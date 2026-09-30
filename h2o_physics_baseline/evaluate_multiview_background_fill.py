#!/usr/bin/env python3
"""Audit four-exo RGB-D background candidates for student old-only holes."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys
import warnings

import numpy as np
import torch
from PIL import Image, ImageDraw, ImageFilter

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from h2o_geometric_baseline.reprojection import (
    load_intrinsics,
    load_pose,
    reproject_rgbd,
)
from h2o_physics_baseline.dataset import H2OPhysicalClipDataset, _gaussian_maps
from h2o_physics_baseline.model import _student_disocclusion_fill, _student_flow_warp
from h2o_physics_baseline.model import PhysicalEgoVideoPredictor


LIMB_SECONDARY_APPEARANCE_RADIUS = 0.24
LIMB_DENSIFY_ITERATIONS = 1


def scaled_rotation(rotation: np.ndarray, scale: float) -> np.ndarray:
    u, _, vt = np.linalg.svd((1.0 - scale) * np.eye(3) + scale * rotation)
    result = u @ vt
    if np.linalg.det(result) < 0:
        u[:, -1] *= -1
        result = u @ vt
    return result


def masked_gaussian(rgb: np.ndarray, valid: np.ndarray, sigma: float) -> np.ndarray:
    radius = max(1, int(np.ceil(2 * sigma)))
    offsets = np.arange(-radius, radius + 1)
    kernel = np.exp(-0.5 * np.square(offsets / sigma))
    kernel /= kernel.sum()
    numerator = rgb * valid[..., None]
    denominator = valid.astype(np.float32)
    for axis in (0, 1):
        numerator = np.apply_along_axis(
            lambda line: np.convolve(line, kernel, mode="same"), axis, numerator
        )
        denominator = np.apply_along_axis(
            lambda line: np.convolve(line, kernel, mode="same"), axis, denominator
        )
    output = numerator / np.maximum(denominator[..., None], 1e-6)
    output[~valid] = 0
    return output.astype(np.float32)


def masked_median3(rgb: np.ndarray, valid: np.ndarray) -> np.ndarray:
    padded_rgb = np.pad(rgb, ((1, 1), (1, 1), (0, 0)), mode="edge")
    padded_valid = np.pad(valid, 1, mode="constant", constant_values=False)
    neighbors = []
    for dy in range(3):
        for dx in range(3):
            value = padded_rgb[dy : dy + rgb.shape[0], dx : dx + rgb.shape[1]].copy()
            neighbor_valid = padded_valid[dy : dy + valid.shape[0], dx : dx + valid.shape[1]]
            value[~neighbor_valid] = np.nan
            neighbors.append(value)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", category=RuntimeWarning)
        output = np.nanmedian(np.stack(neighbors), axis=0)
    output[~valid] = 0
    return np.nan_to_num(output).astype(np.float32)


def depth_bilateral3(
    rgb: np.ndarray, depth: np.ndarray, valid: np.ndarray, depth_sigma_m: float = 0.03
) -> np.ndarray:
    padded_rgb = np.pad(rgb, ((1, 1), (1, 1), (0, 0)), mode="edge")
    padded_depth = np.pad(depth, 1, mode="edge")
    padded_valid = np.pad(valid, 1, mode="constant", constant_values=False)
    numerator = np.zeros_like(rgb, dtype=np.float64)
    denominator = np.zeros(valid.shape, dtype=np.float64)
    for dy in range(3):
        for dx in range(3):
            neighbor_rgb = padded_rgb[dy : dy + rgb.shape[0], dx : dx + rgb.shape[1]]
            neighbor_depth = padded_depth[dy : dy + depth.shape[0], dx : dx + depth.shape[1]]
            neighbor_valid = padded_valid[dy : dy + valid.shape[0], dx : dx + valid.shape[1]]
            spatial = 1.0 if (dy == 1 and dx == 1) else (0.61 if dy == 1 or dx == 1 else 0.37)
            with np.errstate(invalid="ignore"):
                difference = np.where(
                    neighbor_valid & valid, neighbor_depth - depth, np.inf
                )
            weight = spatial * np.exp(
                -0.5 * np.square(difference / depth_sigma_m)
            ) * neighbor_valid
            numerator += neighbor_rgb * weight[..., None]
            denominator += weight
    output = numerator / np.maximum(denominator[..., None], 1e-6)
    output[~valid] = 0
    return output.astype(np.float32)


def hand_support_in_pose(
    joints_world: np.ndarray,
    confidence: np.ndarray,
    intrinsics: np.ndarray,
    camera_pose_world: np.ndarray,
    output_size: int,
) -> np.ndarray:
    """Rasterize exo-triangulated hand joints in a predicted ego camera pose."""
    camera_from_world = np.linalg.inv(camera_pose_world)
    points = joints_world.reshape(-1, 3)
    points_camera = points @ camera_from_world[:3, :3].T + camera_from_world[:3, 3]
    weights = np.zeros((2, len(points)), dtype=np.float32)
    weights[0, :21] = confidence[0]
    weights[1, 21:42] = confidence[1]
    return _gaussian_maps(
        points_camera,
        weights,
        intrinsics[:4],
        output_size,
        sigma_px=3.0,
    ).max(axis=0)


def arm_support_in_pose(
    arm_points_world: np.ndarray,
    confidence: np.ndarray,
    intrinsics: np.ndarray,
    camera_pose_world: np.ndarray,
    output_size: int,
) -> np.ndarray:
    """Rasterize shoulder-elbow-wrist/finger capsules in a target camera."""
    camera_from_world = np.linalg.inv(camera_pose_world)
    points = arm_points_world @ camera_from_world[:3, :3].T + camera_from_world[:3, 3]
    z = points[:, 2]
    valid = np.isfinite(points).all(axis=1) & (z > 1e-5) & (confidence > 0)
    fx, fy, cx, cy = intrinsics[:4]
    u = (fx * points[:, 0] / np.maximum(z, 1e-5) + cx) * output_size / 1280.0
    v = (fy * points[:, 1] / np.maximum(z, 1e-5) + cy) * output_size / 720.0
    canvas = Image.new("L", (output_size, output_size), 0)
    draw = ImageDraw.Draw(canvas)
    # Array order: L/R shoulder, L/R elbow, L/R wrist, then paired
    # pinky/index/thumb landmarks.
    connections = (
        (0, 2), (2, 4), (4, 6), (4, 8), (4, 10),
        (1, 3), (3, 5), (5, 7), (5, 9), (5, 11),
    )
    width = max(3, int(round(output_size * 0.07)))
    for start, end in connections:
        if valid[start] and valid[end]:
            draw.line((float(u[start]), float(v[start]), float(u[end]), float(v[end])), fill=255, width=width)
    radius = width // 2
    for index in np.flatnonzero(valid):
        draw.ellipse(
            (u[index] - radius, v[index] - radius, u[index] + radius, v[index] + radius),
            fill=255,
        )
    return np.asarray(canvas.filter(ImageFilter.GaussianBlur(radius=0.8)), dtype=np.float32) / 255.0


def _densify_dynamic_layer(
    rgb: np.ndarray,
    depth: np.ndarray,
    valid: np.ndarray,
    target_support: np.ndarray,
    iterations: int = 2,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Close sub-pixel forward-warp holes without spreading across limb edges."""
    value = rgb.copy()
    z_value = depth.copy()
    mask = valid.copy()
    height, width = mask.shape
    for _ in range(iterations):
        # Permit growth inside the projected limb support, plus one pixel around
        # any other moving object that is present in the depth-motion mask.
        local_band = np.asarray(
            Image.fromarray(mask.astype(np.uint8) * 255).filter(ImageFilter.MaxFilter(3))
        ) > 0
        allowed = (target_support > 0.025) | local_band
        padded_rgb = np.pad(value, ((1, 1), (1, 1), (0, 0)), mode="constant")
        padded_z = np.pad(z_value, 1, mode="constant", constant_values=np.nan)
        padded_valid = np.pad(mask, 1, mode="constant", constant_values=False)
        rgb_sum = np.zeros_like(value, dtype=np.float64)
        z_sum = np.zeros_like(z_value, dtype=np.float64)
        count = np.zeros_like(z_value, dtype=np.float64)
        for dy in range(3):
            for dx in range(3):
                if dy == 1 and dx == 1:
                    continue
                neighbor_valid = padded_valid[dy : dy + height, dx : dx + width]
                rgb_sum += padded_rgb[dy : dy + height, dx : dx + width] * neighbor_valid[..., None]
                z_sum += np.nan_to_num(
                    padded_z[dy : dy + height, dx : dx + width], nan=0.0
                ) * neighbor_valid
                count += neighbor_valid
        fill = (~mask) & allowed & (count >= 2)
        if not np.any(fill):
            break
        value[fill] = (rgb_sum / np.maximum(count[..., None], 1.0))[fill]
        z_value[fill] = (z_sum / np.maximum(count, 1.0))[fill]
        mask[fill] = True
    value[~mask] = 0
    return value.astype(np.float32), z_value.astype(np.float32), mask


def _limb_depth_consistent_mask(
    depth_mm: np.ndarray,
    intrinsics: np.ndarray,
    camera_pose_world: np.ndarray,
    coarse_mask: np.ndarray,
    hand_joints_world: np.ndarray,
    hand_confidence: np.ndarray,
    arm_points_world: np.ndarray,
    arm_confidence: np.ndarray,
) -> np.ndarray:
    """Keep source RGB-D points physically close to the triangulated limb skeleton."""
    candidate = coarse_mask & np.isfinite(depth_mm) & (depth_mm > 0)
    ys, xs = np.nonzero(candidate)
    if len(xs) == 0:
        return candidate
    z = depth_mm[ys, xs].astype(np.float64) / 1000.0
    fx, fy, cx, cy = intrinsics[:4]
    points_camera = np.column_stack(((xs - cx) * z / fx, (ys - cy) * z / fy, z))
    points_world = points_camera @ camera_pose_world[:3, :3].T + camera_pose_world[:3, 3]
    segments: list[tuple[np.ndarray, np.ndarray, float]] = []
    hand_edges = (
        (0, 1), (1, 2), (2, 3), (3, 4),
        (0, 5), (5, 6), (6, 7), (7, 8),
        (0, 9), (9, 10), (10, 11), (11, 12),
        (0, 13), (13, 14), (14, 15), (15, 16),
        (0, 17), (17, 18), (18, 19), (19, 20),
        (5, 9), (9, 13), (13, 17),
    )
    for hand in range(2):
        joints = hand_joints_world[hand]
        confidence = hand_confidence[hand]
        for start, end in hand_edges:
            if confidence[start] > 0 and confidence[end] > 0:
                # A 3 cm cylinder around every finger segment readily includes
                # the tabletop when a hand is resting on it.  Use a tight seed;
                # sub-pixel cracks are filled later inside the target support.
                segments.append((joints[start], joints[end], 0.020))
    arm_edges = (
        # shoulder--elbow, elbow--wrist, wrist--hand landmarks
        (0, 2, 0.060), (2, 4, 0.045),
        (4, 6, 0.028), (4, 8, 0.028), (4, 10, 0.028),
        (1, 3, 0.060), (3, 5, 0.045),
        (5, 7, 0.028), (5, 9, 0.028), (5, 11, 0.028),
    )
    for start, end, radius in arm_edges:
        if arm_confidence[start] > 0 and arm_confidence[end] > 0:
            segments.append((arm_points_world[start], arm_points_world[end], radius))
    near = np.zeros(len(points_world), dtype=bool)
    for start, end, radius in segments:
        direction = end - start
        length_squared = float(direction @ direction)
        if not np.isfinite(length_squared) or length_squared < 1e-8:
            continue
        position = np.clip(
            ((points_world - start) @ direction) / length_squared, 0.0, 1.0
        )
        closest = start[None] + position[:, None] * direction[None]
        near |= np.sum(np.square(points_world - closest), axis=1) <= radius * radius
    result = np.zeros_like(candidate)
    result[ys[near], xs[near]] = True
    return result


def multiview_dynamic_arm_candidate(
    camera_roots: list[Path],
    frame: int,
    target_root: Path,
    target_pose: np.ndarray,
    output_size: int,
    source_stride: int,
    hand_joints_world: np.ndarray,
    hand_confidence: np.ndarray,
    arm_points_world: np.ndarray,
    arm_confidence: np.ndarray,
    source_motion_masks: dict[tuple[str, int], np.ndarray] | None = None,
    source_person_masks: list[np.ndarray | None] | None = None,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Reproject source-view dynamic RGB-D and a stricter limb-only layer.

    The broad motion candidate is useful for held objects, but it must not be
    treated as limb evidence merely because it falls inside a projected 2-D
    arm capsule.  The fourth return value records pixels that also passed the
    metric 3-D distance test around the hand/arm skeleton.
    """
    rgbs = []
    depths = []
    valids = []
    limb_rgbs = []
    limb_depths = []
    limb_valids = []
    target_intrinsics = load_intrinsics(target_root / "cam_intrinsics.txt")
    stem = f"{frame:06d}"
    for camera_output_index, camera in enumerate(camera_roots):
        rgb = np.asarray(Image.open(camera / "rgb" / f"{stem}.png").convert("RGB"))
        depth = np.asarray(Image.open(camera / "depth" / f"{stem}.png")).copy()
        source_intrinsics = load_intrinsics(camera / "cam_intrinsics.txt")
        source_pose = load_pose(camera / "cam_pose" / f"{stem}.txt")
        limb_support = np.maximum(
            hand_support_in_pose(
                hand_joints_world, hand_confidence, source_intrinsics, source_pose, output_size
            ),
            arm_support_in_pose(
                arm_points_world, arm_confidence, source_intrinsics, source_pose, output_size
            ),
        )
        support = limb_support.copy()
        if source_motion_masks is not None:
            motion = source_motion_masks.get((str(camera), frame))
            if motion is not None:
                motion_low = np.asarray(
                    Image.fromarray(motion.astype(np.uint8) * 255).resize(
                        (output_size, output_size), Image.Resampling.NEAREST
                    )
                ).astype(np.float32) / 255.0
                support = np.maximum(support, motion_low)
        raw_support = np.asarray(
            Image.fromarray((support > 0.1).astype(np.uint8) * 255).resize(
                (depth.shape[1], depth.shape[0]), Image.Resampling.NEAREST
            )
        ) > 0
        raw_limb_support = np.asarray(
            Image.fromarray((limb_support > 0.1).astype(np.uint8) * 255).resize(
                (depth.shape[1], depth.shape[0]), Image.Resampling.NEAREST
            )
        ) > 0
        if source_person_masks is not None:
            person_probability = source_person_masks[camera_output_index]
            if person_probability is not None:
                # Low threshold plus a small support dilation preserves uncertain
                # hand boundaries while excluding the high-confidence table.
                person_low = np.asarray(
                    Image.fromarray(person_probability.astype(np.uint8), "L").filter(
                        ImageFilter.MaxFilter(5)
                    )
                ) > 24
                person_raw = np.asarray(
                    Image.fromarray(person_low.astype(np.uint8) * 255).resize(
                        (depth.shape[1], depth.shape[0]), Image.Resampling.NEAREST
                    )
                ) > 0
                raw_limb_support &= person_raw
        raw_limb_support = _limb_depth_consistent_mask(
            depth,
            source_intrinsics,
            source_pose,
            raw_limb_support,
            hand_joints_world,
            hand_confidence,
            arm_points_world,
            arm_confidence,
        )
        limb_depth = depth.copy()
        limb_depth[~raw_limb_support] = 0
        depth[~raw_support] = 0
        result = reproject_rgbd(
            rgb, depth, source_intrinsics, source_pose,
            target_intrinsics, target_pose,
            output_size=(output_size, output_size), source_stride=source_stride,
        )
        rgbs.append(result.rgb.astype(np.float32) / 255.0)
        depths.append(result.depth_m)
        valids.append(result.valid)
        limb_result = reproject_rgbd(
            rgb, limb_depth, source_intrinsics, source_pose,
            target_intrinsics, target_pose,
            output_size=(output_size, output_size), source_stride=source_stride,
        )
        limb_rgbs.append(limb_result.rgb.astype(np.float32) / 255.0)
        limb_depths.append(limb_result.depth_m)
        limb_valids.append(limb_result.valid)
    rgb_stack = np.stack(rgbs)
    depth_stack = np.stack(depths)
    valid_stack = np.stack(valids)
    limb_rgb_stack = np.stack(limb_rgbs)
    limb_depth_stack = np.stack(limb_depths)
    limb_valid_stack = np.stack(limb_valids)
    target_support = np.maximum(
        hand_support_in_pose(
            hand_joints_world, hand_confidence, target_intrinsics, target_pose, output_size
        ),
        arm_support_in_pose(
            arm_points_world, arm_confidence, target_intrinsics, target_pose, output_size
        ),
    )
    # Keep one view as the appearance authority. Other views may only fill its
    # holes; they never overwrite a main-view pixel. Rank by *metric limb*
    # coverage rather than broad motion coverage: otherwise a table surface
    # moving relative to the fixed-exo background can win the main-view vote.
    limb = target_support > 0.025
    limb_coverage = (
        limb_valid_stack & limb[None]
    ).reshape(len(limb_valid_stack), -1).sum(axis=1)
    total_coverage = valid_stack.reshape(len(valid_stack), -1).sum(axis=1)
    order = np.argsort(-(limb_coverage + 0.1 * total_coverage))
    best_view = int(order[0])

    # Broad motion stays available for the separate held-object channel.
    value = rgb_stack[best_view].copy()
    depth = depth_stack[best_view].copy()
    valid = valid_stack[best_view].copy()

    # Limb compositing uses only RGB-D points that are physically close to the
    # triangulated skeleton.  Previously the primary view used the broad layer,
    # while only secondary fills used this filter; that asymmetry admitted table
    # texture as fully valid arm pixels and made later inpainting unable to act.
    limb_value = limb_rgb_stack[best_view].copy()
    limb_depth = limb_depth_stack[best_view].copy()
    limb_valid = limb_valid_stack[best_view].copy()
    main_limb_samples = limb_value[limb_valid & limb]
    limb_color = (
        np.median(main_limb_samples, axis=0).astype(np.float32)
        if len(main_limb_samples) else np.array([0.65, 0.45, 0.38], dtype=np.float32)
    )
    for view_index in order[1:]:
        candidate_valid = limb_valid_stack[view_index]
        overlap = limb_valid & candidate_valid & limb
        correction = np.zeros(3, dtype=np.float32)
        if int(overlap.sum()) >= 24:
            # Robust exposure/white-balance alignment only. Geometry remains
            # untouched, and the correction is bounded to avoid skin-color drift.
            correction = np.median(
                limb_value[overlap] - limb_rgb_stack[view_index][overlap], axis=0
            ).astype(np.float32)
            correction = np.clip(correction, -0.12, 0.12)
        # Secondary views contribute skin/limb appearance, not arbitrary table
        # or held-object color that merely lies behind a 2D hand capsule.
        appearance_distance = np.sqrt(
            np.sum(np.square(limb_rgb_stack[view_index] - limb_color[None, None]), axis=-1)
        )
        # A permissive threshold turns differently exposed/table-adjacent
        # RGB-D islands into hard bands once they are densified.  Prefer an
        # explicit unknown limb pixel over a weakly matching secondary view;
        # the former can be completed temporally, while the latter is locked.
        fill = (
            (~limb_valid)
            & candidate_valid
            & (appearance_distance <= LIMB_SECONDARY_APPEARANCE_RADIUS)
        )
        if not np.any(fill):
            continue
        candidate_rgb = np.clip(limb_rgb_stack[view_index] + correction, 0.0, 1.0)
        limb_value[fill] = candidate_rgb[fill]
        limb_depth[fill] = limb_depth_stack[view_index][fill]
        limb_valid[fill] = True
    # Grow only one pixel ring. Repeating this operation creates concentric
    # colour terraces around projected samples, while disabling it entirely
    # discards too much reliable structure for the temporal completer.
    limb_value, limb_depth, limb_valid = _densify_dynamic_layer(
        limb_value, limb_depth, limb_valid, target_support,
        iterations=LIMB_DENSIFY_ITERATIONS,
    )
    value, depth, valid = _densify_dynamic_layer(
        value, depth, valid, target_support, iterations=2
    )
    value[limb_valid] = limb_value[limb_valid]
    depth[limb_valid] = limb_depth[limb_valid]
    return value, depth, valid, limb_valid


def causal_static_video_candidates(
    camera_roots: list[Path],
    history_frames: list[int],
    target_root: Path,
    target_pose: np.ndarray,
    output_size: int,
    source_stride: int,
    student_frames: np.ndarray,
    student_joints: np.ndarray,
    student_confidence: np.ndarray,
    arm_frames: np.ndarray | None = None,
    arm_points_world: np.ndarray | None = None,
    arm_confidence: np.ndarray | None = None,
    source_motion_masks: dict[tuple[str, int], np.ndarray] | None = None,
    minimum_agree: int = 2,
) -> dict[str, tuple[np.ndarray, np.ndarray]]:
    """Fuse hand-masked current/past exo RGB-D into a causal static background."""
    rgbs = []
    depths = []
    valids = []
    target_intrinsics = load_intrinsics(target_root / "cam_intrinsics.txt")
    for frame in history_frames:
        state_index = int(np.searchsorted(student_frames, frame))
        if state_index >= len(student_frames) or int(student_frames[state_index]) != frame:
            raise IndexError(f"Student state missing causal video frame {frame}")
        for camera in camera_roots:
            stem = f"{frame:06d}"
            rgb = np.asarray(Image.open(camera / "rgb" / f"{stem}.png").convert("RGB"))
            depth = np.asarray(Image.open(camera / "depth" / f"{stem}.png")).copy()
            source_intrinsics = load_intrinsics(camera / "cam_intrinsics.txt")
            source_pose = load_pose(camera / "cam_pose" / f"{stem}.txt")
            support = hand_support_in_pose(
                student_joints[state_index],
                student_confidence[state_index],
                source_intrinsics,
                source_pose,
                output_size,
            )
            if arm_frames is not None and arm_points_world is not None and arm_confidence is not None:
                arm_index = int(np.searchsorted(arm_frames, frame))
                if arm_index < len(arm_frames) and int(arm_frames[arm_index]) == frame:
                    support = np.maximum(
                        support,
                        arm_support_in_pose(
                            arm_points_world[arm_index],
                            arm_confidence[arm_index],
                            source_intrinsics,
                            source_pose,
                            output_size,
                        ),
                    )
            if source_motion_masks is not None:
                motion = source_motion_masks.get((str(camera), frame))
                if motion is not None:
                    motion_low = np.asarray(
                        Image.fromarray(motion.astype(np.uint8) * 255).resize(
                            (output_size, output_size), Image.Resampling.NEAREST
                        )
                    ).astype(np.float32) / 255.0
                    support = np.maximum(support, motion_low)
            raw_mask = np.asarray(
                Image.fromarray((support > 0.05).astype(np.uint8) * 255).resize(
                    (depth.shape[1], depth.shape[0]), Image.Resampling.NEAREST
                )
            ) > 0
            depth[raw_mask] = 0
            result = reproject_rgbd(
                rgb,
                depth,
                source_intrinsics,
                source_pose,
                target_intrinsics,
                target_pose,
                output_size=(output_size, output_size),
                source_stride=source_stride,
            )
            rgbs.append(result.rgb.astype(np.float32) / 255.0)
            depths.append(result.depth_m)
            valids.append(result.valid)
    rgb_stack = np.stack(rgbs)
    depth_stack = np.stack(depths)
    valid_stack = np.stack(valids)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", category=RuntimeWarning)
        median_z = np.nanmedian(np.where(valid_stack, depth_stack, np.nan), axis=0)
    references = {
        "nearest": np.min(np.where(valid_stack, depth_stack, np.inf), axis=0),
        "median": median_z,
        # A moving hand/object is normally in front of the wanted static
        # surface.  The farthest repeated cluster is therefore a useful video
        # background hypothesis, although it may fail at true depth edges.
        "farthest": np.max(np.where(valid_stack, depth_stack, -np.inf), axis=0),
    }
    outputs = {}
    for name, reference in references.items():
        agree = valid_stack & (np.abs(depth_stack - reference[None]) <= 0.05)
        agree_count = agree.sum(axis=0)
        valid = agree_count >= minimum_agree
        value = (rgb_stack * agree[..., None]).sum(axis=0) / np.maximum(
            agree_count[..., None], 1
        )
        value[~valid] = 0
        outputs[name] = (masked_gaussian(value, valid, sigma=0.8), valid)
    return outputs


def multiview_candidates(
    camera_roots: list[Path],
    frame: int,
    target_root: Path,
    target_pose: np.ndarray,
    output_size: int,
    source_stride: int,
) -> dict[str, tuple[np.ndarray, np.ndarray]]:
    rgbs = []
    depths = []
    valids = []
    stem = f"{frame:06d}"
    target_intrinsics = load_intrinsics(target_root / "cam_intrinsics.txt")
    for camera in camera_roots:
        rgb = np.asarray(Image.open(camera / "rgb" / f"{stem}.png").convert("RGB"))
        depth = np.asarray(Image.open(camera / "depth" / f"{stem}.png"))
        result = reproject_rgbd(
            rgb,
            depth,
            load_intrinsics(camera / "cam_intrinsics.txt"),
            load_pose(camera / "cam_pose" / f"{stem}.txt"),
            target_intrinsics,
            target_pose,
            output_size=(output_size, output_size),
            source_stride=source_stride,
        )
        rgbs.append(result.rgb.astype(np.float32) / 255.0)
        depths.append(result.depth_m)
        valids.append(result.valid)
    rgb_stack = np.stack(rgbs)
    depth_stack = np.stack(depths)
    valid_stack = np.stack(valids)
    any_valid = valid_stack.any(axis=0)

    def gather(indices: np.ndarray) -> np.ndarray:
        yy, xx = np.indices(indices.shape)
        return rgb_stack[indices, yy, xx]

    nearest_depth = np.where(valid_stack, depth_stack, np.inf)
    nearest = gather(np.argmin(nearest_depth, axis=0))
    farthest_depth = np.where(valid_stack, depth_stack, -np.inf)
    farthest = gather(np.argmax(farthest_depth, axis=0))
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", category=RuntimeWarning)
        median_depth = np.nanmedian(np.where(valid_stack, depth_stack, np.nan), axis=0)
    median_distance = np.where(valid_stack, np.abs(depth_stack - median_depth[None]), np.inf)
    median = gather(np.argmin(median_distance, axis=0))
    nearest_z = np.min(nearest_depth, axis=0)
    support_two = valid_stack.sum(axis=0) >= 2
    consensus_candidates = {}
    for tolerance_cm in (2, 5, 10):
        depth_agree = valid_stack & (
            np.abs(depth_stack - nearest_z[None]) <= tolerance_cm / 100.0
        )
        agree_count = depth_agree.sum(axis=0)
        value = (
            (rgb_stack * depth_agree[..., None]).sum(axis=0)
            / np.maximum(agree_count[..., None], 1)
        )
        consensus_candidates[f"depth_consensus_{tolerance_cm}cm"] = (
            value,
            agree_count >= 2,
        )
    for value in (nearest, farthest, median):
        value[~any_valid] = 0
    for value, valid in consensus_candidates.values():
        value[~valid] = 0
    consensus_5cm, consensus_valid = consensus_candidates["depth_consensus_5cm"]
    filtered_candidates = {
        "depth_consensus_5cm_gaussian3": (
            masked_gaussian(consensus_5cm, consensus_valid, sigma=0.8),
            consensus_valid,
        ),
        "depth_consensus_5cm_gaussian5": (
            masked_gaussian(consensus_5cm, consensus_valid, sigma=1.2),
            consensus_valid,
        ),
        "depth_consensus_5cm_median3": (
            masked_median3(consensus_5cm, consensus_valid),
            consensus_valid,
        ),
        "depth_consensus_5cm_bilateral3": (
            depth_bilateral3(consensus_5cm, nearest_z, consensus_valid),
            consensus_valid,
        ),
    }
    return {
        "nearest": (nearest, any_valid),
        "nearest_support2": (nearest, support_two),
        **consensus_candidates,
        **filtered_candidates,
        "farthest": (farthest, any_valid),
        "median_depth": (median, any_valid),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--checkpoint",
        type=Path,
        default=Path(
            "datasets/H2O/experiments/student_warp_fill_v2/fill_seed26/"
            "multiview_anchored_student_flow_warp_fill_local_gate31/best.pt"
        ),
    )
    parser.add_argument(
        "--student-state-root",
        type=Path,
        default=Path("datasets/H2O/student_state_mediapipe_64_32"),
    )
    parser.add_argument("--max-samples", type=int, default=32)
    parser.add_argument("--source-stride", type=int, default=4)
    parser.add_argument(
        "--target-pose",
        choices=("initial", "predicted_translation", "predicted_pose", "future_gt"),
        default="initial",
        help="future_gt is an evaluation-only oracle and is not a deployable input",
    )
    parser.add_argument(
        "--head-summary",
        type=Path,
        default=Path("datasets/H2O/experiments/exo_head_motion/summary.json"),
    )
    parser.add_argument("--translation-scale", type=float, default=0.5)
    parser.add_argument("--rotation-scale", type=float, default=0.75)
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("datasets/H2O/experiments/multiview_background_fill/summary.json"),
    )
    args = parser.parse_args()
    head_motion = {}
    if args.target_pose in {"predicted_translation", "predicted_pose"}:
        head_data = json.loads(args.head_summary.read_text(encoding="utf-8"))
        head_motion = {
            record["pair_id"]: {
                int(frame["frame"]): {
                    "position": np.asarray(
                        frame["estimated_camera_delta_position_world_m"], dtype=np.float64
                    ),
                    "rotation": np.asarray(
                        frame["estimated_delta_rotation_world"], dtype=np.float64
                    ),
                }
                for frame in record["future"]
                if "estimated_camera_delta_position_world_m" in frame
            }
            for record in head_data["per_clip"]
        }
    checkpoint = torch.load(args.checkpoint, map_location="cpu", weights_only=False)
    config = checkpoint["config"]
    model = PhysicalEgoVideoPredictor(condition_mode=config["condition_mode"])
    model.load_state_dict(checkpoint["model"])
    model.eval()
    image_size = int(config["image_size"])
    dataset = H2OPhysicalClipDataset(
        config["index"],
        split="val",
        frames_per_clip=config["frames_per_clip"],
        image_size=image_size,
        stats_path=config["stats"],
        source_cameras=tuple(config["source_cameras"]),
        max_samples=args.max_samples,
        combine_source_cameras=True,
        student_state_root=args.student_state_root,
        student_flow=True,
    )
    geometric_names = (
        "local_average",
        "nearest",
        "nearest_support2",
        "depth_consensus_2cm",
        "depth_consensus_5cm",
        "depth_consensus_10cm",
        "depth_consensus_5cm_gaussian3",
        "depth_consensus_5cm_gaussian5",
        "depth_consensus_5cm_median3",
        "depth_consensus_5cm_bilateral3",
        "farthest",
        "median_depth",
    )
    names = geometric_names + (
        "network",
        "network_depth_consensus_2cm",
        "network_depth_consensus_5cm",
        "network_depth_consensus_10cm",
        "network_depth_consensus_5cm_gaussian3",
        "network_depth_consensus_5cm_gaussian5",
        "network_depth_consensus_5cm_median3",
        "network_depth_consensus_5cm_bilateral3",
        "rendered_new_hand_nearest",
        "rendered_hand_nearest_soft",
        "rendered_hand_consensus_soft",
        "global_ego_rgbd_warp",
        "global_ego_rgbd_warp_exo_fill",
        "layered_static_background",
        "layered_static_background_dynamic_hand",
        "layered_causal_video_background",
        "layered_causal_video_residual",
        "layered_causal_video_residual_median",
        "layered_causal_video_residual_farthest",
    )
    totals = {
        name: {"absolute": 0.0, "pixels": 0.0, "old_absolute": 0.0, "old_pixels": 0.0,
               "changed_absolute": 0.0, "changed_pixels": 0.0,
               "hand_absolute": 0.0, "hand_pixels": 0.0,
               "known_absolute": 0.0, "known_pixels": 0.0,
               "unknown_absolute": 0.0, "unknown_pixels": 0.0}
        for name in names
    }
    covered_old = 0.0
    total_old = 0.0
    per_clip = []
    for index in range(len(dataset)):
        item = dataset[index]
        row = dataset.rows[index]
        anchor = torch.from_numpy(item["ego_anchor"])[None]
        maps = torch.from_numpy(item["initial_ego_maps"])[None]
        target = torch.from_numpy(item["target"])[None]
        warped = _student_flow_warp(anchor, maps)
        local, old_only, new_only = _student_disocclusion_fill(anchor, warped, maps)
        candidate_frames = {name: [] for name in geometric_names[1:]}
        valid_frames = {name: [] for name in geometric_names[1:]}
        source_roots = [Path(directory).parent for directory in json.loads(row["source_rgb_dirs"])]
        target_root = Path(row["target_rgb_dir"]).parent
        first_frame = int(item["frame_numbers"][0])
        initial_target_pose = load_pose(target_root / "cam_pose" / f"{first_frame:06d}.txt")
        initial_ego_rgb = np.asarray(
            Image.open(target_root / "rgb" / f"{first_frame:06d}.png").convert("RGB")
        )
        initial_ego_depth = np.asarray(
            Image.open(target_root / "depth" / f"{first_frame:06d}.png")
        )
        # Remove the first-frame hands before the global camera warp.  Otherwise
        # they are incorrectly treated as part of the static scene and leave a
        # second, camera-warped hand ghost in every future frame.
        initial_hand_support = maps[0, 0, :2].amax(dim=0).numpy()
        initial_hand_mask = initial_hand_support > 0.05
        initial_hand_mask = np.asarray(
            Image.fromarray(initial_hand_mask.astype(np.uint8) * 255).resize(
                (initial_ego_depth.shape[1], initial_ego_depth.shape[0]),
                Image.Resampling.NEAREST,
            )
        ) > 0
        static_ego_depth = initial_ego_depth.copy()
        static_ego_depth[initial_hand_mask] = 0
        target_intrinsics = load_intrinsics(target_root / "cam_intrinsics.txt")
        student_path = args.student_state_root / row["sequence"] / "student_state.npz"
        with np.load(student_path) as archive:
            student_frames = archive["frames"].copy()
            student_joints = archive["hand_joints_world_m"].copy()
            student_confidence = archive["joint_confidence"].copy()
        global_warp_frames = []
        global_valid_frames = []
        static_warp_frames = []
        static_valid_frames = []
        predicted_hand_frames = []
        temporal_static_frames = {name: [] for name in ("nearest", "median", "farthest")}
        temporal_static_valid_frames = {name: [] for name in ("nearest", "median", "farthest")}
        for time_index, frame in enumerate(item["frame_numbers"]):
            target_pose = initial_target_pose
            if args.target_pose == "future_gt":
                target_pose = load_pose(target_root / "cam_pose" / f"{int(frame):06d}.txt")
            elif args.target_pose == "predicted_translation" and int(frame) != first_frame:
                target_pose = initial_target_pose.copy()
                target_pose[:3, 3] += (
                    args.translation_scale
                    * head_motion[item["pair_id"]][int(frame)]["position"]
                )
            elif args.target_pose == "predicted_pose" and int(frame) != first_frame:
                motion = head_motion[item["pair_id"]][int(frame)]
                target_pose = initial_target_pose.copy()
                target_pose[:3, 3] += args.translation_scale * motion["position"]
                target_pose[:3, :3] = (
                    scaled_rotation(motion["rotation"], args.rotation_scale)
                    @ initial_target_pose[:3, :3]
                )
            candidates = multiview_candidates(
                source_roots,
                int(frame),
                target_root,
                target_pose,
                image_size,
                args.source_stride,
            )
            global_warp = reproject_rgbd(
                initial_ego_rgb,
                initial_ego_depth,
                target_intrinsics,
                initial_target_pose,
                target_intrinsics,
                target_pose,
                output_size=(image_size, image_size),
                source_stride=args.source_stride,
            )
            global_warp_frames.append(global_warp.rgb.astype(np.float32) / 255.0)
            global_valid_frames.append(global_warp.valid)
            static_warp = reproject_rgbd(
                initial_ego_rgb,
                static_ego_depth,
                target_intrinsics,
                initial_target_pose,
                target_intrinsics,
                target_pose,
                output_size=(image_size, image_size),
                source_stride=args.source_stride,
            )
            static_warp_frames.append(static_warp.rgb.astype(np.float32) / 255.0)
            static_valid_frames.append(static_warp.valid)
            state_index = int(np.searchsorted(student_frames, int(frame)))
            if state_index >= len(student_frames) or int(student_frames[state_index]) != int(frame):
                raise IndexError(f"Student state missing frame {int(frame)} for {item['pair_id']}")
            predicted_hand_frames.append(
                hand_support_in_pose(
                    student_joints[state_index],
                    student_confidence[state_index],
                    target_intrinsics,
                    target_pose,
                    image_size,
                )
            )
            temporal_candidates = causal_static_video_candidates(
                source_roots,
                [int(value) for value in item["frame_numbers"][: time_index + 1]],
                target_root,
                target_pose,
                image_size,
                args.source_stride,
                student_frames,
                student_joints,
                student_confidence,
            )
            for temporal_name, (temporal_static, temporal_valid) in temporal_candidates.items():
                temporal_static_frames[temporal_name].append(temporal_static)
                temporal_static_valid_frames[temporal_name].append(temporal_valid)
            for name in geometric_names[1:]:
                candidate_frames[name].append(candidates[name][0])
                valid_frames[name].append(candidates[name][1])
        valid = torch.from_numpy(
            np.stack(valid_frames["nearest"]).astype(np.float32)
        )[None, :, None]
        old = old_only
        covered_old += float((valid * old).sum())
        total_old += float(old.sum())
        variants = {"local_average": local}
        candidate_tensors = {}
        valid_tensors = {}
        for name in geometric_names[1:]:
            candidate_valid = torch.from_numpy(
                np.stack(valid_frames[name]).astype(np.float32)
            )[None, :, None]
            candidate = torch.from_numpy(
                np.moveaxis(np.stack(candidate_frames[name]), -1, 1)
            )[None]
            candidate_tensors[name] = candidate
            valid_tensors[name] = candidate_valid
            replacement = candidate * candidate_valid + local * (1.0 - candidate_valid)
            variants[name] = warped * (1.0 - old) + replacement * old
        with torch.inference_mode():
            prediction, _ = model(
                torch.from_numpy(item["source"])[None],
                torch.from_numpy(item["physical_vector"])[None],
                torch.from_numpy(item["physical_maps"])[None],
                torch.from_numpy(item["coarse_rgb"])[None],
                torch.from_numpy(item["coarse_depth"])[None],
                torch.from_numpy(item["visibility_mask"])[None],
                anchor,
                maps,
            )
        variants["network"] = prediction
        for tolerance_cm in (2, 5, 10):
            candidate_name = f"depth_consensus_{tolerance_cm}cm"
            consensus_mask = old * valid_tensors[candidate_name]
            variants[f"network_{candidate_name}"] = (
                prediction * (1.0 - consensus_mask)
                + candidate_tensors[candidate_name] * consensus_mask
            )
        for filter_name in ("gaussian3", "gaussian5", "median3", "bilateral3"):
            candidate_name = f"depth_consensus_5cm_{filter_name}"
            consensus_mask = old * valid_tensors[candidate_name]
            variants[f"network_{candidate_name}"] = (
                prediction * (1.0 - consensus_mask)
                + candidate_tensors[candidate_name] * consensus_mask
            )
        geometry_base = variants["depth_consensus_5cm_gaussian3"]
        current_hand = maps[:, :, :2].amax(dim=2, keepdim=True).clamp(0.0, 1.0)
        current_hand[:, 0] = 0
        nearest_valid = valid_tensors["nearest"]
        nearest_hand_alpha = current_hand * nearest_valid
        new_hand_alpha = new_only * nearest_valid
        variants["rendered_new_hand_nearest"] = (
            geometry_base * (1.0 - new_hand_alpha)
            + candidate_tensors["nearest"] * new_hand_alpha
        )
        variants["rendered_hand_nearest_soft"] = (
            geometry_base * (1.0 - nearest_hand_alpha)
            + candidate_tensors["nearest"] * nearest_hand_alpha
        )
        consensus_hand_alpha = current_hand * valid_tensors["depth_consensus_5cm"]
        variants["rendered_hand_consensus_soft"] = (
            geometry_base * (1.0 - consensus_hand_alpha)
            + candidate_tensors["depth_consensus_5cm"] * consensus_hand_alpha
        )
        global_warp = torch.from_numpy(
            np.moveaxis(np.stack(global_warp_frames), -1, 1)
        )[None]
        global_valid = torch.from_numpy(
            np.stack(global_valid_frames).astype(np.float32)
        )[None, :, None]
        global_warp[:, 0] = anchor[:, 0]
        global_valid[:, 0] = 1
        variants["global_ego_rgbd_warp"] = (
            global_warp * global_valid + anchor * (1.0 - global_valid)
        )
        exo_valid = valid_tensors["depth_consensus_5cm_gaussian3"]
        exo_hole_alpha = (1.0 - global_valid) * exo_valid
        variants["global_ego_rgbd_warp_exo_fill"] = (
            global_warp * global_valid
            + candidate_tensors["depth_consensus_5cm_gaussian3"] * exo_hole_alpha
            + anchor * (1.0 - global_valid) * (1.0 - exo_valid)
        )
        static_warp = torch.from_numpy(
            np.moveaxis(np.stack(static_warp_frames), -1, 1)
        )[None]
        static_valid = torch.from_numpy(
            np.stack(static_valid_frames).astype(np.float32)
        )[None, :, None]
        static_warp[:, 0] = anchor[:, 0]
        static_valid[:, 0] = 1
        static_hole_alpha = (1.0 - static_valid) * exo_valid
        layered_background = (
            static_warp * static_valid
            + candidate_tensors["depth_consensus_5cm_gaussian3"] * static_hole_alpha
            + anchor * (1.0 - static_valid) * (1.0 - exo_valid)
        )
        variants["layered_static_background"] = layered_background
        predicted_hand = torch.from_numpy(
            np.stack(predicted_hand_frames).astype(np.float32)
        )[None, :, None].clamp(0.0, 1.0)
        predicted_hand[:, 0] = 0
        dynamic_alpha = predicted_hand * valid_tensors["nearest"]
        variants["layered_static_background_dynamic_hand"] = (
            layered_background * (1.0 - dynamic_alpha)
            + candidate_tensors["nearest"] * dynamic_alpha
        )
        temporal_tensors = {}
        temporal_valid_tensors = {}
        for temporal_name in temporal_static_frames:
            temporal_tensors[temporal_name] = torch.from_numpy(
                np.moveaxis(np.stack(temporal_static_frames[temporal_name]), -1, 1)
            )[None]
            temporal_valid_tensors[temporal_name] = torch.from_numpy(
                np.stack(temporal_static_valid_frames[temporal_name]).astype(np.float32)
            )[None, :, None]
            temporal_tensors[temporal_name][:, 0] = anchor[:, 0]
            temporal_valid_tensors[temporal_name][:, 0] = 1
        temporal_static = temporal_tensors["nearest"]
        temporal_static_valid = temporal_valid_tensors["nearest"]
        temporal_hole_alpha = (1.0 - static_valid) * temporal_static_valid
        variants["layered_causal_video_background"] = (
            static_warp * static_valid
            + temporal_static * temporal_hole_alpha
            + anchor * (1.0 - static_valid) * (1.0 - temporal_static_valid)
        )
        # Conservative video use: never overwrite a first-frame 3D projection
        # or a current-frame exo observation.  History only resolves pixels for
        # which both of those sources are absent.
        temporal_residual_alpha = (
            (1.0 - static_valid) * (1.0 - exo_valid) * temporal_static_valid
        )
        variants["layered_causal_video_residual"] = (
            layered_background * (1.0 - temporal_residual_alpha)
            + temporal_static * temporal_residual_alpha
        )
        for temporal_name in ("median", "farthest"):
            history = temporal_tensors[temporal_name]
            history_valid = temporal_valid_tensors[temporal_name]
            residual_alpha = (1.0 - static_valid) * (1.0 - exo_valid) * history_valid
            variants[f"layered_causal_video_residual_{temporal_name}"] = (
                layered_background * (1.0 - residual_alpha) + history * residual_alpha
            )
        information_masks = {
            "global_ego_rgbd_warp": global_valid,
            "global_ego_rgbd_warp_exo_fill": torch.maximum(global_valid, exo_valid),
            "layered_static_background": torch.maximum(static_valid, exo_valid),
            "layered_static_background_dynamic_hand": torch.maximum(
                torch.maximum(static_valid, exo_valid), (dynamic_alpha > 0).float()
            ),
            "layered_causal_video_background": torch.maximum(
                static_valid, temporal_static_valid
            ),
            "layered_causal_video_residual": torch.maximum(
                torch.maximum(static_valid, exo_valid), temporal_static_valid
            ),
            "layered_causal_video_residual_median": torch.maximum(
                torch.maximum(static_valid, exo_valid), temporal_valid_tensors["median"]
            ),
            "layered_causal_video_residual_farthest": torch.maximum(
                torch.maximum(static_valid, exo_valid), temporal_valid_tensors["farthest"]
            ),
        }
        changed = ((target - anchor).abs().mean(dim=2, keepdim=True) > 0.05).float()
        changed[:, 0] = 0
        hand_region = maps[:, :, :2].amax(dim=2, keepdim=True)
        clip_record = {"pair_id": item["pair_id"], "variants": {}}
        for name, prediction in variants.items():
            absolute = (prediction - target).abs()
            record = totals[name]
            record["absolute"] += float(absolute.sum())
            record["pixels"] += float(absolute.numel())
            record["old_absolute"] += float((absolute * old).sum())
            record["old_pixels"] += float(old.sum() * 3)
            record["changed_absolute"] += float((absolute * changed).sum())
            record["changed_pixels"] += float(changed.sum() * 3)
            record["hand_absolute"] += float((absolute * hand_region).sum())
            record["hand_pixels"] += float(hand_region.sum() * 3)
            if name in information_masks:
                known = information_masks[name]
                unknown = 1.0 - known
                record["known_absolute"] += float((absolute * known).sum())
                record["known_pixels"] += float(known.sum() * 3)
                record["unknown_absolute"] += float((absolute * unknown).sum())
                record["unknown_pixels"] += float(unknown.sum() * 3)
            clip_record["variants"][name] = {
                "l1": float(absolute.mean()),
                "old_only_l1": float((absolute * old).sum() / (old.sum() * 3).clamp_min(1)),
            }
        per_clip.append(clip_record)
        print(json.dumps({"index": index, **clip_record}, ensure_ascii=False), flush=True)
    metrics = {}
    for name, record in totals.items():
        metrics[name] = {
            "l1": record["absolute"] / max(record["pixels"], 1),
            "old_only_l1": record["old_absolute"] / max(record["old_pixels"], 1),
            "changed_l1": record["changed_absolute"] / max(record["changed_pixels"], 1),
            "hand_l1": record["hand_absolute"] / max(record["hand_pixels"], 1),
        }
        if record["known_pixels"] > 0:
            total_information_pixels = record["known_pixels"] + record["unknown_pixels"]
            metrics[name].update({
                "information_coverage": record["known_pixels"] / total_information_pixels,
                "known_l1": record["known_absolute"] / record["known_pixels"],
                "unknown_l1": record["unknown_absolute"] / max(record["unknown_pixels"], 1),
            })
    result = {
        "protocol": {
            "initial": "four calibrated exo RGB-D + initial ego RGB/pose; no future ego pose/frame",
            "predicted_translation": (
                "four calibrated exo RGB-D + exo-predicted half-scale ego translation; "
                "initial ego rotation; no future ego pose/frame"
            ),
            "predicted_pose": (
                "four calibrated exo RGB-D + exo face/depth predicted ego motion; "
                "no future ego pose/frame"
            ),
            "future_gt": (
                "evaluation oracle: four calibrated exo RGB-D + future GT ego pose; "
                "no future ego RGB/depth"
            ),
        }[args.target_pose],
        "target_pose_mode": args.target_pose,
        "translation_scale": args.translation_scale if args.target_pose.startswith("predicted") else None,
        "rotation_scale": args.rotation_scale if args.target_pose == "predicted_pose" else None,
        "checkpoint": str(args.checkpoint),
        "samples": len(dataset),
        "frames_per_clip": config["frames_per_clip"],
        "image_size": image_size,
        "source_stride": args.source_stride,
        "global_warp_extra_input": "initial ego depth; separate RGB-D-first-frame protocol",
        "old_only_multiview_coverage": covered_old / max(total_old, 1),
        "metrics": metrics,
        "per_clip": per_clip,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print(json.dumps(result | {"per_clip": "omitted"}, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
