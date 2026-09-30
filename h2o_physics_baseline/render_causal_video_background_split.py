#!/usr/bin/env python3
"""Render a clean six-column, 64-frame real-H2O comparison video."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
import subprocess
import sys

import cv2
import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image, ImageDraw, ImageFilter

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from h2o_geometric_baseline.reprojection import load_intrinsics, load_pose, reproject_rgbd
from h2o_physics_baseline.dataset import DEFAULT_INDEX
from h2o_physics_baseline.protocols import LAYERED_CONTRACT_VERSION
from h2o_physics_baseline.annotated_object_layer import (
    annotated_exo_object_geometry_layer,
    annotated_exo_object_layer,
    annotated_exo_object_pose_world,
    build_anchored_object_appearance,
    render_anchored_object_appearance,
)
from h2o_physics_baseline.evaluate_multiview_background_fill import (
    LIMB_DENSIFY_ITERATIONS,
    LIMB_SECONDARY_APPEARANCE_RADIUS,
    arm_support_in_pose,
    causal_static_video_candidates,
    hand_support_in_pose,
    multiview_candidates,
    multiview_dynamic_arm_candidate,
    scaled_rotation,
)
from h2o_physics_baseline.visualize_student_warp_fill import (
    INPUT_BORDER,
    INTERMEDIATE_BORDER,
    PREDICTION_BORDER,
    TARGET_BORDER,
    font,
)


def pil_rgb(value: np.ndarray, size: int) -> Image.Image:
    return Image.fromarray(
        np.rint(np.clip(value, 0, 1) * 255).astype(np.uint8), "RGB"
    ).resize((size, size), Image.Resampling.NEAREST)


def title_tile(
    image: Image.Image,
    title: str,
    size: int,
    border: tuple[int, int, int],
) -> Image.Image:
    """Image plus one black title line; deliberately no header or gray subtitle."""
    tile = Image.new("RGB", (size, size + 44), "white")
    tile.paste(image.resize((size, size), Image.Resampling.NEAREST), (0, 0))
    draw = ImageDraw.Draw(tile)
    draw.rectangle((0, 0, size - 1, size - 1), outline=border, width=5)
    draw.text((8, size + 5), title, font=font(20), fill=(15, 23, 42))
    return tile


CURRENT_EXO_COLOR = INTERMEDIATE_BORDER
HISTORY_EXO_COLOR = (251, 191, 36)
GENERATED_COLOR = PREDICTION_BORDER
FOREGROUND_COLOR = (217, 70, 239)
OBJECT_COLOR = (14, 165, 233)
DYNAMIC_COMPLETION_COLOR = (20, 184, 166)


def feather_fill_layer(
    base_rgb: np.ndarray,
    base_valid: np.ndarray,
    candidate_rgb: np.ndarray,
    candidate_valid: np.ndarray,
    radius: int,
    color_align: bool,
) -> tuple[np.ndarray, np.ndarray]:
    """Fill a background layer while softly crossing observed-source borders.

    Pixels without a base observation still come entirely from the candidate.
    In the overlap, only a narrow band on the base side is blended.  This avoids
    the hard binary switch responsible for broken provenance boundaries while
    leaving the interior of every measured layer unchanged.
    """
    candidate = candidate_rgb.copy()
    overlap = base_valid & candidate_valid
    if color_align and int(overlap.sum()) >= 64:
        # A bounded robust offset compensates exposure/white-balance differences
        # without allowing one camera to recolor the whole scene.
        correction = np.median(
            base_rgb[overlap] - candidate_rgb[overlap], axis=0
        ).astype(np.float32)
        correction = np.clip(correction, -0.08, 0.08)
        candidate = np.clip(candidate + correction, 0.0, 1.0)

    value = base_rgb.copy()
    fill = (~base_valid) & candidate_valid
    value[fill] = candidate[fill]
    if radius > 0 and np.any(overlap):
        distance = cv2.distanceTransform(
            base_valid.astype(np.uint8), cv2.DIST_L2, 3
        )
        candidate_alpha = np.clip((radius + 1.0 - distance) / radius, 0.0, 1.0)
        candidate_alpha *= overlap.astype(np.float32)
        value = (
            value * (1.0 - candidate_alpha[..., None])
            + candidate * candidate_alpha[..., None]
        )
    return value.astype(np.float32), base_valid | candidate_valid


def dilated_mask(mask: np.ndarray, size: int) -> np.ndarray:
    if size % 2 != 1:
        raise ValueError("dilation size must be odd")
    return cv2.dilate(
        mask.astype(np.uint8), np.ones((size, size), dtype=np.uint8)
    ) > 0


def load_anchor_source_hand_occlusion(
    mask_root: Path,
    sequence: str,
    frame: int,
    output_size: int,
    hand_support: np.ndarray,
    arm_support: np.ndarray,
    object_support: np.ndarray,
    object_depth_m: np.ndarray,
    observed_depth_m: np.ndarray,
    safety_margin_px: int,
    front_margin_m: float,
) -> tuple[np.ndarray, dict[str, float]]:
    """Build a conservative first-ego-frame hand provenance exclusion.

    GrabCut supplies dense appearance boundaries, while projected exo-only hand
    and arm state prevents a connected held object from being treated wholesale
    as hand. Any remaining boundary ambiguity is resolved in favour of
    *unknown object surface*: over-exclusion creates a hole, whereas
    under-exclusion transports hand texture as if it were rigid object texture.
    """
    path = mask_root / sequence / "initial_hand_masks.npz"
    if not path.exists():
        raise FileNotFoundError(
            f"Anchor-warp requires a first-frame hand mask, missing {path}"
        )
    with np.load(path) as archive:
        frames = archive["frames"]
        index = int(np.searchsorted(frames, frame))
        if index >= len(frames) or int(frames[index]) != frame:
            raise IndexError(f"First-frame hand mask missing {sequence} frame {frame}")
        per_hand = archive["masks"][index]
    dense = cv2.resize(
        per_hand.max(axis=0),
        (output_size, output_size),
        interpolation=cv2.INTER_LINEAR,
    ) > 32
    hand_zone = dilated_mask(hand_support > 0.025, 17)
    arm_zone = dilated_mask(arm_support > 0.025, 9)
    dense_limb = dense & (hand_zone | arm_zone)
    if np.any(np.max(per_hand, axis=(1, 2)) == 0):
        dense_limb |= hand_zone
    # GrabCut often connects a held object to the fingers.  Calling that entire
    # component "hand" deletes legitimate object texture (notably the chips
    # packet). Resolve the contact boundary by metric depth: an occluder must be
    # measurably in front of the CAD surface, not merely share its 2-D support.
    depth_comparable = (
        dense_limb
        & object_support
        & np.isfinite(object_depth_m)
        & np.isfinite(observed_depth_m)
        & (observed_depth_m > 0)
    )
    depth_residual = observed_depth_m - object_depth_m
    # CAD alignment and sensor depth can have a clip-specific offset. Estimate
    # it from the dominant contact region, then classify relative frontness;
    # otherwise a 1 cm pose bias can make the entire object look like an
    # occluder (the failure observed on k2_4).
    alignment_bias_m = (
        float(np.median(depth_residual[depth_comparable]))
        if np.any(depth_comparable) else 0.0
    )
    depth_ordered = depth_comparable & (
        depth_residual < alignment_bias_m - front_margin_m
    )
    occlusion = depth_ordered
    if safety_margin_px > 0:
        boundary_margin_m = max(0.003, 0.25 * front_margin_m)
        occlusion = (
            dilated_mask(occlusion, 2 * safety_margin_px + 1)
            & depth_comparable
            & (depth_residual < alignment_bias_m - boundary_margin_m)
        )
    diagnostics = {
        "dense_limb_fraction": float(dense_limb.mean()),
        "depth_comparable_fraction": float(depth_comparable.mean()),
        "alignment_bias_m": alignment_bias_m,
        "front_seed_fraction": float(depth_ordered.mean()),
        "excluded_fraction": float(occlusion.mean()),
    }
    return occlusion, diagnostics


def classify_dynamic_foreground(
    candidate_valid: np.ndarray,
    hand_support: np.ndarray,
    arm_support: np.ndarray,
    limb_observed_valid: np.ndarray | None = None,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Separate limb evidence from small hand-adjacent moving-object evidence.

    The RGB-D candidate also contains a generic depth-motion mask.  Previously
    every such pixel was labelled and composited as an arm.  Keep limb pixels
    only near the projected 3-D skeleton; allow a wider region solely around
    hands for a held object, and reject all remote motion responses.
    """
    limb_core = np.maximum(hand_support, arm_support) > 0.025
    arm_zone = dilated_mask(limb_core, 7)
    held_object_zone = dilated_mask(hand_support > 0.025, 25) & (~arm_zone)
    # Broad motion is allowed to propose a held object, but arm pixels must
    # additionally pass the source-space metric limb test.  Without this split,
    # nearby table/runner pixels become protected "observed arm" texture.
    limb_evidence = (
        candidate_valid if limb_observed_valid is None else limb_observed_valid
    )
    arm_valid = limb_evidence & arm_zone
    object_valid = candidate_valid & held_object_zone

    # Isolated depth speckles are not credible held-object evidence.
    component_count, components, stats, _ = cv2.connectedComponentsWithStats(
        object_valid.astype(np.uint8), connectivity=8
    )
    clean_object = np.zeros_like(object_valid)
    for component in range(1, component_count):
        if int(stats[component, cv2.CC_STAT_AREA]) >= 12:
            clean_object[components == component] = True
    object_valid = clean_object
    return arm_valid | object_valid, arm_valid, object_valid


def channel_mask_image(
    static_valid: np.ndarray,
    current_valid: np.ndarray,
    history_valid: np.ndarray,
    arm_valid: np.ndarray,
    object_valid: np.ndarray,
    dynamic_completion: np.ndarray | None = None,
    object_label: str = "持物近邻",
) -> Image.Image:
    """Render mutually exclusive provenance channels used by the final composite."""
    height, width = static_valid.shape
    current_only = (~static_valid) & current_valid
    history_only = (~static_valid) & (~current_valid) & history_valid
    generated = (~static_valid) & (~current_valid) & (~history_valid)
    canvas = np.zeros((height, width, 3), dtype=np.uint8)
    canvas[static_valid] = INPUT_BORDER
    canvas[current_only] = CURRENT_EXO_COLOR
    canvas[history_only] = HISTORY_EXO_COLOR
    canvas[generated] = GENERATED_COLOR
    if dynamic_completion is not None:
        canvas[dynamic_completion] = DYNAMIC_COMPLETION_COLOR
    # Foreground is composited last, so it also has highest mask precedence.
    canvas[object_valid] = OBJECT_COLOR
    canvas[arm_valid] = FOREGROUND_COLOR
    image = Image.fromarray(canvas, "RGB")
    draw = ImageDraw.Draw(image)
    legend = [
        (INPUT_BORDER, "首帧"),
        (CURRENT_EXO_COLOR, "当前exo"),
        (HISTORY_EXO_COLOR, "历史exo"),
        (GENERATED_COLOR, "生成"),
    ]
    if dynamic_completion is not None:
        legend.append((DYNAMIC_COMPLETION_COLOR, "手/臂待补全"))
    legend.extend(((OBJECT_COLOR, object_label), (FOREGROUND_COLOR, "手/臂观测")))
    legend_font = font(max(11, width // 21))
    box_width = max(104, width * 43 // 100)
    row_height = max(17, width // 14)
    draw.rounded_rectangle(
        (5, 5, box_width, 11 + row_height * len(legend)),
        radius=5,
        fill=(15, 23, 42),
    )
    for index, (color, label) in enumerate(legend):
        y = 8 + index * row_height
        draw.rectangle((10, y + 2, 21, y + 13), fill=color)
        draw.text((27, y - 1), label, font=legend_font, fill="white")
    return image


def model_repair_masks(
    static_valid: np.ndarray,
    current_valid: np.ndarray,
    history_valid: np.ndarray,
    foreground_valid: np.ndarray,
    dynamic_support: np.ndarray,
    dynamic_completion: np.ndarray | None = None,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Build background-repair masks plus an explicit missing-limb channel.

    Observed dynamic RGB stays protected.  In a single-view protocol, however,
    the projected kinematic envelope can contain pixels for which that camera
    supplied no limb appearance.  Those pixels are not background: authorize
    the video model there after applying the ordinary dynamic guard.
    """
    labels = np.zeros(static_valid.shape, dtype=np.uint8)
    labels[static_valid] = 1
    labels[(~static_valid) & current_valid] = 2
    labels[(~static_valid) & (~current_valid) & history_valid] = 3
    labels[(~static_valid) & (~current_valid) & (~history_valid)] = 4
    background_edge = np.zeros_like(static_valid)
    background_edge[1:, :] |= labels[1:, :] != labels[:-1, :]
    background_edge[:-1, :] |= labels[:-1, :] != labels[1:, :]
    background_edge[:, 1:] |= labels[:, 1:] != labels[:, :-1]
    background_edge[:, :-1] |= labels[:, :-1] != labels[:, 1:]
    seam = np.asarray(
        Image.fromarray(background_edge.astype(np.uint8) * 255).filter(
            ImageFilter.MaxFilter(5)
        )
    ) > 0
    dynamic = foreground_valid | (dynamic_support > 0.025)
    dynamic_guard = np.asarray(
        Image.fromarray(dynamic.astype(np.uint8) * 255).filter(ImageFilter.MaxFilter(9))
    ) > 0
    seam &= ~dynamic_guard
    unknown = (labels == 4) & (~dynamic_guard)
    repair = seam | unknown
    if dynamic_completion is not None:
        repair |= dynamic_completion
    return labels, unknown, seam, repair


def missing_limb_completion_mask(
    hand_support: np.ndarray,
    arm_support: np.ndarray,
    arm_valid: np.ndarray,
    object_valid: np.ndarray,
) -> np.ndarray:
    """Pixels expected to be limb-shaped but unobserved by the sole exo view.

    This is intentionally a conservative *authorization* mask, not a rendered
    arm.  The skeleton supplies position/extent; actual RGB must come from a
    temporal generative model.  A small dilation treats reprojection cracks as
    observed, while visible held-object pixels retain occlusion priority.
    """
    expected = (arm_support > 0.10) | (hand_support > 0.025)
    observed = dilated_mask(arm_valid, 5)
    object_guard = dilated_mask(object_valid, 3)
    completion = expected & (~observed) & (~object_guard)

    # Remove tiny islands caused by one low-confidence projected landmark.
    count, components, stats, _ = cv2.connectedComponentsWithStats(
        completion.astype(np.uint8), connectivity=8
    )
    clean = np.zeros_like(completion)
    for component in range(1, count):
        if int(stats[component, cv2.CC_STAT_AREA]) >= 20:
            clean[components == component] = True
    return clean


def temporal_smooth(values: np.ndarray, radius: int = 4) -> np.ndarray:
    """Robust symmetric smoothing using only the exo-derived trajectory."""
    values = values.astype(np.float64)
    flat = values.reshape(len(values), -1).copy()
    for channel in range(flat.shape[1]):
        series = flat[:, channel]
        finite = np.isfinite(series)
        if not np.any(finite):
            series[:] = 0
        elif not np.all(finite):
            series[~finite] = np.interp(
                np.flatnonzero(~finite), np.flatnonzero(finite), series[finite]
            )
        robust = np.empty_like(series)
        for index in range(len(series)):
            robust[index] = np.median(
                series[max(0, index - radius) : min(len(series), index + radius + 1)]
            )
        weights = np.arange(1, radius + 2, dtype=np.float64)
        weights = np.concatenate((weights, weights[-2::-1]))
        padded = np.pad(robust, radius, mode="edge")
        flat[:, channel] = np.convolve(padded, weights / weights.sum(), mode="valid")
    return flat.reshape(values.shape).astype(np.float32)


def smooth_world_points(points: np.ndarray) -> np.ndarray:
    return temporal_smooth(points, radius=3)


def smoothed_pose_map(
    frames: list[int], initial_pose: np.ndarray, head_record: dict,
    rotation_scale: float = 0.5,
) -> dict[int, np.ndarray]:
    translations = [np.zeros(3, dtype=np.float64)]
    rotations = [np.eye(3, dtype=np.float64)]
    records = {int(value["frame"]): value for value in head_record["future"]}
    for frame in frames[1:]:
        motion = records[frame]
        if "estimated_head_transform_world" in motion:
            # This path stays valid when initial_pose is inferred rather than read
            # from cam4: the exo-estimated head transform acts on whichever
            # head-mounted camera anchor the protocol supplies.
            moved_pose = np.asarray(motion["estimated_head_transform_world"]) @ initial_pose
            translations.append(moved_pose[:3, 3] - initial_pose[:3, 3])
            rotations.append(moved_pose[:3, :3] @ initial_pose[:3, :3].T)
        else:
            # Backward compatibility for summaries written before the anchor
            # ablation stored the exo-only rigid transform.
            translations.append(np.asarray(motion["estimated_camera_delta_position_world_m"]))
            rotations.append(np.asarray(motion["estimated_delta_rotation_world"]))
    translations = temporal_smooth(np.stack(translations), radius=4)
    rotations_smoothed = temporal_smooth(np.stack(rotations), radius=4)
    rotations_projected = []
    for matrix in rotations_smoothed:
        u, _, vt = np.linalg.svd(matrix)
        rotation = u @ vt
        if np.linalg.det(rotation) < 0:
            u[:, -1] *= -1
            rotation = u @ vt
        rotations_projected.append(rotation)
    translations -= translations[0]
    rotations_projected[0] = np.eye(3)
    result = {}
    for frame, translation, rotation in zip(frames, translations, rotations_projected):
        pose = initial_pose.copy()
        pose[:3, 3] += translation
        pose[:3, :3] = scaled_rotation(rotation, rotation_scale) @ initial_pose[:3, :3]
        result[frame] = pose
    return result


def head_pose_confidence_map(frames: list[int], head_record: dict) -> dict[int, float]:
    """Turn face support, RANSAC agreement and residual into a bounded score."""
    records = {int(value["frame"]): value for value in head_record["future"]}
    detected = dict(zip(frames, head_record.get("detected_exo_views", [])))
    confidence = {frames[0]: 1.0}
    for frame in frames[1:]:
        record = records[frame]
        if "estimated_head_transform_world" not in record:
            confidence[frame] = 0.0
            continue
        if "pose_confidence" in record:
            confidence[frame] = float(np.clip(record["pose_confidence"], 0.0, 1.0))
            continue
        common = max(int(record.get("common_landmarks", 0)), 1)
        inlier_ratio = int(record.get("ransac_inliers", 0)) / common
        view_factor = min(1.0, float(detected.get(frame, 0)) / 2.0)
        residual_factor = float(np.exp(
            -float(record.get("inlier_residual_mean_m", 0.0)) / 0.025
        ))
        confidence[frame] = float(np.clip(
            inlier_ratio * view_factor * residual_factor, 0.0, 1.0
        ))
    return confidence


def calibrated_pose_reliability_map(
    raw_confidence: dict[int, float], calibration_path: Path | None,
) -> dict[int, float]:
    """Map the geometric score to validation-measured good-pose probability."""
    if calibration_path is None:
        # Uncalibrated scores must not silently change the renderer.  They stay
        # available in the manifest for auditing and terminal-model features.
        return {frame: 1.0 for frame in raw_confidence}
    audit = json.loads(calibration_path.read_text(encoding="utf-8"))
    knots = audit["calibration"]["probability_good_pose"]
    x = np.asarray(knots["x"], dtype=np.float64)
    y = np.asarray(knots["y"], dtype=np.float64)
    if len(x) == 0 or len(x) != len(y):
        raise ValueError(f"Invalid pose confidence calibration: {calibration_path}")
    return {
        frame: float(np.clip(np.interp(value, x, y), 0.0, 1.0))
        for frame, value in raw_confidence.items()
    }


def compute_motion_masks(
    camera_roots: list[Path], frames: list[int], size: tuple[int, int] = (160, 90),
    causal: bool = False,
) -> dict[tuple[str, int], np.ndarray]:
    """Fixed-exo depth background model; optionally use only depth seen so far."""
    masks: dict[tuple[str, int], np.ndarray] = {}
    for camera in camera_roots:
        depths = []
        for frame in frames:
            depth = Image.open(camera / "depth" / f"{frame:06d}.png").resize(
                size, Image.Resampling.NEAREST
            )
            depths.append(np.asarray(depth, dtype=np.float32))
        stack = np.stack(depths)
        for index, (frame, depth) in enumerate(zip(frames, stack)):
            history = stack[: index + 1] if causal else stack
            with np.errstate(invalid="ignore"):
                background = np.nanpercentile(
                    np.where(history > 0, history, np.nan), 90, axis=0
                )
            dynamic = (depth > 0) & np.isfinite(background) & ((background - depth) > 30.0)
            # Expand to absorb limb/object boundaries and depth quantization.
            dynamic_image = Image.fromarray(dynamic.astype(np.uint8) * 255).filter(
                ImageFilter.MaxFilter(5)
            )
            masks[(str(camera), frame)] = np.asarray(dynamic_image) > 0
    return masks


def add_systematic_pose_noise(
    poses: dict[int, np.ndarray], rotation_degrees: float,
    translation_m: float, seed: int, mode: str = "systematic",
    rotation_axis: str = "random", translation_axis: str = "random",
) -> tuple[dict[int, np.ndarray], dict[str, object]]:
    """Apply repeatable calibration error or time-growing drift to a trajectory."""
    def unit_vector(name: str, rng: np.random.Generator) -> np.ndarray:
        if name in {"x", "y", "z"}:
            value = np.zeros(3, dtype=np.float64)
            value[{"x": 0, "y": 1, "z": 2}[name]] = 1.0
            return value
        value = rng.normal(size=3)
        return value / max(float(np.linalg.norm(value)), 1e-12)

    rng = np.random.default_rng(seed)
    axis = unit_vector(rotation_axis, rng)
    direction = unit_vector(translation_axis, rng)
    metadata: dict[str, object] = {
        "mode": mode,
        "rotation_axis": axis.tolist(),
        "translation_direction": direction.tolist(),
    }
    if rotation_degrees == 0.0 and translation_m == 0.0:
        return poses, metadata
    result = {}
    items = list(poses.items())
    for index, (frame, original) in enumerate(items):
        factor = index / max(len(items) - 1, 1) if mode == "linear_drift" else 1.0
        rotation_vector = axis * np.deg2rad(rotation_degrees * factor)
        noisy_rotation, _ = cv2.Rodrigues(rotation_vector.astype(np.float64))
        translation = direction * translation_m * factor
        pose = original.copy()
        pose[:3, :3] = pose[:3, :3] @ noisy_rotation
        pose[:3, 3] += translation
        result[frame] = pose
    return result, metadata


def projected_motion_support(
    camera_roots: list[Path],
    frame: int,
    target_root: Path,
    target_pose: np.ndarray,
    motion_masks: dict[tuple[str, int], np.ndarray],
    output_size: int,
) -> np.ndarray:
    target_intrinsics = load_intrinsics(target_root / "cam_intrinsics.txt")
    support = np.zeros((output_size, output_size), dtype=bool)
    stem = f"{frame:06d}"
    for camera in camera_roots:
        depth = np.asarray(Image.open(camera / "depth" / f"{stem}.png")).copy()
        raw_mask = np.asarray(
            Image.fromarray(
                motion_masks[(str(camera), frame)].astype(np.uint8) * 255
            ).resize((depth.shape[1], depth.shape[0]), Image.Resampling.NEAREST)
        ) > 0
        depth[~raw_mask] = 0
        rgb = np.repeat((raw_mask[..., None].astype(np.uint8) * 255), 3, axis=2)
        result = reproject_rgbd(
            rgb, depth,
            load_intrinsics(camera / "cam_intrinsics.txt"),
            load_pose(camera / "cam_pose" / f"{stem}.txt"),
            target_intrinsics, target_pose,
            output_size=(output_size, output_size), source_stride=2,
        )
        support |= result.valid
    return np.asarray(
        Image.fromarray(support.astype(np.uint8) * 255).filter(ImageFilter.MaxFilter(5))
    ) > 0


def warp_generated_background(
    previous_rgb: np.ndarray,
    previous_pose: np.ndarray,
    current_pose: np.ndarray,
    intrinsics: np.ndarray,
    nominal_depth_m: float,
) -> tuple[np.ndarray, np.ndarray]:
    """Move a generated background prior with the predicted camera trajectory.

    Unknown geometry cannot be reprojected exactly.  We therefore backproject a
    dense current-view ray field at the median observed scene depth and sample
    the previous generated background in the previous camera.  This is only a
    motion prior: current geometric observations remain hard constraints.
    """
    height, width = previous_rgb.shape[:2]
    native_width, native_height = intrinsics[4:6]
    fx = intrinsics[0] * width / native_width
    fy = intrinsics[1] * height / native_height
    cx = intrinsics[2] * width / native_width
    cy = intrinsics[3] * height / native_height
    grid_x, grid_y = np.meshgrid(
        np.arange(width, dtype=np.float64), np.arange(height, dtype=np.float64)
    )
    depth = max(float(nominal_depth_m), 0.2)
    current_points = np.stack(
        ((grid_x - cx) * depth / fx, (grid_y - cy) * depth / fy,
         np.full_like(grid_x, depth)), axis=-1,
    )
    world_points = (
        current_points @ current_pose[:3, :3].T + current_pose[:3, 3]
    )
    previous_from_world = np.linalg.inv(previous_pose)
    previous_points = (
        world_points @ previous_from_world[:3, :3].T + previous_from_world[:3, 3]
    )
    z = previous_points[..., 2]
    u = fx * previous_points[..., 0] / np.maximum(z, 1e-6) + cx
    v = fy * previous_points[..., 1] / np.maximum(z, 1e-6) + cy
    valid = (z > 1e-4) & (u >= 0) & (u <= width - 1) & (v >= 0) & (v <= height - 1)
    sample_grid = np.stack(
        (2.0 * u / max(width - 1, 1) - 1.0,
         2.0 * v / max(height - 1, 1) - 1.0), axis=-1,
    ).astype(np.float32)
    source = torch.from_numpy(previous_rgb).permute(2, 0, 1)[None].float()
    grid = torch.from_numpy(sample_grid)[None]
    warped = F.grid_sample(
        source, grid, mode="bilinear", padding_mode="zeros", align_corners=True
    )[0].permute(1, 2, 0).numpy()
    return warped, valid


def generate_unknown_background(
    observed_rgb: np.ndarray,
    known: np.ndarray,
    previous_generated: np.ndarray | None = None,
    previous_pose: np.ndarray | None = None,
    current_pose: np.ndarray | None = None,
    intrinsics: np.ndarray | None = None,
    nominal_depth_m: float = 0.8,
) -> tuple[np.ndarray, np.ndarray]:
    """Generate pixels with no geometric observation, never copying ego frame 0.

    A trajectory-warped previous generated frame supplies temporal structure.
    Remaining new disocclusions are synthesized by normalized multiscale
    diffusion from current observed boundaries.  Known pixels are restored
    exactly after every operation, so generation cannot overwrite evidence.
    """
    known_tensor = torch.from_numpy(known.astype(np.float32))[None, None]
    observed = torch.from_numpy(observed_rgb).permute(2, 0, 1)[None].float()
    generated = torch.zeros_like(observed)
    prior_valid = np.zeros_like(known)
    if (
        previous_generated is not None and previous_pose is not None
        and current_pose is not None and intrinsics is not None
    ):
        prior, prior_valid = warp_generated_background(
            previous_generated, previous_pose, current_pose, intrinsics,
            nominal_depth_m,
        )
        generated = torch.from_numpy(prior).permute(2, 0, 1)[None].float()
    generated = torch.where(known_tensor.bool(), observed, generated)
    support = torch.maximum(
        known_tensor,
        torch.from_numpy(prior_valid.astype(np.float32))[None, None],
    )

    # Propagate only current evidence or the motion-warped generated prior.
    # A 5x5 normalized expansion fills genuinely new field-of-view regions.
    for _ in range(max(observed_rgb.shape) // 2):
        if bool((support > 0.5).all()):
            break
        denominator = F.avg_pool2d(support, 5, stride=1, padding=2)
        numerator = F.avg_pool2d(generated * support, 5, stride=1, padding=2)
        proposal = numerator / denominator.clamp_min(1e-6)
        new_pixels = (support < 0.5) & (denominator > 1e-6)
        generated = torch.where(new_pixels.expand_as(generated), proposal, generated)
        support = torch.where(new_pixels, torch.ones_like(support), support)

    # A few Jacobi steps remove expansion bands while preserving every observed
    # pixel. Replicate padding is essential here: zero padding darkens unknown
    # pixels along the image boundary and can look like a failed completion.
    # The warped prior keeps the result temporally tied to camera motion.
    unknown = ~known_tensor.bool()
    for _ in range(12):
        smooth = F.avg_pool2d(
            F.pad(generated, (1, 1, 1, 1), mode="replicate"),
            3,
            stride=1,
        )
        generated = torch.where(unknown.expand_as(generated), smooth, observed)
    result = generated[0].permute(1, 2, 0).numpy()
    return np.clip(result, 0.0, 1.0), ~known


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--index", type=Path, default=DEFAULT_INDEX)
    parser.add_argument("--pair-id", required=True)
    parser.add_argument("--head-summary", type=Path, required=True)
    parser.add_argument("--student-state-root", type=Path, required=True)
    parser.add_argument("--arm-state-root", type=Path, required=True)
    parser.add_argument(
        "--source-camera-indices",
        default="0,1,2,3",
        help="Comma-separated exo RGB-D appearance cameras; state estimates stay fixed.",
    )
    parser.add_argument(
        "--state-camera-indices",
        default="0,1,2,3",
        help="Cameras used to create the supplied head/hand/arm state files.",
    )
    parser.add_argument(
        "--disable-ego-anchor",
        action="store_true",
        help="Do not use ego frame 0 RGB-D as an inference input.",
    )
    parser.add_argument(
        "--initial-pose-summary",
        type=Path,
        default=None,
        help=(
            "Use a predicted initial camera pose from a mount-prior summary instead "
            "of reading the target clip's cam4 pose. Requires --disable-ego-anchor."
        ),
    )
    parser.add_argument("--render-size", type=int, default=256)
    parser.add_argument(
        "--source-feather-radius",
        type=int,
        default=0,
        help="Blend this many pixels across static/current/history source borders.",
    )
    parser.add_argument(
        "--source-color-align",
        action="store_true",
        help="Apply a bounded robust RGB offset before source-border blending.",
    )
    parser.add_argument(
        "--head-rotation-scale", type=float, default=0.5,
        help="Scale exo-estimated head rotation; 0.5 is the frozen validation setting.",
    )
    parser.add_argument(
        "--causal-motion-masks", action="store_true",
        help="Build each exo depth background from current/past frames only.",
    )
    parser.add_argument("--pose-noise-rotation-deg", type=float, default=0.0)
    parser.add_argument("--pose-noise-translation-m", type=float, default=0.0)
    parser.add_argument("--pose-noise-seed", type=int, default=0)
    parser.add_argument(
        "--pose-noise-mode", choices=("systematic", "linear_drift"),
        default="systematic",
    )
    parser.add_argument(
        "--pose-noise-rotation-axis", choices=("random", "x", "y", "z"),
        default="random",
    )
    parser.add_argument(
        "--pose-noise-translation-axis", choices=("random", "x", "y", "z"),
        default="random",
    )
    parser.add_argument(
        "--frame-step", type=int, default=1,
        help="Render every Nth canonical frame for robustness audits; training uses 1.",
    )
    parser.add_argument(
        "--pose-confidence-calibration", type=Path,
        help=(
            "Validation-only isotonic calibration JSON. When supplied, its "
            "good-pose probability controls geometric evidence during composition."
        ),
    )
    parser.add_argument(
        "--annotated-exo-object",
        action="store_true",
        help="Use cam0--cam3 object-pose annotations as an explicit upper-bound layer.",
    )
    parser.add_argument(
        "--annotated-exo-object-mode",
        choices=(
            "appearance", "cad_gate", "cad_filter",
            "anchor_warp", "anchor_warp_exo_fill",
        ),
        default="appearance",
        help="Object appearance ablation; anchor modes rigidly transport first-ego-frame pixels.",
    )
    parser.add_argument(
        "--object-state-root",
        type=Path,
        default=Path("datasets/H2O/oracle_state"),
    )
    parser.add_argument(
        "--object-alpha",
        type=float,
        default=1.0,
        help="Confidence weight for anchor-warp object appearance; other modes use 1.",
    )
    parser.add_argument(
        "--initial-hand-mask-root",
        type=Path,
        default=Path("datasets/H2O/student_initial_hand_masks_64_32"),
        help="Ego-first-frame hand masks used to reject hand RGB from object anchors.",
    )
    parser.add_argument(
        "--object-source-hand-margin",
        type=int,
        default=5,
        help="Safety dilation in render pixels around first-frame hand occlusion.",
    )
    parser.add_argument(
        "--object-source-front-margin-m",
        type=float,
        default=0.015,
        help="Minimum relative measured-in-front depth gap for hand occlusion.",
    )
    parser.add_argument(
        "--model-input-root",
        type=Path,
        default=None,
        help="Optional directory for raw composite frames and constrained repair masks.",
    )
    parser.add_argument(
        "--complete-single-view-limbs",
        action="store_true",
        help=(
            "Treat the projected hand/arm envelope not covered by the sole exo "
            "view as a dynamic-generation mask instead of locked background."
        ),
    )
    parser.add_argument(
        "--complete-missing-limbs",
        action="store_true",
        help=(
            "For any number of exo views, mark the projected hand/arm envelope "
            "without reliable limb RGB-D as an authorized dynamic-completion mask."
        ),
    )
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if not 0.0 <= args.object_alpha <= 1.0:
        raise ValueError("--object-alpha must be in [0, 1]")
    if args.object_source_hand_margin < 0:
        raise ValueError("--object-source-hand-margin must be non-negative")
    if args.object_source_front_margin_m < 0:
        raise ValueError("--object-source-front-margin-m must be non-negative")
    if args.pose_noise_rotation_deg < 0 or args.pose_noise_translation_m < 0:
        raise ValueError("pose-noise magnitudes must be non-negative")
    if args.frame_step < 1:
        raise ValueError("--frame-step must be positive")
    if args.disable_ego_anchor and args.annotated_exo_object_mode.startswith("anchor_warp"):
        raise ValueError("anchor_warp requires ego frame 0; disable the object anchor too")
    if args.initial_pose_summary is not None and not args.disable_ego_anchor:
        raise ValueError(
            "A first-frame RGB-D geometry anchor has no world alignment after the "
            "head-camera transform is withheld; add --disable-ego-anchor"
        )
    render_size = args.render_size
    poster = args.output.with_name(args.output.stem + "_poster.png")
    # This renderer is the deterministic final layered pipeline.  It must not
    # inherit its data selection from an obsolete learned-model checkpoint.
    # Resolve the requested clip directly from the canonical paired index and
    # collect its four synchronized exo streams in camera order.
    with args.index.open(encoding="utf-8") as handle:
        indexed_rows = list(csv.DictReader(handle))
    matches = [value for value in indexed_rows if value["pair_id"] == args.pair_id]
    if len(matches) != 1:
        raise ValueError(
            f"Expected one canonical row for pair_id={args.pair_id!r}, found {len(matches)}"
        )
    row = dict(matches[0])
    clip_rows = [
        value for value in indexed_rows
        if value["clip_id"] == row["clip_id"]
        and value["target_camera"] == row["target_camera"]
    ]
    by_camera = {value["source_camera"]: value for value in clip_rows}
    required_cameras = tuple(f"cam{index}" for index in range(4))
    missing_cameras = [camera for camera in required_cameras if camera not in by_camera]
    if missing_cameras:
        raise ValueError(
            f"Missing synchronized exo rows for {row['clip_id']}: {missing_cameras}"
        )
    row["source_rgb_dirs"] = json.dumps(
        [by_camera[camera]["source_rgb_dir"] for camera in required_cameras]
    )
    frames = list(
        range(int(row["start_frame"]), int(row["end_frame"]) + 1, args.frame_step)
    )
    first_frame = frames[0]
    target_root = Path(row["target_rgb_dir"]).parent
    all_source_roots = [Path(value).parent for value in json.loads(row["source_rgb_dirs"])]
    source_indices = tuple(
        int(value.strip()) for value in args.source_camera_indices.split(",")
        if value.strip()
    )
    if not source_indices or any(index < 0 or index >= len(all_source_roots) for index in source_indices):
        raise ValueError(f"Invalid --source-camera-indices={args.source_camera_indices}")
    state_indices = tuple(
        int(value.strip()) for value in args.state_camera_indices.split(",")
        if value.strip()
    )
    if not state_indices or any(index < 0 or index >= len(all_source_roots) for index in state_indices):
        raise ValueError(f"Invalid --state-camera-indices={args.state_camera_indices}")
    source_roots = [all_source_roots[index] for index in source_indices]
    if args.complete_single_view_limbs and len(source_roots) != 1:
        raise ValueError("--complete-single-view-limbs is defined for one exo view")
    completion_enabled = (
        args.complete_single_view_limbs or args.complete_missing_limbs
    )
    intrinsics = load_intrinsics(target_root / "cam_intrinsics.txt")
    initial_pose_source = "target_cam4_first_pose"
    if args.initial_pose_summary is None:
        initial_pose = load_pose(target_root / "cam_pose" / f"{first_frame:06d}.txt")
    else:
        pose_summary = json.loads(args.initial_pose_summary.read_text())
        pose_record = next(
            value for value in pose_summary["targets"]
            if value["pair_id"] == args.pair_id
        )
        initial_pose = np.asarray(
            pose_record["predicted_initial_pose_world"], dtype=np.float64
        )
        initial_pose_source = (
            "exo_face_head_frame_plus_cross_split_canonical_mount:"
            f"{args.initial_pose_summary}"
        )
    head_summary = json.loads(args.head_summary.read_text())
    head_record = next(
        value for value in head_summary["per_clip"] if value["pair_id"] == args.pair_id
    )
    poses = smoothed_pose_map(
        frames, initial_pose, head_record, rotation_scale=args.head_rotation_scale
    )
    poses, pose_noise_parameters = add_systematic_pose_noise(
        poses, args.pose_noise_rotation_deg, args.pose_noise_translation_m,
        args.pose_noise_seed, args.pose_noise_mode,
        args.pose_noise_rotation_axis, args.pose_noise_translation_axis,
    )
    pose_confidences = head_pose_confidence_map(frames, head_record)
    pose_reliabilities = calibrated_pose_reliability_map(
        pose_confidences, args.pose_confidence_calibration
    )
    motion_masks = compute_motion_masks(
        source_roots, frames, causal=args.causal_motion_masks
    )
    initial_rgb_u8 = np.asarray(
        Image.open(target_root / "rgb" / f"{first_frame:06d}.png").convert("RGB")
    )
    initial_rgb = np.asarray(
        Image.fromarray(initial_rgb_u8).resize(
            (render_size, render_size), Image.Resampling.BILINEAR
        )
    ).astype(np.float32) / 255.0
    initial_depth = np.asarray(Image.open(target_root / "depth" / f"{first_frame:06d}.png"))
    student_path = args.student_state_root / row["sequence"] / "student_state.npz"
    with np.load(student_path) as archive:
        student_frames = archive["frames"].copy()
        student_joints = archive["hand_joints_world_m"].copy()
        student_confidence = archive["joint_confidence"].copy()
    arm_path = args.arm_state_root / row["sequence"] / "arm_state.npz"
    with np.load(arm_path) as archive:
        arm_frames = archive["frames"].copy()
        arm_points = archive["arm_points_world_m"].copy()
        arm_confidence = archive["confidence"].copy()
        arm_person_masks = (
            archive["person_masks_256"].copy()
            if "person_masks_256" in archive.files else None
        )
        arm_person_camera_indices = (
            archive["camera_indices"].copy()
            if "camera_indices" in archive.files else None
        )
    student_joints = smooth_world_points(student_joints)
    arm_points = smooth_world_points(arm_points)
    initial_hand = hand_support_in_pose(
        student_joints[0], student_confidence[0], intrinsics, initial_pose, render_size
    )
    initial_arm = arm_support_in_pose(
        arm_points[0], arm_confidence[0], intrinsics, initial_pose, render_size
    )
    initial_motion = projected_motion_support(
        source_roots, first_frame, target_root, initial_pose, motion_masks, render_size
    )
    initial_object_layer = None
    anchored_object_appearance = None
    anchor_unfiltered_source_points = None
    anchor_source_occlusion_diagnostics: dict[str, float] = {}
    anchor_source_hand_occlusion = np.zeros(
        (render_size, render_size), dtype=bool
    )
    if args.annotated_exo_object:
        if args.annotated_exo_object_mode.startswith("anchor_warp"):
            initial_object_id, initial_object_pose, _ = annotated_exo_object_pose_world(
                source_roots, first_frame
            )
            initial_object_geometry = annotated_exo_object_geometry_layer(
                source_roots,
                first_frame,
                intrinsics,
                initial_pose,
                args.object_state_root,
                output_size=render_size,
            )
            observed_initial_depth = np.asarray(
                Image.fromarray(initial_depth).resize(
                    (render_size, render_size), Image.Resampling.NEAREST
                ),
                dtype=np.float32,
            ) / 1000.0
            (
                anchor_source_hand_occlusion,
                anchor_source_occlusion_diagnostics,
            ) = load_anchor_source_hand_occlusion(
                args.initial_hand_mask_root,
                row["sequence"],
                first_frame,
                render_size,
                initial_hand,
                initial_arm,
                initial_object_geometry.support,
                initial_object_geometry.depth_m,
                observed_initial_depth,
                args.object_source_hand_margin,
                args.object_source_front_margin_m,
            )
            unfiltered_appearance = build_anchored_object_appearance(
                target_root,
                first_frame,
                initial_object_pose,
                args.object_state_root,
                initial_object_id,
                source_stride=1,
            )
            anchor_unfiltered_source_points = len(
                unfiltered_appearance.points_object_m
            )
            anchored_object_appearance = build_anchored_object_appearance(
                target_root,
                first_frame,
                initial_object_pose,
                args.object_state_root,
                initial_object_id,
                source_stride=1,
                source_hand_occlusion_mask=anchor_source_hand_occlusion,
            )
            initial_object_layer = render_anchored_object_appearance(
                anchored_object_appearance,
                initial_object_pose,
                intrinsics,
                initial_pose,
                args.object_state_root,
                output_size=render_size,
            )
        else:
            object_builder = (
                annotated_exo_object_layer
                if args.annotated_exo_object_mode == "appearance"
                else annotated_exo_object_geometry_layer
            )
            initial_object_layer = object_builder(
                source_roots,
                first_frame,
                intrinsics,
                initial_pose,
                args.object_state_root,
                output_size=render_size,
                **({"source_stride": 2} if args.annotated_exo_object_mode == "appearance" else {}),
            )
    initial_dynamic = np.maximum(initial_hand, initial_arm) > 0.05
    # Background construction is deliberately conservative: remove every
    # detected moving surface so it cannot become a static ghost. This mask is
    # distinct from the much stricter foreground mask composited at the end.
    initial_dynamic |= initial_motion
    if (
        initial_object_layer is not None
        and args.annotated_exo_object_mode != "cad_filter"
    ):
        initial_dynamic |= initial_object_layer.support
    raw_dynamic = np.asarray(
        Image.fromarray(initial_dynamic.astype(np.uint8) * 255).resize(
            (initial_depth.shape[1], initial_depth.shape[0]), Image.Resampling.NEAREST
        )
    ) > 0
    static_depth = initial_depth.copy()
    static_depth[raw_dynamic] = 0
    if args.disable_ego_anchor:
        previous_generated_background = None
        previous_pose = None
    else:
        initial_background_observed = np.where(
            (~initial_dynamic)[..., None], initial_rgb, 0.0
        )
        generated_background, _ = generate_unknown_background(
            initial_background_observed, ~initial_dynamic
        )
        previous_generated_background = generated_background
        previous_pose = initial_pose

    tile_size, gap = render_size, 8
    columns, rows = 5, 2
    canvas_width = tile_size * columns + gap * (columns - 1)
    tile_height = tile_size + 44
    canvas_height = tile_height * rows + gap * (rows - 1)
    rendered_frames = []
    export_directories = {}
    export_records = []
    if args.model_input_root is not None:
        for name in (
            "input_frames", "repair_masks", "unknown_masks", "seam_masks",
            "foreground_masks", "arm_masks", "object_masks",
            "dynamic_completion_masks", "provenance_labels",
            "geometry_frames", "background_frames", "anchor_layer_frames",
            "exo_layer_frames", "object_layer_frames",
        ):
            directory = args.model_input_root / name
            directory.mkdir(parents=True, exist_ok=True)
            export_directories[name] = directory
        if args.annotated_exo_object_mode.startswith("anchor_warp"):
            Image.fromarray(
                anchor_source_hand_occlusion.astype(np.uint8) * 255, "L"
            ).save(args.model_input_root / "anchor_source_hand_exclusion.png")
    for time_index, frame in enumerate(frames):
        object_layer_visual = np.zeros(
            (render_size, render_size, 3), dtype=np.float32
        )
        anchor_layer_visual = np.zeros_like(object_layer_visual)
        exo_layer_visual = np.zeros_like(object_layer_visual)
        pose = poses[frame]
        exo_inputs = [
            np.asarray(
                Image.open(camera / "rgb" / f"{frame:06d}.png").convert("RGB").resize(
                    (render_size, render_size), Image.Resampling.BILINEAR
                )
            ).astype(np.float32) / 255.0
            for camera in source_roots
        ]
        gt = np.asarray(
            Image.open(target_root / "rgb" / f"{frame:06d}.png").convert("RGB").resize(
                (render_size, render_size), Image.Resampling.BILINEAR
            )
        ).astype(np.float32) / 255.0
        if time_index == 0 and not args.disable_ego_anchor:
            static_rgb = initial_rgb.copy()
            static_valid = np.ones((render_size, render_size), dtype=bool)
            current_valid = np.zeros((render_size, render_size), dtype=bool)
            history_valid = np.zeros((render_size, render_size), dtype=bool)
            foreground_valid = np.zeros((render_size, render_size), dtype=bool)
            raw_foreground_valid = np.zeros((render_size, render_size), dtype=bool)
            arm_valid = np.zeros((render_size, render_size), dtype=bool)
            object_valid = np.zeros((render_size, render_size), dtype=bool)
            dynamic_completion = np.zeros((render_size, render_size), dtype=bool)
            dynamic_support = np.zeros((render_size, render_size), dtype=np.float32)
            geometry_rgb = initial_rgb.copy()
            video_output = initial_rgb.copy()
            anchor_background_valid = ~initial_dynamic
            anchor_layer_visual[anchor_background_valid] = initial_rgb[
                anchor_background_valid
            ]
        else:
            if args.disable_ego_anchor:
                static = None
                static_rgb = np.zeros((render_size, render_size, 3), dtype=np.float32)
                static_valid = np.zeros((render_size, render_size), dtype=bool)
            else:
                static = reproject_rgbd(
                    initial_rgb_u8, static_depth, intrinsics, initial_pose, intrinsics, pose,
                    output_size=(render_size, render_size), source_stride=2,
                )
                static_rgb = static.rgb.astype(np.float32) / 255.0
                static_valid = static.valid
            current_static = causal_static_video_candidates(
                source_roots, [frame], target_root, pose, render_size, 2,
                student_frames, student_joints, student_confidence,
                arm_frames, arm_points, arm_confidence, motion_masks,
                minimum_agree=min(2, len(source_roots)),
            )
            current_rgb, current_valid = current_static["farthest"]
            current_geometry, current_geometry_valid = feather_fill_layer(
                static_rgb,
                static_valid,
                current_rgb,
                current_valid,
                args.source_feather_radius,
                args.source_color_align,
            )
            # A bounded causal window keeps the continuous render tractable:
            # up to five real current/past frames, never future frames.
            history_frames = frames[max(0, time_index - 16) : time_index + 1 : 4]
            if history_frames[-1] != frame:
                history_frames.append(frame)
            history = causal_static_video_candidates(
                source_roots, history_frames, target_root, pose, render_size, 2,
                student_frames, student_joints, student_confidence,
                arm_frames, arm_points, arm_confidence, motion_masks,
                minimum_agree=min(2, len(source_roots) * len(history_frames)),
            )
            history_rgb, history_valid = history["farthest"]
            # Export the real RGB stream surviving motion/depth/consensus
            # filtering. Current-frame observations take precedence; history
            # only fills pixels not seen in the current frame. This layer is
            # diagnostic and never introduces new pixels into the composite.
            exo_layer_visual[current_valid] = current_rgb[current_valid]
            history_only_visual = (~current_valid) & history_valid
            exo_layer_visual[history_only_visual] = history_rgb[
                history_only_visual
            ]
            geometry_rgb, geometry_known = feather_fill_layer(
                current_geometry,
                current_geometry_valid,
                history_rgb,
                history_valid,
                args.source_feather_radius,
                args.source_color_align,
            )
            anchor_layer_visual[static_valid] = static_rgb[static_valid]
            state_index = int(np.searchsorted(student_frames, frame))
            arm_index = int(np.searchsorted(arm_frames, frame))
            hand_support = hand_support_in_pose(
                student_joints[state_index], student_confidence[state_index],
                intrinsics, pose, render_size,
            )
            arm_support = arm_support_in_pose(
                arm_points[arm_index], arm_confidence[arm_index], intrinsics, pose, render_size,
            )
            dynamic_support = np.maximum(hand_support, arm_support)
            source_person_masks = None
            if (
                arm_person_masks is not None
                and arm_person_camera_indices is not None
            ):
                stored_camera_to_index = {
                    int(camera): index
                    for index, camera in enumerate(arm_person_camera_indices)
                }
                source_person_masks = [
                    (
                        arm_person_masks[arm_index, stored_camera_to_index[camera]]
                        if camera in stored_camera_to_index else None
                    )
                    for camera in source_indices
                ]
            (
                foreground_rgb,
                foreground_depth,
                raw_foreground_valid,
                limb_observed_valid,
            ) = multiview_dynamic_arm_candidate(
                source_roots, frame, target_root, pose, render_size, 2,
                student_joints[state_index], student_confidence[state_index],
                arm_points[arm_index], arm_confidence[arm_index],
                motion_masks,
                source_person_masks,
            )
            foreground_valid, arm_valid, object_valid = classify_dynamic_foreground(
                raw_foreground_valid, hand_support, arm_support, limb_observed_valid
            )
            heuristic_object_valid = object_valid.copy()
            object_layer = None
            if args.annotated_exo_object:
                if args.annotated_exo_object_mode.startswith("anchor_warp"):
                    if anchored_object_appearance is None:
                        raise RuntimeError("Missing anchored object appearance")
                    object_id, object_pose, _ = annotated_exo_object_pose_world(
                        source_roots, frame
                    )
                    if object_id != anchored_object_appearance.object_id:
                        raise ValueError("Object identity changed within clip")
                    object_layer = render_anchored_object_appearance(
                        anchored_object_appearance,
                        object_pose,
                        intrinsics,
                        pose,
                        args.object_state_root,
                        output_size=render_size,
                    )
                    object_valid = object_layer.valid.copy()
                    object_rgb = object_layer.rgb.copy()
                    object_depth = object_layer.depth_m.copy()
                    if args.annotated_exo_object_mode == "anchor_warp_exo_fill":
                        exo_layer = annotated_exo_object_layer(
                            source_roots,
                            frame,
                            intrinsics,
                            pose,
                            args.object_state_root,
                            output_size=render_size,
                            source_stride=2,
                        )
                        exo_rgb = exo_layer.rgb.copy()
                        overlap = object_valid & exo_layer.valid
                        if overlap.sum() >= 32:
                            correction = np.median(
                                object_rgb[overlap] - exo_rgb[overlap], axis=0
                            )
                            exo_rgb = np.clip(
                                exo_rgb + np.clip(correction, -0.08, 0.08), 0.0, 1.0
                            )
                        fill = (~object_valid) & exo_layer.valid
                        object_rgb[fill] = exo_rgb[fill]
                        object_depth[fill] = exo_layer.depth_m[fill]
                        object_valid |= fill
                else:
                    object_builder = (
                        annotated_exo_object_layer
                        if args.annotated_exo_object_mode == "appearance"
                        else annotated_exo_object_geometry_layer
                    )
                    object_layer = object_builder(
                        source_roots,
                        frame,
                        intrinsics,
                        pose,
                        args.object_state_root,
                        output_size=render_size,
                        **({"source_stride": 2} if args.annotated_exo_object_mode == "appearance" else {}),
                    )
                if args.annotated_exo_object_mode == "appearance":
                    object_valid = object_layer.valid
                    object_rgb = object_layer.rgb
                    object_depth = object_layer.depth_m
                elif args.annotated_exo_object_mode == "cad_gate":
                    # CAD defines identity/support; RGB-D remains a real current
                    # exo observation and is never densified across an edge.
                    object_valid = raw_foreground_valid & dilated_mask(
                        object_layer.support, 5
                    )
                    object_rgb = foreground_rgb
                    object_depth = foreground_depth
                elif args.annotated_exo_object_mode == "cad_filter":
                    # Precision-first use: CAD may reject the loose hand-neighbour
                    # proxy, but cannot introduce pixels unsupported by that proxy.
                    object_valid = heuristic_object_valid & dilated_mask(
                        object_layer.support, 5
                    )
                    object_rgb = foreground_rgb
                    object_depth = foreground_depth
                # The object surface is dynamic evidence too and must be guarded
                # from background inpainting.
                if args.annotated_exo_object_mode == "cad_filter":
                    dynamic_support = np.maximum(
                        dynamic_support, object_valid.astype(np.float32)
                    )
                else:
                    dynamic_support = np.maximum(
                        dynamic_support, object_layer.support.astype(np.float32)
                    )
            observed_depth = (
                static.depth_m[static.valid]
                if static is not None else np.empty(0, dtype=np.float32)
            )
            nominal_depth_m = (
                float(np.nanmedian(observed_depth)) if observed_depth.size else 0.8
            )
            geometry_background, _ = generate_unknown_background(
                geometry_rgb,
                geometry_known,
                previous_generated=previous_generated_background,
                previous_pose=previous_pose,
                current_pose=pose,
                intrinsics=intrinsics,
                nominal_depth_m=nominal_depth_m,
            )
            pose_reliability = pose_reliabilities[frame]
            if (
                pose_reliability < 1.0
                and previous_generated_background is not None
                and previous_pose is not None
            ):
                # The old compositor hard-pasted every reprojected pixel even
                # when the pose estimator itself said the frame was weak.  A
                # calibrated low-confidence frame now falls back continuously
                # to the trajectory-warped prior instead of turning uncertain
                # geometry into an irreversible hard constraint.
                fallback_background, _ = generate_unknown_background(
                    np.zeros_like(geometry_rgb),
                    np.zeros_like(geometry_known),
                    previous_generated=previous_generated_background,
                    previous_pose=previous_pose,
                    current_pose=pose,
                    intrinsics=intrinsics,
                    nominal_depth_m=nominal_depth_m,
                )
                generated_background = (
                    pose_reliability * geometry_background
                    + (1.0 - pose_reliability) * fallback_background
                )
            else:
                generated_background = geometry_background
            if object_layer is not None:
                # Resolve hand/object occlusion in metric target-camera depth.
                arm_in_front = arm_valid & (
                    (~object_valid)
                    | (~np.isfinite(object_depth))
                    | (
                        np.isfinite(foreground_depth)
                        & (foreground_depth <= object_depth + 0.01)
                    )
                )
                object_visible = object_valid & (~arm_in_front)
                foreground_valid = arm_in_front | object_visible
                arm_valid = arm_in_front
                object_valid = object_visible
                video_output = generated_background.copy()
                object_alpha = (
                    args.object_alpha
                    if args.annotated_exo_object_mode.startswith("anchor_warp")
                    else 1.0
                )
                video_output[object_visible] = (
                    (1.0 - object_alpha) * video_output[object_visible]
                    + object_alpha * object_rgb[object_visible]
                )
                object_layer_visual[object_visible] = object_rgb[object_visible]
                video_output[arm_in_front] = foreground_rgb[arm_in_front]
            else:
                dynamic_alpha = foreground_valid.astype(np.float32)
                video_output = (
                    generated_background * (1.0 - dynamic_alpha[..., None])
                    + foreground_rgb * dynamic_alpha[..., None]
                )
            dynamic_completion = (
                missing_limb_completion_mask(
                    hand_support, arm_support, arm_valid, object_valid
                )
                if completion_enabled
                else np.zeros((render_size, render_size), dtype=bool)
            )
            previous_generated_background = generated_background
            previous_pose = pose
        if args.annotated_exo_object_mode.startswith("anchor_warp"):
            object_label = "物体刚体输运"
        elif args.annotated_exo_object:
            object_label = "标注物体层"
        else:
            object_label = "持物近邻"
        provenance_mask = channel_mask_image(
            static_valid, current_valid, history_valid, arm_valid, object_valid,
            dynamic_completion=(dynamic_completion if completion_enabled else None),
            object_label=object_label,
        )
        labels, unknown_mask, seam_mask, repair_mask = model_repair_masks(
            static_valid,
            current_valid,
            history_valid,
            foreground_valid,
            dynamic_support,
            dynamic_completion,
        )
        if export_directories:
            filename = f"{time_index:06d}.png"
            Image.fromarray(
                np.rint(np.clip(video_output, 0, 1) * 255).astype(np.uint8), "RGB"
            ).save(export_directories["input_frames"] / filename)
            for name, value in (
                ("geometry_frames", geometry_rgb),
                ("background_frames", generated_background),
                ("anchor_layer_frames", anchor_layer_visual),
                ("exo_layer_frames", exo_layer_visual),
                ("object_layer_frames", object_layer_visual),
            ):
                Image.fromarray(
                    np.rint(np.clip(value, 0, 1) * 255).astype(np.uint8), "RGB"
                ).save(export_directories[name] / filename)
            for name, mask in (
                ("repair_masks", repair_mask),
                ("unknown_masks", unknown_mask),
                ("seam_masks", seam_mask),
                ("foreground_masks", foreground_valid),
                ("arm_masks", arm_valid),
                ("object_masks", object_valid),
                ("dynamic_completion_masks", dynamic_completion),
            ):
                Image.fromarray(mask.astype(np.uint8) * 255, "L").save(
                    export_directories[name] / filename
                )
            Image.fromarray(labels, "L").save(
                export_directories["provenance_labels"] / filename
            )
            export_records.append(
                {
                    "index": time_index,
                    "dataset_frame": frame,
                    "unknown_fraction": float(unknown_mask.mean()),
                    "seam_fraction": float(seam_mask.mean()),
                    "repair_fraction": float(repair_mask.mean()),
                    "foreground_fraction": float(foreground_valid.mean()),
                    "raw_foreground_fraction": float(raw_foreground_valid.mean()),
                    "arm_fraction": float(arm_valid.mean()),
                    "object_fraction": float(object_valid.mean()),
                    "dynamic_completion_fraction": float(dynamic_completion.mean()),
                    "predicted_camera_pose_world": pose.tolist(),
                    "predicted_camera_pose_confidence": pose_confidences[frame],
                    "calibrated_camera_pose_reliability": pose_reliabilities[frame],
                }
            )
        tiles = [
            *[
                title_tile(
                    pil_rgb(exo_input, tile_size),
                    f"输入 {camera.name} · {frame:06d}",
                    tile_size,
                    INPUT_BORDER,
                )
                for camera, exo_input in zip(source_roots, exo_inputs)
            ],
            title_tile(
                pil_rgb(initial_rgb, tile_size),
                f"输入 ego 首帧 · {first_frame:06d}",
                tile_size,
                INPUT_BORDER,
            ),
            title_tile(pil_rgb(geometry_rgb, tile_size), "几何观测（黑色为未知）", tile_size, INTERMEDIATE_BORDER),
            title_tile(provenance_mask, "信息通道蒙版", tile_size, INTERMEDIATE_BORDER),
            title_tile(pil_rgb(generated_background, tile_size), "运动约束背景生成", tile_size, PREDICTION_BORDER),
            title_tile(pil_rgb(video_output, tile_size), "稳定视频合成", tile_size, PREDICTION_BORDER),
            title_tile(pil_rgb(gt, tile_size), "未来 ego 真值", tile_size, TARGET_BORDER),
        ]
        canvas = Image.new("RGB", (canvas_width, canvas_height), (241, 245, 249))
        for index, tile in enumerate(tiles):
            row_index, column = divmod(index, columns)
            canvas.paste(
                tile,
                (column * (tile_size + gap), row_index * (tile_height + gap)),
            )
        rendered_frames.append(canvas)

    args.output.parent.mkdir(parents=True, exist_ok=True)
    rendered_frames[-1].save(poster)
    fps = 15
    command = [
        "ffmpeg", "-y", "-loglevel", "error",
        "-f", "rawvideo", "-pix_fmt", "rgb24",
        "-s", f"{canvas_width}x{canvas_height}", "-r", str(fps), "-i", "-",
        "-an", "-c:v", "libx264", "-preset", "slow", "-crf", "15",
        "-pix_fmt", "yuv420p", "-movflags", "+faststart", str(args.output),
    ]
    process = subprocess.Popen(command, stdin=subprocess.PIPE)
    assert process.stdin is not None
    for frame in rendered_frames:
        payload = np.asarray(frame, dtype=np.uint8).tobytes()
        process.stdin.write(payload)
    process.stdin.close()
    return_code = process.wait()
    if return_code != 0:
        raise RuntimeError(f"ffmpeg failed with exit code {return_code}")
    if args.model_input_root is not None:
        ego_first_pose = args.initial_pose_summary is None
        ego_first_rgbd = not args.disable_ego_anchor
        annotated_object_pose = args.annotated_exo_object
        if ego_first_rgbd:
            protocol_name = (
                "anchored_gt_mount_annotated_object"
                if annotated_object_pose else "anchored_gt_mount_no_object"
            )
        elif ego_first_pose:
            protocol_name = (
                "exo_only_gt_mount_annotated_object"
                if annotated_object_pose else "exo_only_gt_mount_no_object"
            )
        else:
            protocol_name = (
                "exo_only_estimated_mount_annotated_object"
                if annotated_object_pose else "exo_only_estimated_mount"
            )
        manifest = {
            "pair_id": args.pair_id,
            "protocol_name": protocol_name,
            "input_contract": {
                "version": LAYERED_CONTRACT_VERSION,
                "dataset_split": row.get("split"),
                "exo_rgbd_and_calibration": True,
                "ego_first_rgbd": ego_first_rgbd,
                "ego_first_pose": ego_first_pose,
                "annotated_object_pose": annotated_object_pose,
                "future_ego_rgb_for_inference": False,
                "future_ego_rgb_read_for_visualization": True,
                "head_hand_arm_state_estimated_from_exo": True,
                "motion_mask_history": (
                    "past_and_present" if args.causal_motion_masks else "whole_clip"
                ),
                "offline_noncausal": not args.causal_motion_masks,
                "head_rotation_scale": args.head_rotation_scale,
                "head_rotation_scale_selection": "validation_tuned",
                "pose_noise_rotation_deg": args.pose_noise_rotation_deg,
                "pose_noise_translation_m": args.pose_noise_translation_m,
                "pose_noise_seed": args.pose_noise_seed,
                "pose_noise_parameters": pose_noise_parameters,
                "frame_step": args.frame_step,
                "pose_confidence_calibration": (
                    str(args.pose_confidence_calibration)
                    if args.pose_confidence_calibration is not None else None
                ),
                "render_composition_uses_pose_reliability": (
                    args.pose_confidence_calibration is not None
                ),
            },
            "frame_count": len(frames),
            "image_size": render_size,
            "mask_policy": (
                "repair = no-observation background union 5px background-provenance "
                "seams; final limb RGB-D must pass metric source-space skeleton "
                "proximity and a 7px target projected-skeleton neighborhood; broad "
                "motion is restricted to the separate 25px held-object neighborhood, "
                "and their union plus predicted support receives a 9px repair guard"
            ),
            "limb_observation_policy": (
                "all appearance-authority views use limb-only metric RGB-D; broad "
                "motion pixels cannot be promoted to observed arm texture; optional "
                "source-view person probabilities further reject nearby table RGB-D"
            ),
            "source_person_segmentation_available": arm_person_masks is not None,
            "limb_secondary_appearance_radius": LIMB_SECONDARY_APPEARANCE_RADIUS,
            "limb_densify_iterations": LIMB_DENSIFY_ITERATIONS,
            "source_feather_radius": args.source_feather_radius,
            "source_color_align": args.source_color_align,
            "annotated_exo_object": args.annotated_exo_object,
            "annotated_exo_object_mode": args.annotated_exo_object_mode,
            "object_alpha": args.object_alpha,
            "source_camera_indices": list(source_indices),
            "state_camera_indices": list(state_indices),
            "ego_anchor_enabled": not args.disable_ego_anchor,
            "single_view_limb_completion_enabled": args.complete_single_view_limbs,
            "missing_limb_completion_enabled": completion_enabled,
            "initial_pose_source": initial_pose_source,
            "ego_first_pose_world": initial_pose.tolist(),
            "initial_head_camera_relationship_input": args.initial_pose_summary is None,
            "ablation_scope": (
                "appearance and state streams follow the selected source/state cameras; "
                "the optional initial-pose summary additionally withholds the target "
                "head-camera relationship"
            ),
            "fixed_state_dependencies": {
                "hand": f"MediaPipe 3D estimate from exo cameras {list(state_indices)}",
                "arm": f"MediaPipe RGB-D world points from exo cameras {list(state_indices)}",
                "head": f"temporally smoothed face-pose estimate from exo cameras {list(state_indices)}",
                "object": (
                    "annotated object SE(3) from selected exo annotation stream"
                    if args.annotated_exo_object else "disabled"
                ),
            },
            "object_source_hand_occlusion": {
                "mask_source": str(
                    args.initial_hand_mask_root / row["sequence"]
                    / "initial_hand_masks.npz"
                ),
                "safety_margin_px": args.object_source_hand_margin,
                "front_margin_m": args.object_source_front_margin_m,
                **anchor_source_occlusion_diagnostics,
                "retained_object_source_points": (
                    int(len(anchored_object_appearance.points_object_m))
                    if anchored_object_appearance is not None else None
                ),
                "unfiltered_object_source_points": anchor_unfiltered_source_points,
                "rejected_object_source_points": (
                    int(
                        anchor_unfiltered_source_points
                        - len(anchored_object_appearance.points_object_m)
                    )
                    if anchored_object_appearance is not None
                    and anchor_unfiltered_source_points is not None else None
                ),
                "policy": "excluded pixels remain unknown and are never transported",
            },
            "head_pose_source": "temporally_smoothed_exo_face",
            "frames": export_records,
        }
        (args.model_input_root / "manifest.json").write_text(
            json.dumps(manifest, indent=2, ensure_ascii=False), encoding="utf-8"
        )
    print(args.output)
    print(poster)


if __name__ == "__main__":
    main()
