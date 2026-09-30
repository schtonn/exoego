#!/usr/bin/env python3
"""Stabilize jagged hand boundaries with flow-aligned signed-distance consensus."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import warnings

import cv2
import numpy as np
from PIL import Image


def load_rgb(path: Path, size: int) -> np.ndarray:
    return np.asarray(
        Image.open(path).convert("RGB").resize((size, size), Image.Resampling.BILINEAR)
    )


def load_mask(path: Path, size: int) -> np.ndarray:
    return np.asarray(
        Image.open(path).convert("L").resize((size, size), Image.Resampling.NEAREST)
    ) > 0


def signed_distance(mask: np.ndarray, limit: float = 6.0) -> np.ndarray:
    inside = cv2.distanceTransform(mask.astype(np.uint8), cv2.DIST_L2, 5)
    outside = cv2.distanceTransform((~mask).astype(np.uint8), cv2.DIST_L2, 5)
    return np.clip(inside - outside, -limit, limit).astype(np.float32)


def flow_maps(
    current: np.ndarray, neighbor: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    current_gray = cv2.cvtColor(current, cv2.COLOR_RGB2GRAY)
    neighbor_gray = cv2.cvtColor(neighbor, cv2.COLOR_RGB2GRAY)
    forward = cv2.calcOpticalFlowFarneback(
        current_gray, neighbor_gray, None, 0.5, 4, 21, 4, 7, 1.5, 0
    )
    backward = cv2.calcOpticalFlowFarneback(
        neighbor_gray, current_gray, None, 0.5, 4, 21, 4, 7, 1.5, 0
    )
    height, width = current_gray.shape
    grid_x, grid_y = np.meshgrid(
        np.arange(width, dtype=np.float32), np.arange(height, dtype=np.float32)
    )
    map_x = grid_x + forward[..., 0]
    map_y = grid_y + forward[..., 1]
    sampled_backward = cv2.remap(
        backward, map_x, map_y, cv2.INTER_LINEAR, borderMode=cv2.BORDER_CONSTANT
    )
    consistent = np.linalg.norm(forward + sampled_backward, axis=2) <= 1.5
    consistent &= (
        (map_x >= 0) & (map_x <= width - 1)
        & (map_y >= 0) & (map_y <= height - 1)
    )
    return map_x, map_y, consistent


def remap(value: np.ndarray, map_x: np.ndarray, map_y: np.ndarray, nearest: bool = False) -> np.ndarray:
    return cv2.remap(
        value,
        map_x,
        map_y,
        cv2.INTER_NEAREST if nearest else cv2.INTER_LINEAR,
        borderMode=cv2.BORDER_CONSTANT,
    )


def boundary_perimeter(mask: np.ndarray) -> float:
    contours, _ = cv2.findContours(
        mask.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE
    )
    return float(sum(cv2.arcLength(contour, True) for contour in contours))


def boundary_roughness(mask: np.ndarray, sigma: float = 1.25) -> int:
    """Pixels attributable to sub-scale spikes/bays, not overall contour length."""
    sdf = signed_distance(mask, limit=4.0)
    regular = cv2.GaussianBlur(sdf, (0, 0), sigmaX=sigma, sigmaY=sigma) > 0
    return int((mask ^ regular).sum())


def flow_aligned_xor(
    current_mask: np.ndarray,
    previous_mask: np.ndarray,
    current_rgb: np.ndarray,
    previous_rgb: np.ndarray,
) -> tuple[int, int]:
    map_x, map_y, consistent = flow_maps(current_rgb, previous_rgb)
    warped = remap(previous_mask.astype(np.uint8), map_x, map_y, nearest=True) > 0
    union = current_mask | warped
    valid = consistent & cv2.dilate(
        union.astype(np.uint8), np.ones((5, 5), dtype=np.uint8)
    ).astype(bool)
    return int(((current_mask ^ warped) & valid).sum()), int(valid.sum())


def l1(first: np.ndarray, second: np.ndarray, mask: np.ndarray | None = None) -> float:
    error = np.abs(first.astype(np.float32) - second.astype(np.float32)) / 255.0
    if mask is not None:
        return float(error[mask].mean()) if np.any(mask) else 0.0
    return float(error.mean())


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--sequence-root", type=Path, required=True)
    parser.add_argument("--model-input-root", type=Path, required=True)
    parser.add_argument("--refined-root", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--temporal-radius", type=int, default=2)
    parser.add_argument("--boundary-band-px", type=float, default=3.5)
    parser.add_argument("--add-support", type=float, default=0.60)
    parser.add_argument("--remove-support", type=float, default=0.40)
    parser.add_argument("--object-guard-px", type=int, default=2)
    parser.add_argument("--alpha-width-px", type=float, default=1.25)
    parser.add_argument("--spatial-sigma", type=float, default=0.9)
    args = parser.parse_args()

    manifest = json.loads((args.model_input_root / "manifest.json").read_text())
    frames = [int(value["dataset_frame"]) for value in manifest["frames"]]
    size = int(manifest["image_size"])
    names = [f"{index:06d}.png" for index in range(len(frames))]
    rgbs = [load_rgb(args.refined_root / "frames" / name, size) for name in names]
    hole_masks = [load_mask(args.refined_root / "fill_masks" / name, size) for name in names]
    raw_masks = [load_mask(args.model_input_root / "arm_masks" / name, size) for name in names]
    masks = [raw | hole for raw, hole in zip(raw_masks, hole_masks)]
    object_masks = [load_mask(args.model_input_root / "object_masks" / name, size) for name in names]
    backgrounds = [load_rgb(args.model_input_root / "background_frames" / name, size) for name in names]
    object_rgbs = [load_rgb(args.model_input_root / "object_layer_frames" / name, size) for name in names]
    sdfs = [signed_distance(mask) for mask in masks]

    output_frames = args.output_root / "frames"
    output_masks = args.output_root / "arm_masks"
    output_added = args.output_root / "added_masks"
    output_removed = args.output_root / "removed_masks"
    for directory in (output_frames, output_masks, output_added, output_removed):
        directory.mkdir(parents=True, exist_ok=True)

    stable_masks: list[np.ndarray] = []
    outputs: list[np.ndarray] = []
    per_frame = []
    for index, current in enumerate(rgbs):
        current_mask = masks[index]
        distance_candidates = [sdfs[index], sdfs[index]]
        mask_candidates = [current_mask.astype(np.float32), current_mask.astype(np.float32)]
        texture_candidates: list[np.ndarray] = []
        texture_valid: list[np.ndarray] = []
        for offset in range(-args.temporal_radius, args.temporal_radius + 1):
            neighbor_index = index + offset
            if offset == 0 or not (0 <= neighbor_index < len(frames)):
                continue
            map_x, map_y, consistent = flow_maps(current, rgbs[neighbor_index])
            warped_sdf = remap(sdfs[neighbor_index], map_x, map_y)
            warped_mask = remap(
                masks[neighbor_index].astype(np.uint8), map_x, map_y, nearest=True
            ) > 0
            warped_rgb = remap(rgbs[neighbor_index], map_x, map_y)
            distance_candidates.append(np.where(consistent, warped_sdf, np.nan))
            mask_candidates.append(np.where(consistent, warped_mask.astype(np.float32), np.nan))
            texture_candidates.append(warped_rgb)
            texture_valid.append(consistent & warped_mask)

        with warnings.catch_warnings():
            warnings.simplefilter("ignore", category=RuntimeWarning)
            consensus_sdf = np.nanmedian(np.stack(distance_candidates), axis=0)
            support = np.nanmean(np.stack(mask_candidates), axis=0)
        consensus_sdf = cv2.bilateralFilter(
            consensus_sdf.astype(np.float32), 5, 1.25, 1.8
        )
        if args.spatial_sigma > 0:
            consensus_sdf = cv2.GaussianBlur(
                consensus_sdf,
                (0, 0),
                sigmaX=args.spatial_sigma,
                sigmaY=args.spatial_sigma,
            )
        current_sdf = sdfs[index]
        boundary_band = np.abs(current_sdf) <= args.boundary_band_px
        proposal = consensus_sdf > 0
        object_guard = cv2.dilate(
            object_masks[index].astype(np.uint8),
            cv2.getStructuringElement(
                cv2.MORPH_ELLIPSE,
                (2 * args.object_guard_px + 1, 2 * args.object_guard_px + 1),
            ),
        ) > 0
        added = (
            (~current_mask) & proposal & boundary_band
            & (support >= args.add_support) & (~object_guard)
        )
        removed = (
            current_mask & (~proposal) & boundary_band
            & (support <= args.remove_support)
        )
        stable = (current_mask | added) & (~removed)

        foreground = current.copy()
        if np.any(added):
            fallback = cv2.inpaint(
                current, added.astype(np.uint8) * 255, 3.0, cv2.INPAINT_NS
            )
            foreground[added] = fallback[added]
            if texture_candidates:
                values = np.stack(texture_candidates).astype(np.float32)
                valid = np.stack(texture_valid)
                values[~valid] = np.nan
                with warnings.catch_warnings():
                    warnings.simplefilter("ignore", category=RuntimeWarning)
                    temporal_rgb = np.nanmedian(values, axis=0)
                temporal_valid = added & (valid.sum(axis=0) >= 2) & np.isfinite(temporal_rgb).all(axis=2)
                foreground[temporal_valid] = np.rint(temporal_rgb[temporal_valid]).astype(np.uint8)

        background = backgrounds[index].copy()
        background[object_masks[index]] = object_rgbs[index][object_masks[index]]
        hard = current.copy()
        hard[added] = foreground[added]
        hard[removed] = background[removed]

        stable_sdf = signed_distance(stable, limit=4.0)
        alpha = np.clip(0.5 + stable_sdf / (2.0 * args.alpha_width_px), 0.0, 1.0)
        alpha_band = np.abs(stable_sdf) <= args.alpha_width_px
        # Fill the narrow outside half of the antialias band with the same
        # foreground proposal, without authorizing semantic expansion.
        outside_band = alpha_band & (~stable)
        if np.any(outside_band):
            foreground_band = cv2.inpaint(
                hard, outside_band.astype(np.uint8) * 255, 2.0, cv2.INPAINT_NS
            )
            foreground[outside_band] = foreground_band[outside_band]
        output = hard.astype(np.float32)
        blend = alpha[..., None] * foreground + (1.0 - alpha[..., None]) * background
        output[alpha_band] = blend[alpha_band]
        output = np.rint(np.clip(output, 0, 255)).astype(np.uint8)

        stable_masks.append(stable)
        outputs.append(output)
        Image.fromarray(output, "RGB").save(output_frames / names[index])
        Image.fromarray(stable.astype(np.uint8) * 255, "L").save(output_masks / names[index])
        Image.fromarray(added.astype(np.uint8) * 255, "L").save(output_added / names[index])
        Image.fromarray(removed.astype(np.uint8) * 255, "L").save(output_removed / names[index])
        per_frame.append({
            "frame": frames[index],
            "added_pixels": int(added.sum()),
            "removed_pixels": int(removed.sum()),
            "changed_mask_pixels": int((added | removed).sum()),
        })

    raw_xor = raw_valid = stable_xor = stable_valid = 0
    raw_perimeter = stable_perimeter = 0.0
    raw_roughness = stable_roughness = 0
    original_l1 = refined_l1 = 0.0
    boundary_original_l1 = boundary_refined_l1 = 0.0
    boundary_frames = 0
    for index, frame in enumerate(frames):
        raw_perimeter += boundary_perimeter(masks[index])
        stable_perimeter += boundary_perimeter(stable_masks[index])
        raw_roughness += boundary_roughness(masks[index])
        stable_roughness += boundary_roughness(stable_masks[index])
        target = load_rgb(args.sequence_root / "cam4/rgb" / f"{frame:06d}.png", size)
        original_l1 += l1(rgbs[index], target)
        refined_l1 += l1(outputs[index], target)
        evaluation_band = cv2.dilate(
            (masks[index] ^ stable_masks[index]).astype(np.uint8),
            np.ones((5, 5), dtype=np.uint8),
        ) > 0
        if np.any(evaluation_band):
            boundary_original_l1 += l1(rgbs[index], target, evaluation_band)
            boundary_refined_l1 += l1(outputs[index], target, evaluation_band)
            boundary_frames += 1
        if index > 0:
            numerator, denominator = flow_aligned_xor(
                masks[index], masks[index - 1], rgbs[index], rgbs[index - 1]
            )
            raw_xor += numerator
            raw_valid += denominator
            numerator, denominator = flow_aligned_xor(
                stable_masks[index], stable_masks[index - 1], outputs[index], outputs[index - 1]
            )
            stable_xor += numerator
            stable_valid += denominator

    metrics = {
        "protocol": {
            "temporal_radius": args.temporal_radius,
            "boundary_band_px": args.boundary_band_px,
            "add_support": args.add_support,
            "remove_support": args.remove_support,
            "object_guard_px": args.object_guard_px,
            "alpha_width_px": args.alpha_width_px,
            "spatial_sigma": args.spatial_sigma,
            "future_ego_used_for_inference": False,
        },
        "frames": len(frames),
        "added_pixels": int(sum(value["added_pixels"] for value in per_frame)),
        "removed_pixels": int(sum(value["removed_pixels"] for value in per_frame)),
        "changed_mask_pixels": int(sum(value["changed_mask_pixels"] for value in per_frame)),
        "flow_aligned_boundary_xor_before": raw_xor / max(raw_valid, 1),
        "flow_aligned_boundary_xor_after": stable_xor / max(stable_valid, 1),
        "relative_boundary_xor_change": (
            (stable_xor / max(stable_valid, 1)) / max(raw_xor / max(raw_valid, 1), 1e-12) - 1.0
        ),
        "mean_perimeter_before": raw_perimeter / len(frames),
        "mean_perimeter_after": stable_perimeter / len(frames),
        "relative_perimeter_change": stable_perimeter / max(raw_perimeter, 1e-12) - 1.0,
        "mean_high_frequency_boundary_pixels_before": raw_roughness / len(frames),
        "mean_high_frequency_boundary_pixels_after": stable_roughness / len(frames),
        "relative_high_frequency_boundary_change": (
            stable_roughness / max(raw_roughness, 1) - 1.0
        ),
        "original_l1": original_l1 / len(frames),
        "refined_l1": refined_l1 / len(frames),
        "relative_full_l1_change": refined_l1 / max(original_l1, 1e-12) - 1.0,
        "boundary_original_l1": boundary_original_l1 / max(boundary_frames, 1),
        "boundary_refined_l1": boundary_refined_l1 / max(boundary_frames, 1),
        "relative_boundary_l1_change": (
            boundary_refined_l1 / max(boundary_original_l1, 1e-12) - 1.0
            if boundary_frames else 0.0
        ),
        "per_frame": per_frame,
    }
    (args.output_root / "metrics.json").write_text(
        json.dumps(metrics, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )
    print(json.dumps({key: value for key, value in metrics.items() if key != "per_frame"}, indent=2))


if __name__ == "__main__":
    main()
