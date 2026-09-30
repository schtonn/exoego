#!/usr/bin/env python3
"""Use flow-consistent hand observations to validate and texture larger holes."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import warnings

import cv2
import numpy as np
from PIL import Image


def load_rgb(path: Path) -> np.ndarray:
    return np.asarray(Image.open(path).convert("RGB"))


def load_mask(path: Path) -> np.ndarray:
    return np.asarray(Image.open(path)) > 0


def l1(first: np.ndarray, second: np.ndarray, mask: np.ndarray | None = None) -> float:
    error = np.abs(first.astype(np.float32) - second.astype(np.float32)) / 255.0
    if mask is not None:
        return float(error[mask].mean()) if np.any(mask) else 0.0
    return float(error.mean())


def boundary_jump(rgb: np.ndarray, mask: np.ndarray) -> tuple[float, int]:
    """Sum RGB jumps across the authorized-region boundary."""
    value = rgb.astype(np.float32) / 255.0
    total = 0.0
    count = 0
    for first, second, crossing in (
        (value[:, :-1], value[:, 1:], mask[:, :-1] != mask[:, 1:]),
        (value[:-1], value[1:], mask[:-1] != mask[1:]),
    ):
        if np.any(crossing):
            total += float(np.abs(first - second).mean(axis=2)[crossing].sum())
            count += int(crossing.sum())
    return total, count


def flow_candidate(
    current: np.ndarray,
    neighbor: np.ndarray,
    neighbor_hand: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
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
    warped_rgb = cv2.remap(
        neighbor, map_x, map_y, cv2.INTER_LINEAR,
        borderMode=cv2.BORDER_REFLECT_101,
    )
    warped_hand = cv2.remap(
        neighbor_hand.astype(np.uint8), map_x, map_y, cv2.INTER_NEAREST,
        borderMode=cv2.BORDER_CONSTANT,
    ) > 0
    sampled_backward = cv2.remap(
        backward, map_x, map_y, cv2.INTER_LINEAR,
        borderMode=cv2.BORDER_CONSTANT,
    )
    consistency = np.linalg.norm(forward + sampled_backward, axis=2) <= 1.5
    in_bounds = (
        (map_x >= 0) & (map_x <= width - 1)
        & (map_y >= 0) & (map_y <= height - 1)
    )
    return warped_rgb, warped_hand & consistency & in_bounds


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--sequence-root", type=Path, required=True)
    parser.add_argument("--model-input-root", type=Path, required=True)
    parser.add_argument("--prediction-root", type=Path, required=True)
    parser.add_argument("--selected-hole-root", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--always-fill-area", type=int, default=24)
    parser.add_argument("--minimum-temporal-support", type=float, default=0.50)
    parser.add_argument("--temporal-radius", type=int, default=2)
    args = parser.parse_args()

    manifest = json.loads((args.model_input_root / "manifest.json").read_text())
    frames = [int(value["dataset_frame"]) for value in manifest["frames"]]
    size = int(manifest["image_size"])
    names = [f"{index:06d}.png" for index in range(len(frames))]
    predictions = [load_rgb(args.prediction_root / name) for name in names]
    arm_masks = [load_mask(args.model_input_root / "arm_masks" / name) for name in names]
    proposed_masks = [load_mask(args.selected_hole_root / name) for name in names]

    output_frames = args.output_root / "frames"
    output_masks = args.output_root / "fill_masks"
    output_support = args.output_root / "temporal_support"
    for directory in (output_frames, output_masks, output_support):
        directory.mkdir(parents=True, exist_ok=True)

    totals = {
        "proposed_components": 0,
        "accepted_components": 0,
        "rejected_large_components": 0,
        "accepted_pixels": 0,
        "temporally_textured_pixels": 0,
        "original_l1": 0.0,
        "repaired_l1": 0.0,
        "original_fill_l1": 0.0,
        "repaired_fill_l1": 0.0,
        "frames_with_fill": 0,
        "improved_frames": 0,
        "outside_max_difference": 0,
        "boundary_jump_before_sum": 0.0,
        "boundary_jump_after_sum": 0.0,
        "boundary_edge_count": 0,
    }
    per_frame = []
    for index, (frame, current, proposed) in enumerate(
        zip(frames, predictions, proposed_masks)
    ):
        candidates = []
        candidate_valid = []
        for offset in range(-args.temporal_radius, args.temporal_radius + 1):
            neighbor_index = index + offset
            if offset == 0 or not (0 <= neighbor_index < len(frames)):
                continue
            value, valid = flow_candidate(
                current, predictions[neighbor_index], arm_masks[neighbor_index]
            )
            candidates.append(value)
            candidate_valid.append(valid)
        if candidates:
            candidate_stack = np.stack(candidates)
            valid_stack = np.stack(candidate_valid)
            support_count = valid_stack.sum(axis=0)
        else:
            candidate_stack = np.empty((0, size, size, 3), dtype=np.uint8)
            valid_stack = np.empty((0, size, size), dtype=bool)
            support_count = np.zeros((size, size), dtype=np.int32)

        accepted = np.zeros_like(proposed)
        count, labels, stats, _ = cv2.connectedComponentsWithStats(
            proposed.astype(np.uint8), connectivity=8
        )
        proposed_components = accepted_components = rejected_components = 0
        for component_index in range(1, count):
            component = labels == component_index
            area = int(stats[component_index, cv2.CC_STAT_AREA])
            proposed_components += 1
            # A single neighboring flow can be accidentally consistent across a
            # real finger gap. Require agreement from at least two neighbors
            # before a component larger than the unconditional tiny-hole limit
            # is treated as missing hand surface.
            support_fraction = float((component & (support_count >= 2)).sum()) / max(area, 1)
            if area <= args.always_fill_area or support_fraction >= args.minimum_temporal_support:
                accepted |= component
                accepted_components += 1
            else:
                rejected_components += 1

        fallback = cv2.inpaint(
            current, accepted.astype(np.uint8) * 255, 3.0, cv2.INPAINT_NS
        )
        repaired = current.copy()
        repaired[accepted] = fallback[accepted]
        temporal_texture = accepted & (support_count >= 2)
        if np.any(temporal_texture):
            # Median is robust to one bad neighboring warp. Invalid entries are
            # excluded with NaN rather than replaced by black.
            values = candidate_stack.astype(np.float32)
            values[~valid_stack] = np.nan
            with warnings.catch_warnings():
                warnings.simplefilter("ignore", category=RuntimeWarning)
                median = np.nanmedian(values, axis=0)
            valid_median = temporal_texture & np.isfinite(median).all(axis=2)
            repaired[valid_median] = np.rint(median[valid_median]).astype(np.uint8)

        name = names[index]
        Image.fromarray(repaired, "RGB").save(output_frames / name)
        Image.fromarray(accepted.astype(np.uint8) * 255, "L").save(output_masks / name)
        support_visual = np.clip(support_count * 64, 0, 255).astype(np.uint8)
        Image.fromarray(support_visual, "L").save(output_support / name)

        target = np.asarray(
            Image.open(args.sequence_root / "cam4/rgb" / f"{frame:06d}.png")
            .convert("RGB").resize((size, size), Image.Resampling.BILINEAR)
        )
        original_l1 = l1(current, target)
        repaired_l1 = l1(repaired, target)
        outside_max = int(
            np.abs(repaired.astype(np.int16) - current.astype(np.int16))[~accepted].max()
            if np.any(~accepted) else 0
        )
        frame_record = {
            "frame": frame,
            "proposed_components": proposed_components,
            "accepted_components": accepted_components,
            "rejected_large_components": rejected_components,
            "accepted_pixels": int(accepted.sum()),
            "temporally_textured_pixels": int(temporal_texture.sum()),
            "original_l1": original_l1,
            "repaired_l1": repaired_l1,
        }
        boundary_before, boundary_count = boundary_jump(current, accepted)
        boundary_after, _ = boundary_jump(repaired, accepted)
        frame_record["boundary_jump_before"] = boundary_before / max(boundary_count, 1)
        frame_record["boundary_jump_after"] = boundary_after / max(boundary_count, 1)
        if np.any(accepted):
            original_fill = l1(current, target, accepted)
            repaired_fill = l1(repaired, target, accepted)
            frame_record["original_fill_l1"] = original_fill
            frame_record["repaired_fill_l1"] = repaired_fill
            totals["original_fill_l1"] += original_fill
            totals["repaired_fill_l1"] += repaired_fill
            totals["frames_with_fill"] += 1
            totals["improved_frames"] += int(repaired_fill < original_fill)
        totals["proposed_components"] += proposed_components
        totals["accepted_components"] += accepted_components
        totals["rejected_large_components"] += rejected_components
        totals["accepted_pixels"] += int(accepted.sum())
        totals["temporally_textured_pixels"] += int(temporal_texture.sum())
        totals["original_l1"] += original_l1
        totals["repaired_l1"] += repaired_l1
        totals["outside_max_difference"] = max(totals["outside_max_difference"], outside_max)
        totals["boundary_jump_before_sum"] += boundary_before
        totals["boundary_jump_after_sum"] += boundary_after
        totals["boundary_edge_count"] += boundary_count
        per_frame.append(frame_record)

    filled_frames = max(totals["frames_with_fill"], 1)
    metrics = {
        "protocol": {
            "always_fill_area_px": args.always_fill_area,
            "minimum_temporal_support": args.minimum_temporal_support,
            "temporal_radius": args.temporal_radius,
            "flow_consistency_px": 1.5,
            "color": "two-or-more flow-consistent neighbors, otherwise Navier-Stokes",
        },
        "frames": len(frames),
        "proposed_components": totals["proposed_components"],
        "accepted_components": totals["accepted_components"],
        "rejected_large_components": totals["rejected_large_components"],
        "accepted_pixels": totals["accepted_pixels"],
        "temporally_textured_pixels": totals["temporally_textured_pixels"],
        "frames_with_fill": totals["frames_with_fill"],
        "improved_filled_frames": totals["improved_frames"],
        "original_l1": totals["original_l1"] / len(frames),
        "repaired_l1": totals["repaired_l1"] / len(frames),
        "relative_full_l1_change": totals["repaired_l1"] / max(totals["original_l1"], 1e-12) - 1.0,
        "original_fill_l1": totals["original_fill_l1"] / filled_frames,
        "repaired_fill_l1": totals["repaired_fill_l1"] / filled_frames,
        "relative_fill_l1_change": (
            totals["repaired_fill_l1"] / max(totals["original_fill_l1"], 1e-12) - 1.0
            if totals["frames_with_fill"] else 0.0
        ),
        "outside_max_difference": totals["outside_max_difference"],
        "boundary_jump_before": (
            totals["boundary_jump_before_sum"] / max(totals["boundary_edge_count"], 1)
        ),
        "boundary_jump_after": (
            totals["boundary_jump_after_sum"] / max(totals["boundary_edge_count"], 1)
        ),
        "relative_boundary_jump_change": (
            totals["boundary_jump_after_sum"]
            / max(totals["boundary_jump_before_sum"], 1e-12) - 1.0
            if totals["boundary_edge_count"] else 0.0
        ),
        "per_frame": per_frame,
    }
    args.output_root.mkdir(parents=True, exist_ok=True)
    (args.output_root / "metrics.json").write_text(
        json.dumps(metrics, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )
    print(json.dumps({key: value for key, value in metrics.items() if key != "per_frame"}, indent=2))


if __name__ == "__main__":
    main()
