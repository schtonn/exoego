#!/usr/bin/env python3
"""Separate CAD/object appearance quality from estimated-ego-pose alignment."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

import numpy as np
from PIL import Image

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from h2o_geometric_baseline.reprojection import load_intrinsics, load_pose
from h2o_physics_baseline.annotated_object_layer import (
    annotated_exo_object_layer,
    annotated_exo_object_pose_world,
    load_geometry,
    rasterize_object_support,
)
from h2o_physics_baseline.render_causal_video_background_split import smoothed_pose_map


def centroid(mask: np.ndarray) -> np.ndarray:
    y, x = np.nonzero(mask)
    return np.array([x.mean(), y.mean()]) if len(x) else np.array([np.nan, np.nan])


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--sequence", required=True)
    parser.add_argument("--start-frame", type=int, required=True)
    parser.add_argument("--end-frame", type=int, required=True)
    parser.add_argument("--pair-id", required=True)
    parser.add_argument("--head-summary", type=Path, required=True)
    parser.add_argument("--absolute-head-poses", type=Path, default=None)
    parser.add_argument("--baseline-root", type=Path, default=None)
    parser.add_argument("--explicit-root", type=Path, default=None)
    parser.add_argument("--size", type=int, default=256)
    parser.add_argument("--appearance-stride", type=int, default=4)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    sequence_root = Path("datasets/H2O/raw") / args.sequence
    sources = [sequence_root / f"cam{index}" for index in range(4)]
    target = sequence_root / "cam4"
    frames = list(range(args.start_frame, args.end_frame + 1))
    intrinsics = load_intrinsics(target / "cam_intrinsics.txt")
    initial_pose = load_pose(target / "cam_pose" / f"{frames[0]:06d}.txt")
    if args.absolute_head_poses is not None:
        pose_data = json.loads(args.absolute_head_poses.read_text())
        predicted_poses = {
            int(frame): np.asarray(pose, dtype=np.float64)
            for frame, pose in zip(pose_data["frames"], pose_data["poses"])
        }
    else:
        head_summary = json.loads(args.head_summary.read_text())
        head_record = next(value for value in head_summary["per_clip"] if value["pair_id"] == args.pair_id)
        predicted_poses = smoothed_pose_map(frames, initial_pose, head_record)

    ious, centroid_errors, appearance_l1, visible_fractions, coverage = [], [], [], [], []
    baseline_visible_l1, explicit_visible_l1 = [], []
    per_frame = []
    for index, frame in enumerate(frames):
        object_id, object_pose, _ = annotated_exo_object_pose_world(sources, frame)
        geometry = load_geometry(str(Path("datasets/H2O/oracle_state").resolve()), object_id)
        actual_pose = load_pose(target / "cam_pose" / f"{frame:06d}.txt")
        actual_support, _ = rasterize_object_support(
            geometry, object_pose, intrinsics, actual_pose, (args.size, args.size)
        )
        predicted_support, _ = rasterize_object_support(
            geometry, object_pose, intrinsics, predicted_poses[frame], (args.size, args.size)
        )
        union = actual_support | predicted_support
        iou = float((actual_support & predicted_support).sum() / max(union.sum(), 1))
        center_error = float(np.linalg.norm(centroid(actual_support) - centroid(predicted_support)))
        ious.append(iou)
        centroid_errors.append(center_error)
        record = {"frame": frame, "support_iou": iou, "centroid_error_px": center_error}

        if index > 0 and (index % args.appearance_stride == 0 or index == len(frames) - 1):
            layer = annotated_exo_object_layer(
                sources, frame, intrinsics, actual_pose, Path("datasets/H2O/oracle_state"),
                output_size=args.size, source_stride=2,
            )
            target_rgb = np.asarray(
                Image.open(target / "rgb" / f"{frame:06d}.png").convert("RGB").resize(
                    (args.size, args.size), Image.Resampling.BILINEAR
                ), dtype=np.float32,
            ) / 255.0
            target_depth = np.asarray(
                Image.open(target / "depth" / f"{frame:06d}.png").resize(
                    (args.size, args.size), Image.Resampling.NEAREST
                ), dtype=np.float32,
            ) / 1000.0
            # Pixels whose observed target depth agrees with the rendered object
            # are visible object surface, not an occluding hand.
            visible = (
                layer.valid & (target_depth > 0) & np.isfinite(layer.depth_m)
                & (np.abs(target_depth - layer.depth_m) <= 0.025)
            )
            error = float(np.abs(layer.rgb - target_rgb)[visible].mean()) if np.any(visible) else None
            visible_fraction = float(visible.mean())
            layer_coverage = float(visible.sum() / max(actual_support.sum(), 1))
            if error is not None:
                appearance_l1.append(error)
            visible_fractions.append(visible_fraction)
            coverage.append(layer_coverage)
            record.update(
                appearance_l1_visible=error,
                visible_fraction=visible_fraction,
                visible_coverage_of_cad_support=layer_coverage,
            )
            name = f"{index:06d}.png"
            if args.baseline_root is not None and np.any(visible):
                baseline = np.asarray(
                    Image.open(args.baseline_root / "input_frames" / name).convert("RGB"),
                    dtype=np.float32,
                ) / 255.0
                value = float(np.abs(baseline - target_rgb)[visible].mean())
                baseline_visible_l1.append(value)
                record["baseline_l1_on_true_visible_object"] = value
            if args.explicit_root is not None and np.any(visible):
                explicit = np.asarray(
                    Image.open(args.explicit_root / "input_frames" / name).convert("RGB"),
                    dtype=np.float32,
                ) / 255.0
                value = float(np.abs(explicit - target_rgb)[visible].mean())
                explicit_visible_l1.append(value)
                record["estimated_pose_layer_l1_on_true_visible_object"] = value
        per_frame.append(record)

    result = {
        "pair_id": args.pair_id,
        "frame_count": len(frames),
        "support_alignment": {
            "mean_iou": float(np.mean(ious)),
            "median_iou": float(np.median(ious)),
            "mean_centroid_error_px": float(np.mean(centroid_errors)),
            "p90_centroid_error_px": float(np.percentile(centroid_errors, 90)),
        },
        "appearance_with_true_ego_pose": {
            "sample_count": len(appearance_l1),
            "visible_surface_l1": float(np.mean(appearance_l1)),
            "visible_fraction": float(np.mean(visible_fractions)),
            "visible_coverage_of_cad_support": float(np.mean(coverage)),
            "baseline_l1_on_true_visible_object": (
                float(np.mean(baseline_visible_l1)) if baseline_visible_l1 else None
            ),
            "estimated_pose_layer_l1_on_true_visible_object": (
                float(np.mean(explicit_visible_l1)) if explicit_visible_l1 else None
            ),
        },
        "per_frame": per_frame,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps({key: value for key, value in result.items() if key != "per_frame"}, indent=2))


if __name__ == "__main__":
    main()
