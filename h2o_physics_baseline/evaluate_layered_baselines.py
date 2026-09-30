#!/usr/bin/env python3
"""Compare simple baselines and layered outputs in one evaluation harness."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
import sys

import numpy as np
from PIL import Image

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from h2o_geometric_baseline.reprojection import load_intrinsics, reproject_rgbd
from h2o_physics_baseline.protocols import validate_layered_manifest


def load_rgb(path: Path, size: int) -> np.ndarray:
    return np.asarray(
        Image.open(path).convert("RGB").resize((size, size), Image.Resampling.BILINEAR),
        dtype=np.float32,
    ) / 255.0


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, required=True, help="Layered scene root")
    parser.add_argument(
        "--index", type=Path,
        default=Path("datasets/H2O/oracle_state/paired_physical_clips.csv"),
    )
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    model_root = args.root / "model_input"
    if not (model_root / "manifest.json").exists():
        model_root = args.root
    manifest = json.loads((model_root / "manifest.json").read_text())
    validate_layered_manifest(manifest)
    with args.index.open(encoding="utf-8") as handle:
        rows = [row for row in csv.DictReader(handle) if row["pair_id"] == manifest["pair_id"]]
    if len(rows) != 1:
        raise ValueError(f"Expected one index row, found {len(rows)}")
    row = rows[0]
    target_root = Path(row["target_rgb_dir"]).parent
    size = int(manifest["image_size"])
    first_dataset_frame = int(manifest["frames"][0]["dataset_frame"])
    first_rgb_u8 = np.asarray(
        Image.open(target_root / "rgb" / f"{first_dataset_frame:06d}.png").convert("RGB")
    )
    first_rgb = load_rgb(
        target_root / "rgb" / f"{first_dataset_frame:06d}.png", size
    )
    first_depth = np.asarray(
        Image.open(target_root / "depth" / f"{first_dataset_frame:06d}.png")
    )
    intrinsics = load_intrinsics(target_root / "cam_intrinsics.txt")
    first_pose = np.asarray(manifest["ego_first_pose_world"], dtype=np.float64)

    stream_roots = {"layered_no_completion": model_root / "input_frames"}
    if (args.root / "propainter_composed/frames").is_dir():
        stream_roots["propainter_composed"] = args.root / "propainter_composed/frames"
    if (args.root / "final_fusion/frames").is_dir():
        stream_roots["terminal_fusion"] = args.root / "final_fusion/frames"
    names = [*stream_roots, "copy_first_frame", "warp_first_frame_copy_fill"]
    totals = {
        name: {
            "absolute": 0.0, "square": 0.0, "pixels": 0,
            "temporal": 0.0, "temporal_pixels": 0,
        } for name in names
    }
    prior_predictions: dict[str, np.ndarray] = {}
    prior_target = None
    warp_coverage = []

    for record in manifest["frames"]:
        index = int(record["index"])
        if index == 0:
            continue
        frame = int(record["dataset_frame"])
        filename = f"{index:06d}.png"
        target = load_rgb(target_root / "rgb" / f"{frame:06d}.png", size)
        pose = np.asarray(record["predicted_camera_pose_world"], dtype=np.float64)
        warped = reproject_rgbd(
            first_rgb_u8, first_depth, intrinsics, first_pose, intrinsics, pose,
            output_size=(size, size), source_stride=1,
        )
        warped_rgb = warped.rgb.astype(np.float32) / 255.0
        warped_filled = np.where(warped.valid[..., None], warped_rgb, first_rgb)
        warp_coverage.append(float(warped.valid.mean()))
        predictions = {
            name: load_rgb(root / filename, size)
            for name, root in stream_roots.items()
        } | {
            "copy_first_frame": first_rgb,
            "warp_first_frame_copy_fill": warped_filled,
        }
        for name, prediction in predictions.items():
            difference = prediction - target
            totals[name]["absolute"] += float(np.abs(difference).sum())
            totals[name]["square"] += float(np.square(difference).sum())
            totals[name]["pixels"] += difference.size
            if prior_target is not None:
                delta_error = np.abs(
                    (prediction - prior_predictions[name]) - (target - prior_target)
                )
                totals[name]["temporal"] += float(delta_error.sum())
                totals[name]["temporal_pixels"] += delta_error.size
        prior_predictions = predictions
        prior_target = target

    metrics = {}
    for name, values in totals.items():
        mse = values["square"] / values["pixels"]
        metrics[name] = {
            "l1": values["absolute"] / values["pixels"],
            "psnr": float(-10.0 * np.log10(max(mse, 1e-12))),
            "temporal_delta_l1": values["temporal"] / values["temporal_pixels"],
        }
    result = {
        "pair_id": manifest["pair_id"],
        "dataset_split": row["split"],
        "frame_count_evaluated": len(manifest["frames"]) - 1,
        "warp_first_frame_valid_coverage": float(np.mean(warp_coverage)),
        "warp_hole_policy": "copy unwarped first frame; reported explicitly in baseline name",
        "metrics": metrics,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
