#!/usr/bin/env python3
"""Measure ground-truth ego-camera motion and relate it to fixed-pose fill quality.

The future cam4 poses are evaluation-only here.  They are never supplied to the
generation model or to the fixed-initial-pose multiview background candidate.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

import numpy as np
import torch

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from h2o_geometric_baseline.reprojection import load_pose
from h2o_physics_baseline.dataset import H2OPhysicalClipDataset


def rotation_angle_deg(rotation: np.ndarray) -> float:
    cosine = np.clip((np.trace(rotation) - 1.0) * 0.5, -1.0, 1.0)
    return float(np.degrees(np.arccos(cosine)))


def correlation(x: list[float], y: list[float]) -> float:
    if len(x) < 2 or np.std(x) < 1e-12 or np.std(y) < 1e-12:
        return float("nan")
    return float(np.corrcoef(x, y)[0, 1])


def describe(values: list[float]) -> dict[str, float]:
    array = np.asarray(values, dtype=np.float64)
    return {
        "mean": float(array.mean()),
        "median": float(np.median(array)),
        "p90": float(np.percentile(array, 90)),
        "max": float(array.max()),
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
        "--fill-summary",
        type=Path,
        default=Path("datasets/H2O/experiments/multiview_background_fill/summary.json"),
    )
    parser.add_argument("--max-samples", type=int, default=32)
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("datasets/H2O/experiments/multiview_background_fill/ego_motion_audit.json"),
    )
    args = parser.parse_args()

    config = torch.load(args.checkpoint, map_location="cpu", weights_only=False)["config"]
    dataset = H2OPhysicalClipDataset(
        config["index"],
        split="val",
        frames_per_clip=config["frames_per_clip"],
        image_size=config["image_size"],
        stats_path=config["stats"],
        source_cameras=tuple(config["source_cameras"]),
        max_samples=args.max_samples,
        combine_source_cameras=True,
    )
    fill = json.loads(args.fill_summary.read_text(encoding="utf-8"))
    fill_by_pair = {record["pair_id"]: record for record in fill["per_clip"]}

    records = []
    endpoint_translation = []
    endpoint_rotation = []
    old_improvement = []
    full_improvement = []
    for index in range(len(dataset)):
        item = dataset[index]
        row = dataset.rows[index]
        target_root = Path(row["target_rgb_dir"]).parent
        poses = [
            load_pose(target_root / "cam_pose" / f"{int(frame):06d}.txt")
            for frame in item["frame_numbers"]
        ]
        initial = poses[0]
        translations = [float(np.linalg.norm(pose[:3, 3] - initial[:3, 3])) for pose in poses]
        rotations = [rotation_angle_deg(initial[:3, :3].T @ pose[:3, :3]) for pose in poses]
        fill_record = fill_by_pair[item["pair_id"]]["variants"]
        old_gain = (
            fill_record["local_average"]["old_only_l1"]
            - fill_record["nearest"]["old_only_l1"]
        )
        total_gain = fill_record["local_average"]["l1"] - fill_record["nearest"]["l1"]
        endpoint_translation.append(translations[-1])
        endpoint_rotation.append(rotations[-1])
        old_improvement.append(old_gain)
        full_improvement.append(total_gain)
        records.append(
            {
                "pair_id": item["pair_id"],
                "frames": [int(value) for value in item["frame_numbers"]],
                "translation_from_first_m": translations,
                "rotation_from_first_deg": rotations,
                "nearest_minus_local_gain": {
                    "old_only_l1": old_gain,
                    "full_l1": total_gain,
                },
            }
        )

    future_translations = [value for record in records for value in record["translation_from_first_m"][1:]]
    future_rotations = [value for record in records for value in record["rotation_from_first_deg"][1:]]
    result = {
        "note": "cam4 future pose is evaluation-only; positive fill gain means nearest exo RGB-D beats local average",
        "samples": len(records),
        "future_frame_motion": {
            "translation_m": describe(future_translations),
            "rotation_deg": describe(future_rotations),
        },
        "endpoint_motion": {
            "translation_m": describe(endpoint_translation),
            "rotation_deg": describe(endpoint_rotation),
        },
        "pearson_endpoint_motion_vs_fill_gain": {
            "translation_vs_old_only": correlation(endpoint_translation, old_improvement),
            "rotation_vs_old_only": correlation(endpoint_rotation, old_improvement),
            "translation_vs_full": correlation(endpoint_translation, full_improvement),
            "rotation_vs_full": correlation(endpoint_rotation, full_improvement),
        },
        "clips_improved_old_only": int(np.sum(np.asarray(old_improvement) > 0)),
        "clips_degraded_old_only": int(np.sum(np.asarray(old_improvement) < 0)),
        "per_clip": records,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print(json.dumps(result | {"per_clip": "omitted"}, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
