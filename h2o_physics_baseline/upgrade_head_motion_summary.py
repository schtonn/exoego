#!/usr/bin/env python3
"""Recover exo-only rigid head transforms in legacy face-motion summaries."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import numpy as np


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument(
        "--index",
        type=Path,
        default=Path("datasets/H2O/oracle_state/paired_physical_clips.csv"),
    )
    args = parser.parse_args()
    with args.index.open(encoding="utf-8") as handle:
        rows = {row["pair_id"]: row for row in csv.DictReader(handle)}
    summary = json.loads(args.input.read_text())
    upgraded = 0
    for clip in summary["per_clip"]:
        row = rows[clip["pair_id"]]
        first_frame = int(clip["frames"][0])
        pose_path = Path(row["target_cam_pose_dir"]) / f"{first_frame:06d}.txt"
        initial_pose = np.loadtxt(pose_path).reshape(4, 4)
        for motion in clip["future"]:
            if "estimated_head_transform_world" in motion:
                continue
            if "estimated_camera_delta_position_world_m" not in motion:
                continue
            full_pose = initial_pose.copy()
            delta_rotation = np.asarray(motion["estimated_delta_rotation_world"])
            full_pose[:3, :3] = delta_rotation @ initial_pose[:3, :3]
            full_pose[:3, 3] = initial_pose[:3, 3] + np.asarray(
                motion["estimated_camera_delta_position_world_m"]
            )
            motion["estimated_head_transform_world"] = (
                full_pose @ np.linalg.inv(initial_pose)
            ).tolist()
            upgraded += 1
    summary["legacy_upgrade"] = {
        "source": str(args.input),
        "recovered_head_transforms": upgraded,
        "note": (
            "Algebraic recovery from legacy camera delta and its original initial pose; "
            "the recovered transform can be applied to an independently inferred anchor."
        ),
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(summary, indent=2, ensure_ascii=False) + "\n")
    print(json.dumps(summary["legacy_upgrade"], ensure_ascii=False))


if __name__ == "__main__":
    main()
