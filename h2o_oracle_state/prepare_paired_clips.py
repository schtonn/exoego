#!/usr/bin/env python3
"""Expand H2O physical clips into explicit exo-to-ego training pairs."""

from __future__ import annotations

import argparse
import csv
import json
from collections import Counter
from pathlib import Path


DEFAULT_ROOT = Path("datasets/H2O/raw")
DEFAULT_STATE = Path("datasets/H2O/oracle_state")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=DEFAULT_ROOT)
    parser.add_argument("--state-root", type=Path, default=DEFAULT_STATE)
    parser.add_argument("--input", type=Path)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    input_path = args.input or args.state_root / "physical_clips_64f_stride32.csv"
    output_path = args.output or args.state_root / "paired_physical_clips.csv"
    rows = list(csv.DictReader(input_path.open(encoding="utf-8")))
    output_rows = []
    missing = []
    for row in rows:
        sequence_root = args.root / row["sequence"]
        start = int(row["start_frame"])
        end = int(row["end_frame"])
        target = sequence_root / "cam4"
        target_start = target / "rgb" / f"{start:06d}.png"
        target_end = target / "rgb" / f"{end:06d}.png"
        for source_index in range(4):
            source = sequence_root / f"cam{source_index}"
            source_start = source / "rgb" / f"{start:06d}.png"
            source_end = source / "rgb" / f"{end:06d}.png"
            required = (
                source_start,
                source_end,
                target_start,
                target_end,
                source / "cam_intrinsics.txt",
                target / "cam_intrinsics.txt",
            )
            absent = [str(path) for path in required if not path.is_file()]
            if absent:
                missing.extend(absent)
                continue
            output_rows.append(
                {
                    "pair_id": f"{row['clip_id']}_cam{source_index}_to_cam4",
                    "clip_id": row["clip_id"],
                    "split": row["split"],
                    "sequence": row["sequence"],
                    "source_camera": f"cam{source_index}",
                    "target_camera": "cam4",
                    "start_frame": start,
                    "end_frame": end,
                    "length": row["length"],
                    "source_rgb_dir": str(source / "rgb"),
                    "target_rgb_dir": str(target / "rgb"),
                    "source_depth_dir": str(source / "depth"),
                    "target_depth_dir": str(target / "depth"),
                    "source_cam_pose_dir": str(source / "cam_pose"),
                    "target_cam_pose_dir": str(target / "cam_pose"),
                    "source_intrinsics": str(source / "cam_intrinsics.txt"),
                    "target_intrinsics": str(target / "cam_intrinsics.txt"),
                    "state_path": row["state_path"],
                    "object_id": row["object_id"],
                    "object_name": row["object_name"],
                    "dominant_action_id": row["dominant_action_id"],
                    "dominant_action_name": row["dominant_action_name"],
                    "labels_available": row["labels_available"],
                    "left_presence_fraction": row["left_presence_fraction"],
                    "right_presence_fraction": row["right_presence_fraction"],
                    "left_contact_fraction": row["left_contact_fraction"],
                    "right_contact_fraction": row["right_contact_fraction"],
                    "object_speed_p95_mps": row["object_speed_p95_mps"],
                    "camera_speed_p95_mps": row["camera_speed_p95_mps"],
                    "quality_ok": row["quality_ok"],
                }
            )
    if missing:
        sample = "\n".join(missing[:10])
        raise FileNotFoundError(f"Missing {len(missing)} required clip endpoints; first entries:\n{sample}")
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(output_rows[0]))
        writer.writeheader()
        writer.writerows(output_rows)
    summary = {
        "input_clips": len(rows),
        "source_cameras_per_clip": 4,
        "paired_clips": len(output_rows),
        "target_camera": "cam4",
        "split_counts": dict(Counter(row["split"] for row in output_rows)),
        "source_camera_counts": dict(Counter(row["source_camera"] for row in output_rows)),
        "object_counts": dict(Counter(row["object_name"] for row in output_rows)),
        "output": str(output_path),
    }
    summary_path = output_path.with_suffix(".summary.json")
    summary_path.write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
