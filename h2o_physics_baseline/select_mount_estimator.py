#!/usr/bin/env python3
"""Select a precomputed exo-only mount estimator for downstream rendering."""

from __future__ import annotations

import argparse
import json
from pathlib import Path


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--estimator", choices=("face_knn",), required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    summary = json.loads(args.input.read_text())
    for record in summary["targets"]:
        candidate = record["face_knn_candidate"]
        evaluation = record["evaluation"]
        record["canonical_candidate"] = {
            "predicted_initial_pose_world": record["predicted_initial_pose_world"],
            "translation_error_m": evaluation["translation_error_m"],
            "rotation_error_deg": evaluation["rotation_error_deg"],
        }
        record["predicted_initial_pose_world"] = candidate[
            "predicted_initial_pose_world"
        ]
        evaluation["translation_error_m"] = evaluation[
            "face_knn_translation_error_m"
        ]
        evaluation["rotation_error_deg"] = evaluation[
            "face_knn_rotation_error_deg"
        ]
    summary["selected_prediction_estimator"] = args.estimator
    summary["protocol"] += "; face-shape KNN selected only on calibration-split CV"
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(summary, indent=2, ensure_ascii=False) + "\n")
    print(args.output)


if __name__ == "__main__":
    main()
