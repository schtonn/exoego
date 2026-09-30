#!/usr/bin/env python3
"""Aggregate mount-estimator pilots with an exo-only coverage gate."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np


METRICS = ("l1", "dynamic_union_l1", "background_l1", "temporal_delta_l1")
VARIANT = "exo4_no_anchor_no_mount"


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--pair", action="append", required=True,
        help="NAME=CANONICAL_METRICS_JSON,CANDIDATE_METRICS_JSON",
    )
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    records = []
    for specification in args.pair:
        name, paths = specification.split("=", 1)
        canonical_path, candidate_path = (Path(value) for value in paths.split(",", 1))
        canonical = json.loads(canonical_path.read_text())["metrics"][VARIANT]
        candidate = json.loads(candidate_path.read_text())["metrics"][VARIANT]
        use_candidate = (
            candidate["mean_geometrically_observed_fraction"]
            > canonical["mean_geometrically_observed_fraction"]
        )
        selected = candidate if use_candidate else canonical
        records.append({
            "scene": name,
            "selected": "face_knn" if use_candidate else "canonical",
            "canonical_observed_fraction": canonical[
                "mean_geometrically_observed_fraction"
            ],
            "face_knn_observed_fraction": candidate[
                "mean_geometrically_observed_fraction"
            ],
            "canonical": {key: canonical[key] for key in METRICS},
            "face_knn": {key: candidate[key] for key in METRICS},
            "selected_metrics": {key: selected[key] for key in METRICS},
        })

    means = {}
    for key in METRICS:
        canonical_mean = float(np.mean([row["canonical"][key] for row in records]))
        candidate_mean = float(np.mean([row["face_knn"][key] for row in records]))
        gated_mean = float(np.mean([row["selected_metrics"][key] for row in records]))
        means[key] = {
            "canonical": canonical_mean,
            "face_knn": candidate_mean,
            "coverage_gated": gated_mean,
            "face_knn_relative_to_canonical_percent": 100.0 * (
                candidate_mean / canonical_mean - 1.0
            ),
            "coverage_gated_relative_to_canonical_percent": 100.0 * (
                gated_mean / canonical_mean - 1.0
            ),
        }
    result = {
        "gate": "select face_knn for the whole clip only when exo geometric observed fraction increases",
        "uses_target_ego_for_selection": False,
        "scene_count": len(records),
        "candidate_selected_count": sum(row["selected"] == "face_knn" for row in records),
        "means": means,
        "per_scene": records,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2, ensure_ascii=False) + "\n")
    print(json.dumps(result["means"], indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
