#!/usr/bin/env python3
"""Aggregate per-scene input-ablation metrics with equal scene weighting."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np


VARIANTS = (
    "full4_anchor",
    "exo4_no_anchor",
    "cam0_anchor",
    "cam0_no_anchor",
    "exo4_no_anchor_no_mount",
)
SCALARS = (
    "mean_geometrically_observed_fraction",
    "l1",
    "psnr",
    "dynamic_union_l1",
    "background_l1",
    "temporal_delta_l1",
)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--metric", type=Path, action="append", required=True)
    parser.add_argument("--mount-summary", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    scenes = [json.loads(path.read_text()) for path in args.metric]
    means: dict[str, dict[str, float | dict[str, float]]] = {}
    for variant in VARIANTS:
        means[variant] = {
            scalar: float(np.mean([scene["metrics"][variant][scalar] for scene in scenes]))
            for scalar in SCALARS
        }

    baseline = means["full4_anchor"]
    for variant in VARIANTS:
        means[variant]["relative_to_full_percent"] = {
            scalar: 100.0 * (float(means[variant][scalar]) / float(baseline[scalar]) - 1.0)
            for scalar in ("l1", "dynamic_union_l1", "background_l1", "temporal_delta_l1")
        }

    result = {
        "aggregation": "equal weight per scene; frame 0 excluded in each scene",
        "scene_count": len(scenes),
        "frame_count_evaluated": sum(scene["frame_count_evaluated"] for scene in scenes),
        "means": means,
        "per_scene": {
            scene["pair_id"]: scene["metrics"]
            for scene in scenes
        },
    }
    if args.mount_summary is not None:
        mount_summary = json.loads(args.mount_summary.read_text())
        pose_by_pair = {
            record["pair_id"]: record["evaluation"]
            for record in mount_summary["targets"]
        }
        mount_records = []
        for scene in scenes:
            pair_id = scene["pair_id"]
            known = scene["metrics"]["exo4_no_anchor"]
            withheld = scene["metrics"]["exo4_no_anchor_no_mount"]
            pose = pose_by_pair[pair_id]
            mount_records.append({
                "pair_id": pair_id,
                "initial_translation_error_cm": 100.0 * pose["translation_error_m"],
                "initial_rotation_error_deg": pose["rotation_error_deg"],
                "known_mount_l1": known["l1"],
                "withheld_mount_l1": withheld["l1"],
                "withheld_relative_to_known_percent": 100.0 * (
                    withheld["l1"] / known["l1"] - 1.0
                ),
                "known_mount_observed_fraction": known[
                    "mean_geometrically_observed_fraction"
                ],
                "withheld_mount_observed_fraction": withheld[
                    "mean_geometrically_observed_fraction"
                ],
            })
        translation = np.asarray([
            record["initial_translation_error_cm"] for record in mount_records
        ])
        rotation = np.asarray([
            record["initial_rotation_error_deg"] for record in mount_records
        ])
        degradation = np.asarray([
            record["withheld_relative_to_known_percent"] for record in mount_records
        ])
        result["mount_ablation"] = {
            "known_mount_mean_l1": means["exo4_no_anchor"]["l1"],
            "withheld_mount_mean_l1": means["exo4_no_anchor_no_mount"]["l1"],
            "withheld_relative_to_known_percent": 100.0 * (
                float(means["exo4_no_anchor_no_mount"]["l1"])
                / float(means["exo4_no_anchor"]["l1"]) - 1.0
            ),
            "translation_error_cm_mean": float(translation.mean()),
            "rotation_error_deg_mean": float(rotation.mean()),
            "translation_error_vs_l1_degradation_pearson": float(
                np.corrcoef(translation, degradation)[0, 1]
            ),
            "rotation_error_vs_l1_degradation_pearson": float(
                np.corrcoef(rotation, degradation)[0, 1]
            ),
            "per_scene": mount_records,
        }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(result["means"], indent=2))


if __name__ == "__main__":
    main()
