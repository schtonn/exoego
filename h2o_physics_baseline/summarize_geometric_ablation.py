#!/usr/bin/env python3
"""Collect the A-D geometry/state pilot with extended known/hole metrics."""

from __future__ import annotations

import argparse
import json
from pathlib import Path


def reduction(reference: float, value: float) -> float:
    return 100.0 * (reference - value) / reference


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("root", type=Path)
    args = parser.parse_args()
    modes = ("rgb_only", "geometry", "geometry_state", "geometry_state_distance")
    runs = {}
    for mode in modes:
        evaluation = json.loads((args.root / mode / "evaluation_extended.json").read_text(encoding="utf-8"))
        runs[mode] = {"checkpoint_epoch": evaluation["checkpoint_epoch"], **evaluation["metrics"]}
    a, b, c, d = (runs[mode] for mode in modes)
    output = {
        "protocol": "cam3->cam4; 64 train pairs; 32 val pairs; 4x64x64; 3 epochs; matched seed",
        "runs": runs,
        "comparisons": {
            "B_geometry_vs_A_rgb": {
                "loss_reduction_percent": reduction(a["loss"], b["loss"]),
                "hand_l1_reduction_percent": reduction(a["hand_l1"], b["hand_l1"]),
                "psnr_gain_db": b["psnr"] - a["psnr"],
                "temporal_l1_reduction_percent": reduction(a["temporal_l1"], b["temporal_l1"]),
            },
            "C_state_vs_B_geometry": {
                "loss_reduction_percent": reduction(b["loss"], c["loss"]),
                "hand_l1_reduction_percent": reduction(b["hand_l1"], c["hand_l1"]),
                "hole_l1_reduction_percent": reduction(b["hole_l1"], c["hole_l1"]),
                "psnr_gain_db": c["psnr"] - b["psnr"],
            },
            "D_distance_vs_C_state": {
                "loss_reduction_percent": reduction(c["loss"], d["loss"]),
                "hand_l1_reduction_percent": reduction(c["hand_l1"], d["hand_l1"]),
                "hole_l1_reduction_percent": reduction(c["hole_l1"], d["hole_l1"]),
                "psnr_gain_db": d["psnr"] - c["psnr"],
            },
        },
        "conclusion": "Geometry is strongly beneficial. This pilot provides no evidence that the current state/distance adapter improves geometry refinement; C and D are slightly worse than B.",
        "warning": "Engineering pilot only; follow with full-split multi-seed experiments.",
    }
    destination = args.root / "ablation_extended_summary.json"
    destination.write_text(json.dumps(output, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(output, indent=2))


if __name__ == "__main__":
    main()
