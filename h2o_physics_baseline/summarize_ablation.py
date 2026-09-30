#!/usr/bin/env python3
"""Summarize matched rgb/state conditioning runs from their histories."""

from __future__ import annotations

import argparse
import json
from pathlib import Path


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("root", type=Path)
    args = parser.parse_args()
    records = {}
    for mode in ("rgb_state", "rgb_only", "state_only"):
        path = args.root / mode / "history.json"
        history = json.loads(path.read_text(encoding="utf-8"))
        best = min(history, key=lambda item: item["val"]["loss"])
        records[mode] = {"epoch": best["epoch"], **best["val"]}
    rgb = records["rgb_only"]
    combined = records["rgb_state"]
    delta = {
        "loss_reduction_percent": 100 * (rgb["loss"] - combined["loss"]) / rgb["loss"],
        "l1_reduction_percent": 100 * (rgb["l1"] - combined["l1"]) / rgb["l1"],
        "hand_l1_reduction_percent": 100 * (rgb["hand_l1"] - combined["hand_l1"]) / rgb["hand_l1"],
        "temporal_l1_reduction_percent": 100
        * (rgb["temporal_l1"] - combined["temporal_l1"])
        / rgb["temporal_l1"],
        "psnr_gain_db": combined["psnr"] - rgb["psnr"],
    }
    output = {
        "selection": "lowest validation objective independently per condition mode",
        "runs": records,
        "rgb_state_vs_rgb_only": delta,
        "warning": "Small engineering pilot only; not a statistically powered research result.",
    }
    destination = args.root / "ablation_summary.json"
    destination.write_text(json.dumps(output, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(output, indent=2))


if __name__ == "__main__":
    main()
