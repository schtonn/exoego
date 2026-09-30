#!/usr/bin/env python3
"""Rejected diagnostic exporter for the deprecated white-color ICP ablation."""

from __future__ import annotations

import argparse
import json
from pathlib import Path


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--start-frame", type=int, required=True)
    parser.add_argument("--radius-m", type=float, default=0.16)
    parser.add_argument("--brightness", type=int, default=150)
    parser.add_argument("--chroma", type=int, default=50)
    parser.add_argument("--threshold-m", type=float, default=0.02)
    args = parser.parse_args()
    source = json.loads(args.input.read_text())
    setting = next(
        value for value in source["settings"]
        if value["radius_m"] == args.radius_m
        and value["brightness"] == args.brightness
        and value["chroma"] == args.chroma
        and value["threshold_m"] == args.threshold_m
    )
    poses = setting["poses"]
    output = {
        "protocol": "four exo RGB-D helmet ICP + initial ego pose; no future ego inputs",
        "pair_id": source["pair_id"],
        "setting": {
            key: setting[key]
            for key in ("radius_m", "brightness", "chroma", "threshold_m")
        },
        "frames": list(range(args.start_frame, args.start_frame + len(poses))),
        "poses": poses,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(output, indent=2) + "\n")
    print(args.output)


if __name__ == "__main__":
    main()
