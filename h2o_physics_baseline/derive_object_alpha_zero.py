#!/usr/bin/env python3
"""Recover the same-background alpha=0 object ablation from alpha=a and alpha=1 renders."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
from PIL import Image


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--soft-root", type=Path, required=True)
    parser.add_argument("--hard-root", type=Path, required=True)
    parser.add_argument("--alpha", type=float, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    args = parser.parse_args()
    if not 0.0 < args.alpha < 1.0:
        raise ValueError("alpha must be strictly between zero and one")
    manifest = json.loads((args.soft_root / "manifest.json").read_text())
    output_frames = args.output_root / "input_frames"
    output_frames.mkdir(parents=True, exist_ok=True)
    for record in manifest["frames"]:
        name = f"{int(record['index']):06d}.png"
        soft = np.asarray(Image.open(args.soft_root / "input_frames" / name).convert("RGB"), dtype=np.float32) / 255.0
        hard = np.asarray(Image.open(args.hard_root / "input_frames" / name).convert("RGB"), dtype=np.float32) / 255.0
        background = np.clip((soft - args.alpha * hard) / (1.0 - args.alpha), 0.0, 1.0)
        Image.fromarray(np.rint(background * 255).astype(np.uint8), "RGB").save(output_frames / name)
    for directory in ("object_masks", "arm_masks", "foreground_masks"):
        destination = args.output_root / directory
        destination.mkdir(parents=True, exist_ok=True)
        for record in manifest["frames"]:
            name = f"{int(record['index']):06d}.png"
            Image.open(args.soft_root / directory / name).save(destination / name)
    manifest["object_alpha"] = 0.0
    manifest["derived_from_alpha"] = args.alpha
    (args.output_root / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    print(args.output_root)


if __name__ == "__main__":
    main()
