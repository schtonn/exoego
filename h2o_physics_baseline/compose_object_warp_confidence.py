#!/usr/bin/env python3
"""Apply confidence-weighted rigid object transport to an already repaired video."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
from PIL import Image


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--base-frames", type=Path, required=True)
    parser.add_argument("--warp-root", type=Path, required=True)
    parser.add_argument("--alpha", type=float, default=0.25)
    parser.add_argument("--output-root", type=Path, required=True)
    args = parser.parse_args()
    manifest = json.loads((args.warp_root / "manifest.json").read_text())
    output_frames = args.output_root / "input_frames"
    output_frames.mkdir(parents=True, exist_ok=True)
    mask_names = (
        "object_masks", "arm_masks", "foreground_masks", "provenance_labels",
    )
    for directory in mask_names:
        (args.output_root / directory).mkdir(parents=True, exist_ok=True)
    for record in manifest["frames"]:
        name = f"{int(record['index']):06d}.png"
        base = np.asarray(Image.open(args.base_frames / name).convert("RGB"), dtype=np.float32) / 255.0
        warp = np.asarray(
            Image.open(args.warp_root / "input_frames" / name).convert("RGB"), dtype=np.float32
        ) / 255.0
        object_mask = np.asarray(Image.open(args.warp_root / "object_masks" / name)) > 127
        result = base.copy()
        result[object_mask] = (
            (1.0 - args.alpha) * base[object_mask] + args.alpha * warp[object_mask]
        )
        Image.fromarray(np.rint(result * 255).astype(np.uint8), "RGB").save(output_frames / name)
        for directory in mask_names:
            Image.open(args.warp_root / directory / name).save(args.output_root / directory / name)
    manifest["object_alpha"] = args.alpha
    manifest["base_frames"] = str(args.base_frames)
    manifest["composition_order"] = "background repair first; object confidence fusion last"
    (args.output_root / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    print(args.output_root)


if __name__ == "__main__":
    main()
