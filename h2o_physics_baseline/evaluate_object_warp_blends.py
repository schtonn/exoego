#!/usr/bin/env python3
"""Sweep confidence-limited blends of rigid object transport into a base video."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
from PIL import Image, ImageFilter


def rgb(path: Path, size: int | None = None) -> np.ndarray:
    image = Image.open(path).convert("RGB")
    if size is not None and image.size != (size, size):
        image = image.resize((size, size), Image.Resampling.BILINEAR)
    return np.asarray(image, dtype=np.float32) / 255.0


def mask(path: Path, erosion: int) -> np.ndarray:
    image = Image.open(path).convert("L")
    if erosion:
        image = image.filter(ImageFilter.MinFilter(2 * erosion + 1))
    return np.asarray(image) > 127


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--base-root", type=Path, required=True)
    parser.add_argument("--warp-root", type=Path, required=True)
    parser.add_argument("--target-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    manifest = json.loads((args.warp_root / "manifest.json").read_text())
    settings = [(erosion, alpha) for erosion in (0, 1, 2, 3) for alpha in (0.1, 0.25, 0.5, 0.75, 1.0)]
    totals = {
        setting: {"absolute": 0.0, "pixels": 0, "object_absolute": 0.0, "object_pixels": 0,
                  "temporal_absolute": 0.0, "base_temporal_absolute": 0.0,
                  "temporal_pixels": 0, "modified": 0}
        for setting in settings
    }
    base_total = 0.0
    base_pixels = 0
    previous = {}
    for record in manifest["frames"]:
        index, frame = int(record["index"]), int(record["dataset_frame"])
        name = f"{index:06d}.png"
        base = rgb(args.base_root / "input_frames" / name)
        warp = rgb(args.warp_root / "input_frames" / name)
        target = rgb(args.target_root / f"{frame:06d}.png", manifest["image_size"])
        base_total += float(np.abs(base - target).sum())
        base_pixels += base.size
        for setting in settings:
            erosion, alpha = setting
            support = mask(args.warp_root / "object_masks" / name, erosion)
            prediction = base.copy()
            prediction[support] = (1.0 - alpha) * base[support] + alpha * warp[support]
            absolute = np.abs(prediction - target)
            values = totals[setting]
            values["absolute"] += float(absolute.sum())
            values["pixels"] += prediction.size
            values["object_absolute"] += float(absolute[support].sum())
            values["object_pixels"] += int(support.sum()) * 3
            values["modified"] += int(support.sum())
            if setting in previous:
                prior_prediction, prior_target, prior_support, prior_base = previous[setting]
                temporal_support = support | prior_support
                delta_error = np.abs(
                    (prediction - prior_prediction) - (target - prior_target)
                )
                values["temporal_absolute"] += float(delta_error[temporal_support].sum())
                base_delta_error = np.abs(
                    (base - prior_base) - (target - prior_target)
                )
                values["base_temporal_absolute"] += float(
                    base_delta_error[temporal_support].sum()
                )
                values["temporal_pixels"] += int(temporal_support.sum()) * 3
            previous[setting] = (prediction, target, support, base)

    base_l1 = base_total / base_pixels
    results = []
    for (erosion, alpha), values in totals.items():
        l1 = values["absolute"] / values["pixels"]
        results.append({
            "erosion_px": erosion,
            "alpha": alpha,
            "all_l1": l1,
            "all_change_percent": 100.0 * (l1 / base_l1 - 1.0),
            "object_l1": (
                values["object_absolute"] / values["object_pixels"]
                if values["object_pixels"] else None
            ),
            "object_temporal_delta_l1": (
                values["temporal_absolute"] / values["temporal_pixels"]
                if values["temporal_pixels"] else None
            ),
            "base_object_temporal_delta_l1": (
                values["base_temporal_absolute"] / values["temporal_pixels"]
                if values["temporal_pixels"] else None
            ),
            "mean_modified_fraction": values["modified"] / (len(manifest["frames"]) * manifest["image_size"] ** 2),
        })
    output = {
        "pair_id": manifest["pair_id"],
        "base_l1": base_l1,
        "best_all_l1": min(results, key=lambda value: value["all_l1"]),
        "variants": results,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(output, indent=2) + "\n")
    print(json.dumps(output, indent=2))


if __name__ == "__main__":
    main()
