#!/usr/bin/env python3
"""Compare the former hand-neighbour object proxy with an explicit CAD object layer."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
from PIL import Image


def load_rgb(path: Path, size: int) -> np.ndarray:
    image = Image.open(path).convert("RGB")
    if image.size != (size, size):
        image = image.resize((size, size), Image.Resampling.BILINEAR)
    return np.asarray(image, dtype=np.float32) / 255.0


def load_mask(path: Path, size: int) -> np.ndarray:
    image = Image.open(path).convert("L")
    if image.size != (size, size):
        image = image.resize((size, size), Image.Resampling.NEAREST)
    return np.asarray(image) > 127


def masked_l1(pred: np.ndarray, target: np.ndarray, mask: np.ndarray) -> float | None:
    if not np.any(mask):
        return None
    return float(np.abs(pred - target)[mask].mean())


def summarize(values: list[float | None]) -> float | None:
    valid = [value for value in values if value is not None]
    return float(np.mean(valid)) if valid else None


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--old-root", type=Path, required=True)
    parser.add_argument("--new-root", type=Path, required=True)
    parser.add_argument("--target-rgb-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    manifest = json.loads((args.new_root / "manifest.json").read_text())
    size = int(manifest["image_size"])
    records = manifest["frames"]
    region_names = [
        "all", "new_object", "old_proxy", "overlap", "new_only", "old_only",
        "object_union", "arm", "outside_object_union", "changed",
    ]
    errors = {
        variant: {region: [] for region in region_names}
        for variant in ("old_proxy", "explicit_object")
    }
    error_sums = {
        variant: {region: 0.0 for region in region_names}
        for variant in ("old_proxy", "explicit_object")
    }
    error_counts = {
        variant: {region: 0 for region in region_names}
        for variant in ("old_proxy", "explicit_object")
    }
    temporal_errors = {variant: [] for variant in ("old_proxy", "explicit_object")}
    mask_iou: list[float] = []
    fractions = {name: [] for name in ("new_object", "old_proxy", "overlap", "new_only", "old_only")}
    prior = None

    for record in records:
        index = int(record["index"])
        frame = int(record["dataset_frame"])
        name = f"{index:06d}.png"
        old = load_rgb(args.old_root / "input_frames" / name, size)
        new = load_rgb(args.new_root / "input_frames" / name, size)
        target = load_rgb(args.target_rgb_root / f"{frame:06d}.png", size)
        old_object = load_mask(args.old_root / "object_masks" / name, size)
        new_object = load_mask(args.new_root / "object_masks" / name, size)
        arm = load_mask(args.new_root / "arm_masks" / name, size)
        overlap = old_object & new_object
        union = old_object | new_object
        regions = {
            "all": np.ones((size, size), dtype=bool),
            "new_object": new_object,
            "old_proxy": old_object,
            "overlap": overlap,
            "new_only": new_object & ~old_object,
            "old_only": old_object & ~new_object,
            "object_union": union,
            "arm": arm,
            "outside_object_union": ~union,
            "changed": np.any(np.abs(new - old) > (1.0 / 255.0), axis=2),
        }
        for variant, image in (("old_proxy", old), ("explicit_object", new)):
            absolute = np.abs(image - target)
            for region, mask in regions.items():
                errors[variant][region].append(masked_l1(image, target, mask))
                error_sums[variant][region] += float(absolute[mask].sum())
                error_counts[variant][region] += int(mask.sum()) * 3

        denominator = np.count_nonzero(union)
        mask_iou.append(float(np.count_nonzero(overlap) / denominator) if denominator else 1.0)
        for key, mask in (
            ("new_object", new_object), ("old_proxy", old_object), ("overlap", overlap),
            ("new_only", new_object & ~old_object), ("old_only", old_object & ~new_object),
        ):
            fractions[key].append(float(mask.mean()))

        if prior is not None:
            temporal_mask = union | prior["union"]
            target_delta = target - prior["target"]
            for variant, image in (("old_proxy", old), ("explicit_object", new)):
                pred_delta = image - prior[variant]
                temporal_errors[variant].append(masked_l1(pred_delta, target_delta, temporal_mask))
        prior = {"old_proxy": old, "explicit_object": new, "target": target, "union": union}

    result = {
        "pair_id": manifest["pair_id"],
        "frame_count": len(records),
        "l1": {
            variant: {region: summarize(values) for region, values in region_map.items()}
            for variant, region_map in errors.items()
        },
        "relative_l1_change_percent": {
            region: (
                100.0 * (summarize(errors["explicit_object"][region])
                         / summarize(errors["old_proxy"][region]) - 1.0)
                if summarize(errors["old_proxy"][region]) not in (None, 0.0)
                and summarize(errors["explicit_object"][region]) is not None else None
            )
            for region in region_names
        },
        "pixel_weighted_l1": {
            variant: {
                region: (
                    error_sums[variant][region] / error_counts[variant][region]
                    if error_counts[variant][region] else None
                )
                for region in region_names
            }
            for variant in ("old_proxy", "explicit_object")
        },
        "object_union_temporal_delta_l1": {
            variant: summarize(values) for variant, values in temporal_errors.items()
        },
        "mask": {
            "mean_iou": float(np.mean(mask_iou)),
            "mean_fraction": {key: float(np.mean(values)) for key, values in fractions.items()},
        },
    }
    result["pixel_weighted_relative_l1_change_percent"] = {}
    for region in region_names:
        old_value = result["pixel_weighted_l1"]["old_proxy"][region]
        new_value = result["pixel_weighted_l1"]["explicit_object"][region]
        result["pixel_weighted_relative_l1_change_percent"][region] = (
            100.0 * (new_value / old_value - 1.0)
            if old_value not in (None, 0.0) and new_value is not None else None
        )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
