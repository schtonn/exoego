#!/usr/bin/env python3
"""Evaluate reduced RGB-D input contracts with fixed motion/state estimates."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
from PIL import Image


def rgb(path: Path, size: int) -> np.ndarray:
    return np.asarray(
        Image.open(path).convert("RGB").resize(
            (size, size), Image.Resampling.BILINEAR
        ),
        dtype=np.float32,
    ) / 255.0


def mask(path: Path, size: int) -> np.ndarray:
    return np.asarray(
        Image.open(path).convert("L").resize(
            (size, size), Image.Resampling.NEAREST
        )
    ) > 127


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--variant", action="append", required=True,
        help="NAME=MODEL_INPUT_ROOT; repeat for every input contract",
    )
    parser.add_argument("--target-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    roots = {}
    for value in args.variant:
        name, root = value.split("=", 1)
        roots[name] = Path(root)
    manifests = {
        name: json.loads((root / "manifest.json").read_text())
        for name, root in roots.items()
    }
    prediction_roots = {}
    for name, root in roots.items():
        # Current exports wrap streams in model_input; the three original
        # mount-prior exports put the same manifest and streams at scene root.
        container = root.parent if root.name == "model_input" else root
        prediction_roots[name] = (
            container / "final_fusion" / "frames"
            if (container / "final_fusion" / "frames").is_dir()
            else container / "propainter_composed" / "frames"
            if (container / "propainter_composed" / "frames").is_dir()
            else root / "input_frames"
        )
    reference = next(iter(manifests.values()))
    size = int(reference["image_size"])
    records = reference["frames"]
    totals = {
        name: {
            "absolute": 0.0, "square": 0.0, "pixels": 0,
            "dynamic_absolute": 0.0, "dynamic_pixels": 0,
            "background_absolute": 0.0, "background_pixels": 0,
            "temporal_absolute": 0.0, "temporal_pixels": 0,
        }
        for name in roots
    }
    prior_prediction: dict[str, np.ndarray] = {}
    prior_target = None
    observed_fraction = {
        name: float(np.mean([1.0 - row["unknown_fraction"] for row in manifest["frames"]]))
        for name, manifest in manifests.items()
    }
    # Frame 0 is excluded from reconstruction metrics: an ego-anchor protocol
    # copies it exactly by definition, whereas it is a prediction for no-anchor.
    for record in records[1:]:
        index = int(record["index"])
        frame = int(record["dataset_frame"])
        filename = f"{index:06d}.png"
        target = rgb(args.target_root / f"{frame:06d}.png", size)
        predictions = {
            name: rgb(prediction_roots[name] / filename, size)
            for name in roots
        }
        dynamic = np.zeros((size, size), dtype=bool)
        for root in roots.values():
            dynamic |= mask(root / "foreground_masks" / filename, size)
        for name, prediction in predictions.items():
            absolute = np.abs(prediction - target)
            values = totals[name]
            values["absolute"] += float(absolute.sum())
            values["square"] += float(np.square(prediction - target).sum())
            values["pixels"] += prediction.size
            values["dynamic_absolute"] += float(absolute[dynamic].sum())
            values["dynamic_pixels"] += int(dynamic.sum()) * 3
            values["background_absolute"] += float(absolute[~dynamic].sum())
            values["background_pixels"] += int((~dynamic).sum()) * 3
            if prior_target is not None:
                delta_error = np.abs(
                    (prediction - prior_prediction[name]) - (target - prior_target)
                )
                values["temporal_absolute"] += float(delta_error.sum())
                values["temporal_pixels"] += prediction.size
        prior_prediction = predictions
        prior_target = target

    metrics = {}
    for name, values in totals.items():
        mse = values["square"] / values["pixels"]
        metrics[name] = {
            "source_camera_indices": manifests[name]["source_camera_indices"],
            "state_camera_indices": manifests[name].get(
                "state_camera_indices", [0, 1, 2, 3]
            ),
            "ego_anchor_enabled": manifests[name]["ego_anchor_enabled"],
            "mean_geometrically_observed_fraction": observed_fraction[name],
            "l1": values["absolute"] / values["pixels"],
            "psnr": float(-10.0 * np.log10(max(mse, 1e-12))),
            "dynamic_union_l1": (
                values["dynamic_absolute"] / values["dynamic_pixels"]
                if values["dynamic_pixels"] else None
            ),
            "background_l1": values["background_absolute"] / values["background_pixels"],
            "temporal_delta_l1": values["temporal_absolute"] / values["temporal_pixels"],
        }
    baseline = metrics["full4_anchor"]
    for name, value in metrics.items():
        value["relative_to_full_percent"] = {
            key: 100.0 * (value[key] / baseline[key] - 1.0)
            for key in ("l1", "dynamic_union_l1", "background_l1", "temporal_delta_l1")
            if value[key] is not None and baseline[key] not in (None, 0.0)
        }
    result = {
        "pair_id": reference["pair_id"],
        "frame_count_evaluated": len(records) - 1,
        "scope": (
            "input-consistent RGB-D appearance and head/hand/arm state ablation "
            "with constrained ProPainter completion and standard temporally "
            "constrained terminal fusion"
        ),
        "metrics": metrics,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
