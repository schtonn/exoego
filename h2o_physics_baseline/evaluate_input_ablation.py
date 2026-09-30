#!/usr/bin/env python3
"""Evaluate layered input ablations by source region and optional GT masks."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
from PIL import Image


PROVENANCE_NAMES = {
    1: "ego_anchor", 2: "current_exo", 3: "history_exo",
    4: "generated", 5: "hand_arm", 6: "object",
}


def rgb(path: Path, size: int) -> np.ndarray:
    return np.asarray(
        Image.open(path).convert("RGB").resize(
            (size, size), Image.Resampling.BILINEAR
        ), dtype=np.float32,
    ) / 255.0


def mask(path: Path, size: int) -> np.ndarray:
    return np.asarray(
        Image.open(path).convert("L").resize(
            (size, size), Image.Resampling.NEAREST
        )
    ) > 127


def optional_mask(root: Path | None, index: int, frame: int, size: int) -> np.ndarray | None:
    if root is None:
        return None
    for stem in (f"{frame:06d}", f"{index:06d}"):
        path = root / f"{stem}.png"
        if path.exists():
            return mask(path, size)
    raise FileNotFoundError(f"No GT mask for dataset frame {frame} below {root}")


def add_region(total: dict, name: str, error: np.ndarray, region: np.ndarray) -> None:
    entry = total.setdefault(name, {"absolute": 0.0, "pixels": 0})
    entry["absolute"] += float(error[region].sum())
    entry["pixels"] += int(region.sum()) * 3


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--variant", action="append", required=True,
        help="NAME=MODEL_INPUT_ROOT; repeat for every input contract",
    )
    parser.add_argument("--target-root", type=Path, required=True)
    parser.add_argument("--gt-hand-mask-root", type=Path)
    parser.add_argument("--gt-object-mask-root", type=Path)
    parser.add_argument("--baseline-name", default="full4_anchor")
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
            "temporal_absolute": 0.0, "temporal_pixels": 0,
            "regions": {}, "authorized_pixels": 0, "spatial_pixels": 0,
            "hand_intersection": 0, "hand_union": 0,
            "object_intersection": 0, "object_union": 0,
        } for name in roots
    }
    prior_prediction: dict[str, np.ndarray] = {}
    prior_target = None
    observed_fraction = {
        name: float(np.mean([
            1.0 - row["unknown_fraction"] for row in manifest["frames"]
        ])) for name, manifest in manifests.items()
    }

    # Frame 0 is copied by anchored protocols and is excluded consistently.
    for record in records[1:]:
        index = int(record["index"])
        frame = int(record["dataset_frame"])
        filename = f"{index:06d}.png"
        target = rgb(args.target_root / f"{frame:06d}.png", size)
        predictions = {
            name: rgb(prediction_roots[name] / filename, size)
            for name in roots
        }
        gt_hand = optional_mask(args.gt_hand_mask_root, index, frame, size)
        gt_object = optional_mask(args.gt_object_mask_root, index, frame, size)
        predicted_foregrounds = {
            name: mask(root / "foreground_masks" / filename, size)
            for name, root in roots.items()
        }
        predicted_arms = {
            name: mask(root / "arm_masks" / filename, size)
            for name, root in roots.items()
        }
        predicted_objects = {
            name: mask(root / "object_masks" / filename, size)
            for name, root in roots.items()
        }
        dynamic_union = np.logical_or.reduce(list(predicted_foregrounds.values()))

        for name, prediction in predictions.items():
            absolute = np.abs(prediction - target)
            values = totals[name]
            values["absolute"] += float(absolute.sum())
            values["square"] += float(np.square(prediction - target).sum())
            values["pixels"] += prediction.size
            own_foreground = predicted_foregrounds[name]
            own_arm = predicted_arms[name]
            own_object = predicted_objects[name]
            add_region(values["regions"], "predicted_foreground", absolute, own_foreground)
            add_region(values["regions"], "predicted_hand_arm", absolute, own_arm)
            add_region(values["regions"], "predicted_object", absolute, own_object)
            add_region(values["regions"], "predicted_background", absolute, ~own_foreground)
            add_region(values["regions"], "predicted_dynamic_union", absolute, dynamic_union)
            if gt_hand is not None:
                add_region(values["regions"], "gt_hand", absolute, gt_hand)
                values["hand_intersection"] += int((own_arm & gt_hand).sum())
                values["hand_union"] += int((own_arm | gt_hand).sum())
            if gt_object is not None:
                add_region(values["regions"], "gt_object", absolute, gt_object)
                values["object_intersection"] += int((own_object & gt_object).sum())
                values["object_union"] += int((own_object | gt_object).sum())

            labels = np.asarray(
                Image.open(roots[name] / "provenance_labels" / filename).convert("L")
                .resize((size, size), Image.Resampling.NEAREST)
            )
            for label, region_name in PROVENANCE_NAMES.items():
                add_region(values["regions"], f"provenance_{region_name}", absolute, labels == label)
            repair = mask(roots[name] / "repair_masks" / filename, size)
            authorized = repair & (~own_foreground) & (~own_object)
            values["authorized_pixels"] += int(authorized.sum())
            values["spatial_pixels"] += int(authorized.size)
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
        region_metrics = {
            region: {
                "l1": item["absolute"] / item["pixels"] if item["pixels"] else None,
                "pixel_fraction": item["pixels"] / values["pixels"],
            } for region, item in values["regions"].items()
        }
        metrics[name] = {
            "source_camera_indices": manifests[name]["source_camera_indices"],
            "state_camera_indices": manifests[name].get("state_camera_indices", [0, 1, 2, 3]),
            "ego_anchor_enabled": manifests[name]["ego_anchor_enabled"],
            "mean_geometrically_observed_fraction": observed_fraction[name],
            "authorized_fraction": values["authorized_pixels"] / values["spatial_pixels"],
            "l1": values["absolute"] / values["pixels"],
            "psnr": float(-10.0 * np.log10(max(mse, 1e-12))),
            "temporal_delta_l1": values["temporal_absolute"] / values["temporal_pixels"],
            "regions": region_metrics,
            "hand_mask_iou": (
                values["hand_intersection"] / values["hand_union"]
                if values["hand_union"] else None
            ),
            "object_mask_iou": (
                values["object_intersection"] / values["object_union"]
                if values["object_union"] else None
            ),
        }
        metrics[name]["dynamic_union_l1"] = region_metrics[
            "predicted_dynamic_union"
        ]["l1"]

    if args.baseline_name in metrics:
        baseline = metrics[args.baseline_name]
        for value in metrics.values():
            value["relative_to_baseline_percent"] = {
                key: 100.0 * (value[key] / baseline[key] - 1.0)
                for key in ("l1", "dynamic_union_l1", "temporal_delta_l1")
                if value[key] is not None and baseline[key] not in (None, 0.0)
            }
    result = {
        "pair_id": reference["pair_id"],
        "frame_count_evaluated": len(records) - 1,
        "dynamic_union_definition": "union of variants' predicted foreground masks; not ground truth",
        "gt_mask_metrics_enabled": {
            "hand": args.gt_hand_mask_root is not None,
            "object": args.gt_object_mask_root is not None,
        },
        "metrics": metrics,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
