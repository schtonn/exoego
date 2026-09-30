#!/usr/bin/env python3
"""Evaluate repeating the first ego frame without using the exo video."""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

import torch
from torch.utils.data import DataLoader

from h2o_physics_baseline.dataset import H2OPhysicalClipDataset
from h2o_physics_baseline.train import reconstruction_terms


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--frames-per-clip", type=int, default=4)
    parser.add_argument("--image-size", type=int, default=64)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--max-val-samples", type=int, default=32)
    parser.add_argument("--source-camera", default="cam3")
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("datasets/H2O/experiments/anchored_pilot/static_anchor.json"),
    )
    args = parser.parse_args()
    dataset = H2OPhysicalClipDataset(
        split="val",
        frames_per_clip=args.frames_per_clip,
        image_size=args.image_size,
        source_cameras=(args.source_camera,),
        max_samples=args.max_val_samples,
        include_reprojection=False,
    )
    loader = DataLoader(dataset, batch_size=args.batch_size, shuffle=False, num_workers=0)
    totals = {
        name: 0.0
        for name in (
            "l1",
            "mse",
            "temporal_l1",
            "prediction_motion_l1",
            "target_motion_l1",
            "hand_l1",
            "changed_fraction",
            "changed_l1",
            "static_anchor_changed_l1",
            "changed_gain_over_static",
            "unchanged_l1",
            "static_anchor_unchanged_l1",
            "unchanged_anchor_drift_l1",
        )
    }
    samples = 0
    for batch in loader:
        target = batch["target"].float()
        prediction = batch["ego_anchor"].float()
        maps = batch["physical_maps"].float()
        terms = reconstruction_terms(prediction, target, maps, ego_anchor=prediction)
        count = target.shape[0]
        samples += count
        for name in totals:
            totals[name] += float(terms[name]) * count
    metrics = {name: value / samples for name, value in totals.items()}
    metrics["psnr"] = -10.0 * math.log10(max(metrics["mse"], 1e-12))
    result = {
        "method": "repeat_first_ego_frame",
        "uses_exo_video": False,
        "samples": samples,
        "frames_per_clip": args.frames_per_clip,
        "image_size": args.image_size,
        "source_camera_for_split_matching_only": args.source_camera,
        "metrics": metrics,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
