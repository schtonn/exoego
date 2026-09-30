#!/usr/bin/env python3
"""Exercise the complete H2O loader/model/loss path on real data."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import torch

from h2o_physics_baseline.dataset import H2OPhysicalClipDataset
from h2o_physics_baseline.model import PhysicalEgoVideoPredictor
from h2o_physics_baseline.train import reconstruction_terms


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--device", default="cuda:0" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--image-size", type=int, default=64)
    parser.add_argument("--frames", type=int, default=4)
    parser.add_argument(
        "--index", type=Path,
        default=Path("datasets/H2O/oracle_state/paired_physical_clips.csv"),
    )
    parser.add_argument(
        "--stats", type=Path,
        default=Path("datasets/H2O/oracle_state/train_feature_stats.json"),
    )
    args = parser.parse_args()
    dataset = H2OPhysicalClipDataset(
        split="train",
        index_path=args.index,
        stats_path=args.stats,
        frames_per_clip=args.frames,
        image_size=args.image_size,
        max_samples=2,
        include_reprojection=True,
    )
    sample = dataset[0]
    assert sample["source"].shape == (args.frames, 3, args.image_size, args.image_size)
    assert sample["ego_anchor"].shape == (args.frames, 3, args.image_size, args.image_size)
    assert np.array_equal(sample["ego_anchor"][0], sample["target"][0])
    assert np.array_equal(sample["ego_anchor"][0], sample["ego_anchor"][-1])
    assert sample["physical_vector"].shape == (args.frames, dataset.physical_dim)
    assert sample["physical_maps"].shape == (
        args.frames,
        dataset.map_channels,
        args.image_size,
        args.image_size,
    )
    assert sample["initial_ego_maps"].shape == sample["physical_maps"].shape
    assert np.isfinite(sample["physical_vector"]).all()
    assert np.isfinite(sample["physical_maps"]).all()
    assert sample["coarse_rgb"].shape == (args.frames, 3, args.image_size, args.image_size)
    assert sample["visibility_mask"].shape == (args.frames, 1, args.image_size, args.image_size)
    assert 0 < float(sample["visibility_mask"].mean()) < 1
    device = torch.device(args.device)
    tensors = {
        name: torch.from_numpy(sample[name]).unsqueeze(0).to(device)
        for name in (
            "source",
            "target",
            "ego_anchor",
            "physical_vector",
            "physical_maps",
            "initial_ego_maps",
            "coarse_rgb",
            "coarse_depth",
            "visibility_mask",
        )
    }
    model = PhysicalEgoVideoPredictor(condition_mode="geometry_state_distance").to(device)
    prediction, tokens = model(
        tensors["source"],
        tensors["physical_vector"],
        tensors["physical_maps"],
        tensors["coarse_rgb"],
        tensors["coarse_depth"],
        tensors["visibility_mask"],
        tensors["ego_anchor"],
        tensors["initial_ego_maps"],
    )
    terms = reconstruction_terms(
        prediction,
        tensors["target"],
        tensors["physical_maps"],
        tensors["visibility_mask"],
        tensors["coarse_rgb"],
        tensors["ego_anchor"],
    )
    sum(terms.values()).backward()
    result = {
        "pair_id": sample["pair_id"],
        "source": list(tensors["source"].shape),
        "physical_vector": list(tensors["physical_vector"].shape),
        "physical_maps": list(tensors["physical_maps"].shape),
        "prediction": list(prediction.shape),
        "condition_tokens": list(tokens.shape),
        "metrics": {name: float(value.detach()) for name, value in terms.items()},
        "parameters": sum(parameter.numel() for parameter in model.parameters()),
        "device": str(device),
    }
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
