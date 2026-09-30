#!/usr/bin/env python3
"""Measure whether an anchored model uses condition timing or only its support."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch
from torch.utils.data import Dataset

from h2o_physics_baseline.dataset import H2OPhysicalClipDataset
from h2o_physics_baseline.model import PhysicalEgoVideoPredictor
from h2o_physics_baseline.protocols import validate_protocol
from h2o_physics_baseline.train import TrainConfig, evaluate, make_loader, seed_everything


class TemporalMapCounterfactual(Dataset):
    """Replace the initial-view state sequence while leaving every other input fixed."""

    def __init__(self, dataset: Dataset, variant: str) -> None:
        self.dataset = dataset
        self.variant = variant

    def __len__(self) -> int:
        return len(self.dataset)

    def __getitem__(self, index: int) -> dict:
        item = self.dataset[index]
        maps = item["initial_ego_maps"]
        if self.variant == "frozen_first":
            maps = maps[:1].repeat(maps.shape[0], axis=0)
        elif self.variant == "reversed_future":
            # Retain the observed t=0 condition and reverse only future states.
            maps = maps.copy()
            maps[1:] = maps[:0:-1]
        elif self.variant != "actual":
            raise ValueError(f"Unknown counterfactual variant: {self.variant}")
        return item | {"initial_ego_maps": maps}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("run_dir", type=Path)
    parser.add_argument("--device", default="cuda:0" if torch.cuda.is_available() else "cpu")
    args = parser.parse_args()

    checkpoint = torch.load(args.run_dir / "best.pt", map_location="cpu", weights_only=False)
    values = dict(checkpoint["config"])
    values["device"] = args.device
    values.setdefault("reprojection_source_stride", 2)
    values.setdefault("evaluation_protocol", "oracle")
    values.setdefault("changed_region_weight", 0.0)
    values.setdefault("anchor_preservation_weight", 0.0)
    values.setdefault("student_state_root", None)
    values.setdefault("student_mask_root", None)
    values.setdefault("initialize_from", None)
    values.setdefault("freeze_base", False)
    values.setdefault(
        "required_test_information",
        tuple(sorted(validate_protocol(values["condition_mode"], values["evaluation_protocol"]))),
    )
    values["source_cameras"] = tuple(values["source_cameras"])
    config = TrainConfig(**values)
    seed_everything(config.seed)

    uses_geometry = config.condition_mode.startswith("geometry")
    uses_multiview = config.condition_mode.startswith("multiview")
    dataset = H2OPhysicalClipDataset(
        config.index,
        split="val",
        frames_per_clip=config.frames_per_clip,
        image_size=config.image_size,
        stats_path=config.stats,
        source_cameras=config.source_cameras,
        max_samples=config.max_val_samples,
        include_reprojection=uses_geometry,
        reprojection_source_stride=config.reprojection_source_stride,
        combine_source_cameras=uses_multiview,
        student_state_root=config.student_state_root,
        student_mask_root=config.student_mask_root,
        oracle_hand_only="oracle_hand" in config.condition_mode,
        student_flow="student_flow" in config.condition_mode,
        student_mask_flow="student_mask_flow" in config.condition_mode,
        source_hand_maps="exo_hand" in config.condition_mode,
        source_joint_geometry="exo_joint" in config.condition_mode,
    )
    device = torch.device(args.device)
    model = PhysicalEgoVideoPredictor(condition_mode=config.condition_mode).to(device)
    model.load_state_dict(checkpoint["model"])

    result = {
        "checkpoint": str(args.run_dir / "best.pt"),
        "checkpoint_epoch": checkpoint["epoch"],
        "condition_mode": config.condition_mode,
        "variants": {},
    }
    for variant in ("actual", "frozen_first", "reversed_future"):
        loader = make_loader(TemporalMapCounterfactual(dataset, variant), config, train=False)
        metrics, _, _ = evaluate(model, loader, config, device)
        result["variants"][variant] = metrics
        print(json.dumps({"variant": variant, "metrics": metrics}, ensure_ascii=False), flush=True)

    destination = args.run_dir / "evaluation_temporal_counterfactuals.json"
    destination.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
