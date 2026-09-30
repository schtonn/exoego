#!/usr/bin/env python3
"""Re-evaluate a saved run with current diagnostics without changing weights."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch

from h2o_physics_baseline.dataset import H2OPhysicalClipDataset
from h2o_physics_baseline.model import PhysicalEgoVideoPredictor
from h2o_physics_baseline.protocols import validate_protocol
from h2o_physics_baseline.train import TrainConfig, evaluate, make_loader, seed_everything


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("run_dir", type=Path)
    parser.add_argument("--device", default="cuda:0" if torch.cuda.is_available() else "cpu")
    args = parser.parse_args()
    checkpoint = torch.load(args.run_dir / "best.pt", map_location="cpu", weights_only=False)
    config_values = dict(checkpoint["config"])
    config_values["device"] = args.device
    config_values.setdefault("reprojection_source_stride", 2)
    config_values.setdefault("evaluation_protocol", "oracle")
    config_values.setdefault("changed_region_weight", 0.0)
    config_values.setdefault("anchor_preservation_weight", 0.0)
    config_values.setdefault("student_state_root", None)
    config_values.setdefault("student_mask_root", None)
    config_values.setdefault("initialize_from", None)
    config_values.setdefault("freeze_base", False)
    config_values.setdefault(
        "required_test_information",
        tuple(sorted(validate_protocol(config_values["condition_mode"], "oracle"))),
    )
    config_values["source_cameras"] = tuple(config_values["source_cameras"])
    config = TrainConfig(**config_values)
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
    loader = make_loader(dataset, config, train=False)
    device = torch.device(args.device)
    model = PhysicalEgoVideoPredictor(condition_mode=config.condition_mode).to(device)
    model.load_state_dict(checkpoint["model"])
    metrics, _, _ = evaluate(model, loader, config, device)
    output = {
        "checkpoint": str(args.run_dir / "best.pt"),
        "checkpoint_epoch": checkpoint["epoch"],
        "condition_mode": config.condition_mode,
        "metrics": metrics,
    }
    destination = args.run_dir / "evaluation_extended.json"
    destination.write_text(json.dumps(output, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(output, indent=2))


if __name__ == "__main__":
    main()
