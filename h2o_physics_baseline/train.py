#!/usr/bin/env python3
"""Train and evaluate the compact H2O physical conditioning baseline."""

from __future__ import annotations

import argparse
import json
import math
import random
import time
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np
import torch
from PIL import Image, ImageDraw
from torch import nn
from torch.nn import functional
from torch.utils.data import DataLoader

from h2o_physics_baseline.dataset import DEFAULT_INDEX, DEFAULT_STATS, H2OPhysicalClipDataset
from h2o_physics_baseline.model import PhysicalEgoVideoPredictor
from h2o_physics_baseline.protocols import PROTOCOL_ALLOWED, validate_protocol


@dataclass
class TrainConfig:
    index: str
    stats: str
    output_dir: str
    condition_mode: str
    evaluation_protocol: str
    required_test_information: tuple[str, ...]
    frames_per_clip: int
    image_size: int
    batch_size: int
    epochs: int
    learning_rate: float
    weight_decay: float
    temporal_weight: float
    hand_region_weight: float
    changed_region_weight: float
    anchor_preservation_weight: float
    student_state_root: str | None
    student_mask_root: str | None
    workers: int
    seed: int
    max_train_samples: int | None
    max_val_samples: int | None
    max_steps_per_epoch: int | None
    source_cameras: tuple[str, ...]
    reprojection_source_stride: int
    device: str
    initialize_from: str | None
    freeze_base: bool


def seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def to_device(batch: dict, device: torch.device) -> dict[str, torch.Tensor]:
    names = (
        "source",
        "target",
        "ego_anchor",
        "physical_vector",
        "physical_maps",
        "initial_ego_maps",
        "source_hand_maps",
        "source_joint_uv",
        "source_joint_confidence",
        "target_joint_uv",
        "target_joint_confidence",
        "coarse_rgb",
        "coarse_depth",
        "visibility_mask",
    )
    values = {
        name: batch[name].to(device=device, dtype=torch.float32, non_blocking=True) for name in names
    }
    for name, value in values.items():
        if not torch.isfinite(value).all():
            raise FloatingPointError(f"Non-finite values in batch field {name}")
    return values


def reconstruction_terms(
    prediction: torch.Tensor,
    target: torch.Tensor,
    maps: torch.Tensor,
    visibility_mask: torch.Tensor | None = None,
    coarse_rgb: torch.Tensor | None = None,
    ego_anchor: torch.Tensor | None = None,
    change_threshold: float = 0.05,
) -> dict[str, torch.Tensor]:
    absolute = (prediction - target).abs()
    l1 = absolute.mean()
    mse = torch.square(prediction - target).mean()
    prediction_delta = prediction[:, 1:] - prediction[:, :-1]
    target_delta = target[:, 1:] - target[:, :-1]
    temporal = (prediction_delta - target_delta).abs().mean()
    if prediction.shape[1] >= 3:
        prediction_acceleration = prediction[:, 2:] - 2 * prediction[:, 1:-1] + prediction[:, :-2]
        target_acceleration = target[:, 2:] - 2 * target[:, 1:-1] + target[:, :-2]
        temporal_acceleration = (prediction_acceleration - target_acceleration).abs().mean()
    else:
        temporal_acceleration = torch.zeros((), device=prediction.device, dtype=prediction.dtype)
    # Channels 0:4 cover hands and contact-weighted joints. A soft floor keeps
    # this metric stable for frames where both hands are outside the ego view.
    hand_mask = maps[:, :, :4].amax(dim=2, keepdim=True)
    numerator = (absolute * hand_mask).sum()
    denominator = hand_mask.sum() * prediction.shape[2]
    hand_l1 = numerator / denominator.clamp_min(1.0)
    binary_hand = (hand_mask > 0.1).to(prediction.dtype)
    flat_hand = binary_hand.reshape(-1, 1, prediction.shape[-2], prediction.shape[-1])
    dilated_hand = functional.max_pool2d(flat_hand, 3, stride=1, padding=1)
    eroded_hand = 1.0 - functional.max_pool2d(1.0 - flat_hand, 3, stride=1, padding=1)
    hand_boundary = (dilated_hand - eroded_hand).reshape_as(binary_hand)
    hand_boundary_l1 = (absolute * hand_boundary).sum() / (
        hand_boundary.sum() * prediction.shape[2]
    ).clamp_min(1.0)
    if visibility_mask is None:
        visibility_mask = torch.zeros_like(prediction[:, :, :1])
    known_l1 = (absolute * visibility_mask).sum() / (visibility_mask.sum() * prediction.shape[2]).clamp_min(1.0)
    hole_mask = 1.0 - visibility_mask
    hole_l1 = (absolute * hole_mask).sum() / (hole_mask.sum() * prediction.shape[2]).clamp_min(1.0)
    if coarse_rgb is None:
        coarse_known_l1 = torch.zeros((), device=prediction.device, dtype=prediction.dtype)
    else:
        coarse_absolute = (coarse_rgb - target).abs()
        coarse_known_l1 = (coarse_absolute * visibility_mask).sum() / (
            visibility_mask.sum() * prediction.shape[2]
        ).clamp_min(1.0)
    if ego_anchor is None:
        changed_fraction = torch.zeros((), device=prediction.device, dtype=prediction.dtype)
        changed_l1 = torch.zeros_like(changed_fraction)
        anchor_changed_l1 = torch.zeros_like(changed_fraction)
        unchanged_l1 = torch.zeros_like(changed_fraction)
        anchor_unchanged_l1 = torch.zeros_like(changed_fraction)
        unchanged_anchor_drift_l1 = torch.zeros_like(changed_fraction)
    else:
        changed = ((target - ego_anchor).abs().mean(dim=2, keepdim=True) > change_threshold).to(
            prediction.dtype
        )
        changed[:, 0] = 0
        changed_fraction = changed.mean()
        changed_denominator = (changed.sum() * prediction.shape[2]).clamp_min(1.0)
        changed_l1 = (absolute * changed).sum() / changed_denominator
        anchor_changed_l1 = ((ego_anchor - target).abs() * changed).sum() / changed_denominator
        unchanged = 1.0 - changed
        unchanged_denominator = (unchanged.sum() * prediction.shape[2]).clamp_min(1.0)
        unchanged_l1 = (absolute * unchanged).sum() / unchanged_denominator
        anchor_unchanged_l1 = ((ego_anchor - target).abs() * unchanged).sum() / unchanged_denominator
        unchanged_anchor_drift_l1 = ((prediction - ego_anchor).abs() * unchanged).sum() / unchanged_denominator
    return {
        "l1": l1,
        "mse": mse,
        "temporal_l1": temporal,
        "temporal_acceleration_l1": temporal_acceleration,
        "prediction_motion_l1": prediction_delta.abs().mean(),
        "target_motion_l1": target_delta.abs().mean(),
        "hand_l1": hand_l1,
        "hand_boundary_l1": hand_boundary_l1,
        "known_l1": known_l1,
        "hole_l1": hole_l1,
        "coarse_known_l1": coarse_known_l1,
        "changed_fraction": changed_fraction,
        "changed_l1": changed_l1,
        "static_anchor_changed_l1": anchor_changed_l1,
        "changed_gain_over_static": anchor_changed_l1 - changed_l1,
        "unchanged_l1": unchanged_l1,
        "static_anchor_unchanged_l1": anchor_unchanged_l1,
        "unchanged_anchor_drift_l1": unchanged_anchor_drift_l1,
    }


def objective(terms: dict[str, torch.Tensor], config: TrainConfig) -> torch.Tensor:
    return (
        terms["l1"]
        + config.temporal_weight * terms["temporal_l1"]
        + config.hand_region_weight * terms["hand_l1"]
        + config.changed_region_weight * terms["changed_l1"]
        + config.anchor_preservation_weight * terms["unchanged_anchor_drift_l1"]
    )


def make_loader(dataset: H2OPhysicalClipDataset, config: TrainConfig, train: bool) -> DataLoader:
    generator = torch.Generator().manual_seed(config.seed + (0 if train else 1))
    return DataLoader(
        dataset,
        batch_size=config.batch_size,
        shuffle=train,
        num_workers=config.workers,
        pin_memory=config.device.startswith("cuda"),
        persistent_workers=config.workers > 0,
        generator=generator,
        drop_last=train and len(dataset) >= config.batch_size,
    )


def train_epoch(
    model: nn.Module,
    loader: DataLoader,
    optimizer: torch.optim.Optimizer,
    config: TrainConfig,
    device: torch.device,
) -> dict[str, float]:
    model.train()
    totals = {
        name: 0.0
        for name in (
            "loss",
            "l1",
            "mse",
            "temporal_l1",
            "temporal_acceleration_l1",
            "prediction_motion_l1",
            "target_motion_l1",
            "hand_l1",
            "hand_boundary_l1",
            "known_l1",
            "hole_l1",
            "coarse_known_l1",
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
    for step, batch in enumerate(loader):
        if config.max_steps_per_epoch is not None and step >= config.max_steps_per_epoch:
            break
        tensors = to_device(batch, device)
        optimizer.zero_grad(set_to_none=True)
        with torch.autocast(device_type=device.type, dtype=torch.bfloat16, enabled=device.type == "cuda"):
            prediction, _ = model(
                tensors["source"],
                tensors["physical_vector"],
                tensors["physical_maps"],
                tensors["coarse_rgb"],
                tensors["coarse_depth"],
                tensors["visibility_mask"],
                tensors["ego_anchor"],
                tensors["initial_ego_maps"],
                tensors["source_hand_maps"],
                tensors["source_joint_uv"],
                tensors["source_joint_confidence"],
                tensors["target_joint_uv"],
                tensors["target_joint_confidence"],
            )
            terms = reconstruction_terms(
                prediction,
                tensors["target"],
                tensors["physical_maps"],
                tensors["visibility_mask"],
                tensors["coarse_rgb"],
                tensors["ego_anchor"],
            )
            loss = objective(terms, config)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        count = tensors["source"].shape[0]
        samples += count
        totals["loss"] += float(loss.detach()) * count
        for name, value in terms.items():
            totals[name] += float(value.detach()) * count
    if samples == 0:
        raise RuntimeError("Training loader produced no samples")
    return {name: value / samples for name, value in totals.items()} | {"samples": samples}


@torch.inference_mode()
def evaluate(
    model: nn.Module,
    loader: DataLoader,
    config: TrainConfig,
    device: torch.device,
) -> tuple[dict[str, float], dict | None, torch.Tensor | None]:
    model.eval()
    totals = {
        name: 0.0
        for name in (
            "loss",
            "l1",
            "mse",
            "temporal_l1",
            "temporal_acceleration_l1",
            "prediction_motion_l1",
            "target_motion_l1",
            "hand_l1",
            "hand_boundary_l1",
            "known_l1",
            "hole_l1",
            "coarse_known_l1",
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
    preview_batch = None
    preview_prediction = None
    for batch in loader:
        tensors = to_device(batch, device)
        with torch.autocast(device_type=device.type, dtype=torch.bfloat16, enabled=device.type == "cuda"):
            prediction, _ = model(
                tensors["source"],
                tensors["physical_vector"],
                tensors["physical_maps"],
                tensors["coarse_rgb"],
                tensors["coarse_depth"],
                tensors["visibility_mask"],
                tensors["ego_anchor"],
                tensors["initial_ego_maps"],
                tensors["source_hand_maps"],
                tensors["source_joint_uv"],
                tensors["source_joint_confidence"],
                tensors["target_joint_uv"],
                tensors["target_joint_confidence"],
            )
            terms = reconstruction_terms(
                prediction,
                tensors["target"],
                tensors["physical_maps"],
                tensors["visibility_mask"],
                tensors["coarse_rgb"],
                tensors["ego_anchor"],
            )
            loss = objective(terms, config)
        count = tensors["source"].shape[0]
        samples += count
        totals["loss"] += float(loss) * count
        for name, value in terms.items():
            totals[name] += float(value) * count
        if preview_batch is None:
            preview_batch = batch
            preview_prediction = prediction.float().cpu()
    metrics = {name: value / samples for name, value in totals.items()}
    metrics["psnr"] = -10.0 * math.log10(max(metrics["mse"], 1e-12))
    metrics["samples"] = samples
    return metrics, preview_batch, preview_prediction


def _frame_to_image(frame: torch.Tensor | np.ndarray, size: int) -> Image.Image:
    array = frame.detach().cpu().numpy() if isinstance(frame, torch.Tensor) else frame
    array = np.moveaxis(np.clip(array, 0, 1), 0, -1)
    return Image.fromarray(np.rint(array * 255).astype(np.uint8)).resize((size, size))


def save_preview(batch: dict, prediction: torch.Tensor, path: Path) -> None:
    frames = min(6, prediction.shape[1])
    tile = 192
    header = 26
    canvas = Image.new("RGB", (frames * tile, 3 * tile + header), "white")
    draw = ImageDraw.Draw(canvas)
    draw.text((5, 5), "rows: exo source | ego prediction | ego target", fill="black")
    source_preview = batch["source"][0]
    if source_preview.ndim == 5:
        source_preview = source_preview[0]
    for index in range(frames):
        values = (source_preview[index], prediction[0, index], batch["target"][0, index])
        for row, value in enumerate(values):
            canvas.paste(_frame_to_image(value, tile), (index * tile, header + row * tile))
    path.parent.mkdir(parents=True, exist_ok=True)
    canvas.save(path)


def build_config(args: argparse.Namespace) -> TrainConfig:
    required = validate_protocol(args.condition_mode, args.evaluation_protocol)
    return TrainConfig(
        index=str(args.index),
        stats=str(args.stats),
        output_dir=str(args.output_dir),
        condition_mode=args.condition_mode,
        evaluation_protocol=args.evaluation_protocol,
        required_test_information=tuple(sorted(required)),
        frames_per_clip=args.frames_per_clip,
        image_size=args.image_size,
        batch_size=args.batch_size,
        epochs=args.epochs,
        learning_rate=args.learning_rate,
        weight_decay=args.weight_decay,
        temporal_weight=args.temporal_weight,
        hand_region_weight=args.hand_region_weight,
        changed_region_weight=args.changed_region_weight,
        anchor_preservation_weight=args.anchor_preservation_weight,
        student_state_root=str(args.student_state_root) if args.student_state_root else None,
        student_mask_root=str(args.student_mask_root) if args.student_mask_root else None,
        workers=args.workers,
        seed=args.seed,
        max_train_samples=args.max_train_samples,
        max_val_samples=args.max_val_samples,
        max_steps_per_epoch=args.max_steps_per_epoch,
        source_cameras=tuple(args.source_cameras),
        reprojection_source_stride=args.reprojection_source_stride,
        device=args.device,
        initialize_from=str(args.initialize_from) if args.initialize_from else None,
        freeze_base=args.freeze_base,
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--index", type=Path, default=DEFAULT_INDEX)
    parser.add_argument("--stats", type=Path, default=DEFAULT_STATS)
    parser.add_argument("--output-dir", type=Path, default=Path("datasets/H2O/experiments/physical_baseline"))
    parser.add_argument("--condition-mode", choices=sorted(PhysicalEgoVideoPredictor.valid_modes), default="rgb_state")
    parser.add_argument(
        "--evaluation-protocol",
        choices=sorted(PROTOCOL_ALLOWED),
        default="oracle",
        help="Test-time information contract. Non-oracle protocols reject modes with target-view leakage.",
    )
    parser.add_argument("--frames-per-clip", type=int, default=8)
    parser.add_argument("--image-size", type=int, default=128)
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--learning-rate", type=float, default=2e-4)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--temporal-weight", type=float, default=0.25)
    parser.add_argument("--hand-region-weight", type=float, default=0.5)
    parser.add_argument(
        "--changed-region-weight",
        type=float,
        default=0.0,
        help="Training-only weight on pixels that differ from the supplied ego anchor.",
    )
    parser.add_argument(
        "--anchor-preservation-weight",
        type=float,
        default=0.0,
        help="Training-only penalty on prediction drift from the anchor outside changed pixels.",
    )
    parser.add_argument(
        "--student-state-root",
        type=Path,
        help="Precomputed exo-only student state root; required by student condition modes.",
    )
    parser.add_argument(
        "--student-mask-root",
        type=Path,
        help="Precomputed initial-ego hand masks; required by student mask-flow modes.",
    )
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--seed", type=int, default=20260925)
    parser.add_argument("--max-train-samples", type=int)
    parser.add_argument("--max-val-samples", type=int)
    parser.add_argument("--max-steps-per-epoch", type=int)
    parser.add_argument("--source-cameras", nargs="+", default=["cam0", "cam1", "cam2", "cam3"])
    parser.add_argument("--reprojection-source-stride", type=int, default=2)
    parser.add_argument("--device", default="cuda:0" if torch.cuda.is_available() else "cpu")
    parser.add_argument(
        "--initialize-from",
        type=Path,
        help="Checkpoint whose shape-compatible model weights initialize this run.",
    )
    parser.add_argument(
        "--freeze-base",
        action="store_true",
        help="Train only source-joint and joint-residual modules after initialization.",
    )
    args = parser.parse_args()
    if args.image_size % 16:
        parser.error("--image-size must be divisible by 16")
    config = build_config(args)
    if "student" in config.condition_mode and config.student_state_root is None:
        parser.error("student condition modes require --student-state-root")
    if "student_mask_flow" in config.condition_mode and config.student_mask_root is None:
        parser.error("student mask-flow modes require --student-mask-root")
    seed_everything(config.seed)
    output_dir = Path(config.output_dir) / config.condition_mode
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "config.json").write_text(json.dumps(asdict(config), indent=2) + "\n", encoding="utf-8")

    uses_geometry = config.condition_mode.startswith("geometry")
    uses_multiview = config.condition_mode.startswith("multiview")
    train_data = H2OPhysicalClipDataset(
        config.index,
        split="train",
        frames_per_clip=config.frames_per_clip,
        image_size=config.image_size,
        stats_path=config.stats,
        source_cameras=config.source_cameras,
        max_samples=config.max_train_samples,
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
    val_data = H2OPhysicalClipDataset(
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
    train_loader = make_loader(train_data, config, train=True)
    val_loader = make_loader(val_data, config, train=False)
    device = torch.device(config.device)
    model = PhysicalEgoVideoPredictor(condition_mode=config.condition_mode).to(device)
    if config.initialize_from:
        initialization = torch.load(config.initialize_from, map_location="cpu", weights_only=False)
        source_state = initialization["model"]
        target_state = model.state_dict()
        copied = []
        for name, value in source_state.items():
            if name in target_state and target_state[name].shape == value.shape:
                target_state[name] = value
                copied.append(name)
        model.load_state_dict(target_state)
        print(
            json.dumps(
                {"initialized_from": config.initialize_from, "copied_tensors": len(copied)},
                ensure_ascii=False,
            ),
            flush=True,
        )
    if config.freeze_base:
        for name, parameter in model.named_parameters():
            parameter.requires_grad = name.startswith(("source_joint_", "joint_motion_head"))
    optimizer = torch.optim.AdamW(
        [parameter for parameter in model.parameters() if parameter.requires_grad],
        lr=config.learning_rate,
        weight_decay=config.weight_decay,
    )
    parameter_count = sum(parameter.numel() for parameter in model.parameters())
    repair_parameter_count = (
        sum(parameter.numel() for parameter in model.local_motion_head.parameters())
        if hasattr(model, "local_motion_head")
        else 0
    )
    history = []
    best_loss = float("inf")
    start_time = time.time()
    for epoch in range(1, config.epochs + 1):
        train_metrics = train_epoch(model, train_loader, optimizer, config, device)
        val_metrics, preview_batch, preview_prediction = evaluate(model, val_loader, config, device)
        record = {"epoch": epoch, "train": train_metrics, "val": val_metrics}
        history.append(record)
        print(json.dumps(record, ensure_ascii=False), flush=True)
        checkpoint = {
            "model": model.state_dict(),
            "optimizer": optimizer.state_dict(),
            "epoch": epoch,
            "config": asdict(config),
            "parameter_count": parameter_count,
            "repair_parameter_count": repair_parameter_count,
            "val": val_metrics,
        }
        torch.save(checkpoint, output_dir / "latest.pt")
        if val_metrics["loss"] < best_loss:
            best_loss = val_metrics["loss"]
            torch.save(checkpoint, output_dir / "best.pt")
        if preview_batch is not None and preview_prediction is not None:
            save_preview(preview_batch, preview_prediction, output_dir / f"preview_epoch_{epoch:03d}.jpg")
        (output_dir / "history.json").write_text(
            json.dumps(history, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
        )
    best_record = min(history, key=lambda item: item["val"]["loss"])
    summary = {
        "condition_mode": config.condition_mode,
        "parameter_count": parameter_count,
        "repair_parameter_count": repair_parameter_count,
        "train_samples": len(train_data),
        "val_samples": len(val_data),
        "best_val_loss": best_loss,
        "best_epoch": best_record["epoch"],
        "best_val": best_record["val"],
        "elapsed_seconds": time.time() - start_time,
        "last": history[-1],
    }
    (output_dir / "summary.json").write_text(
        json.dumps(summary, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )


if __name__ == "__main__":
    main()
