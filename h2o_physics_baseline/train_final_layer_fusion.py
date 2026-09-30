#!/usr/bin/env python3
"""Train/evaluate an authorized fusion head on final-pipeline layer exports."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import random
import sys

import numpy as np
import torch
from torch.utils.data import DataLoader
import torch.nn.functional as F

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from h2o_physics_baseline.final_layer_fusion import (
    AuthorizedFinalFusion,
    FinalLayerWindowDataset,
    RGB_STREAMS,
    parameter_count,
)


def masked_mean(value: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    return (value * mask).sum() / (mask.sum() * value.shape[1]).clamp_min(1.0)


def temporal_delta(value: torch.Tensor) -> torch.Tensor:
    return value[:, :, 1:] - value[:, :, :-1]


def temporal_acceleration(value: torch.Tensor) -> torch.Tensor:
    delta = temporal_delta(value)
    return delta[:, :, 1:] - delta[:, :, :-1]


def augment_training_batch(tensors: dict[str, torch.Tensor]) -> None:
    """Apply one matched image-space transform to every spatial tensor."""
    if random.random() < 0.5:
        for key, value in tuple(tensors.items()):
            # Batched image, mask, provenance, and reliability tensors all use
            # width as their final dimension.  Flipping only RGB silently
            # misregistered half of the training samples from their edit masks.
            if value.ndim >= 4:
                tensors[key] = torch.flip(value, dims=(-1,))
    gain = random.uniform(0.9, 1.1)
    offset = random.uniform(-0.03, 0.03)
    rgb_channels = len(RGB_STREAMS) * 3
    tensors["input"][:, :rgb_channels] = (
        tensors["input"][:, :rgb_channels] * gain + offset
    ).clamp(0.0, 1.0)
    for key in ("base", "proposal", "target"):
        tensors[key] = (tensors[key] * gain + offset).clamp(0.0, 1.0)


@torch.no_grad()
def evaluate(model: AuthorizedFinalFusion, loader: DataLoader, device: torch.device) -> dict[str, float]:
    model.eval()
    totals: dict[str, float] = {}
    count = 0
    for batch in loader:
        tensors = {key: value.to(device) for key, value in batch.items() if torch.is_tensor(value)}
        output, _, _ = model(
            tensors["input"], tensors["base"], tensors["proposal"], tensors["edit_weight"]
        )
        target = tensors["target"]
        auth = tensors["authorized"]
        locked = 1.0 - auth
        noneditable = (tensors["edit_weight"] == 0).float()
        metrics = {
            "l1": F.l1_loss(output, target),
            "authorized_l1": masked_mean((output - target).abs(), auth),
            "editable_l1": masked_mean(
                (output - target).abs(), tensors["edit_weight"]
            ),
            "base_l1": F.l1_loss(tensors["base"], target),
            "proposal_l1": F.l1_loss(tensors["proposal"], target),
            "base_authorized_l1": masked_mean((tensors["base"] - target).abs(), auth),
            "proposal_authorized_l1": masked_mean(
                (tensors["proposal"] - target).abs(), auth
            ),
            "authorized_fraction": auth.mean(),
            "soft_only_fraction": (
                (tensors["edit_weight"] > 0) & (auth == 0)
            ).float().mean(),
            "mean_edit_weight": tensors["edit_weight"].mean(),
            "locked_mean_change": masked_mean(
                (output - tensors["proposal"]).abs(), locked
            ),
            "noneditable_max_change": (
                (output - tensors["proposal"]).abs() * noneditable
            ).max(),
            "delta_l1": F.l1_loss(temporal_delta(output), temporal_delta(target)),
            "acceleration_l1": F.l1_loss(
                temporal_acceleration(output), temporal_acceleration(target)
            ),
        }
        for key, value in metrics.items():
            totals[key] = totals.get(key, 0.0) + float(value)
        count += 1
    return {key: value / max(count, 1) for key, value in totals.items()}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--train-root", type=Path, action="append", required=True)
    parser.add_argument("--val-root", type=Path, action="append", required=True)
    parser.add_argument(
        "--index", type=Path,
        default=Path("datasets/H2O/oracle_state/paired_physical_clips.csv"),
    )
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--width", type=int, default=16)
    parser.add_argument("--blocks", type=int, default=1)
    parser.add_argument(
        "--blend-mode", choices=("convex", "legacy_extrapolating"),
        default="convex",
    )
    parser.add_argument("--residual-limit", type=float, default=0.03)
    parser.add_argument(
        "--allow-split-mismatch", action="store_true",
        help="Permit non-train/non-validation roots (for explicit diagnostics only).",
    )
    parser.add_argument("--window", type=int, default=5)
    parser.add_argument("--stride", type=int, default=2)
    parser.add_argument("--epochs", type=int, default=5)
    parser.add_argument("--patience", type=int, default=3)
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--lr", type=float, default=2e-4)
    # Temporal consistency is part of the standard terminal-fusion objective;
    # zeroing these values is an ablation, not the default training protocol.
    parser.add_argument("--temporal-weight", type=float, default=0.3)
    parser.add_argument("--acceleration-weight", type=float, default=0.15)
    parser.add_argument("--temporal-regression-weight", type=float, default=1.0)
    parser.add_argument(
        "--require-temporal-nonregression",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Only replace the current final output when delta L1 does not regress.",
    )
    parser.add_argument("--temporal-tolerance", type=float, default=0.0)
    parser.add_argument("--seed", type=int, default=26)
    parser.add_argument("--device", default="cuda")
    parser.add_argument(
        "--augmentation", action=argparse.BooleanOptionalAction, default=True,
        help="Matched horizontal and photometric augmentation for the tiny training set.",
    )
    args = parser.parse_args()

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    device = torch.device(args.device if args.device != "cuda" or torch.cuda.is_available() else "cpu")
    train_data = FinalLayerWindowDataset(args.train_root, args.index, args.window, args.stride)
    val_data = FinalLayerWindowDataset(args.val_root, args.index, args.window, args.window)
    train_splits = sorted({clip.split for clip in train_data.clips})
    val_splits = sorted({clip.split for clip in val_data.clips})
    if not args.allow_split_mismatch and (
        train_splits != ["train"] or val_splits != ["val"]
    ):
        raise ValueError(
            "Expected train roots from split=train and validation roots from "
            f"split=val; found train={train_splits}, val={val_splits}"
        )
    train_loader = DataLoader(train_data, batch_size=args.batch_size, shuffle=True, num_workers=0)
    val_loader = DataLoader(val_data, batch_size=1, shuffle=False, num_workers=0)
    model = AuthorizedFinalFusion(
        args.width, args.blocks, args.residual_limit, args.blend_mode
    ).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)
    args.output.mkdir(parents=True, exist_ok=True)

    initial_metrics = evaluate(model, val_loader, device)
    history = [{"epoch": 0, "train_loss": None, **initial_metrics}]
    print(json.dumps(history[0], ensure_ascii=False), flush=True)
    best = initial_metrics["editable_l1"]
    best_epoch = 0
    initial_checkpoint = {
        "model": model.state_dict(),
        "config": vars(args) | {"parameter_count": parameter_count(model)},
        "metrics": initial_metrics,
    }
    torch.save(initial_checkpoint, args.output / "best.pt")
    for epoch in range(1, args.epochs + 1):
        model.train()
        running = 0.0
        for batch in train_loader:
            tensors = {key: value.to(device) for key, value in batch.items() if torch.is_tensor(value)}
            if args.augmentation:
                augment_training_batch(tensors)
            output, blend_control, residual = model(
                tensors["input"], tensors["base"], tensors["proposal"], tensors["edit_weight"]
            )
            auth = tensors["authorized"]
            edit_weight = tensors["edit_weight"]
            target = tensors["target"]
            # Supervise every pixel the model is permitted to change.  The old
            # authorized-only loss gave soft reliability corrections no direct
            # reconstruction signal even though the forward pass exposed them.
            reconstruction = masked_mean((output - target).abs(), edit_weight)
            delta_mask = torch.minimum(
                edit_weight[:, :, 1:], edit_weight[:, :, :-1]
            )
            masked_temporal = masked_mean(
                (temporal_delta(output) - temporal_delta(target)).abs(), delta_mask
            )
            full_temporal = F.l1_loss(temporal_delta(output), temporal_delta(target))
            full_acceleration = F.l1_loss(
                temporal_acceleration(output), temporal_acceleration(target)
            )
            proposal_temporal = F.l1_loss(
                temporal_delta(tensors["proposal"]), temporal_delta(target)
            )
            proposal_acceleration = F.l1_loss(
                temporal_acceleration(tensors["proposal"]), temporal_acceleration(target)
            )
            regression = F.relu(full_temporal - proposal_temporal) + F.relu(
                full_acceleration - proposal_acceleration
            )
            regularizer = masked_mean(residual.abs(), auth) + 0.05 * masked_mean(
                (blend_control[:, :, 1:] - blend_control[:, :, :-1]).abs(), delta_mask
            )
            loss = (
                reconstruction
                + args.temporal_weight * masked_temporal
                + args.acceleration_weight * full_acceleration
                + args.temporal_regression_weight * regression
                + 0.03 * regularizer
            )
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            optimizer.step()
            running += float(loss.detach())
        metrics = evaluate(model, val_loader, device)
        record = {"epoch": epoch, "train_loss": running / max(len(train_loader), 1), **metrics}
        history.append(record)
        print(json.dumps(record, ensure_ascii=False), flush=True)
        checkpoint = {
            "model": model.state_dict(),
            "config": vars(args) | {"parameter_count": parameter_count(model)},
            "metrics": metrics,
        }
        torch.save(checkpoint, args.output / "latest.pt")
        temporal_allowed = (
            not args.require_temporal_nonregression
            or metrics["delta_l1"]
            <= initial_metrics["delta_l1"] + args.temporal_tolerance
        )
        if metrics["editable_l1"] < best and temporal_allowed:
            best = metrics["editable_l1"]
            best_epoch = epoch
            torch.save(checkpoint, args.output / "best.pt")
        if epoch - best_epoch >= args.patience:
            print(
                json.dumps({"early_stop": epoch, "best_epoch": best_epoch}),
                flush=True,
            )
            break

    summary = {
        "protocol": "current final layered pipeline; terminal authorized fusion only",
        "train_pair_ids": [clip.pair_id for clip in train_data.clips],
        "val_pair_ids": [clip.pair_id for clip in val_data.clips],
        "train_splits": train_splits,
        "val_splits": val_splits,
        "train_windows": len(train_data),
        "val_windows": len(val_data),
        "parameter_count": parameter_count(model),
        "best_epoch": best_epoch,
        "best_editable_l1": best,
        "initialization": "exact current constrained-ProPainter output",
        "correction_policy": (
            "full edits in authorized repair pixels; reliability-weighted corrections "
            "up to 25% at seams and hand/object boundaries, smaller corrections on "
            "transported source interiors"
        ),
        "history": history,
    }
    (args.output / "summary.json").write_text(
        json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8"
    )


if __name__ == "__main__":
    main()
