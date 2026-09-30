#!/usr/bin/env python3
"""Evaluate non-learned initial-frame warps induced by estimated hand motion."""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

import torch
import torch.nn.functional as functional

from h2o_physics_baseline.dataset import H2OPhysicalClipDataset
from h2o_physics_baseline.train import TrainConfig, make_loader, reconstruction_terms, seed_everything


def warp_anchor(
    anchor: torch.Tensor, maps: torch.Tensor, scale: float, support_power: float
) -> torch.Tensor:
    """Backward-warp the anchor at current joint locations using initial-to-current flow."""
    batch, time, _, height, width = anchor.shape
    ys = (torch.arange(height, device=anchor.device, dtype=anchor.dtype) + 0.5) * 2 / height - 1
    xs = (torch.arange(width, device=anchor.device, dtype=anchor.dtype) + 0.5) * 2 / width - 1
    yy, xx = torch.meshgrid(ys, xs, indexing="ij")
    base = torch.stack((xx, yy), dim=-1)[None].expand(batch * time, -1, -1, -1)
    flow = maps[:, :, 2:4].reshape(batch * time, 2, height, width).permute(0, 2, 3, 1)
    if support_power > 0:
        support = maps[:, :, :2].amax(dim=2).reshape(batch * time, height, width, 1)
        flow = flow * support.pow(support_power)
    grid = base - 2.0 * scale * flow
    warped = functional.grid_sample(
        anchor.reshape(batch * time, 3, height, width),
        grid,
        mode="bilinear",
        padding_mode="border",
        align_corners=False,
    )
    return warped.reshape(batch, time, 3, height, width)


def fill_old_hand_location(
    anchor: torch.Tensor,
    transported: torch.Tensor,
    maps: torch.Tensor,
    blur_kernel: int,
    mask_dilation: int,
) -> torch.Tensor:
    """Replace initial-only hand support with a local background estimate."""
    batch, time, _, height, width = anchor.shape
    uses_mask_support = bool(torch.any(maps[:, :, 4:5] > 0))
    if uses_mask_support:
        initial_support = maps[:, :1, 4:5].expand(-1, time, -1, -1, -1)
        current_support = maps[:, :, 4:5]
    else:
        initial_support = maps[:, :1, :2].amax(dim=2, keepdim=True).expand(-1, time, -1, -1, -1)
        current_support = maps[:, :, :2].amax(dim=2, keepdim=True)
    old_only = torch.relu(initial_support - current_support)
    if mask_dilation > 1:
        old_only = functional.max_pool2d(
            old_only.reshape(batch * time, 1, height, width),
            kernel_size=mask_dilation,
            stride=1,
            padding=mask_dilation // 2,
        ).reshape(batch, time, 1, height, width)
    blurred = functional.avg_pool2d(
        anchor.reshape(batch * time, 3, height, width),
        kernel_size=blur_kernel,
        stride=1,
        padding=blur_kernel // 2,
        count_include_pad=False,
    ).reshape(batch, time, 3, height, width)
    return transported * (1.0 - old_only) + blurred * old_only


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("run_dir", type=Path, help="A student-flow run whose data split/config will be reused")
    parser.add_argument("--device", default="cuda:0" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--scales", nargs="+", type=float, default=[-1.0, 0.5, 1.0, 1.5])
    parser.add_argument("--support-powers", nargs="+", type=float, default=[0.0, 0.5, 1.0, 2.0])
    parser.add_argument(
        "--student-mask-root",
        type=Path,
        help="Use dense mask flow from this initial-hand-mask cache instead of Gaussian joint flow.",
    )
    parser.add_argument(
        "--dense-mask-transport",
        action="store_true",
        help="Diagnostic: transport the full mask instead of using it only for disocclusion.",
    )
    args = parser.parse_args()

    checkpoint = torch.load(args.run_dir / "best.pt", map_location="cpu", weights_only=False)
    values = dict(checkpoint["config"])
    values["device"] = args.device
    values.setdefault("reprojection_source_stride", 2)
    values.setdefault("changed_region_weight", 0.0)
    values.setdefault("anchor_preservation_weight", 0.0)
    values.setdefault("student_state_root", None)
    values.setdefault("student_mask_root", None)
    values.setdefault("initialize_from", None)
    values.setdefault("freeze_base", False)
    if args.student_mask_root is not None:
        values["student_mask_root"] = str(args.student_mask_root)
    values["required_test_information"] = tuple(values["required_test_information"])
    values["source_cameras"] = tuple(values["source_cameras"])
    config = TrainConfig(**values)
    seed_everything(config.seed)
    dataset = H2OPhysicalClipDataset(
        config.index,
        split="val",
        frames_per_clip=config.frames_per_clip,
        image_size=config.image_size,
        stats_path=config.stats,
        source_cameras=config.source_cameras,
        max_samples=config.max_val_samples,
        combine_source_cameras=True,
        student_state_root=config.student_state_root,
        student_mask_root=config.student_mask_root,
        student_flow=args.student_mask_root is None,
        student_mask_flow=args.student_mask_root is not None,
        student_dense_mask_flow=args.dense_mask_transport,
    )
    loader = make_loader(dataset, config, train=False)
    device = torch.device(args.device)
    names = ("l1", "mse", "hand_l1", "changed_l1", "changed_gain_over_static", "unchanged_anchor_drift_l1")
    variants = [("anchor", 0.0, 0.0)] + [
        (f"scale={scale},support_power={power}", scale, power)
        for scale in args.scales
        for power in args.support_powers
    ]
    fill_variants = [
        (f"warp_fill,blur={blur},dilation={dilation}", blur, dilation)
        for blur in (5, 9, 15, 21)
        for dilation in (1, 5, 9)
    ]
    variants += [(name, 1.0, 2.0) for name, _, _ in fill_variants]
    totals = {name: {metric: 0.0 for metric in names} for name, _, _ in variants}
    samples = 0
    for batch in loader:
        anchor = batch["ego_anchor"].to(device=device, dtype=torch.float32)
        target = batch["target"].to(device=device, dtype=torch.float32)
        maps = batch["initial_ego_maps"].to(device=device, dtype=torch.float32)
        physical_maps = batch["physical_maps"].to(device=device, dtype=torch.float32)
        count = anchor.shape[0]
        samples += count
        for variant, scale, support_power in variants:
            prediction = anchor if scale == 0 else warp_anchor(anchor, maps, scale, support_power)
            if variant.startswith("warp_fill"):
                _, blur, dilation = next(value for value in fill_variants if value[0] == variant)
                prediction = fill_old_hand_location(anchor, prediction, maps, blur, dilation)
            terms = reconstruction_terms(prediction, target, physical_maps, ego_anchor=anchor)
            for name in names:
                totals[variant][name] += float(terms[name]) * count
    result = {"samples": samples, "variants": {}}
    for variant, _, _ in variants:
        metrics = {name: value / samples for name, value in totals[variant].items()}
        metrics["psnr"] = -10.0 * math.log10(max(metrics["mse"], 1e-12))
        result["variants"][variant] = metrics
    if args.dense_mask_transport:
        filename = "evaluation_deterministic_dense_mask_warp.json"
    elif args.student_mask_root is not None:
        filename = "evaluation_deterministic_mask_warp.json"
    else:
        filename = "evaluation_deterministic_warp.json"
    destination = args.run_dir / filename
    destination.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
