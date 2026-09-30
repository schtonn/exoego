#!/usr/bin/env python3
"""Evaluate final-fusion checkpoints on complete held-out layered clips."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

import numpy as np
from PIL import Image
import torch

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from h2o_physics_baseline.final_layer_fusion import discover_clip
from h2o_physics_baseline.render_final_data_scale_comparison import infer_clip_models


def rgb(path: Path, size: tuple[int, int] = (256, 256)) -> torch.Tensor:
    image = Image.open(path).convert("RGB").resize(size, Image.Resampling.BILINEAR)
    return torch.from_numpy(np.asarray(image, dtype=np.float32) / 255.0).permute(2, 0, 1)


def mask(path: Path) -> torch.Tensor:
    return torch.from_numpy(np.asarray(Image.open(path).convert("L")) > 127)[None]


def clip_tensors(root: Path, index: Path) -> dict[str, torch.Tensor | str]:
    clip = discover_clip(root, index)
    manifest = json.loads((root / "model_input/manifest.json").read_text())
    base, proposal, target, authorization = [], [], [], []
    for record in manifest["frames"]:
        i = int(record["index"])
        name = f"{i:06d}.png"
        base.append(rgb(root / "model_input/input_frames" / name))
        proposal.append(rgb(root / "propainter_composed/frames" / name))
        target.append(rgb(clip.target_rgb_dir / f"{int(record['dataset_frame']):06d}.png"))
        repair = mask(root / "model_input/repair_masks" / name)
        foreground = mask(root / "model_input/foreground_masks" / name)
        physical_object = mask(root / "model_input/object_masks" / name)
        authorization.append(repair & (~foreground) & (~physical_object))
    return {
        "pair_id": clip.pair_id,
        "base": torch.stack(base),
        "current": torch.stack(proposal),
        "target": torch.stack(target),
        "authorized": torch.stack(authorization),
    }


def metrics(output: torch.Tensor, data: dict[str, torch.Tensor | str]) -> dict[str, float]:
    target = data["target"]
    current = data["current"]
    authorized = data["authorized"].float()
    assert isinstance(target, torch.Tensor) and isinstance(current, torch.Tensor)
    assert isinstance(authorized, torch.Tensor)
    absolute = (output - target).abs()
    denominator = (authorized.sum() * 3).clamp_min(1)
    delta_output = output[1:] - output[:-1]
    delta_target = target[1:] - target[:-1]
    accel_output = delta_output[1:] - delta_output[:-1]
    accel_target = delta_target[1:] - delta_target[:-1]
    return {
        "l1": float(absolute.mean()),
        "authorized_l1": float((absolute * authorized).sum() / denominator),
        "delta_l1": float((delta_output - delta_target).abs().mean()),
        "acceleration_l1": float((accel_output - accel_target).abs().mean()),
        "locked_max_change": float(((output - current).abs() * (1 - authorized)).max()),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--clip-root", type=Path, action="append", required=True)
    parser.add_argument("--checkpoint", action="append", required=True, help="name=checkpoint.pt")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--index", type=Path, default=Path("datasets/H2O/oracle_state/paired_physical_clips.csv"))
    parser.add_argument("--device", default="cuda")
    args = parser.parse_args()
    checkpoints = {}
    for specification in args.checkpoint:
        name, path = specification.split("=", 1)
        checkpoints[name] = Path(path)
    device = torch.device(args.device if args.device != "cuda" or torch.cuda.is_available() else "cpu")
    per_clip = []
    for root in args.clip_root:
        data = clip_tensors(root, args.index)
        record = {"pair_id": data["pair_id"], "models": {}}
        record["models"]["layered"] = metrics(data["base"], data)
        record["models"]["current"] = metrics(data["current"], data)
        predictions = infer_clip_models(
            root, list(checkpoints.values()), args.index, device
        )
        for (name, _), frames in zip(checkpoints.items(), predictions):
            prediction = torch.stack(frames)
            record["models"][name] = metrics(prediction, data)
        per_clip.append(record)
        print(json.dumps(record, ensure_ascii=False), flush=True)
    model_names = list(per_clip[0]["models"])
    aggregate = {
        name: {
            metric: float(np.mean([clip["models"][name][metric] for clip in per_clip]))
            for metric in per_clip[0]["models"][name]
        }
        for name in model_names
    }
    result = {"clips": per_clip, "equal_clip_average": aggregate}
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2, ensure_ascii=False), encoding="utf-8")
    print(json.dumps({"equal_clip_average": aggregate}, ensure_ascii=False), flush=True)


if __name__ == "__main__":
    main()
