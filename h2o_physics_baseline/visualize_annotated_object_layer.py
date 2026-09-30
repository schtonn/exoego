#!/usr/bin/env python3
"""Visualize the annotated-exo H2O object-layer upper bound on real frames."""

from __future__ import annotations

import argparse
from pathlib import Path
import sys

import numpy as np
from PIL import Image, ImageDraw, ImageFont

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from h2o_geometric_baseline.reprojection import load_intrinsics, load_pose
from h2o_physics_baseline.annotated_object_layer import annotated_exo_object_layer


FONT = Path("/usr/share/fonts/opentype/noto/NotoSansCJK-Bold.ttc")


def tile(image: np.ndarray, label: str, border: tuple[int, int, int], size: int) -> Image.Image:
    value = Image.fromarray(np.rint(np.clip(image, 0, 1) * 255).astype(np.uint8), "RGB")
    panel = Image.new("RGB", (size, size + 42), "white")
    panel.paste(value, (0, 0))
    draw = ImageDraw.Draw(panel)
    draw.rectangle((0, 0, size - 1, size - 1), outline=border, width=5)
    draw.text((8, size + 4), label, font=ImageFont.truetype(str(FONT), 18), fill=(15, 23, 42))
    return panel


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--sequence", required=True)
    parser.add_argument("--frame", type=int, required=True)
    parser.add_argument("--target-pose", type=Path, default=None)
    parser.add_argument("--size", type=int, default=256)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    root = Path("datasets/H2O/raw") / args.sequence
    sources = [root / f"cam{index}" for index in range(4)]
    target = root / "cam4"
    pose = (
        load_pose(args.target_pose)
        if args.target_pose is not None
        else load_pose(target / "cam_pose" / f"{args.frame:06d}.txt")
    )
    layer = annotated_exo_object_layer(
        sources,
        args.frame,
        load_intrinsics(target / "cam_intrinsics.txt"),
        pose,
        Path("datasets/H2O/oracle_state"),
        output_size=args.size,
    )
    target_rgb = np.asarray(
        Image.open(target / "rgb" / f"{args.frame:06d}.png").convert("RGB").resize(
            (args.size, args.size), Image.Resampling.BILINEAR
        )
    ).astype(np.float32) / 255.0
    support_rgb = np.zeros_like(target_rgb)
    support_rgb[layer.support] = np.asarray((14, 165, 233)) / 255.0
    valid_rgb = np.zeros_like(target_rgb)
    valid_rgb[layer.valid] = np.asarray((22, 163, 74)) / 255.0
    overlay = target_rgb.copy()
    overlay[layer.valid] = 0.25 * target_rgb[layer.valid] + 0.75 * layer.rgb[layer.valid]
    images = [
        tile(support_rgb, f"CAD投影 · {layer.object_name}", (245, 158, 11), args.size),
        tile(valid_rgb, "四路RGB-D有效物体点", (245, 158, 11), args.size),
        tile(layer.rgb, "exo物体外观层", (126, 34, 206), args.size),
        tile(overlay, "叠加到ego真值检查", (126, 34, 206), args.size),
        tile(target_rgb, "ego真值（仅评测）", (220, 38, 38), args.size),
    ]
    canvas = Image.new("RGB", (len(images) * args.size + 8 * (len(images) - 1), args.size + 42), (241, 245, 249))
    for index, panel in enumerate(images):
        canvas.paste(panel, (index * (args.size + 8), 0))
    args.output.parent.mkdir(parents=True, exist_ok=True)
    canvas.save(args.output)
    print(args.output)
    print(
        {
            "object": layer.object_name,
            "support_fraction": float(layer.support.mean()),
            "valid_fraction": float(layer.valid.mean()),
            "coverage_of_support": float(layer.valid.sum() / max(layer.support.sum(), 1)),
            "source_point_count": layer.source_point_count,
        }
    )


if __name__ == "__main__":
    main()
