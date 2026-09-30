#!/usr/bin/env python3
"""Render a real-data 2x5 comparison for confidence-weighted object transport."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import subprocess
import sys

import numpy as np
from PIL import Image

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from h2o_physics_baseline.render_causal_video_background_split import (
    CURRENT_EXO_COLOR,
    FOREGROUND_COLOR,
    GENERATED_COLOR,
    HISTORY_EXO_COLOR,
    OBJECT_COLOR,
    pil_rgb,
    title_tile,
)
from h2o_physics_baseline.visualize_student_warp_fill import (
    INPUT_BORDER,
    INTERMEDIATE_BORDER,
    PREDICTION_BORDER,
    TARGET_BORDER,
)


def load_rgb(path: Path, size: int) -> np.ndarray:
    return np.asarray(
        Image.open(path).convert("RGB").resize((size, size), Image.Resampling.BILINEAR),
        dtype=np.float32,
    ) / 255.0


def load_mask(path: Path, size: int) -> np.ndarray:
    return np.asarray(
        Image.open(path).convert("L").resize((size, size), Image.Resampling.NEAREST)
    ) > 127


def provenance_image(root: Path, name: str, size: int) -> Image.Image:
    labels = np.asarray(
        Image.open(root / "provenance_labels" / name).resize(
            (size, size), Image.Resampling.NEAREST
        )
    )
    canvas = np.zeros((size, size, 3), dtype=np.uint8)
    for value, color in (
        (1, INPUT_BORDER), (2, CURRENT_EXO_COLOR),
        (3, HISTORY_EXO_COLOR), (4, GENERATED_COLOR),
    ):
        canvas[labels == value] = color
    object_mask = load_mask(root / "object_masks" / name, size)
    arm_mask = load_mask(root / "arm_masks" / name, size)
    canvas[object_mask] = OBJECT_COLOR
    canvas[arm_mask] = FOREGROUND_COLOR
    return Image.fromarray(canvas, "RGB")


def source_occlusion_mask_image(root: Path, size: int) -> Image.Image:
    """Show forbidden source pixels without changing the ego input image."""
    path = root / "anchor_source_hand_exclusion.png"
    value = np.zeros((size, size, 3), dtype=np.uint8)
    if not path.exists():
        return Image.fromarray(value, "RGB")
    mask = load_mask(path, size)
    value[mask] = FOREGROUND_COLOR
    return Image.fromarray(value, "RGB")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--sequence-root", type=Path, required=True)
    parser.add_argument("--base-root", type=Path, required=True)
    parser.add_argument("--warp-root", type=Path, required=True)
    parser.add_argument("--alpha", type=float, default=0.25)
    parser.add_argument("--scene-label", default="")
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    manifest = json.loads((args.warp_root / "manifest.json").read_text())
    size = int(manifest["image_size"])
    first_frame = int(manifest["frames"][0]["dataset_frame"])
    initial_ego = load_rgb(args.sequence_root / "cam4/rgb" / f"{first_frame:06d}.png", size)
    output_frames = args.output.with_suffix("").with_name(args.output.stem + "_frames")
    output_frames.mkdir(parents=True, exist_ok=True)
    gap, tile_height = 8, size + 44
    canvas_width = 6 * size + 5 * gap
    canvas_height = 2 * tile_height + gap
    frames = []
    for record in manifest["frames"]:
        index, frame = int(record["index"]), int(record["dataset_frame"])
        name = f"{index:06d}.png"
        exo = [
            load_rgb(args.sequence_root / f"cam{camera}/rgb/{frame:06d}.png", size)
            for camera in range(4)
        ]
        base = load_rgb(args.base_root / "input_frames" / name, size)
        warp = load_rgb(args.warp_root / "input_frames" / name, size)
        target = load_rgb(args.sequence_root / f"cam4/rgb/{frame:06d}.png", size)
        object_mask = load_mask(args.warp_root / "object_masks" / name, size)
        # Keep the repaired background identical in all comparison columns;
        # only the object pixels may differ.  The earlier visualization loaded
        # the whole pre-repair warp frame here, confounding object appearance
        # with unrelated background completion differences.
        hard = base.copy()
        hard[object_mask] = warp[object_mask]
        object_layer = np.zeros_like(warp)
        object_layer[object_mask] = warp[object_mask]
        soft = base.copy()
        soft[object_mask] = (
            (1.0 - args.alpha) * base[object_mask] + args.alpha * hard[object_mask]
        )
        Image.fromarray(np.rint(soft * 255).astype(np.uint8), "RGB").save(
            output_frames / name
        )
        panels = [
            *[
                title_tile(
                    pil_rgb(value, size),
                    (
                        f"{args.scene_label} · 输入 cam{camera} · {frame:06d}"
                        if args.scene_label and camera == 0
                        else f"输入 cam{camera} · {frame:06d}"
                    ),
                    size,
                    INPUT_BORDER,
                )
                for camera, value in enumerate(exo)
            ],
            title_tile(
                pil_rgb(initial_ego, size),
                f"输入 ego 首帧 · {first_frame:06d}",
                size,
                INPUT_BORDER,
            ),
            title_tile(
                source_occlusion_mask_image(args.warp_root, size),
                "首帧手遮挡蒙版",
                size,
                INTERMEDIATE_BORDER,
            ),
            title_tile(
                pil_rgb(base, size),
                f"{args.scene_label} · 原稳定合成" if args.scene_label else "原稳定合成",
                size,
                PREDICTION_BORDER,
            ),
            title_tile(
                provenance_image(args.warp_root, name, size),
                "蓝色=物体刚体输运", size, INTERMEDIATE_BORDER,
            ),
            title_tile(
                pil_rgb(object_layer, size),
                "过滤后对象层",
                size,
                INTERMEDIATE_BORDER,
            ),
            title_tile(pil_rgb(hard, size), "100%刚体输运", size, PREDICTION_BORDER),
            title_tile(pil_rgb(soft, size), f"{int(args.alpha * 100)}%置信融合", size, PREDICTION_BORDER),
            title_tile(pil_rgb(target, size), "未来 ego 真值", size, TARGET_BORDER),
        ]
        canvas = Image.new("RGB", (canvas_width, canvas_height), (241, 245, 249))
        for panel_index, panel in enumerate(panels):
            row, column = divmod(panel_index, 6)
            canvas.paste(panel, (column * (size + gap), row * (tile_height + gap)))
        frames.append(canvas)

    args.output.parent.mkdir(parents=True, exist_ok=True)
    frames[-1].save(args.output.with_name(args.output.stem + "_poster.png"))
    command = [
        "ffmpeg", "-y", "-loglevel", "error", "-f", "rawvideo", "-pix_fmt", "rgb24",
        "-s", f"{canvas_width}x{canvas_height}", "-r", "15", "-i", "-", "-an",
        "-c:v", "libx264", "-preset", "slow", "-crf", "15", "-pix_fmt", "yuv420p",
        "-movflags", "+faststart", str(args.output),
    ]
    process = subprocess.Popen(command, stdin=subprocess.PIPE)
    assert process.stdin is not None
    for frame in frames:
        process.stdin.write(np.asarray(frame, dtype=np.uint8).tobytes())
    process.stdin.close()
    if process.wait() != 0:
        raise RuntimeError("ffmpeg failed")
    print(args.output)
    print(args.output.with_name(args.output.stem + "_poster.png"))


if __name__ == "__main__":
    main()
