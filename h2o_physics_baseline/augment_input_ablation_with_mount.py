#!/usr/bin/env python3
"""Append a head-camera mount ablation row to an existing 4x6 scene video."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import subprocess
import sys

import cv2
import numpy as np
from PIL import Image

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from h2o_physics_baseline.render_causal_video_background_split import pil_rgb, title_tile
from h2o_physics_baseline.render_input_ablation_comparison import difference_panel
from h2o_physics_baseline.render_object_warp_comparison import load_rgb, provenance_image
from h2o_physics_baseline.visualize_student_warp_fill import (
    INTERMEDIATE_BORDER,
    PREDICTION_BORDER,
)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--base-video", type=Path, required=True)
    parser.add_argument("--known-output-root", type=Path, required=True)
    parser.add_argument("--no-mount-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--fps", type=int, default=15)
    args = parser.parse_args()

    manifest = json.loads((args.no_mount_root / "manifest.json").read_text())
    size = int(manifest["image_size"])
    metrics = json.loads(
        (args.no_mount_root / "propainter_composed" / "metrics.json").read_text()
    )
    gap, tile_height, columns = 8, size + 44, 6
    capture = cv2.VideoCapture(str(args.base_video))
    width = int(capture.get(cv2.CAP_PROP_FRAME_WIDTH))
    base_height = int(capture.get(cv2.CAP_PROP_FRAME_HEIGHT))
    expected_width = columns * size + (columns - 1) * gap
    if width != expected_width:
        raise ValueError(f"Unexpected base-video width {width}, expected {expected_width}")
    output_height = base_height + gap + tile_height
    args.output.parent.mkdir(parents=True, exist_ok=True)
    process = subprocess.Popen(
        [
            "ffmpeg", "-y", "-loglevel", "error", "-f", "rawvideo",
            "-pix_fmt", "rgb24", "-s", f"{width}x{output_height}",
            "-r", str(args.fps), "-i", "-", "-an", "-c:v", "libx264",
            "-preset", "slow", "-crf", "15", "-pix_fmt", "yuv420p",
            "-movflags", "+faststart", str(args.output),
        ],
        stdin=subprocess.PIPE,
    )
    assert process.stdin is not None
    final_canvas = None
    records = manifest["frames"]
    for record in records:
        ok, base_bgr = capture.read()
        if not ok:
            raise RuntimeError("Base video ended before the mount-ablation frames")
        base = Image.fromarray(cv2.cvtColor(base_bgr, cv2.COLOR_BGR2RGB), "RGB")
        index = int(record["index"])
        name = f"{index:06d}.png"
        known = load_rgb(args.known_output_root / name, size)
        baseline = load_rgb(args.no_mount_root / "input_frames" / name, size)
        raw_propainter = load_rgb(
            args.no_mount_root / "propainter_unknown" / "input_frames" / "frames"
            / f"{index:04d}.png",
            size,
        )
        repaired = load_rgb(
            args.no_mount_root / "propainter_composed" / "frames" / name, size
        )
        panels = [
            title_tile(pil_rgb(known, size), "已知首帧头-相机关系", size, PREDICTION_BORDER),
            title_tile(pil_rgb(baseline, size), "无关系几何合成", size, PREDICTION_BORDER),
            title_tile(
                provenance_image(args.no_mount_root, name, size),
                "无关系信息来源蒙版", size, INTERMEDIATE_BORDER,
            ),
            title_tile(pil_rgb(raw_propainter, size), "无关系ProPainter提议", size, PREDICTION_BORDER),
            title_tile(
                pil_rgb(repaired, size),
                f"无关系最终输出 · L1={metrics['all']['repaired_l1']:.3f}",
                size,
                PREDICTION_BORDER,
            ),
            title_tile(
                difference_panel(known, repaired, size),
                "隐藏关系造成的输出差异", size, INTERMEDIATE_BORDER,
            ),
        ]
        row = Image.new("RGB", (width, tile_height), (241, 245, 249))
        for column, panel in enumerate(panels):
            row.paste(panel, (column * (size + gap), 0))
        canvas = Image.new("RGB", (width, output_height), (241, 245, 249))
        canvas.paste(base, (0, 0))
        canvas.paste(row, (0, base_height + gap))
        process.stdin.write(np.asarray(canvas, dtype=np.uint8).tobytes())
        final_canvas = canvas
    capture.release()
    process.stdin.close()
    if process.wait() != 0:
        raise RuntimeError("ffmpeg failed")
    assert final_canvas is not None
    final_canvas.save(args.output.with_name(args.output.stem + "_poster.png"))
    print(args.output)


if __name__ == "__main__":
    main()
