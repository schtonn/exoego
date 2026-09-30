#!/usr/bin/env python3
"""Insert repaired prediction frames into the existing 2x5 diagnostic panel."""

from __future__ import annotations

import argparse
from pathlib import Path
import subprocess

import cv2
import numpy as np
from PIL import Image, ImageDraw, ImageFont


FONT = Path("/usr/share/fonts/opentype/noto/NotoSansCJK-Bold.ttc")
PREDICTION_BORDER = (126, 34, 206)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--base-panel", type=Path, required=True)
    parser.add_argument("--replacement-frames", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--tile-size", type=int, default=256)
    parser.add_argument("--label-height", type=int, default=44)
    parser.add_argument("--gap", type=int, default=8)
    parser.add_argument("--column", type=int, default=3)
    parser.add_argument("--row", type=int, default=1)
    parser.add_argument("--label", default="最终约束修复")
    args = parser.parse_args()

    capture = cv2.VideoCapture(str(args.base_panel))
    fps = capture.get(cv2.CAP_PROP_FPS) or 15.0
    width = int(capture.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(capture.get(cv2.CAP_PROP_FRAME_HEIGHT))
    args.output.parent.mkdir(parents=True, exist_ok=True)
    process = subprocess.Popen(
        [
            "ffmpeg", "-y", "-loglevel", "error", "-f", "rawvideo",
            "-pix_fmt", "rgb24", "-s", f"{width}x{height}", "-r", str(fps),
            "-i", "-", "-an", "-c:v", "libx264", "-preset", "slow",
            "-crf", "15", "-pix_fmt", "yuv420p", "-movflags", "+faststart",
            str(args.output),
        ],
        stdin=subprocess.PIPE,
    )
    assert process.stdin is not None
    x = args.column * (args.tile_size + args.gap)
    y = args.row * (args.tile_size + args.label_height + args.gap)
    frame_index = 0
    last = None
    label_font = ImageFont.truetype(str(FONT), 20)
    while True:
        ok, bgr = capture.read()
        if not ok:
            break
        panel = Image.fromarray(cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB), "RGB")
        replacement_path = args.replacement_frames / f"{frame_index:06d}.png"
        replacement = Image.open(replacement_path).convert("RGB").resize(
            (args.tile_size, args.tile_size), Image.Resampling.BILINEAR
        )
        panel.paste(replacement, (x, y))
        draw = ImageDraw.Draw(panel)
        draw.rectangle(
            (x, y, x + args.tile_size - 1, y + args.tile_size - 1),
            outline=PREDICTION_BORDER,
            width=5,
        )
        draw.rectangle(
            (
                x,
                y + args.tile_size,
                x + args.tile_size - 1,
                y + args.tile_size + args.label_height - 1,
            ),
            fill="white",
        )
        draw.text(
            (x + 8, y + args.tile_size + 5),
            args.label,
            font=label_font,
            fill=(15, 23, 42),
        )
        payload = np.asarray(panel, dtype=np.uint8)
        process.stdin.write(payload.tobytes())
        last = panel
        frame_index += 1
    capture.release()
    process.stdin.close()
    if process.wait() != 0:
        raise RuntimeError("ffmpeg failed")
    if last is None:
        raise RuntimeError(f"No frames decoded from {args.base_panel}")
    last.save(args.output.with_name(args.output.stem + "_poster.png"))
    print(args.output)
    print(f"frames={frame_index}")


if __name__ == "__main__":
    main()
