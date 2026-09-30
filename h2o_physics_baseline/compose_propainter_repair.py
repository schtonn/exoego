#!/usr/bin/env python3
"""Hard-constrained ProPainter fusion and real-H2O comparison rendering."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import subprocess

import cv2
import numpy as np
from PIL import Image, ImageDraw, ImageFont


FONT = Path("/usr/share/fonts/opentype/noto/NotoSansCJK-Bold.ttc")
INPUT_BORDER = (22, 163, 74)
INTERMEDIATE_BORDER = (245, 158, 11)
PREDICTION_BORDER = (126, 34, 206)
TARGET_BORDER = (220, 38, 38)
FOREGROUND_COLOR = (217, 70, 239)
OBJECT_COLOR = (14, 165, 233)
DYNAMIC_COMPLETION_COLOR = (20, 184, 166)


def load_rgb(path: Path, size: int) -> np.ndarray:
    return np.asarray(
        Image.open(path).convert("RGB").resize((size, size), Image.Resampling.BILINEAR)
    ).astype(np.float32) / 255.0


def load_mask(path: Path, size: int) -> np.ndarray:
    return np.asarray(
        Image.open(path).convert("L").resize((size, size), Image.Resampling.NEAREST)
    ) > 127


def labelled_tile(
    value: np.ndarray,
    label: str,
    border: tuple[int, int, int],
    size: int,
) -> Image.Image:
    image = Image.fromarray(
        np.rint(np.clip(value, 0, 1) * 255).astype(np.uint8), "RGB"
    )
    tile = Image.new("RGB", (size, size + 44), "white")
    tile.paste(image, (0, 0))
    draw = ImageDraw.Draw(tile)
    draw.rectangle((0, 0, size - 1, size - 1), outline=border, width=5)
    draw.text((8, size + 5), label, font=ImageFont.truetype(str(FONT), 20), fill=(15, 23, 42))
    return tile


def mask_visual(
    unknown: np.ndarray,
    seam: np.ndarray,
    arm: np.ndarray,
    object_mask: np.ndarray,
    dynamic_completion: np.ndarray | None = None,
) -> np.ndarray:
    value = np.zeros((*unknown.shape, 3), dtype=np.float32)
    value[seam] = np.asarray(INTERMEDIATE_BORDER, dtype=np.float32) / 255.0
    value[unknown] = np.asarray(PREDICTION_BORDER, dtype=np.float32) / 255.0
    if dynamic_completion is not None:
        value[dynamic_completion] = (
            np.asarray(DYNAMIC_COMPLETION_COLOR, dtype=np.float32) / 255.0
        )
    value[object_mask] = np.asarray(OBJECT_COLOR, dtype=np.float32) / 255.0
    value[arm] = np.asarray(FOREGROUND_COLOR, dtype=np.float32) / 255.0
    return value


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model-input-root", type=Path, required=True)
    parser.add_argument("--propainter-frames", type=Path, required=True)
    parser.add_argument(
        "--baseline-frames",
        type=Path,
        default=None,
        help=(
            "Optional already-repaired frame directory used as the locked base. "
            "This supports a second, dynamic-foreground-only ProPainter pass."
        ),
    )
    parser.add_argument(
        "--primary-mask-root",
        type=Path,
        default=None,
        help="Mask actually authorized for the primary ProPainter pass (default: repair_masks).",
    )
    parser.add_argument(
        "--seam-propainter-frames",
        type=Path,
        default=None,
        help="Optional ProPainter pass trained only on cleaned known-source seams.",
    )
    parser.add_argument(
        "--known-seam-mask-root",
        type=Path,
        default=None,
        help="Clean known-to-known seam masks paired with --seam-propainter-frames.",
    )
    parser.add_argument("--target-rgb-dir", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--size", type=int, default=256)
    parser.add_argument("--fps", type=int, default=15)
    parser.add_argument("--feather-radius", type=int, default=2)
    args = parser.parse_args()

    manifest = json.loads((args.model_input_root / "manifest.json").read_text())
    args.output_root.mkdir(parents=True, exist_ok=True)
    frame_root = args.output_root / "frames"
    frame_root.mkdir(parents=True, exist_ok=True)
    baseline_frames = []
    repaired_frames = []
    target_frames = []
    region_totals = {
        name: {"baseline_absolute": 0.0, "repaired_absolute": 0.0, "pixels": 0.0}
        for name in (
            "all", "repair", "unknown", "seam", "known_seam", "foreground",
            "dynamic_completion", "locked"
        )
    }
    comparison_frames = []
    for record in manifest["frames"]:
        index = int(record["index"])
        frame = int(record["dataset_frame"])
        filename = f"{index:06d}.png"
        baseline = load_rgb(
            (args.baseline_frames or (args.model_input_root / "input_frames")) / filename,
            args.size,
        )
        proposed = load_rgb(args.propainter_frames / f"{index:04d}.png", args.size)
        repair = load_mask(args.model_input_root / "repair_masks" / filename, args.size)
        primary_mask = load_mask(
            (args.primary_mask_root or (args.model_input_root / "repair_masks")) / filename,
            args.size,
        )
        unknown = load_mask(args.model_input_root / "unknown_masks" / filename, args.size)
        seam = load_mask(args.model_input_root / "seam_masks" / filename, args.size)
        foreground = load_mask(args.model_input_root / "foreground_masks" / filename, args.size)
        arm_path = args.model_input_root / "arm_masks" / filename
        object_path = args.model_input_root / "object_masks" / filename
        arm = load_mask(arm_path, args.size) if arm_path.exists() else foreground
        object_mask = (
            load_mask(object_path, args.size)
            if object_path.exists()
            else np.zeros_like(foreground)
        )
        dynamic_completion_path = (
            args.model_input_root / "dynamic_completion_masks" / filename
        )
        dynamic_completion = (
            load_mask(dynamic_completion_path, args.size)
            if dynamic_completion_path.exists() else np.zeros_like(foreground)
        )
        known_seam = (
            load_mask(args.known_seam_mask_root / filename, args.size)
            if args.known_seam_mask_root is not None
            else seam & (~unknown)
        )
        target = load_rgb(args.target_rgb_dir / f"{frame:06d}.png", args.size)

        kernel_size = args.feather_radius * 2 + 1
        kernel = np.ones((kernel_size, kernel_size), np.uint8)
        authorized = cv2.dilate(primary_mask.astype(np.uint8), kernel) > 0
        alpha = cv2.GaussianBlur(
            primary_mask.astype(np.float32),
            (0, 0),
            sigmaX=max(0.8, args.feather_radius / 2),
        )
        alpha[primary_mask] = 1.0
        alpha[~authorized] = 0.0
        repaired = baseline * (1.0 - alpha[..., None]) + proposed * alpha[..., None]
        if args.seam_propainter_frames is not None:
            seam_proposed = load_rgb(
                args.seam_propainter_frames / f"{index:04d}.png", args.size
            )
            seam_authorized = cv2.dilate(known_seam.astype(np.uint8), kernel) > 0
            seam_alpha = cv2.GaussianBlur(
                known_seam.astype(np.float32),
                (0, 0),
                sigmaX=max(0.8, args.feather_radius / 2),
            )
            seam_alpha[known_seam] = 1.0
            seam_alpha[~seam_authorized] = 0.0
            # The seam-only pass has priority on observed-source boundaries;
            # unknown pixels still come from the general completion pass.
            repaired = (
                repaired * (1.0 - seam_alpha[..., None])
                + seam_proposed * seam_alpha[..., None]
            )
        repaired = np.clip(repaired, 0.0, 1.0)
        Image.fromarray(np.rint(repaired * 255).astype(np.uint8), "RGB").save(
            frame_root / filename
        )

        regions = {
            "all": np.ones_like(repair),
            "repair": repair,
            "unknown": unknown,
            "seam": seam,
            "known_seam": known_seam,
            "foreground": foreground,
            "dynamic_completion": dynamic_completion,
            "locked": ~authorized,
        }
        baseline_error = np.abs(baseline - target).mean(axis=-1)
        repaired_error = np.abs(repaired - target).mean(axis=-1)
        for name, region in regions.items():
            region_totals[name]["baseline_absolute"] += float(baseline_error[region].sum())
            region_totals[name]["repaired_absolute"] += float(repaired_error[region].sum())
            region_totals[name]["pixels"] += float(region.sum())

        tiles = [
            labelled_tile(baseline, "当前分层合成", PREDICTION_BORDER, args.size),
            labelled_tile(
                mask_visual(
                    unknown, known_seam, arm, object_mask, dynamic_completion
                ),
                "橙融合 紫背景生成 青动态补全 粉手臂 蓝持物",
                INTERMEDIATE_BORDER,
                args.size,
            ),
            labelled_tile(repaired, "ProPainter约束修复", PREDICTION_BORDER, args.size),
            labelled_tile(target, "未来 ego 真值", TARGET_BORDER, args.size),
        ]
        gap = 8
        canvas = Image.new(
            "RGB", (args.size * len(tiles) + gap * (len(tiles) - 1), args.size + 44),
            (241, 245, 249),
        )
        for column, tile in enumerate(tiles):
            canvas.paste(tile, (column * (args.size + gap), 0))
        comparison_frames.append(canvas)
        baseline_frames.append(baseline)
        repaired_frames.append(repaired)
        target_frames.append(target)

    metrics = {}
    for name, total in region_totals.items():
        pixels = max(total["pixels"], 1.0)
        baseline_l1 = total["baseline_absolute"] / pixels
        repaired_l1 = total["repaired_absolute"] / pixels
        metrics[name] = {
            "baseline_l1": baseline_l1,
            "repaired_l1": repaired_l1,
            "relative_improvement_percent": 100.0 * (baseline_l1 - repaired_l1) / max(baseline_l1, 1e-8),
            "pixel_fraction": total["pixels"] / (len(manifest["frames"]) * args.size * args.size),
        }
    baseline_stack = np.stack(baseline_frames)
    repaired_stack = np.stack(repaired_frames)
    target_stack = np.stack(target_frames)
    for name, stack in (("baseline", baseline_stack), ("repaired", repaired_stack)):
        temporal_error = np.abs(
            np.diff(stack, axis=0) - np.diff(target_stack, axis=0)
        ).mean()
        metrics[f"{name}_temporal_delta_l1"] = float(temporal_error)
    metrics["locked_output_difference_l1"] = float(
        np.abs(repaired_stack - baseline_stack).mean(axis=-1)[
            np.stack([
                ~(
                    cv2.dilate(
                        load_mask(
                            (args.primary_mask_root or (args.model_input_root / "repair_masks"))
                            / f"{int(r['index']):06d}.png",
                            args.size,
                        ).astype(np.uint8),
                        np.ones((args.feather_radius * 2 + 1,) * 2, np.uint8),
                    ) > 0
                )
                for r in manifest["frames"]
            ])
        ].mean()
    )
    (args.output_root / "metrics.json").write_text(
        json.dumps(metrics, indent=2, ensure_ascii=False), encoding="utf-8"
    )

    width, height = comparison_frames[0].size
    video_path = args.output_root / "comparison.mp4"
    process = subprocess.Popen(
        [
            "ffmpeg", "-y", "-loglevel", "error", "-f", "rawvideo",
            "-pix_fmt", "rgb24", "-s", f"{width}x{height}", "-r", str(args.fps),
            "-i", "-", "-an", "-c:v", "libx264", "-preset", "slow", "-crf", "15",
            "-pix_fmt", "yuv420p", "-movflags", "+faststart", str(video_path),
        ],
        stdin=subprocess.PIPE,
    )
    assert process.stdin is not None
    for frame in comparison_frames:
        process.stdin.write(np.asarray(frame, dtype=np.uint8).tobytes())
    process.stdin.close()
    if process.wait() != 0:
        raise RuntimeError("ffmpeg failed")
    comparison_frames[-1].save(args.output_root / "comparison_poster.png")
    print(video_path)
    print(args.output_root / "metrics.json")


if __name__ == "__main__":
    main()
