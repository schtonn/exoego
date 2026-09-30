#!/usr/bin/env python3
"""Append data-scale comparisons to the established 4x6 process video format."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
import subprocess
import sys

import cv2
import numpy as np
from PIL import Image, ImageDraw, ImageFont
import torch
from torch.utils.data import DataLoader

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from h2o_physics_baseline.final_layer_fusion import (
    AuthorizedFinalFusion,
    FinalLayerWindowDataset,
)
from h2o_physics_baseline.render_causal_video_background_split import (
    CURRENT_EXO_COLOR,
    FOREGROUND_COLOR,
    GENERATED_COLOR,
    HISTORY_EXO_COLOR,
    OBJECT_COLOR,
)


INPUT_BORDER = (22, 163, 74)
INTERMEDIATE_BORDER = (245, 158, 11)
PREDICTION_BORDER = (126, 34, 206)
TARGET_BORDER = (220, 38, 38)
FONT_PATH = Path("/usr/share/fonts/opentype/noto/NotoSansCJK-Bold.ttc")


def tile(image: Image.Image, label: str, border: tuple[int, int, int], size: int) -> Image.Image:
    value = Image.new("RGB", (size, size + 44), "white")
    value.paste(image.convert("RGB").resize((size, size), Image.Resampling.BILINEAR), (0, 0))
    draw = ImageDraw.Draw(value)
    draw.rectangle((0, 0, size - 1, size - 1), outline=border, width=5)
    draw.text((8, size + 5), label, font=ImageFont.truetype(str(FONT_PATH), 20), fill=(15, 23, 42))
    return value


def tensor_image(value: torch.Tensor) -> Image.Image:
    array = value.detach().cpu().permute(1, 2, 0).numpy()
    return Image.fromarray(np.rint(np.clip(array, 0, 1) * 255).astype(np.uint8), "RGB")


def load_rgb(path: Path, size: int) -> np.ndarray:
    return np.asarray(
        Image.open(path).convert("RGB").resize((size, size), Image.Resampling.BILINEAR),
        dtype=np.float32,
    ) / 255.0


def load_mask(path: Path, size: int) -> np.ndarray:
    return np.asarray(
        Image.open(path).convert("L").resize((size, size), Image.Resampling.NEAREST)
    ) > 127


def rgb_image(value: np.ndarray) -> Image.Image:
    return Image.fromarray(
        np.rint(np.clip(value, 0, 1) * 255).astype(np.uint8), "RGB"
    )


def depth_panel(path: Path, size: int) -> Image.Image:
    depth = np.asarray(
        Image.open(path).resize((size, size), Image.Resampling.NEAREST),
        dtype=np.float32,
    )
    valid = depth > 0
    value = np.zeros((size, size), dtype=np.uint8)
    if np.any(valid):
        low, high = np.percentile(depth[valid], (2, 98))
        value[valid] = np.rint(
            255.0 * np.clip((depth[valid] - low) / max(high - low, 1.0), 0, 1)
        ).astype(np.uint8)
    colored = cv2.cvtColor(
        cv2.applyColorMap(value, cv2.COLORMAP_TURBO), cv2.COLOR_BGR2RGB
    )
    colored[~valid] = 0
    return Image.fromarray(colored, "RGB")


def provenance_image(
    labels: np.ndarray,
    arm_mask: np.ndarray,
    object_mask: np.ndarray,
) -> Image.Image:
    canvas = np.zeros((*labels.shape, 3), dtype=np.uint8)
    for value, color in (
        (1, INPUT_BORDER),
        (2, CURRENT_EXO_COLOR),
        (3, HISTORY_EXO_COLOR),
        (4, GENERATED_COLOR),
    ):
        canvas[labels == value] = color
    canvas[object_mask] = OBJECT_COLOR
    canvas[arm_mask] = FOREGROUND_COLOR
    return Image.fromarray(canvas, "RGB")


def isolated_stream(rgb: np.ndarray, valid: np.ndarray) -> Image.Image:
    value = np.zeros_like(rgb)
    value[valid] = rgb[valid]
    return rgb_image(value)


def difference_panel(first: np.ndarray, second: np.ndarray) -> Image.Image:
    difference = np.abs(first - second).mean(axis=2)
    scale = max(float(np.percentile(difference, 98)), 1e-3)
    value = np.rint(255.0 * np.clip(difference / scale, 0, 1)).astype(np.uint8)
    colored = cv2.cvtColor(
        cv2.applyColorMap(value, cv2.COLORMAP_TURBO), cv2.COLOR_BGR2RGB
    )
    return Image.fromarray(colored, "RGB")


def load_model(path: Path, device: torch.device) -> AuthorizedFinalFusion:
    checkpoint = torch.load(path, map_location="cpu", weights_only=False)
    config = checkpoint["config"]
    model = AuthorizedFinalFusion(int(config["width"]), int(config["blocks"]))
    model.load_state_dict(checkpoint["model"])
    return model.to(device).eval()


@torch.no_grad()
def infer_clip_models(
    root: Path,
    checkpoints: list[Path],
    index: Path,
    device: torch.device,
    batch_size: int = 4,
    require_full_protocol: bool = True,
) -> list[list[torch.Tensor]]:
    """Read every layered window once and evaluate all checkpoints on it.

    The earlier implementation rebuilt the dataset and decoded the same 33
    input channels once per checkpoint.  Sharing each batch keeps comparison
    inference identical while removing the dominant repeated image I/O.
    """
    dataset = FinalLayerWindowDataset(
        [root], index, window=5, stride=1,
        require_full_protocol=require_full_protocol,
    )
    frame_count = len(dataset.clips[0].dataset_frames)
    models = [load_model(checkpoint, device) for checkpoint in checkpoints]
    accumulators: list[list[torch.Tensor | None]] = [
        [None] * frame_count for _ in models
    ]
    counts = np.zeros(frame_count, dtype=np.int32)
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=False, num_workers=0)
    sample_cursor = 0
    for sample in loader:
        inputs = sample["input"].to(device)
        base = sample["base"].to(device)
        proposal = sample["proposal"].to(device)
        authorized = sample["authorized"].to(device)
        outputs = [
            model(inputs, base, proposal, authorized)[0].cpu()
            for model in models
        ]
        current_batch = outputs[0].shape[0]
        for batch_index in range(current_batch):
            _, start = dataset.samples[sample_cursor + batch_index]
            for local_index in range(outputs[0].shape[2]):
                frame_index = start + local_index
                for model_index, output in enumerate(outputs):
                    value = output[batch_index, :, local_index]
                    previous = accumulators[model_index][frame_index]
                    accumulators[model_index][frame_index] = (
                        value.clone() if previous is None else previous + value
                    )
                counts[frame_index] += 1
        sample_cursor += current_batch
    return [
        [
            value / int(counts[frame_index])
            for frame_index, value in enumerate(model_accumulator)
            if value is not None
        ]
        for model_accumulator in accumulators
    ]


def infer_clip(
    root: Path,
    checkpoint: Path,
    index: Path,
    device: torch.device,
) -> list[torch.Tensor]:
    """Compatibility wrapper for a single checkpoint."""
    return infer_clip_models(root, [checkpoint], index, device)[0]


def render_clip(
    root: Path,
    output: Path,
    checkpoints: list[Path],
    index_path: Path,
    device: torch.device,
    size: int,
    fps: int,
) -> None:
    manifest = json.loads((root / "model_input/manifest.json").read_text())
    pair_id = manifest["pair_id"]
    with index_path.open(encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))
    row = next(value for value in rows if value["pair_id"] == pair_id)
    clip_rows = [value for value in rows if value["clip_id"] == row["clip_id"]]
    sources = {
        value["source_camera"]: Path(value["source_rgb_dir"]) for value in clip_rows
    }
    predictions = infer_clip_models(root, checkpoints, index_path, device)
    first_frame = int(manifest["frames"][0]["dataset_frame"])
    target_rgb_root = Path(row["target_rgb_dir"])
    target_camera_root = target_rgb_root.parent
    ego_first = load_rgb(target_rgb_root / f"{first_frame:06d}.png", size)
    scene_tokens = pair_id.split("_")
    scene_label = "_".join(scene_tokens[1:3]) if len(scene_tokens) >= 3 else pair_id
    gap = 8
    tile_height = size + 44
    width = 6 * size + 5 * gap
    height = 4 * tile_height + 3 * gap
    output.parent.mkdir(parents=True, exist_ok=True)
    process = subprocess.Popen(
        [
            "ffmpeg", "-y", "-loglevel", "error", "-f", "rawvideo",
            "-pix_fmt", "rgb24", "-s", f"{width}x{height}", "-r", str(fps),
            "-i", "-", "-an", "-c:v", "libx264", "-preset", "slow",
            "-crf", "15", "-pix_fmt", "yuv420p", "-movflags", "+faststart",
            str(output),
        ],
        stdin=subprocess.PIPE,
    )
    assert process.stdin is not None
    last_canvas: Image.Image | None = None
    for record in manifest["frames"]:
        frame_index = int(record["index"])
        dataset_frame = int(record["dataset_frame"])
        filename = f"{frame_index:06d}.png"
        model_input = root / "model_input"
        labels = np.asarray(Image.open(model_input / "provenance_labels" / filename))
        arm_mask = load_mask(model_input / "arm_masks" / filename, size)
        object_mask = load_mask(model_input / "object_masks" / filename, size) & (~arm_mask)
        background_priority = ~(arm_mask | object_mask)
        static_mask = (labels == 1) & background_priority
        current_mask = (labels == 2) & background_priority
        history_mask = (labels == 3) & background_priority
        generated_mask = (labels == 4) & background_priority

        geometry = load_rgb(model_input / "geometry_frames" / filename, size)
        generated_background = load_rgb(model_input / "background_frames" / filename, size)
        layered = load_rgb(model_input / "input_frames" / filename, size)
        current = load_rgb(root / "propainter_composed/frames" / filename, size)
        raw_propainter = load_rgb(
            root / "propainter_unknown/input_frames/frames" / f"{frame_index:04d}.png",
            size,
        )
        anchor_rgb = (
            layered
            if frame_index == 0
            else load_rgb(model_input / "anchor_layer_frames" / filename, size)
        )
        exo_rgb = load_rgb(model_input / "exo_layer_frames" / filename, size)
        object_rgb = load_rgb(model_input / "object_layer_frames" / filename, size)
        prediction_images = [tensor_image(value[frame_index]) for value in predictions]
        temporal = np.asarray(prediction_images[2], dtype=np.float32) / 255.0
        target = load_rgb(target_rgb_root / f"{dataset_frame:06d}.png", size)
        panels = [
            tile(
                Image.open(sources[f"cam{camera}"] / f"{dataset_frame:06d}.png"),
                f"{scene_label} · 输入 cam{camera} · {dataset_frame:06d}" if camera == 0
                else f"输入 cam{camera} · {dataset_frame:06d}",
                INPUT_BORDER,
                size,
            )
            for camera in range(4)
        ] + [
            tile(rgb_image(ego_first), f"输入 ego 首帧 · {first_frame:06d}", INPUT_BORDER, size),
            tile(depth_panel(target_camera_root / "depth" / f"{first_frame:06d}.png", size), "ego首帧深度（仅首帧协议）", INPUT_BORDER, size),
            tile(rgb_image(geometry), "几何观测（黑=未知）", INTERMEDIATE_BORDER, size),
            tile(rgb_image(generated_background), "相机运动约束背景", INTERMEDIATE_BORDER, size),
            tile(provenance_image(labels, arm_mask, object_mask), "信息来源蒙版", INTERMEDIATE_BORDER, size),
            tile(rgb_image(layered), "ProPainter前分层合成", PREDICTION_BORDER, size),
            tile(rgb_image(raw_propainter), "ProPainter原始提议", PREDICTION_BORDER, size),
            tile(difference_panel(layered, current), "ProPainter授权修改量", INTERMEDIATE_BORDER, size),
            tile(isolated_stream(anchor_rgb, static_mask), "首帧几何流", INPUT_BORDER, size),
            tile(isolated_stream(exo_rgb, current_mask), "当前exo流", CURRENT_EXO_COLOR, size),
            tile(isolated_stream(exo_rgb, history_mask), "历史exo流", HISTORY_EXO_COLOR, size),
            tile(isolated_stream(generated_background, generated_mask), "无观测生成流", GENERATED_COLOR, size),
            tile(isolated_stream(layered, arm_mask), "手/臂动态流", FOREGROUND_COLOR, size),
            tile(isolated_stream(object_rgb, object_mask), "物体流", OBJECT_COLOR, size),
            tile(rgb_image(current), "当前最终", PREDICTION_BORDER, size),
            tile(prediction_images[0], "2段训练", PREDICTION_BORDER, size),
            tile(prediction_images[1], "8段训练", PREDICTION_BORDER, size),
            tile(prediction_images[2], "8段时序约束", PREDICTION_BORDER, size),
            tile(difference_panel(current, temporal), "时序约束修改量", INTERMEDIATE_BORDER, size),
            tile(rgb_image(target), "未来 ego 真值", TARGET_BORDER, size),
        ]
        canvas = Image.new("RGB", (width, height), (241, 245, 249))
        for panel_index, panel in enumerate(panels):
            row_index, column_index = divmod(panel_index, 6)
            canvas.paste(panel, (column_index * (size + gap), row_index * (tile_height + gap)))
        process.stdin.write(np.asarray(canvas, dtype=np.uint8).tobytes())
        last_canvas = canvas
    process.stdin.close()
    if process.wait() != 0:
        raise RuntimeError(f"ffmpeg failed for {output}")
    if last_canvas is None:
        raise RuntimeError(f"No frames rendered for {output}")
    last_canvas.save(output.with_name(output.stem + "_poster.png"))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--clip-root", type=Path, action="append", required=True)
    parser.add_argument("--data2-checkpoint", type=Path, required=True)
    parser.add_argument("--data8-checkpoint", type=Path, required=True)
    parser.add_argument("--temporal-checkpoint", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--index", type=Path, default=Path("datasets/H2O/oracle_state/paired_physical_clips.csv"))
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--size", type=int, default=256)
    parser.add_argument("--fps", type=int, default=15)
    args = parser.parse_args()
    device = torch.device(args.device if args.device != "cuda" or torch.cuda.is_available() else "cpu")
    checkpoints = [args.data2_checkpoint, args.data8_checkpoint, args.temporal_checkpoint]
    for root in args.clip_root:
        pair_id = json.loads((root / "model_input/manifest.json").read_text())["pair_id"]
        render_clip(
            root, args.output_root / f"{pair_id}_data_scale.mp4", checkpoints,
            args.index, device, args.size, args.fps,
        )


if __name__ == "__main__":
    main()
