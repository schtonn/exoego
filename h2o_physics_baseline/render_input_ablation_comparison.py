#!/usr/bin/env python3
"""Render one 4x6 process-and-input-ablation video for a single H2O scene."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import subprocess
import sys

import cv2
import numpy as np
from PIL import Image
import torch

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
from h2o_physics_baseline.render_object_warp_comparison import (
    load_mask,
    load_rgb,
    provenance_image,
)
from h2o_physics_baseline.render_final_data_scale_comparison import infer_clip_models
from h2o_physics_baseline.visualize_student_warp_fill import (
    INPUT_BORDER,
    INTERMEDIATE_BORDER,
    PREDICTION_BORDER,
    TARGET_BORDER,
)


VARIANTS = (
    ("full4_anchor", "4exo+首帧"),
    ("exo4_no_anchor", "4exo无首帧"),
    ("cam0_anchor", "cam0+首帧"),
    ("cam0_no_anchor", "cam0无首帧"),
)
MOUNT_VARIANT = ("exo4_no_anchor_no_mount", "4exo无首帧无位置关系")
DISPLAY_VARIANTS = VARIANTS + (MOUNT_VARIANT,)


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
    colored = cv2.cvtColor(cv2.applyColorMap(value, cv2.COLORMAP_TURBO), cv2.COLOR_BGR2RGB)
    colored[~valid] = 0
    return Image.fromarray(colored, "RGB")


def difference_panel(first: np.ndarray, second: np.ndarray, size: int) -> Image.Image:
    difference = np.abs(first - second).mean(axis=2)
    scale = max(float(np.percentile(difference, 98)), 1e-3)
    value = np.rint(255.0 * np.clip(difference / scale, 0, 1)).astype(np.uint8)
    colored = cv2.cvtColor(cv2.applyColorMap(value, cv2.COLORMAP_TURBO), cv2.COLOR_BGR2RGB)
    return Image.fromarray(colored, "RGB").resize((size, size), Image.Resampling.NEAREST)


def labels(root: Path, name: str, size: int) -> np.ndarray:
    return np.asarray(
        Image.open(root / "provenance_labels" / name).resize(
            (size, size), Image.Resampling.NEAREST
        ),
        dtype=np.uint8,
    )


def isolated_stream(rgb_value: np.ndarray, valid: np.ndarray) -> Image.Image:
    """Show only pixels assigned to one mutually exclusive provenance color."""
    value = np.zeros_like(rgb_value)
    value[valid] = rgb_value[valid]
    return pil_rgb(value, value.shape[0])


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--scene-label", required=True)
    parser.add_argument("--sequence-root", type=Path, required=True)
    parser.add_argument("--ablation-root", type=Path, required=True)
    parser.add_argument(
        "--full-root", type=Path, default=None,
        help="Optional existing full4_anchor root when only reduced-input variants live below ablation-root.",
    )
    parser.add_argument(
        "--no-mount-root", type=Path, required=True,
        help="4exo without ego first frame or supplied head-camera mount.",
    )
    parser.add_argument(
        "--final-checkpoint", type=Path, default=None,
        help="Apply the same standard terminal fusion model to all five input variants.",
    )
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    roots = {name: args.ablation_root / name / "model_input" for name, _ in VARIANTS}
    variant_roots = {name: args.ablation_root / name for name, _ in VARIANTS}
    if args.full_root is not None:
        variant_roots["full4_anchor"] = args.full_root
        roots["full4_anchor"] = args.full_root / "model_input"
    variant_roots[MOUNT_VARIANT[0]] = args.no_mount_root
    wrapped_no_mount = args.no_mount_root / "model_input"
    roots[MOUNT_VARIANT[0]] = (
        wrapped_no_mount
        if (wrapped_no_mount / "manifest.json").exists()
        else args.no_mount_root
    )
    prediction_roots = {
        name: variant_roots[name] / "propainter_composed" / "frames"
        for name, _ in DISPLAY_VARIANTS
    }
    missing_predictions = [
        str(root) for root in prediction_roots.values() if not root.is_dir()
    ]
    if missing_predictions:
        raise FileNotFoundError(
            "Run constrained ProPainter composition first: "
            + ", ".join(missing_predictions)
        )
    manifest = json.loads((roots["full4_anchor"] / "manifest.json").read_text())
    learned_outputs = None
    if args.final_checkpoint is not None:
        device = torch.device(
            args.device if args.device != "cuda" or torch.cuda.is_available() else "cpu"
        )
        checkpoint_path = str(args.final_checkpoint.resolve())
        learned_outputs = {}
        for variant, _ in DISPLAY_VARIANTS:
            frame_root = variant_roots[variant] / "final_fusion" / "frames"
            cache_manifest_path = frame_root.parent / "manifest.json"
            cache_manifest = (
                json.loads(cache_manifest_path.read_text())
                if cache_manifest_path.exists() else {}
            )
            cached = (
                cache_manifest.get("checkpoint") == checkpoint_path
                and cache_manifest.get("frame_count") == len(manifest["frames"])
                and len(list(frame_root.glob("*.png"))) == len(manifest["frames"])
            )
            if cached:
                learned_outputs[variant] = [
                    torch.from_numpy(
                        load_rgb(frame_root / f"{frame_index:06d}.png", int(manifest["image_size"]))
                    ).permute(2, 0, 1)
                    for frame_index in range(len(manifest["frames"]))
                ]
                continue
            frames = infer_clip_models(
                variant_roots[variant], [args.final_checkpoint],
                Path("datasets/H2O/oracle_state/paired_physical_clips.csv"),
                device, require_full_protocol=False,
            )[0]
            learned_outputs[variant] = frames
            frame_root.mkdir(parents=True, exist_ok=True)
            for frame_index, frame in enumerate(frames):
                value = np.rint(
                    np.clip(frame.permute(1, 2, 0).numpy(), 0, 1) * 255
                ).astype(np.uint8)
                Image.fromarray(value, "RGB").save(frame_root / f"{frame_index:06d}.png")
            cache_manifest_path.write_text(
                json.dumps(
                    {
                        "checkpoint": checkpoint_path,
                        "pair_id": manifest["pair_id"],
                        "variant": variant,
                        "frame_count": len(frames),
                        "temporal_loss_is_standard": True,
                    },
                    indent=2,
                    ensure_ascii=False,
                ),
                encoding="utf-8",
            )
    size = int(manifest["image_size"])
    first_frame = int(manifest["frames"][0]["dataset_frame"])
    ego_first = load_rgb(args.sequence_root / f"cam4/rgb/{first_frame:06d}.png", size)
    gap, tile_height = 8, size + 44
    columns, rows = 6, 4
    canvas_width = columns * size + (columns - 1) * gap
    canvas_height = rows * tile_height + (rows - 1) * gap
    rendered = []

    for record in manifest["frames"]:
        index, frame = int(record["index"]), int(record["dataset_frame"])
        name = f"{index:06d}.png"
        exo = [
            load_rgb(args.sequence_root / f"cam{camera}/rgb/{frame:06d}.png", size)
            for camera in range(4)
        ]
        outputs = {
            variant: (
                learned_outputs[variant][index].permute(1, 2, 0).numpy()
                if learned_outputs is not None
                else load_rgb(prediction_roots[variant] / name, size)
            )
            for variant, _ in DISPLAY_VARIANTS
        }
        target = load_rgb(args.sequence_root / f"cam4/rgb/{frame:06d}.png", size)
        full_root = roots["full4_anchor"]
        source_labels = labels(full_root, name, size)
        arm_mask = load_mask(full_root / "arm_masks" / name, size)
        object_mask = load_mask(full_root / "object_masks" / name, size) & (~arm_mask)
        background_priority = ~(arm_mask | object_mask)
        static_mask = (source_labels == 1) & background_priority
        current_mask = (source_labels == 2) & background_priority
        history_mask = (source_labels == 3) & background_priority
        generated_mask = (source_labels == 4) & background_priority
        geometry = load_rgb(full_root / "geometry_frames" / name, size)
        generated_background = load_rgb(full_root / "background_frames" / name, size)
        pre_propainter = load_rgb(full_root / "input_frames" / name, size)
        constrained_propainter = load_rgb(
            prediction_roots["full4_anchor"] / name, size
        )
        raw_propainter = load_rgb(
            variant_roots["full4_anchor"] / "propainter_unknown"
            / "input_frames" / "frames" / f"{index:04d}.png",
            size,
        )
        anchor_rgb = (
            pre_propainter
            if index == 0
            else load_rgb(full_root / "anchor_layer_frames" / name, size)
        )
        exo_rgb = load_rgb(full_root / "exo_layer_frames" / name, size)
        object_rgb = load_rgb(full_root / "object_layer_frames" / name, size)
        panels = [
            *[
                title_tile(
                    pil_rgb(value, size),
                    f"{args.scene_label} · 输入 cam{camera} · {frame:06d}" if camera == 0
                    else f"输入 cam{camera} · {frame:06d}",
                    size,
                    INPUT_BORDER,
                )
                for camera, value in enumerate(exo)
            ],
            title_tile(pil_rgb(ego_first, size), f"输入 ego 首帧 · {first_frame:06d}", size, INPUT_BORDER),
            title_tile(
                depth_panel(args.sequence_root / f"cam4/depth/{first_frame:06d}.png", size),
                "ego首帧深度（仅首帧协议）", size, INPUT_BORDER,
            ),
            title_tile(pil_rgb(geometry, size), "几何观测（黑=未知）", size, INTERMEDIATE_BORDER),
            title_tile(pil_rgb(generated_background, size), "相机运动约束背景", size, INTERMEDIATE_BORDER),
            title_tile(provenance_image(full_root, name, size), "信息来源蒙版", size, INTERMEDIATE_BORDER),
            title_tile(pil_rgb(pre_propainter, size), "ProPainter前分层合成", size, PREDICTION_BORDER),
            title_tile(pil_rgb(raw_propainter, size), "ProPainter原始提议", size, PREDICTION_BORDER),
            title_tile(difference_panel(pre_propainter, constrained_propainter, size), "ProPainter授权修改量", size, INTERMEDIATE_BORDER),
            title_tile(isolated_stream(anchor_rgb, static_mask), "首帧几何流", size, INPUT_BORDER),
            title_tile(isolated_stream(exo_rgb, current_mask), "当前exo流", size, CURRENT_EXO_COLOR),
            title_tile(isolated_stream(exo_rgb, history_mask), "历史exo流", size, HISTORY_EXO_COLOR),
            title_tile(isolated_stream(generated_background, generated_mask), "无观测生成流", size, GENERATED_COLOR),
            title_tile(isolated_stream(pre_propainter, arm_mask), "手/臂动态流", size, FOREGROUND_COLOR),
            title_tile(isolated_stream(object_rgb, object_mask), "物体流", size, OBJECT_COLOR),
        ]
        for variant, label in VARIANTS:
            panels.append(
                title_tile(
                    pil_rgb(outputs[variant], size),
                    label,
                    size,
                    PREDICTION_BORDER,
                )
            )
        panels.extend((
            title_tile(
                pil_rgb(outputs[MOUNT_VARIANT[0]], size),
                MOUNT_VARIANT[1], size, PREDICTION_BORDER,
            ),
            title_tile(pil_rgb(target, size), "未来 ego 真值", size, TARGET_BORDER),
        ))
        canvas = Image.new("RGB", (canvas_width, canvas_height), (241, 245, 249))
        for panel_index, panel in enumerate(panels):
            row, column = divmod(panel_index, columns)
            canvas.paste(panel, (column * (size + gap), row * (tile_height + gap)))
        rendered.append(canvas)

    args.output.parent.mkdir(parents=True, exist_ok=True)
    poster = args.output.with_name(args.output.stem + "_poster.png")
    rendered[-1].save(poster)
    command = [
        "ffmpeg", "-y", "-loglevel", "error", "-f", "rawvideo", "-pix_fmt", "rgb24",
        "-s", f"{canvas_width}x{canvas_height}", "-r", "15", "-i", "-", "-an",
        "-c:v", "libx264", "-preset", "slow", "-crf", "15", "-pix_fmt", "yuv420p",
        "-movflags", "+faststart", str(args.output),
    ]
    process = subprocess.Popen(command, stdin=subprocess.PIPE)
    assert process.stdin is not None
    for frame in rendered:
        process.stdin.write(np.asarray(frame, dtype=np.uint8).tobytes())
    process.stdin.close()
    if process.wait() != 0:
        raise RuntimeError("ffmpeg failed")
    print(args.output)
    print(poster)


if __name__ == "__main__":
    main()
