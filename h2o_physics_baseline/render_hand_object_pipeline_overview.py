#!/usr/bin/env python3
"""Render a dataset-derived overview of the hand/object processing paths."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

import numpy as np
from PIL import Image, ImageDraw

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from h2o_physics_baseline.render_causal_video_background_split import (
    FOREGROUND_COLOR,
    OBJECT_COLOR,
    title_tile,
)
from h2o_physics_baseline.visualize_student_warp_fill import (
    INPUT_BORDER,
    INTERMEDIATE_BORDER,
    PREDICTION_BORDER,
    TARGET_BORDER,
)


def rgb(path: Path, size: int) -> np.ndarray:
    return np.asarray(
        Image.open(path).convert("RGB").resize(
            (size, size), Image.Resampling.BILINEAR
        )
    )


def mask(path: Path, size: int) -> np.ndarray:
    return np.asarray(
        Image.open(path).convert("L").resize(
            (size, size), Image.Resampling.NEAREST
        )
    ) > 0


def isolated(image: np.ndarray, valid: np.ndarray) -> Image.Image:
    output = np.zeros_like(image)
    output[valid] = image[valid]
    return Image.fromarray(output, "RGB")


def color_mask(valid: np.ndarray, color: tuple[int, int, int]) -> Image.Image:
    output = np.zeros((*valid.shape, 3), dtype=np.uint8)
    output[valid] = color
    return Image.fromarray(output, "RGB")


def focused_isolated(image: np.ndarray, valid: np.ndarray, size: int) -> Image.Image:
    """Center a sparse real layer so its retained appearance is inspectable."""
    ys, xs = np.nonzero(valid)
    if len(xs) == 0:
        return Image.new("RGB", (size, size), "black")
    margin = 5
    x0, x1 = max(0, int(xs.min()) - margin), min(size, int(xs.max()) + margin + 1)
    y0, y1 = max(0, int(ys.min()) - margin), min(size, int(ys.max()) + margin + 1)
    crop = np.zeros((y1 - y0, x1 - x0, 3), dtype=np.uint8)
    local = valid[y0:y1, x0:x1]
    crop[local] = image[y0:y1, x0:x1][local]
    scale = min(0.82 * size / max(crop.shape[1], 1), 0.82 * size / max(crop.shape[0], 1))
    resized = Image.fromarray(crop, "RGB").resize(
        (max(1, int(round(crop.shape[1] * scale))), max(1, int(round(crop.shape[0] * scale)))),
        Image.Resampling.NEAREST,
    )
    output = Image.new("RGB", (size, size), "black")
    output.paste(resized, ((size - resized.width) // 2, (size - resized.height) // 2))
    return output


def exo_mosaic(sequence_root: Path, frame: int, size: int) -> Image.Image:
    half = size // 2
    result = Image.new("RGB", (size, size), "black")
    for camera in range(4):
        value = Image.open(
            sequence_root / f"cam{camera}/rgb/{frame:06d}.png"
        ).convert("RGB").resize((half, half), Image.Resampling.BILINEAR)
        result.paste(value, ((camera % 2) * half, (camera // 2) * half))
    return result


def draw_arrow(
    canvas: Image.Image,
    start: tuple[int, int],
    end: tuple[int, int],
) -> None:
    draw = ImageDraw.Draw(canvas)
    color = (100, 116, 139)
    draw.line((*start, *end), fill=color, width=4)
    draw.polygon(
        ((end[0], end[1]), (end[0] - 10, end[1] - 7), (end[0] - 10, end[1] + 7)),
        fill=color,
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--sequence-root", type=Path, required=True)
    parser.add_argument("--model-input-root", type=Path, required=True)
    parser.add_argument("--refined-root", type=Path, required=True)
    parser.add_argument("--frame-index", type=int, default=14)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    manifest = json.loads((args.model_input_root / "manifest.json").read_text())
    size = int(manifest["image_size"])
    record = manifest["frames"][args.frame_index]
    frame = int(record["dataset_frame"])
    first_frame = int(manifest["frames"][0]["dataset_frame"])
    name = f"{args.frame_index:06d}.png"
    first_name = "000000.png"

    initial_ego = rgb(args.sequence_root / f"cam4/rgb/{first_frame:06d}.png", size)
    current_target = rgb(args.sequence_root / f"cam4/rgb/{frame:06d}.png", size)
    prepaint = rgb(args.model_input_root / "input_frames" / name, size)
    background = rgb(args.model_input_root / "background_frames" / name, size)
    object_rgb = rgb(args.model_input_root / "object_layer_frames" / name, size)
    final = rgb(args.refined_root / "frames" / name, size)

    raw_arm = mask(args.model_input_root / "arm_masks" / name, size)
    stable_arm = mask(args.refined_root / "arm_masks" / name, size)
    object_valid = mask(args.model_input_root / "object_masks" / name, size) & (~stable_arm)
    # The object can be fully hand-occluded in frame zero. Its appearance is
    # nevertheless anchored there in object coordinates; show the first frame
    # where that filtered anchored surface becomes visible after rigid motion.
    anchor_name = next(
        candidate
        for candidate in (f"{index:06d}.png" for index in range(len(manifest["frames"])))
        if np.any(mask(args.model_input_root / "object_masks" / candidate, size))
    )
    anchor_object_rgb = rgb(
        args.model_input_root / "object_layer_frames" / anchor_name, size
    )
    anchor_object = mask(
        args.model_input_root / "object_masks" / anchor_name, size
    )
    exclusion = mask(
        args.model_input_root / "anchor_source_hand_exclusion.png", size
    )

    exclusion_panel = initial_ego.copy()
    exclusion_panel[exclusion] = np.rint(
        0.35 * exclusion_panel[exclusion]
        + 0.65 * np.asarray(FOREGROUND_COLOR)
    ).astype(np.uint8)
    interaction_mask = np.zeros((size, size, 3), dtype=np.uint8)
    interaction_mask[object_valid] = OBJECT_COLOR
    interaction_mask[stable_arm] = FOREGROUND_COLOR
    interaction_rgb = np.zeros_like(final)
    interaction_rgb[object_valid | stable_arm] = final[object_valid | stable_arm]

    panels = [
        # Hand/arm path.
        (exo_mosaic(args.sequence_root, frame, size), "四路exo", INPUT_BORDER),
        (color_mask(raw_arm, FOREGROUND_COLOR), "手/臂投影", INTERMEDIATE_BORDER),
        (isolated(prepaint, raw_arm), "exo手/臂观测", FOREGROUND_COLOR),
        (color_mask(stable_arm, FOREGROUND_COLOR), "时序稳定mask", INTERMEDIATE_BORDER),
        (isolated(final, stable_arm), "手/臂流", FOREGROUND_COLOR),
        # Object path.
        (Image.fromarray(initial_ego, "RGB"), "ego首帧", INPUT_BORDER),
        (Image.fromarray(exclusion_panel, "RGB"), "手遮挡过滤", INTERMEDIATE_BORDER),
        (focused_isolated(anchor_object_rgb, anchor_object, size), "物体外观锚点", OBJECT_COLOR),
        (color_mask(object_valid, OBJECT_COLOR), "exo物体SE(3)", INTERMEDIATE_BORDER),
        (isolated(object_rgb, object_valid), "物体流", OBJECT_COLOR),
        # Interaction and final composition.
        (Image.fromarray(background, "RGB"), "背景流", INTERMEDIATE_BORDER),
        (Image.fromarray(interaction_mask, "RGB"), "手/物mask", INTERMEDIATE_BORDER),
        (Image.fromarray(interaction_rgb, "RGB"), "深度遮挡", INTERMEDIATE_BORDER),
        (Image.fromarray(final, "RGB"), "最终输出", PREDICTION_BORDER),
        (Image.fromarray(current_target, "RGB"), "未来ego真值", TARGET_BORDER),
    ]

    columns, rows = 5, 3
    arrow_gap, row_gap = 42, 18
    tile_height = size + 44
    canvas_width = columns * size + (columns - 1) * arrow_gap
    canvas_height = rows * tile_height + (rows - 1) * row_gap
    canvas = Image.new("RGB", (canvas_width, canvas_height), (241, 245, 249))
    for index, (image, label, border) in enumerate(panels):
        row, column = divmod(index, columns)
        x = column * (size + arrow_gap)
        y = row * (tile_height + row_gap)
        canvas.paste(title_tile(image, label, size, border), (x, y))
        if column < columns - 1:
            draw_arrow(
                canvas,
                (x + size + 7, y + size // 2),
                (x + size + arrow_gap - 7, y + size // 2),
            )

    args.output.parent.mkdir(parents=True, exist_ok=True)
    canvas.save(args.output)
    print(args.output)


if __name__ == "__main__":
    main()
