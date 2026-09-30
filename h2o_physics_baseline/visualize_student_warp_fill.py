#!/usr/bin/env python3
"""Render student_warp_fill with real H2O frames and true intermediate tensors."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import torch
from PIL import Image, ImageDraw, ImageFont
from torch.utils.data._utils.collate import default_collate

from h2o_physics_baseline.dataset import H2OPhysicalClipDataset
from h2o_physics_baseline.model import (
    PhysicalEgoVideoPredictor,
    _student_disocclusion_fill,
    _student_flow_warp,
)


FONT = Path("/usr/share/fonts/opentype/noto/NotoSansCJK-Bold.ttc")
INPUT_BORDER = (22, 163, 74)
INTERMEDIATE_BORDER = (245, 158, 11)
PREDICTION_BORDER = (126, 34, 206)
TARGET_BORDER = (220, 38, 38)
CONNECTIONS = (
    (0, 1), (1, 2), (2, 3), (3, 4),
    (0, 5), (5, 6), (6, 7), (7, 8),
    (0, 9), (9, 10), (10, 11), (11, 12),
    (0, 13), (13, 14), (14, 15), (15, 16),
    (0, 17), (17, 18), (18, 19), (19, 20),
)


def font(size: int) -> ImageFont.FreeTypeFont:
    return ImageFont.truetype(str(FONT), size=size)


def rgb_image(value: torch.Tensor | np.ndarray, size: int) -> Image.Image:
    array = value.detach().cpu().numpy() if isinstance(value, torch.Tensor) else value
    array = np.moveaxis(np.clip(array, 0, 1), 0, -1)
    image = Image.fromarray(np.rint(array * 255).astype(np.uint8), "RGB")
    return image.resize((size, size), Image.Resampling.NEAREST)


def heat_image(value: torch.Tensor | np.ndarray, size: int, color: tuple[int, int, int]) -> Image.Image:
    array = value.detach().cpu().numpy() if isinstance(value, torch.Tensor) else value
    array = np.clip(array.squeeze(), 0, 1)
    rgb = np.zeros((*array.shape, 3), dtype=np.uint8)
    for channel, component in enumerate(color):
        rgb[..., channel] = np.rint(array * component).astype(np.uint8)
    return Image.fromarray(rgb, "RGB").resize((size, size), Image.Resampling.NEAREST)


def draw_skeleton(
    image: Image.Image,
    uv: np.ndarray,
    confidence: np.ndarray,
    width: int = 4,
) -> Image.Image:
    output = image.copy()
    draw = ImageDraw.Draw(output)
    colors = ((44, 220, 120), (36, 190, 255))
    for hand in range(2):
        points = uv[hand * 21 : (hand + 1) * 21]
        scores = confidence[hand * 21 : (hand + 1) * 21]
        pixels = np.stack(
            ((points[:, 0] + 1) * output.width / 2, (points[:, 1] + 1) * output.height / 2),
            axis=1,
        )
        for start, end in CONNECTIONS:
            if scores[start] > 0 and scores[end] > 0:
                draw.line((*pixels[start], *pixels[end]), fill=colors[hand], width=width)
        for point, score in zip(pixels, scores):
            if score > 0:
                radius = max(2, width)
                draw.ellipse(
                    (point[0] - radius, point[1] - radius, point[0] + radius, point[1] + radius),
                    fill=colors[hand],
                    outline=(5, 20, 30),
                    width=1,
                )
    return output


def draw_ego_trajectory(
    image: Image.Image,
    initial_uv: np.ndarray,
    current_uv: np.ndarray,
    initial_confidence: np.ndarray,
    current_confidence: np.ndarray,
) -> Image.Image:
    output = draw_skeleton(image, initial_uv, initial_confidence, width=5)
    # Current pose is magenta to make the initial-to-current displacement explicit.
    draw = ImageDraw.Draw(output)
    current_pixels = np.stack(
        ((current_uv[:, 0] + 1) * output.width / 2, (current_uv[:, 1] + 1) * output.height / 2),
        axis=1,
    )
    for hand in range(2):
        offset = hand * 21
        for start, end in CONNECTIONS:
            if current_confidence[offset + start] > 0 and current_confidence[offset + end] > 0:
                draw.line(
                    (*current_pixels[offset + start], *current_pixels[offset + end]),
                    fill=(255, 55, 180),
                    width=5,
                )
        valid_initial = initial_confidence[offset : offset + 21] > 0
        valid_current = current_confidence[offset : offset + 21] > 0
        valid = valid_initial & valid_current
        if np.any(valid):
            initial_center = np.mean(initial_uv[offset : offset + 21][valid], axis=0)
            current_center = np.mean(current_uv[offset : offset + 21][valid], axis=0)
            p0 = ((initial_center[0] + 1) * output.width / 2, (initial_center[1] + 1) * output.height / 2)
            p1 = ((current_center[0] + 1) * output.width / 2, (current_center[1] + 1) * output.height / 2)
            draw.line((*p0, *p1), fill=(255, 220, 40), width=7)
            angle = np.arctan2(p1[1] - p0[1], p1[0] - p0[0])
            for delta in (-0.6, 0.6):
                tip = (p1[0] - 18 * np.cos(angle + delta), p1[1] - 18 * np.sin(angle + delta))
                draw.line((*p1, *tip), fill=(255, 220, 40), width=7)
    return output


def labelled_tile(
    image: Image.Image,
    title: str,
    subtitle: str,
    width: int,
    image_height: int,
    border: tuple[int, int, int],
) -> Image.Image:
    tile = Image.new("RGB", (width, image_height + 66), "white")
    resized = image.resize((width, image_height), Image.Resampling.LANCZOS)
    tile.paste(resized, (0, 0))
    draw = ImageDraw.Draw(tile)
    draw.rectangle((0, 0, width - 1, image_height - 1), outline=border, width=5)
    draw.text((10, image_height + 5), title, font=font(21), fill=(15, 23, 42))
    draw.text((10, image_height + 34), subtitle, font=font(14), fill=(71, 85, 105))
    return tile


def choose_clip(dataset: H2OPhysicalClipDataset) -> tuple[int, dict, float]:
    best_index = 0
    best_item = dataset[0]
    best_score = -1.0
    for index in range(len(dataset)):
        item = dataset[index]
        initial = item["target_joint_uv"][0]
        current = item["target_joint_uv"][-1]
        valid = (item["target_joint_confidence"][0] > 0) & (
            item["target_joint_confidence"][-1] > 0
        )
        score = float(np.linalg.norm(current[valid] - initial[valid], axis=1).mean()) if np.any(valid) else 0.0
        if score > best_score:
            best_index, best_item, best_score = index, item, score
    return best_index, best_item, best_score


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--checkpoint",
        type=Path,
        default=Path(
            "datasets/H2O/experiments/student_warp_fill_v2/fill_seed26/"
            "multiview_anchored_student_flow_warp_fill_local_gate31/best.pt"
        ),
    )
    parser.add_argument(
        "--student-state-root",
        type=Path,
        default=Path("datasets/H2O/student_state_mediapipe_64_32"),
    )
    parser.add_argument("--output-dir", type=Path, default=Path("h2o_physics_baseline/figures"))
    args = parser.parse_args()

    checkpoint = torch.load(args.checkpoint, map_location="cpu", weights_only=False)
    config = checkpoint["config"]
    dataset = H2OPhysicalClipDataset(
        config["index"],
        split="val",
        frames_per_clip=config["frames_per_clip"],
        image_size=config["image_size"],
        stats_path=config["stats"],
        source_cameras=tuple(config["source_cameras"]),
        max_samples=config["max_val_samples"],
        combine_source_cameras=True,
        student_state_root=args.student_state_root,
        student_flow=True,
        source_joint_geometry=True,
    )
    index, item, motion_score = choose_clip(dataset)
    batch = default_collate([item])
    model = PhysicalEgoVideoPredictor(condition_mode=config["condition_mode"])
    model.load_state_dict(checkpoint["model"])
    model.eval()
    with torch.inference_mode():
        prediction, _ = model(
            batch["source"],
            batch["physical_vector"],
            batch["physical_maps"],
            batch["coarse_rgb"],
            batch["coarse_depth"],
            batch["visibility_mask"],
            batch["ego_anchor"],
            batch["initial_ego_maps"],
        )
        warped = _student_flow_warp(batch["ego_anchor"], batch["initial_ego_maps"])
        filled, old_only, new_only = _student_disocclusion_fill(
            batch["ego_anchor"], warped, batch["initial_ego_maps"]
        )

    final_time = config["frames_per_clip"] - 1
    row = dataset.rows[index]
    frame = int(item["frame_numbers"][final_time])
    source_dirs = json.loads(row["source_rgb_dirs"])
    exo_tiles = []
    for view, directory in enumerate(source_dirs):
        raw = Image.open(Path(directory) / f"{frame:06d}.png").convert("RGB")
        raw = draw_skeleton(
            raw,
            item["source_joint_uv"][view, final_time],
            item["source_joint_confidence"][view, final_time],
            width=5,
        )
        exo_tiles.append(
            labelled_tile(
                raw,
                f"cam{view}：真实 exo 帧",
                "彩色骨架 = student 3D 关节反投影",
                360,
                203,
                INPUT_BORDER,
            )
        )

    size = 230
    anchor = rgb_image(batch["ego_anchor"][0, final_time], size)
    trajectory = draw_ego_trajectory(
        anchor,
        item["target_joint_uv"][0],
        item["target_joint_uv"][final_time],
        item["target_joint_confidence"][0],
        item["target_joint_confidence"][final_time],
    )
    step_images = (
        (anchor, "① ego 首帧", "唯一 ego 图像输入", INPUT_BORDER),
        (trajectory, "② 初始相机内投影", "由输入标定和 student 状态确定", INPUT_BORDER),
        (rgb_image(warped[0, final_time], size), "③ 确定性 warp", "沿 student flow 搬运首帧像素", INTERMEDIATE_BORDER),
        (rgb_image(filled[0, final_time], size), "④ old-only fill", "旧手位置用局部背景均值替换", INTERMEDIATE_BORDER),
        (rgb_image(prediction[0, final_time], size), "⑤ 局部修复输出", "31×31 gate 内学习残差", PREDICTION_BORDER),
        (rgb_image(batch["target"][0, final_time], size), "⑥ 未来 ego 真值", "只用于训练/评测，不是推理输入", TARGET_BORDER),
    )
    step_tiles = [
        labelled_tile(image, title, subtitle, size, size, border)
        for image, title, subtitle, border in step_images
    ]

    initial_support = batch["initial_ego_maps"][0, 0, :2].amax(dim=0)
    current_support = batch["initial_ego_maps"][0, final_time, :2].amax(dim=0)
    diagnostic_images = (
        (heat_image(initial_support, 205, (255, 145, 35)), "初始 support", "输入条件：首帧手部区域", INPUT_BORDER),
        (heat_image(current_support, 205, (50, 170, 255)), "当前 support", "输入条件：当前手部区域", INPUT_BORDER),
        (heat_image(old_only[0, final_time, 0], 205, (255, 120, 40)), "old-only", "中间量：需要擦除/补洞", INTERMEDIATE_BORDER),
        (heat_image(new_only[0, final_time, 0], 205, (70, 150, 255)), "new-only", "中间量：新出现手部区域", INTERMEDIATE_BORDER),
    )
    diagnostic_tiles = [
        labelled_tile(image, title, subtitle, 205, 205, border)
        for image, title, subtitle, border in diagnostic_images
    ]

    canvas = Image.new("RGB", (1530, 1110), (248, 250, 252))
    draw = ImageDraw.Draw(canvas)
    draw.text((35, 22), "student_warp_fill：H2O 真实样本逐步结果", font=font(34), fill=(15, 23, 42))
    draw.text(
        (35, 70),
        f"{item['pair_id']}　frame={frame}　自动选择依据：验证集 student 手部投影位移最大（score={motion_score:.3f}）",
        font=font(17),
        fill=(71, 85, 105),
    )
    legend = (
        (INPUT_BORDER, "输入/物理条件"),
        (INTERMEDIATE_BORDER, "模型中间结果"),
        (PREDICTION_BORDER, "最终预测"),
        (TARGET_BORDER, "评测真值"),
    )
    legend_x = 35
    for color, label in legend:
        draw.rounded_rectangle((legend_x, 100, legend_x + 25, 120), radius=4, fill=color)
        draw.text((legend_x + 32, 98), label, font=font(14), fill=(51, 65, 85))
        legend_x += 150
    for column, tile in enumerate(exo_tiles):
        canvas.paste(tile, (35 + column * 372, 135))
    draw.text((35, 421), "从真实输入到模型输出（64×64 张量按最近邻放大，便于看清像素变化）", font=font(23), fill=(15, 23, 42))
    for column, tile in enumerate(step_tiles):
        canvas.paste(tile, (35 + column * 247, 463))
    draw.text((35, 778), "模型内部的遮挡拓扑图", font=font(23), fill=(15, 23, 42))
    for column, tile in enumerate(diagnostic_tiles):
        canvas.paste(tile, (35 + column * 220, 820))
    draw.rounded_rectangle((935, 820, 1495, 1081), radius=16, fill=(241, 245, 249), outline=(148, 163, 184), width=2)
    draw.text((960, 840), "这张图能说明什么", font=font(22), fill=(15, 23, 42))
    lines = (
        "✓ exo 只负责产生可观测的 3D 手轨迹",
        "✓ ego 首帧像素按正确方向被搬运",
        "✓ old-only 显式指出幽灵手应被擦除的位置",
        "✗ 物体没有被 student 状态建模",
        "✗ 新暴露背景和新手部外观只能粗略修补",
        "红框真值只用于评测，推理时不可见。",
    )
    for line, y in zip(lines, range(885, 1060, 30)):
        draw.text((960, y), line, font=font(16), fill=(51, 65, 85))

    args.output_dir.mkdir(parents=True, exist_ok=True)
    main_path = args.output_dir / "student_warp_fill_real_example.png"
    canvas.save(main_path)

    # A temporal grid exposes how each deterministic stage changes all four frames.
    temporal = Image.new("RGB", (1190, 1265), (248, 250, 252))
    temporal_draw = ImageDraw.Draw(temporal)
    temporal_draw.text((35, 20), "同一真实 clip 的四帧演化", font=font(34), fill=(15, 23, 42))
    temporal_draw.text((35, 67), "列是同步时刻；行是实际中间结果。未来 ego 真值仅作评测对照。", font=font(17), fill=(71, 85, 105))
    legend_x = 650
    for color, label in (
        (INTERMEDIATE_BORDER, "中间结果"),
        (PREDICTION_BORDER, "最终预测"),
        (TARGET_BORDER, "评测真值"),
    ):
        temporal_draw.rounded_rectangle((legend_x, 67, legend_x + 24, 87), radius=4, fill=color)
        temporal_draw.text((legend_x + 31, 65), label, font=font(14), fill=(51, 65, 85))
        legend_x += 165
    row_names = ("确定性 warp", "old-only fill", "学习式输出", "未来 ego 真值")
    row_values = (warped[0], filled[0], prediction[0], batch["target"][0])
    for time_index, frame_number in enumerate(item["frame_numbers"]):
        temporal_draw.text((205 + time_index * 242, 108), f"t{time_index} / frame {int(frame_number)}", font=font(17), fill=(15, 23, 42))
    for row_index, (name, values) in enumerate(zip(row_names, row_values)):
        y = 145 + row_index * 270
        temporal_draw.text((28, y + 103), name, font=font(19), fill=(15, 23, 42))
        for time_index in range(config["frames_per_clip"]):
            image = rgb_image(values[time_index], 230)
            temporal.paste(image, (205 + time_index * 242, y))
            if row_index < 2:
                color = INTERMEDIATE_BORDER
            elif row_index == 2:
                color = PREDICTION_BORDER
            else:
                color = TARGET_BORDER
            temporal_draw.rectangle(
                (205 + time_index * 242, y, 434 + time_index * 242, y + 229),
                outline=color,
                width=4,
            )
    temporal_draw.text((205, 1230), "提示：t0 几乎等于 ego 首帧；差异主要随手部投影轨迹在局部出现。", font=font(17), fill=(71, 85, 105))
    temporal_path = args.output_dir / "student_warp_fill_real_sequence.png"
    temporal.save(temporal_path)

    metadata = {
        "checkpoint": str(args.checkpoint),
        "dataset_index": index,
        "pair_id": item["pair_id"],
        "sequence": item["sequence"],
        "frame_numbers": [int(value) for value in item["frame_numbers"]],
        "selected_final_frame": frame,
        "motion_score_normalized": motion_score,
        "outputs": [str(main_path), str(temporal_path)],
    }
    (args.output_dir / "student_warp_fill_real_example.json").write_text(
        json.dumps(metadata, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )
    print(json.dumps(metadata, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
