#!/usr/bin/env python3
"""Show real H2O examples for denoising and face/depth ego-motion geometry."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

import numpy as np
import torch
from PIL import Image, ImageDraw

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from h2o_geometric_baseline.reprojection import load_pose
from h2o_physics_baseline.dataset import H2OPhysicalClipDataset
from h2o_physics_baseline.evaluate_multiview_background_fill import (
    multiview_candidates,
    scaled_rotation,
)
from h2o_physics_baseline.model import _student_disocclusion_fill, _student_flow_warp
from h2o_physics_baseline.visualize_student_warp_fill import (
    INPUT_BORDER,
    INTERMEDIATE_BORDER,
    PREDICTION_BORDER,
    TARGET_BORDER,
    font,
    heat_image,
    labelled_tile,
    rgb_image,
)


def render_case(
    pair_id: str,
    label: str,
    fixed_metrics: dict,
    moving_metrics: dict,
    dataset: H2OPhysicalClipDataset,
    motion_by_pair: dict,
    source_stride: int,
) -> Image.Image:
    index = next(i for i, row in enumerate(dataset.rows) if row["pair_id"] == pair_id)
    item = dataset[index]
    row = dataset.rows[index]
    anchor = torch.from_numpy(item["ego_anchor"])[None]
    maps = torch.from_numpy(item["initial_ego_maps"])[None]
    warped = _student_flow_warp(anchor, maps)
    local, old_only, _ = _student_disocclusion_fill(anchor, warped, maps)
    time_index = len(item["frame_numbers"]) - 1
    frame = int(item["frame_numbers"][time_index])
    initial_frame = int(item["frame_numbers"][0])
    source_roots = [Path(value).parent for value in json.loads(row["source_rgb_dirs"])]
    target_root = Path(row["target_rgb_dir"]).parent
    initial_pose = load_pose(target_root / "cam_pose" / f"{initial_frame:06d}.txt")
    motion = motion_by_pair[pair_id][frame]
    moving_pose = initial_pose.copy()
    moving_pose[:3, 3] += motion["position"]
    moving_pose[:3, :3] = scaled_rotation(motion["rotation"], 0.5) @ initial_pose[:3, :3]
    fixed_candidates = multiview_candidates(
        source_roots, frame, target_root, initial_pose, int(anchor.shape[-1]), source_stride
    )
    moving_candidates = multiview_candidates(
        source_roots, frame, target_root, moving_pose, int(anchor.shape[-1]), source_stride
    )

    def compose(candidate_name: str, candidates: dict) -> tuple[torch.Tensor, torch.Tensor]:
        rgb, valid = candidates[candidate_name]
        candidate = torch.from_numpy(np.moveaxis(rgb, -1, 0))[None]
        valid_tensor = torch.from_numpy(valid.astype(np.float32))[None, None]
        replacement = candidate * valid_tensor + local[:, time_index] * (1.0 - valid_tensor)
        output = warped[:, time_index] * (1.0 - old_only[:, time_index]) + replacement * old_only[:, time_index]
        return candidate[0], output[0]

    raw_candidate, _ = compose("depth_consensus_5cm", fixed_candidates)
    denoised_candidate, fixed_output = compose(
        "depth_consensus_5cm_gaussian3", fixed_candidates
    )
    _, moving_output = compose("depth_consensus_5cm_gaussian3", moving_candidates)
    tile_width = 205
    gap = 7
    top = []
    for camera_index, camera_root in enumerate(source_roots):
        top.append(
            labelled_tile(
                Image.open(camera_root / "rgb" / f"{frame:06d}.png").convert("RGB"),
                f"输入 exo cam{camera_index}",
                f"frame {frame}",
                tile_width,
                116,
                INPUT_BORDER,
            )
        )
    top.append(
        labelled_tile(
            rgb_image(anchor[0, 0], tile_width),
            "输入 ego 首帧",
            f"frame {initial_frame}",
            tile_width,
            116,
            INPUT_BORDER,
        )
    )
    motion_card = Image.new("RGB", (tile_width, 182), "white")
    motion_draw = ImageDraw.Draw(motion_card)
    motion_draw.rectangle((0, 0, tile_width - 1, 115), outline=INPUT_BORDER, width=5)
    translation_cm = np.linalg.norm(motion["position"]) * 100
    rotation_deg = np.degrees(
        np.arccos(np.clip((np.trace(motion["rotation"]) - 1) / 2, -1, 1))
    )
    motion_draw.text((12, 18), "密集人脸+深度", font=font(18), fill=(15, 23, 42))
    motion_draw.text((12, 51), f"平移 {translation_cm:.2f} cm", font=font(16), fill=(30, 64, 175))
    motion_draw.text((12, 80), f"旋转×0.5 {rotation_deg * 0.5:.2f}°", font=font(16), fill=(30, 64, 175))
    motion_draw.text((10, 123), "头动输入", font=font(19), fill=(15, 23, 42))
    motion_draw.text((10, 151), "仅来自四路 exo RGB-D", font=font(13), fill=(71, 85, 105))
    top.append(motion_card)
    bottom = [
        labelled_tile(
            heat_image(old_only[0, time_index], tile_width, (255, 154, 0)),
            "old-only 区域",
            "仅在此处补背景",
            tile_width,
            tile_width,
            INTERMEDIATE_BORDER,
        ),
        labelled_tile(
            rgb_image(raw_candidate, tile_width),
            "原始几何背景",
            "可见离群噪点",
            tile_width,
            tile_width,
            INTERMEDIATE_BORDER,
        ),
        labelled_tile(
            rgb_image(denoised_candidate, tile_width),
            "3×3 归一化高斯",
            "不跨无效孔洞平均",
            tile_width,
            tile_width,
            INTERMEDIATE_BORDER,
        ),
        labelled_tile(
            rgb_image(fixed_output, tile_width),
            "固定头几何输出",
            f"L1 {fixed_metrics['l1']:.4f}",
            tile_width,
            tile_width,
            PREDICTION_BORDER,
        ),
        labelled_tile(
            rgb_image(moving_output, tile_width),
            "预测头动几何输出",
            f"L1 {moving_metrics['l1']:.4f}",
            tile_width,
            tile_width,
            PREDICTION_BORDER,
        ),
        labelled_tile(
            rgb_image(torch.from_numpy(item["target"])[time_index], tile_width),
            "未来 ego 真值",
            "仅用于评估",
            tile_width,
            tile_width,
            TARGET_BORDER,
        ),
    ]
    width = tile_width * 6 + gap * 5
    header = 60
    panel = Image.new(
        "RGB", (width, header + top[0].height + gap + bottom[0].height), (241, 245, 249)
    )
    draw = ImageDraw.Draw(panel)
    gain = fixed_metrics["l1"] - moving_metrics["l1"]
    draw.text((8, 3), label, font=font(23), fill=(15, 23, 42))
    draw.text(
        (8, 33), f"{pair_id}　头动收益 ΔL1={gain:+.4f}",
        font=font(15), fill=(51, 65, 85),
    )
    x = 0
    for tile in top:
        panel.paste(tile, (x, header))
        x += tile_width + gap
    x = 0
    y = header + top[0].height + gap
    for tile in bottom:
        panel.paste(tile, (x, y))
        x += tile_width + gap
    return panel


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
        "--fixed-summary", type=Path,
        default=Path("datasets/H2O/experiments/multiview_background_fill/summary_fixed_stride2.json"),
    )
    parser.add_argument(
        "--moving-summary", type=Path,
        default=Path("datasets/H2O/experiments/multiview_background_fill/summary_face_pose_stride2.json"),
    )
    parser.add_argument(
        "--head-summary", type=Path,
        default=Path("datasets/H2O/experiments/exo_face_head_motion/summary.json"),
    )
    parser.add_argument(
        "--student-state-root", type=Path,
        default=Path("datasets/H2O/student_state_mediapipe_64_32"),
    )
    parser.add_argument(
        "--output", type=Path,
        default=Path("h2o_physics_baseline/figures/background_denoise_head_success_failure.png"),
    )
    parser.add_argument("--source-stride", type=int, default=2)
    args = parser.parse_args()
    checkpoint = torch.load(args.checkpoint, map_location="cpu", weights_only=False)
    config = checkpoint["config"]
    dataset = H2OPhysicalClipDataset(
        config["index"], split="val", frames_per_clip=config["frames_per_clip"],
        image_size=config["image_size"], stats_path=config["stats"],
        source_cameras=tuple(config["source_cameras"]), max_samples=config["max_val_samples"],
        combine_source_cameras=True, student_state_root=args.student_state_root, student_flow=True,
    )
    fixed = json.loads(args.fixed_summary.read_text(encoding="utf-8"))
    moving = json.loads(args.moving_summary.read_text(encoding="utf-8"))
    fixed_by_pair = {record["pair_id"]: record for record in fixed["per_clip"]}
    moving_by_pair = {record["pair_id"]: record for record in moving["per_clip"]}
    head = json.loads(args.head_summary.read_text(encoding="utf-8"))
    motion_by_pair = {
        record["pair_id"]: {
            int(value["frame"]): {
                "position": np.asarray(value["estimated_camera_delta_position_world_m"]),
                "rotation": np.asarray(value["estimated_delta_rotation_world"]),
            }
            for value in record["future"] if "estimated_camera_delta_position_world_m" in value
        }
        for record in head["per_clip"]
    }
    candidate = "depth_consensus_5cm_gaussian3"
    scored = []
    for pair_id, record in moving_by_pair.items():
        fixed_metrics = fixed_by_pair[pair_id]["variants"][candidate]
        moving_metrics = record["variants"][candidate]
        scored.append((fixed_metrics["l1"] - moving_metrics["l1"], pair_id))
    best_pair = max(scored)[1]
    worst_pair = min(scored)[1]
    panels = []
    for pair_id, label in (
        (best_pair, "成功例：去噪后，头动继续修正桌面投影位置"),
        (worst_pair, "失败例：头姿残差仍会造成背景错位"),
    ):
        panels.append(
            render_case(
                pair_id, label,
                fixed_by_pair[pair_id]["variants"][candidate],
                moving_by_pair[pair_id]["variants"][candidate],
                dataset, motion_by_pair, args.source_stride,
            )
        )
    gap = 18
    output = Image.new("RGB", (panels[0].width, panels[0].height * 2 + gap), (226, 232, 240))
    output.paste(panels[0], (0, 0))
    output.paste(panels[1], (0, panels[0].height + gap))
    args.output.parent.mkdir(parents=True, exist_ok=True)
    output.save(args.output)
    metadata = {"output": str(args.output), "best_pair_id": best_pair, "worst_pair_id": worst_pair}
    args.output.with_suffix(".json").write_text(
        json.dumps(metadata, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )
    print(json.dumps(metadata, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
