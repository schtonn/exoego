#!/usr/bin/env python3
"""Contrast local pose-assisted filling with a true full-frame ego-camera warp."""

from __future__ import annotations

import json
from pathlib import Path
import sys

import numpy as np
import torch
from PIL import Image, ImageDraw

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from h2o_geometric_baseline.reprojection import load_intrinsics, load_pose, reproject_rgbd
from h2o_physics_baseline.dataset import H2OPhysicalClipDataset
from h2o_physics_baseline.evaluate_multiview_background_fill import multiview_candidates, scaled_rotation
from h2o_physics_baseline.model import _student_disocclusion_fill, _student_flow_warp
from h2o_physics_baseline.visualize_student_warp_fill import (
    INPUT_BORDER, INTERMEDIATE_BORDER, PREDICTION_BORDER, TARGET_BORDER,
    font, labelled_tile, rgb_image,
)


PAIR_ID = "subject3_k2_4_000288_000351_cam0_to_cam4"
OUTPUT = Path("h2o_physics_baseline/figures/local_fill_vs_global_head_warp.png")


def rotation_difference_deg(first: np.ndarray, second: np.ndarray) -> float:
    relative = first @ second.T
    cosine = np.clip((np.trace(relative) - 1.0) / 2.0, -1.0, 1.0)
    return float(np.degrees(np.arccos(cosine)))


def main() -> None:
    checkpoint_path = Path(
        "datasets/H2O/experiments/student_warp_fill_v2/fill_seed26/"
        "multiview_anchored_student_flow_warp_fill_local_gate31/best.pt"
    )
    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    config = checkpoint["config"]
    dataset = H2OPhysicalClipDataset(
        config["index"], split="val", frames_per_clip=4, image_size=64,
        stats_path=config["stats"], source_cameras=tuple(config["source_cameras"]),
        max_samples=32, combine_source_cameras=True,
        student_state_root="datasets/H2O/student_state_mediapipe_64_32", student_flow=True,
    )
    index = next(i for i, row in enumerate(dataset.rows) if row["pair_id"] == PAIR_ID)
    item, row = dataset[index], dataset.rows[index]
    anchor = torch.from_numpy(item["ego_anchor"])[None]
    maps = torch.from_numpy(item["initial_ego_maps"])[None]
    warped = _student_flow_warp(anchor, maps)
    local, old_only, _ = _student_disocclusion_fill(anchor, warped, maps)
    time_index = 3
    frame, initial_frame = int(item["frame_numbers"][3]), int(item["frame_numbers"][0])
    target_root = Path(row["target_rgb_dir"]).parent
    source_roots = [Path(value).parent for value in json.loads(row["source_rgb_dirs"])]
    head = json.loads(Path("datasets/H2O/experiments/exo_face_head_motion/summary.json").read_text())
    head_record = next(record for record in head["per_clip"] if record["pair_id"] == PAIR_ID)
    motion = next(value for value in head_record["future"] if int(value["frame"]) == frame)
    initial_pose = load_pose(target_root / "cam_pose" / f"{initial_frame:06d}.txt")
    predicted_pose = initial_pose.copy()
    predicted_pose[:3, 3] += np.asarray(motion["estimated_camera_delta_position_world_m"])
    predicted_pose[:3, :3] = (
        scaled_rotation(np.asarray(motion["estimated_delta_rotation_world"]), 0.5)
        @ initial_pose[:3, :3]
    )
    ground_truth_pose = load_pose(target_root / "cam_pose" / f"{frame:06d}.txt")
    predicted_translation_cm = np.linalg.norm(predicted_pose[:3, 3] - initial_pose[:3, 3]) * 100
    ground_truth_translation_cm = np.linalg.norm(ground_truth_pose[:3, 3] - initial_pose[:3, 3]) * 100
    predicted_rotation_deg = rotation_difference_deg(predicted_pose[:3, :3], initial_pose[:3, :3])
    ground_truth_rotation_deg = rotation_difference_deg(ground_truth_pose[:3, :3], initial_pose[:3, :3])
    candidates = multiview_candidates(source_roots, frame, target_root, predicted_pose, 64, 2)
    background_rgb, background_valid_np = candidates["depth_consensus_5cm_gaussian3"]
    background = torch.from_numpy(np.moveaxis(background_rgb, -1, 0))[None]
    background_valid = torch.from_numpy(background_valid_np.astype(np.float32))[None, None]
    replacement = background * background_valid + local[:, time_index] * (1 - background_valid)
    local_fill = warped[:, time_index] * (1 - old_only[:, time_index]) + replacement * old_only[:, time_index]

    initial_rgb = np.asarray(Image.open(target_root / "rgb" / f"{initial_frame:06d}.png").convert("RGB"))
    initial_depth = np.asarray(Image.open(target_root / "depth" / f"{initial_frame:06d}.png"))
    intrinsics = load_intrinsics(target_root / "cam_intrinsics.txt")
    result = reproject_rgbd(
        initial_rgb, initial_depth, intrinsics, initial_pose, intrinsics, predicted_pose,
        output_size=(64, 64), source_stride=2,
    )
    global_rgb = torch.from_numpy(np.moveaxis(result.rgb.astype(np.float32) / 255.0, -1, 0))[None]
    global_valid = torch.from_numpy(result.valid.astype(np.float32))[None, None]
    global_warp = global_rgb * global_valid + anchor[:, time_index] * (1 - global_valid)
    hole_alpha = (1 - global_valid) * background_valid
    global_fill = (
        global_rgb * global_valid + background * hole_alpha
        + anchor[:, time_index] * (1 - global_valid) * (1 - background_valid)
    )
    oracle_result = reproject_rgbd(
        initial_rgb, initial_depth, intrinsics, initial_pose, intrinsics, ground_truth_pose,
        output_size=(64, 64), source_stride=2,
    )
    oracle_rgb = torch.from_numpy(
        np.moveaxis(oracle_result.rgb.astype(np.float32) / 255.0, -1, 0)
    )[None]
    oracle_valid = torch.from_numpy(oracle_result.valid.astype(np.float32))[None, None]
    oracle_warp = oracle_rgb * oracle_valid + anchor[:, time_index] * (1 - oracle_valid)
    target = torch.from_numpy(item["target"])[time_index]
    summary = json.loads(Path(
        "datasets/H2O/experiments/multiview_background_fill/summary_global_warp.json"
    ).read_text())
    metrics = next(record["variants"] for record in summary["per_clip"] if record["pair_id"] == PAIR_ID)
    oracle_summary = json.loads(Path(
        "datasets/H2O/experiments/multiview_background_fill/summary_layered_global_warp_oracle_pose.json"
    ).read_text())
    oracle_metrics = next(
        record["variants"] for record in oracle_summary["per_clip"] if record["pair_id"] == PAIR_ID
    )

    width, gap = 236, 8
    top = []
    for camera_index, root in enumerate(source_roots):
        top.append(labelled_tile(
            Image.open(root / "rgb" / f"{frame:06d}.png").convert("RGB"),
            f"输入 exo cam{camera_index}", f"frame {frame}", width, 133, INPUT_BORDER,
        ))
    top.append(labelled_tile(
        rgb_image(anchor[0, 0], width), "输入 ego 首帧RGB-D", f"frame {initial_frame}",
        width, 133, INPUT_BORDER,
    ))
    bottom = [
        labelled_tile(
            rgb_image(anchor[0, time_index], width), "静态首帧", "没有全局头动", width, width, INTERMEDIATE_BORDER,
        ),
        labelled_tile(
            rgb_image(local_fill[0], width), "头姿辅助局部补洞",
            f"L1 {metrics['depth_consensus_5cm_gaussian3']['l1']:.4f}；画面仍固定",
            width, width, PREDICTION_BORDER,
        ),
        labelled_tile(
            rgb_image(global_warp[0], width), "真正整幅头动 warp",
            f"预测 {predicted_translation_cm:.2f}cm / {predicted_rotation_deg:.2f}°",
            width, width, PREDICTION_BORDER,
        ),
        labelled_tile(
            rgb_image(oracle_warp[0], width), "真值位姿 warp（诊断）",
            f"L1 {oracle_metrics['global_ego_rgbd_warp']['l1']:.4f}；仍有渲染孔洞",
            width, width, INTERMEDIATE_BORDER,
        ),
        labelled_tile(
            rgb_image(target, width), "未来 ego 真值",
            f"真值 {ground_truth_translation_cm:.2f}cm / {ground_truth_rotation_deg:.2f}°",
            width, width, TARGET_BORDER,
        ),
    ]
    panel_width = width * 5 + gap * 4
    header = 68
    output = Image.new("RGB", (panel_width, header + top[0].height + gap + bottom[0].height), (241, 245, 249))
    draw = ImageDraw.Draw(output)
    draw.text((8, 3), "局部补洞 ≠ 生成全局头动", font=font(25), fill=(15, 23, 42))
    draw.text(
        (8, 36), "预测旋转明显偏小；换成真值位姿后方向正确，但单层前向点云warp仍有孔洞和旧手残影",
        font=font(15), fill=(51, 65, 85),
    )
    for column, tile in enumerate(top):
        output.paste(tile, (column * (width + gap), header))
    y = header + top[0].height + gap
    for column, tile in enumerate(bottom):
        output.paste(tile, (column * (width + gap), y))
    OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    output.save(OUTPUT)
    print(OUTPUT)


if __name__ == "__main__":
    main()
