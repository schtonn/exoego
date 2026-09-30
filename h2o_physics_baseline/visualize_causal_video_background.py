#!/usr/bin/env python3
"""Show real H2O success/failure cases for causal temporal background fusion."""

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
from h2o_physics_baseline.evaluate_multiview_background_fill import (
    causal_static_video_candidates,
    multiview_candidates,
    scaled_rotation,
)
from h2o_physics_baseline.visualize_student_warp_fill import (
    INPUT_BORDER,
    INTERMEDIATE_BORDER,
    PREDICTION_BORDER,
    TARGET_BORDER,
    font,
    labelled_tile,
)


PAIRS = (
    ("视频补洞成功", "subject3_k2_5_000608_000671_cam0_to_cam4"),
    ("视频补洞失败", "subject3_o2_7_000480_000543_cam0_to_cam4"),
)
OUTPUT = Path("h2o_physics_baseline/figures/causal_video_background_success_failure.png")


def pil_rgb(value: np.ndarray, size: int) -> Image.Image:
    array = np.clip(value, 0, 1)
    return Image.fromarray(np.rint(array * 255).astype(np.uint8), "RGB").resize(
        (size, size), Image.Resampling.NEAREST
    )


def predicted_pose_for_frame(
    pair_id: str,
    frame: int,
    first_frame: int,
    initial_pose: np.ndarray,
    head_summary: dict,
) -> np.ndarray:
    if frame == first_frame:
        return initial_pose
    record = next(value for value in head_summary["per_clip"] if value["pair_id"] == pair_id)
    motion = next(value for value in record["future"] if int(value["frame"]) == frame)
    pose = initial_pose.copy()
    pose[:3, 3] += np.asarray(motion["estimated_camera_delta_position_world_m"])
    pose[:3, :3] = (
        scaled_rotation(np.asarray(motion["estimated_delta_rotation_world"]), 0.5)
        @ initial_pose[:3, :3]
    )
    return pose


def render_row(
    label: str,
    pair_id: str,
    dataset: H2OPhysicalClipDataset,
    head_summary: dict,
    metrics_summary: dict,
    tile_size: int,
) -> tuple[list[Image.Image], str]:
    index = next(i for i, row in enumerate(dataset.rows) if row["pair_id"] == pair_id)
    item, row = dataset[index], dataset.rows[index]
    frames = [int(value) for value in item["frame_numbers"]]
    first_frame, frame = frames[0], frames[-1]
    target_root = Path(row["target_rgb_dir"]).parent
    source_roots = [Path(value).parent for value in json.loads(row["source_rgb_dirs"])]
    intrinsics = load_intrinsics(target_root / "cam_intrinsics.txt")
    initial_pose = load_pose(target_root / "cam_pose" / f"{first_frame:06d}.txt")
    target_pose = predicted_pose_for_frame(pair_id, frame, first_frame, initial_pose, head_summary)
    initial_rgb_u8 = np.asarray(
        Image.open(target_root / "rgb" / f"{first_frame:06d}.png").convert("RGB")
    )
    initial_rgb = np.asarray(
        Image.fromarray(initial_rgb_u8).resize((64, 64), Image.Resampling.BILINEAR)
    ).astype(np.float32) / 255.0
    initial_depth = np.asarray(Image.open(target_root / "depth" / f"{first_frame:06d}.png"))
    support = item["initial_ego_maps"][0, :2].max(axis=0) > 0.05
    raw_mask = np.asarray(
        Image.fromarray(support.astype(np.uint8) * 255).resize(
            (initial_depth.shape[1], initial_depth.shape[0]), Image.Resampling.NEAREST
        )
    ) > 0
    static_depth = initial_depth.copy()
    static_depth[raw_mask] = 0
    static = reproject_rgbd(
        initial_rgb_u8,
        static_depth,
        intrinsics,
        initial_pose,
        intrinsics,
        target_pose,
        output_size=(64, 64),
        source_stride=2,
    )
    current_rgb, current_valid = multiview_candidates(
        source_roots, frame, target_root, target_pose, 64, 2
    )["depth_consensus_5cm_gaussian3"]
    static_rgb = static.rgb.astype(np.float32) / 255.0
    static_valid = static.valid
    current_alpha = (~static_valid) & current_valid
    current_output = np.where(
        static_valid[..., None], static_rgb,
        np.where(current_alpha[..., None], current_rgb, initial_rgb),
    )
    student_path = Path("datasets/H2O/student_state_mediapipe_64_32") / row["sequence"] / "student_state.npz"
    with np.load(student_path) as archive:
        history = causal_static_video_candidates(
            source_roots,
            frames,
            target_root,
            target_pose,
            64,
            2,
            archive["frames"],
            archive["hand_joints_world_m"],
            archive["joint_confidence"],
        )
    history_rgb, history_valid = history["farthest"]
    recovered = (~static_valid) & (~current_valid) & history_valid
    video_output = np.where(recovered[..., None], history_rgb, current_output)
    gt = np.asarray(
        Image.open(target_root / "rgb" / f"{frame:06d}.png").convert("RGB").resize(
            (64, 64), Image.Resampling.BILINEAR
        )
    ).astype(np.float32) / 255.0
    clip_metrics = next(
        value["variants"] for value in metrics_summary["per_clip"] if value["pair_id"] == pair_id
    )
    static_l1 = clip_metrics["layered_static_background"]["l1"]
    video_l1 = clip_metrics["layered_causal_video_residual_farthest"]["l1"]
    recovered_view = current_output.copy()
    recovered_view[recovered] = 0.25 * recovered_view[recovered] + 0.75 * np.array([0.0, 1.0, 0.85])
    tiles = [
        labelled_tile(pil_rgb(initial_rgb, tile_size), "输入 ego 首帧", "仅首帧RGB-D", tile_size, tile_size, INPUT_BORDER),
        labelled_tile(pil_rgb(static_rgb, tile_size), "静态背景头动投影", "黑区=没有几何信息", tile_size, tile_size, INTERMEDIATE_BORDER),
        labelled_tile(pil_rgb(current_output, tile_size), "当前exo补洞", f"L1 {static_l1:.4f}", tile_size, tile_size, PREDICTION_BORDER),
        labelled_tile(pil_rgb(recovered_view, tile_size), "视频新增覆盖", "青色=仅历史帧可见", tile_size, tile_size, INTERMEDIATE_BORDER),
        labelled_tile(pil_rgb(video_output, tile_size), "因果视频背景", f"L1 {video_l1:.4f}", tile_size, tile_size, PREDICTION_BORDER),
        labelled_tile(pil_rgb(gt, tile_size), "未来 ego 真值", "仅用于评估", tile_size, tile_size, TARGET_BORDER),
    ]
    return tiles, f"{label}：新增覆盖 {recovered.mean():.2%}，ΔL1 {video_l1 - static_l1:+.4f}"


def main() -> None:
    checkpoint = torch.load(
        "datasets/H2O/experiments/student_warp_fill_v2/fill_seed26/"
        "multiview_anchored_student_flow_warp_fill_local_gate31/best.pt",
        map_location="cpu",
        weights_only=False,
    )
    config = checkpoint["config"]
    dataset = H2OPhysicalClipDataset(
        config["index"], split="val", frames_per_clip=4, image_size=64,
        stats_path=config["stats"], source_cameras=tuple(config["source_cameras"]),
        max_samples=32, combine_source_cameras=True,
        student_state_root="datasets/H2O/student_state_mediapipe_64_32", student_flow=True,
    )
    head_summary = json.loads(Path(
        "datasets/H2O/experiments/exo_face_head_motion/summary.json"
    ).read_text())
    metrics_summary = json.loads(Path(
        "datasets/H2O/experiments/multiview_background_fill/summary_causal_video_background.json"
    ).read_text())
    tile_size, gap, header = 196, 8, 67
    rows = [render_row(label, pair, dataset, head_summary, metrics_summary, tile_size) for label, pair in PAIRS]
    row_height = rows[0][0][0].height
    width = tile_size * 6 + gap * 5
    row_block = 24 + row_height + 14
    output = Image.new("RGB", (width, header + row_block * 2), (241, 245, 249))
    draw = ImageDraw.Draw(output)
    draw.text((8, 3), "视频增加覆盖，但新增像素未必可信", font=font(25), fill=(15, 23, 42))
    draw.text((8, 36), "因果设置：每个时刻只累计当前及过去的四路exo RGB-D；先剔除手，再融合静态背景", font=font(15), fill=(51, 65, 85))
    y = header
    for tiles, row_label in rows:
        draw.text((8, y - 1), row_label, font=font(15), fill=(51, 65, 85))
        tile_y = y + 24
        for column, tile in enumerate(tiles):
            output.paste(tile, (column * (tile_size + gap), tile_y))
        y = tile_y + row_height + 14
    OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    output.save(OUTPUT)
    print(OUTPUT)


if __name__ == "__main__":
    main()
