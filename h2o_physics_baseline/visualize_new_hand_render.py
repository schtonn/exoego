#!/usr/bin/env python3
"""Compare learned repair with narrow exo rendering on real H2O hand motion."""

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
from h2o_physics_baseline.evaluate_multiview_background_fill import multiview_candidates, scaled_rotation
from h2o_physics_baseline.model import PhysicalEgoVideoPredictor, _student_disocclusion_fill, _student_flow_warp
from h2o_physics_baseline.visualize_student_warp_fill import (
    INPUT_BORDER, INTERMEDIATE_BORDER, PREDICTION_BORDER, TARGET_BORDER,
    font, heat_image, labelled_tile, rgb_image,
)


def batch(item: dict, name: str) -> torch.Tensor:
    return torch.from_numpy(item[name])[None]


def render_case(pair_id: str, label: str, metrics: dict, dataset, model, motion_by_pair) -> Image.Image:
    index = next(i for i, row in enumerate(dataset.rows) if row["pair_id"] == pair_id)
    item, row = dataset[index], dataset.rows[index]
    anchor, maps = batch(item, "ego_anchor"), batch(item, "initial_ego_maps")
    with torch.inference_mode():
        network, _ = model(
            batch(item, "source"), batch(item, "physical_vector"), batch(item, "physical_maps"),
            batch(item, "coarse_rgb"), batch(item, "coarse_depth"), batch(item, "visibility_mask"),
            anchor, maps,
        )
    warped = _student_flow_warp(anchor, maps)
    local, old_only, new_only = _student_disocclusion_fill(anchor, warped, maps)
    time_index = len(item["frame_numbers"]) - 1
    frame, initial_frame = int(item["frame_numbers"][time_index]), int(item["frame_numbers"][0])
    source_roots = [Path(value).parent for value in json.loads(row["source_rgb_dirs"])]
    target_root = Path(row["target_rgb_dir"]).parent
    pose = load_pose(target_root / "cam_pose" / f"{initial_frame:06d}.txt")
    motion = motion_by_pair[pair_id][frame]
    pose[:3, 3] += motion["position"]
    pose[:3, :3] = scaled_rotation(motion["rotation"], 0.5) @ pose[:3, :3]
    candidates = multiview_candidates(source_roots, frame, target_root, pose, 64, 2)

    def candidate(name: str) -> tuple[torch.Tensor, torch.Tensor]:
        rgb, valid = candidates[name]
        return (
            torch.from_numpy(np.moveaxis(rgb, -1, 0))[None],
            torch.from_numpy(valid.astype(np.float32))[None, None],
        )

    background, background_valid = candidate("depth_consensus_5cm_gaussian3")
    nearest, nearest_valid = candidate("nearest")
    replacement = background * background_valid + local[:, time_index] * (1 - background_valid)
    geometry = warped[:, time_index] * (1 - old_only[:, time_index]) + replacement * old_only[:, time_index]
    network_background = (
        network[:, time_index] * (1 - old_only[:, time_index] * background_valid)
        + background * old_only[:, time_index] * background_valid
    )
    new_alpha = new_only[:, time_index] * nearest_valid
    rendered = geometry * (1 - new_alpha) + nearest * new_alpha

    width, gap = 236, 8
    top = []
    for camera_index, root in enumerate(source_roots):
        top.append(labelled_tile(
            Image.open(root / "rgb" / f"{frame:06d}.png").convert("RGB"),
            f"输入 exo cam{camera_index}", f"frame {frame}", width, 133, INPUT_BORDER,
        ))
    top.append(labelled_tile(
        rgb_image(anchor[0, 0], width), "输入 ego 首帧", f"frame {initial_frame}",
        width, 133, INPUT_BORDER,
    ))
    bottom = [
        labelled_tile(
            heat_image(new_only[0, time_index], width, (255, 154, 0)),
            "new-only 手部前缘", "只在橙色窄区域取 exo 外观", width, width, INTERMEDIATE_BORDER,
        ),
        labelled_tile(
            rgb_image(geometry[0], width), "头动几何基线",
            f"L1 {metrics['depth_consensus_5cm_gaussian3']['l1']:.4f}",
            width, width, PREDICTION_BORDER,
        ),
        labelled_tile(
            rgb_image(network_background[0], width), "原学习修复+背景",
            f"L1 {metrics['network_depth_consensus_5cm_gaussian3']['l1']:.4f}",
            width, width, PREDICTION_BORDER,
        ),
        labelled_tile(
            rgb_image(rendered[0], width), "exo 新手表面替换",
            f"L1 {metrics['rendered_new_hand_nearest']['l1']:.4f}",
            width, width, PREDICTION_BORDER,
        ),
        labelled_tile(
            rgb_image(torch.from_numpy(item["target"])[time_index], width),
            "未来 ego 真值", "仅用于评估", width, width, TARGET_BORDER,
        ),
    ]
    panel_width = width * 5 + gap * 4
    header = 62
    panel = Image.new("RGB", (panel_width, header + top[0].height + gap + bottom[0].height), (241, 245, 249))
    draw = ImageDraw.Draw(panel)
    gain = metrics["depth_consensus_5cm_gaussian3"]["l1"] - metrics["rendered_new_hand_nearest"]["l1"]
    draw.text((8, 4), label, font=font(23), fill=(15, 23, 42))
    draw.text((8, 34), f"{pair_id}　新手表面收益 ΔL1={gain:+.4f}", font=font(15), fill=(51, 65, 85))
    for column, tile in enumerate(top):
        panel.paste(tile, (column * (width + gap), header))
    y = header + top[0].height + gap
    for column, tile in enumerate(bottom):
        panel.paste(tile, (column * (width + gap), y))
    return panel


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--checkpoint", type=Path,
        default=Path("datasets/H2O/experiments/student_warp_fill_v2/fill_seed26/multiview_anchored_student_flow_warp_fill_local_gate31/best.pt"),
    )
    parser.add_argument(
        "--summary", type=Path,
        default=Path("datasets/H2O/experiments/multiview_background_fill/summary_rendered_hand.json"),
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
        default=Path("h2o_physics_baseline/figures/new_hand_render_success_failure.png"),
    )
    args = parser.parse_args()
    checkpoint = torch.load(args.checkpoint, map_location="cpu", weights_only=False)
    config = checkpoint["config"]
    dataset = H2OPhysicalClipDataset(
        config["index"], split="val", frames_per_clip=config["frames_per_clip"], image_size=64,
        stats_path=config["stats"], source_cameras=tuple(config["source_cameras"]),
        max_samples=config["max_val_samples"], combine_source_cameras=True,
        student_state_root=args.student_state_root, student_flow=True,
    )
    model = PhysicalEgoVideoPredictor(condition_mode=config["condition_mode"])
    model.load_state_dict(checkpoint["model"])
    model.eval()
    summary = json.loads(args.summary.read_text(encoding="utf-8"))
    by_pair = {record["pair_id"]: record for record in summary["per_clip"]}
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
    scores = [
        (
            record["variants"]["depth_consensus_5cm_gaussian3"]["l1"]
            - record["variants"]["rendered_new_hand_nearest"]["l1"],
            pair_id,
        )
        for pair_id, record in by_pair.items()
    ]
    best_pair, worst_pair = max(scores)[1], min(scores)[1]
    panels = [
        render_case(best_pair, "成功例：只补新出现的手部表面", by_pair[best_pair]["variants"], dataset, model, motion_by_pair),
        render_case(worst_pair, "失败例：外部视角前景仍可能错位", by_pair[worst_pair]["variants"], dataset, model, motion_by_pair),
    ]
    gap = 18
    output = Image.new("RGB", (panels[0].width, panels[0].height * 2 + gap), (226, 232, 240))
    output.paste(panels[0], (0, 0))
    output.paste(panels[1], (0, panels[0].height + gap))
    args.output.parent.mkdir(parents=True, exist_ok=True)
    output.save(args.output)
    metadata = {"output": str(args.output), "best_pair_id": best_pair, "worst_pair_id": worst_pair}
    args.output.with_suffix(".json").write_text(json.dumps(metadata, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print(json.dumps(metadata, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
