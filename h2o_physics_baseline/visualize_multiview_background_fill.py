#!/usr/bin/env python3
"""Visualize the best and worst RGB-D consensus-fill clips with real H2O frames."""

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
from h2o_physics_baseline.evaluate_multiview_background_fill import multiview_candidates
from h2o_physics_baseline.model import (
    PhysicalEgoVideoPredictor,
    _student_disocclusion_fill,
    _student_flow_warp,
)
from h2o_physics_baseline.visualize_student_warp_fill import (
    FONT,
    INPUT_BORDER,
    INTERMEDIATE_BORDER,
    PREDICTION_BORDER,
    TARGET_BORDER,
    font,
    heat_image,
    labelled_tile,
    rgb_image,
)


def tensor(item: dict, name: str) -> torch.Tensor:
    return torch.from_numpy(item[name])[None]


def render_case(
    pair_id: str,
    label: str,
    metrics: dict,
    dataset: H2OPhysicalClipDataset,
    model: PhysicalEgoVideoPredictor,
) -> Image.Image:
    index = next(i for i, row in enumerate(dataset.rows) if row["pair_id"] == pair_id)
    item = dataset[index]
    row = dataset.rows[index]
    anchor = tensor(item, "ego_anchor")
    maps = tensor(item, "initial_ego_maps")
    with torch.inference_mode():
        prediction, _ = model(
            tensor(item, "source"),
            tensor(item, "physical_vector"),
            tensor(item, "physical_maps"),
            tensor(item, "coarse_rgb"),
            tensor(item, "coarse_depth"),
            tensor(item, "visibility_mask"),
            anchor,
            maps,
        )
        warped = _student_flow_warp(anchor, maps)
        _, old_only, _ = _student_disocclusion_fill(anchor, warped, maps)

    time_index = len(item["frame_numbers"]) - 1
    frame = int(item["frame_numbers"][time_index])
    source_roots = [Path(value).parent for value in json.loads(row["source_rgb_dirs"])]
    target_root = Path(row["target_rgb_dir"]).parent
    initial_frame = int(item["frame_numbers"][0])
    initial_pose = load_pose(target_root / "cam_pose" / f"{initial_frame:06d}.txt")
    candidate, valid = multiview_candidates(
        source_roots, frame, target_root, initial_pose, int(prediction.shape[-1]), 4
    )["depth_consensus_5cm"]
    candidate_tensor = torch.from_numpy(np.moveaxis(candidate, -1, 0))
    valid_tensor = torch.from_numpy(valid.astype(np.float32))[None]
    replace = old_only[0, time_index] * valid_tensor
    hybrid = prediction[0, time_index] * (1.0 - replace) + candidate_tensor * replace

    tile_width = 236
    gap = 8
    top = []
    for camera_index, camera_root in enumerate(source_roots):
        image = Image.open(camera_root / "rgb" / f"{frame:06d}.png").convert("RGB")
        top.append(
            labelled_tile(
                image,
                f"输入 exo cam{camera_index}",
                f"frame {frame}",
                tile_width,
                133,
                INPUT_BORDER,
            )
        )
    top.append(
        labelled_tile(
            rgb_image(anchor[0, 0], tile_width),
            "输入 ego 首帧",
            f"frame {initial_frame}",
            tile_width,
            133,
            INPUT_BORDER,
        )
    )
    bottom = [
        labelled_tile(
            heat_image(old_only[0, time_index], tile_width, (255, 154, 0)),
            "old-only 区域",
            "需要补出的背景",
            tile_width,
            tile_width,
            INTERMEDIATE_BORDER,
        ),
        labelled_tile(
            rgb_image(prediction[0, time_index], tile_width),
            "原学习预测",
            f"L1 {metrics['network']['l1']:.4f}",
            tile_width,
            tile_width,
            PREDICTION_BORDER,
        ),
        labelled_tile(
            rgb_image(candidate_tensor, tile_width),
            "多视角深度一致背景",
            "≥2 路且深度差≤5 cm",
            tile_width,
            tile_width,
            INTERMEDIATE_BORDER,
        ),
        labelled_tile(
            rgb_image(hybrid, tile_width),
            "混合预测",
            f"L1 {metrics['network_depth_consensus_5cm']['l1']:.4f}",
            tile_width,
            tile_width,
            PREDICTION_BORDER,
        ),
        labelled_tile(
            rgb_image(tensor(item, "target")[0, time_index], tile_width),
            "未来 ego 真值",
            "仅用于评估",
            tile_width,
            tile_width,
            TARGET_BORDER,
        ),
    ]
    width = tile_width * 5 + gap * 4
    header = 62
    height = header + top[0].height + gap + bottom[0].height
    panel = Image.new("RGB", (width, height), (241, 245, 249))
    draw = ImageDraw.Draw(panel)
    gain = metrics["network"]["l1"] - metrics["network_depth_consensus_5cm"]["l1"]
    draw.text((8, 4), label, font=font(24), fill=(15, 23, 42))
    draw.text(
        (8, 34),
        f"{pair_id}　混合收益 ΔL1={gain:+.4f}",
        font=font(16),
        fill=(51, 65, 85),
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
        "--summary",
        type=Path,
        default=Path("datasets/H2O/experiments/multiview_background_fill/summary_hybrid.json"),
    )
    parser.add_argument(
        "--student-state-root",
        type=Path,
        default=Path("datasets/H2O/student_state_mediapipe_64_32"),
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("h2o_physics_baseline/figures/multiview_background_fill_success_failure.png"),
    )
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
    )
    model = PhysicalEgoVideoPredictor(condition_mode=config["condition_mode"])
    model.load_state_dict(checkpoint["model"])
    model.eval()
    summary = json.loads(args.summary.read_text(encoding="utf-8"))
    scored = []
    for record in summary["per_clip"]:
        variants = record["variants"]
        gain = variants["network"]["l1"] - variants["network_depth_consensus_5cm"]["l1"]
        scored.append((gain, record))
    best = max(scored, key=lambda value: value[0])[1]
    worst = min(scored, key=lambda value: value[0])[1]
    panels = [
        render_case(best["pair_id"], "成功例：一致背景纠正大块空洞", best["variants"], dataset, model),
        render_case(worst["pair_id"], "失败例：几何错位覆盖了更好的学习结果", worst["variants"], dataset, model),
    ]
    gap = 18
    output = Image.new(
        "RGB", (panels[0].width, sum(panel.height for panel in panels) + gap), (226, 232, 240)
    )
    output.paste(panels[0], (0, 0))
    output.paste(panels[1], (0, panels[0].height + gap))
    args.output.parent.mkdir(parents=True, exist_ok=True)
    output.save(args.output)
    metadata = {
        "output": str(args.output),
        "best_pair_id": best["pair_id"],
        "worst_pair_id": worst["pair_id"],
        "border_legend": {
            "green": "input",
            "orange": "geometric intermediate",
            "purple": "prediction",
            "red": "evaluation-only ground truth",
        },
    }
    args.output.with_suffix(".json").write_text(
        json.dumps(metadata, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )
    print(json.dumps(metadata, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
