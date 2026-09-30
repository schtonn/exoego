#!/usr/bin/env python3
"""Render a coherent 4x6 input/process/source/ablation study video."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import subprocess
import sys

import numpy as np
from PIL import Image, ImageDraw

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from h2o_geometric_baseline.reprojection import load_pose
from h2o_physics_baseline.render_causal_video_background_split import (
    CURRENT_EXO_COLOR,
    FOREGROUND_COLOR,
    GENERATED_COLOR,
    HISTORY_EXO_COLOR,
    OBJECT_COLOR,
    pil_rgb,
    title_tile,
)
from h2o_physics_baseline.render_input_ablation_comparison import (
    VARIANTS,
    difference_panel,
    isolated_stream,
    labels,
)
from h2o_physics_baseline.render_object_warp_comparison import (
    load_mask,
    load_rgb,
    provenance_image,
)
from h2o_physics_baseline.visualize_student_warp_fill import (
    INPUT_BORDER,
    INTERMEDIATE_BORDER,
    PREDICTION_BORDER,
    TARGET_BORDER,
)


def project_iso(points: np.ndarray, size: int) -> np.ndarray:
    projection = np.asarray(((0.92, 0.0, -0.58), (-0.28, -0.90, -0.44)))
    value = points @ projection.T
    extent = max(float(np.max(np.abs(value))), 0.09)
    return value * (0.39 * size / extent) + np.asarray((0.5 * size, 0.53 * size))


def mount_relation_panel(
    sequence_root: Path,
    first_frame: int,
    pair_id: str,
    mount_summary: dict,
    size: int,
) -> Image.Image:
    """Dataset-derived 3-D head/camera transform, not an illustrative cartoon."""
    record = next(value for value in mount_summary["targets"] if value["pair_id"] == pair_id)
    predicted_camera = np.asarray(record["predicted_initial_pose_world"], dtype=np.float64)
    canonical_mount = np.asarray(
        mount_summary["canonical_head_from_camera"], dtype=np.float64
    )
    exo_head = predicted_camera @ np.linalg.inv(canonical_mount)
    ego_camera = load_pose(
        sequence_root / "cam4/cam_pose" / f"{first_frame:06d}.txt"
    )
    head_from_camera = np.linalg.inv(exo_head) @ ego_camera
    camera_center = head_from_camera[:3, 3]
    camera_rotation = head_from_camera[:3, :3]

    axis_length = 0.055
    frustum_depth = 0.055
    frustum_width = 0.037
    frustum_height = 0.025
    points = [np.zeros(3), camera_center]
    head_axes = [axis_length * np.eye(3)[:, axis] for axis in range(3)]
    camera_axes = [
        camera_center + axis_length * camera_rotation[:, axis] for axis in range(3)
    ]
    corners_local = np.asarray(
        [
            (-frustum_width, -frustum_height, frustum_depth),
            (frustum_width, -frustum_height, frustum_depth),
            (frustum_width, frustum_height, frustum_depth),
            (-frustum_width, frustum_height, frustum_depth),
        ]
    )
    corners = corners_local @ camera_rotation.T + camera_center
    points.extend(head_axes)
    points.extend(camera_axes)
    points.extend(corners)
    projected = project_iso(np.stack(points), size)

    image = Image.new("RGB", (size, size), (248, 250, 252))
    draw = ImageDraw.Draw(image)
    draw.ellipse((18, 18, size - 18, size - 18), outline=(203, 213, 225), width=2)
    origin, camera = projected[0], projected[1]
    draw.line((*origin, *camera), fill=(15, 23, 42), width=4)
    axis_colors = ((239, 68, 68), (34, 197, 94), (59, 130, 246))
    for axis, color in enumerate(axis_colors):
        draw.line((*origin, *projected[2 + axis]), fill=color, width=5)
        draw.line((*camera, *projected[5 + axis]), fill=color, width=5)
    frustum = projected[8:12]
    for corner in frustum:
        draw.line((*camera, *corner), fill=(126, 34, 206), width=3)
    for index in range(4):
        draw.line((*frustum[index], *frustum[(index + 1) % 4]), fill=(126, 34, 206), width=3)
    draw.ellipse((origin[0] - 6, origin[1] - 6, origin[0] + 6, origin[1] + 6), fill=(15, 23, 42))
    draw.ellipse((camera[0] - 6, camera[1] - 6, camera[0] + 6, camera[1] + 6), fill=(126, 34, 206))
    return image


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--scene-label", required=True)
    parser.add_argument("--sequence-root", type=Path, required=True)
    parser.add_argument("--ablation-root", type=Path, required=True)
    parser.add_argument("--no-mount-root", type=Path, required=True)
    parser.add_argument("--mount-summary", type=Path, required=True)
    parser.add_argument("--refined-full4-root", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--fps", type=int, default=15)
    args = parser.parse_args()

    roots = {name: args.ablation_root / name / "model_input" for name, _ in VARIANTS}
    predictions = {
        name: args.ablation_root / name / "propainter_composed/frames"
        for name, _ in VARIANTS
    }
    manifest = json.loads((roots["full4_anchor"] / "manifest.json").read_text())
    mount_summary = json.loads(args.mount_summary.read_text())
    size = int(manifest["image_size"])
    pair_id = manifest["pair_id"]
    first_frame = int(manifest["frames"][0]["dataset_frame"])
    ego_first = load_rgb(args.sequence_root / f"cam4/rgb/{first_frame:06d}.png", size)
    mount_image = mount_relation_panel(
        args.sequence_root, first_frame, pair_id, mount_summary, size
    )
    no_mount_prediction_root = args.no_mount_root / "propainter_composed/frames"

    gap, columns, rows = 8, 6, 4
    tile_height = size + 44
    canvas_width = columns * size + (columns - 1) * gap
    canvas_height = rows * tile_height + (rows - 1) * gap
    rendered = []
    full_root = roots["full4_anchor"]

    for record in manifest["frames"]:
        index = int(record["index"])
        frame = int(record["dataset_frame"])
        name = f"{index:06d}.png"
        exo = [
            load_rgb(args.sequence_root / f"cam{camera}/rgb/{frame:06d}.png", size)
            for camera in range(4)
        ]
        outputs = {
            variant: load_rgb(predictions[variant] / name, size)
            for variant, _ in VARIANTS
        }
        refined_added = np.zeros((size, size), dtype=bool)
        refined_removed = np.zeros((size, size), dtype=bool)
        refined_arm = None
        if args.refined_full4_root is not None:
            outputs["full4_anchor"] = load_rgb(
                args.refined_full4_root / "frames" / name, size
            )
        no_mount = load_rgb(no_mount_prediction_root / name, size)
        target = load_rgb(args.sequence_root / f"cam4/rgb/{frame:06d}.png", size)
        source_labels = labels(full_root, name, size)
        raw_arm_mask = load_mask(full_root / "arm_masks" / name, size)
        raw_object_mask = load_mask(full_root / "object_masks" / name, size)
        if args.refined_full4_root is not None:
            refined_arm_path = args.refined_full4_root / "arm_masks" / name
            refined_fill_path = args.refined_full4_root / "fill_masks" / name
            if refined_arm_path.exists():
                refined_arm = load_mask(refined_arm_path, size)
                refined_added = refined_arm & (~raw_arm_mask)
                refined_removed = raw_arm_mask & (~refined_arm)
            elif refined_fill_path.exists():
                refined_added = load_mask(refined_fill_path, size)
        arm_mask = refined_arm if refined_arm is not None else (raw_arm_mask | refined_added)
        object_mask = raw_object_mask & (~arm_mask)
        priority = ~(arm_mask | object_mask)
        static_mask = (source_labels == 1) & priority
        current_mask = (source_labels == 2) & priority
        history_mask = (source_labels == 3) & priority
        generated_mask = (source_labels == 4) & priority
        geometry = load_rgb(full_root / "geometry_frames" / name, size)
        background = load_rgb(full_root / "background_frames" / name, size)
        pre_propainter = load_rgb(full_root / "input_frames" / name, size)
        raw_propainter = load_rgb(
            args.ablation_root / "full4_anchor/propainter_unknown/input_frames/frames"
            / f"{index:04d}.png",
            size,
        )
        anchor_rgb = (
            pre_propainter if index == 0
            else load_rgb(full_root / "anchor_layer_frames" / name, size)
        )
        exo_rgb = load_rgb(full_root / "exo_layer_frames" / name, size)
        object_rgb = load_rgb(full_root / "object_layer_frames" / name, size)
        hand_rgb = outputs["full4_anchor"]
        provenance = np.asarray(provenance_image(full_root, name, size)).copy()
        # Pixels rejected by temporal boundary consensus return to their actual
        # underlying stream instead of remaining falsely marked as hand/arm.
        provenance[refined_removed] = GENERATED_COLOR
        for label, color in (
            (1, INPUT_BORDER),
            (2, CURRENT_EXO_COLOR),
            (3, HISTORY_EXO_COLOR),
            (4, GENERATED_COLOR),
        ):
            provenance[refined_removed & (source_labels == label)] = color
        provenance[refined_removed & raw_object_mask] = OBJECT_COLOR
        provenance[arm_mask] = FOREGROUND_COLOR
        panels = [
            # Row 1: inputs only.
            *[
                title_tile(
                    pil_rgb(value, size),
                    f"输入 cam{camera} · {frame:06d}",
                    size,
                    INPUT_BORDER,
                )
                for camera, value in enumerate(exo)
            ],
            title_tile(pil_rgb(ego_first, size), f"输入 ego首帧 · {first_frame:06d}", size, INPUT_BORDER),
            title_tile(mount_image, "头—ego相机关系", size, INPUT_BORDER),
            # Row 2: processing chain.
            title_tile(pil_rgb(geometry, size), "几何观测（黑=未知）", size, INTERMEDIATE_BORDER),
            title_tile(pil_rgb(background, size), "相机运动约束背景", size, INTERMEDIATE_BORDER),
            title_tile(Image.fromarray(provenance, "RGB"), "信息来源蒙版", size, INTERMEDIATE_BORDER),
            title_tile(pil_rgb(pre_propainter, size), "ProPainter前分层合成", size, PREDICTION_BORDER),
            title_tile(pil_rgb(raw_propainter, size), "ProPainter原始提议", size, PREDICTION_BORDER),
            title_tile(difference_panel(pre_propainter, outputs["full4_anchor"], size), "最终授权修改量", size, INTERMEDIATE_BORDER),
            # Row 3: mutually exclusive source streams.
            title_tile(isolated_stream(anchor_rgb, static_mask), "首帧几何流", size, INPUT_BORDER),
            title_tile(isolated_stream(exo_rgb, current_mask), "当前exo流", size, CURRENT_EXO_COLOR),
            title_tile(isolated_stream(exo_rgb, history_mask), "历史exo流", size, HISTORY_EXO_COLOR),
            title_tile(isolated_stream(background, generated_mask), "无观测背景生成流", size, GENERATED_COLOR),
            title_tile(isolated_stream(hand_rgb, arm_mask), "手/臂流", size, FOREGROUND_COLOR),
            title_tile(isolated_stream(object_rgb, object_mask), "物体流", size, OBJECT_COLOR),
            # Row 4: every input ablation in one place.
            *[
                title_tile(
                    pil_rgb(outputs[variant], size),
                    label,
                    size,
                    PREDICTION_BORDER,
                )
                for variant, label in VARIANTS
            ],
            title_tile(
                pil_rgb(no_mount, size),
                "4exo无空间关系",
                size,
                PREDICTION_BORDER,
            ),
            title_tile(pil_rgb(target, size), "未来 ego 真值", size, TARGET_BORDER),
        ]
        if len(panels) != columns * rows:
            raise AssertionError(f"Expected 24 panels, got {len(panels)}")
        canvas = Image.new("RGB", (canvas_width, canvas_height), (241, 245, 249))
        for panel_index, panel in enumerate(panels):
            row, column = divmod(panel_index, columns)
            canvas.paste(panel, (column * (size + gap), row * (tile_height + gap)))
        rendered.append(canvas)

    args.output.parent.mkdir(parents=True, exist_ok=True)
    poster = args.output.with_name(args.output.stem + "_poster.png")
    rendered[-1].save(poster)
    process = subprocess.Popen(
        [
            "ffmpeg", "-y", "-loglevel", "error", "-f", "rawvideo", "-pix_fmt", "rgb24",
            "-s", f"{canvas_width}x{canvas_height}", "-r", str(args.fps), "-i", "-", "-an",
            "-c:v", "libx264", "-preset", "slow", "-crf", "15", "-pix_fmt", "yuv420p",
            "-movflags", "+faststart", str(args.output),
        ],
        stdin=subprocess.PIPE,
    )
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
