#!/usr/bin/env python3
"""Render a compact diagnostic for the rejected nearest-limb texture fill."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
import subprocess
import sys

import cv2
import numpy as np
from PIL import Image
from scipy import ndimage

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from h2o_geometric_baseline.reprojection import load_intrinsics
from h2o_physics_baseline.evaluate_multiview_background_fill import (
    arm_support_in_pose,
    hand_support_in_pose,
)
from h2o_physics_baseline.final_layer_fusion import discover_clip
from h2o_physics_baseline.render_causal_video_background_split import (
    DYNAMIC_COMPLETION_COLOR,
    FOREGROUND_COLOR,
    INTERMEDIATE_BORDER,
    PREDICTION_BORDER,
    TARGET_BORDER,
    missing_limb_completion_mask,
    title_tile,
)


def rgb(path: Path, size: int) -> np.ndarray:
    return np.asarray(
        Image.open(path).convert("RGB").resize((size, size), Image.Resampling.BILINEAR),
        dtype=np.float32,
    ) / 255.0


def mask(path: Path, size: int) -> np.ndarray:
    return np.asarray(
        Image.open(path).convert("L").resize((size, size), Image.Resampling.NEAREST)
    ) > 127


def nearest_texture(frame: np.ndarray, observed: np.ndarray) -> np.ndarray | None:
    if not np.any(observed):
        return None
    indices = ndimage.distance_transform_edt(
        ~observed, return_distances=False, return_indices=True
    )
    value = frame[indices[0], indices[1]]
    return cv2.GaussianBlur(value, (0, 0), sigmaX=1.0)


def pil(value: np.ndarray) -> Image.Image:
    return Image.fromarray(np.rint(np.clip(value, 0, 1) * 255).astype(np.uint8), "RGB")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--clip-root", type=Path, required=True)
    parser.add_argument("--index", type=Path, required=True)
    parser.add_argument("--student-state-root", type=Path, required=True)
    parser.add_argument("--arm-state-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--completion-alpha", type=float, default=0.75)
    parser.add_argument("--observed-alpha", type=float, default=0.25)
    parser.add_argument(
        "--hand-mask-root", type=Path,
        help="Optional evaluation-only projected hand-mask directory.",
    )
    args = parser.parse_args()

    clip = discover_clip(args.clip_root, args.index)
    manifest = json.loads((args.clip_root / "model_input/manifest.json").read_text())
    with args.index.open(encoding="utf-8") as handle:
        row = next(row for row in csv.DictReader(handle) if row["pair_id"] == clip.pair_id)
    with np.load(args.student_state_root / row["sequence"] / "student_state.npz") as state:
        hand_frames = state["frames"].copy()
        hand_joints = state["hand_joints_world_m"].copy()
        hand_confidence = state["joint_confidence"].copy()
    with np.load(args.arm_state_root / row["sequence"] / "arm_state.npz") as state:
        arm_frames = state["frames"].copy()
        arm_points = state["arm_points_world_m"].copy()
        arm_confidence = state["confidence"].copy()
    size = int(manifest["image_size"])
    intrinsics = load_intrinsics(clip.target_rgb_dir.parent / "cam_intrinsics.txt")
    rendered = []
    outputs, targets, hand_masks = [], [], []
    for record in manifest["frames"]:
        local = f"{int(record['index']):06d}.png"
        frame = int(record["dataset_frame"])
        current = rgb(args.clip_root / "propainter_composed/frames" / local, size)
        layered = rgb(args.clip_root / "model_input/input_frames" / local, size)
        observed = mask(args.clip_root / "model_input/arm_masks" / local, size)
        object_mask = mask(args.clip_root / "model_input/object_masks" / local, size)
        pose = np.asarray(record["predicted_camera_pose_world"], dtype=np.float64)
        hi = int(np.searchsorted(hand_frames, frame))
        ai = int(np.searchsorted(arm_frames, frame))
        hand_support = hand_support_in_pose(
            hand_joints[hi], hand_confidence[hi], intrinsics, pose, size
        )
        arm_support = arm_support_in_pose(
            arm_points[ai], arm_confidence[ai], intrinsics, pose, size
        )
        completion = missing_limb_completion_mask(
            hand_support, arm_support, observed, object_mask
        )
        expected = (
            ((hand_support > 0.025) | (arm_support > 0.10)) & (~object_mask)
        )
        texture = nearest_texture(layered, observed)
        filled = current.copy()
        if texture is not None:
            correction = expected.astype(np.float32) * args.observed_alpha
            correction[completion] = args.completion_alpha
            filled = (
                current * (1.0 - correction[..., None])
                + texture * correction[..., None]
            )
        channels = np.zeros_like(current)
        channels[observed] = np.asarray(FOREGROUND_COLOR) / 255.0
        channels[completion] = np.asarray(DYNAMIC_COMPLETION_COLOR) / 255.0
        target = rgb(clip.target_rgb_dir / f"{frame:06d}.png", size)
        outputs.append(filled)
        targets.append(target)
        if args.hand_mask_root is not None:
            hand_masks.append(mask(args.hand_mask_root / f"{frame:06d}.png", size))
        tiles = (
            title_tile(pil(current), "当前输出", size, PREDICTION_BORDER),
            title_tile(pil(channels), "手臂观测与补齐区", size, INTERMEDIATE_BORDER),
            title_tile(pil(filled), "纹理延展（未采用）", size, PREDICTION_BORDER),
            title_tile(pil(target), "ego 真值", size, TARGET_BORDER),
        )
        gap, tile_height = 8, size + 44
        canvas = Image.new("RGB", (4 * size + 3 * gap, tile_height), (241, 245, 249))
        for index, tile in enumerate(tiles):
            canvas.paste(tile, (index * (size + gap), 0))
        rendered.append(canvas)

    args.output.parent.mkdir(parents=True, exist_ok=True)
    command = [
        "ffmpeg", "-y", "-loglevel", "error", "-f", "rawvideo",
        "-pix_fmt", "rgb24", "-s", f"{rendered[0].width}x{rendered[0].height}",
        "-r", "15", "-i", "-", "-an", "-c:v", "libx264", "-preset", "slow",
        "-crf", "15", "-pix_fmt", "yuv420p", "-movflags", "+faststart",
        str(args.output),
    ]
    process = subprocess.Popen(command, stdin=subprocess.PIPE)
    assert process.stdin is not None
    for frame in rendered:
        process.stdin.write(np.asarray(frame, dtype=np.uint8).tobytes())
    process.stdin.close()
    if process.wait() != 0:
        raise RuntimeError("ffmpeg failed")
    output_stack = np.stack(outputs)
    target_stack = np.stack(targets)
    error = np.abs(output_stack - target_stack).mean(axis=-1)
    metrics = {
        "full_l1": float(error.mean()),
        "delta_l1": float(np.abs(
            np.diff(output_stack, axis=0) - np.diff(target_stack, axis=0)
        ).mean()),
        "completion_alpha": args.completion_alpha,
        "observed_alpha": args.observed_alpha,
    }
    if hand_masks:
        hand = np.stack(hand_masks)
        metrics["hand_l1"] = float(error[hand].mean())
        metrics["hand_fraction"] = float(hand.mean())
    args.output.with_suffix(".json").write_text(
        json.dumps(metrics, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps(metrics))
    print(args.output)


if __name__ == "__main__":
    main()
