#!/usr/bin/env python3
"""Precompute deployable 3D arm landmarks from four exo RGB-D streams."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
import sys
from types import SimpleNamespace

import mediapipe as mp
import numpy as np
from mediapipe.tasks import python
from mediapipe.tasks.python import vision
from PIL import Image

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from h2o_geometric_baseline.reprojection import load_intrinsics, load_pose


ARM_LANDMARKS = (11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22)


def load_rows(index: Path, split: str, maximum: int) -> list[dict[str, str]]:
    with index.open(encoding="utf-8") as handle:
        rows = [
            row for row in csv.DictReader(handle)
            if row["split"] == split and row["source_camera"] == "cam0"
        ]
    if maximum < len(rows):
        indices = np.linspace(0, len(rows) - 1, maximum, dtype=np.int64)
        rows = [rows[int(value)] for value in indices]
    return rows


def depth_point_world(
    landmark: object,
    depth_mm: np.ndarray,
    intrinsics: np.ndarray,
    camera_pose: np.ndarray,
    radius: int,
) -> np.ndarray | None:
    height, width = depth_mm.shape
    x = int(round(float(landmark.x) * width))
    y = int(round(float(landmark.y) * height))
    if not (0 <= x < width and 0 <= y < height):
        return None
    patch = depth_mm[
        max(0, y - radius) : min(height, y + radius + 1),
        max(0, x - radius) : min(width, x + radius + 1),
    ].astype(np.float64) / 1000.0
    values = patch[np.isfinite(patch) & (patch >= 0.1) & (patch <= 5.0)]
    if len(values) < 3:
        return None
    z = float(np.median(values))
    fx, fy, cx, cy = intrinsics[:4]
    point_camera = np.asarray(((x - cx) * z / fx, (y - cy) * z / fy, z))
    return camera_pose[:3, :3] @ point_camera + camera_pose[:3, 3]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--index", type=Path,
        default=Path("datasets/H2O/oracle_state/paired_physical_clips.csv"),
    )
    parser.add_argument(
        "--pose-model", type=Path,
        default=Path("models/mediapipe/pose_landmarker_full.task"),
    )
    parser.add_argument("--split", choices=("train", "val", "test"), default="val")
    parser.add_argument("--max-samples", type=int, default=32)
    parser.add_argument("--frames-per-clip", type=int, default=64)
    parser.add_argument("--pair-id", type=str, required=True)
    parser.add_argument("--depth-radius", type=int, default=3)
    parser.add_argument("--camera-indices", default="0,1,2,3")
    parser.add_argument("--output-root", type=Path, required=True)
    args = parser.parse_args()
    camera_indices = tuple(
        int(value.strip()) for value in args.camera_indices.split(",") if value.strip()
    )
    if not camera_indices or any(index < 0 or index > 3 for index in camera_indices):
        raise ValueError(f"Invalid --camera-indices={args.camera_indices}")

    # Explicit identity selection must not depend on a preceding linspace
    # subsample; otherwise a valid requested training clip can disappear.
    with args.index.open(encoding="utf-8") as handle:
        rows = [
            row for row in csv.DictReader(handle)
            if row["split"] == args.split and row["source_camera"] == "cam0"
            and row["pair_id"] == args.pair_id
        ]
    if len(rows) != 1:
        raise ValueError(f"Expected one selected pair, found {len(rows)} for {args.pair_id}")
    row = rows[0]
    frames = np.rint(
        np.linspace(int(row["start_frame"]), int(row["end_frame"]), args.frames_per_clip)
    ).astype(np.int32)
    sequence_root = Path(row["source_rgb_dir"]).parents[1]
    camera_roots = [sequence_root / f"cam{index}" for index in camera_indices]
    points_world = np.full((len(frames), len(ARM_LANDMARKS), 3), np.nan, dtype=np.float32)
    confidence = np.zeros((len(frames), len(ARM_LANDMARKS)), dtype=np.float32)
    detected_views = np.zeros(len(frames), dtype=np.int32)
    # Low-resolution source-view person probabilities let the renderer reject
    # table RGB-D that is metrically close to a resting forearm.  Keep the
    # camera dimension explicit so reduced-input protocols cannot borrow masks
    # from withheld views.
    person_masks = np.zeros(
        (len(frames), len(camera_roots), 256, 256), dtype=np.uint8
    )
    options = vision.PoseLandmarkerOptions(
        base_options=python.BaseOptions(model_asset_path=str(args.pose_model)),
        running_mode=vision.RunningMode.IMAGE,
        num_poses=1,
        min_pose_detection_confidence=0.1,
        min_pose_presence_confidence=0.1,
        min_tracking_confidence=0.1,
        output_segmentation_masks=True,
    )
    with vision.PoseLandmarker.create_from_options(options) as detector:
        for frame_index, frame in enumerate(frames):
            observations = [[] for _ in ARM_LANDMARKS]
            for camera_output_index, camera_root in enumerate(camera_roots):
                stem = f"{int(frame):06d}"
                image = np.asarray(Image.open(camera_root / "rgb" / f"{stem}.png").convert("RGB"))
                result = detector.detect(mp.Image(image_format=mp.ImageFormat.SRGB, data=image))
                if not result.pose_landmarks:
                    continue
                detected_views[frame_index] += 1
                if result.segmentation_masks:
                    probability = np.squeeze(
                        result.segmentation_masks[0].numpy_view()
                    )
                    probability_u8 = np.rint(
                        np.clip(probability, 0.0, 1.0) * 255.0
                    ).astype(np.uint8)
                    person_masks[frame_index, camera_output_index] = np.asarray(
                        Image.fromarray(probability_u8, "L").resize(
                            (256, 256), Image.Resampling.BILINEAR
                        ),
                        dtype=np.uint8,
                    )
                depth = np.asarray(Image.open(camera_root / "depth" / f"{stem}.png"))
                intrinsics = load_intrinsics(camera_root / "cam_intrinsics.txt")
                pose = load_pose(camera_root / "cam_pose" / f"{stem}.txt")
                landmarks = result.pose_landmarks[0]
                for output_index, landmark_index in enumerate(ARM_LANDMARKS):
                    landmark = landmarks[landmark_index]
                    visibility = landmark.visibility if landmark.visibility is not None else 1.0
                    presence = landmark.presence if landmark.presence is not None else 1.0
                    if min(visibility, presence) < 0.1:
                        continue
                    point = depth_point_world(
                        SimpleNamespace(x=landmark.x, y=landmark.y),
                        depth, intrinsics, pose, args.depth_radius,
                    )
                    if point is not None:
                        observations[output_index].append(point)
            for landmark_index, values in enumerate(observations):
                if values:
                    points_world[frame_index, landmark_index] = np.median(
                        np.stack(values), axis=0
                    )
                    confidence[frame_index, landmark_index] = len(values) / len(camera_roots)
            print(
                json.dumps({
                    "frame": int(frame),
                    "views": int(detected_views[frame_index]),
                    "arm_points": int(np.isfinite(points_world[frame_index]).all(axis=1).sum()),
                }),
                flush=True,
            )
    destination = args.output_root / row["sequence"]
    destination.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        destination / "arm_state.npz",
        frames=frames,
        landmark_indices=np.asarray(ARM_LANDMARKS, dtype=np.int32),
        arm_points_world_m=points_world,
        confidence=confidence,
        detected_views=detected_views,
        camera_indices=np.asarray(camera_indices, dtype=np.int32),
        person_masks_256=person_masks,
    )
    print(destination / "arm_state.npz")


if __name__ == "__main__":
    main()
