#!/usr/bin/env python3
"""Precompute deployable multi-exo hand state for selected H2O clips.

Four RGB views are detected independently with MediaPipe.  The highest-score
Left/Right detection in each view is associated by handedness, then calibrated
RANSAC triangulation produces world-space joints.  No cam4 frames, future cam4
poses, or hand/object annotations are used to form the estimates.  Ground truth
is loaded only after estimation to report state error.
"""

from __future__ import annotations

import argparse
import csv
import json
from collections import defaultdict
from pathlib import Path

import mediapipe as mp
import numpy as np
from mediapipe.tasks import python
from mediapipe.tasks.python import vision
from PIL import Image

from h2o_geometric_baseline.mediapipe_multiview_audit import (
    CAMERAS,
    HAND_NAMES,
    frame_projection_matrix,
    reproject,
    robust_triangulate,
)
from h2o_geometric_baseline.reprojection import load_intrinsics, load_pose


DEFAULT_INDEX = Path("datasets/H2O/oracle_state/paired_physical_clips.csv")
DEFAULT_MODEL = Path("models/mediapipe/hand_landmarker.task")
DEFAULT_OUTPUT = Path("datasets/H2O/student_state_mediapipe_64_32")


def selected_rows(index: Path, split: str, maximum: int) -> list[dict[str, str]]:
    with index.open(encoding="utf-8") as handle:
        rows = [
            row
            for row in csv.DictReader(handle)
            if row["split"] == split and row["source_camera"] == "cam0"
        ]
    if maximum < len(rows):
        indices = np.linspace(0, len(rows) - 1, maximum, dtype=np.int64)
        rows = [rows[int(index)] for index in indices]
    return rows


def percentile(values: list[float], scale: float = 1.0) -> dict[str, float | int | None]:
    if not values:
        return {"count": 0, "mean": None, "median": None, "p95": None}
    array = np.asarray(values, dtype=np.float64) * scale
    return {
        "count": int(len(array)),
        "mean": float(array.mean()),
        "median": float(np.median(array)),
        "p95": float(np.percentile(array, 95)),
    }


def depth_point_world(
    uv: np.ndarray,
    depth_mm: np.ndarray,
    intrinsics: np.ndarray,
    camera_pose: np.ndarray,
    radius: int,
) -> np.ndarray | None:
    """Lift one RGB landmark with only the selected camera's depth."""
    height, width = depth_mm.shape
    x, y = np.rint(uv).astype(np.int64)
    if not (0 <= x < width and 0 <= y < height):
        return None
    patch = depth_mm[
        max(0, y - radius) : min(height, y + radius + 1),
        max(0, x - radius) : min(width, x + radius + 1),
    ].astype(np.float64) / 1000.0
    valid = patch[np.isfinite(patch) & (patch >= 0.1) & (patch <= 5.0)]
    if len(valid) < 3:
        return None
    z = float(np.median(valid))
    fx, fy, cx, cy = intrinsics[:4]
    camera_point = np.asarray(((x - cx) * z / fx, (y - cy) * z / fy, z))
    return camera_pose[:3, :3] @ camera_point + camera_pose[:3, 3]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--index", type=Path, default=DEFAULT_INDEX)
    parser.add_argument("--model", type=Path, default=DEFAULT_MODEL)
    parser.add_argument("--output-root", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--frames-per-clip", type=int, default=4)
    parser.add_argument("--pair-id", type=str, default=None)
    parser.add_argument("--max-train-samples", type=int, default=64)
    parser.add_argument("--max-val-samples", type=int, default=32)
    parser.add_argument("--min-detection-confidence", type=float, default=0.1)
    parser.add_argument("--min-presence-confidence", type=float, default=0.1)
    parser.add_argument("--ransac-threshold-px", type=float, default=25.0)
    parser.add_argument("--depth-radius", type=int, default=3)
    parser.add_argument(
        "--camera-indices",
        default="0,1,2,3",
        help="Comma-separated exo cameras. One camera uses its RGB-D depth lift.",
    )
    args = parser.parse_args()
    camera_indices = tuple(
        int(value.strip()) for value in args.camera_indices.split(",") if value.strip()
    )
    if not camera_indices or any(index < 0 or index >= len(CAMERAS) for index in camera_indices):
        raise ValueError(f"Invalid --camera-indices={args.camera_indices}")
    cameras = tuple(CAMERAS[index] for index in camera_indices)

    if args.pair_id is not None:
        # An explicit pair is an identity selection, not a request to search a
        # lossy linspace subsample.  This makes batch expansion reproducible.
        with args.index.open(encoding="utf-8") as handle:
            rows = [row for row in csv.DictReader(handle) if row["pair_id"] == args.pair_id]
        if not rows:
            raise ValueError(f"Pair id not found: {args.pair_id}")
    else:
        rows = selected_rows(args.index, "train", args.max_train_samples) + selected_rows(
            args.index, "val", args.max_val_samples
        )
    sequence_frames: dict[str, set[int]] = defaultdict(set)
    state_paths: dict[str, Path] = {}
    for row in rows:
        frames = np.rint(
            np.linspace(int(row["start_frame"]), int(row["end_frame"]), args.frames_per_clip)
        ).astype(np.int32)
        sequence_frames[row["sequence"]].update(int(frame) for frame in frames)
        state_paths[row["sequence"]] = Path(row["state_path"])

    options = vision.HandLandmarkerOptions(
        base_options=python.BaseOptions(model_asset_path=str(args.model)),
        running_mode=vision.RunningMode.IMAGE,
        num_hands=2,
        min_hand_detection_confidence=args.min_detection_confidence,
        min_hand_presence_confidence=args.min_presence_confidence,
        min_tracking_confidence=0.1,
    )
    mpjpe: list[float] = []
    center_errors: list[float] = []
    reprojection_errors: list[float] = []
    observation_counts: list[float] = []
    total_joints = 0
    estimated_joints = 0
    sequence_summaries = []

    with vision.HandLandmarker.create_from_options(options) as detector:
        for sequence in sorted(sequence_frames):
            frames = np.asarray(sorted(sequence_frames[sequence]), dtype=np.int32)
            joints_world = np.full((len(frames), 2, 21, 3), np.nan, dtype=np.float32)
            joint_confidence = np.zeros((len(frames), 2, 21), dtype=np.float32)
            reprojection_rmse = np.full((len(frames), 2, 21), np.nan, dtype=np.float32)
            sequence_root = Path("datasets/H2O/raw") / sequence
            sequence_detections = 0
            for frame_index, frame in enumerate(frames):
                stem = f"{int(frame):06d}"
                observations: list[list[list[tuple[np.ndarray, np.ndarray]]]] = [
                    [[] for _ in range(21)] for _ in range(2)
                ]
                depth_observations: list[list[list[np.ndarray]]] = [
                    [[] for _ in range(21)] for _ in range(2)
                ]
                for camera in cameras:
                    camera_root = sequence_root / camera
                    image_path = camera_root / "rgb" / f"{stem}.png"
                    if not image_path.exists():
                        continue
                    image = mp.Image.create_from_file(str(image_path))
                    result = detector.detect(image)
                    predictions = [
                        np.asarray(
                            [[point.x * image.width, point.y * image.height] for point in hand],
                            dtype=np.float64,
                        )
                        for hand in result.hand_landmarks
                    ]
                    labels = [categories[0].category_name for categories in result.handedness]
                    scores = [float(categories[0].score) for categories in result.handedness]
                    sequence_detections += len(predictions)
                    matrix = frame_projection_matrix(camera_root, stem)
                    depth = (
                        np.asarray(Image.open(camera_root / "depth" / f"{stem}.png"))
                        if len(cameras) == 1 else None
                    )
                    intrinsics = load_intrinsics(camera_root / "cam_intrinsics.txt")
                    camera_pose = load_pose(camera_root / "cam_pose" / f"{stem}.txt")
                    for hand_index, hand_name in enumerate(HAND_NAMES):
                        candidates = [index for index, label in enumerate(labels) if label == hand_name]
                        if not candidates:
                            continue
                        prediction = predictions[max(candidates, key=lambda index: scores[index])]
                        for joint_index, uv in enumerate(prediction):
                            observations[hand_index][joint_index].append((uv, matrix))
                            if depth is not None:
                                point = depth_point_world(
                                    uv, depth, intrinsics, camera_pose, args.depth_radius
                                )
                                if point is not None:
                                    depth_observations[hand_index][joint_index].append(point)

                for hand_index in range(2):
                    for joint_index in range(21):
                        values = observations[hand_index][joint_index]
                        total_joints += 1
                        if len(cameras) == 1:
                            lifted = depth_observations[hand_index][joint_index]
                            if not lifted:
                                continue
                            estimate = np.median(np.stack(lifted), axis=0)
                            inliers = len(lifted)
                        else:
                            if len(values) < 2:
                                continue
                            estimate, inliers = robust_triangulate(
                                values, args.ransac_threshold_px
                            )
                        if not np.isfinite(estimate).all():
                            continue
                        errors = [
                            float(np.linalg.norm(reproject(estimate, matrix) - uv))
                            for uv, matrix in values
                        ]
                        joints_world[frame_index, hand_index, joint_index] = estimate
                        joint_confidence[frame_index, hand_index, joint_index] = inliers / len(cameras)
                        reprojection_rmse[frame_index, hand_index, joint_index] = float(
                            np.sqrt(np.mean(np.square(errors)))
                        )
                        estimated_joints += 1
                        observation_counts.append(float(len(values)))
                        reprojection_errors.append(reprojection_rmse[frame_index, hand_index, joint_index])

            hand_presence = np.isfinite(joints_world).all(axis=-1).mean(axis=-1) >= 0.5
            destination = args.output_root / sequence
            destination.mkdir(parents=True, exist_ok=True)
            np.savez_compressed(
                destination / "student_state.npz",
                frames=frames,
                hand_joints_world_m=joints_world,
                hand_presence=hand_presence,
                joint_confidence=joint_confidence,
                reprojection_rmse_px=reprojection_rmse,
            )

            with np.load(state_paths[sequence]) as oracle:
                oracle_indices = np.searchsorted(oracle["frames"], frames)
                truth = oracle["hand_joints_world_m"][oracle_indices]
                truth_presence = oracle["hand_presence"][oracle_indices]
                finite = np.isfinite(joints_world).all(axis=-1) & truth_presence[:, :, None]
                errors = np.linalg.norm(joints_world - truth, axis=-1)
                mpjpe.extend(errors[finite].tolist())
                for frame_index in range(len(frames)):
                    for hand_index in range(2):
                        valid = finite[frame_index, hand_index]
                        if int(valid.sum()) >= 11:
                            predicted_center = joints_world[frame_index, hand_index, valid].mean(axis=0)
                            target_center = truth[frame_index, hand_index, valid].mean(axis=0)
                            center_errors.append(float(np.linalg.norm(predicted_center - target_center)))
            sequence_summaries.append(
                {
                    "sequence": sequence,
                    "frames": int(len(frames)),
                    "rgb_detections": int(sequence_detections),
                    "joint_coverage": float(np.isfinite(joints_world).all(axis=-1).mean()),
                }
            )

    report = {
        "configuration": {
            "frames_per_clip": args.frames_per_clip,
            "max_train_samples": args.max_train_samples,
            "max_val_samples": args.max_val_samples,
            "selected_clips": len(rows),
            "sequences": len(sequence_frames),
            "uses_cam4_for_estimation": False,
            "uses_ground_truth_for_estimation": False,
            "camera_indices": list(camera_indices),
            "camera_names": list(cameras),
            "single_camera_3d": "RGB-D depth lift" if len(cameras) == 1 else None,
            "association": "MediaPipe handedness, highest confidence per side",
            "object_state": "unavailable",
        },
        "total_joints": total_joints,
        "estimated_joints": estimated_joints,
        "joint_coverage": estimated_joints / max(total_joints, 1),
        "observing_exo_views": percentile(observation_counts),
        "reprojection_rmse_px": percentile(reprojection_errors),
        "mpjpe_mm_against_withheld_ground_truth": percentile(mpjpe, 1000.0),
        "hand_center_error_mm": percentile(center_errors, 1000.0),
        "sequence_summaries": sequence_summaries,
    }
    args.output_root.mkdir(parents=True, exist_ok=True)
    (args.output_root / "summary.json").write_text(
        json.dumps(report, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
