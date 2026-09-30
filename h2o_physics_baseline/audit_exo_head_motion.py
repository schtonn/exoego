#!/usr/bin/env python3
"""Estimate ego-camera motion from pose landmarks observed by four exo cameras.

MediaPipe sees only the four exo RGB streams.  Head landmarks are triangulated
with known calibration, a rigid transform is fitted relative to the first
frame, and that transform moves the known initial ego-camera pose.  Future
cam4 poses are used only to report error and a static-head baseline.
"""

from __future__ import annotations

import argparse
import csv
import json
from collections import defaultdict, OrderedDict
from pathlib import Path
import sys

import mediapipe as mp
import numpy as np
from mediapipe.tasks import python
from mediapipe.tasks.python import vision

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from h2o_geometric_baseline.mediapipe_multiview_audit import (
    frame_projection_matrix,
    robust_triangulate,
)
from h2o_geometric_baseline.reprojection import load_intrinsics, load_pose


# Nose, eye centers represented by outer-eye landmarks, and ears.
HEAD_LANDMARKS = (0, 2, 5, 7, 8)


def rotation_angle_deg(rotation: np.ndarray) -> float:
    cosine = np.clip((np.trace(rotation) - 1.0) * 0.5, -1.0, 1.0)
    return float(np.degrees(np.arccos(cosine)))


def rigid_transform(source: np.ndarray, target: np.ndarray) -> np.ndarray:
    source_center = source.mean(axis=0)
    target_center = target.mean(axis=0)
    u, _, vt = np.linalg.svd((source - source_center).T @ (target - target_center))
    rotation = vt.T @ u.T
    if np.linalg.det(rotation) < 0:
        vt[-1] *= -1
        rotation = vt.T @ u.T
    transform = np.eye(4, dtype=np.float64)
    transform[:3, :3] = rotation
    transform[:3, 3] = target_center - rotation @ source_center
    return transform


def scaled_rotation(rotation: np.ndarray, scale: float) -> np.ndarray:
    """Project a linear identity/rotation blend back onto SO(3)."""
    u, _, vt = np.linalg.svd((1.0 - scale) * np.eye(3) + scale * rotation)
    result = u @ vt
    if np.linalg.det(result) < 0:
        u[:, -1] *= -1
        result = u @ vt
    return result


def describe(values: list[float]) -> dict[str, float | int | None]:
    if not values:
        return {"count": 0, "mean": None, "median": None, "p90": None, "max": None}
    array = np.asarray(values, dtype=np.float64)
    return {
        "count": len(values),
        "mean": float(array.mean()),
        "median": float(np.median(array)),
        "p90": float(np.percentile(array, 90)),
        "max": float(array.max()),
    }


def load_combined_rows(
    index_path: Path, max_samples: int, split: str = "val"
) -> list[dict[str, str]]:
    with index_path.open(encoding="utf-8") as handle:
        rows = [row for row in csv.DictReader(handle) if row["split"] == split]
    grouped: OrderedDict[str, list[dict[str, str]]] = OrderedDict()
    for row in rows:
        grouped.setdefault(row["clip_id"], []).append(row)
    combined = []
    cameras = ("cam0", "cam1", "cam2", "cam3")
    for clip_rows in grouped.values():
        by_camera = {row["source_camera"]: row for row in clip_rows}
        if not all(camera in by_camera for camera in cameras):
            continue
        representative = dict(by_camera["cam0"])
        representative["source_rgb_dirs"] = json.dumps(
            [by_camera[camera]["source_rgb_dir"] for camera in cameras]
        )
        combined.append(representative)
    if max_samples < len(combined):
        selected = np.linspace(0, len(combined) - 1, max_samples, dtype=np.int64)
        combined = [combined[int(index)] for index in selected]
    return combined


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--index",
        type=Path,
        default=Path("datasets/H2O/oracle_state/paired_physical_clips.csv"),
    )
    parser.add_argument(
        "--model", type=Path, default=Path("models/mediapipe/pose_landmarker_full.task")
    )
    parser.add_argument("--max-samples", type=int, default=32)
    parser.add_argument("--frames-per-clip", type=int, default=4)
    parser.add_argument("--visibility", type=float, default=0.1)
    parser.add_argument("--ransac-threshold-px", type=float, default=30.0)
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("datasets/H2O/experiments/exo_head_motion/summary.json"),
    )
    args = parser.parse_args()

    rows = load_combined_rows(args.index, args.max_samples)
    options = vision.PoseLandmarkerOptions(
        base_options=python.BaseOptions(model_asset_path=str(args.model)),
        running_mode=vision.RunningMode.IMAGE,
        num_poses=1,
        min_pose_detection_confidence=0.1,
        min_pose_presence_confidence=0.1,
        min_tracking_confidence=0.1,
        output_segmentation_masks=False,
    )
    counters = defaultdict(int)
    translation_errors = []
    rotation_errors = []
    static_translation_errors = []
    static_rotation_errors = []
    motion_scales = (0.0, 0.25, 0.5, 0.75, 1.0)
    translation_scale_errors = {scale: [] for scale in motion_scales}
    rotation_scale_errors = {scale: [] for scale in motion_scales}
    per_clip = []

    with vision.PoseLandmarker.create_from_options(options) as detector:
        for clip_index, row in enumerate(rows):
            frames = np.rint(
                np.linspace(int(row["start_frame"]), int(row["end_frame"]), args.frames_per_clip)
            ).astype(np.int64)
            camera_roots = [Path(value).parent for value in json.loads(row["source_rgb_dirs"])]
            target_root = Path(row["target_rgb_dir"]).parent
            triangulated_frames: list[dict[int, np.ndarray]] = []
            views_per_frame = []
            for frame in frames:
                stem = f"{int(frame):06d}"
                observations: dict[int, list[tuple[np.ndarray, np.ndarray]]] = {
                    index: [] for index in HEAD_LANDMARKS
                }
                detected_views = 0
                for camera_root in camera_roots:
                    counters["camera_images"] += 1
                    result = detector.detect(
                        mp.Image.create_from_file(str(camera_root / "rgb" / f"{stem}.png"))
                    )
                    if not result.pose_landmarks:
                        continue
                    detected_views += 1
                    counters["camera_images_detected"] += 1
                    landmarks = result.pose_landmarks[0]
                    intrinsics = load_intrinsics(camera_root / "cam_intrinsics.txt")
                    width, height = intrinsics[4:6]
                    projection = frame_projection_matrix(camera_root, stem)
                    for landmark_index in HEAD_LANDMARKS:
                        landmark = landmarks[landmark_index]
                        visibility = landmark.visibility if landmark.visibility is not None else 1.0
                        presence = landmark.presence if landmark.presence is not None else 1.0
                        if min(visibility, presence) < args.visibility:
                            continue
                        uv = np.asarray((landmark.x * width, landmark.y * height), dtype=np.float64)
                        if 0 <= uv[0] < width and 0 <= uv[1] < height:
                            observations[landmark_index].append((uv, projection))
                points = {}
                for landmark_index, landmark_observations in observations.items():
                    if len(landmark_observations) < 2:
                        continue
                    point, inliers = robust_triangulate(
                        landmark_observations, args.ransac_threshold_px
                    )
                    if np.isfinite(point).all() and inliers >= 2:
                        points[landmark_index] = point
                        counters["triangulated_head_landmarks"] += 1
                if len(points) >= 3:
                    counters["frames_with_rigid_head"] += 1
                triangulated_frames.append(points)
                views_per_frame.append(detected_views)

            initial_points = triangulated_frames[0]
            initial_pose = load_pose(target_root / "cam_pose" / f"{int(frames[0]):06d}.txt")
            clip_record = {
                "pair_id": row["pair_id"],
                "frames": frames.tolist(),
                "detected_exo_views": views_per_frame,
                "triangulated_head_landmarks": [len(points) for points in triangulated_frames],
                "future": [],
            }
            for time_index in range(1, len(frames)):
                current_points = triangulated_frames[time_index]
                common = sorted(set(initial_points) & set(current_points))
                gt_pose = load_pose(
                    target_root / "cam_pose" / f"{int(frames[time_index]):06d}.txt"
                )
                static_t = float(np.linalg.norm(initial_pose[:3, 3] - gt_pose[:3, 3]))
                static_r = rotation_angle_deg(initial_pose[:3, :3].T @ gt_pose[:3, :3])
                static_translation_errors.append(static_t)
                static_rotation_errors.append(static_r)
                record = {
                    "frame": int(frames[time_index]),
                    "common_landmarks": len(common),
                    "static_translation_error_m": static_t,
                    "static_rotation_error_deg": static_r,
                }
                if len(common) >= 3:
                    source = np.stack([initial_points[index] for index in common])
                    target = np.stack([current_points[index] for index in common])
                    predicted_pose = rigid_transform(source, target) @ initial_pose
                    translation_error = float(
                        np.linalg.norm(predicted_pose[:3, 3] - gt_pose[:3, 3])
                    )
                    rotation_error = rotation_angle_deg(
                        predicted_pose[:3, :3].T @ gt_pose[:3, :3]
                    )
                    translation_errors.append(translation_error)
                    rotation_errors.append(rotation_error)
                    estimated_delta_rotation = (
                        predicted_pose[:3, :3] @ initial_pose[:3, :3].T
                    )
                    estimated_delta_position = predicted_pose[:3, 3] - initial_pose[:3, 3]
                    for scale in motion_scales:
                        scaled_position = initial_pose[:3, 3] + scale * estimated_delta_position
                        scaled_orientation = (
                            scaled_rotation(estimated_delta_rotation, scale)
                            @ initial_pose[:3, :3]
                        )
                        translation_scale_errors[scale].append(
                            float(np.linalg.norm(scaled_position - gt_pose[:3, 3]))
                        )
                        rotation_scale_errors[scale].append(
                            rotation_angle_deg(scaled_orientation.T @ gt_pose[:3, :3])
                        )
                    counters["future_frames_estimated"] += 1
                    record.update(
                        {
                            "estimated_translation_error_m": translation_error,
                            "estimated_rotation_error_deg": rotation_error,
                            "estimated_camera_delta_position_world_m": estimated_delta_position.tolist(),
                            "estimated_delta_rotation_world": estimated_delta_rotation.tolist(),
                        }
                    )
                clip_record["future"].append(record)
            per_clip.append(clip_record)
            print(
                json.dumps(
                    {
                        "clip": clip_index,
                        "pair_id": row["pair_id"],
                        "views": views_per_frame,
                        "points": clip_record["triangulated_head_landmarks"],
                        "estimated": sum("estimated_translation_error_m" in x for x in clip_record["future"]),
                    },
                    ensure_ascii=False,
                ),
                flush=True,
            )

    result = {
        "protocol": "four exo RGB + calibration + initial ego pose; future ego pose evaluation-only",
        "model": str(args.model),
        "samples": len(rows),
        "camera_detection_fraction": counters["camera_images_detected"] / max(counters["camera_images"], 1),
        "rigid_head_frame_fraction": counters["frames_with_rigid_head"] / max(len(rows) * args.frames_per_clip, 1),
        "future_pose_estimate_fraction": counters["future_frames_estimated"] / max(len(rows) * (args.frames_per_clip - 1), 1),
        "estimated_pose_error": {
            "translation_m": describe(translation_errors),
            "rotation_deg": describe(rotation_errors),
        },
        "static_initial_pose_error": {
            "translation_m": describe(static_translation_errors),
            "rotation_deg": describe(static_rotation_errors),
        },
        "motion_scale_sweep": {
            str(scale): {
                "translation_m": describe(translation_scale_errors[scale]),
                "rotation_deg": describe(rotation_scale_errors[scale]),
            }
            for scale in motion_scales
        },
        "counters": dict(counters),
        "per_clip": per_clip,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print(json.dumps(result | {"per_clip": "omitted"}, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
