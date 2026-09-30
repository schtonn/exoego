#!/usr/bin/env python3
"""Estimate ego motion from dense exo face landmarks sampled on exo depth.

Only four exo RGB-D streams, calibration, and the initial ego pose are inputs.
Future ego poses are loaded exclusively for evaluation.
"""

from __future__ import annotations

import argparse
import itertools
import json
from collections import defaultdict
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
SCRIPT_ROOT = Path(__file__).resolve().parent
if str(SCRIPT_ROOT) not in sys.path:
    sys.path.insert(0, str(SCRIPT_ROOT))

from h2o_geometric_baseline.reprojection import load_intrinsics, load_pose
from audit_exo_head_motion import (
    describe,
    load_combined_rows,
    rigid_transform,
    rotation_angle_deg,
    scaled_rotation,
)


# Stable facial outline / eye / nose landmarks; mouth points are deliberately omitted.
FACE_LANDMARKS = (1, 10, 33, 133, 152, 234, 263, 362, 454)
# Selected once on the three-scene validation audit.  Larger inferred motion
# has measurably larger target-camera pose error even when face-fit residuals
# stay small, so it must lower transport confidence rather than being treated
# as equally trustworthy.
MOTION_ROTATION_RISK_SCALE_DEG = 10.0
MOTION_TRANSLATION_RISK_SCALE_M = 0.01


def temporal_smooth(values: np.ndarray, radius: int = 4) -> np.ndarray:
    """Mirror the deployed render trajectory smoothing without importing Torch."""
    flat = values.astype(np.float64).reshape(len(values), -1).copy()
    for channel in range(flat.shape[1]):
        series = flat[:, channel]
        finite = np.isfinite(series)
        if not np.any(finite):
            series[:] = 0
        elif not np.all(finite):
            series[~finite] = np.interp(
                np.flatnonzero(~finite), np.flatnonzero(finite), series[finite]
            )
        robust = np.asarray([
            np.median(series[max(0, index - radius) : min(len(series), index + radius + 1)])
            for index in range(len(series))
        ])
        weights = np.arange(1, radius + 2, dtype=np.float64)
        weights = np.concatenate((weights, weights[-2::-1]))
        flat[:, channel] = np.convolve(
            np.pad(robust, radius, mode="edge"), weights / weights.sum(), mode="valid"
        )
    return flat.reshape(values.shape).astype(np.float32)


def deployed_pose_map(
    frames: list[int], initial_pose: np.ndarray, clip_record: dict,
    rotation_scale: float,
) -> dict[int, np.ndarray]:
    records = {int(value["frame"]): value for value in clip_record["future"]}
    translations = [np.zeros(3, dtype=np.float64)]
    rotations = [np.eye(3, dtype=np.float64)]
    for frame in frames[1:]:
        motion = records[frame]
        if "estimated_head_transform_world" in motion:
            moved = np.asarray(motion["estimated_head_transform_world"]) @ initial_pose
            translations.append(moved[:3, 3] - initial_pose[:3, 3])
            rotations.append(moved[:3, :3] @ initial_pose[:3, :3].T)
        else:
            translations.append(np.full(3, np.nan))
            rotations.append(np.full((3, 3), np.nan))
    translations_array = temporal_smooth(np.stack(translations), radius=4)
    rotations_array = temporal_smooth(np.stack(rotations), radius=4)
    translations_array -= translations_array[0]
    result = {}
    for index, (frame, translation, matrix) in enumerate(
        zip(frames, translations_array, rotations_array)
    ):
        u, _, vt = np.linalg.svd(matrix)
        rotation = u @ vt
        if np.linalg.det(rotation) < 0:
            u[:, -1] *= -1
            rotation = u @ vt
        if index == 0:
            rotation = np.eye(3)
        pose = initial_pose.copy()
        pose[:3, 3] += translation
        pose[:3, :3] = scaled_rotation(rotation, rotation_scale) @ initial_pose[:3, :3]
        result[frame] = pose
    return result


def confidence_map(frames: list[int], clip_record: dict) -> dict[int, float]:
    records = {int(value["frame"]): value for value in clip_record["future"]}
    result = {frames[0]: 1.0}
    for frame in frames[1:]:
        record = records[frame]
        result[frame] = float(np.clip(record.get("pose_confidence", 0.0), 0.0, 1.0))
    return result


def average_ranks(values: np.ndarray) -> np.ndarray:
    """Ranks with averaged ties, sufficient for a dependency-free Spearman audit."""
    order = np.argsort(values, kind="mergesort")
    ranks = np.empty(len(values), dtype=np.float64)
    start = 0
    while start < len(values):
        end = start + 1
        while end < len(values) and values[order[end]] == values[order[start]]:
            end += 1
        ranks[order[start:end]] = 0.5 * (start + end - 1)
        start = end
    return ranks


def spearman(values: list[float], targets: list[float]) -> float | None:
    if len(values) < 3:
        return None
    x = average_ranks(np.asarray(values, dtype=np.float64))
    y = average_ranks(np.asarray(targets, dtype=np.float64))
    if x.std() <= 1e-12 or y.std() <= 1e-12:
        return None
    return float(np.corrcoef(x, y)[0, 1])


def isotonic_knots(
    values: list[float], targets: list[float], *, increasing: bool
) -> dict[str, list[float]]:
    """Fit a one-dimensional weighted PAV calibration and return interpolation knots."""
    x = np.asarray(values, dtype=np.float64)
    y = np.asarray(targets, dtype=np.float64)
    unique_x, inverse, counts = np.unique(x, return_inverse=True, return_counts=True)
    grouped = np.zeros(len(unique_x), dtype=np.float64)
    np.add.at(grouped, inverse, y)
    grouped /= counts
    working = grouped if increasing else -grouped
    blocks: list[list[float | int]] = []
    for index, (mean, weight) in enumerate(zip(working, counts)):
        blocks.append([index, index + 1, float(weight), float(mean)])
        while len(blocks) >= 2 and blocks[-2][3] > blocks[-1][3]:
            left, right = blocks[-2], blocks[-1]
            total_weight = float(left[2]) + float(right[2])
            total_mean = (
                float(left[3]) * float(left[2])
                + float(right[3]) * float(right[2])
            ) / total_weight
            blocks[-2:] = [[int(left[0]), int(right[1]), total_weight, total_mean]]
    fitted = np.empty(len(unique_x), dtype=np.float64)
    for start, end, _, mean in blocks:
        fitted[int(start) : int(end)] = float(mean)
    if not increasing:
        fitted *= -1.0
    return {"x": unique_x.tolist(), "y": fitted.tolist()}


def pose_guided_crop(
    image: np.ndarray, pose_result: object, minimum_side: int = 192
) -> tuple[np.ndarray, tuple[int, int, int, int]] | None:
    if not pose_result.pose_landmarks:
        return None
    height, width = image.shape[:2]
    landmarks = pose_result.pose_landmarks[0]
    points = []
    for index in range(11):
        landmark = landmarks[index]
        visibility = landmark.visibility if landmark.visibility is not None else 1.0
        presence = landmark.presence if landmark.presence is not None else 1.0
        if min(visibility, presence) >= 0.1 and 0 <= landmark.x <= 1 and 0 <= landmark.y <= 1:
            points.append((landmark.x * width, landmark.y * height))
    if len(points) < 3:
        return None
    points = np.asarray(points)
    center = points.mean(axis=0)
    spread = float(np.max(points.max(axis=0) - points.min(axis=0)))
    side = int(round(max(minimum_side, spread * 3.0)))
    x0 = max(0, int(round(center[0] - side / 2)))
    y0 = max(0, int(round(center[1] - side / 2)))
    x1 = min(width, x0 + side)
    y1 = min(height, y0 + side)
    x0 = max(0, x1 - side)
    y0 = max(0, y1 - side)
    if x1 - x0 < 64 or y1 - y0 < 64:
        return None
    return np.ascontiguousarray(image[y0:y1, x0:x1]), (x0, y0, x1, y1)


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
    x0, x1 = max(0, x - radius), min(width, x + radius + 1)
    y0, y1 = max(0, y - radius), min(height, y + radius + 1)
    patch = depth_mm[y0:y1, x0:x1].astype(np.float64) / 1000.0
    values = patch[np.isfinite(patch) & (patch >= 0.1) & (patch <= 5.0)]
    if len(values) < 3:
        return None
    z = float(np.median(values))
    fx, fy, cx, cy = intrinsics[:4]
    camera_point = np.asarray(((x - cx) * z / fx, (y - cy) * z / fy, z))
    return camera_pose[:3, :3] @ camera_point + camera_pose[:3, 3]


def robust_face_transform(
    source: np.ndarray, target: np.ndarray, threshold_m: float
) -> tuple[np.ndarray, np.ndarray]:
    if len(source) < 3:
        return np.eye(4), np.zeros(len(source), dtype=bool)
    best = np.zeros(len(source), dtype=bool)
    best_error = np.inf
    for subset in itertools.combinations(range(len(source)), 3):
        transform = rigid_transform(source[list(subset)], target[list(subset)])
        predicted = source @ transform[:3, :3].T + transform[:3, 3]
        errors = np.linalg.norm(predicted - target, axis=1)
        inliers = errors <= threshold_m
        score = float(errors[inliers].mean()) if np.any(inliers) else np.inf
        if inliers.sum() > best.sum() or (inliers.sum() == best.sum() and score < best_error):
            best, best_error = inliers, score
    if best.sum() < 3:
        return np.eye(4), best
    return rigid_transform(source[best], target[best]), best


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--index",
        type=Path,
        default=Path("datasets/H2O/oracle_state/paired_physical_clips.csv"),
    )
    parser.add_argument(
        "--model", type=Path, default=Path("models/mediapipe/face_landmarker.task")
    )
    parser.add_argument(
        "--pose-model", type=Path, default=Path("models/mediapipe/pose_landmarker_full.task")
    )
    parser.add_argument("--max-samples", type=int, default=32)
    parser.add_argument("--split", choices=("train", "val", "test"), default="val")
    parser.add_argument("--frames-per-clip", type=int, default=4)
    parser.add_argument("--pair-id", action="append", dest="pair_ids")
    parser.add_argument("--depth-radius", type=int, default=3)
    parser.add_argument("--ransac-threshold-m", type=float, default=0.025)
    parser.add_argument("--camera-indices", default="0,1,2,3")
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("datasets/H2O/experiments/exo_face_head_motion/summary.json"),
    )
    parser.add_argument(
        "--audit-output", type=Path,
        help=(
            "Optional GT-only evaluation/calibration JSON. Future ego poses are "
            "not read at all unless this argument is supplied."
        ),
    )
    parser.add_argument("--deployment-rotation-scale", type=float, default=0.5)
    parser.add_argument("--good-rotation-deg", type=float, default=2.0)
    parser.add_argument("--good-translation-m", type=float, default=0.02)
    args = parser.parse_args()
    camera_indices = tuple(
        int(value.strip()) for value in args.camera_indices.split(",") if value.strip()
    )
    if not camera_indices or any(index < 0 or index > 3 for index in camera_indices):
        raise ValueError(f"Invalid --camera-indices={args.camera_indices}")

    rows = load_combined_rows(
        args.index,
        (10**9 if args.pair_ids is not None else args.max_samples),
        split=args.split,
    )
    if args.pair_ids is not None:
        requested = set(args.pair_ids)
        rows = [row for row in rows if row["pair_id"] in requested]
        if not rows:
            raise ValueError(
                f"Pair ids not found in split={args.split}: {sorted(requested)}"
            )
        missing = requested - {row["pair_id"] for row in rows}
        if missing:
            raise ValueError(f"Pair ids missing in split={args.split}: {sorted(missing)}")
    options = vision.FaceLandmarkerOptions(
        base_options=python.BaseOptions(model_asset_path=str(args.model)),
        running_mode=vision.RunningMode.IMAGE,
        num_faces=1,
        min_face_detection_confidence=0.1,
        min_face_presence_confidence=0.1,
        min_tracking_confidence=0.1,
        output_face_blendshapes=False,
        output_facial_transformation_matrixes=False,
    )
    pose_options = vision.PoseLandmarkerOptions(
        base_options=python.BaseOptions(model_asset_path=str(args.pose_model)),
        running_mode=vision.RunningMode.IMAGE,
        num_poses=1,
        min_pose_detection_confidence=0.1,
        min_pose_presence_confidence=0.1,
        min_tracking_confidence=0.1,
        output_segmentation_masks=False,
    )
    counters = defaultdict(int)
    scales = (0.0, 0.25, 0.5, 0.75, 1.0)
    rotation_scale_errors = {scale: [] for scale in scales}
    audit_samples: list[dict[str, float | int | str]] = []
    inlier_counts = []
    per_clip = []

    with (
        vision.FaceLandmarker.create_from_options(options) as detector,
        vision.PoseLandmarker.create_from_options(pose_options) as pose_detector,
    ):
        for clip_index, row in enumerate(rows):
            frames = np.rint(
                np.linspace(int(row["start_frame"]), int(row["end_frame"]), args.frames_per_clip)
            ).astype(np.int64)
            all_camera_roots = [
                Path(value).parent for value in json.loads(row["source_rgb_dirs"])
            ]
            camera_roots = [all_camera_roots[index] for index in camera_indices]
            target_root = Path(row["target_rgb_dir"]).parent
            points_by_frame = []
            detected_views = []
            for frame in frames:
                stem = f"{int(frame):06d}"
                observations = {index: [] for index in FACE_LANDMARKS}
                detections = 0
                for camera_root in camera_roots:
                    counters["camera_images"] += 1
                    image_path = camera_root / "rgb" / f"{stem}.png"
                    image = np.asarray(Image.open(image_path).convert("RGB"))
                    mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=image)
                    pose_result = pose_detector.detect(mp_image)
                    crop = pose_guided_crop(image, pose_result)
                    result = None
                    crop_box = (0, 0, image.shape[1], image.shape[0])
                    if crop is not None:
                        crop_image, crop_box = crop
                        result = detector.detect(
                            mp.Image(image_format=mp.ImageFormat.SRGB, data=crop_image)
                        )
                        counters["pose_guided_crops"] += 1
                    if result is None or not result.face_landmarks:
                        result = detector.detect(mp_image)
                        crop_box = (0, 0, image.shape[1], image.shape[0])
                    if not result.face_landmarks:
                        continue
                    detections += 1
                    counters["camera_images_detected"] += 1
                    landmarks = result.face_landmarks[0]
                    depth = np.asarray(Image.open(camera_root / "depth" / f"{stem}.png"))
                    intrinsics = load_intrinsics(camera_root / "cam_intrinsics.txt")
                    pose = load_pose(camera_root / "cam_pose" / f"{stem}.txt")
                    x0, y0, x1, y1 = crop_box
                    crop_width, crop_height = x1 - x0, y1 - y0
                    for landmark_index in FACE_LANDMARKS:
                        crop_landmark = landmarks[landmark_index]
                        landmark = SimpleNamespace(
                            x=(x0 + crop_landmark.x * crop_width) / image.shape[1],
                            y=(y0 + crop_landmark.y * crop_height) / image.shape[0],
                        )
                        point = depth_point_world(
                            landmark, depth, intrinsics, pose, args.depth_radius
                        )
                        if point is not None:
                            observations[landmark_index].append(point)
                            counters["valid_depth_observations"] += 1
                aggregated = {
                    index: np.median(np.stack(values), axis=0)
                    for index, values in observations.items()
                    if len(values) >= 1
                }
                if len(aggregated) >= 3:
                    counters["frames_with_face_cloud"] += 1
                points_by_frame.append(aggregated)
                detected_views.append(detections)

            initial_points = points_by_frame[0]
            initial_pose = load_pose(target_root / "cam_pose" / f"{int(frames[0]):06d}.txt")
            clip_record = {
                "pair_id": row["pair_id"],
                "frames": frames.tolist(),
                "detected_exo_views": detected_views,
                "face_points": [len(points) for points in points_by_frame],
                "future": [],
            }
            for time_index in range(1, len(frames)):
                current_points = points_by_frame[time_index]
                common = sorted(set(initial_points) & set(current_points))
                record = {"frame": int(frames[time_index]), "common_landmarks": len(common)}
                if len(common) >= 3:
                    source = np.stack([initial_points[index] for index in common])
                    target = np.stack([current_points[index] for index in common])
                    head_transform, inliers = robust_face_transform(
                        source, target, args.ransac_threshold_m
                    )
                    if inliers.sum() >= 3:
                        counters["future_frames_estimated"] += 1
                        inlier_counts.append(int(inliers.sum()))
                        full_pose = head_transform @ initial_pose
                        delta_position = full_pose[:3, 3] - initial_pose[:3, 3]
                        delta_rotation = full_pose[:3, :3] @ initial_pose[:3, :3].T
                        predicted = (
                            source @ head_transform[:3, :3].T
                            + head_transform[:3, 3]
                        )
                        inlier_residuals = np.linalg.norm(
                            predicted[inliers] - target[inliers], axis=1
                        )
                        inlier_ratio = float(inliers.sum() / len(common))
                        view_factor = min(1.0, detected_views[time_index] / 2.0)
                        residual_factor = float(np.exp(
                            -float(inlier_residuals.mean())
                            / max(args.ransac_threshold_m, 1e-6)
                        ))
                        geometric_confidence = (
                            inlier_ratio * view_factor * residual_factor
                        )
                        motion_rotation_deg = rotation_angle_deg(
                            head_transform[:3, :3]
                        )
                        motion_translation_m = float(np.linalg.norm(delta_position))
                        motion_risk_factor = float(np.exp(
                            -motion_rotation_deg / MOTION_ROTATION_RISK_SCALE_DEG
                            -motion_translation_m / MOTION_TRANSLATION_RISK_SCALE_M
                        ))
                        record.update(
                            {
                                "ransac_inliers": int(inliers.sum()),
                                "inlier_residual_mean_m": float(inlier_residuals.mean()),
                                "inlier_residual_p95_m": float(
                                    np.percentile(inlier_residuals, 95)
                                ),
                                "geometric_pose_confidence": float(
                                    geometric_confidence
                                ),
                                "estimated_motion_rotation_deg": motion_rotation_deg,
                                "estimated_motion_translation_m": motion_translation_m,
                                "motion_risk_factor": motion_risk_factor,
                                "pose_confidence": float(
                                    geometric_confidence * motion_risk_factor
                                ),
                                # Keep the exo-only rigid head transform itself.  The
                                # legacy camera deltas below are convenient when the
                                # true initial ego pose is known, but their translation
                                # is tied to that initial camera lever arm.  Consumers
                                # that ablate the initial head-to-camera relationship
                                # must instead apply this transform to their predicted
                                # initial pose.
                                "estimated_head_transform_world": head_transform.tolist(),
                                "estimated_camera_delta_position_world_m": delta_position.tolist(),
                                "estimated_delta_rotation_world": delta_rotation.tolist(),
                            }
                        )
                clip_record["future"].append(record)
            per_clip.append(clip_record)
            if args.audit_output is not None:
                confidence_by_frame = confidence_map(
                    frames.tolist(), clip_record
                )
                deployed = deployed_pose_map(
                    frames.tolist(), initial_pose, clip_record,
                    rotation_scale=args.deployment_rotation_scale,
                )
                swept = {
                    scale: deployed_pose_map(
                        frames.tolist(), initial_pose, clip_record,
                        rotation_scale=scale,
                    )
                    for scale in scales
                }
                for frame in frames[1:]:
                    gt_pose = load_pose(
                        target_root / "cam_pose" / f"{int(frame):06d}.txt"
                    )
                    predicted = deployed[int(frame)]
                    rotation_error = rotation_angle_deg(
                        predicted[:3, :3].T @ gt_pose[:3, :3]
                    )
                    translation_error = float(np.linalg.norm(
                        predicted[:3, 3] - gt_pose[:3, 3]
                    ))
                    audit_samples.append({
                        "pair_id": row["pair_id"],
                        "frame": int(frame),
                        "raw_confidence": confidence_by_frame[int(frame)],
                        "rotation_error_deg": rotation_error,
                        "translation_error_m": translation_error,
                    })
                    for scale in scales:
                        rotation_scale_errors[scale].append(rotation_angle_deg(
                            swept[scale][int(frame)][:3, :3].T @ gt_pose[:3, :3]
                        ))
            print(
                json.dumps(
                    {
                        "clip": clip_index,
                        "pair_id": row["pair_id"],
                        "views": detected_views,
                        "points": clip_record["face_points"],
                        "estimated": sum("ransac_inliers" in value for value in clip_record["future"]),
                    },
                    ensure_ascii=False,
                ),
                flush=True,
            )

    result = {
        "protocol": (
            f"exo cameras {list(camera_indices)} RGB-D + calibration + initial ego pose; "
            "future ego poses not read"
        ),
        "camera_indices": list(camera_indices),
        "model": str(args.model),
        "samples": len(rows),
        "split": args.split,
        "face_landmarks": list(FACE_LANDMARKS),
        "confidence_motion_risk_scales": {
            "rotation_deg": MOTION_ROTATION_RISK_SCALE_DEG,
            "translation_m": MOTION_TRANSLATION_RISK_SCALE_M,
        },
        "camera_detection_fraction": counters["camera_images_detected"] / max(counters["camera_images"], 1),
        "face_cloud_frame_fraction": counters["frames_with_face_cloud"] / max(len(rows) * args.frames_per_clip, 1),
        "future_pose_estimate_fraction": counters["future_frames_estimated"] / max(len(rows) * (args.frames_per_clip - 1), 1),
        "ransac_inliers": describe(inlier_counts),
        "counters": dict(counters),
        "per_clip": per_clip,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print(json.dumps(result | {"per_clip": "omitted"}, indent=2, ensure_ascii=False))
    if args.audit_output is not None:
        confidence = [float(value["raw_confidence"]) for value in audit_samples]
        rotation_error = [float(value["rotation_error_deg"]) for value in audit_samples]
        translation_error = [float(value["translation_error_m"]) for value in audit_samples]
        good = [
            float(rotation <= args.good_rotation_deg and translation <= args.good_translation_m)
            for rotation, translation in zip(rotation_error, translation_error)
        ]
        audit = {
            "protocol": "GT future cam4 poses used only in this audit/calibration file",
            "split": args.split,
            "pair_ids": [row["pair_id"] for row in rows],
            "deployment_rotation_scale": args.deployment_rotation_scale,
            "good_pose_thresholds": {
                "rotation_deg": args.good_rotation_deg,
                "translation_m": args.good_translation_m,
            },
            "spearman": {
                "confidence_vs_rotation_error": spearman(confidence, rotation_error),
                "confidence_vs_translation_error": spearman(confidence, translation_error),
            },
            "calibration": {
                "probability_good_pose": isotonic_knots(
                    confidence, good, increasing=True
                ),
                "expected_rotation_error_deg": isotonic_knots(
                    confidence, rotation_error, increasing=False
                ),
                "expected_translation_error_m": isotonic_knots(
                    confidence, translation_error, increasing=False
                ),
            },
            "rotation_scale_sweep": {
                str(scale): describe(rotation_scale_errors[scale])
                for scale in scales
            },
            "samples": audit_samples,
        }
        args.audit_output.parent.mkdir(parents=True, exist_ok=True)
        args.audit_output.write_text(
            json.dumps(audit, indent=2, ensure_ascii=False) + "\n",
            encoding="utf-8",
        )


if __name__ == "__main__":
    main()
