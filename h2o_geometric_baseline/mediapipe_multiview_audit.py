#!/usr/bin/env python3
"""Audit an off-the-shelf RGB hand detector on calibrated H2O exo views.

This differs from ``multiview_triangulation_audit.py``: observations come from
MediaPipe detections on RGB, not from projected ground truth.  Ground truth is
used only for evaluation and oracle hand-identity association.  Consequently
the reported triangulation error is a detector upper bound; a deployable
pipeline must also solve cross-view identity association without annotations.
"""

from __future__ import annotations

import argparse
import itertools
import json
import math
from collections import defaultdict
from pathlib import Path

import mediapipe as mp
import numpy as np
from mediapipe.tasks import python
from mediapipe.tasks.python import vision
from PIL import Image, ImageDraw


DEFAULT_RAW = Path("datasets/H2O/raw")
DEFAULT_MODEL = Path("models/mediapipe/hand_landmarker.task")
DEFAULT_OUTPUT = Path("datasets/H2O/experiments/mediapipe_multiview_audit")
DEFAULT_SEQUENCES = ("subject1/h1/0", "subject2/k2/3", "subject3/o1/4", "subject4/o2/5")
CAMERAS = ("cam0", "cam1", "cam2", "cam3")
HAND_NAMES = ("Left", "Right")


def load_flat(path: Path, expected: int | None = None) -> np.ndarray:
    values = np.loadtxt(path, dtype=np.float64).reshape(-1)
    if expected is not None and values.size != expected:
        raise ValueError(f"{path}: expected {expected} values, got {values.size}")
    return values


def load_hand(path: Path) -> tuple[np.ndarray, np.ndarray]:
    values = load_flat(path, 128)
    presence = values[[0, 64]] > 0.5
    joints = np.stack((values[1:64].reshape(21, 3), values[65:128].reshape(21, 3)))
    return presence, joints


def project_camera(points: np.ndarray, intrinsics: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    depth = points[:, 2]
    valid = np.isfinite(points).all(axis=1) & (depth > 1e-6)
    uv = np.full((len(points), 2), np.nan, dtype=np.float64)
    fx, fy, cx, cy = intrinsics[:4]
    uv[valid, 0] = fx * points[valid, 0] / depth[valid] + cx
    uv[valid, 1] = fy * points[valid, 1] / depth[valid] + cy
    return uv, valid


def projection_matrix(camera_root: Path) -> np.ndarray:
    intrinsics = load_flat(camera_root / "cam_intrinsics.txt")
    intrinsic = np.array(
        [[intrinsics[0], 0, intrinsics[2]], [0, intrinsics[1], intrinsics[3]], [0, 0, 1]],
        dtype=np.float64,
    )
    pose = load_flat(camera_root / "cam_pose" / "000000.txt", 16).reshape(4, 4)
    return intrinsic @ np.linalg.inv(pose)[:3]


def frame_projection_matrix(camera_root: Path, stem: str) -> np.ndarray:
    intrinsics = load_flat(camera_root / "cam_intrinsics.txt")
    intrinsic = np.array(
        [[intrinsics[0], 0, intrinsics[2]], [0, intrinsics[1], intrinsics[3]], [0, 0, 1]],
        dtype=np.float64,
    )
    pose = load_flat(camera_root / "cam_pose" / f"{stem}.txt", 16).reshape(4, 4)
    return intrinsic @ np.linalg.inv(pose)[:3]


def transform_points(transform: np.ndarray, points: np.ndarray) -> np.ndarray:
    return points @ transform[:3, :3].T + transform[:3, 3]


def triangulate(observations: list[tuple[np.ndarray, np.ndarray]]) -> np.ndarray:
    rows = []
    for uv, matrix in observations:
        rows.extend((uv[0] * matrix[2] - matrix[0], uv[1] * matrix[2] - matrix[1]))
    _, _, vectors = np.linalg.svd(np.stack(rows), full_matrices=False)
    homogeneous = vectors[-1]
    if abs(homogeneous[3]) < 1e-10:
        return np.full(3, np.nan)
    return homogeneous[:3] / homogeneous[3]


def reproject(point: np.ndarray, matrix: np.ndarray) -> np.ndarray:
    projected = matrix @ np.append(point, 1.0)
    if projected[2] <= 1e-8:
        return np.full(2, np.nan)
    return projected[:2] / projected[2]


def robust_triangulate(
    observations: list[tuple[np.ndarray, np.ndarray]], threshold_px: float
) -> tuple[np.ndarray, int]:
    if len(observations) == 2:
        return triangulate(observations), 2
    best_inliers: list[int] = []
    best_error = math.inf
    for first, second in itertools.combinations(range(len(observations)), 2):
        estimate = triangulate([observations[first], observations[second]])
        errors = np.asarray(
            [np.linalg.norm(reproject(estimate, matrix) - uv) for uv, matrix in observations]
        )
        inliers = np.flatnonzero(np.isfinite(errors) & (errors <= threshold_px)).tolist()
        score = float(errors[inliers].mean()) if inliers else math.inf
        if len(inliers) > len(best_inliers) or (len(inliers) == len(best_inliers) and score < best_error):
            best_inliers, best_error = inliers, score
    if len(best_inliers) < 2:
        return np.full(3, np.nan), len(best_inliers)
    return triangulate([observations[index] for index in best_inliers]), len(best_inliers)


def hand_scale(uv: np.ndarray, inside: np.ndarray) -> float:
    points = uv[inside]
    if len(points) < 2:
        return 1.0
    return max(float(np.linalg.norm(points.max(axis=0) - points.min(axis=0))), 20.0)


def optimal_assignment(cost: np.ndarray) -> list[tuple[int, int]]:
    ground_truth, predictions = cost.shape
    count = min(ground_truth, predictions)
    if count == 0:
        return []
    best: list[tuple[int, int]] = []
    best_cost = math.inf
    for gt_subset in itertools.combinations(range(ground_truth), count):
        for pred_subset in itertools.permutations(range(predictions), count):
            value = sum(float(cost[gt, pred]) for gt, pred in zip(gt_subset, pred_subset))
            if value < best_cost:
                best_cost = value
                best = list(zip(gt_subset, pred_subset))
    return best


def summary(values: list[float], scale: float = 1.0) -> dict[str, float | int | None]:
    if not values:
        return {"count": 0, "mean": None, "median": None, "p95": None}
    array = np.asarray(values, dtype=np.float64) * scale
    return {
        "count": int(len(array)),
        "mean": float(array.mean()),
        "median": float(np.median(array)),
        "p95": float(np.percentile(array, 95)),
    }


def draw_overlay(
    image_path: Path,
    gt_uv: list[np.ndarray],
    detections: list[np.ndarray],
    handedness: list[str],
    output: Path,
) -> None:
    image = Image.open(image_path).convert("RGB")
    draw = ImageDraw.Draw(image)
    colors = ("#ff3030", "#ff9f1c")
    for hand_index, points in enumerate(gt_uv):
        for x, y in points:
            draw.ellipse((x - 3, y - 3, x + 3, y + 3), outline=colors[hand_index], width=2)
        draw.text(tuple(points[0]), f"GT {HAND_NAMES[hand_index]}", fill=colors[hand_index])
    for index, points in enumerate(detections):
        for x, y in points:
            draw.ellipse((x - 2, y - 2, x + 2, y + 2), fill="#00e676")
        draw.text(tuple(points[0] + np.array((0, 12))), f"MP {handedness[index]}", fill="#00a844")
    output.parent.mkdir(parents=True, exist_ok=True)
    image.resize((640, 360), Image.Resampling.LANCZOS).save(output)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--raw-root", type=Path, default=DEFAULT_RAW)
    parser.add_argument("--model", type=Path, default=DEFAULT_MODEL)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--sequences", nargs="+", default=list(DEFAULT_SEQUENCES))
    parser.add_argument("--start", type=int, default=100)
    parser.add_argument("--end", type=int, default=157)
    parser.add_argument("--step", type=int, default=3)
    parser.add_argument("--min-detection-confidence", type=float, default=0.1)
    parser.add_argument("--min-presence-confidence", type=float, default=0.1)
    parser.add_argument("--match-scale-threshold", type=float, default=0.75)
    parser.add_argument("--ransac-threshold-px", type=float, default=25.0)
    parser.add_argument("--previews", type=int, default=8)
    args = parser.parse_args()

    options = vision.HandLandmarkerOptions(
        base_options=python.BaseOptions(model_asset_path=str(args.model)),
        running_mode=vision.RunningMode.IMAGE,
        num_hands=2,
        min_hand_detection_confidence=args.min_detection_confidence,
        min_hand_presence_confidence=args.min_presence_confidence,
        min_tracking_confidence=0.1,
    )
    counters = defaultdict(int)
    per_camera = {camera: defaultdict(int) for camera in CAMERAS}
    pixel_errors: list[float] = []
    normalized_errors: list[float] = []
    wrist_errors: list[float] = []
    handedness_direct: list[bool] = []
    handedness_flipped: list[bool] = []
    triangulation_all: list[float] = []
    triangulation_robust: list[float] = []
    triangulated_view_counts: list[int] = []
    ransac_inlier_counts: list[int] = []
    handed_triangulation_all: list[float] = []
    handed_triangulation_robust: list[float] = []
    handed_view_counts: list[int] = []
    handed_ransac_inliers: list[int] = []
    preview_count = 0

    with vision.HandLandmarker.create_from_options(options) as detector:
        for sequence in args.sequences:
            sequence_root = args.raw_root / sequence
            stems = [f"{frame:06d}" for frame in range(args.start, args.end + 1, args.step)]
            for stem in stems:
                # Observations indexed by physical hand then joint.
                multiview: list[list[list[tuple[np.ndarray, np.ndarray]]]] = [
                    [[] for _ in range(21)] for _ in range(2)
                ]
                handed_multiview: list[list[list[tuple[np.ndarray, np.ndarray]]]] = [
                    [[] for _ in range(21)] for _ in range(2)
                ]
                reference_world: list[np.ndarray | None] = [None, None]
                for camera in CAMERAS:
                    camera_root = sequence_root / camera
                    image_path = camera_root / "rgb" / f"{stem}.png"
                    hand_path = camera_root / "hand_pose" / f"{stem}.txt"
                    pose_path = camera_root / "cam_pose" / f"{stem}.txt"
                    if not (image_path.exists() and hand_path.exists() and pose_path.exists()):
                        continue
                    counters["camera_images"] += 1
                    per_camera[camera]["camera_images"] += 1
                    intrinsics = load_flat(camera_root / "cam_intrinsics.txt")
                    width, height = int(intrinsics[4]), int(intrinsics[5])
                    presence, joints_camera = load_hand(hand_path)
                    gt_uv_all: list[np.ndarray] = []
                    eligible_hands: list[int] = []
                    scales: list[float] = []
                    for hand_index in range(2):
                        uv, valid = project_camera(joints_camera[hand_index], intrinsics)
                        inside = (
                            valid
                            & (uv[:, 0] >= 0)
                            & (uv[:, 0] < width)
                            & (uv[:, 1] >= 0)
                            & (uv[:, 1] < height)
                        )
                        gt_uv_all.append(uv)
                        scales.append(hand_scale(uv, inside))
                        if presence[hand_index] and int(inside.sum()) >= 11:
                            eligible_hands.append(hand_index)
                            counters["eligible_hands"] += 1
                            per_camera[camera]["eligible_hands"] += 1
                        if camera == CAMERAS[0] and presence[hand_index]:
                            pose = load_flat(pose_path, 16).reshape(4, 4)
                            reference_world[hand_index] = transform_points(pose, joints_camera[hand_index])

                    result = detector.detect(mp.Image.create_from_file(str(image_path)))
                    predictions = [
                        np.asarray([[point.x * width, point.y * height] for point in hand], dtype=np.float64)
                        for hand in result.hand_landmarks
                    ]
                    labels = [categories[0].category_name for categories in result.handedness]
                    scores = [float(categories[0].score) for categories in result.handedness]
                    counters["raw_detections"] += len(predictions)
                    per_camera[camera]["raw_detections"] += len(predictions)
                    matrix = frame_projection_matrix(camera_root, stem)
                    # Deployable association baseline: trust MediaPipe handedness
                    # and keep the highest-confidence prediction for each side.
                    for hand_index, hand_name in enumerate(HAND_NAMES):
                        candidates = [
                            index for index, label in enumerate(labels) if label == hand_name
                        ]
                        if not candidates:
                            continue
                        pred_index = max(candidates, key=lambda index: scores[index])
                        for joint_index, uv in enumerate(predictions[pred_index]):
                            handed_multiview[hand_index][joint_index].append((uv, matrix))
                    if eligible_hands and predictions:
                        cost = np.asarray(
                            [
                                [np.median(np.linalg.norm(gt_uv_all[hand] - pred, axis=1)) for pred in predictions]
                                for hand in eligible_hands
                            ]
                        )
                        for gt_local, pred_index in optimal_assignment(cost):
                            hand_index = eligible_hands[gt_local]
                            scale = scales[hand_index]
                            point_error = np.linalg.norm(gt_uv_all[hand_index] - predictions[pred_index], axis=1)
                            if float(np.median(point_error) / scale) > args.match_scale_threshold:
                                counters["rejected_assignments"] += 1
                                continue
                            counters["matched_hands"] += 1
                            per_camera[camera]["matched_hands"] += 1
                            pixel_errors.extend(point_error.tolist())
                            normalized_errors.extend((point_error / scale).tolist())
                            wrist_errors.append(float(point_error[0]))
                            handedness_direct.append(labels[pred_index] == HAND_NAMES[hand_index])
                            handedness_flipped.append(labels[pred_index] != HAND_NAMES[hand_index])
                            for joint_index, uv in enumerate(predictions[pred_index]):
                                multiview[hand_index][joint_index].append((uv, matrix))

                    if preview_count < args.previews:
                        draw_overlay(
                            image_path,
                            gt_uv_all,
                            predictions,
                            labels,
                            args.output_dir / "previews" / f"{sequence.replace('/', '_')}_{stem}_{camera}.jpg",
                        )
                        preview_count += 1

                for hand_index in range(2):
                    if reference_world[hand_index] is None:
                        continue
                    for joint_index, observations in enumerate(multiview[hand_index]):
                        counters["present_world_joints"] += 1
                        if len(observations) < 2:
                            continue
                        counters["triangulated_joints"] += 1
                        triangulated_view_counts.append(len(observations))
                        estimate_all = triangulate(observations)
                        estimate_robust, inliers = robust_triangulate(
                            observations, args.ransac_threshold_px
                        )
                        ground_truth = reference_world[hand_index][joint_index]
                        if np.isfinite(estimate_all).all():
                            triangulation_all.append(float(np.linalg.norm(estimate_all - ground_truth)))
                        if np.isfinite(estimate_robust).all():
                            triangulation_robust.append(float(np.linalg.norm(estimate_robust - ground_truth)))
                            ransac_inlier_counts.append(inliers)
                    for joint_index, observations in enumerate(handed_multiview[hand_index]):
                        counters["handed_present_world_joints"] += 1
                        if len(observations) < 2:
                            continue
                        counters["handed_triangulated_joints"] += 1
                        handed_view_counts.append(len(observations))
                        estimate_all = triangulate(observations)
                        estimate_robust, inliers = robust_triangulate(
                            observations, args.ransac_threshold_px
                        )
                        ground_truth = reference_world[hand_index][joint_index]
                        if np.isfinite(estimate_all).all():
                            handed_triangulation_all.append(
                                float(np.linalg.norm(estimate_all - ground_truth))
                            )
                        if np.isfinite(estimate_robust).all():
                            handed_triangulation_robust.append(
                                float(np.linalg.norm(estimate_robust - ground_truth))
                            )
                            handed_ransac_inliers.append(inliers)

    normalized = np.asarray(normalized_errors, dtype=np.float64)
    metrics = {
        "sequences": args.sequences,
        "frame_range": [args.start, args.end, args.step],
        "uses_real_rgb_detector": True,
        "detector": "MediaPipe Hand Landmarker float16 v1",
        "hand_identity_association": "oracle nearest-to-ground-truth assignment for evaluation",
        "camera_images": counters["camera_images"],
        "eligible_hands": counters["eligible_hands"],
        "raw_detections": counters["raw_detections"],
        "matched_hands": counters["matched_hands"],
        "matched_hand_recall": counters["matched_hands"] / max(counters["eligible_hands"], 1),
        "detections_per_image": counters["raw_detections"] / max(counters["camera_images"], 1),
        "keypoint_error_px": summary(pixel_errors),
        "wrist_error_px": summary(wrist_errors),
        "normalized_keypoint_error": summary(normalized_errors),
        "pck": {
            "0.1_hand_diagonal": float(np.mean(normalized <= 0.1)) if len(normalized) else None,
            "0.2_hand_diagonal": float(np.mean(normalized <= 0.2)) if len(normalized) else None,
            "0.5_hand_diagonal": float(np.mean(normalized <= 0.5)) if len(normalized) else None,
        },
        "handedness_accuracy_direct": float(np.mean(handedness_direct)) if handedness_direct else None,
        "handedness_accuracy_if_flipped": float(np.mean(handedness_flipped)) if handedness_flipped else None,
        "per_camera": {
            camera: {
                **values,
                "matched_hand_recall": values["matched_hands"] / max(values["eligible_hands"], 1),
            }
            for camera, values in per_camera.items()
        },
        "present_world_joints": counters["present_world_joints"],
        "triangulated_joints": counters["triangulated_joints"],
        "triangulated_joint_fraction": counters["triangulated_joints"]
        / max(counters["present_world_joints"], 1),
        "triangulation_views": summary([float(value) for value in triangulated_view_counts]),
        "triangulation_all_views_mm": summary(triangulation_all, 1000.0),
        "triangulation_ransac_mm": summary(triangulation_robust, 1000.0),
        "ransac_inliers": summary([float(value) for value in ransac_inlier_counts]),
        "handedness_association": {
            "method": "highest-confidence detection per MediaPipe Left/Right label",
            "uses_ground_truth_for_association": False,
            "present_world_joints": counters["handed_present_world_joints"],
            "triangulated_joints": counters["handed_triangulated_joints"],
            "triangulated_joint_fraction": counters["handed_triangulated_joints"]
            / max(counters["handed_present_world_joints"], 1),
            "triangulation_views": summary([float(value) for value in handed_view_counts]),
            "triangulation_all_views_mm": summary(handed_triangulation_all, 1000.0),
            "triangulation_ransac_mm": summary(handed_triangulation_robust, 1000.0),
            "ransac_inliers": summary([float(value) for value in handed_ransac_inliers]),
        },
        "limitations": [
            "ground truth is used for hand-identity matching",
            "object state is not detected",
            "MediaPipe and H2O joint definitions may differ systematically",
        ],
    }
    args.output_dir.mkdir(parents=True, exist_ok=True)
    (args.output_dir / "summary.json").write_text(json.dumps(metrics, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(metrics, indent=2))


if __name__ == "__main__":
    main()
