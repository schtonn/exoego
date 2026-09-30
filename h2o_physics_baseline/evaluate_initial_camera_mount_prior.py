#!/usr/bin/env python3
"""Ablate the initial ego-camera/head relationship with an exo-only head estimate.

The target clip's cam4 pose is never used to predict its initial camera pose.  A
face-centred 6-DoF head frame is recovered from exo RGB-D, and a canonical
head-to-camera transform is learned from calibration clips in another split.
The target cam4 pose is loaded only after prediction to report pose error.
"""

from __future__ import annotations

import argparse
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
SCRIPT_ROOT = Path(__file__).resolve().parent
if str(SCRIPT_ROOT) not in sys.path:
    sys.path.insert(0, str(SCRIPT_ROOT))

from h2o_geometric_baseline.reprojection import load_intrinsics, load_pose
from h2o_geometric_baseline.mediapipe_multiview_audit import (
    frame_projection_matrix,
    robust_triangulate,
)
from audit_exo_face_head_motion import (
    FACE_LANDMARKS,
    depth_point_world,
    pose_guided_crop,
)
from audit_exo_head_motion import describe, load_combined_rows, rotation_angle_deg


REQUIRED_FRAME_LANDMARKS = (1, 10, 33, 133, 152, 263, 362)
SHAPE_LANDMARKS = (1, 10, 33, 152, 234, 263, 454)


def project_rotation(matrix: np.ndarray) -> np.ndarray:
    u, _, vt = np.linalg.svd(matrix)
    rotation = u @ vt
    if np.linalg.det(rotation) < 0:
        u[:, -1] *= -1
        rotation = u @ vt
    return rotation


def head_frame(points: dict[int, np.ndarray]) -> np.ndarray | None:
    """Construct a face-centred, camera-like (right/down/forward) world frame."""
    if not all(index in points for index in REQUIRED_FRAME_LANDMARKS):
        return None
    left_eye = 0.5 * (points[33] + points[133])
    right_eye = 0.5 * (points[263] + points[362])
    eye_mid = 0.5 * (left_eye + right_eye)
    x_axis = right_eye - left_eye
    x_norm = np.linalg.norm(x_axis)
    if x_norm < 0.025:
        return None
    x_axis /= x_norm
    down_hint = points[152] - points[10]
    down_hint -= x_axis * np.dot(down_hint, x_axis)
    down_norm = np.linalg.norm(down_hint)
    if down_norm < 0.04:
        return None
    y_axis = down_hint / down_norm
    z_axis = np.cross(x_axis, y_axis)
    z_norm = np.linalg.norm(z_axis)
    if z_norm < 1e-6:
        return None
    z_axis /= z_norm
    # The nose protrudes toward the facial forward hemisphere.  Use it to
    # resolve the otherwise ambiguous sign of the surface normal.
    if np.dot(z_axis, points[1] - eye_mid) < 0:
        z_axis *= -1
    y_axis = np.cross(z_axis, x_axis)
    y_axis /= np.linalg.norm(y_axis)
    pose = np.eye(4, dtype=np.float64)
    pose[:3, :3] = np.stack((x_axis, y_axis, z_axis), axis=1)
    pose[:3, 3] = eye_mid
    return pose


def face_shape_features(points: dict[int, np.ndarray]) -> list[float] | None:
    """Metric exo-only face geometry used to condition the mount prior."""
    if not all(index in points for index in SHAPE_LANDMARKS):
        return None
    frame = head_frame(points)
    if frame is None:
        return None
    eye_mid = 0.5 * (points[33] + points[263])
    nose_local = frame[:3, :3].T @ (points[1] - eye_mid)
    return [
        float(np.linalg.norm(points[454] - points[234])),
        float(np.linalg.norm(points[152] - points[10])),
        float(np.linalg.norm(points[263] - points[33])),
        float(nose_local[0]),
        float(nose_local[1]),
        float(nose_local[2]),
    ]


def detect_face_cloud(
    row: dict[str, str],
    detector: object,
    pose_detector: object,
    camera_indices: tuple[int, ...],
    depth_radius: int,
    predicted_depth_root: Path | None,
) -> tuple[dict[int, np.ndarray], int, dict[str, object]]:
    frame = int(row["start_frame"])
    stem = f"{frame:06d}"
    roots = [Path(value).parent for value in json.loads(row["source_rgb_dirs"])]
    observations = {index: [] for index in FACE_LANDMARKS}
    image_observations = {index: [] for index in FACE_LANDMARKS}
    raw_points_by_view: dict[int, dict[int, np.ndarray]] = {}
    camera_poses: dict[int, np.ndarray] = {}
    detections = 0
    for camera_index in camera_indices:
        camera_root = roots[camera_index]
        image = np.asarray(Image.open(camera_root / "rgb" / f"{stem}.png").convert("RGB"))
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
        if result is None or not result.face_landmarks:
            result = detector.detect(mp_image)
            crop_box = (0, 0, image.shape[1], image.shape[0])
        if not result.face_landmarks:
            continue
        detections += 1
        landmarks = result.face_landmarks[0]
        if predicted_depth_root is None:
            depth = np.asarray(Image.open(camera_root / "depth" / f"{stem}.png"))
        else:
            depth_m = np.load(
                predicted_depth_root / row["sequence"]
                / f"cam{camera_index}" / f"{stem}.npy"
            )
            # depth_point_world consumes millimetres to share the exact same
            # back-projection path as the H2O sensor-depth condition.
            depth = depth_m.astype(np.float64) * 1000.0
        intrinsics = load_intrinsics(camera_root / "cam_intrinsics.txt")
        camera_pose = load_pose(camera_root / "cam_pose" / f"{stem}.txt")
        camera_poses[camera_index] = camera_pose
        raw_points_by_view[camera_index] = {}
        projection = frame_projection_matrix(camera_root, stem)
        x0, y0, x1, y1 = crop_box
        crop_width, crop_height = x1 - x0, y1 - y0
        for index in FACE_LANDMARKS:
            landmark = landmarks[index]
            full_landmark = SimpleNamespace(
                x=(x0 + landmark.x * crop_width) / image.shape[1],
                y=(y0 + landmark.y * crop_height) / image.shape[0],
            )
            uv = np.asarray(
                (full_landmark.x * image.shape[1], full_landmark.y * image.shape[0]),
                dtype=np.float64,
            )
            image_observations[index].append((uv, projection))
            point = depth_point_world(
                full_landmark, depth, intrinsics, camera_pose, depth_radius
            )
            if point is not None:
                observations[index].append(point)
                raw_points_by_view[camera_index][index] = point
    diagnostics: dict[str, object] = {}
    if predicted_depth_root is not None:
        # DA-V2 metric depth has a substantial image-dependent scale bias on
        # these close facial crops.  Recover metric scale from calibrated exo
        # multiview geometry only (no H2O depth and no ego input), then retain
        # DA's within-image depth variation.
        triangulated = {}
        for index, values in image_observations.items():
            if len(values) < 2:
                continue
            point, inliers = robust_triangulate(values, threshold_px=20.0)
            if inliers >= 2 and np.isfinite(point).all():
                triangulated[index] = point
        observations = {index: [] for index in FACE_LANDMARKS}
        view_scales = {}
        for camera_index, raw_points in raw_points_by_view.items():
            pose = camera_poses[camera_index]
            world_to_camera = np.linalg.inv(pose)
            ratios = []
            for index, raw_point in raw_points.items():
                if index not in triangulated:
                    continue
                predicted_z = (
                    world_to_camera[:3, :3] @ raw_point + world_to_camera[:3, 3]
                )[2]
                true_z = (
                    world_to_camera[:3, :3] @ triangulated[index]
                    + world_to_camera[:3, 3]
                )[2]
                if predicted_z > 0.1 and true_z > 0.1:
                    ratios.append(float(true_z / predicted_z))
            if len(ratios) < 3:
                continue
            scale = float(np.median(ratios))
            view_scales[str(camera_index)] = scale
            center = pose[:3, 3]
            for index, raw_point in raw_points.items():
                observations[index].append(center + scale * (raw_point - center))
        diagnostics = {
            "triangulated_landmarks": len(triangulated),
            "depth_anything_view_scales": view_scales,
        }
    return (
        {
            index: np.median(np.stack(values), axis=0)
            for index, values in observations.items()
            if values
        },
        detections,
        diagnostics,
    )


def average_mount(transforms: list[np.ndarray]) -> np.ndarray:
    result = np.eye(4, dtype=np.float64)
    result[:3, :3] = project_rotation(
        np.mean(np.stack([value[:3, :3] for value in transforms]), axis=0)
    )
    result[:3, 3] = np.median(
        np.stack([value[:3, 3] for value in transforms]), axis=0
    )
    return result


def robust_mount_inliers(transforms: list[np.ndarray]) -> np.ndarray:
    """Reject calibration failures without consulting any target-clip pose."""
    translations = np.stack([value[:3, 3] for value in transforms])
    center = np.median(translations, axis=0)
    translation_residual = np.linalg.norm(translations - center, axis=1)
    rotations = [value[:3, :3] for value in transforms]
    pairwise = np.asarray(
        [
            [rotation_angle_deg(first.T @ second) for second in rotations]
            for first in rotations
        ]
    )
    medoid = int(np.argmin(np.median(pairwise, axis=1)))
    rotation_residual = pairwise[medoid]
    translation_median = float(np.median(translation_residual))
    rotation_median = float(np.median(rotation_residual))
    # Floors preserve natural inter-person mount variation; multiples reject
    # only gross face/depth failures such as the 180-degree DA outliers.
    translation_threshold = max(0.08, 4.0 * translation_median)
    rotation_threshold = max(30.0, 3.0 * rotation_median)
    return (translation_residual <= translation_threshold) & (
        rotation_residual <= rotation_threshold
    )


def knn_mount(
    train_features: np.ndarray,
    train_mounts: list[np.ndarray],
    target_feature: np.ndarray,
    neighbors: int,
) -> np.ndarray:
    center = np.median(train_features, axis=0)
    scale = np.median(np.abs(train_features - center), axis=0)
    scale = np.maximum(scale, 1e-4)
    distances = np.linalg.norm((train_features - target_feature) / scale, axis=1)
    selected = np.argsort(distances)[: min(neighbors, len(train_mounts))]
    return average_mount([train_mounts[int(index)] for index in selected])


def mount_proxy_error(predicted: np.ndarray, target: np.ndarray) -> float:
    """Frustum-point displacement proxy selected without target ego imagery."""
    camera_points = np.asarray([
        (0.0, 0.0, 0.0),
        (-0.5, -0.4, 1.0),
        (0.5, -0.4, 1.0),
        (0.5, 0.4, 1.0),
        (-0.5, 0.4, 1.0),
    ])
    predicted_points = camera_points @ predicted[:3, :3].T + predicted[:3, 3]
    target_points = camera_points @ target[:3, :3].T + target[:3, 3]
    return float(np.linalg.norm(predicted_points - target_points, axis=1).mean())


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
        "--pose-model",
        type=Path,
        default=Path("models/mediapipe/pose_landmarker_full.task"),
    )
    parser.add_argument("--calibration-split", default="train")
    parser.add_argument("--target-split", default="val")
    parser.add_argument("--calibration-samples", type=int, default=64)
    parser.add_argument("--target-max-samples", type=int, default=32)
    parser.add_argument("--target-pair-id", action="append", required=True)
    parser.add_argument("--camera-indices", default="0,1,2,3")
    parser.add_argument("--depth-radius", type=int, default=3)
    parser.add_argument(
        "--predicted-depth-root",
        type=Path,
        default=None,
        help="Optional Depth Anything cache; when set, H2O exo depth is not read.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("datasets/H2O/experiments/initial_camera_mount_prior/summary.json"),
    )
    args = parser.parse_args()
    camera_indices = tuple(
        int(value.strip()) for value in args.camera_indices.split(",") if value.strip()
    )
    if not camera_indices or any(index < 0 or index > 3 for index in camera_indices):
        raise ValueError(f"Invalid --camera-indices={args.camera_indices}")

    calibration_rows = load_combined_rows(
        args.index, args.calibration_samples, split=args.calibration_split
    )
    target_rows = load_combined_rows(
        args.index, args.target_max_samples, split=args.target_split
    )
    requested = set(args.target_pair_id)
    target_rows = [row for row in target_rows if row["pair_id"] in requested]
    missing = requested - {row["pair_id"] for row in target_rows}
    if missing:
        raise ValueError(f"Target pair ids not selected: {sorted(missing)}")

    face_options = vision.FaceLandmarkerOptions(
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
    calibration_records = []
    target_head_frames = {}
    with (
        vision.FaceLandmarker.create_from_options(face_options) as detector,
        vision.PoseLandmarker.create_from_options(pose_options) as pose_detector,
    ):
        for index, row in enumerate(calibration_rows):
            points, detections, depth_diagnostics = detect_face_cloud(
                row, detector, pose_detector, camera_indices, args.depth_radius,
                args.predicted_depth_root,
            )
            frame_pose = head_frame(points)
            shape_features = face_shape_features(points)
            if frame_pose is None or shape_features is None:
                print(json.dumps({"calibration": index, "status": "head-frame-failed"}), flush=True)
                continue
            frame = int(row["start_frame"])
            target_root = Path(row["target_rgb_dir"]).parent
            camera_pose = load_pose(target_root / "cam_pose" / f"{frame:06d}.txt")
            mount = np.linalg.inv(frame_pose) @ camera_pose
            calibration_records.append(
                {
                    "pair_id": row["pair_id"],
                    "sequence": row["sequence"],
                    "detected_exo_views": detections,
                    "depth_diagnostics": depth_diagnostics,
                    "face_shape_features": shape_features,
                    "head_from_camera": mount.tolist(),
                }
            )
            print(
                json.dumps(
                    {"calibration": index, "pair_id": row["pair_id"], "views": detections},
                    ensure_ascii=False,
                ),
                flush=True,
            )
        for row in target_rows:
            points, detections, depth_diagnostics = detect_face_cloud(
                row, detector, pose_detector, camera_indices, args.depth_radius,
                args.predicted_depth_root,
            )
            frame_pose = head_frame(points)
            shape_features = face_shape_features(points)
            if frame_pose is None or shape_features is None:
                raise RuntimeError(f"Target head-frame detection failed: {row['pair_id']}")
            target_head_frames[row["pair_id"]] = (
                frame_pose, shape_features, row, detections, depth_diagnostics
            )

    if len(calibration_records) < 3:
        raise RuntimeError("Too few valid calibration head frames")
    mounts = [np.asarray(record["head_from_camera"]) for record in calibration_records]
    mount_inliers = robust_mount_inliers(mounts)
    canonical_mount = average_mount(
        [mount for mount, keep in zip(mounts, mount_inliers) if keep]
    )
    inlier_mounts = [mount for mount, keep in zip(mounts, mount_inliers) if keep]
    inlier_features = np.asarray([
        record["face_shape_features"]
        for record, keep in zip(calibration_records, mount_inliers)
        if keep
    ], dtype=np.float64)
    candidate_neighbors = [
        value for value in (1, 2, 4, 8, 16)
        if value < len(inlier_mounts)
    ]
    knn_cross_validation = {}
    for neighbors in candidate_neighbors:
        errors = []
        for held_out in range(len(inlier_mounts)):
            keep = np.arange(len(inlier_mounts)) != held_out
            predicted_mount = knn_mount(
                inlier_features[keep],
                [mount for index, mount in enumerate(inlier_mounts) if keep[index]],
                inlier_features[held_out],
                neighbors,
            )
            errors.append(mount_proxy_error(predicted_mount, inlier_mounts[held_out]))
        knn_cross_validation[str(neighbors)] = float(np.mean(errors))
    selected_neighbors = min(
        candidate_neighbors,
        key=lambda value: knn_cross_validation[str(value)],
    )
    calibration_translation_residuals = []
    calibration_rotation_residuals = []
    for mount in mounts:
        calibration_translation_residuals.append(
            float(np.linalg.norm(mount[:3, 3] - canonical_mount[:3, 3]))
        )
        calibration_rotation_residuals.append(
            rotation_angle_deg(canonical_mount[:3, :3].T @ mount[:3, :3])
        )

    target_records = []
    for pair_id, value in target_head_frames.items():
        frame_pose, shape_features, row, detections, depth_diagnostics = value
        predicted_pose = frame_pose @ canonical_mount
        conditioned_mount = knn_mount(
            inlier_features, inlier_mounts,
            np.asarray(shape_features, dtype=np.float64), selected_neighbors,
        )
        conditioned_pose = frame_pose @ conditioned_mount
        frame = int(row["start_frame"])
        target_root = Path(row["target_rgb_dir"]).parent
        # Evaluation-only read: it happens strictly after prediction.
        ground_truth_pose = load_pose(target_root / "cam_pose" / f"{frame:06d}.txt")
        target_records.append(
            {
                "pair_id": pair_id,
                "sequence": row["sequence"],
                "frame": frame,
                "detected_exo_views": detections,
                "depth_diagnostics": depth_diagnostics,
                "face_shape_features": shape_features,
                "predicted_initial_pose_world": predicted_pose.tolist(),
                "face_knn_candidate": {
                    "neighbors": selected_neighbors,
                    "predicted_initial_pose_world": conditioned_pose.tolist(),
                },
                "evaluation": {
                    "translation_error_m": float(
                        np.linalg.norm(predicted_pose[:3, 3] - ground_truth_pose[:3, 3])
                    ),
                    "rotation_error_deg": rotation_angle_deg(
                        predicted_pose[:3, :3].T @ ground_truth_pose[:3, :3]
                    ),
                    "face_knn_translation_error_m": float(
                        np.linalg.norm(
                            conditioned_pose[:3, 3] - ground_truth_pose[:3, 3]
                        )
                    ),
                    "face_knn_rotation_error_deg": rotation_angle_deg(
                        conditioned_pose[:3, :3].T @ ground_truth_pose[:3, :3]
                    ),
                },
            }
        )

    result = {
        "protocol": (
            "target initial cam4 pose withheld; exo face RGB-D head frame + canonical "
            f"head-camera mount learned from {args.calibration_split} split"
        ),
        "depth_source": (
            f"Depth Anything cache: {args.predicted_depth_root}"
            if args.predicted_depth_root is not None else "H2O exo metric depth"
        ),
        "camera_indices": list(camera_indices),
        "calibration_split": args.calibration_split,
        "target_split": args.target_split,
        "calibration_requested": len(calibration_rows),
        "calibration_valid": len(calibration_records),
        "calibration_mount_inliers": int(mount_inliers.sum()),
        "canonical_head_from_camera": canonical_mount.tolist(),
        "face_conditioned_mount": {
            "feature_order": [
                "face_width_m", "face_height_m", "eye_span_m",
                "nose_x_m", "nose_y_m", "nose_z_m",
            ],
            "selection": "leave-one-out frustum displacement on calibration split",
            "candidate_neighbor_proxy_errors": knn_cross_validation,
            "selected_neighbors": selected_neighbors,
        },
        "calibration_mount_residual": {
            "translation_m": describe(calibration_translation_residuals),
            "rotation_deg": describe(calibration_rotation_residuals),
        },
        "targets": target_records,
        "calibration_records": calibration_records,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2, ensure_ascii=False) + "\n")
    print(json.dumps(result | {"calibration_records": "omitted"}, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
