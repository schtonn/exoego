#!/usr/bin/env python3
"""Rejected diagnostic: white-color ICP is not a reliable semantic head tracker.

Retained only to reproduce the negative ablation. Do not use its poses in the
main renderer: view-dependent white surfaces reintroduce pose jitter.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

import numpy as np
from PIL import Image
from scipy.spatial import cKDTree

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from h2o_geometric_baseline.reprojection import load_intrinsics, load_pose
from h2o_physics_baseline.render_causal_video_background_split import (
    smoothed_pose_map,
    temporal_smooth,
)


def rigid_transform(source: np.ndarray, target: np.ndarray) -> np.ndarray:
    source_center = source.mean(axis=0)
    target_center = target.mean(axis=0)
    covariance = (source - source_center).T @ (target - target_center)
    u, _, vt = np.linalg.svd(covariance)
    rotation = vt.T @ u.T
    if np.linalg.det(rotation) < 0:
        vt[-1] *= -1
        rotation = vt.T @ u.T
    transform = np.eye(4, dtype=np.float64)
    transform[:3, :3] = rotation
    transform[:3, 3] = target_center - rotation @ source_center
    return transform


def rotation_angle_deg(rotation: np.ndarray) -> float:
    return float(np.degrees(np.arccos(np.clip((np.trace(rotation) - 1.0) / 2.0, -1.0, 1.0))))


def rgbd_near_position(
    camera_roots: list[Path], frame: int, center_world: np.ndarray,
    radius_m: float = 0.28, stride: int = 4,
) -> tuple[np.ndarray, np.ndarray]:
    points, colors = [], []
    for camera in camera_roots:
        stem = f"{frame:06d}"
        depth = np.asarray(Image.open(camera / "depth" / f"{stem}.png"))[::stride, ::stride]
        rgb = np.asarray(Image.open(camera / "rgb" / f"{stem}.png").convert("RGB"))[::stride, ::stride]
        intrinsics = load_intrinsics(camera / "cam_intrinsics.txt")
        pose = load_pose(camera / "cam_pose" / f"{stem}.txt")
        y, x = np.indices(depth.shape)
        x = x.astype(np.float64) * stride
        y = y.astype(np.float64) * stride
        z = depth.astype(np.float64) / 1000.0
        valid = (z >= 0.1) & (z <= 5.0)
        fx, fy, cx, cy = intrinsics[:4]
        local = np.column_stack(
            ((x[valid] - cx) * z[valid] / fx, (y[valid] - cy) * z[valid] / fy, z[valid])
        )
        world = local @ pose[:3, :3].T + pose[:3, 3]
        near = np.linalg.norm(world - center_world, axis=1) <= radius_m
        points.append(world[near])
        colors.append(rgb[valid][near])
    return np.concatenate(points), np.concatenate(colors)


def select_helmet(
    points: np.ndarray, colors: np.ndarray, center: np.ndarray,
    radius: float, brightness: int, chroma: int, voxel_m: float = 0.004,
) -> np.ndarray:
    distance = np.linalg.norm(points - center, axis=1)
    value = colors.max(axis=1)
    color_chroma = colors.max(axis=1).astype(np.int16) - colors.min(axis=1).astype(np.int16)
    keep = (distance <= radius) & (value >= brightness) & (color_chroma <= chroma)
    chosen = points[keep]
    if not len(chosen):
        return chosen
    cells = np.rint(chosen / voxel_m).astype(np.int64)
    _, indices = np.unique(cells, axis=0, return_index=True)
    return chosen[np.sort(indices)]


def transform_points(points: np.ndarray, transform: np.ndarray) -> np.ndarray:
    return points @ transform[:3, :3].T + transform[:3, 3]


def robust_icp(
    source: np.ndarray, target: np.ndarray, initial: np.ndarray,
    threshold_m: float, iterations: int = 12,
) -> tuple[np.ndarray, float, int]:
    if len(source) < 30 or len(target) < 30:
        return initial, float("nan"), 0
    tree = cKDTree(target)
    transform = initial.copy()
    final_distances = np.empty(0)
    for _ in range(iterations):
        moved = transform_points(source, transform)
        distances, indices = tree.query(moved, k=1, workers=1)
        candidate = distances <= threshold_m
        if candidate.sum() < 30:
            break
        cutoff = np.quantile(distances[candidate], 0.75)
        keep = candidate & (distances <= cutoff)
        delta = rigid_transform(moved[keep], target[indices[keep]])
        transform = delta @ transform
        final_distances = distances[keep]
        if np.linalg.norm(delta[:3, 3]) < 1e-5 and rotation_angle_deg(delta[:3, :3]) < 0.01:
            break
    rmse = float(np.sqrt(np.mean(final_distances ** 2))) if len(final_distances) else float("nan")
    return transform, rmse, int(len(final_distances))


def pose_errors(poses: list[np.ndarray], truth: list[np.ndarray]) -> dict[str, float]:
    translation = [np.linalg.norm(a[:3, 3] - b[:3, 3]) for a, b in zip(poses, truth)]
    rotation = [rotation_angle_deg(a[:3, :3].T @ b[:3, :3]) for a, b in zip(poses, truth)]
    return {
        "translation_mean_m": float(np.mean(translation)),
        "translation_p90_m": float(np.percentile(translation, 90)),
        "rotation_mean_deg": float(np.mean(rotation)),
        "rotation_p90_deg": float(np.percentile(rotation, 90)),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--sequence", required=True)
    parser.add_argument(
        "--dataset-root", type=Path, default=Path("datasets/H2O/raw")
    )
    parser.add_argument("--start-frame", type=int, required=True)
    parser.add_argument("--end-frame", type=int, required=True)
    parser.add_argument("--pair-id", required=True)
    parser.add_argument("--head-summary", type=Path, required=True)
    parser.add_argument("--stride", type=int, default=4)
    parser.add_argument(
        "--fixed-only", action="store_true",
        help="Evaluate only the preregistered 0.16/150/50/0.02 setting.",
    )
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    root = args.dataset_root / args.sequence
    sources = [root / f"cam{index}" for index in range(4)]
    target = root / "cam4"
    frames = list(range(args.start_frame, args.end_frame + 1))
    initial_pose = load_pose(target / "cam_pose" / f"{frames[0]:06d}.txt")
    summary = json.loads(args.head_summary.read_text())
    record = next(value for value in summary["per_clip"] if value["pair_id"] == args.pair_id)
    face_map = smoothed_pose_map(frames, initial_pose, record)
    face_poses = [face_map[frame] for frame in frames]
    truth = [load_pose(target / "cam_pose" / f"{frame:06d}.txt") for frame in frames]

    clouds = []
    for frame, pose in zip(frames, face_poses):
        points, colors = rgbd_near_position(
            sources, frame, pose[:3, 3], radius_m=0.28, stride=args.stride
        )
        clouds.append((points, colors))
        print(json.dumps({"frame": frame, "near_points": len(points)}), flush=True)

    settings = [
        (radius, brightness, chroma, threshold)
        for radius in (0.16, 0.20, 0.24)
        for brightness in (150, 180)
        for chroma in (50, 80)
        for threshold in (0.02, 0.035)
    ]
    if args.fixed_only:
        settings = [(0.16, 150, 50, 0.02)]
    results = []
    for radius, brightness, chroma, threshold in settings:
        reference = select_helmet(
            *clouds[0], face_poses[0][:3, 3], radius, brightness, chroma
        )
        refined, rmses, inliers, counts = [initial_pose], [], [], [len(reference)]
        for frame_index in range(1, len(frames)):
            current = select_helmet(
                *clouds[frame_index], face_poses[frame_index][:3, 3],
                radius, brightness, chroma,
            )
            initial_transform = face_poses[frame_index] @ np.linalg.inv(initial_pose)
            transform, rmse, count = robust_icp(reference, current, initial_transform, threshold)
            refined.append(transform @ initial_pose)
            rmses.append(rmse)
            inliers.append(count)
            counts.append(len(current))
        # Smooth only the translation; independently averaging rotation matrices
        # can erase the small real rotations we are trying to recover.
        translation = temporal_smooth(np.stack([pose[:3, 3] for pose in refined]), radius=2)
        smoothed = [pose.copy() for pose in refined]
        for pose, value in zip(smoothed, translation):
            pose[:3, 3] = value
        smoothed[0] = initial_pose.copy()
        result = {
            "radius_m": radius,
            "brightness": brightness,
            "chroma": chroma,
            "threshold_m": threshold,
            "helmet_points_mean": float(np.mean(counts)),
            "icp_inliers_mean": float(np.mean(inliers)),
            "icp_rmse_mean_m": float(np.nanmean(rmses)),
            "raw": pose_errors(refined, truth),
            "translation_smoothed": pose_errors(smoothed, truth),
            "poses": [pose.tolist() for pose in smoothed],
        }
        results.append(result)
        print(json.dumps({key: value for key, value in result.items() if key != "poses"}), flush=True)

    baseline = pose_errors(face_poses, truth)
    best = min(results, key=lambda value: value["translation_smoothed"]["rotation_mean_deg"])
    output = {
        "protocol": "four exo RGB-D + initial ego pose; GT ego poses evaluation-only",
        "pair_id": args.pair_id,
        "baseline": baseline,
        "selection_note": "best is reported diagnostically; validate fixed settings on held-out clips",
        "best": best,
        "settings": results,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(output, indent=2) + "\n")
    print(json.dumps({"baseline": baseline, "best": {k: v for k, v in best.items() if k != "poses"}}, indent=2))


if __name__ == "__main__":
    main()
