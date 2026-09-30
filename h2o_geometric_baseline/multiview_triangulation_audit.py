#!/usr/bin/env python3
"""Audit calibrated multi-exo hand triangulation under 2D keypoint noise.

The 2D observations are projected from H2O ground-truth 3D joints and then
perturbed, so this measures the geometric ceiling and noise sensitivity rather
than the accuracy of any particular hand detector.
"""

from __future__ import annotations

import argparse
import itertools
import json
from pathlib import Path

import numpy as np

from h2o_geometric_baseline.reprojection import load_intrinsics, load_pose


DEFAULT_RAW = Path("datasets/H2O/raw")
DEFAULT_STATE = Path("datasets/H2O/oracle_state")
DEFAULT_OUTPUT = Path("datasets/H2O/experiments/multiview_triangulation")
DEFAULT_SEQUENCES = ("subject1/h1/0", "subject2/k2/3", "subject3/o1/4", "subject4/o2/5")


def projection_matrix(camera: Path, frame: int) -> tuple[np.ndarray, np.ndarray]:
    fx, fy, cx, cy, width, height = load_intrinsics(camera / "cam_intrinsics.txt")
    intrinsic = np.array([[fx, 0, cx], [0, fy, cy], [0, 0, 1]], dtype=np.float64)
    world_to_camera = np.linalg.inv(load_pose(camera / "cam_pose" / f"{frame:06d}.txt"))
    return intrinsic @ world_to_camera[:3], np.array([width, height], dtype=np.float64)


def project(point_world: np.ndarray, matrix: np.ndarray, size: np.ndarray) -> tuple[np.ndarray, bool]:
    homogeneous = np.append(point_world, 1.0)
    projected = matrix @ homogeneous
    if projected[2] <= 1e-6:
        return np.zeros(2), False
    uv = projected[:2] / projected[2]
    valid = bool(0 <= uv[0] < size[0] and 0 <= uv[1] < size[1])
    return uv, valid


def triangulate(observations: list[tuple[np.ndarray, np.ndarray]]) -> np.ndarray:
    rows = []
    for uv, matrix in observations:
        rows.append(uv[0] * matrix[2] - matrix[0])
        rows.append(uv[1] * matrix[2] - matrix[1])
    _, _, vectors = np.linalg.svd(np.stack(rows), full_matrices=False)
    homogeneous = vectors[-1]
    return homogeneous[:3] / homogeneous[3]


def percentile_summary(values: list[float]) -> dict[str, float | int]:
    array = np.asarray(values, dtype=np.float64)
    return {
        "points": int(len(array)),
        "mean_mm": float(array.mean() * 1000),
        "median_mm": float(np.median(array) * 1000),
        "p95_mm": float(np.percentile(array, 95) * 1000),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--raw-root", type=Path, default=DEFAULT_RAW)
    parser.add_argument("--state-root", type=Path, default=DEFAULT_STATE)
    parser.add_argument("--sequences", nargs="+", default=list(DEFAULT_SEQUENCES))
    parser.add_argument("--start", type=int, default=100)
    parser.add_argument("--end", type=int, default=157)
    parser.add_argument("--step", type=int, default=3)
    parser.add_argument("--trials", type=int, default=5)
    parser.add_argument("--seed", type=int, default=20260926)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    args = parser.parse_args()
    rng = np.random.default_rng(args.seed)
    cameras = ("cam0", "cam1", "cam2", "cam3")
    noises = (0.0, 1.0, 3.0, 5.0, 10.0)
    errors = {(views, noise): [] for views in (2, 3, 4) for noise in noises}
    visible_counts = []

    for sequence in args.sequences:
        sequence_root = args.raw_root / sequence
        with np.load(args.state_root / sequence / "oracle_state.npz") as state:
            state_frames = state["frames"]
            hands_world = state["hand_joints_world_m"]
            presence = state["hand_presence"]
            for frame in range(args.start, args.end + 1, args.step):
                index = int(np.searchsorted(state_frames, frame))
                if index >= len(state_frames) or int(state_frames[index]) != frame:
                    continue
                matrices_sizes = [projection_matrix(sequence_root / camera, frame) for camera in cameras]
                for hand in range(2):
                    if not presence[index, hand]:
                        continue
                    for point_world in hands_world[index, hand]:
                        observations = []
                        for camera_index, (matrix, size) in enumerate(matrices_sizes):
                            uv, valid = project(point_world, matrix, size)
                            if valid:
                                observations.append((camera_index, uv, matrix))
                        visible_counts.append(len(observations))
                        for view_count in (2, 3, 4):
                            if len(observations) < view_count:
                                continue
                            combinations = list(itertools.combinations(observations, view_count))
                            for noise in noises:
                                repeats = 1 if noise == 0 else args.trials
                                for trial in range(repeats):
                                    selected = combinations[trial % len(combinations)]
                                    noisy = [
                                        (uv + rng.normal(0, noise, size=2), matrix)
                                        for _, uv, matrix in selected
                                    ]
                                    estimate = triangulate(noisy)
                                    errors[(view_count, noise)].append(float(np.linalg.norm(estimate - point_world)))

    conditions = []
    for views in (2, 3, 4):
        for noise in noises:
            values = errors[(views, noise)]
            if values:
                conditions.append(
                    {"views": views, "keypoint_noise_sigma_px": noise, **percentile_summary(values)}
                )
    config = {
        "sequences": args.sequences,
        "frames": list(range(args.start, args.end + 1, args.step)),
        "trials": args.trials,
        "seed": args.seed,
        "input_2d_is_projected_ground_truth": True,
        "models_real_detector_occlusion_or_identity_errors": False,
        "mean_visible_exo_cameras": float(np.mean(visible_counts)),
        "fraction_visible_in_all_four": float(np.mean(np.asarray(visible_counts) == 4)),
    }
    args.output_dir.mkdir(parents=True, exist_ok=True)
    (args.output_dir / "summary.json").write_text(
        json.dumps({"config": config, "conditions": conditions}, indent=2) + "\n", encoding="utf-8"
    )
    lines = [
        "# H2O四路exo手关节三角化几何审计",
        "",
        "二维点由真值3D投影后加入高斯噪声，因此这是标定几何上限，不含真实检测器的遮挡、错检和身份交换。",
        "",
        f"平均可见exo数：{config['mean_visible_exo_cameras']:.2f}；四路均在画面内：{config['fraction_visible_in_all_four']:.1%}。",
        "",
        "| 视角数 | 2D噪声σ(px) | 3D mean(mm) | median(mm) | p95(mm) |",
        "|---:|---:|---:|---:|---:|",
    ]
    for row in conditions:
        lines.append(
            f"| {row['views']} | {row['keypoint_noise_sigma_px']:.0f} | {row['mean_mm']:.2f} | "
            f"{row['median_mm']:.2f} | {row['p95_mm']:.2f} |"
        )
    lines += [
        "",
        "只有当真实2D检测误差与可见性审计也落在可接受范围，三角化状态才能进入推理接口；否则仍只能作为Oracle上限。",
        "",
    ]
    (args.output_dir / "REPORT.md").write_text("\n".join(lines), encoding="utf-8")
    print(json.dumps({"output_dir": str(args.output_dir), "config": config, "conditions": conditions}, indent=2))


if __name__ == "__main__":
    main()
