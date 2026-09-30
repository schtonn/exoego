#!/usr/bin/env python3
"""Measure how target-camera pose errors damage exo-to-ego reprojection.

This is an audit, not a target-pose estimator.  It deliberately perturbs the
ground-truth H2O cam4 pose while keeping the target RGB-D used for evaluation
fixed.  The result establishes the pose-accuracy budget that a deployable
exo-only camera module would have to meet.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
from dataclasses import dataclass
from pathlib import Path

import numpy as np
from PIL import Image

from h2o_geometric_baseline.reprojection import (
    audit_reprojection,
    load_intrinsics,
    load_pose,
    reproject_rgbd,
)


DEFAULT_RAW = Path("datasets/H2O/raw")
DEFAULT_OUTPUT = Path("datasets/H2O/experiments/trajectory_sensitivity")


@dataclass(frozen=True)
class Perturbation:
    name: str
    rotation_axis: tuple[float, float, float] = (0.0, 0.0, 0.0)
    rotation_deg: float = 0.0
    translation_axis: tuple[float, float, float] = (0.0, 0.0, 0.0)
    translation_cm: float = 0.0


def _rotation(axis: tuple[float, float, float], angle_deg: float) -> np.ndarray:
    vector = np.asarray(axis, dtype=np.float64)
    norm = np.linalg.norm(vector)
    if norm == 0 or angle_deg == 0:
        return np.eye(3, dtype=np.float64)
    x, y, z = vector / norm
    angle = math.radians(angle_deg)
    c, s, one_minus_c = math.cos(angle), math.sin(angle), 1.0 - math.cos(angle)
    return np.array(
        [
            [c + x * x * one_minus_c, x * y * one_minus_c - z * s, x * z * one_minus_c + y * s],
            [y * x * one_minus_c + z * s, c + y * y * one_minus_c, y * z * one_minus_c - x * s],
            [z * x * one_minus_c - y * s, z * y * one_minus_c + x * s, c + z * z * one_minus_c],
        ],
        dtype=np.float64,
    )


def perturb_pose(pose_world: np.ndarray, perturbation: Perturbation, sign: int) -> np.ndarray:
    """Apply an error in the local cam4 coordinate system."""
    local_error = np.eye(4, dtype=np.float64)
    local_error[:3, :3] = _rotation(
        perturbation.rotation_axis, sign * perturbation.rotation_deg
    )
    local_error[:3, 3] = (
        np.asarray(perturbation.translation_axis, dtype=np.float64)
        * sign
        * perturbation.translation_cm
        / 100.0
    )
    return pose_world @ local_error


def conditions() -> list[Perturbation]:
    values_deg = (1.0, 3.0, 5.0, 10.0)
    values_cm = (1.0, 3.0, 5.0, 10.0)
    result = [Perturbation("oracle_pose")]
    result += [Perturbation(f"yaw_{v:g}deg", (0, 1, 0), v) for v in values_deg]
    result += [Perturbation(f"pitch_{v:g}deg", (1, 0, 0), v) for v in values_deg]
    result += [Perturbation(f"lateral_{v:g}cm", translation_axis=(1, 0, 0), translation_cm=v) for v in values_cm]
    result += [Perturbation(f"forward_{v:g}cm", translation_axis=(0, 0, 1), translation_cm=v) for v in values_cm]
    return result


def load_rgb_depth(camera: Path, frame: int) -> tuple[np.ndarray, np.ndarray]:
    stem = f"{frame:06d}"
    rgb = np.asarray(Image.open(camera / "rgb" / f"{stem}.png").convert("RGB"))
    depth = np.asarray(Image.open(camera / "depth" / f"{stem}.png"))
    return rgb, depth


def resize_target(
    rgb: np.ndarray, depth_mm: np.ndarray, size: tuple[int, int]
) -> tuple[np.ndarray, np.ndarray]:
    rgb_resized = np.asarray(Image.fromarray(rgb).resize(size, Image.Resampling.BILINEAR))
    depth_resized = np.asarray(Image.fromarray(depth_mm).resize(size, Image.Resampling.NEAREST))
    return rgb_resized, depth_resized.astype(np.float32) / 1000.0


def extra_rgb_metrics(result_rgb: np.ndarray, valid: np.ndarray, target_rgb: np.ndarray) -> dict[str, float]:
    if not np.any(valid):
        return {"all_warp_rgb_l1": float("nan"), "all_warp_rgb_psnr_db": float("nan")}
    error = (result_rgb[valid].astype(np.float32) - target_rgb[valid].astype(np.float32)) / 255.0
    mse = float(np.square(error).mean())
    return {
        "all_warp_rgb_l1": float(np.abs(error).mean()),
        "all_warp_rgb_psnr_db": float(-10.0 * np.log10(max(mse, 1e-12))),
    }


def mean_finite(rows: list[dict], key: str) -> float:
    values = np.asarray([row[key] for row in rows], dtype=np.float64)
    return float(np.nanmean(values))


def write_csv(path: Path, rows: list[dict]) -> None:
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def write_report(path: Path, config: dict, summary: list[dict]) -> None:
    baseline = summary[0]
    lines = [
        "# H2O目标ego相机位姿误差敏感性",
        "",
        "本实验只回答一个问题：如果cam4位姿不是Oracle，而是带误差的估计值，",
        "cam3 RGB-D几何重投影会退化多快。target RGB-D只用于审计，未参与重投影。",
        "",
        f"- 序列：`{config['sequence']}`，源视角：`{config['source_camera']}`",
        f"- 帧：{config['start']}–{config['end']}，步长{config['step']}，共{config['frames']}帧",
        f"- 分辨率：{config['width']}×{config['height']}，深度一致阈值：{config['depth_tolerance_m'] * 100:g} cm",
        "- 旋转和平移误差施加在cam4局部坐标系；非零固定误差对正负方向取平均。",
        "",
        "| 位姿条件 | target深度一致/整图 | 一致/warp | 全warp RGB L1 | 全warp PSNR | 相对Oracle保留率 |",
        "|---|---:|---:|---:|---:|---:|",
    ]
    for row in summary:
        retention = row["depth_consistent_fraction_of_image"] / max(
            baseline["depth_consistent_fraction_of_image"], 1e-12
        )
        lines.append(
            f"| {row['condition']} | {row['depth_consistent_fraction_of_image']:.4f} | "
            f"{row['depth_consistent_fraction_of_warp']:.4f} | {row['all_warp_rgb_l1']:.4f} | "
            f"{row['all_warp_rgb_psnr_db']:.2f} | {retention:.1%} |"
        )
    lines += [
        "",
        "说明：`一致/整图`是最直接的可用几何覆盖；`全warp RGB`不筛掉深度不一致像素，",
        "因此不会因只保留少量容易像素而虚高。该实验评估的是对真实cam4复现的要求，",
        "不能证明从exo能够估出这些位姿。",
        "",
    ]
    path.write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--raw-root", type=Path, default=DEFAULT_RAW)
    parser.add_argument("--sequence", default="subject1/h1/0")
    parser.add_argument("--source-camera", choices=["cam0", "cam1", "cam2", "cam3"], default="cam3")
    parser.add_argument("--start", type=int, default=70)
    parser.add_argument("--end", type=int, default=129)
    parser.add_argument("--step", type=int, default=3)
    parser.add_argument("--width", type=int, default=320)
    parser.add_argument("--height", type=int, default=180)
    parser.add_argument("--source-stride", type=int, default=2)
    parser.add_argument("--depth-tolerance-m", type=float, default=0.03)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    args = parser.parse_args()
    if args.end < args.start or args.step < 1:
        parser.error("require end >= start and step >= 1")

    sequence_root = args.raw_root / args.sequence
    source = sequence_root / args.source_camera
    target = sequence_root / "cam4"
    size = (args.width, args.height)
    source_intrinsics = load_intrinsics(source / "cam_intrinsics.txt")
    target_intrinsics = load_intrinsics(target / "cam_intrinsics.txt")
    frame_numbers = list(range(args.start, args.end + 1, args.step))
    per_frame: list[dict] = []

    for frame in frame_numbers:
        source_rgb, source_depth = load_rgb_depth(source, frame)
        target_rgb_native, target_depth_native = load_rgb_depth(target, frame)
        target_rgb, target_depth = resize_target(target_rgb_native, target_depth_native, size)
        source_pose = load_pose(source / "cam_pose" / f"{frame:06d}.txt")
        true_target_pose = load_pose(target / "cam_pose" / f"{frame:06d}.txt")
        for condition in conditions():
            signs = (1,) if condition.name == "oracle_pose" else (-1, 1)
            for sign in signs:
                result = reproject_rgbd(
                    source_rgb,
                    source_depth,
                    source_intrinsics,
                    source_pose,
                    target_intrinsics,
                    perturb_pose(true_target_pose, condition, sign),
                    output_size=size,
                    source_stride=args.source_stride,
                )
                metrics, _ = audit_reprojection(
                    result, target_rgb, target_depth, args.depth_tolerance_m
                )
                metrics.update(extra_rgb_metrics(result.rgb, result.valid, target_rgb))
                per_frame.append(
                    {
                        "condition": condition.name,
                        "frame": frame,
                        "sign": sign,
                        **metrics,
                    }
                )

    metric_names = [
        "raw_coverage_fraction",
        "depth_consistent_fraction_of_image",
        "depth_consistent_fraction_of_warp",
        "depth_abs_error_median_m",
        "depth_abs_error_p95_m",
        "consistent_rgb_l1",
        "consistent_rgb_psnr_db",
        "all_warp_rgb_l1",
        "all_warp_rgb_psnr_db",
    ]
    summary = []
    for condition in conditions():
        rows = [row for row in per_frame if row["condition"] == condition.name]
        summary.append(
            {
                "condition": condition.name,
                "rotation_deg": condition.rotation_deg,
                "translation_cm": condition.translation_cm,
                "evaluations": len(rows),
                **{name: mean_finite(rows, name) for name in metric_names},
            }
        )

    args.output_dir.mkdir(parents=True, exist_ok=True)
    config = {
        "sequence": args.sequence,
        "source_camera": args.source_camera,
        "target_camera": "cam4",
        "start": args.start,
        "end": args.end,
        "step": args.step,
        "frames": len(frame_numbers),
        "width": args.width,
        "height": args.height,
        "source_stride": args.source_stride,
        "depth_tolerance_m": args.depth_tolerance_m,
        "target_rgb_or_depth_used_for_reprojection": False,
        "target_rgb_or_depth_used_for_audit": True,
    }
    write_csv(args.output_dir / "per_frame.csv", per_frame)
    write_csv(args.output_dir / "summary.csv", summary)
    (args.output_dir / "summary.json").write_text(
        json.dumps({"config": config, "conditions": summary}, indent=2) + "\n", encoding="utf-8"
    )
    write_report(args.output_dir / "REPORT.md", config, summary)
    print(json.dumps({"output_dir": str(args.output_dir), "config": config, "conditions": summary}, indent=2))


if __name__ == "__main__":
    main()
