#!/usr/bin/env python3
"""Calibrated RGB-D point reprojection with a target-view z-buffer."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
from PIL import Image


@dataclass
class ReprojectionResult:
    rgb: np.ndarray
    depth_m: np.ndarray
    valid: np.ndarray
    source_point_count: int
    projected_point_count: int


def load_intrinsics(path: str | Path) -> np.ndarray:
    values = np.loadtxt(path, dtype=np.float64).reshape(-1)
    if len(values) < 6:
        raise ValueError(f"Expected fx fy cx cy width height in {path}")
    return values[:6]


def load_pose(path: str | Path) -> np.ndarray:
    values = np.loadtxt(path, dtype=np.float64).reshape(-1)
    if len(values) != 16:
        raise ValueError(f"Expected a 4x4 pose in {path}, got {len(values)} values")
    return values.reshape(4, 4)


def reproject_rgbd(
    source_rgb: np.ndarray,
    source_depth_mm: np.ndarray,
    source_intrinsics: np.ndarray,
    source_pose_world: np.ndarray,
    target_intrinsics: np.ndarray,
    target_pose_world: np.ndarray,
    output_size: tuple[int, int] | None = None,
    source_stride: int = 1,
    min_depth_m: float = 0.1,
    max_depth_m: float = 5.0,
) -> ReprojectionResult:
    """Reproject one source RGB-D frame to the target camera.

    Poses follow the H2O camera-to-world convention. Only source-view data and
    calibration are used; target RGB/depth are not inputs to the reprojection.
    """
    if source_stride < 1:
        raise ValueError("source_stride must be positive")
    source_height, source_width = source_depth_mm.shape
    if source_rgb.shape[:2] != (source_height, source_width):
        raise ValueError("RGB and depth sizes differ")
    target_width_native = int(round(target_intrinsics[4]))
    target_height_native = int(round(target_intrinsics[5]))
    output_width, output_height = output_size or (target_width_native, target_height_native)
    ys = np.arange(0, source_height, source_stride, dtype=np.int32)
    xs = np.arange(0, source_width, source_stride, dtype=np.int32)
    grid_x, grid_y = np.meshgrid(xs, ys)
    depth_m = source_depth_mm[::source_stride, ::source_stride].astype(np.float64) / 1000.0
    valid_depth = np.isfinite(depth_m) & (depth_m >= min_depth_m) & (depth_m <= max_depth_m)
    x = grid_x[valid_depth].astype(np.float64)
    y = grid_y[valid_depth].astype(np.float64)
    z = depth_m[valid_depth]
    fx, fy, cx, cy = source_intrinsics[:4]
    points_source = np.column_stack(((x - cx) * z / fx, (y - cy) * z / fy, z))
    colors = source_rgb[grid_y[valid_depth], grid_x[valid_depth]]

    target_from_source = np.linalg.inv(target_pose_world) @ source_pose_world
    points_target = points_source @ target_from_source[:3, :3].T + target_from_source[:3, 3]
    target_z = points_target[:, 2]
    in_front = target_z > 1e-5
    points_target = points_target[in_front]
    target_z = target_z[in_front]
    colors = colors[in_front]
    scale_x = output_width / target_width_native
    scale_y = output_height / target_height_native
    target_fx, target_fy, target_cx, target_cy = target_intrinsics[:4]
    target_u = np.rint((target_fx * points_target[:, 0] / target_z + target_cx) * scale_x).astype(np.int32)
    target_v = np.rint((target_fy * points_target[:, 1] / target_z + target_cy) * scale_y).astype(np.int32)
    in_image = (
        (target_u >= 0)
        & (target_u < output_width)
        & (target_v >= 0)
        & (target_v < output_height)
    )
    target_u = target_u[in_image]
    target_v = target_v[in_image]
    target_z = target_z[in_image]
    colors = colors[in_image]

    pixel_index = target_v.astype(np.int64) * output_width + target_u
    order = np.lexsort((target_z, pixel_index))
    sorted_pixels = pixel_index[order]
    first = np.empty(len(order), dtype=bool)
    if len(order):
        first[0] = True
        first[1:] = sorted_pixels[1:] != sorted_pixels[:-1]
    winners = order[first]
    winning_pixels = pixel_index[winners]
    output_rgb = np.zeros((output_height * output_width, 3), dtype=np.uint8)
    output_depth = np.full(output_height * output_width, np.nan, dtype=np.float32)
    output_valid = np.zeros(output_height * output_width, dtype=bool)
    output_rgb[winning_pixels] = colors[winners]
    output_depth[winning_pixels] = target_z[winners].astype(np.float32)
    output_valid[winning_pixels] = True
    return ReprojectionResult(
        rgb=output_rgb.reshape(output_height, output_width, 3),
        depth_m=output_depth.reshape(output_height, output_width),
        valid=output_valid.reshape(output_height, output_width),
        source_point_count=int(valid_depth.sum()),
        projected_point_count=int(len(winning_pixels)),
    )


def load_and_reproject(
    source_camera: Path,
    target_camera: Path,
    frame: int,
    output_size: tuple[int, int] | None = None,
    source_stride: int = 1,
) -> tuple[ReprojectionResult, np.ndarray, np.ndarray]:
    stem = f"{frame:06d}"
    source_rgb = np.asarray(Image.open(source_camera / "rgb" / f"{stem}.png").convert("RGB"))
    source_depth = np.asarray(Image.open(source_camera / "depth" / f"{stem}.png"))
    result = reproject_rgbd(
        source_rgb=source_rgb,
        source_depth_mm=source_depth,
        source_intrinsics=load_intrinsics(source_camera / "cam_intrinsics.txt"),
        source_pose_world=load_pose(source_camera / "cam_pose" / f"{stem}.txt"),
        target_intrinsics=load_intrinsics(target_camera / "cam_intrinsics.txt"),
        target_pose_world=load_pose(target_camera / "cam_pose" / f"{stem}.txt"),
        output_size=output_size,
        source_stride=source_stride,
    )
    target_rgb = np.asarray(Image.open(target_camera / "rgb" / f"{stem}.png").convert("RGB"))
    target_depth = np.asarray(Image.open(target_camera / "depth" / f"{stem}.png"))
    if output_size and target_rgb.shape[1::-1] != output_size:
        target_rgb = np.asarray(Image.fromarray(target_rgb).resize(output_size, Image.Resampling.BILINEAR))
        target_depth = np.asarray(Image.fromarray(target_depth).resize(output_size, Image.Resampling.NEAREST))
    return result, target_rgb, target_depth.astype(np.float32) / 1000.0


def audit_reprojection(
    result: ReprojectionResult,
    target_rgb: np.ndarray,
    target_depth_m: np.ndarray,
    depth_tolerance_m: float = 0.03,
) -> tuple[dict[str, float | int], dict[str, np.ndarray]]:
    target_depth_valid = np.isfinite(target_depth_m) & (target_depth_m > 0)
    comparable = result.valid & target_depth_valid
    depth_error = np.full(result.depth_m.shape, np.nan, dtype=np.float32)
    depth_error[comparable] = result.depth_m[comparable] - target_depth_m[comparable]
    consistent = comparable & (np.abs(depth_error) <= depth_tolerance_m)
    behind_target = comparable & (depth_error > depth_tolerance_m)
    in_front_target = comparable & (depth_error < -depth_tolerance_m)
    rgb_error = np.abs(result.rgb.astype(np.float32) - target_rgb.astype(np.float32)) / 255.0
    consistent_mse = float(np.square(rgb_error[consistent]).mean()) if np.any(consistent) else float("nan")
    metrics: dict[str, float | int] = {
        "source_depth_points": result.source_point_count,
        "zbuffer_pixels": result.projected_point_count,
        "raw_coverage_fraction": float(result.valid.mean()),
        "target_depth_comparable_fraction": float(comparable.mean()),
        "depth_consistent_fraction_of_image": float(consistent.mean()),
        "depth_consistent_fraction_of_warp": float(consistent.sum() / max(1, result.valid.sum())),
        "behind_target_fraction_of_warp": float(behind_target.sum() / max(1, result.valid.sum())),
        "in_front_target_fraction_of_warp": float(in_front_target.sum() / max(1, result.valid.sum())),
        "depth_abs_error_median_m": float(np.nanmedian(np.abs(depth_error))) if np.any(comparable) else float("nan"),
        "depth_abs_error_p95_m": float(np.nanpercentile(np.abs(depth_error), 95)) if np.any(comparable) else float("nan"),
        "consistent_rgb_l1": float(rgb_error[consistent].mean()) if np.any(consistent) else float("nan"),
        "consistent_rgb_psnr_db": float(-10 * np.log10(max(consistent_mse, 1e-12))),
    }
    return metrics, {
        "comparable": comparable,
        "consistent": consistent,
        "behind_target": behind_target,
        "in_front_target": in_front_target,
        "depth_error_m": depth_error,
    }
