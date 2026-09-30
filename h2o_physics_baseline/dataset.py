#!/usr/bin/env python3
"""Dataset and deterministic target-view physical conditioning for H2O."""

from __future__ import annotations

import csv
import json
import math
from collections import OrderedDict
from pathlib import Path
from typing import Any

import numpy as np
from PIL import Image

from h2o_geometric_baseline.reprojection import load_intrinsics, load_pose, reproject_rgbd


DEFAULT_INDEX = Path("datasets/H2O/oracle_state/paired_physical_clips.csv")
DEFAULT_STATS = Path("datasets/H2O/oracle_state/train_feature_stats.json")


def _read_intrinsics(path: str | Path) -> np.ndarray:
    values = np.loadtxt(path, dtype=np.float32).reshape(-1)
    if len(values) < 4:
        raise ValueError(f"Expected at least fx, fy, cx, cy in {path}, got {values}")
    return values[:4]


def _rotation_6d(rotation: np.ndarray) -> np.ndarray:
    # Keep the first two columns in the same row-major order used by
    # prepare_training_metadata.py: r00,r01,r10,r11,r20,r21.
    return rotation[:, :, :2].reshape(len(rotation), 6)


def _normalise(value: np.ndarray, stats: dict[str, Any], name: str) -> np.ndarray:
    feature = stats["features"][name]
    mean = np.asarray(feature["mean"], dtype=np.float32)
    std = np.maximum(np.asarray(feature["std"], dtype=np.float32), 1e-6)
    return (value.astype(np.float32) - mean) / std


def _gaussian_maps(
    points_xyz: np.ndarray,
    weights: np.ndarray,
    intrinsics: np.ndarray,
    output_size: int,
    original_size: tuple[int, int] = (1280, 720),
    sigma_px: float = 3.0,
) -> np.ndarray:
    """Rasterize weighted 3D camera-frame points into low-resolution heatmaps."""
    channels, points = weights.shape
    maps = np.zeros((channels, output_size, output_size), dtype=np.float32)
    z = points_xyz[:, 2]
    valid = np.isfinite(points_xyz).all(axis=1) & (z > 1e-5)
    if not np.any(valid):
        return maps
    fx, fy, cx, cy = intrinsics
    u = (fx * points_xyz[:, 0] / np.maximum(z, 1e-5) + cx) * output_size / original_size[0]
    v = (fy * points_xyz[:, 1] / np.maximum(z, 1e-5) + cy) * output_size / original_size[1]
    radius = max(1, int(math.ceil(3 * sigma_px)))
    for point_index in np.flatnonzero(valid):
        center_x = float(u[point_index])
        center_y = float(v[point_index])
        x0 = max(0, int(math.floor(center_x)) - radius)
        x1 = min(output_size, int(math.floor(center_x)) + radius + 1)
        y0 = max(0, int(math.floor(center_y)) - radius)
        y1 = min(output_size, int(math.floor(center_y)) + radius + 1)
        if x0 >= x1 or y0 >= y1:
            continue
        yy, xx = np.mgrid[y0:y1, x0:x1]
        kernel = np.exp(-((xx - center_x) ** 2 + (yy - center_y) ** 2) / (2 * sigma_px**2))
        maps[:, y0:y1, x0:x1] = np.maximum(
            maps[:, y0:y1, x0:x1], weights[:, point_index, None, None] * kernel[None]
        )
    return maps


def _gaussian_flow_maps(
    initial_xyz: np.ndarray,
    current_xyz: np.ndarray,
    confidence: np.ndarray,
    intrinsics: np.ndarray,
    output_size: int,
    original_size: tuple[int, int] = (1280, 720),
    sigma_px: float = 3.0,
) -> np.ndarray:
    """Rasterize signed initial-to-current image displacement at current joints."""
    maps = np.zeros((2, output_size, output_size), dtype=np.float32)
    weight_sum = np.zeros((output_size, output_size), dtype=np.float32)
    initial_z = initial_xyz[:, 2]
    current_z = current_xyz[:, 2]
    valid = (
        np.isfinite(initial_xyz).all(axis=1)
        & np.isfinite(current_xyz).all(axis=1)
        & (initial_z > 1e-5)
        & (current_z > 1e-5)
        & (confidence > 0)
    )
    if not np.any(valid):
        return maps
    fx, fy, cx, cy = intrinsics

    def project(points: np.ndarray, depth: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        u = (fx * points[:, 0] / np.maximum(depth, 1e-5) + cx) * output_size / original_size[0]
        v = (fy * points[:, 1] / np.maximum(depth, 1e-5) + cy) * output_size / original_size[1]
        return u, v

    initial_u, initial_v = project(initial_xyz, initial_z)
    current_u, current_v = project(current_xyz, current_z)
    displacement = np.stack(
        ((current_u - initial_u) / output_size, (current_v - initial_v) / output_size), axis=1
    )
    displacement = np.clip(displacement, -1.0, 1.0)
    radius = max(1, int(math.ceil(3 * sigma_px)))
    for point_index in np.flatnonzero(valid):
        center_x = float(current_u[point_index])
        center_y = float(current_v[point_index])
        x0 = max(0, int(math.floor(center_x)) - radius)
        x1 = min(output_size, int(math.floor(center_x)) + radius + 1)
        y0 = max(0, int(math.floor(center_y)) - radius)
        y1 = min(output_size, int(math.floor(center_y)) + radius + 1)
        if x0 >= x1 or y0 >= y1:
            continue
        yy, xx = np.mgrid[y0:y1, x0:x1]
        kernel = np.exp(-((xx - center_x) ** 2 + (yy - center_y) ** 2) / (2 * sigma_px**2))
        weight = confidence[point_index] * kernel
        maps[:, y0:y1, x0:x1] += displacement[point_index, :, None, None] * weight[None]
        weight_sum[y0:y1, x0:x1] += weight
    maps /= np.maximum(weight_sum[None], 1e-8)
    return maps


def _dense_mask_flow_maps(
    initial_xyz: np.ndarray,
    current_xyz: np.ndarray,
    confidence: np.ndarray,
    intrinsics: np.ndarray,
    initial_masks: np.ndarray,
    output_size: int,
    original_size: tuple[int, int] = (1280, 720),
) -> np.ndarray:
    """Transport initial hand masks with nearest-joint piecewise translations."""
    if initial_masks.shape != (2, output_size, output_size):
        raise ValueError(
            f"Expected two {output_size}x{output_size} initial masks, got {initial_masks.shape}"
        )
    maps = np.zeros((5, output_size, output_size), dtype=np.float32)
    flow_weight = np.zeros((output_size, output_size), dtype=np.float32)
    yy, xx = np.mgrid[:output_size, :output_size].astype(np.float32)
    fx, fy, cx, cy = intrinsics

    def project(points: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        depth = points[:, 2]
        valid = np.isfinite(points).all(axis=1) & (depth > 1e-5)
        u = (fx * points[:, 0] / np.maximum(depth, 1e-5) + cx) * output_size / original_size[0]
        v = (fy * points[:, 1] / np.maximum(depth, 1e-5) + cy) * output_size / original_size[1]
        return u.astype(np.float32), v.astype(np.float32), valid

    def bilinear(mask: np.ndarray, source_x: np.ndarray, source_y: np.ndarray) -> np.ndarray:
        x0 = np.floor(source_x).astype(np.int32)
        y0 = np.floor(source_y).astype(np.int32)
        x1 = x0 + 1
        y1 = y0 + 1
        valid = (x0 >= 0) & (y0 >= 0) & (x1 < output_size) & (y1 < output_size)
        result = np.zeros_like(source_x, dtype=np.float32)
        if not np.any(valid):
            return result
        wx = source_x - x0
        wy = source_y - y0
        result[valid] = (
            mask[y0[valid], x0[valid]] * (1 - wx[valid]) * (1 - wy[valid])
            + mask[y0[valid], x1[valid]] * wx[valid] * (1 - wy[valid])
            + mask[y1[valid], x0[valid]] * (1 - wx[valid]) * wy[valid]
            + mask[y1[valid], x1[valid]] * wx[valid] * wy[valid]
        )
        return result

    for hand in range(2):
        initial_u, initial_v, initial_valid = project(initial_xyz[hand])
        current_u, current_v, current_valid = project(current_xyz[hand])
        valid = initial_valid & current_valid & (confidence[hand] > 0)
        if not np.any(valid) or not np.any(initial_masks[hand] > 0):
            continue
        current_points = np.stack((current_u[valid], current_v[valid]), axis=1)
        displacement = np.stack(
            (current_u[valid] - initial_u[valid], current_v[valid] - initial_v[valid]), axis=1
        )
        squared = (
            (xx[..., None] - current_points[:, 0]) ** 2
            + (yy[..., None] - current_points[:, 1]) ** 2
        )
        nearest = np.argmin(squared, axis=2)
        dense_dx = displacement[nearest, 0]
        dense_dy = displacement[nearest, 1]
        current_mask = bilinear(initial_masks[hand], xx - dense_dx, yy - dense_dy)
        current_mask = np.clip(current_mask, 0, 1)
        maps[hand] = current_mask
        maps[2] += dense_dx / output_size * current_mask
        maps[3] += dense_dy / output_size * current_mask
        flow_weight += current_mask
    nonzero = flow_weight > 1e-6
    maps[2, nonzero] /= flow_weight[nonzero]
    maps[3, nonzero] /= flow_weight[nonzero]
    return maps


class _NpzCache:
    def __init__(self, capacity: int = 8):
        self.capacity = capacity
        self.entries: OrderedDict[str, dict[str, np.ndarray]] = OrderedDict()

    def get(self, path: str | Path) -> dict[str, np.ndarray]:
        key = str(path)
        if key in self.entries:
            self.entries.move_to_end(key)
            return self.entries[key]
        with np.load(key) as archive:
            value = {name: archive[name] for name in archive.files}
        self.entries[key] = value
        if len(self.entries) > self.capacity:
            self.entries.popitem(last=False)
        return value


class H2OPhysicalClipDataset:
    """Read paired RGB clips and their compact Oracle physical condition.

    The returned physical vector has 317 dimensions per frame:
    hand positions (126), hand-object offsets (126), object translation/rotation
    (9), object and camera velocities (12), joint surface proximity (42), and
    left/right visibility (2). The five spatial maps are left/right hand layout,
    left/right contact-weighted layout, and object-center layout in cam4.
    """

    physical_dim = 317
    map_channels = 5

    def __init__(
        self,
        index_path: str | Path = DEFAULT_INDEX,
        split: str = "train",
        frames_per_clip: int = 8,
        image_size: int = 128,
        stats_path: str | Path = DEFAULT_STATS,
        source_cameras: tuple[str, ...] | None = None,
        max_samples: int | None = None,
        cache_size: int = 8,
        include_reprojection: bool = False,
        reprojection_source_stride: int = 2,
        combine_source_cameras: bool = False,
        student_state_root: str | Path | None = None,
        student_mask_root: str | Path | None = None,
        oracle_hand_only: bool = False,
        student_flow: bool = False,
        student_mask_flow: bool = False,
        student_dense_mask_flow: bool = False,
        source_hand_maps: bool = False,
        source_joint_geometry: bool = False,
    ) -> None:
        if frames_per_clip < 2:
            raise ValueError("frames_per_clip must be at least 2")
        if image_size < 32:
            raise ValueError("image_size must be at least 32")
        with Path(index_path).open(encoding="utf-8") as handle:
            rows = [row for row in csv.DictReader(handle) if row["split"] == split]
        if source_cameras:
            rows = [row for row in rows if row["source_camera"] in source_cameras]
        if combine_source_cameras:
            camera_order = tuple(source_cameras or ("cam0", "cam1", "cam2", "cam3"))
            grouped: OrderedDict[str, list[dict[str, str]]] = OrderedDict()
            for row in rows:
                grouped.setdefault(row["clip_id"], []).append(row)
            combined = []
            for clip_rows in grouped.values():
                by_camera = {row["source_camera"]: row for row in clip_rows}
                if not all(camera in by_camera for camera in camera_order):
                    continue
                representative = dict(by_camera[camera_order[0]])
                representative["source_camera"] = "+".join(camera_order)
                representative["source_rgb_dirs"] = json.dumps(
                    [by_camera[camera]["source_rgb_dir"] for camera in camera_order]
                )
                combined.append(representative)
            rows = combined
        if max_samples is not None and max_samples < len(rows):
            # Cover the full split for quick experiments instead of taking only
            # the first sequence/camera block from the ordered CSV.
            selected = np.linspace(0, len(rows) - 1, max_samples, dtype=np.int64)
            rows = [rows[int(index)] for index in selected]
        if not rows:
            raise ValueError(f"No rows found for split={split!r}")
        self.rows = rows
        self.split = split
        self.frames_per_clip = frames_per_clip
        self.image_size = image_size
        self.stats = json.loads(Path(stats_path).read_text(encoding="utf-8"))
        self.include_reprojection = include_reprojection
        self.combine_source_cameras = combine_source_cameras
        self.reprojection_source_stride = reprojection_source_stride
        self.student_state_root = Path(student_state_root) if student_state_root else None
        self.student_mask_root = Path(student_mask_root) if student_mask_root else None
        self.oracle_hand_only = oracle_hand_only
        self.student_flow = student_flow
        self.student_mask_flow = student_mask_flow
        self.student_dense_mask_flow = student_dense_mask_flow
        self.source_hand_maps = source_hand_maps
        self.source_joint_geometry = source_joint_geometry
        self.state_cache = _NpzCache(cache_size)
        self.contact_cache = _NpzCache(cache_size)
        self.student_cache = _NpzCache(cache_size)
        self.student_mask_cache = _NpzCache(cache_size)

    def _source_hand_layouts(
        self,
        row: dict[str, str],
        frame_numbers: np.ndarray,
        student_state: dict[str, np.ndarray] | None,
    ) -> np.ndarray:
        """Project exo-estimated world hands into each calibrated source view."""
        if self.combine_source_cameras:
            source_directories = json.loads(row["source_rgb_dirs"])
        else:
            source_directories = [row["source_rgb_dir"]]
        shape = (
            len(source_directories),
            len(frame_numbers),
            2,
            self.image_size,
            self.image_size,
        )
        if not self.source_hand_maps:
            result = np.zeros(shape, dtype=np.float32)
            return result if self.combine_source_cameras else result[0]
        if student_state is None:
            raise ValueError("source_hand_maps requires exo-estimated student state")
        student_indices = np.searchsorted(student_state["frames"], frame_numbers)
        if np.any(student_indices >= len(student_state["frames"])) or not np.array_equal(
            student_state["frames"][student_indices], frame_numbers
        ):
            raise IndexError("Student state does not contain every requested source frame")
        world_hands = student_state["hand_joints_world_m"][student_indices].astype(np.float32)
        confidence = student_state["joint_confidence"][student_indices].astype(np.float32)
        result = np.zeros(shape, dtype=np.float32)
        for view_index, directory in enumerate(source_directories):
            camera_root = Path(directory).parent
            intrinsics_values = np.loadtxt(
                camera_root / "cam_intrinsics.txt", dtype=np.float32
            ).reshape(-1)
            intrinsics = intrinsics_values[:4]
            original_size = (
                int(intrinsics_values[4]) if len(intrinsics_values) > 4 else 1280,
                int(intrinsics_values[5]) if len(intrinsics_values) > 5 else 720,
            )
            for time_index, frame in enumerate(frame_numbers):
                world_to_camera = np.linalg.inv(
                    load_pose(camera_root / "cam_pose" / f"{int(frame):06d}.txt")
                )
                points = world_hands[time_index].reshape(42, 3)
                camera_points = points @ world_to_camera[:3, :3].T + world_to_camera[:3, 3]
                weights = np.zeros((2, 42), dtype=np.float32)
                weights[0, :21] = confidence[time_index, 0]
                weights[1, 21:] = confidence[time_index, 1]
                result[view_index, time_index] = _gaussian_maps(
                    camera_points,
                    weights,
                    intrinsics,
                    self.image_size,
                    original_size=original_size,
                )
        return result if self.combine_source_cameras else result[0]

    def _crossview_joint_coordinates(
        self,
        row: dict[str, str],
        frame_numbers: np.ndarray,
        state: dict[str, np.ndarray],
        indices: np.ndarray,
        student_state: dict[str, np.ndarray] | None,
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """Return calibrated source/initial-ego joint coordinates for feature lifting."""
        if self.combine_source_cameras:
            source_directories = json.loads(row["source_rgb_dirs"])
        else:
            source_directories = [row["source_rgb_dir"]]
        views = len(source_directories)
        time = len(frame_numbers)
        source_uv = np.zeros((views, time, 42, 2), dtype=np.float32)
        source_confidence = np.zeros((views, time, 42), dtype=np.float32)
        target_uv = np.zeros((time, 42, 2), dtype=np.float32)
        target_confidence = np.zeros((time, 42), dtype=np.float32)
        if not self.source_joint_geometry:
            if not self.combine_source_cameras:
                source_uv, source_confidence = source_uv[0], source_confidence[0]
            return source_uv, source_confidence, target_uv, target_confidence
        if student_state is None:
            raise ValueError("source_joint_geometry requires exo-estimated student state")
        student_indices = np.searchsorted(student_state["frames"], frame_numbers)
        if np.any(student_indices >= len(student_state["frames"])) or not np.array_equal(
            student_state["frames"][student_indices], frame_numbers
        ):
            raise IndexError("Student state does not contain every requested joint frame")
        world_points = student_state["hand_joints_world_m"][student_indices].reshape(
            time, 42, 3
        ).astype(np.float32)
        confidence = student_state["joint_confidence"][student_indices].reshape(
            time, 42
        ).astype(np.float32)

        def project(
            points: np.ndarray,
            world_to_camera: np.ndarray,
            intrinsics_path: Path,
        ) -> tuple[np.ndarray, np.ndarray]:
            values = np.loadtxt(intrinsics_path, dtype=np.float32).reshape(-1)
            fx, fy, cx, cy = values[:4]
            width = float(values[4]) if len(values) > 4 else 1280.0
            height = float(values[5]) if len(values) > 5 else 720.0
            camera = points @ world_to_camera[:3, :3].T + world_to_camera[:3, 3]
            depth = camera[:, 2]
            u = fx * camera[:, 0] / np.maximum(depth, 1e-5) + cx
            v = fy * camera[:, 1] / np.maximum(depth, 1e-5) + cy
            valid = (
                np.isfinite(camera).all(axis=1)
                & (depth > 1e-5)
                & (u >= 0)
                & (u < width)
                & (v >= 0)
                & (v < height)
            )
            uv = np.stack((2.0 * u / width - 1.0, 2.0 * v / height - 1.0), axis=1)
            return np.nan_to_num(uv).astype(np.float32), valid.astype(np.float32)

        for view_index, directory in enumerate(source_directories):
            camera_root = Path(directory).parent
            for time_index, frame in enumerate(frame_numbers):
                world_to_camera = np.linalg.inv(
                    load_pose(camera_root / "cam_pose" / f"{int(frame):06d}.txt")
                )
                uv, valid = project(
                    world_points[time_index],
                    world_to_camera,
                    camera_root / "cam_intrinsics.txt",
                )
                source_uv[view_index, time_index] = uv
                source_confidence[view_index, time_index] = confidence[time_index] * valid

        anchor_from_world = np.linalg.inv(state["camera_pose_world"][indices[0]])
        target_intrinsics = Path(row["target_intrinsics"])
        for time_index in range(time):
            uv, valid = project(world_points[time_index], anchor_from_world, target_intrinsics)
            target_uv[time_index] = uv
            target_confidence[time_index] = confidence[time_index] * valid
        if not self.combine_source_cameras:
            source_uv, source_confidence = source_uv[0], source_confidence[0]
        return source_uv, source_confidence, target_uv, target_confidence

    def __len__(self) -> int:
        return len(self.rows)

    def _load_rgb(self, directory: str, frame: int) -> np.ndarray:
        path = Path(directory) / f"{frame:06d}.png"
        with Image.open(path) as image:
            image = image.convert("RGB").resize(
                (self.image_size, self.image_size), Image.Resampling.BILINEAR
            )
            value = np.asarray(image, dtype=np.float32) / 255.0
        return np.moveaxis(value, -1, 0)

    def _geometric_condition(
        self, row: dict[str, str], frame_numbers: np.ndarray
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        if not self.include_reprojection:
            shape = (len(frame_numbers), 1, self.image_size, self.image_size)
            return (
                np.zeros((len(frame_numbers), 3, self.image_size, self.image_size), dtype=np.float32),
                np.zeros(shape, dtype=np.float32),
                np.zeros(shape, dtype=np.float32),
            )
        source_rgb_dir = Path(row["source_rgb_dir"])
        source_camera = source_rgb_dir.parent
        target_camera = Path(row["target_rgb_dir"]).parent
        source_intrinsics = load_intrinsics(source_camera / "cam_intrinsics.txt")
        target_intrinsics = load_intrinsics(target_camera / "cam_intrinsics.txt")
        rgb_values = []
        depth_values = []
        mask_values = []
        for frame in frame_numbers:
            stem = f"{int(frame):06d}"
            source_rgb = np.asarray(Image.open(source_camera / "rgb" / f"{stem}.png").convert("RGB"))
            source_depth = np.asarray(Image.open(source_camera / "depth" / f"{stem}.png"))
            result = reproject_rgbd(
                source_rgb,
                source_depth,
                source_intrinsics,
                load_pose(source_camera / "cam_pose" / f"{stem}.txt"),
                target_intrinsics,
                load_pose(target_camera / "cam_pose" / f"{stem}.txt"),
                output_size=(self.image_size, self.image_size),
                source_stride=self.reprojection_source_stride,
            )
            rgb_values.append(np.moveaxis(result.rgb.astype(np.float32) / 255.0, -1, 0))
            depth_values.append(np.nan_to_num(result.depth_m, nan=0.0)[None] / 5.0)
            mask_values.append(result.valid.astype(np.float32)[None])
        return np.stack(rgb_values), np.stack(depth_values), np.stack(mask_values)

    def _frame_indices(self, row: dict[str, str], state_frames: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        frame_numbers = np.rint(
            np.linspace(int(row["start_frame"]), int(row["end_frame"]), self.frames_per_clip)
        ).astype(np.int32)
        indices = np.searchsorted(state_frames, frame_numbers)
        if np.any(indices >= len(state_frames)) or not np.array_equal(state_frames[indices], frame_numbers):
            raise IndexError(f"State frames do not contain requested RGB frames for {row['pair_id']}")
        return frame_numbers, indices

    def _physical_condition(
        self,
        state: dict[str, np.ndarray],
        contact: dict[str, np.ndarray],
        indices: np.ndarray,
        intrinsics: np.ndarray,
    ) -> tuple[np.ndarray, np.ndarray]:
        hands = state["hand_joints_cam4_m"][indices].astype(np.float32)
        presence = state["hand_presence"][indices].astype(np.float32)
        object_pose = state["object_pose_cam4"][indices].astype(np.float32)
        object_translation = object_pose[:, :3, 3]
        relative = hands - object_translation[:, None, None]
        camera_rotation_world = state["camera_pose_world"][indices, :3, :3].astype(np.float32)

        def world_to_camera_vector(name: str) -> np.ndarray:
            world = state[name][indices].astype(np.float32)
            return np.einsum("tji,tj->ti", camera_rotation_world, world)

        hand_mask = presence[:, :, None, None]
        normalised_hands = np.nan_to_num(
            _normalise(hands, self.stats, "hand_joint_cam4_m"), nan=0.0, posinf=0.0, neginf=0.0
        ) * hand_mask
        normalised_relative = np.nan_to_num(
            _normalise(relative, self.stats, "hand_minus_object_cam4_m"),
            nan=0.0,
            posinf=0.0,
            neginf=0.0,
        ) * hand_mask
        groups = [
            normalised_hands.reshape(len(indices), -1),
            normalised_relative.reshape(len(indices), -1),
            _normalise(object_translation, self.stats, "object_translation_cam4_m"),
            _normalise(_rotation_6d(object_pose[:, :3, :3]), self.stats, "object_rotation_6d_cam4"),
            _normalise(
                world_to_camera_vector("object_linear_velocity_world_mps"),
                self.stats,
                "object_linear_velocity_cam4_mps",
            ),
            _normalise(
                world_to_camera_vector("object_angular_velocity_world_radps"),
                self.stats,
                "object_angular_velocity_cam4_radps",
            ),
            _normalise(
                world_to_camera_vector("camera_linear_velocity_world_mps"),
                self.stats,
                "camera_linear_velocity_local_mps",
            ),
            _normalise(
                world_to_camera_vector("camera_angular_velocity_world_radps"),
                self.stats,
                "camera_angular_velocity_local_radps",
            ),
        ]
        distance = np.nan_to_num(
            contact["joint_surface_distance_m"][indices].astype(np.float32),
            nan=np.inf,
            posinf=np.inf,
            neginf=0.0,
        )
        proximity = np.exp(-np.square(distance / 0.02)) * presence[:, :, None]
        groups.extend((proximity.reshape(len(indices), -1), presence))
        vector = np.concatenate(groups, axis=1).astype(np.float32)
        if vector.shape[1] != self.physical_dim:
            raise AssertionError(f"Physical vector has {vector.shape[1]} rather than {self.physical_dim} dims")

        maps = []
        for time_index in range(len(indices)):
            point_cloud = np.concatenate(
                (hands[time_index, 0], hands[time_index, 1], object_translation[time_index, None]), axis=0
            )
            weights = np.zeros((self.map_channels, 43), dtype=np.float32)
            weights[0, :21] = presence[time_index, 0]
            weights[1, 21:42] = presence[time_index, 1]
            weights[2, :21] = proximity[time_index, 0]
            weights[3, 21:42] = proximity[time_index, 1]
            weights[4, 42] = 1.0
            maps.append(_gaussian_maps(point_cloud, weights, intrinsics, self.image_size))
        return vector, np.stack(maps)

    def _initial_ego_maps(
        self,
        state: dict[str, np.ndarray],
        indices: np.ndarray,
        intrinsics: np.ndarray,
        student_state: dict[str, np.ndarray] | None = None,
        student_masks: dict[str, np.ndarray] | None = None,
        oracle_hand_only: bool = False,
        student_flow: bool = False,
        student_mask_flow: bool = False,
        student_dense_mask_flow: bool = False,
    ) -> np.ndarray:
        """Project world hand/object state into the *initial* ego camera only.

        This deliberately avoids every future cam4 pose.  The current pilot uses
        Oracle world state as an upper bound; a deployable version must replace
        it with calibrated multi-exo triangulation.
        """
        anchor_from_world = np.linalg.inv(state["camera_pose_world"][indices[0]])

        def transform(points: np.ndarray) -> np.ndarray:
            return points @ anchor_from_world[:3, :3].T + anchor_from_world[:3, 3]

        if student_state is None:
            hands = transform(state["hand_joints_world_m"][indices].astype(np.float32))
            if oracle_hand_only:
                objects = np.full((len(indices), 3), np.nan, dtype=np.float32)
            else:
                objects = transform(state["object_center_world_m"][indices].astype(np.float32))
            confidence = np.repeat(
                state["hand_presence"][indices, :, None].astype(np.float32), 21, axis=2
            )
        else:
            requested_frames = state["frames"][indices]
            student_indices = np.searchsorted(student_state["frames"], requested_frames)
            if np.any(student_indices >= len(student_state["frames"])) or not np.array_equal(
                student_state["frames"][student_indices], requested_frames
            ):
                raise IndexError("Student state does not contain every requested clip frame")
            hands = transform(
                student_state["hand_joints_world_m"][student_indices].astype(np.float32)
            )
            confidence = student_state["joint_confidence"][student_indices].astype(np.float32)
            # No object detector is used in this student baseline.  NaNs keep
            # the object channel exactly empty during rasterization.
            objects = np.full((len(indices), 3), np.nan, dtype=np.float32)
        if student_mask_flow:
            if student_masks is None:
                raise ValueError("student_mask_flow requires precomputed initial hand masks")
            initial_frame = int(state["frames"][indices[0]])
            mask_index = int(np.searchsorted(student_masks["frames"], initial_frame))
            if (
                mask_index >= len(student_masks["frames"])
                or int(student_masks["frames"][mask_index]) != initial_frame
            ):
                raise IndexError(f"Initial hand mask missing frame {initial_frame}")
            initial_masks = student_masks["masks"][mask_index].astype(np.float32) / 255.0
            if student_dense_mask_flow:
                return np.stack(
                    [
                        _dense_mask_flow_maps(
                            hands[0],
                            hands[time_index],
                            confidence[time_index],
                            intrinsics,
                            initial_masks,
                            self.image_size,
                        )
                        for time_index in range(len(indices))
                    ]
                )
            hybrid_maps = []
            for time_index in range(len(indices)):
                points = np.concatenate(
                    (
                        hands[time_index, 0],
                        hands[time_index, 1],
                        np.full((1, 3), np.nan, dtype=np.float32),
                    ),
                    axis=0,
                )
                weights = np.zeros((self.map_channels, 43), dtype=np.float32)
                weights[0, :21] = confidence[time_index, 0]
                weights[1, 21:42] = confidence[time_index, 1]
                frame_maps = _gaussian_maps(points, weights, intrinsics, self.image_size)
                frame_maps[2:4] = _gaussian_flow_maps(
                    hands[0].reshape(42, 3),
                    hands[time_index].reshape(42, 3),
                    confidence[time_index].reshape(42),
                    intrinsics,
                    self.image_size,
                )
                dense_masks = _dense_mask_flow_maps(
                    hands[0],
                    hands[time_index],
                    confidence[time_index],
                    intrinsics,
                    initial_masks,
                    self.image_size,
                )
                # The real anchor mask controls only occlusion topology. Pixel
                # transport remains the conservative, already-audited sparse
                # joint flow in channels 2:4.
                frame_maps[4] = dense_masks[:2].max(axis=0)
                hybrid_maps.append(frame_maps)
            return np.stack(hybrid_maps)
        maps = []
        for time_index in range(len(indices)):
            points = np.concatenate(
                (hands[time_index, 0], hands[time_index, 1], objects[time_index, None]), axis=0
            )
            weights = np.zeros((self.map_channels, 43), dtype=np.float32)
            weights[0, :21] = confidence[time_index, 0]
            weights[1, 21:42] = confidence[time_index, 1]
            if student_state is None and not oracle_hand_only:
                weights[4, 42] = 1.0
            frame_maps = _gaussian_maps(points, weights, intrinsics, self.image_size)
            if student_flow:
                initial_hands = hands[0].reshape(42, 3)
                current_hands = hands[time_index].reshape(42, 3)
                frame_maps[2:4] = _gaussian_flow_maps(
                    initial_hands,
                    current_hands,
                    confidence[time_index].reshape(42),
                    intrinsics,
                    self.image_size,
                )
            maps.append(frame_maps)
        return np.stack(maps)

    def __getitem__(self, index: int) -> dict[str, Any]:
        row = self.rows[index]
        state = self.state_cache.get(row["state_path"])
        contact_path = Path(row["state_path"]).with_name("surface_contact.npz")
        contact = self.contact_cache.get(contact_path)
        frame_numbers, indices = self._frame_indices(row, state["frames"])
        if self.combine_source_cameras:
            source = np.stack(
                [
                    np.stack([self._load_rgb(directory, int(f)) for f in frame_numbers])
                    for directory in json.loads(row["source_rgb_dirs"])
                ]
            )
        else:
            source = np.stack([self._load_rgb(row["source_rgb_dir"], int(f)) for f in frame_numbers])
        target = np.stack([self._load_rgb(row["target_rgb_dir"], int(f)) for f in frame_numbers])
        # Explicit Anchored-protocol input: only the first ego frame is exposed,
        # then repeated for tensor convenience. No future target frame leaks.
        ego_anchor = np.repeat(target[:1], len(frame_numbers), axis=0)
        intrinsics = _read_intrinsics(row["target_intrinsics"])
        vector, maps = self._physical_condition(state, contact, indices, intrinsics)
        student_state = None
        if self.student_state_root is not None:
            student_state = self.student_cache.get(
                self.student_state_root / row["sequence"] / "student_state.npz"
            )
        student_masks = None
        if self.student_mask_root is not None:
            student_masks = self.student_mask_cache.get(
                self.student_mask_root / row["sequence"] / "initial_hand_masks.npz"
            )
        initial_ego_maps = self._initial_ego_maps(
            state,
            indices,
            intrinsics,
            student_state=student_state,
            student_masks=student_masks,
            oracle_hand_only=self.oracle_hand_only,
            student_flow=self.student_flow,
            student_mask_flow=self.student_mask_flow,
            student_dense_mask_flow=self.student_dense_mask_flow,
        )
        source_hand_maps = self._source_hand_layouts(row, frame_numbers, student_state)
        (
            source_joint_uv,
            source_joint_confidence,
            target_joint_uv,
            target_joint_confidence,
        ) = self._crossview_joint_coordinates(
            row, frame_numbers, state, indices, student_state
        )
        coarse_rgb, coarse_depth, visibility_mask = self._geometric_condition(row, frame_numbers)
        if not np.isfinite(vector).all() or not np.isfinite(maps).all():
            raise FloatingPointError(f"Non-finite physical condition for {row['pair_id']}")
        return {
            "source": source,
            "target": target,
            "ego_anchor": ego_anchor,
            "physical_vector": vector,
            "physical_maps": maps,
            "initial_ego_maps": initial_ego_maps,
            "source_hand_maps": source_hand_maps,
            "source_joint_uv": source_joint_uv,
            "source_joint_confidence": source_joint_confidence,
            "target_joint_uv": target_joint_uv,
            "target_joint_confidence": target_joint_confidence,
            "coarse_rgb": coarse_rgb,
            "coarse_depth": coarse_depth,
            "visibility_mask": visibility_mask,
            "frame_numbers": frame_numbers,
            "pair_id": row["pair_id"],
            "sequence": row["sequence"],
            "source_camera": row["source_camera"],
        }
