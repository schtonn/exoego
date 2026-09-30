"""Annotated-exo rigid-object layer for H2O upper-bound experiments.

Only cam0--cam3 object poses, RGB-D and calibration are inputs.  Cam4 object
poses are deliberately never read here.  The annotations make this an upper
bound, not a deployable inference component.
"""

from __future__ import annotations

from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path

import cv2
import numpy as np
from PIL import Image
from scipy.spatial import cKDTree

from h2o_geometric_baseline.reprojection import load_intrinsics, load_pose
from h2o_oracle_state.h2o_oracle_state import OBJECTS


@dataclass
class ObjectGeometry:
    object_id: int
    object_name: str
    surface_points_m: np.ndarray
    vertices_m: np.ndarray
    faces: np.ndarray
    surface_tree: cKDTree
    bounds_min_m: np.ndarray
    bounds_max_m: np.ndarray


@dataclass
class ObjectLayer:
    object_id: int
    object_name: str
    pose_world: np.ndarray
    rgb: np.ndarray
    depth_m: np.ndarray
    valid: np.ndarray
    support: np.ndarray
    source_point_count: int


@dataclass
class AnchoredObjectAppearance:
    """First-ego-frame object pixels stored in rigid object coordinates."""

    object_id: int
    object_name: str
    points_object_m: np.ndarray
    colors: np.ndarray


@lru_cache(maxsize=16)
def load_geometry(state_root_text: str, object_id: int) -> ObjectGeometry:
    state_root = Path(state_root_text)
    object_name = OBJECTS[object_id][0]
    with np.load(
        state_root / "object_geometry" / f"{object_name}_surface_20000.npz"
    ) as archive:
        surface = archive["points_object_m"].astype(np.float64)
        vertices = archive["vertices_object_m"].astype(np.float64)
        faces = archive["faces"].astype(np.int32)
    return ObjectGeometry(
        object_id=object_id,
        object_name=object_name,
        surface_points_m=surface,
        vertices_m=vertices,
        faces=faces,
        surface_tree=cKDTree(surface),
        bounds_min_m=vertices.min(axis=0),
        bounds_max_m=vertices.max(axis=0),
    )


def _load_annotated_pose(camera: Path, frame: int) -> tuple[int, np.ndarray]:
    values = np.loadtxt(
        camera / "obj_pose_rt" / f"{frame:06d}.txt", dtype=np.float64
    ).reshape(-1)
    if len(values) != 17:
        raise ValueError(f"Expected object id + 4x4 pose in {camera}")
    return int(round(values[0])), values[1:].reshape(4, 4)


def annotated_exo_object_pose_world(
    camera_roots: list[Path], frame: int
) -> tuple[int, np.ndarray, dict[str, float]]:
    """Fuse redundant cam0--cam3 annotations into one world-frame SE(3)."""
    identifiers = []
    poses = []
    for camera in camera_roots:
        identifier, object_from_local = _load_annotated_pose(camera, frame)
        camera_world = load_pose(camera / "cam_pose" / f"{frame:06d}.txt")
        identifiers.append(identifier)
        poses.append(camera_world @ object_from_local)
    if len(set(identifiers)) != 1:
        raise ValueError(f"Object-id disagreement at frame {frame}: {identifiers}")
    translations = np.stack([pose[:3, 3] for pose in poses])
    rotation_sum = np.stack([pose[:3, :3] for pose in poses]).sum(axis=0)
    u, _, vt = np.linalg.svd(rotation_sum)
    rotation = u @ vt
    if np.linalg.det(rotation) < 0:
        u[:, -1] *= -1
        rotation = u @ vt
    pose_world = np.eye(4, dtype=np.float64)
    pose_world[:3, :3] = rotation
    pose_world[:3, 3] = np.median(translations, axis=0)
    translation_spread = np.linalg.norm(translations - pose_world[:3, 3], axis=1)
    rotation_spread = []
    for pose in poses:
        relative = pose[:3, :3].T @ rotation
        rotation_spread.append(
            np.degrees(
                np.arccos(np.clip((np.trace(relative) - 1.0) / 2.0, -1.0, 1.0))
            )
        )
    return identifiers[0], pose_world, {
        "translation_spread_max_m": float(translation_spread.max()),
        "rotation_spread_max_deg": float(np.max(rotation_spread)),
    }


def transform_object(points: np.ndarray, pose_world: np.ndarray) -> np.ndarray:
    return points @ pose_world[:3, :3].T + pose_world[:3, 3]


def world_to_camera(points_world: np.ndarray, camera_pose_world: np.ndarray) -> np.ndarray:
    camera_from_world = np.linalg.inv(camera_pose_world)
    return points_world @ camera_from_world[:3, :3].T + camera_from_world[:3, 3]


def project_camera_points(
    points_camera: np.ndarray,
    intrinsics: np.ndarray,
    output_size: tuple[int, int],
) -> tuple[np.ndarray, np.ndarray]:
    width, height = output_size
    native_width, native_height = intrinsics[4:6]
    scale_x, scale_y = width / native_width, height / native_height
    z = points_camera[:, 2]
    valid = np.isfinite(points_camera).all(axis=1) & (z > 1e-5)
    uv = np.full((len(points_camera), 2), np.nan, dtype=np.float64)
    fx, fy, cx, cy = intrinsics[:4]
    uv[valid, 0] = (fx * points_camera[valid, 0] / z[valid] + cx) * scale_x
    uv[valid, 1] = (fy * points_camera[valid, 1] / z[valid] + cy) * scale_y
    return uv, valid


def rasterize_object_support(
    geometry: ObjectGeometry,
    pose_world: np.ndarray,
    target_intrinsics: np.ndarray,
    target_pose_world: np.ndarray,
    output_size: tuple[int, int],
) -> tuple[np.ndarray, np.ndarray]:
    """Rasterize CAD triangles for support and dense surface points for depth."""
    width, height = output_size
    vertices_camera = world_to_camera(
        transform_object(geometry.vertices_m, pose_world), target_pose_world
    )
    uv, vertex_valid = project_camera_points(
        vertices_camera, target_intrinsics, output_size
    )
    support = np.zeros((height, width), dtype=np.uint8)
    visible_vertices = uv[vertex_valid]
    if len(visible_vertices) >= 3:
        # All eight H2O manipulation objects are compact containers/books. A
        # projected vertex hull is a conservative support and is orders of
        # magnitude faster than looping over thousands of triangles per frame.
        hull = cv2.convexHull(np.rint(visible_vertices).astype(np.int32))
        cv2.fillConvexPoly(support, hull, 1)

    surface_camera = world_to_camera(
        transform_object(geometry.surface_points_m, pose_world), target_pose_world
    )
    surface_uv, surface_valid = project_camera_points(
        surface_camera, target_intrinsics, output_size
    )
    xy = np.rint(surface_uv[surface_valid]).astype(np.int32)
    z = surface_camera[surface_valid, 2]
    in_image = (
        (xy[:, 0] >= 0)
        & (xy[:, 0] < width)
        & (xy[:, 1] >= 0)
        & (xy[:, 1] < height)
    )
    xy, z = xy[in_image], z[in_image]
    flat_depth = np.full(width * height, np.inf, dtype=np.float32)
    if len(xy):
        flat = xy[:, 1].astype(np.int64) * width + xy[:, 0]
        np.minimum.at(flat_depth, flat, z.astype(np.float32))
    depth = flat_depth.reshape(height, width)
    depth[~np.isfinite(depth)] = np.nan
    return support.astype(bool), depth


def _source_object_points(
    camera: Path,
    frame: int,
    pose_world: np.ndarray,
    geometry: ObjectGeometry,
    source_stride: int,
    surface_tolerance_m: float,
    source_exclusion_mask: np.ndarray | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    stem = f"{frame:06d}"
    rgb = np.asarray(Image.open(camera / "rgb" / f"{stem}.png").convert("RGB"))
    depth_mm = np.asarray(Image.open(camera / "depth" / f"{stem}.png"))
    intrinsics = load_intrinsics(camera / "cam_intrinsics.txt")
    camera_pose = load_pose(camera / "cam_pose" / f"{stem}.txt")

    # CAD projection supplies a cheap 2-D prefilter before metric 3-D testing.
    native_size = (depth_mm.shape[1], depth_mm.shape[0])
    vertices_camera = world_to_camera(
        transform_object(geometry.vertices_m, pose_world), camera_pose
    )
    uv, valid_vertex = project_camera_points(vertices_camera, intrinsics, native_size)
    silhouette = np.zeros(depth_mm.shape, dtype=np.uint8)
    visible_vertices = uv[valid_vertex]
    if len(visible_vertices) >= 3:
        # This is only a cheap prefilter. The later object-local surface-distance
        # test rejects distant background, while an explicit source-view mask is
        # required for hands touching or occluding the object: proximity to the
        # CAD surface alone cannot distinguish their RGB provenance.
        hull = cv2.convexHull(np.rint(visible_vertices).astype(np.int32))
        cv2.fillConvexPoly(silhouette, hull, 1)

    if source_exclusion_mask is None:
        exclusion = np.zeros(depth_mm.shape, dtype=bool)
    else:
        exclusion = cv2.resize(
            source_exclusion_mask.astype(np.uint8),
            (depth_mm.shape[1], depth_mm.shape[0]),
            interpolation=cv2.INTER_NEAREST,
        ) > 0

    ys = np.arange(0, depth_mm.shape[0], source_stride, dtype=np.int32)
    xs = np.arange(0, depth_mm.shape[1], source_stride, dtype=np.int32)
    gx, gy = np.meshgrid(xs, ys)
    depth = depth_mm[::source_stride, ::source_stride].astype(np.float64) / 1000.0
    candidate = (
        (silhouette[::source_stride, ::source_stride] > 0)
        & (~exclusion[::source_stride, ::source_stride])
        & np.isfinite(depth)
        & (depth >= 0.1)
        & (depth <= 5.0)
    )
    x, y, z = gx[candidate], gy[candidate], depth[candidate]
    if len(z) == 0:
        return np.empty((0, 3)), np.empty((0, 3), dtype=np.uint8)
    fx, fy, cx, cy = intrinsics[:4]
    points_camera = np.column_stack(((x - cx) * z / fx, (y - cy) * z / fy, z))
    points_world = points_camera @ camera_pose[:3, :3].T + camera_pose[:3, 3]
    points_object = (points_world - pose_world[:3, 3]) @ pose_world[:3, :3]
    margin = surface_tolerance_m
    inside_bounds = np.all(
        (points_object >= geometry.bounds_min_m - margin)
        & (points_object <= geometry.bounds_max_m + margin),
        axis=1,
    )
    nearest_distance = np.full(len(points_object), np.inf, dtype=np.float64)
    if np.any(inside_bounds):
        nearest_distance[inside_bounds] = geometry.surface_tree.query(
            points_object[inside_bounds], k=1, workers=1
        )[0]
    keep = inside_bounds & (nearest_distance <= surface_tolerance_m)
    return points_world[keep], rgb[y[keep], x[keep]]


def _zbuffer_points(
    points_world: np.ndarray,
    colors: np.ndarray,
    target_intrinsics: np.ndarray,
    target_pose_world: np.ndarray,
    output_size: tuple[int, int],
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    width, height = output_size
    points_camera = world_to_camera(points_world, target_pose_world)
    uv, valid = project_camera_points(points_camera, target_intrinsics, output_size)
    xy = np.rint(uv[valid]).astype(np.int32)
    z = points_camera[valid, 2]
    colors = colors[valid]
    in_image = (
        (xy[:, 0] >= 0)
        & (xy[:, 0] < width)
        & (xy[:, 1] >= 0)
        & (xy[:, 1] < height)
    )
    xy, z, colors = xy[in_image], z[in_image], colors[in_image]
    flat = xy[:, 1].astype(np.int64) * width + xy[:, 0]
    order = np.lexsort((z, flat))
    sorted_flat = flat[order]
    first = np.ones(len(order), dtype=bool)
    if len(order) > 1:
        first[1:] = sorted_flat[1:] != sorted_flat[:-1]
    winners = order[first]
    pixels = flat[winners]
    rgb = np.zeros((height * width, 3), dtype=np.float32)
    depth = np.full(height * width, np.nan, dtype=np.float32)
    mask = np.zeros(height * width, dtype=bool)
    rgb[pixels] = colors[winners].astype(np.float32) / 255.0
    depth[pixels] = z[winners].astype(np.float32)
    mask[pixels] = True
    return rgb.reshape(height, width, 3), depth.reshape(height, width), mask.reshape(height, width)


def _densify_within_support(
    rgb: np.ndarray,
    depth: np.ndarray,
    valid: np.ndarray,
    support: np.ndarray,
    iterations: int = 3,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    value, z_value, mask = rgb.copy(), depth.copy(), valid.copy()
    height, width = mask.shape
    for _ in range(iterations):
        padded_rgb = np.pad(value, ((1, 1), (1, 1), (0, 0)))
        padded_z = np.pad(np.nan_to_num(z_value, nan=0.0), 1)
        padded_valid = np.pad(mask, 1)
        rgb_sum = np.zeros_like(value)
        z_sum = np.zeros_like(z_value)
        count = np.zeros_like(z_value, dtype=np.float32)
        for dy in range(3):
            for dx in range(3):
                neighbor = padded_valid[dy : dy + height, dx : dx + width]
                rgb_sum += padded_rgb[dy : dy + height, dx : dx + width] * neighbor[..., None]
                z_sum += padded_z[dy : dy + height, dx : dx + width] * neighbor
                count += neighbor
        fill = (~mask) & support & (count >= 2)
        value[fill] = (rgb_sum / np.maximum(count[..., None], 1))[fill]
        z_value[fill] = (z_sum / np.maximum(count, 1))[fill]
        mask[fill] = True
    return value, z_value, mask


def annotated_exo_object_layer(
    camera_roots: list[Path],
    frame: int,
    target_intrinsics: np.ndarray,
    target_pose_world: np.ndarray,
    state_root: Path,
    output_size: int = 256,
    source_stride: int = 2,
    surface_tolerance_m: float = 0.018,
) -> ObjectLayer:
    object_id, pose_world, _ = annotated_exo_object_pose_world(camera_roots, frame)
    geometry = load_geometry(str(state_root.resolve()), object_id)
    support, cad_depth = rasterize_object_support(
        geometry,
        pose_world,
        target_intrinsics,
        target_pose_world,
        (output_size, output_size),
    )
    points, colors = [], []
    for camera in camera_roots:
        camera_points, camera_colors = _source_object_points(
            camera,
            frame,
            pose_world,
            geometry,
            source_stride,
            surface_tolerance_m,
        )
        points.append(camera_points)
        colors.append(camera_colors)
    points_world = np.concatenate(points) if points else np.empty((0, 3))
    point_colors = np.concatenate(colors) if colors else np.empty((0, 3), dtype=np.uint8)
    if len(points_world):
        rgb, depth, valid = _zbuffer_points(
            points_world,
            point_colors,
            target_intrinsics,
            target_pose_world,
            (output_size, output_size),
        )
        valid &= support
        rgb[~valid] = 0
        depth[~valid] = np.nan
        rgb, depth, valid = _densify_within_support(rgb, depth, valid, support)
    else:
        rgb = np.zeros((output_size, output_size, 3), dtype=np.float32)
        depth = cad_depth
        valid = np.zeros((output_size, output_size), dtype=bool)
    # Prefer measured RGB-D depth, but retain CAD depth for support diagnostics.
    depth = np.where(valid, depth, cad_depth)
    return ObjectLayer(
        object_id=object_id,
        object_name=geometry.object_name,
        pose_world=pose_world,
        rgb=rgb,
        depth_m=depth,
        valid=valid,
        support=support,
        source_point_count=len(points_world),
    )


def annotated_exo_object_geometry_layer(
    camera_roots: list[Path],
    frame: int,
    target_intrinsics: np.ndarray,
    target_pose_world: np.ndarray,
    state_root: Path,
    output_size: int = 256,
) -> ObjectLayer:
    """Project only the annotated CAD support, without synthesizing appearance."""
    object_id, pose_world, _ = annotated_exo_object_pose_world(camera_roots, frame)
    geometry = load_geometry(str(state_root.resolve()), object_id)
    support, depth = rasterize_object_support(
        geometry, pose_world, target_intrinsics, target_pose_world,
        (output_size, output_size),
    )
    return ObjectLayer(
        object_id=object_id,
        object_name=geometry.object_name,
        pose_world=pose_world,
        rgb=np.zeros((output_size, output_size, 3), dtype=np.float32),
        depth_m=depth,
        valid=np.zeros((output_size, output_size), dtype=bool),
        support=support,
        source_point_count=0,
    )


def build_anchored_object_appearance(
    ego_camera_root: Path,
    frame: int,
    object_pose_world: np.ndarray,
    state_root: Path,
    object_id: int,
    source_stride: int = 1,
    surface_tolerance_m: float = 0.018,
    source_hand_occlusion_mask: np.ndarray | None = None,
) -> AnchoredObjectAppearance:
    """Extract visible first-frame object RGB-D, excluding hand provenance.

    A hand in contact with the object may lie within the CAD distance tolerance.
    The caller must therefore provide a first-frame hand/arm occlusion mask when
    the appearance will be transported to future frames. Masked pixels remain
    unobserved object surface; they are never relabelled as object appearance.
    """
    geometry = load_geometry(str(state_root.resolve()), object_id)
    points_world, colors = _source_object_points(
        ego_camera_root,
        frame,
        object_pose_world,
        geometry,
        source_stride,
        surface_tolerance_m,
        source_exclusion_mask=source_hand_occlusion_mask,
    )
    points_object = (
        (points_world - object_pose_world[:3, 3]) @ object_pose_world[:3, :3]
        if len(points_world) else np.empty((0, 3), dtype=np.float64)
    )
    return AnchoredObjectAppearance(
        object_id=object_id,
        object_name=geometry.object_name,
        points_object_m=points_object,
        colors=colors,
    )


def render_anchored_object_appearance(
    appearance: AnchoredObjectAppearance,
    object_pose_world: np.ndarray,
    target_intrinsics: np.ndarray,
    target_pose_world: np.ndarray,
    state_root: Path,
    output_size: int = 256,
    densify_iterations: int = 2,
) -> ObjectLayer:
    """Rigidly transport first-frame object appearance using its annotated SE(3)."""
    geometry = load_geometry(str(state_root.resolve()), appearance.object_id)
    support, cad_depth = rasterize_object_support(
        geometry,
        object_pose_world,
        target_intrinsics,
        target_pose_world,
        (output_size, output_size),
    )
    if len(appearance.points_object_m):
        points_world = transform_object(appearance.points_object_m, object_pose_world)
        rgb, depth, valid = _zbuffer_points(
            points_world,
            appearance.colors,
            target_intrinsics,
            target_pose_world,
            (output_size, output_size),
        )
        valid &= support
        rgb[~valid] = 0
        depth[~valid] = np.nan
        rgb, depth, valid = _densify_within_support(
            rgb, depth, valid, support, iterations=densify_iterations
        )
    else:
        rgb = np.zeros((output_size, output_size, 3), dtype=np.float32)
        depth = np.full((output_size, output_size), np.nan, dtype=np.float32)
        valid = np.zeros((output_size, output_size), dtype=bool)
    depth = np.where(valid, depth, cad_depth)
    return ObjectLayer(
        object_id=appearance.object_id,
        object_name=appearance.object_name,
        pose_world=object_pose_world,
        rgb=rgb,
        depth_m=depth,
        valid=valid,
        support=support,
        source_point_count=len(appearance.points_object_m),
    )
