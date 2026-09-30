#!/usr/bin/env python3
"""Render H2O RGB, calibrated projections, and continuous distance cues."""

from __future__ import annotations

import argparse
import os
from pathlib import Path

os.environ.setdefault("MPLCONFIGDIR", "/tmp/matplotlib-h2o-render")

import matplotlib

matplotlib.use("Agg")
import matplotlib.animation as animation
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
import numpy as np
from PIL import Image

from h2o_oracle_state import ACTIONS, HAND_EDGES, OBJECTS, load_obj_vertices, transform_points


DEFAULT_RAW = Path("datasets/H2O/raw")
DEFAULT_STATE = Path("datasets/H2O/oracle_state")
DEFAULT_PREVIEW = Path("datasets/H2O/previews")
DEFAULT_CALIBRATION = DEFAULT_PREVIEW / "subject1_h1_0_cam3" / "calibration_extract"

LEFT = "#d62728"
RIGHT = "#1f77b4"
OBJECT = "#2ca02c"

EGO_FRUSTUM_LOCAL = np.array(
    [
        [-0.055, -0.035, 0.11],
        [0.055, -0.035, 0.11],
        [0.055, 0.035, 0.11],
        [-0.055, 0.035, 0.11],
    ]
)


def project_world(points: np.ndarray, world_to_camera: np.ndarray, intrinsics: np.ndarray):
    camera_points = transform_points(world_to_camera, points)
    depth = camera_points[:, 2]
    uv = np.full((len(points), 2), np.nan, dtype=np.float64)
    valid = depth > 1e-6
    fx, fy, cx, cy = intrinsics[:4]
    uv[valid, 0] = fx * camera_points[valid, 0] / depth[valid] + cx
    uv[valid, 1] = fy * camera_points[valid, 1] / depth[valid] + cy
    return uv, depth


def build_hand_2d(axis, color: str, label: str, *, cmap=None, norm=None):
    if cmap is None:
        scatter = axis.scatter([], [], s=24, color=color, label=label, zorder=4)
    else:
        scatter = axis.scatter(
            [], [], s=32, c=np.empty(0), cmap=cmap, norm=norm,
            edgecolors="black", linewidths=0.25, label=label, zorder=4,
        )
    lines = [axis.plot([], [], color=color, linewidth=1.8, alpha=0.9, zorder=3)[0] for _ in HAND_EDGES]
    return scatter, lines


def update_hand_2d(scatter, lines, uv: np.ndarray, present: bool) -> None:
    if not present:
        scatter.set_offsets(np.empty((0, 2)))
        for line in lines:
            line.set_data([], [])
        return
    scatter.set_offsets(uv)
    for artist, (start, end) in zip(lines, HAND_EDGES):
        edge = uv[[start, end]]
        artist.set_data(edge[:, 0], edge[:, 1])


def image_roi(
    uv_values: list[np.ndarray],
    width: float,
    height: float,
    *,
    percentile: float = 0.5,
    clamp_to_image: bool = True,
) -> tuple[float, float, float, float]:
    points = np.concatenate(uv_values, axis=0)
    valid = np.all(np.isfinite(points), axis=1)
    points = points[valid]
    if clamp_to_image:
        points = points[
            (points[:, 0] >= 0.0)
            & (points[:, 0] <= width)
            & (points[:, 1] >= 0.0)
            & (points[:, 1] <= height)
        ]
    if not len(points):
        return 0.0, width, 0.0, height
    low = np.percentile(points, percentile, axis=0)
    high = np.percentile(points, 100.0 - percentile, axis=0)
    span = np.maximum(high - low, np.array([220.0, 160.0]))
    center = (low + high) / 2.0
    span *= 1.22
    target_aspect = width / height
    if span[0] / span[1] < target_aspect:
        span[0] = span[1] * target_aspect
    else:
        span[1] = span[0] / target_aspect
    x0, y0 = center - span / 2.0
    x1, y1 = center + span / 2.0
    if clamp_to_image:
        if x0 < 0:
            x1 -= x0
            x0 = 0.0
        if x1 > width:
            x0 -= x1 - width
            x1 = width
        if y0 < 0:
            y1 -= y0
            y0 = 0.0
        if y1 > height:
            y0 -= y1 - height
            y1 = height
        return max(0.0, x0), min(width, x1), max(0.0, y0), min(height, y1)
    return x0, x1, y0, y1


def render(args: argparse.Namespace) -> None:
    state_dir = args.state_root / args.sequence
    state_path = state_dir / "oracle_state.npz"
    if not state_path.is_file():
        raise FileNotFoundError(f"Build oracle state first: {state_path}")
    state = np.load(state_path)
    all_frames = state["frames"]
    selected = np.flatnonzero((all_frames >= args.start) & (all_frames <= args.end))
    if not len(selected):
        raise ValueError("Requested frame range is outside the oracle state")

    frames = all_frames[selected]
    sequence_parts = args.sequence.split("/")
    subject, room, take = sequence_parts
    rgb_dir = args.rgb_root / f"{subject}_{room}_{take}_cam{args.exo_camera}" / args.sequence / f"cam{args.exo_camera}" / "rgb"
    rgb_paths = [rgb_dir / f"{int(frame):06d}.png" for frame in frames]
    missing = [path for path in rgb_paths if not path.is_file()]
    if missing:
        raise FileNotFoundError(f"Missing {len(missing)} RGB frames; first: {missing[0]}")

    calibration_dir = args.calibration_root / args.sequence / f"cam{args.exo_camera}"
    intrinsics_path = calibration_dir / "cam_intrinsics.txt"
    if not intrinsics_path.is_file():
        raise FileNotFoundError(f"Missing exocentric calibration: {intrinsics_path}")
    intrinsics = np.loadtxt(intrinsics_path).reshape(-1)
    if len(intrinsics) < 6:
        raise ValueError(f"Expected fx fy cx cy width height in {intrinsics_path}")
    image_width, image_height = float(intrinsics[4]), float(intrinsics[5])
    exo_camera_world_pose = np.stack(
        [
            np.loadtxt(calibration_dir / "cam_pose" / f"{int(frame):06d}.txt").reshape(4, 4)
            for frame in frames
        ]
    )
    exo_world_to_camera = np.linalg.inv(exo_camera_world_pose)

    object_ids = state["object_id"][selected]
    if len(np.unique(object_ids)) != 1:
        raise ValueError("This renderer currently expects one rigid object per clip")
    object_id = int(object_ids[0])
    object_name, mesh_relative = OBJECTS[object_id]
    local_vertices = load_obj_vertices(args.raw_root / "object" / mesh_relative)
    vertex_step = max(1, len(local_vertices) // args.max_object_points)
    local_plot_vertices = local_vertices[::vertex_step]

    hand_world = state["hand_joints_world_m"][selected]
    presence = state["hand_presence"][selected]
    object_world_pose = state["object_pose_world"][selected]
    camera_world_pose = state["camera_pose_world"][selected]
    action = state["action_label"][selected]
    surface_contact_path = state_dir / "surface_contact.npz"
    if surface_contact_path.is_file():
        with np.load(surface_contact_path) as surface_contact:
            if not np.array_equal(surface_contact["frames"], all_frames):
                raise ValueError(f"Frame mismatch in {surface_contact_path}")
            contact_distance = surface_contact["surface_distance_m"][selected]
            joint_surface_distance = surface_contact["joint_surface_distance_m"][selected]
    else:
        contact_distance = state["contact_distance_m"][selected]
        joint_surface_distance = np.repeat(contact_distance[:, :, None], 21, axis=2)
    object_speed = np.linalg.norm(state["object_linear_velocity_world_mps"][selected], axis=1)
    camera_speed = np.linalg.norm(state["camera_linear_velocity_world_mps"][selected], axis=1)

    interaction_roi_values = []
    scene_roi_values = []
    roi_step = max(1, len(frames) // 20)
    roi_vertex_step = max(1, len(local_plot_vertices) // 300)
    for local_index in range(0, len(frames), roi_step):
        object_world = transform_points(object_world_pose[local_index], local_plot_vertices[::roi_vertex_step])
        for points in (object_world, hand_world[local_index].reshape(-1, 3)):
            uv, depth = project_world(points, exo_world_to_camera[local_index], intrinsics)
            visible_uv = uv[depth > 1e-6]
            interaction_roi_values.append(visible_uv)
            scene_roi_values.append(visible_uv)
        ego_pose = camera_world_pose[local_index]
        ego_origin = ego_pose[:3, 3]
        ego_axes = ego_origin + ego_pose[:3, :3].T * 0.06
        ego_frustum = transform_points(ego_pose, EGO_FRUSTUM_LOCAL)
        ego_points = np.vstack([ego_origin, ego_axes, ego_frustum])
        uv, depth = project_world(ego_points, exo_world_to_camera[local_index], intrinsics)
        scene_roi_values.append(uv[depth > 1e-6])
    projection_roi = image_roi(interaction_roi_values, image_width, image_height)
    scene_roi = image_roi(
        scene_roi_values,
        image_width,
        image_height,
        percentile=0.0,
        clamp_to_image=False,
    )

    first_rgb = np.asarray(Image.open(rgb_paths[0]).convert("RGB"))
    figure = plt.figure(figsize=(16, 9), dpi=120, facecolor="#f8f8f8")
    grid = figure.add_gridspec(
        2,
        2,
        left=0.025,
        right=0.955,
        bottom=0.06,
        top=0.985,
        wspace=0.07,
        hspace=0.08,
        height_ratios=(0.9, 1.1),
    )
    rgb_axis = figure.add_subplot(grid[0, 0])
    cam_axis = figure.add_subplot(grid[0, 1])
    world_axis = figure.add_subplot(grid[1, 0])
    timeline_axis = figure.add_subplot(grid[1, 1])

    rgb_artist = rgb_axis.imshow(first_rgb)
    rgb_axis.axis("off")

    overlay_rgb_artist = cam_axis.imshow(first_rgb, alpha=0.48)
    cam_object = cam_axis.scatter([], [], s=3, alpha=0.40, color=OBJECT, label=object_name, zorder=2)
    cam_left, cam_left_lines = build_hand_2d(cam_axis, LEFT, "left hand")
    cam_right, cam_right_lines = build_hand_2d(cam_axis, RIGHT, "right hand")
    cam_axis.set_xlim(projection_roi[0], projection_roi[1])
    cam_axis.set_ylim(projection_roi[3], projection_roi[2])
    cam_axis.set_aspect("equal")
    cam_axis.axis("off")
    cam_axis.legend(loc="upper left", fontsize=7)

    distance_cmap = plt.get_cmap("turbo_r").copy()
    distance_cmap.set_bad("#eeeeee")
    distance_norm = matplotlib.colors.Normalize(vmin=0.0, vmax=args.distance_heatmap_max_cm)
    world_axis.set_facecolor("#f2f2f2")
    world_axis.add_patch(
        Rectangle(
            (0, 0),
            image_width,
            image_height,
            fill=False,
            edgecolor="#777777",
            linestyle="--",
            linewidth=1.0,
            alpha=0.65,
            label=f"cam{args.exo_camera} image boundary",
        )
    )
    world_object = world_axis.scatter([], [], s=3, alpha=0.38, color=OBJECT, label=object_name, zorder=1)
    world_left, world_left_lines = build_hand_2d(
        world_axis, LEFT, "left hand", cmap=distance_cmap, norm=distance_norm
    )
    world_right, world_right_lines = build_hand_2d(
        world_axis, RIGHT, "right hand", cmap=distance_cmap, norm=distance_norm
    )
    world_camera = world_axis.scatter(
        [], [], marker="D", s=58, color="black", label="ego camera", zorder=5
    )
    camera_axis_lines = [
        world_axis.plot([], [], linewidth=2, color=color, zorder=4)[0]
        for color in ("#d62728", "#2ca02c", "#1f77b4")
    ]
    camera_frustum_lines = [
        world_axis.plot([], [], linewidth=1.8, color="black", alpha=0.9, zorder=4)[0]
        for _ in range(8)
    ]
    camera_to_object = world_axis.plot(
        [], [], linewidth=1.2, color="black", linestyle=":", alpha=0.7, label="camera→object", zorder=2
    )[0]
    world_axis.set_xlim(scene_roi[0], scene_roi[1])
    world_axis.set_ylim(scene_roi[3], scene_roi[2])
    world_axis.set_aspect("equal")
    world_axis.axis("off")
    world_axis.legend(loc="upper left", fontsize=7)

    time = (frames - frames[0]) / float(args.source_fps)
    distance_heatmap = joint_surface_distance.transpose(1, 2, 0).reshape(42, len(frames)) * 100.0
    heatmap = timeline_axis.imshow(
        distance_heatmap,
        cmap=distance_cmap,
        norm=distance_norm,
        interpolation="nearest",
        aspect="auto",
        extent=(time[0], time[-1], 42, 0),
    )
    timeline_axis.axhline(21, color="white", linewidth=1.5, alpha=0.9)
    cursor = timeline_axis.axvline(time[0], color="#333333", linewidth=2)
    timeline_axis.set_xlim(time[0], time[-1])
    timeline_axis.set_xlabel("clip time [s]")
    timeline_axis.set_yticks((10.5, 31.5), labels=("left joints", "right joints"))
    colorbar = figure.colorbar(heatmap, ax=timeline_axis, fraction=0.035, pad=0.02)
    colorbar.set_label("distance [cm]; red = near")
    status = world_axis.text(
        0.01,
        0.03,
        "",
        transform=world_axis.transAxes,
        va="bottom",
        ha="left",
        fontsize=10,
        bbox={"boxstyle": "round", "facecolor": "white", "alpha": 0.85, "edgecolor": "#777777"},
    )
    frame_text = figure.text(0.5, 0.025, "", ha="center", fontsize=11, family="monospace")

    def update(local_index: int):
        frame = int(frames[local_index])
        rgb_frame = np.asarray(Image.open(rgb_paths[local_index]).convert("RGB"))
        rgb_artist.set_data(rgb_frame)
        overlay_rgb_artist.set_data(rgb_frame)

        vertices_world = transform_points(object_world_pose[local_index], local_plot_vertices)
        object_uv, object_depth = project_world(
            vertices_world, exo_world_to_camera[local_index], intrinsics
        )
        visible_object = object_depth > 1e-6
        cam_object.set_offsets(object_uv[visible_object])
        world_object.set_offsets(object_uv[visible_object])
        hand_uv = []
        for hand_index in range(2):
            uv, _ = project_world(
                hand_world[local_index, hand_index], exo_world_to_camera[local_index], intrinsics
            )
            hand_uv.append(uv)
        update_hand_2d(cam_left, cam_left_lines, hand_uv[0], bool(presence[local_index, 0]))
        update_hand_2d(cam_right, cam_right_lines, hand_uv[1], bool(presence[local_index, 1]))
        update_hand_2d(world_left, world_left_lines, hand_uv[0], bool(presence[local_index, 0]))
        update_hand_2d(world_right, world_right_lines, hand_uv[1], bool(presence[local_index, 1]))
        world_left.set_array(
            np.ma.masked_invalid(joint_surface_distance[local_index, 0] * 100.0)
            if presence[local_index, 0]
            else np.empty(0)
        )
        world_right.set_array(
            np.ma.masked_invalid(joint_surface_distance[local_index, 1] * 100.0)
            if presence[local_index, 1]
            else np.empty(0)
        )

        camera_pose = camera_world_pose[local_index]
        origin = camera_pose[:3, 3]
        origin_uv, origin_depth = project_world(
            origin[None], exo_world_to_camera[local_index], intrinsics
        )
        world_camera.set_offsets(origin_uv if origin_depth[0] > 1e-6 else np.empty((0, 2)))
        for dimension, artist in enumerate(camera_axis_lines):
            end = origin + camera_pose[:3, dimension] * 0.06
            axis_uv, axis_depth = project_world(
                np.stack([origin, end]), exo_world_to_camera[local_index], intrinsics
            )
            artist.set_data(axis_uv[:, 0], axis_uv[:, 1]) if np.all(axis_depth > 1e-6) else artist.set_data([], [])

        # A compact pinhole frustum makes camera position and viewing direction
        # visible even when the origin marker overlaps a grid line.
        frustum_world = transform_points(camera_pose, EGO_FRUSTUM_LOCAL)
        for corner, artist in zip(frustum_world, camera_frustum_lines[:4]):
            edge_uv, edge_depth = project_world(
                np.stack([origin, corner]), exo_world_to_camera[local_index], intrinsics
            )
            artist.set_data(edge_uv[:, 0], edge_uv[:, 1]) if np.all(edge_depth > 1e-6) else artist.set_data([], [])
        for edge_index, artist in enumerate(camera_frustum_lines[4:]):
            edge = frustum_world[[edge_index, (edge_index + 1) % 4]]
            edge_uv, edge_depth = project_world(edge, exo_world_to_camera[local_index], intrinsics)
            artist.set_data(edge_uv[:, 0], edge_uv[:, 1]) if np.all(edge_depth > 1e-6) else artist.set_data([], [])
        object_center = object_world_pose[local_index, :3, 3]
        link_uv, link_depth = project_world(
            np.stack([origin, object_center]), exo_world_to_camera[local_index], intrinsics
        )
        camera_to_object.set_data(link_uv[:, 0], link_uv[:, 1]) if np.all(link_depth > 1e-6) else camera_to_object.set_data([], [])

        cursor.set_xdata([time[local_index], time[local_index]])
        action_id = int(action[local_index])
        action_name = ACTIONS[action_id] if 0 <= action_id < len(ACTIONS) else "unavailable"
        status.set_text(
            f"action: {action_name} ({action_id})\n"
            f"minimum surface distance  L={contact_distance[local_index, 0] * 100.0:.2f}  "
            f"R={contact_distance[local_index, 1] * 100.0:.2f} cm\n"
            f"object speed={object_speed[local_index]:.3f} m/s  camera={camera_speed[local_index]:.3f} m/s"
        )
        frame_text.set_text(
            f"sequence={args.sequence}   frame={frame:06d}   t={time[local_index]:.2f}s   object={object_name}"
        )
        return [rgb_artist, cam_object, world_object, cursor, status, frame_text]

    update(0)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    clip = animation.FuncAnimation(figure, update, frames=len(frames), interval=1000 / args.output_fps, blit=False)
    writer = animation.FFMpegWriter(
        fps=args.output_fps,
        codec="libx264",
        bitrate=args.bitrate_kbps,
        extra_args=["-pix_fmt", "yuv420p", "-movflags", "+faststart"],
        metadata={
            "title": "H2O synchronized physical-state audit",
            "comment": "Calibrated H2O state and continuous hand-object distance",
        },
    )
    clip.save(args.output, writer=writer, dpi=args.dpi)
    update(len(frames) // 2)
    figure.savefig(args.output.with_suffix(".png"), dpi=args.dpi)
    plt.close(figure)
    print(f"Wrote {len(frames)} synchronized frames to {args.output}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--sequence", default="subject1/h1/0")
    parser.add_argument("--start", type=int, default=70)
    parser.add_argument("--end", type=int, default=189)
    parser.add_argument("--exo-camera", type=int, default=3)
    parser.add_argument("--raw-root", type=Path, default=DEFAULT_RAW)
    parser.add_argument("--state-root", type=Path, default=DEFAULT_STATE)
    parser.add_argument("--rgb-root", type=Path, default=DEFAULT_PREVIEW)
    parser.add_argument("--calibration-root", type=Path, default=DEFAULT_CALIBRATION)
    parser.add_argument(
        "--output",
        type=Path,
        default=DEFAULT_PREVIEW / "subject1_h1_0_cam3" / "h2o_known_vs_derived_000070_000189.mp4",
    )
    parser.add_argument("--source-fps", type=float, default=30.0)
    parser.add_argument("--output-fps", type=float, default=30.0)
    parser.add_argument("--max-object-points", type=int, default=1400)
    parser.add_argument("--distance-heatmap-max-cm", type=float, default=5.0)
    parser.add_argument("--bitrate-kbps", type=int, default=10000)
    parser.add_argument("--dpi", type=int, default=160)
    return parser.parse_args()


if __name__ == "__main__":
    render(parse_args())
