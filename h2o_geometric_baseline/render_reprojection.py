#!/usr/bin/env python3
"""Render deterministic cam0-3 to cam4 reprojection audits as images or video."""

from __future__ import annotations

import argparse
import json
import subprocess
import tempfile
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw, ImageFont

from h2o_geometric_baseline.reprojection import (
    audit_reprojection,
    load_and_reproject,
    load_intrinsics,
)
from h2o_oracle_state.h2o_oracle_state import HAND_EDGES, OBJECTS


DEFAULT_RAW = Path("datasets/H2O/raw")
DEFAULT_STATE = Path("datasets/H2O/oracle_state")
DEFAULT_OUTPUT = Path("datasets/H2O/previews/geometric_reprojection")


def project(points: np.ndarray, intrinsics: np.ndarray, size: tuple[int, int]) -> tuple[np.ndarray, np.ndarray]:
    native_width, native_height = intrinsics[4:6]
    width, height = size
    z = points[:, 2]
    valid = np.isfinite(points).all(axis=1) & (z > 1e-5)
    uv = np.full((len(points), 2), np.nan, dtype=np.float64)
    uv[valid, 0] = (intrinsics[0] * points[valid, 0] / z[valid] + intrinsics[2]) * width / native_width
    uv[valid, 1] = (intrinsics[1] * points[valid, 1] / z[valid] + intrinsics[3]) * height / native_height
    return uv, valid


def convex_hull(points: np.ndarray) -> list[tuple[float, float]]:
    unique = sorted(set(map(tuple, points.tolist())))
    if len(unique) <= 1:
        return unique

    def cross(origin, first, second):
        return (first[0] - origin[0]) * (second[1] - origin[1]) - (first[1] - origin[1]) * (
            second[0] - origin[0]
        )

    lower = []
    for point in unique:
        while len(lower) >= 2 and cross(lower[-2], lower[-1], point) <= 0:
            lower.pop()
        lower.append(point)
    upper = []
    for point in reversed(unique):
        while len(upper) >= 2 and cross(upper[-2], upper[-1], point) <= 0:
            upper.pop()
        upper.append(point)
    return lower[:-1] + upper[:-1]


def distance_color(distance_m: float) -> tuple[int, int, int]:
    alpha = float(np.clip(distance_m / 0.05, 0, 1))
    return (int(255 * (1 - alpha)), int(70 + 100 * alpha), int(255 * alpha))


def physical_overlay(
    base_rgb: np.ndarray,
    state_root: Path,
    sequence: str,
    frame: int,
    target_intrinsics: np.ndarray,
) -> Image.Image:
    state_path = state_root / sequence / "oracle_state.npz"
    contact_path = state_root / sequence / "surface_contact.npz"
    with np.load(state_path) as state, np.load(contact_path) as contact:
        position = int(np.searchsorted(state["frames"], frame))
        if position >= len(state["frames"]) or int(state["frames"][position]) != frame:
            raise IndexError(f"Frame {frame} absent from {state_path}")
        hands = state["hand_joints_cam4_m"][position]
        presence = state["hand_presence"][position]
        distances = contact["joint_surface_distance_m"][position]
        object_id = int(state["object_id"][position])
        object_pose = state["object_pose_cam4"][position]
    image = Image.fromarray(base_rgb).convert("RGBA")
    overlay = Image.new("RGBA", image.size, (0, 0, 0, 0))
    draw = ImageDraw.Draw(overlay)
    for hand_index in range(2):
        if not presence[hand_index]:
            continue
        uv, valid = project(hands[hand_index], target_intrinsics, image.size)
        for start, end in HAND_EDGES:
            if valid[start] and valid[end]:
                color = distance_color(float(np.nanmean(distances[hand_index, [start, end]]))) + (230,)
                draw.line([tuple(uv[start]), tuple(uv[end])], fill=color, width=3)
        for joint, ok, distance in zip(uv, valid, distances[hand_index]):
            if ok:
                color = distance_color(float(distance)) + (255,)
                x, y = joint
                draw.ellipse((x - 3, y - 3, x + 3, y + 3), fill=color)
    object_name = OBJECTS[object_id][0]
    geometry_path = state_root / "object_geometry" / f"{object_name}_surface_20000.npz"
    with np.load(geometry_path) as geometry:
        vertices = geometry["vertices_object_m"].astype(np.float64)
    vertices_cam4 = vertices @ object_pose[:3, :3].T + object_pose[:3, 3]
    uv, valid = project(vertices_cam4, target_intrinsics, image.size)
    width, height = image.size
    valid &= (
        (uv[:, 0] >= 0) & (uv[:, 0] < width) & (uv[:, 1] >= 0) & (uv[:, 1] < height)
    )
    if valid.sum() >= 3:
        hull = convex_hull(uv[valid])
        draw.polygon(hull, fill=(255, 225, 45, 40), outline=(255, 235, 70, 255), width=3)
    return Image.alpha_composite(image, overlay).convert("RGB")


def add_label(image: Image.Image, label: str) -> Image.Image:
    image = image.copy()
    draw = ImageDraw.Draw(image, "RGBA")
    font = ImageFont.load_default(size=18)
    box = draw.textbbox((0, 0), label, font=font)
    draw.rounded_rectangle((8, 8, box[2] + 20, box[3] + 18), radius=5, fill=(0, 0, 0, 150))
    draw.text((14, 12), label, fill="white", font=font)
    return image


def render_frame(args: argparse.Namespace, frame: int) -> tuple[Image.Image, dict]:
    sequence_root = args.raw_root / args.sequence
    source = sequence_root / args.source_camera
    target = sequence_root / "cam4"
    size = (args.width, args.height)
    result, target_rgb, target_depth = load_and_reproject(
        source, target, frame, output_size=size, source_stride=args.source_stride
    )
    metrics, audit = audit_reprojection(result, target_rgb, target_depth, args.depth_tolerance_m)
    source_path = source / "rgb" / f"{frame:06d}.png"
    source_rgb = Image.open(source_path).convert("RGB").resize(size, Image.Resampling.BILINEAR)
    warp = result.rgb.copy()
    overlay = physical_overlay(
        warp,
        args.state_root,
        args.sequence,
        frame,
        load_intrinsics(target / "cam_intrinsics.txt"),
    )
    mask_rgb = np.zeros_like(warp)
    mask_rgb[result.valid] = (235, 235, 235)
    mask_rgb[audit["behind_target"]] = (230, 65, 65)
    mask_rgb[audit["in_front_target"]] = (65, 130, 230)
    error = np.zeros_like(warp)
    finite = np.isfinite(audit["depth_error_m"])
    scaled_error = np.clip(np.abs(audit["depth_error_m"]) / 0.10, 0, 1)
    error[..., 0][finite] = np.rint(255 * scaled_error[finite]).astype(np.uint8)
    error[..., 1][finite] = np.rint(255 * (1 - scaled_error[finite])).astype(np.uint8)
    panels = [
        add_label(source_rgb, args.source_camera + " RGB-D source"),
        add_label(Image.fromarray(warp), "source-only z-buffer warp"),
        add_label(Image.fromarray(target_rgb), "cam4 RGB target (audit only)"),
        add_label(Image.fromarray(mask_rgb), "coverage; red=behind target depth"),
        add_label(overlay, "warp + Oracle hand/object geometry"),
        add_label(Image.fromarray(error), "target-depth error 0-10 cm (audit only)"),
    ]
    canvas = Image.new("RGB", (args.width * 3, args.height * 2), (12, 12, 12))
    for index, panel in enumerate(panels):
        canvas.paste(panel, ((index % 3) * args.width, (index // 3) * args.height))
    metrics.update(
        {
            "sequence": args.sequence,
            "source_camera": args.source_camera,
            "target_camera": "cam4",
            "frame": frame,
            "source_stride": args.source_stride,
            "output_width": args.width,
            "output_height": args.height,
            "target_rgb_or_depth_used_for_reprojection": False,
        }
    )
    return canvas, metrics


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--raw-root", type=Path, default=DEFAULT_RAW)
    parser.add_argument("--state-root", type=Path, default=DEFAULT_STATE)
    parser.add_argument("--sequence", default="subject1/h1/0")
    parser.add_argument("--source-camera", choices=["cam0", "cam1", "cam2", "cam3"], default="cam3")
    parser.add_argument("--frame", type=int, default=100)
    parser.add_argument("--start", type=int)
    parser.add_argument("--end", type=int)
    parser.add_argument("--width", type=int, default=640)
    parser.add_argument("--height", type=int, default=360)
    parser.add_argument("--source-stride", type=int, default=1)
    parser.add_argument("--depth-tolerance-m", type=float, default=0.03)
    parser.add_argument("--fps", type=int, default=30)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    output_root = DEFAULT_OUTPUT / args.sequence.replace("/", "_") / args.source_camera
    output_root.mkdir(parents=True, exist_ok=True)
    if args.start is None:
        canvas, metrics = render_frame(args, args.frame)
        output = args.output or output_root / f"frame_{args.frame:06d}.png"
        canvas.save(output)
        output.with_suffix(".json").write_text(json.dumps(metrics, indent=2) + "\n", encoding="utf-8")
        print(json.dumps({"output": str(output), **metrics}, indent=2))
        return
    if args.end is None or args.end < args.start:
        parser.error("--end must be provided and be >= --start")
    output = args.output or output_root / f"clip_{args.start:06d}_{args.end:06d}.mp4"
    metrics_by_frame = []
    with tempfile.TemporaryDirectory(prefix="h2o-reprojection-", dir="/tmp") as temporary:
        temporary_path = Path(temporary)
        for output_index, frame in enumerate(range(args.start, args.end + 1)):
            canvas, metrics = render_frame(args, frame)
            canvas.save(temporary_path / f"{output_index:06d}.png")
            metrics_by_frame.append(metrics)
        subprocess.run(
            [
                "ffmpeg",
                "-y",
                "-loglevel",
                "error",
                "-framerate",
                str(args.fps),
                "-i",
                str(temporary_path / "%06d.png"),
                "-c:v",
                "libx264",
                "-pix_fmt",
                "yuv420p",
                str(output),
            ],
            check=True,
        )
    summary = {
        "output": str(output),
        "frames": len(metrics_by_frame),
        "mean_raw_coverage_fraction": float(np.mean([x["raw_coverage_fraction"] for x in metrics_by_frame])),
        "mean_depth_consistent_fraction_of_warp": float(
            np.mean([x["depth_consistent_fraction_of_warp"] for x in metrics_by_frame])
        ),
        "mean_consistent_rgb_l1": float(np.mean([x["consistent_rgb_l1"] for x in metrics_by_frame])),
        "per_frame": metrics_by_frame,
    }
    output.with_suffix(".json").write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({key: value for key, value in summary.items() if key != "per_frame"}, indent=2))


if __name__ == "__main__":
    main()
