#!/usr/bin/env python3
"""Render a full-resolution cam0-cam4 H2O calibration contact sheet."""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw, ImageFont
from scipy.spatial import ConvexHull

from h2o_oracle_state import HAND_EDGES, OBJECTS, load_flat, load_obj_vertices, transform_points


DEFAULT_ROOT = Path("datasets/H2O/raw")
DEFAULT_OUTPUT = Path("datasets/H2O/previews/multiview_qa")
HAND_COLORS = ((38, 214, 255, 255), (255, 70, 190, 255))


def project(points: np.ndarray, intrinsics: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    depth = points[:, 2]
    valid = depth > 1e-8
    uv = np.full((len(points), 2), np.nan, dtype=np.float64)
    fx, fy, cx, cy = intrinsics[:4]
    uv[valid, 0] = fx * points[valid, 0] / depth[valid] + cx
    uv[valid, 1] = fy * points[valid, 1] / depth[valid] + cy
    return uv, valid


def draw_panel(root: Path, sequence: str, camera_index: int, frame: int) -> Image.Image:
    camera = root / sequence / f"cam{camera_index}"
    stem = f"{frame:06d}"
    image = Image.open(camera / "rgb" / f"{stem}.png").convert("RGBA")
    overlay = Image.new("RGBA", image.size, (0, 0, 0, 0))
    draw = ImageDraw.Draw(overlay)
    intrinsics = load_flat(camera / "cam_intrinsics.txt", 6)

    hand = load_flat(camera / "hand_pose" / f"{stem}.txt", 128)
    presence = hand[[0, 64]] > 0.5
    hands = (hand[1:64].reshape(21, 3), hand[65:128].reshape(21, 3))
    for hand_index, joints in enumerate(hands):
        if not presence[hand_index]:
            continue
        uv, valid = project(joints, intrinsics)
        color = HAND_COLORS[hand_index]
        for start, end in HAND_EDGES:
            if valid[start] and valid[end]:
                draw.line(
                    [tuple(uv[start]), tuple(uv[end])],
                    fill=color,
                    width=5,
                )
        for point, ok in zip(uv, valid):
            if ok:
                x, y = point
                draw.ellipse((x - 5, y - 5, x + 5, y + 5), fill=color)

    object_values = load_flat(camera / "obj_pose_rt" / f"{stem}.txt", 17)
    object_id = int(round(object_values[0]))
    object_pose = object_values[1:].reshape(4, 4)
    vertices = load_obj_vertices(root / "object" / OBJECTS[object_id][1])
    vertices_camera = transform_points(object_pose, vertices)
    uv, valid = project(vertices_camera, intrinsics)
    width, height = image.size
    valid &= (
        (uv[:, 0] >= -width)
        & (uv[:, 0] < 2 * width)
        & (uv[:, 1] >= -height)
        & (uv[:, 1] < 2 * height)
    )
    object_uv = uv[valid]
    if len(object_uv) >= 3:
        hull = ConvexHull(object_uv)
        polygon = [tuple(point) for point in object_uv[hull.vertices]]
        draw.polygon(polygon, fill=(255, 225, 45, 38), outline=(255, 240, 80, 255), width=5)
    for point in object_uv[:: max(1, len(object_uv) // 100)]:
        x, y = point
        draw.ellipse((x - 2, y - 2, x + 2, y + 2), fill=(255, 240, 80, 180))

    label = f"cam{camera_index}"
    font = ImageFont.load_default(size=28)
    box = draw.textbbox((0, 0), label, font=font)
    label_width = box[2] - box[0]
    label_height = box[3] - box[1]
    x, y = 24, height - label_height - 24
    draw.rounded_rectangle(
        (x - 12, y - 8, x + label_width + 12, y + label_height + 8),
        radius=8,
        fill=(0, 0, 0, 150),
    )
    draw.text((x, y), label, font=font, fill=(255, 255, 255, 255))
    return Image.alpha_composite(image, overlay).convert("RGB")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=DEFAULT_ROOT)
    parser.add_argument("--sequence", default="subject1/h1/0")
    parser.add_argument("--frame", type=int, default=100)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    panels = [draw_panel(args.root, args.sequence, index, args.frame) for index in range(5)]
    canvas = Image.new("RGB", (3840, 1440), (12, 12, 12))
    for index, panel in enumerate(panels[:3]):
        canvas.paste(panel, (index * 1280, 0))
    canvas.paste(panels[3], (640, 720))
    canvas.paste(panels[4], (1920, 720))
    output = args.output or (
        DEFAULT_OUTPUT
        / args.sequence.replace("/", "_")
        / f"frame_{args.frame:06d}_cam0-cam4.png"
    )
    output.parent.mkdir(parents=True, exist_ok=True)
    canvas.save(output, optimize=True)
    print(output)


if __name__ == "__main__":
    main()
