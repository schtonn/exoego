#!/usr/bin/env python3
"""Audit H2O cam0-cam4 calibration and annotation consistency."""

from __future__ import annotations

import argparse
import csv
import json
import math
from pathlib import Path

import numpy as np
from PIL import Image


DEFAULT_ROOT = Path("datasets/H2O/raw")
DEFAULT_OUTPUT = Path("datasets/H2O/oracle_state")


def load_flat(path: Path, expected: int) -> np.ndarray:
    values = np.loadtxt(path, dtype=np.float64).reshape(-1)
    if values.size != expected or not np.all(np.isfinite(values)):
        raise ValueError(f"Invalid {path}: expected {expected} finite values")
    return values


def transform_points(transform: np.ndarray, points: np.ndarray) -> np.ndarray:
    return points @ transform[:3, :3].T + transform[:3, 3]


def load_hand(path: Path) -> tuple[np.ndarray, np.ndarray]:
    values = load_flat(path, 128)
    presence = values[[0, 64]] > 0.5
    joints = np.stack((values[1:64].reshape(21, 3), values[65:128].reshape(21, 3)))
    return presence, joints


def load_object(path: Path) -> tuple[int, np.ndarray]:
    values = load_flat(path, 17)
    return int(round(values[0])), values[1:].reshape(4, 4)


def rotation_angle_deg(left: np.ndarray, right: np.ndarray) -> float:
    relative = left.T @ right
    cosine = float(np.clip((np.trace(relative) - 1.0) / 2.0, -1.0, 1.0))
    return math.degrees(math.acos(cosine))


def project(points: np.ndarray, intrinsics: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    depth = points[:, 2]
    uv = np.full((len(points), 2), np.nan, dtype=np.float64)
    valid = depth > 1e-8
    fx, fy, cx, cy = intrinsics[:4]
    uv[valid, 0] = fx * points[valid, 0] / depth[valid] + cx
    uv[valid, 1] = fy * points[valid, 1] / depth[valid] + cy
    return uv, valid


def stats(values: list[float], prefix: str) -> dict[str, float | None]:
    data = np.asarray(values, dtype=np.float64)
    data = data[np.isfinite(data)]
    if not len(data):
        return {f"{prefix}_{name}": None for name in ("mean", "p95", "max")}
    return {
        f"{prefix}_mean": float(np.mean(data)),
        f"{prefix}_p95": float(np.percentile(data, 95)),
        f"{prefix}_max": float(np.max(data)),
    }


def frame_stems(sequence: Path) -> list[str]:
    return sorted(path.stem for path in (sequence / "cam4" / "cam_pose").glob("*.txt"))


def sample_stems(stems: list[str], stride: int) -> list[str]:
    selected = stems[::stride]
    if stems and selected[-1] != stems[-1]:
        selected.append(stems[-1])
    return selected


def audit_camera(sequence: Path, camera_index: int, stride: int, root: Path) -> dict:
    reference = sequence / "cam4"
    camera = sequence / f"cam{camera_index}"
    all_stems = frame_stems(sequence)
    selected = sample_stems(all_stems, stride)
    intrinsics = load_flat(camera / "cam_intrinsics.txt", 6)
    width, height = int(intrinsics[4]), int(intrinsics[5])
    first_rgb = camera / "rgb" / f"{all_stems[0]}.png"
    with Image.open(first_rgb) as image:
        rgb_width, rgb_height = image.size

    hand_world_errors: list[float] = []
    hand_camera_errors: list[float] = []
    hand_pixel_errors: list[float] = []
    object_translation_errors: list[float] = []
    object_rotation_errors: list[float] = []
    in_frame: list[float] = []
    camera_positions = []
    camera_rotations = []
    presence_disagreements = 0
    object_id_disagreements = 0
    compared_points = 0

    for stem in selected:
        reference_pose = load_flat(reference / "cam_pose" / f"{stem}.txt", 16).reshape(4, 4)
        camera_pose = load_flat(camera / "cam_pose" / f"{stem}.txt", 16).reshape(4, 4)
        world_to_camera = np.linalg.inv(camera_pose)
        camera_positions.append(camera_pose[:3, 3])
        camera_rotations.append(camera_pose[:3, :3])

        reference_presence, reference_hand = load_hand(reference / "hand_pose" / f"{stem}.txt")
        camera_presence, camera_hand = load_hand(camera / "hand_pose" / f"{stem}.txt")
        presence_disagreements += int(np.count_nonzero(reference_presence != camera_presence))
        for hand_index in range(2):
            if not (reference_presence[hand_index] and camera_presence[hand_index]):
                continue
            reference_world = transform_points(reference_pose, reference_hand[hand_index])
            camera_world = transform_points(camera_pose, camera_hand[hand_index])
            reference_in_camera = transform_points(world_to_camera, reference_world)
            world_error = np.linalg.norm(reference_world - camera_world, axis=1)
            camera_error = np.linalg.norm(reference_in_camera - camera_hand[hand_index], axis=1)
            reference_uv, reference_valid = project(reference_in_camera, intrinsics)
            camera_uv, camera_valid = project(camera_hand[hand_index], intrinsics)
            valid = reference_valid & camera_valid
            pixel_error = np.linalg.norm(reference_uv[valid] - camera_uv[valid], axis=1)
            inside = (
                valid
                & (reference_uv[:, 0] >= 0)
                & (reference_uv[:, 0] < width)
                & (reference_uv[:, 1] >= 0)
                & (reference_uv[:, 1] < height)
            )
            hand_world_errors.extend(world_error.tolist())
            hand_camera_errors.extend(camera_error.tolist())
            hand_pixel_errors.extend(pixel_error.tolist())
            in_frame.extend(inside[valid].astype(float).tolist())
            compared_points += int(np.count_nonzero(valid))

        reference_id, reference_object_camera = load_object(
            reference / "obj_pose_rt" / f"{stem}.txt"
        )
        camera_id, camera_object_camera = load_object(camera / "obj_pose_rt" / f"{stem}.txt")
        object_id_disagreements += int(reference_id != camera_id)
        reference_object_world = reference_pose @ reference_object_camera
        camera_object_world = camera_pose @ camera_object_camera
        object_translation_errors.append(
            float(np.linalg.norm(reference_object_world[:3, 3] - camera_object_world[:3, 3]))
        )
        object_rotation_errors.append(
            rotation_angle_deg(reference_object_world[:3, :3], camera_object_world[:3, :3])
        )

    positions = np.asarray(camera_positions)
    rotations = np.asarray(camera_rotations)
    origin = positions[0]
    base_rotation = rotations[0]
    camera_position_range = float(np.max(np.linalg.norm(positions - origin, axis=1)))
    camera_rotation_range = float(
        max(rotation_angle_deg(base_rotation, rotation) for rotation in rotations)
    )
    counts = {
        name: len(list((camera / name).glob("*")))
        for name in ("rgb", "depth", "cam_pose", "hand_pose", "obj_pose_rt")
    }
    row = {
        "sequence": str(sequence.relative_to(root)),
        "camera": f"cam{camera_index}",
        "frames": len(all_stems),
        "sampled_frames": len(selected),
        **{f"{name}_files": count for name, count in counts.items()},
        "file_counts_match": all(count == len(all_stems) for count in counts.values()),
        "fx": float(intrinsics[0]),
        "fy": float(intrinsics[1]),
        "cx": float(intrinsics[2]),
        "cy": float(intrinsics[3]),
        "intrinsic_width": width,
        "intrinsic_height": height,
        "rgb_width": rgb_width,
        "rgb_height": rgb_height,
        "resolution_matches": (width, height) == (rgb_width, rgb_height),
        "presence_disagreements": presence_disagreements,
        "object_id_disagreements": object_id_disagreements,
        "compared_hand_points": compared_points,
        "projected_in_frame_fraction": float(np.mean(in_frame)) if in_frame else None,
        "camera_position_range_m": camera_position_range,
        "camera_rotation_range_deg": camera_rotation_range,
    }
    row.update(stats(hand_world_errors, "hand_world_error_m"))
    row.update(stats(hand_camera_errors, "hand_camera_error_m"))
    row.update(stats(hand_pixel_errors, "hand_pixel_error_px"))
    row.update(stats(object_translation_errors, "object_translation_error_m"))
    row.update(stats(object_rotation_errors, "object_rotation_error_deg"))
    return row


def summarize(rows: list[dict], stride: int) -> dict:
    def aggregate(key: str) -> dict[str, float | None]:
        return stats([row[key] for row in rows if row[key] is not None], key)

    per_camera = {}
    for camera in (f"cam{index}" for index in range(5)):
        selected = [row for row in rows if row["camera"] == camera]
        per_camera[camera] = {
            "rows": len(selected),
            "sampled_frames": sum(row["sampled_frames"] for row in selected),
            "hand_pixel_error_px_p95_max": max(row["hand_pixel_error_px_p95"] for row in selected),
            "object_translation_error_m_p95_max": max(
                row["object_translation_error_m_p95"] for row in selected
            ),
            "projected_in_frame_fraction_mean": float(
                np.mean([row["projected_in_frame_fraction"] for row in selected])
            ),
        }
    checks = {
        "all_file_counts_match": all(row["file_counts_match"] for row in rows),
        "all_resolutions_match": all(row["resolution_matches"] for row in rows),
        "no_object_id_disagreements": not any(row["object_id_disagreements"] for row in rows),
        "hand_pixel_p95_below_1px": max(row["hand_pixel_error_px_p95"] for row in rows) < 1.0,
        "object_translation_p95_below_1mm": max(
            row["object_translation_error_m_p95"] for row in rows
        )
        < 0.001,
        "object_rotation_p95_below_0_1deg": max(
            row["object_rotation_error_deg_p95"] for row in rows
        )
        < 0.1,
    }
    return {
        "stride": stride,
        "sequences": len({row["sequence"] for row in rows}),
        "camera_rows": len(rows),
        "sampled_frame_camera_pairs": sum(row["sampled_frames"] for row in rows),
        "checks": checks,
        "passed": all(checks.values()),
        "observations": {
            "hand_presence_is_view_specific": True,
            "hand_presence_disagreements": sum(
                row["presence_disagreements"] for row in rows
            ),
            "rows_with_hand_presence_disagreement": sum(
                row["presence_disagreements"] > 0 for row in rows
            ),
            "hand_presence_disagreement_fraction": sum(
                row["presence_disagreements"] for row in rows
            )
            / max(1, 2 * sum(row["sampled_frames"] for row in rows)),
        },
        "per_camera": per_camera,
        "global": {
            "hand_world_error_m_p95_max": max(row["hand_world_error_m_p95"] for row in rows),
            "hand_pixel_error_px_p95_max": max(row["hand_pixel_error_px_p95"] for row in rows),
            "object_translation_error_m_p95_max": max(
                row["object_translation_error_m_p95"] for row in rows
            ),
            "object_rotation_error_deg_p95_max": max(
                row["object_rotation_error_deg_p95"] for row in rows
            ),
        },
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=DEFAULT_ROOT)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--stride", type=int, default=30)
    parser.add_argument("--sequence", action="append", help="Relative sequence; repeatable")
    args = parser.parse_args()
    if args.sequence:
        sequences = [args.root / value for value in args.sequence]
    else:
        sequences = sorted(path.parent for path in args.root.glob("subject*/*/*/cam4"))
    rows = []
    for sequence_index, sequence in enumerate(sequences, 1):
        for camera_index in range(5):
            rows.append(audit_camera(sequence, camera_index, args.stride, args.root))
        if sequence_index % 10 == 0 or sequence_index == len(sequences):
            print(f"Audited {sequence_index}/{len(sequences)} sequences", flush=True)
    args.output.mkdir(parents=True, exist_ok=True)
    csv_path = args.output / "multiview_calibration_audit.csv"
    with csv_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    summary = summarize(rows, args.stride)
    summary_path = args.output / "multiview_calibration_audit.summary.json"
    summary_path.write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
