#!/usr/bin/env python3
"""Test whether an exo-derived 3-D hand mesh can safely gate small-hole repair.

The inference mask uses only the existing multi-view student hand joints and the
same predicted ego camera trajectory as the main render.  Cam4 hand joints and
camera poses are read only to construct an evaluation reference.
"""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
import sys

import cv2
import numpy as np
from PIL import Image

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from h2o_geometric_baseline.reprojection import load_intrinsics, load_pose


FINGER_CHAINS = ((1, 2, 3, 4), (5, 6, 7, 8), (9, 10, 11, 12),
                 (13, 14, 15, 16), (17, 18, 19, 20))
PALM_FACES = ((0, 1, 5), (0, 5, 9), (0, 9, 13), (0, 13, 17))


def temporal_smooth(values: np.ndarray, radius: int = 4) -> np.ndarray:
    flat = values.reshape(len(values), -1).astype(np.float64).copy()
    for channel in range(flat.shape[1]):
        series = flat[:, channel]
        finite = np.isfinite(series)
        if not np.any(finite):
            series[:] = 0
        elif not np.all(finite):
            series[~finite] = np.interp(
                np.flatnonzero(~finite), np.flatnonzero(finite), series[finite]
            )
        robust = np.empty_like(series)
        for index in range(len(series)):
            robust[index] = np.median(
                series[max(0, index - radius):min(len(series), index + radius + 1)]
            )
        weights = np.arange(1, radius + 2, dtype=np.float64)
        weights = np.concatenate((weights, weights[-2::-1]))
        padded = np.pad(robust, radius, mode="edge")
        flat[:, channel] = np.convolve(padded, weights / weights.sum(), mode="valid")
    return flat.reshape(values.shape)


def scaled_rotation(rotation: np.ndarray, amount: float) -> np.ndarray:
    vector, _ = cv2.Rodrigues(rotation)
    result, _ = cv2.Rodrigues(vector * amount)
    return result


def predicted_pose_map(
    frames: list[int], initial_pose: np.ndarray, head_record: dict
) -> dict[int, np.ndarray]:
    translations = [np.zeros(3)]
    rotations = [np.eye(3)]
    records = {int(value["frame"]): value for value in head_record["future"]}
    for frame in frames[1:]:
        motion = records[frame]
        moved = np.asarray(motion["estimated_head_transform_world"]) @ initial_pose
        translations.append(moved[:3, 3] - initial_pose[:3, 3])
        rotations.append(moved[:3, :3] @ initial_pose[:3, :3].T)
    translations = temporal_smooth(np.stack(translations), radius=4)
    rotation_values = temporal_smooth(np.stack(rotations), radius=4)
    projected = []
    for value in rotation_values:
        u, _, vt = np.linalg.svd(value)
        value = u @ vt
        if np.linalg.det(value) < 0:
            u[:, -1] *= -1
            value = u @ vt
        projected.append(value)
    translations -= translations[0]
    projected[0] = np.eye(3)
    output = {}
    for frame, translation, rotation in zip(frames, translations, projected):
        pose = initial_pose.copy()
        pose[:3, 3] += translation
        pose[:3, :3] = scaled_rotation(rotation, 0.5) @ initial_pose[:3, :3]
        output[frame] = pose
    return output


def interpolate_state(
    frames: np.ndarray,
    joints: np.ndarray,
    confidence: np.ndarray,
    frame: int,
) -> tuple[np.ndarray, np.ndarray]:
    right = int(np.searchsorted(frames, frame))
    if right == 0:
        return joints[0], confidence[0]
    if right >= len(frames):
        return joints[-1], confidence[-1]
    left = right - 1
    span = max(int(frames[right] - frames[left]), 1)
    alpha = float(frame - frames[left]) / span
    return (
        (1.0 - alpha) * joints[left] + alpha * joints[right],
        (1.0 - alpha) * confidence[left] + alpha * confidence[right],
    )


def append_tube(
    vertices: list[np.ndarray], faces: list[tuple[int, int, int]],
    start: np.ndarray, end: np.ndarray, start_radius: float, end_radius: float,
    sides: int = 8,
) -> None:
    axis = end - start
    length = float(np.linalg.norm(axis))
    if not np.isfinite(length) or length < 1e-6:
        return
    axis /= length
    reference = np.asarray((0.0, 0.0, 1.0))
    if abs(float(axis @ reference)) > 0.9:
        reference = np.asarray((0.0, 1.0, 0.0))
    basis_u = np.cross(axis, reference)
    basis_u /= max(float(np.linalg.norm(basis_u)), 1e-8)
    basis_v = np.cross(axis, basis_u)
    base = len(vertices)
    for center, radius in ((start, start_radius), (end, end_radius)):
        for index in range(sides):
            angle = 2.0 * math.pi * index / sides
            vertices.append(center + radius * (math.cos(angle) * basis_u + math.sin(angle) * basis_v))
    for index in range(sides):
        nxt = (index + 1) % sides
        faces.append((base + index, base + nxt, base + sides + index))
        faces.append((base + nxt, base + sides + nxt, base + sides + index))


def append_octahedron(
    vertices: list[np.ndarray], faces: list[tuple[int, int, int]],
    center: np.ndarray, radius: float,
) -> None:
    base = len(vertices)
    axes = np.eye(3) * radius
    vertices.extend([center + axes[0], center - axes[0], center + axes[1],
                     center - axes[1], center + axes[2], center - axes[2]])
    faces.extend((base + a, base + b, base + c) for a, b, c in (
        (0, 2, 4), (2, 1, 4), (1, 3, 4), (3, 0, 4),
        (2, 0, 5), (1, 2, 5), (3, 1, 5), (0, 3, 5),
    ))


def hand_mesh(joints_camera: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    vertices: list[np.ndarray] = []
    faces: list[tuple[int, int, int]] = []
    for face in PALM_FACES:
        base = len(vertices)
        vertices.extend(joints_camera[list(face)])
        faces.append((base, base + 1, base + 2))
    for chain_index, chain in enumerate(FINGER_CHAINS):
        connected = (0,) + chain
        for segment, (start_index, end_index) in enumerate(zip(connected[:-1], connected[1:])):
            # Conservative anatomical widths: the mesh only authorizes repair
            # inside an already enclosed 2-D hole and never expands the outline.
            start_radius = (0.0115, 0.0100, 0.0080, 0.0065)[segment]
            end_radius = (0.0100, 0.0080, 0.0065, 0.0050)[segment]
            if chain_index == 0:
                start_radius *= 1.05
                end_radius *= 1.05
            append_tube(vertices, faces, joints_camera[start_index], joints_camera[end_index],
                        start_radius, end_radius)
        for segment, joint_index in enumerate(connected):
            radius = (0.0120, 0.0100, 0.0080, 0.0065, 0.0050)[segment]
            append_octahedron(vertices, faces, joints_camera[joint_index], radius)
    return np.asarray(vertices), np.asarray(faces, dtype=np.int32)


def project_mesh_mask(
    joints_camera: np.ndarray, valid_joints: np.ndarray,
    intrinsics: np.ndarray, size: int,
) -> np.ndarray:
    mask = np.zeros((size, size), dtype=np.uint8)
    native_width, native_height = intrinsics[4:6]
    fx = intrinsics[0] * size / native_width
    fy = intrinsics[1] * size / native_height
    cx = intrinsics[2] * size / native_width
    cy = intrinsics[3] * size / native_height
    for hand in range(2):
        if int(valid_joints[hand].sum()) < 12:
            continue
        value = joints_camera[hand]
        if not np.isfinite(value).all() or np.any(value[:, 2] <= 1e-4):
            continue
        vertices, faces = hand_mesh(value)
        uv = np.stack((fx * vertices[:, 0] / vertices[:, 2] + cx,
                       fy * vertices[:, 1] / vertices[:, 2] + cy), axis=1)
        for face in faces:
            polygon = np.rint(uv[face]).astype(np.int32)
            cv2.fillConvexPoly(mask, polygon, 255, lineType=cv2.LINE_AA)
    return mask > 0


def load_oracle_joints(sequence_root: Path, frame: int) -> tuple[np.ndarray, np.ndarray]:
    values = np.loadtxt(sequence_root / "cam4/hand_pose" / f"{frame:06d}.txt").reshape(-1)
    joints = np.zeros((2, 21, 3), dtype=np.float64)
    valid = np.zeros((2, 21), dtype=bool)
    for hand in range(2):
        chunk = values[64 * hand:64 * (hand + 1)]
        joints[hand] = chunk[1:].reshape(21, 3)
        valid[hand] = bool(chunk[0] > 0.5)
    return joints, valid


def enclosed_small_components(mask: np.ndarray, maximum_area: int) -> list[np.ndarray]:
    inverse = (~mask).astype(np.uint8)
    count, labels, stats, _ = cv2.connectedComponentsWithStats(inverse, connectivity=8)
    components = []
    height, width = mask.shape
    for component in range(1, count):
        x, y, w, h, area = (int(value) for value in stats[component])
        if x == 0 or y == 0 or x + w == width or y + h == height:
            continue
        if 1 <= area <= maximum_area:
            components.append(labels == component)
    return components


def boundary_jaggedness(component: np.ndarray) -> float:
    contours, _ = cv2.findContours(component.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    perimeter = sum(cv2.arcLength(contour, True) for contour in contours)
    area = max(float(component.sum()), 1.0)
    return float(perimeter * perimeter / (4.0 * math.pi * area))


def overlay(rgb: np.ndarray, mask: np.ndarray, color: tuple[int, int, int]) -> np.ndarray:
    value = rgb.astype(np.float32).copy()
    value[mask] = 0.48 * value[mask] + 0.52 * np.asarray(color, dtype=np.float32)
    return np.rint(value).astype(np.uint8)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--sequence-root", type=Path, required=True)
    parser.add_argument("--model-input-root", type=Path, required=True)
    parser.add_argument("--student-state", type=Path, required=True)
    parser.add_argument("--head-summary", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--maximum-hole-area", type=int, default=128)
    args = parser.parse_args()

    manifest = json.loads((args.model_input_root / "manifest.json").read_text())
    frames = [int(value["dataset_frame"]) for value in manifest["frames"]]
    pair_id = manifest["pair_id"]
    size = int(manifest["image_size"])
    intrinsics = load_intrinsics(args.sequence_root / "cam4/cam_intrinsics.txt")
    initial_pose = load_pose(args.sequence_root / "cam4/cam_pose" / f"{frames[0]:06d}.txt")
    head_summary = json.loads(args.head_summary.read_text())
    head_record = next(value for value in head_summary["per_clip"] if value["pair_id"] == pair_id)
    poses = predicted_pose_map(frames, initial_pose, head_record)
    with np.load(args.student_state) as archive:
        state_frames = archive["frames"].copy()
        state_joints = archive["hand_joints_world_m"].copy()
        state_confidence = archive["joint_confidence"].copy()

    records = []
    candidate_count = true_count = accepted_count = correct_count = 0
    jagged_candidate = jagged_true = jagged_accepted = jagged_correct = 0
    sweep = {
        f"dilate_{dilation}_overlap_{threshold:.2f}": {
            "dilation_px": dilation,
            "overlap_threshold": threshold,
            "accepted": 0,
            "correct": 0,
            "jagged_accepted": 0,
            "jagged_correct": 0,
        }
        for dilation in (0, 2, 4, 6, 8, 12)
        for threshold in (0.25, 0.50)
    }
    silhouette_intersection = silhouette_union = 0
    fill_masks: list[np.ndarray] = []
    estimated_masks: list[np.ndarray] = []
    oracle_masks: list[np.ndarray] = []
    for index, frame in enumerate(frames):
        raw = np.asarray(Image.open(args.model_input_root / "arm_masks" / f"{index:06d}.png")) > 0
        if index == 0:
            fill_masks.append(np.zeros_like(raw))
            estimated_masks.append(np.zeros_like(raw))
            oracle_masks.append(np.zeros_like(raw))
            continue
        joints_world, confidence = interpolate_state(
            state_frames, state_joints, state_confidence, frame
        )
        camera_from_world = np.linalg.inv(poses[frame])
        estimated_camera = joints_world @ camera_from_world[:3, :3].T + camera_from_world[:3, 3]
        estimated = project_mesh_mask(estimated_camera, confidence >= 0.15, intrinsics, size)
        oracle_joints, oracle_valid = load_oracle_joints(args.sequence_root, frame)
        oracle = project_mesh_mask(oracle_joints, oracle_valid, intrinsics, size)
        estimated_masks.append(estimated)
        oracle_masks.append(oracle)
        silhouette_intersection += int((estimated & oracle).sum())
        silhouette_union += int((estimated | oracle).sum())

        accepted = np.zeros_like(raw)
        dilated_estimated = {
            dilation: (
                estimated if dilation == 0 else cv2.dilate(
                    estimated.astype(np.uint8),
                    cv2.getStructuringElement(
                        cv2.MORPH_ELLIPSE, (2 * dilation + 1, 2 * dilation + 1)
                    ),
                ) > 0
            )
            for dilation in (0, 2, 4, 6, 8, 12)
        }
        frame_candidates = frame_true = frame_accepted = frame_correct = 0
        for component in enclosed_small_components(raw, args.maximum_hole_area):
            area = int(component.sum())
            oracle_ratio = float((component & oracle).sum()) / area
            estimated_ratio = float((component & estimated).sum()) / area
            is_true = oracle_ratio >= 0.5
            is_accepted = estimated_ratio >= 0.5
            is_jagged = boundary_jaggedness(component) >= 1.45
            frame_candidates += 1
            frame_true += int(is_true)
            frame_accepted += int(is_accepted)
            frame_correct += int(is_true and is_accepted)
            if is_accepted:
                accepted |= component
            if is_jagged:
                jagged_candidate += 1
                jagged_true += int(is_true)
                jagged_accepted += int(is_accepted)
                jagged_correct += int(is_true and is_accepted)
            for values in sweep.values():
                relaxed_ratio = float(
                    (component & dilated_estimated[values["dilation_px"]]).sum()
                ) / area
                relaxed_accept = relaxed_ratio >= values["overlap_threshold"]
                values["accepted"] += int(relaxed_accept)
                values["correct"] += int(relaxed_accept and is_true)
                if is_jagged:
                    values["jagged_accepted"] += int(relaxed_accept)
                    values["jagged_correct"] += int(relaxed_accept and is_true)
        fill_masks.append(accepted)
        candidate_count += frame_candidates
        true_count += frame_true
        accepted_count += frame_accepted
        correct_count += frame_correct
        records.append({
            "frame": frame,
            "candidate_components": frame_candidates,
            "oracle_supported_components": frame_true,
            "mesh_accepted_components": frame_accepted,
            "correct_mesh_components": frame_correct,
            "filled_pixels": int(accepted.sum()),
        })

    precision = correct_count / max(accepted_count, 1)
    recall = correct_count / max(true_count, 1)
    f1 = 2.0 * precision * recall / max(precision + recall, 1e-12)
    for values in sweep.values():
        values["precision"] = values["correct"] / max(values["accepted"], 1)
        values["recall"] = values["correct"] / max(true_count, 1)
        values["jagged_precision"] = values["jagged_correct"] / max(values["jagged_accepted"], 1)
        values["jagged_recall"] = values["jagged_correct"] / max(jagged_true, 1)
        values["jagged_f1"] = (
            2.0 * values["jagged_precision"] * values["jagged_recall"]
            / max(values["jagged_precision"] + values["jagged_recall"], 1e-12)
        )
    metrics = {
        "protocol": {
            "inference": "multi-view student 3-D joints + predicted ego trajectory",
            "evaluation_only": "cam4 hand_pose joints",
            "mano_parameters_used": False,
            "reason": "licensed MANO model files are not present locally",
        },
        "frames_evaluated": len(frames) - 1,
        "maximum_hole_area_px": args.maximum_hole_area,
        "estimated_vs_oracle_mesh_iou": silhouette_intersection / max(silhouette_union, 1),
        "small_hole_components": candidate_count,
        "oracle_hand_hole_components": true_count,
        "mesh_accepted_components": accepted_count,
        "mesh_correct_components": correct_count,
        "mesh_gate_precision": precision,
        "mesh_gate_recall": recall,
        "mesh_gate_f1": f1,
        "ungated_2d_precision": true_count / max(candidate_count, 1),
        "ungated_2d_recall": 1.0 if true_count else 0.0,
        "jagged_components": jagged_candidate,
        "jagged_oracle_hand_components": jagged_true,
        "jagged_mesh_accepted": jagged_accepted,
        "jagged_mesh_correct": jagged_correct,
        "jagged_mesh_precision": jagged_correct / max(jagged_accepted, 1),
        "jagged_mesh_recall": jagged_correct / max(jagged_true, 1),
        "planar_component_mesh_neighborhood_sweep": sweep,
        "per_frame": records,
    }
    args.output_root.mkdir(parents=True, exist_ok=True)
    (args.output_root / "metrics.json").write_text(
        json.dumps(metrics, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )

    ranked = np.argsort([-int(mask.sum()) for mask in fill_masks])[:6]
    rows = []
    for index in ranked:
        frame = frames[int(index)]
        rgb = np.asarray(Image.open(args.sequence_root / "cam4/rgb" / f"{frame:06d}.png").convert("RGB").resize((size, size)))
        raw = np.asarray(Image.open(args.model_input_root / "arm_masks" / f"{int(index):06d}.png")) > 0
        row = np.concatenate((
            rgb,
            overlay(rgb, raw, (217, 70, 239)),
            overlay(rgb, estimated_masks[int(index)], (34, 197, 94)),
            overlay(rgb, fill_masks[int(index)], (20, 184, 166)),
            overlay(rgb, oracle_masks[int(index)], (239, 68, 68)),
        ), axis=1)
        rows.append(row)
        Image.fromarray(fill_masks[int(index)].astype(np.uint8) * 255).save(
            args.output_root / f"mesh_fill_{frame:06d}.png"
        )
    if rows:
        Image.fromarray(np.concatenate(rows, axis=0)).save(args.output_root / "audit.png")
    print(json.dumps({key: value for key, value in metrics.items() if key != "per_frame"}, indent=2))


if __name__ == "__main__":
    main()
