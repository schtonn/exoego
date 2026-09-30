#!/usr/bin/env python3
"""Build and audit a compact oracle 4D state from H2O pose annotations.

The implementation follows the coordinate convention used by the official
H2OPlayer:

* ``cam_pose`` maps cam4 coordinates to the dataset world frame (C2W).
* ``obj_pose_rt`` maps object-local coordinates to cam4 (O2C).
* therefore ``object_pose_world = cam_pose @ object_pose_cam4``.

The pose-only release does not include RGB or camera intrinsics, so pixel
reprojection is deliberately deferred until a full subject archive is ready.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
from pathlib import Path
from typing import Iterable

import numpy as np


DEFAULT_ROOT = Path("datasets/H2O/raw")
DEFAULT_OUTPUT = Path("datasets/H2O/oracle_state")

OBJECTS = {
    1: ("book", Path("book/book.obj")),
    2: ("espresso", Path("espresso/espresso.obj")),
    3: ("lotion", Path("lotion/lotion.obj")),
    4: ("spray", Path("spray/lotion_spray.obj")),
    5: ("milk", Path("milk/milk.obj")),
    6: ("cocoa", Path("cocoa/cocoa.obj")),
    7: ("chips", Path("chips/chips.obj")),
    8: ("cappuccino", Path("cappuccino/cappuccino.obj")),
}

ACTIONS = [
    "background",
    "grab book",
    "grab espresso",
    "grab lotion",
    "grab spray",
    "grab milk",
    "grab cocoa",
    "grab chips",
    "grab cappuccino",
    "place book",
    "place espresso",
    "place lotion",
    "place spray",
    "place milk",
    "place cocoa",
    "place chips",
    "place cappuccino",
    "open lotion",
    "open milk",
    "open chips",
    "close lotion",
    "close milk",
    "close chips",
    "pour milk",
    "take out espresso",
    "take out cocoa",
    "take out chips",
    "take out cappuccino",
    "put in espresso",
    "put in cocoa",
    "put in cappuccino",
    "apply lotion",
    "apply spray",
    "read book",
    "read espresso",
    "spray spray",
    "squeeze lotion",
]

# H2O uses wrist + four joints for each of five fingers.
HAND_EDGES = tuple(
    [(0, base) for base in (1, 5, 9, 13, 17)]
    + [(base + offset, base + offset + 1) for base in (1, 5, 9, 13, 17) for offset in range(3)]
)

CORE_FRAME_DIRS = (
    "hand_pose",
    "hand_pose_mano",
    "obj_pose",
    "obj_pose_rt",
    "cam_pose",
)
OPTIONAL_FRAME_DIRS = ("action_label", "verb_label")
ALL_FRAME_DIRS = CORE_FRAME_DIRS + OPTIONAL_FRAME_DIRS


def load_flat(path: Path, expected: int | None = None) -> np.ndarray:
    values = np.loadtxt(path, dtype=np.float64).reshape(-1)
    if expected is not None and values.size != expected:
        raise ValueError(f"{path}: expected {expected} values, found {values.size}")
    if not np.all(np.isfinite(values)):
        raise ValueError(f"{path}: contains non-finite values")
    return values


def load_obj_vertices(path: Path) -> np.ndarray:
    vertices: list[list[float]] = []
    with path.open("r", encoding="utf-8", errors="replace") as handle:
        for line in handle:
            if line.startswith("v "):
                fields = line.split()
                vertices.append([float(fields[1]), float(fields[2]), float(fields[3])])
    if not vertices:
        raise ValueError(f"No vertices found in {path}")
    return np.asarray(vertices, dtype=np.float64)


def transform_points(transform: np.ndarray, points: np.ndarray) -> np.ndarray:
    return points @ transform[:3, :3].T + transform[:3, 3]


def rotation_errors(transforms: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    rotations = transforms[:, :3, :3]
    identity = np.eye(3, dtype=np.float64)
    orthogonality = np.linalg.norm(
        rotations.transpose(0, 2, 1) @ rotations - identity[None], axis=(1, 2)
    )
    determinant = np.linalg.det(rotations)
    bottom_row = np.max(
        np.abs(transforms[:, 3, :] - np.array([0.0, 0.0, 0.0, 1.0])[None]), axis=1
    )
    return orthogonality, determinant, bottom_row


def relative_angular_velocity(rotations: np.ndarray, fps: float) -> np.ndarray:
    """Return a stable world-frame rotation-vector velocity for each frame."""
    output = np.zeros((len(rotations), 3), dtype=np.float64)
    for index in range(1, len(rotations)):
        previous = rotations[index - 1]
        current = rotations[index]
        relative = previous.T @ current
        cosine = float(np.clip((np.trace(relative) - 1.0) / 2.0, -1.0, 1.0))
        angle = math.acos(cosine)
        skew = np.array(
            [relative[2, 1] - relative[1, 2],
             relative[0, 2] - relative[2, 0],
             relative[1, 0] - relative[0, 1]],
            dtype=np.float64,
        )
        if angle < 1e-7:
            body_rotvec = 0.5 * skew
        else:
            sine = math.sin(angle)
            if abs(sine) < 1e-7:
                # Near pi: use the dominant eigenvector of R with eigenvalue 1.
                values, vectors = np.linalg.eig(relative)
                axis = np.real(vectors[:, int(np.argmin(np.abs(values - 1.0)))])
                axis /= max(np.linalg.norm(axis), 1e-12)
                body_rotvec = axis * angle
            else:
                body_rotvec = skew * (angle / (2.0 * sine))
        output[index] = previous @ body_rotvec * fps
    return output


def finite_percentiles(values: np.ndarray) -> dict[str, float | None]:
    finite = np.asarray(values)[np.isfinite(values)]
    if finite.size == 0:
        return {key: None for key in ("min", "p05", "median", "p95", "max")}
    percentiles = np.percentile(finite, [0, 5, 50, 95, 100])
    return {
        key: float(value)
        for key, value in zip(("min", "p05", "median", "p95", "max"), percentiles)
    }


def filter_short_true_runs(values: np.ndarray, minimum_run: int) -> np.ndarray:
    result = np.asarray(values, dtype=bool).copy()
    if minimum_run <= 1:
        return result
    for hand_index in range(result.shape[1]):
        start = 0
        while start < len(result):
            if not result[start, hand_index]:
                start += 1
                continue
            end = start + 1
            while end < len(result) and result[end, hand_index]:
                end += 1
            if end - start < minimum_run:
                result[start:end, hand_index] = False
            start = end
    return result


def frame_stems(cam4: Path) -> list[str]:
    sets = []
    for directory in CORE_FRAME_DIRS:
        path = cam4 / directory
        if not path.is_dir():
            raise FileNotFoundError(f"Missing required directory: {path}")
        sets.append({item.stem for item in path.glob("*.txt")})
    common = set.intersection(*sets)
    if not common:
        raise ValueError(f"No complete frames under {cam4}")
    return sorted(common, key=int)


def sequence_directories(root: Path) -> Iterable[Path]:
    for cam4 in sorted(root.glob("subject*/*/*/cam4")):
        if (cam4 / "hand_pose").is_dir():
            yield cam4.parent


def write_inventory(root: Path, output: Path) -> None:
    output.mkdir(parents=True, exist_ok=True)
    rows = []
    total_frames = 0
    for sequence in sequence_directories(root):
        cam4 = sequence / "cam4"
        counts = {
            directory: len(list((cam4 / directory).glob("*.txt")))
            if (cam4 / directory).is_dir()
            else 0
            for directory in ALL_FRAME_DIRS
        }
        complete = min(counts[key] for key in CORE_FRAME_DIRS) if counts else 0
        total_frames += complete
        rows.append(
            {
                "sequence": sequence.relative_to(root).as_posix(),
                "complete_frames": complete,
                **{f"{key}_files": value for key, value in counts.items()},
            }
        )
    fields = ["sequence", "complete_frames"] + [f"{key}_files" for key in ALL_FRAME_DIRS]
    with (output / "inventory.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)
    subjects: dict[str, dict[str, int]] = {}
    for row in rows:
        subject = str(row["sequence"]).split("/")[0]
        entry = subjects.setdefault(subject, {"sequences": 0, "complete_frames": 0})
        entry["sequences"] += 1
        entry["complete_frames"] += int(row["complete_frames"])
    summary = {
        "root": str(root),
        "sequence_count": len(rows),
        "complete_frame_count": total_frames,
        "subjects": subjects,
        "core_frame_directories": list(CORE_FRAME_DIRS),
        "optional_frame_directories": list(OPTIONAL_FRAME_DIRS),
    }
    (output / "inventory_summary.json").write_text(
        json.dumps(summary, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )
    print(json.dumps(summary, indent=2, ensure_ascii=False))


def equalize_axes_3d(axis, arrays: list[np.ndarray]) -> None:
    valid = [array[np.all(np.isfinite(array), axis=1)] for array in arrays if array.size]
    valid = [array for array in valid if len(array)]
    if not valid:
        return
    points = np.concatenate(valid, axis=0)
    low = points.min(axis=0)
    high = points.max(axis=0)
    center = (low + high) / 2.0
    radius = max(float((high - low).max()) / 2.0, 0.05)
    axis.set_xlim(center[0] - radius, center[0] + radius)
    axis.set_ylim(center[1] - radius, center[1] + radius)
    axis.set_zlim(center[2] - radius, center[2] + radius)


def plot_outputs(
    output: Path,
    frames: np.ndarray,
    hand_world: np.ndarray,
    hand_presence: np.ndarray,
    object_centers: np.ndarray,
    camera_centers: np.ndarray,
    contact_distance: np.ndarray,
    contact_candidate: np.ndarray,
    action_label: np.ndarray,
    object_speed: np.ndarray,
    camera_speed: np.ndarray,
    fps: float,
    preview_frame_index: int,
    object_vertices_world: np.ndarray,
) -> None:
    import os

    os.environ.setdefault("MPLCONFIGDIR", "/tmp/matplotlib-h2o-oracle")
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    # Global paths.
    figure = plt.figure(figsize=(10, 8))
    axis = figure.add_subplot(111, projection="3d")
    axis.plot(*camera_centers.T, color="black", linewidth=1.0, label="ego camera")
    axis.plot(*object_centers.T, color="tab:green", linewidth=2.0, label="object center")
    colors = ("tab:red", "tab:blue")
    labels = ("left wrist", "right wrist")
    for hand in range(2):
        wrist = hand_world[:, hand, 0].copy()
        wrist[~hand_presence[:, hand]] = np.nan
        axis.plot(*wrist.T, color=colors[hand], linewidth=1.0, label=labels[hand])
    axis.set_title("H2O oracle trajectories in dataset world coordinates")
    axis.set_xlabel("x [m]")
    axis.set_ylabel("y [m]")
    axis.set_zlabel("z [m]")
    axis.legend(loc="best")
    equalize_axes_3d(axis, [camera_centers, object_centers, hand_world[:, 0, 0], hand_world[:, 1, 0]])
    figure.tight_layout()
    figure.savefig(output / "trajectory_3d.png", dpi=160)
    plt.close(figure)

    # Time series for state auditing.
    time = (frames - frames[0]).astype(np.float64) / fps
    figure, axes = plt.subplots(4, 1, figsize=(13, 10), sharex=True)
    axes[0].step(time, action_label, where="post", color="tab:purple")
    axes[0].set_ylabel("action id")
    for hand, label, color in zip(range(2), ("left", "right"), colors):
        axes[1].plot(time, contact_distance[:, hand] * 100.0, label=label, color=color)
        active = np.flatnonzero(contact_candidate[:, hand])
        axes[1].scatter(time[active], contact_distance[active, hand] * 100.0, s=4, color=color)
    axes[1].set_ylabel("joint→object [cm]")
    axes[1].set_ylim(bottom=0)
    axes[1].legend()
    axes[2].plot(time, object_speed, label="object", color="tab:green")
    axes[2].plot(time, camera_speed, label="camera", color="black", alpha=0.8)
    axes[2].set_ylabel("linear speed [m/s]")
    axes[2].legend()
    axes[3].step(time, hand_presence[:, 0], where="post", label="left", color=colors[0])
    axes[3].step(time, hand_presence[:, 1], where="post", label="right", color=colors[1])
    axes[3].set_ylabel("present")
    axes[3].set_xlabel(f"time [s], assuming {fps:g} fps")
    axes[3].legend()
    figure.suptitle("Oracle state audit timeline")
    figure.tight_layout()
    figure.savefig(output / "state_timeline.png", dpi=160)
    plt.close(figure)

    # A representative world-space frame. Object points are subsampled only for plotting.
    figure = plt.figure(figsize=(9, 8))
    axis = figure.add_subplot(111, projection="3d")
    vertices = object_vertices_world[:: max(1, len(object_vertices_world) // 1500)]
    axis.scatter(*vertices.T, s=1, alpha=0.2, color="tab:green", label="object vertices")
    for hand, label, color in zip(range(2), ("left hand", "right hand"), colors):
        if not hand_presence[preview_frame_index, hand]:
            continue
        joints = hand_world[preview_frame_index, hand]
        axis.scatter(*joints.T, s=16, color=color, label=label)
        for start, end in HAND_EDGES:
            edge = joints[[start, end]]
            axis.plot(*edge.T, color=color, linewidth=1.5)
    camera = camera_centers[preview_frame_index]
    axis.scatter(*camera, marker="^", s=55, color="black", label="ego camera")
    axis.set_title(f"Frame {int(frames[preview_frame_index]):06d} in world coordinates")
    axis.set_xlabel("x [m]")
    axis.set_ylabel("y [m]")
    axis.set_zlabel("z [m]")
    axis.legend(loc="best")
    equalize_axes_3d(
        axis,
        [vertices, hand_world[preview_frame_index, 0], hand_world[preview_frame_index, 1], camera[None]],
    )
    figure.tight_layout()
    figure.savefig(output / f"frame_{int(frames[preview_frame_index]):06d}_world.png", dpi=180)
    plt.close(figure)


def build_state(args: argparse.Namespace) -> None:
    root = args.root.resolve()
    sequence = (root / args.sequence).resolve()
    try:
        sequence.relative_to(root)
    except ValueError as error:
        raise ValueError("Sequence must be inside the H2O root") from error
    cam4 = sequence / "cam4"
    stems = frame_stems(cam4)
    frames = np.asarray([int(stem) for stem in stems], dtype=np.int32)
    count = len(frames)

    output = args.output / args.sequence
    output.mkdir(parents=True, exist_ok=True)

    hand_presence = np.zeros((count, 2), dtype=bool)
    hand_cam = np.full((count, 2, 21, 3), np.nan, dtype=np.float64)
    hand_world = np.full_like(hand_cam, np.nan)
    mano_parameters = np.full((count, 124), np.nan, dtype=np.float64)
    object_id = np.zeros(count, dtype=np.int16)
    object_pose_cam = np.full((count, 4, 4), np.nan, dtype=np.float64)
    object_pose_world = np.full_like(object_pose_cam, np.nan)
    object_center_world = np.full((count, 3), np.nan, dtype=np.float64)
    camera_pose_world = np.full((count, 4, 4), np.nan, dtype=np.float64)
    action_label = np.zeros(count, dtype=np.int16)
    verb_label = np.zeros(count, dtype=np.int16)
    joint_object_vertex_distance = np.full((count, 2, 21), np.nan, dtype=np.float64)
    object_mesh_cache: dict[int, np.ndarray] = {}
    roundtrip_errors = []

    for index, stem in enumerate(stems):
        hand = load_flat(cam4 / "hand_pose" / f"{stem}.txt", 128)
        mano_parameters[index] = load_flat(cam4 / "hand_pose_mano" / f"{stem}.txt", 124)
        hand_presence[index] = hand[[0, 64]] > 0.5
        hand_cam[index, 0] = hand[1:64].reshape(21, 3)
        hand_cam[index, 1] = hand[65:128].reshape(21, 3)

        camera = load_flat(cam4 / "cam_pose" / f"{stem}.txt", 16).reshape(4, 4)
        camera_pose_world[index] = camera
        for hand_index in range(2):
            if hand_presence[index, hand_index]:
                hand_world[index, hand_index] = transform_points(camera, hand_cam[index, hand_index])
                recovered = transform_points(np.linalg.inv(camera), hand_world[index, hand_index])
                roundtrip_errors.append(float(np.max(np.abs(recovered - hand_cam[index, hand_index]))))

        object_values = load_flat(cam4 / "obj_pose_rt" / f"{stem}.txt", 17)
        identifier = int(round(float(object_values[0])))
        object_id[index] = identifier
        if identifier not in OBJECTS:
            raise ValueError(f"Frame {stem}: unsupported object id {identifier}")
        pose_cam = object_values[1:].reshape(4, 4)
        pose_world = camera @ pose_cam
        object_pose_cam[index] = pose_cam
        object_pose_world[index] = pose_world

        if identifier not in object_mesh_cache:
            mesh_path = root / "object" / OBJECTS[identifier][1]
            object_mesh_cache[identifier] = load_obj_vertices(mesh_path)
        local_vertices = object_mesh_cache[identifier]
        object_center_world[index] = transform_points(pose_world, local_vertices.mean(axis=0)[None])[0]

        vertices_cam = transform_points(pose_cam, local_vertices)
        for hand_index in range(2):
            if not hand_presence[index, hand_index]:
                continue
            difference = hand_cam[index, hand_index, :, None, :] - vertices_cam[None, :, :]
            squared = np.einsum("jkq,jkq->jk", difference, difference)
            joint_object_vertex_distance[index, hand_index] = np.sqrt(squared.min(axis=1))

        action_path = cam4 / "action_label" / f"{stem}.txt"
        verb_path = cam4 / "verb_label" / f"{stem}.txt"
        action_label[index] = int(load_flat(action_path, 1)[0]) if action_path.is_file() else -1
        verb_label[index] = int(load_flat(verb_path, 1)[0]) if verb_path.is_file() else -1

    fps = float(args.fps)
    camera_center = camera_pose_world[:, :3, 3]
    camera_linear_velocity = np.zeros_like(camera_center)
    object_linear_velocity = np.zeros_like(object_center_world)
    if count > 1:
        camera_linear_velocity[1:] = np.diff(camera_center, axis=0) * fps
        object_linear_velocity[1:] = np.diff(object_center_world, axis=0) * fps
    camera_angular_velocity = relative_angular_velocity(camera_pose_world[:, :3, :3], fps)
    object_angular_velocity = relative_angular_velocity(object_pose_world[:, :3, :3], fps)

    hand_joint_velocity = np.full_like(hand_world, np.nan)
    for hand_index in range(2):
        for index in range(1, count):
            if hand_presence[index - 1, hand_index] and hand_presence[index, hand_index]:
                hand_joint_velocity[index, hand_index] = (
                    hand_world[index, hand_index] - hand_world[index - 1, hand_index]
                ) * fps

    contact_distance = np.nanmin(joint_object_vertex_distance, axis=2)
    raw_contact = contact_distance <= float(args.contact_threshold_m)
    raw_contact &= hand_presence
    contact_candidate = filter_short_true_runs(raw_contact, int(args.min_contact_frames))
    temperature = max(float(args.contact_temperature_m), 1e-6)
    contact_probability = 1.0 / (
        1.0 + np.exp(np.clip((contact_distance - args.contact_threshold_m) / temperature, -60, 60))
    )
    contact_probability[~hand_presence] = np.nan

    camera_ortho, camera_det, camera_bottom = rotation_errors(camera_pose_world)
    object_ortho, object_det, object_bottom = rotation_errors(object_pose_world)
    bone_lengths = np.full((count, 2, len(HAND_EDGES)), np.nan, dtype=np.float64)
    for hand_index in range(2):
        for edge_index, (start, end) in enumerate(HAND_EDGES):
            valid = hand_presence[:, hand_index]
            bone_lengths[valid, hand_index, edge_index] = np.linalg.norm(
                hand_world[valid, hand_index, end] - hand_world[valid, hand_index, start], axis=1
            )
    bone_median = np.nanmedian(bone_lengths, axis=0)
    relative_bone_error = np.abs(bone_lengths - bone_median[None]) / np.maximum(
        bone_median[None], 1e-9
    )

    np.savez_compressed(
        output / "oracle_state.npz",
        frames=frames,
        timestamps_s=(frames - frames[0]) / fps,
        hand_presence=hand_presence,
        hand_joints_cam4_m=hand_cam,
        hand_joints_world_m=hand_world,
        hand_joint_velocity_world_mps=hand_joint_velocity,
        mano_parameters_cam4=mano_parameters,
        object_id=object_id,
        object_pose_cam4=object_pose_cam,
        object_pose_world=object_pose_world,
        object_center_world_m=object_center_world,
        object_linear_velocity_world_mps=object_linear_velocity,
        object_angular_velocity_world_radps=object_angular_velocity,
        camera_pose_world=camera_pose_world,
        camera_linear_velocity_world_mps=camera_linear_velocity,
        camera_angular_velocity_world_radps=camera_angular_velocity,
        action_label=action_label,
        verb_label=verb_label,
        joint_object_vertex_distance_m=joint_object_vertex_distance,
        contact_distance_m=contact_distance,
        contact_probability=contact_probability,
        contact_candidate=contact_candidate,
        bone_lengths_m=bone_lengths,
    )

    action_counts = {
        str(int(identifier)): {
            "name": ACTIONS[int(identifier)] if 0 <= int(identifier) < len(ACTIONS) else "unavailable",
            "frames": int(np.count_nonzero(action_label == identifier)),
        }
        for identifier in np.unique(action_label)
    }
    contact_by_action = {}
    for identifier in np.unique(action_label):
        selected = action_label == identifier
        action = int(identifier)
        contact_by_action[str(action)] = {
            "name": ACTIONS[action] if 0 <= action < len(ACTIONS) else "unavailable",
            "frames": int(selected.sum()),
            "left_contact_fraction": float(contact_candidate[selected, 0].mean()),
            "right_contact_fraction": float(contact_candidate[selected, 1].mean()),
        }
    object_counts = {
        str(int(identifier)): {
            "name": OBJECTS[int(identifier)][0],
            "frames": int(np.count_nonzero(object_id == identifier)),
        }
        for identifier in np.unique(object_id)
    }
    summary = {
        "sequence": args.sequence,
        "frame_count": count,
        "frame_range": [int(frames[0]), int(frames[-1])],
        "fps_assumed": fps,
        "duration_s": float((frames[-1] - frames[0]) / fps),
        "coordinate_contract": {
            "cam_pose": "cam4-to-world (C2W)",
            "obj_pose_rt": "object-local-to-cam4 (O2C)",
            "object_pose_world": "cam_pose @ obj_pose_rt",
            "hand_pose": "cam4 coordinates, transformed by cam_pose",
            "units": "meters",
            "source": "official H2OPlayer transform behavior; pixel validation awaits RGB archives",
        },
        "objects": object_counts,
        "actions": action_counts,
        "contact_by_action": contact_by_action,
        "hand_presence_fraction": {
            "left": float(hand_presence[:, 0].mean()),
            "right": float(hand_presence[:, 1].mean()),
        },
        "contact_definition": {
            "kind": "candidate proxy: minimum 3D hand-joint to transformed object-mesh-vertex distance",
            "threshold_m": float(args.contact_threshold_m),
            "temperature_m": temperature,
            "minimum_consecutive_frames": int(args.min_contact_frames),
            "limitation": "joint centers and nearest vertices are not signed hand-mesh/object-surface distances",
        },
        "contact_candidate_fraction": {
            "left": float(contact_candidate[:, 0].mean()),
            "right": float(contact_candidate[:, 1].mean()),
        },
        "contact_distance_m": {
            "left": finite_percentiles(contact_distance[:, 0]),
            "right": finite_percentiles(contact_distance[:, 1]),
        },
        "camera_linear_speed_mps": finite_percentiles(np.linalg.norm(camera_linear_velocity, axis=1)),
        "camera_angular_speed_radps": finite_percentiles(np.linalg.norm(camera_angular_velocity, axis=1)),
        "object_linear_speed_mps": finite_percentiles(np.linalg.norm(object_linear_velocity, axis=1)),
        "object_angular_speed_radps": finite_percentiles(np.linalg.norm(object_angular_velocity, axis=1)),
        "numeric_audit": {
            "camera_rotation_orthogonality_max": float(camera_ortho.max()),
            "camera_rotation_determinant_range": [float(camera_det.min()), float(camera_det.max())],
            "camera_bottom_row_error_max": float(camera_bottom.max()),
            "object_rotation_orthogonality_max": float(object_ortho.max()),
            "object_rotation_determinant_range": [float(object_det.min()), float(object_det.max())],
            "object_bottom_row_error_max": float(object_bottom.max()),
            "hand_cam_world_cam_roundtrip_error_m_max": float(max(roundtrip_errors, default=0.0)),
            "hand_bone_relative_error": finite_percentiles(relative_bone_error),
        },
    }
    (output / "summary.json").write_text(
        json.dumps(summary, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )

    with (output / "frame_state.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow(
            [
                "frame", "time_s", "action_id", "action_name", "verb_id", "object_id", "object_name",
                "left_present", "right_present", "left_contact_distance_m", "right_contact_distance_m",
                "left_contact_candidate", "right_contact_candidate", "object_speed_mps", "camera_speed_mps",
            ]
        )
        for index, frame in enumerate(frames):
            action = int(action_label[index])
            identifier = int(object_id[index])
            writer.writerow(
                [
                    int(frame), float((frame - frames[0]) / fps), action,
                    ACTIONS[action] if 0 <= action < len(ACTIONS) else "unavailable", int(verb_label[index]),
                    identifier, OBJECTS[identifier][0], int(hand_presence[index, 0]),
                    int(hand_presence[index, 1]), float(contact_distance[index, 0]),
                    float(contact_distance[index, 1]), int(contact_candidate[index, 0]),
                    int(contact_candidate[index, 1]), float(np.linalg.norm(object_linear_velocity[index])),
                    float(np.linalg.norm(camera_linear_velocity[index])),
                ]
            )

    preview_index = args.preview_frame_index
    if preview_index is None:
        per_frame = np.nanmin(contact_distance, axis=1)
        foreground = np.flatnonzero((action_label > 0) & np.any(contact_candidate, axis=1))
        preview_index = (
            int(foreground[np.nanargmin(per_frame[foreground])])
            if len(foreground)
            else int(np.nanargmin(per_frame))
        )
    if not 0 <= preview_index < count:
        raise ValueError(f"preview-frame-index must be in [0, {count - 1}]")
    plots: list[str] = []
    if not args.skip_plots:
        plot_outputs(
            output,
            frames,
            hand_world,
            hand_presence,
            object_center_world,
            camera_center,
            contact_distance,
            contact_candidate,
            action_label,
            np.linalg.norm(object_linear_velocity, axis=1),
            np.linalg.norm(camera_linear_velocity, axis=1),
            fps,
            preview_index,
            transform_points(
                object_pose_world[preview_index],
                object_mesh_cache[int(object_id[preview_index])],
            ),
        )
        plots = [
            "trajectory_3d.png",
            "state_timeline.png",
            f"frame_{int(frames[preview_index]):06d}_world.png",
        ]

    metadata = {
        "schema_version": 1,
        "state_file": "oracle_state.npz",
        "summary_file": "summary.json",
        "frame_table": "frame_state.csv",
        "plots": plots,
        "preview_frame": int(frames[preview_index]),
    }
    (output / "manifest.json").write_text(
        json.dumps(metadata, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )
    if not args.quiet:
        print(json.dumps(summary, indent=2, ensure_ascii=False))
        print(f"Wrote oracle state to {output}")


def parser() -> argparse.ArgumentParser:
    result = argparse.ArgumentParser(description=__doc__)
    subparsers = result.add_subparsers(dest="command", required=True)

    inventory = subparsers.add_parser("inventory", help="inventory all pose-only sequences")
    inventory.add_argument("--root", type=Path, default=DEFAULT_ROOT)
    inventory.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)

    build = subparsers.add_parser("build", help="build one oracle state sequence")
    build.add_argument("--root", type=Path, default=DEFAULT_ROOT)
    build.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    build.add_argument("--sequence", default="subject1/h1/0")
    build.add_argument("--fps", type=float, default=30.0)
    build.add_argument("--contact-threshold-m", type=float, default=0.02)
    build.add_argument("--contact-temperature-m", type=float, default=0.005)
    build.add_argument("--min-contact-frames", type=int, default=3)
    build.add_argument("--preview-frame-index", type=int)
    build.add_argument("--skip-plots", action="store_true")
    build.add_argument("--quiet", action="store_true")
    return result


def main() -> None:
    args = parser().parse_args()
    if args.command == "inventory":
        write_inventory(args.root.resolve(), args.output.resolve())
    else:
        build_state(args)


if __name__ == "__main__":
    main()
