#!/usr/bin/env python3
"""Build a minimal surface-contact proxy and slip metric for H2O oracle states."""

from __future__ import annotations

import argparse
import csv
import json
from collections import Counter
from pathlib import Path

import numpy as np
from scipy.spatial import cKDTree

from h2o_oracle_state import ACTIONS, OBJECTS


DEFAULT_STATE_ROOT = Path("datasets/H2O/oracle_state")
DEFAULT_GEOMETRY_ROOT = DEFAULT_STATE_ROOT / "object_geometry"


def split_for_subject(subject: str) -> str:
    return {"subject1": "train", "subject2": "train", "subject3": "val", "subject4": "test"}.get(
        subject, "unknown"
    )


def true_runs(values: np.ndarray):
    start = 0
    while start < len(values):
        if not values[start]:
            start += 1
            continue
        end = start + 1
        while end < len(values) and values[end]:
            end += 1
        yield start, end - 1
        start = end


def joint_contacts_with_hysteresis(
    distances: np.ndarray,
    presence: np.ndarray,
    enter_threshold_m: float,
    exit_threshold_m: float,
    minimum_run: int,
) -> np.ndarray:
    """Return stable per-joint contact using two thresholds and a run filter."""
    result = np.zeros(distances.shape, dtype=bool)
    for hand in range(distances.shape[1]):
        for joint in range(distances.shape[2]):
            active = False
            for frame in range(distances.shape[0]):
                distance = distances[frame, hand, joint]
                if not presence[frame, hand] or not np.isfinite(distance):
                    active = False
                elif active:
                    active = distance <= exit_threshold_m
                else:
                    active = distance <= enter_threshold_m
                result[frame, hand, joint] = active
            for start, end in list(true_runs(result[:, hand, joint])):
                if end - start + 1 < minimum_run:
                    result[start : end + 1, hand, joint] = False
    return result


def dominant(values: np.ndarray) -> int:
    return Counter(int(value) for value in values).most_common(1)[0][0]


def finite_summary(values: np.ndarray) -> dict[str, float | None]:
    finite = np.asarray(values)[np.isfinite(values)]
    if not len(finite):
        return {key: None for key in ("median", "p95", "max")}
    return {
        "median": float(np.median(finite)),
        "p95": float(np.percentile(finite, 95)),
        "max": float(finite.max()),
    }


def prepare(args: argparse.Namespace) -> None:
    state_root = args.state_root.resolve()
    geometry_root = args.geometry_root.resolve()
    state_paths = sorted(state_root.glob("subject*/*/*/oracle_state.npz"))
    if not state_paths:
        raise FileNotFoundError(f"No oracle states under {state_root}")

    geometry_cache: dict[int, tuple[np.ndarray, np.ndarray, cKDTree]] = {}
    event_rows: list[dict] = []
    all_surface_distances: list[np.ndarray] = []
    all_vertex_distances: list[np.ndarray] = []
    all_contact_slip: list[np.ndarray] = []
    all_contact_joint_counts: list[np.ndarray] = []
    changed_frames = 0
    compared_frames = 0

    for state_path in state_paths:
        sequence = state_path.parent.relative_to(state_root).as_posix()
        subject = sequence.split("/")[0]
        split = split_for_subject(subject)
        with np.load(state_path) as state:
            frames = state["frames"]
            frame_count = len(frames)
            object_id = int(state["object_id"][0])
            object_name = OBJECTS[object_id][0]
            if object_id not in geometry_cache:
                geometry_path = geometry_root / f"{object_name}_surface_{args.surface_samples}.npz"
                if not geometry_path.is_file():
                    raise FileNotFoundError(f"Build object geometry first: {geometry_path}")
                geometry = np.load(geometry_path)
                surface_points = geometry["points_object_m"].astype(np.float64)
                surface_normals = geometry["normals_object"].astype(np.float64)
                geometry_cache[object_id] = (surface_points, surface_normals, cKDTree(surface_points))
            surface_points, surface_normals, tree = geometry_cache[object_id]

            object_pose = state["object_pose_world"]
            rotation = object_pose[:, :3, :3]
            translation = object_pose[:, :3, 3]
            hand_world = state["hand_joints_world_m"]
            presence = state["hand_presence"]
            local_hand = np.einsum(
                "fji,fhkj->fhki", rotation, hand_world - translation[:, None, None, :]
            )
            flat_hand = local_hand.reshape(-1, 3)
            valid_flat = np.all(np.isfinite(flat_hand), axis=1)
            joint_distance_flat = np.full(len(flat_hand), np.nan, dtype=np.float64)
            nearest_index_flat = np.full(len(flat_hand), -1, dtype=np.int32)
            queried_distance, queried_index = tree.query(flat_hand[valid_flat], workers=-1)
            joint_distance_flat[valid_flat] = queried_distance
            nearest_index_flat[valid_flat] = queried_index.astype(np.int32)
            joint_distance = joint_distance_flat.reshape(frame_count, 2, 21)
            nearest_index = nearest_index_flat.reshape(frame_count, 2, 21)

            sortable_distance = np.where(np.isfinite(joint_distance), joint_distance, np.inf)
            closest_joint = np.argmin(sortable_distance, axis=2).astype(np.int16)
            frame_index = np.arange(frame_count)[:, None]
            hand_index = np.arange(2)[None, :]
            surface_distance = sortable_distance[frame_index, hand_index, closest_joint]
            closest_surface = nearest_index[frame_index, hand_index, closest_joint]
            valid_hand = presence & np.isfinite(surface_distance) & (closest_surface >= 0)
            surface_distance[~valid_hand] = np.nan
            closest_joint[~valid_hand] = -1
            closest_surface[~valid_hand] = -1

            safe_nearest_index = np.maximum(nearest_index, 0)
            local_surface_point = surface_points[safe_nearest_index]
            local_surface_normal = surface_normals[safe_nearest_index]
            world_surface_point = (
                np.einsum("fij,fhkj->fhki", rotation, local_surface_point)
                + translation[:, None, None, :]
            )
            world_surface_normal = np.einsum("fij,fhkj->fhki", rotation, local_surface_normal)

            hand_velocity_all = state["hand_joint_velocity_world_mps"].copy()
            # The canonical state uses backward differences and therefore leaves
            # the first visible frame undefined.  A forward difference supplies
            # that boundary value without adding a smoother or another model.
            for current_hand in range(2):
                forward_valid = presence[:-1, current_hand] & presence[1:, current_hand]
                missing_now = ~np.isfinite(hand_velocity_all[:-1, current_hand]).all(axis=(1, 2))
                fill = forward_valid & missing_now
                hand_velocity_all[:-1, current_hand][fill] = (
                    hand_world[1:, current_hand][fill] - hand_world[:-1, current_hand][fill]
                ) * args.fps
            object_center = state["object_center_world_m"]
            surface_radius = world_surface_point - object_center[:, None, None, :]
            surface_velocity = state["object_linear_velocity_world_mps"][:, None, None, :] + np.cross(
                state["object_angular_velocity_world_radps"][:, None, None, :], surface_radius
            )
            relative_velocity = hand_velocity_all - surface_velocity
            signed_normal_velocity = np.sum(relative_velocity * world_surface_normal, axis=3)
            tangential_velocity = (
                relative_velocity - signed_normal_velocity[:, :, :, None] * world_surface_normal
            )
            joint_normal_speed_abs = np.abs(signed_normal_velocity)
            joint_tangential_speed = np.linalg.norm(tangential_velocity, axis=3)
            valid_joint = np.isfinite(joint_distance) & presence[:, :, None]
            joint_normal_speed_abs[~valid_joint] = np.nan
            joint_tangential_speed[~valid_joint] = np.nan

            joint_contact = joint_contacts_with_hysteresis(
                joint_distance,
                presence,
                args.contact_threshold_m,
                args.contact_exit_threshold_m,
                args.min_contact_frames,
            )
            contact = np.any(joint_contact, axis=2)

            def active_joint_mean(values: np.ndarray) -> np.ndarray:
                selected_values = np.where(joint_contact & np.isfinite(values), values, 0.0)
                counts = np.sum(joint_contact & np.isfinite(values), axis=2)
                result = np.full(counts.shape, np.nan, dtype=np.float64)
                np.divide(selected_values.sum(axis=2), counts, out=result, where=counts > 0)
                return result

            normal_speed_abs = active_joint_mean(joint_normal_speed_abs)
            tangential_speed = active_joint_mean(joint_tangential_speed)
            old_contact = state["contact_candidate"]
            compared_frames += int(valid_hand.sum())
            changed_frames += int(np.count_nonzero((contact != old_contact) & presence))
            all_surface_distances.append(surface_distance[valid_hand])
            old_distance = state["contact_distance_m"]
            all_vertex_distances.append(old_distance[presence])
            all_contact_slip.append(tangential_speed[contact & np.isfinite(tangential_speed)])
            all_contact_joint_counts.append(joint_contact.sum(axis=2)[contact])

            output_path = state_path.parent / "surface_contact.npz"
            np.savez_compressed(
                output_path,
                frames=frames,
                surface_distance_m=surface_distance.astype(np.float32),
                joint_surface_distance_m=joint_distance.astype(np.float32),
                joint_contact_candidate=joint_contact,
                nearest_surface_sample_index_per_joint=nearest_index,
                closest_hand_joint_index=closest_joint,
                closest_surface_sample_index=closest_surface,
                normal_relative_speed_abs_mps=normal_speed_abs.astype(np.float32),
                tangential_slip_speed_mps=tangential_speed.astype(np.float32),
                contact_candidate=contact,
            )

            actions = state["action_label"]
            for current_hand, hand_name in enumerate(("left", "right")):
                for start, end in true_runs(contact[:, current_hand]):
                    section = slice(start, end + 1)
                    action_id = dominant(actions[section])
                    action_name = ACTIONS[action_id] if 0 <= action_id < len(ACTIONS) else "unavailable"
                    slip = tangential_speed[section, current_hand]
                    event_rows.append(
                        {
                            "event_id": f"{sequence.replace('/', '_')}_{hand_name}_{int(frames[start]):06d}_{int(frames[end]):06d}",
                            "split": split,
                            "sequence": sequence,
                            "surface_contact_path": str(output_path),
                            "hand": hand_name,
                            "start_index": start,
                            "stop_index_exclusive": end + 1,
                            "start_frame": int(frames[start]),
                            "end_frame": int(frames[end]),
                            "length": end - start + 1,
                            "object_id": object_id,
                            "object_name": object_name,
                            "dominant_action_id": action_id,
                            "dominant_action_name": action_name,
                            "minimum_surface_distance_m": float(np.nanmin(surface_distance[section, current_hand])),
                            "median_surface_distance_m": float(np.nanmedian(surface_distance[section, current_hand])),
                            "median_tangential_slip_mps": float(np.nanmedian(slip)),
                            "p95_tangential_slip_mps": float(np.nanpercentile(slip, 95)),
                        }
                    )

    event_path = state_root / "surface_contact_events.csv"
    with event_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(event_rows[0]))
        writer.writeheader()
        writer.writerows(event_rows)

    surface_values = np.concatenate(all_surface_distances)
    vertex_values = np.concatenate(all_vertex_distances)
    slip_values = np.concatenate(all_contact_slip) if any(len(values) for values in all_contact_slip) else np.array([])
    contact_joint_counts = np.concatenate(all_contact_joint_counts)
    summary = {
        "state_file_count": len(state_paths),
        "surface_samples_per_object": args.surface_samples,
        "distance_semantics": "unsigned nearest sampled object-surface distance from 21 hand joints",
        "contact_enter_threshold_m": args.contact_threshold_m,
        "contact_exit_threshold_m": args.contact_exit_threshold_m,
        "minimum_consecutive_frames": args.min_contact_frames,
        "slip_semantics": "mean tangential speed of active contact joints relative to the rigid object surface",
        "contact_event_count": len(event_rows),
        "contact_event_counts": dict(Counter(row["hand"] for row in event_rows)),
        "candidate_change_fraction_vs_vertex_proxy": changed_frames / max(compared_frames, 1),
        "surface_distance_m": finite_summary(surface_values),
        "previous_vertex_distance_m": finite_summary(vertex_values),
        "contact_tangential_slip_mps": finite_summary(slip_values),
        "active_contact_joint_count": finite_summary(contact_joint_counts),
        "events_csv": str(event_path),
    }
    summary_path = state_root / "surface_contact.summary.json"
    summary_path.write_text(json.dumps(summary, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print(json.dumps(summary, indent=2, ensure_ascii=False))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--state-root", type=Path, default=DEFAULT_STATE_ROOT)
    parser.add_argument("--geometry-root", type=Path, default=DEFAULT_GEOMETRY_ROOT)
    parser.add_argument("--surface-samples", type=int, default=20_000)
    parser.add_argument("--contact-threshold-m", type=float, default=0.02)
    parser.add_argument("--contact-exit-threshold-m", type=float, default=0.025)
    parser.add_argument("--min-contact-frames", type=int, default=3)
    parser.add_argument("--fps", type=float, default=30.0)
    return parser.parse_args()


if __name__ == "__main__":
    prepare(parse_args())
