#!/usr/bin/env python3
"""Build event-aligned indices and train-only normalization statistics for H2O."""

from __future__ import annotations

import argparse
import csv
import json
from collections import Counter
from pathlib import Path

import numpy as np

from h2o_oracle_state import ACTIONS, OBJECTS


DEFAULT_STATE_ROOT = Path("datasets/H2O/oracle_state")


def split_for_subject(subject: str) -> str:
    return {"subject1": "train", "subject2": "train", "subject3": "val", "subject4": "test"}.get(
        subject, "unknown"
    )


def runs(values: np.ndarray):
    """Yield inclusive constant-value runs as (start, end, value)."""
    if not len(values):
        return
    start = 0
    for index in range(1, len(values)):
        if values[index] != values[start]:
            yield start, index - 1, values[start]
            start = index
    yield start, len(values) - 1, values[start]


def dominant(values: np.ndarray) -> int:
    return Counter(int(value) for value in values).most_common(1)[0][0]


def describe_components(chunks: list[np.ndarray], labels: list[str]) -> dict:
    values = np.concatenate(chunks, axis=0).reshape(-1, len(labels)).astype(np.float64)
    result = {"components": labels, "count": [], "mean": [], "std": [], "min": [], "p01": [], "p50": [], "p99": [], "max": []}
    for component in range(values.shape[1]):
        column = values[:, component]
        column = column[np.isfinite(column)]
        result["count"].append(int(len(column)))
        if not len(column):
            for key in ("mean", "std", "min", "p01", "p50", "p99", "max"):
                result[key].append(None)
            continue
        result["mean"].append(float(column.mean()))
        result["std"].append(float(column.std()))
        result["min"].append(float(column.min()))
        result["p01"].append(float(np.percentile(column, 1)))
        result["p50"].append(float(np.percentile(column, 50)))
        result["p99"].append(float(np.percentile(column, 99)))
        result["max"].append(float(column.max()))
    return result


def prepare(args: argparse.Namespace) -> None:
    root = args.state_root.resolve()
    state_paths = sorted(root.glob("subject*/*/*/oracle_state.npz"))
    if not state_paths:
        raise FileNotFoundError(f"No oracle_state.npz under {root}")

    action_rows: list[dict] = []
    contact_rows: list[dict] = []
    train_features: dict[str, list[np.ndarray]] = {
        "hand_joint_cam4_m": [],
        "hand_minus_object_cam4_m": [],
        "object_translation_cam4_m": [],
        "object_rotation_6d_cam4": [],
        "object_linear_velocity_cam4_mps": [],
        "object_angular_velocity_cam4_radps": [],
        "camera_linear_velocity_local_mps": [],
        "camera_angular_velocity_local_radps": [],
        "contact_distance_m": [],
    }

    for state_path in state_paths:
        sequence = state_path.parent.relative_to(root).as_posix()
        subject = sequence.split("/")[0]
        split = split_for_subject(subject)
        with np.load(state_path) as state:
            frames = state["frames"]
            actions = state["action_label"]
            object_id = int(state["object_id"][0])
            object_name = OBJECTS[object_id][0]
            contact = state["contact_candidate"]
            distances = state["contact_distance_m"]
            object_speed = np.linalg.norm(state["object_linear_velocity_world_mps"], axis=1)
            camera_speed = np.linalg.norm(state["camera_linear_velocity_world_mps"], axis=1)

            for start, end, raw_action_id in runs(actions):
                action_id = int(raw_action_id)
                action_name = ACTIONS[action_id] if 0 <= action_id < len(ACTIONS) else "unavailable"
                section = slice(start, end + 1)
                action_rows.append(
                    {
                        "segment_id": f"{sequence.replace('/', '_')}_action_{int(frames[start]):06d}_{int(frames[end]):06d}",
                        "split": split,
                        "sequence": sequence,
                        "state_path": str(state_path),
                        "start_index": start,
                        "stop_index_exclusive": end + 1,
                        "start_frame": int(frames[start]),
                        "end_frame": int(frames[end]),
                        "length": end - start + 1,
                        "object_id": object_id,
                        "object_name": object_name,
                        "action_id": action_id,
                        "action_name": action_name,
                        "labels_available": int(action_id >= 0),
                        "left_contact_fraction": float(contact[section, 0].mean()),
                        "right_contact_fraction": float(contact[section, 1].mean()),
                        "object_speed_p95_mps": float(np.percentile(object_speed[section], 95)),
                        "camera_speed_p95_mps": float(np.percentile(camera_speed[section], 95)),
                    }
                )

            for hand_index, hand_name in enumerate(("left", "right")):
                for start, end, active in runs(contact[:, hand_index]):
                    if not bool(active):
                        continue
                    section = slice(start, end + 1)
                    action_id = dominant(actions[section])
                    action_name = ACTIONS[action_id] if 0 <= action_id < len(ACTIONS) else "unavailable"
                    contact_rows.append(
                        {
                            "event_id": f"{sequence.replace('/', '_')}_{hand_name}_{int(frames[start]):06d}_{int(frames[end]):06d}",
                            "split": split,
                            "sequence": sequence,
                            "state_path": str(state_path),
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
                            "minimum_distance_m": float(np.nanmin(distances[section, hand_index])),
                            "median_distance_m": float(np.nanmedian(distances[section, hand_index])),
                            "object_speed_p95_mps": float(np.percentile(object_speed[section], 95)),
                        }
                    )

            if split == "train":
                hand = state["hand_joints_cam4_m"]
                presence = state["hand_presence"]
                hand_valid = hand[presence].reshape(-1, 3)
                object_pose_cam = state["object_pose_cam4"]
                object_translation = object_pose_cam[:, :3, 3]
                hand_relative = (hand - object_translation[:, None, None, :])[presence].reshape(-1, 3)
                camera_rotation = state["camera_pose_world"][:, :3, :3]

                def world_to_camera(vectors: np.ndarray) -> np.ndarray:
                    return np.einsum("fji,fj->fi", camera_rotation, vectors)

                train_features["hand_joint_cam4_m"].append(hand_valid)
                train_features["hand_minus_object_cam4_m"].append(hand_relative)
                train_features["object_translation_cam4_m"].append(object_translation)
                train_features["object_rotation_6d_cam4"].append(object_pose_cam[:, :3, :2].reshape(-1, 6))
                train_features["object_linear_velocity_cam4_mps"].append(
                    world_to_camera(state["object_linear_velocity_world_mps"])
                )
                train_features["object_angular_velocity_cam4_radps"].append(
                    world_to_camera(state["object_angular_velocity_world_radps"])
                )
                train_features["camera_linear_velocity_local_mps"].append(
                    world_to_camera(state["camera_linear_velocity_world_mps"])
                )
                train_features["camera_angular_velocity_local_radps"].append(
                    world_to_camera(state["camera_angular_velocity_world_radps"])
                )
                train_features["contact_distance_m"].append(distances)

    action_path = args.action_output or root / "action_segments.csv"
    contact_path = args.contact_output or root / "contact_events.csv"
    for path, rows in ((action_path, action_rows), (contact_path, contact_rows)):
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("w", newline="", encoding="utf-8") as handle:
            writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
            writer.writeheader()
            writer.writerows(rows)

    labels = {
        "hand_joint_cam4_m": ["x", "y", "z"],
        "hand_minus_object_cam4_m": ["x", "y", "z"],
        "object_translation_cam4_m": ["x", "y", "z"],
        "object_rotation_6d_cam4": ["r00", "r01", "r10", "r11", "r20", "r21"],
        "object_linear_velocity_cam4_mps": ["x", "y", "z"],
        "object_angular_velocity_cam4_radps": ["x", "y", "z"],
        "camera_linear_velocity_local_mps": ["x", "y", "z"],
        "camera_angular_velocity_local_radps": ["x", "y", "z"],
        "contact_distance_m": ["left", "right"],
    }
    feature_stats = {
        "scope": "subject1+subject2 train split only; no val/test leakage",
        "coordinate_contract": "camera-frame features; world velocities rotated by R_world_cam^T",
        "features": {
            name: describe_components(chunks, labels[name]) for name, chunks in train_features.items()
        },
    }
    stats_path = args.stats_output or root / "train_feature_stats.json"
    stats_path.write_text(json.dumps(feature_stats, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")

    summary = {
        "indexed_state_files": len(state_paths),
        "action_segment_count": len(action_rows),
        "labeled_action_segment_count": sum(row["labels_available"] for row in action_rows),
        "contact_event_count": len(contact_rows),
        "contact_event_counts": dict(Counter(row["hand"] for row in contact_rows)),
        "split_action_segment_counts": dict(Counter(row["split"] for row in action_rows)),
        "split_contact_event_counts": dict(Counter(row["split"] for row in contact_rows)),
        "action_segments_csv": str(action_path),
        "contact_events_csv": str(contact_path),
        "train_feature_stats_json": str(stats_path),
    }
    summary_path = root / "training_metadata.summary.json"
    summary_path.write_text(json.dumps(summary, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print(json.dumps(summary, indent=2, ensure_ascii=False))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--state-root", type=Path, default=DEFAULT_STATE_ROOT)
    parser.add_argument("--action-output", type=Path)
    parser.add_argument("--contact-output", type=Path)
    parser.add_argument("--stats-output", type=Path)
    return parser.parse_args()


if __name__ == "__main__":
    prepare(parse_args())
