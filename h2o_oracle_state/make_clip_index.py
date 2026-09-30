#!/usr/bin/env python3
"""Create fixed-length physical clip indices from built H2O oracle states."""

from __future__ import annotations

import argparse
import csv
import json
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np

from h2o_oracle_state import ACTIONS, OBJECTS


DEFAULT_STATE_ROOT = Path("datasets/H2O/oracle_state")


def split_for_subject(subject: str) -> str:
    return {"subject1": "train", "subject2": "train", "subject3": "val", "subject4": "test"}.get(
        subject, "unknown"
    )


def dominant(values: np.ndarray) -> int:
    counts = Counter(int(value) for value in values)
    return counts.most_common(1)[0][0]


def make_index(args: argparse.Namespace) -> None:
    root = args.state_root.resolve()
    files = sorted(root.glob("subject*/*/*/oracle_state.npz"))
    if not files:
        raise FileNotFoundError(f"No oracle_state.npz files under {root}")
    rows = []
    skipped_nonfinite = 0
    skipped_discontinuous = 0
    sequence_counts: Counter[str] = Counter()
    split_counts: Counter[str] = Counter()
    object_counts: Counter[str] = Counter()
    action_counts: Counter[str] = Counter()
    for state_path in files:
        state = np.load(state_path)
        frames = state["frames"]
        sequence = state_path.parent.relative_to(root).as_posix()
        subject = sequence.split("/")[0]
        split = split_for_subject(subject)
        sequence_counts[sequence] += 1
        for start in range(0, len(frames) - args.length + 1, args.stride):
            stop = start + args.length
            window_frames = frames[start:stop]
            if not np.all(np.diff(window_frames) == 1):
                skipped_discontinuous += 1
                continue
            object_pose = state["object_pose_world"][start:stop]
            hand_world = state["hand_joints_world_m"][start:stop]
            presence = state["hand_presence"][start:stop]
            finite_hands = np.isfinite(hand_world[presence]).all() if np.any(presence) else True
            if not np.isfinite(object_pose).all() or not finite_hands:
                skipped_nonfinite += 1
                continue
            actions = state["action_label"][start:stop]
            action_id = dominant(actions)
            object_id = dominant(state["object_id"][start:stop])
            contact = state["contact_candidate"][start:stop]
            contact_change = np.any(contact[1:] != contact[:-1], axis=0)
            bone = state["bone_lengths_m"][start:stop]
            bone_median = np.nanmedian(bone, axis=0)
            relative_bone = np.abs(bone - bone_median[None]) / np.maximum(bone_median[None], 1e-9)
            bone_p95 = float(np.nanpercentile(relative_bone, 95))
            quality_ok = bone_p95 <= args.max_bone_relative_p95
            identifier = f"{sequence.replace('/', '_')}_{int(window_frames[0]):06d}_{int(window_frames[-1]):06d}"
            action_name = ACTIONS[action_id] if 0 <= action_id < len(ACTIONS) else "unavailable"
            object_name = OBJECTS[object_id][0]
            row = {
                "clip_id": identifier,
                "split": split,
                "sequence": sequence,
                "state_path": str(state_path),
                "start_index": start,
                "stop_index_exclusive": stop,
                "start_frame": int(window_frames[0]),
                "end_frame": int(window_frames[-1]),
                "length": args.length,
                "object_id": object_id,
                "object_name": object_name,
                "dominant_action_id": action_id,
                "dominant_action_name": action_name,
                "unique_action_ids": ";".join(str(int(value)) for value in np.unique(actions)),
                "labels_available": int(action_id >= 0),
                "left_presence_fraction": float(presence[:, 0].mean()),
                "right_presence_fraction": float(presence[:, 1].mean()),
                "left_contact_fraction": float(contact[:, 0].mean()),
                "right_contact_fraction": float(contact[:, 1].mean()),
                "left_contact_transition": int(contact_change[0]),
                "right_contact_transition": int(contact_change[1]),
                "object_speed_p95_mps": float(
                    np.percentile(np.linalg.norm(state["object_linear_velocity_world_mps"][start:stop], axis=1), 95)
                ),
                "camera_speed_p95_mps": float(
                    np.percentile(np.linalg.norm(state["camera_linear_velocity_world_mps"][start:stop], axis=1), 95)
                ),
                "bone_relative_error_p95": bone_p95,
                "quality_ok": int(quality_ok),
            }
            rows.append(row)
            split_counts[split] += 1
            object_counts[object_name] += 1
            action_counts[action_name] += 1

    output = args.output or root / f"physical_clips_{args.length}f_stride{args.stride}.csv"
    output.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        raise RuntimeError("No clips passed structural checks")
    with output.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    summary = {
        "state_root": str(root),
        "indexed_state_files": len(files),
        "clip_length": args.length,
        "stride": args.stride,
        "clip_count": len(rows),
        "quality_ok_count": sum(int(row["quality_ok"]) for row in rows),
        "skipped_nonfinite": skipped_nonfinite,
        "skipped_discontinuous": skipped_discontinuous,
        "split_counts": dict(split_counts),
        "object_counts": dict(object_counts),
        "dominant_action_counts": dict(action_counts),
        "max_bone_relative_p95": args.max_bone_relative_p95,
        "csv": str(output),
    }
    summary_path = output.with_suffix(".summary.json")
    summary_path.write_text(json.dumps(summary, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print(json.dumps(summary, indent=2, ensure_ascii=False))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--state-root", type=Path, default=DEFAULT_STATE_ROOT)
    parser.add_argument("--length", type=int, default=64)
    parser.add_argument("--stride", type=int, default=32)
    parser.add_argument("--max-bone-relative-p95", type=float, default=0.05)
    parser.add_argument("--output", type=Path)
    return parser.parse_args()


if __name__ == "__main__":
    make_index(parse_args())
