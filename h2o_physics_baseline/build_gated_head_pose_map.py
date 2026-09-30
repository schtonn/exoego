#!/usr/bin/env python3
"""Rejected diagnostic for the deprecated white-color helmet ICP ablation."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from h2o_geometric_baseline.reprojection import load_pose
from h2o_physics_baseline.render_causal_video_background_split import smoothed_pose_map


def angle_deg(rotation: np.ndarray) -> float:
    return float(np.degrees(np.arccos(np.clip((np.trace(rotation) - 1.0) / 2.0, -1.0, 1.0))))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--sequence", required=True)
    parser.add_argument("--start-frame", type=int, required=True)
    parser.add_argument("--end-frame", type=int, required=True)
    parser.add_argument("--pair-id", required=True)
    parser.add_argument("--face-summary", type=Path, required=True)
    parser.add_argument("--helmet-summary", type=Path, required=True)
    parser.add_argument("--motion-threshold-deg", type=float, default=2.0)
    parser.add_argument("--helmet-translation-weight", type=float, default=0.5)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    target = Path("datasets/H2O/raw") / args.sequence / "cam4"
    frames = list(range(args.start_frame, args.end_frame + 1))
    initial = load_pose(target / "cam_pose" / f"{frames[0]:06d}.txt")
    face_data = json.loads(args.face_summary.read_text())
    face_record = next(value for value in face_data["per_clip"] if value["pair_id"] == args.pair_id)
    face_map = smoothed_pose_map(frames, initial, face_record)
    helmet_data = json.loads(args.helmet_summary.read_text())
    helmet = [np.asarray(value, dtype=np.float64) for value in helmet_data["settings"][0]["poses"]]
    motion = [angle_deg(np.asarray(value["estimated_delta_rotation_world"])) for value in face_record["future"]]
    motion_mean = float(np.mean(motion))
    use_helmet = motion_mean > args.motion_threshold_deg

    output_poses = []
    for index, frame in enumerate(frames):
        face = face_map[frame]
        if use_helmet:
            pose = helmet[index].copy()
            pose[:3, 3] = (
                args.helmet_translation_weight * helmet[index][:3, 3]
                + (1.0 - args.helmet_translation_weight) * face[:3, 3]
            )
        else:
            pose = face.copy()
        output_poses.append(pose)

    truth = [load_pose(target / "cam_pose" / f"{frame:06d}.txt") for frame in frames]
    translation = [np.linalg.norm(a[:3, 3] - b[:3, 3]) for a, b in zip(output_poses, truth)]
    rotation = [angle_deg(a[:3, :3].T @ b[:3, :3]) for a, b in zip(output_poses, truth)]
    result = {
        "protocol": "exo-only clip gate: helmet rotation for mean face motion > threshold",
        "pair_id": args.pair_id,
        "frames": frames,
        "gate": {
            "mean_face_motion_deg": motion_mean,
            "threshold_deg": args.motion_threshold_deg,
            "used_helmet": use_helmet,
            "helmet_translation_weight": args.helmet_translation_weight,
        },
        "evaluation_only": {
            "translation_mean_m": float(np.mean(translation)),
            "translation_p90_m": float(np.percentile(translation, 90)),
            "rotation_mean_deg": float(np.mean(rotation)),
            "rotation_p90_deg": float(np.percentile(rotation, 90)),
        },
        "poses": [pose.tolist() for pose in output_poses],
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps({key: value for key, value in result.items() if key != "poses"}, indent=2))


if __name__ == "__main__":
    main()
