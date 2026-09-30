#!/usr/bin/env python3
"""Export evaluation-only hand and visible-object masks from H2O annotations."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
import sys

import numpy as np
from PIL import Image

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from h2o_geometric_baseline.reprojection import load_intrinsics, load_pose
from h2o_physics_baseline.annotated_object_layer import (
    load_geometry,
    rasterize_object_support,
)
from h2o_physics_baseline.audit_kinematic_hand_mesh_holes import (
    load_oracle_joints,
    project_mesh_mask,
)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model-input-root", type=Path, required=True)
    parser.add_argument(
        "--index", type=Path,
        default=Path("datasets/H2O/oracle_state/paired_physical_clips.csv"),
    )
    parser.add_argument(
        "--state-root", type=Path, default=Path("datasets/H2O/oracle_state")
    )
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--object-depth-tolerance-m", type=float, default=0.025)
    args = parser.parse_args()

    manifest = json.loads((args.model_input_root / "manifest.json").read_text())
    with args.index.open(encoding="utf-8") as handle:
        rows = [row for row in csv.DictReader(handle) if row["pair_id"] == manifest["pair_id"]]
    if len(rows) != 1:
        raise ValueError(f"Expected one index row, found {len(rows)}")
    row = rows[0]
    sequence_root = Path(row["target_rgb_dir"]).parent.parent
    target_root = sequence_root / "cam4"
    size = int(manifest["image_size"])
    intrinsics = load_intrinsics(target_root / "cam_intrinsics.txt")
    state_path = args.state_root / row["sequence"] / "oracle_state.npz"
    with np.load(state_path) as state:
        state_frames = state["frames"].copy()
        object_ids = state["object_id"].copy()
        object_poses = state["object_pose_world"].copy()

    hand_root = args.output_root / "hand"
    object_root = args.output_root / "object"
    hand_root.mkdir(parents=True, exist_ok=True)
    object_root.mkdir(parents=True, exist_ok=True)
    hand_fractions, object_fractions = [], []
    for record in manifest["frames"]:
        frame = int(record["dataset_frame"])
        joints, valid = load_oracle_joints(sequence_root, frame)
        hand = project_mesh_mask(joints, valid, intrinsics, size)

        state_index = int(np.searchsorted(state_frames, frame))
        if state_index >= len(state_frames) or int(state_frames[state_index]) != frame:
            raise IndexError(f"Oracle state missing frame {frame}")
        object_id = int(object_ids[state_index])
        object_mask = np.zeros((size, size), dtype=bool)
        if object_id > 0:
            geometry = load_geometry(str(args.state_root.resolve()), object_id)
            camera_pose = load_pose(target_root / "cam_pose" / f"{frame:06d}.txt")
            support, rendered_depth = rasterize_object_support(
                geometry, object_poses[state_index], intrinsics, camera_pose,
                (size, size),
            )
            observed_depth = np.asarray(
                Image.open(target_root / "depth" / f"{frame:06d}.png").resize(
                    (size, size), Image.Resampling.NEAREST
                ), dtype=np.float32,
            ) / 1000.0
            object_mask = (
                support & np.isfinite(rendered_depth) & (observed_depth > 0)
                & (np.abs(observed_depth - rendered_depth)
                   <= args.object_depth_tolerance_m)
            )
        filename = f"{frame:06d}.png"
        Image.fromarray(hand.astype(np.uint8) * 255, "L").save(hand_root / filename)
        Image.fromarray(object_mask.astype(np.uint8) * 255, "L").save(object_root / filename)
        hand_fractions.append(float(hand.mean()))
        object_fractions.append(float(object_mask.mean()))

    summary = {
        "pair_id": manifest["pair_id"],
        "evaluation_only": True,
        "hand_reference": "mesh proxy projected from annotated cam4 hand joints",
        "object_reference": (
            "annotated CAD silhouette filtered by cam4 observed-depth agreement"
        ),
        "object_depth_tolerance_m": args.object_depth_tolerance_m,
        "mean_hand_fraction": float(np.mean(hand_fractions)),
        "mean_visible_object_fraction": float(np.mean(object_fractions)),
    }
    (args.output_root / "summary.json").write_text(
        json.dumps(summary, indent=2) + "\n"
    )
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
