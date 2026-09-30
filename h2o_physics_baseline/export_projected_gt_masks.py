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
    parser.add_argument(
        "--student-state-root", type=Path,
        help="Optional exo-estimated hand state root for hand-only precision/recall.",
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
    predicted_hand_root = args.output_root / "predicted_hand"
    hand_root.mkdir(parents=True, exist_ok=True)
    object_root.mkdir(parents=True, exist_ok=True)
    if args.student_state_root is not None:
        predicted_hand_root.mkdir(parents=True, exist_ok=True)
        student_path = args.student_state_root / row["sequence"] / "student_state.npz"
        with np.load(student_path) as student:
            student_frames = student["frames"].copy()
            student_joints = student["hand_joints_world_m"].copy()
            student_confidence = student["joint_confidence"].copy()
    else:
        student_frames = student_joints = student_confidence = None
    hand_fractions, object_fractions = [], []
    pixel_counts = {
        "hand_true": 0, "hand_predicted": 0, "hand_intersection": 0,
        "object_true": 0, "object_predicted": 0, "object_intersection": 0,
        "arm_predicted": 0, "arm_hand_intersection": 0,
    }
    for record in manifest["frames"]:
        frame = int(record["dataset_frame"])
        local_index = int(record["index"])
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
        local_filename = f"{local_index:06d}.png"
        predicted_object = np.asarray(
            Image.open(args.model_input_root / "object_masks" / local_filename).convert("L")
        ) > 127
        predicted_arm = np.asarray(
            Image.open(args.model_input_root / "arm_masks" / local_filename).convert("L")
        ) > 127
        pixel_counts["object_true"] += int(object_mask.sum())
        pixel_counts["object_predicted"] += int(predicted_object.sum())
        pixel_counts["object_intersection"] += int(
            (object_mask & predicted_object).sum()
        )
        pixel_counts["arm_predicted"] += int(predicted_arm.sum())
        pixel_counts["arm_hand_intersection"] += int((predicted_arm & hand).sum())
        pixel_counts["hand_true"] += int(hand.sum())
        if student_frames is not None:
            student_index = int(np.searchsorted(student_frames, frame))
            if (
                student_index >= len(student_frames)
                or int(student_frames[student_index]) != frame
            ):
                raise IndexError(f"Student hand state missing frame {frame}")
            camera_from_world = np.linalg.inv(
                np.asarray(record["predicted_camera_pose_world"], dtype=np.float64)
            )
            world = student_joints[student_index]
            homogeneous = np.concatenate(
                (world, np.ones((*world.shape[:-1], 1), dtype=world.dtype)), axis=-1
            )
            predicted_camera = np.einsum(
                "ij,hkj->hki", camera_from_world, homogeneous
            )[..., :3]
            predicted_valid = student_confidence[student_index] > 0
            predicted_hand = project_mesh_mask(
                predicted_camera, predicted_valid, intrinsics, size
            )
            Image.fromarray(
                predicted_hand.astype(np.uint8) * 255, "L"
            ).save(predicted_hand_root / filename)
            pixel_counts["hand_predicted"] += int(predicted_hand.sum())
            pixel_counts["hand_intersection"] += int((predicted_hand & hand).sum())
        hand_fractions.append(float(hand.mean()))
        object_fractions.append(float(object_mask.mean()))

    def precision_recall(prefix: str) -> dict[str, float | int]:
        true = pixel_counts[f"{prefix}_true"]
        predicted = pixel_counts[f"{prefix}_predicted"]
        intersection = pixel_counts[f"{prefix}_intersection"]
        precision = intersection / max(predicted, 1)
        recall = intersection / max(true, 1)
        return {
            "true_pixels": true,
            "predicted_pixels": predicted,
            "intersection_pixels": intersection,
            "precision": precision,
            "recall": recall,
            "f1": 2.0 * precision * recall / max(precision + recall, 1e-12),
        }

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
        "hand_mesh_proxy_metrics": (
            precision_recall("hand")
            if args.student_state_root is not None else None
        ),
        "visible_object_metrics": precision_recall("object"),
        "arm_envelope_audit": {
            "arm_reference_available": False,
            "reason": (
                "H2O cam4 hand joints define a hand proxy but not a forearm "
                "silhouette; arm-mask IoU against the hand proxy is invalid."
            ),
            "hand_proxy_recall_inside_arm_envelope": (
                pixel_counts["arm_hand_intersection"]
                / max(pixel_counts["hand_true"], 1)
            ),
            "hand_proxy_fraction_of_arm_envelope": (
                pixel_counts["arm_hand_intersection"]
                / max(pixel_counts["arm_predicted"], 1)
            ),
            "hand_true_pixels": pixel_counts["hand_true"],
            "arm_predicted_pixels": pixel_counts["arm_predicted"],
            "intersection_pixels": pixel_counts["arm_hand_intersection"],
        },
    }
    (args.output_root / "summary.json").write_text(
        json.dumps(summary, indent=2) + "\n"
    )
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
