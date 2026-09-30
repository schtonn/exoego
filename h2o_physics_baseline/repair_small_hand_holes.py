#!/usr/bin/env python3
"""Repair persistent small jagged hand holes without changing the outer outline."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

import cv2
import numpy as np
from PIL import Image

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from h2o_geometric_baseline.reprojection import load_intrinsics, load_pose
from h2o_physics_baseline.audit_kinematic_hand_mesh_holes import (
    boundary_jaggedness,
    enclosed_small_components,
    interpolate_state,
    predicted_pose_map,
    project_mesh_mask,
)


def l1(first: np.ndarray, second: np.ndarray, mask: np.ndarray | None = None) -> float:
    error = np.abs(first.astype(np.float32) - second.astype(np.float32)) / 255.0
    if mask is not None:
        return float(error[mask].mean()) if np.any(mask) else 0.0
    return float(error.mean())


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--sequence-root", type=Path, required=True)
    parser.add_argument("--model-input-root", type=Path, required=True)
    parser.add_argument("--prediction-root", type=Path, required=True)
    parser.add_argument("--student-state", type=Path, required=True)
    parser.add_argument("--head-summary", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--maximum-hole-area", type=int, default=128)
    parser.add_argument("--minimum-jaggedness", type=float, default=1.45)
    parser.add_argument("--mesh-neighborhood-px", type=int, default=0)
    parser.add_argument("--minimum-mesh-overlap", type=float, default=0.75)
    parser.add_argument("--object-guard-px", type=int, default=2)
    parser.add_argument(
        "--color-completion", choices=("telea", "navier_stokes"),
        default="navier_stokes",
    )
    args = parser.parse_args()

    manifest = json.loads((args.model_input_root / "manifest.json").read_text())
    frames = [int(value["dataset_frame"]) for value in manifest["frames"]]
    pair_id = manifest["pair_id"]
    size = int(manifest["image_size"])
    intrinsics = load_intrinsics(args.sequence_root / "cam4/cam_intrinsics.txt")
    initial_pose = load_pose(args.sequence_root / "cam4/cam_pose" / f"{frames[0]:06d}.txt")
    summary = json.loads(args.head_summary.read_text())
    record = next(value for value in summary["per_clip"] if value["pair_id"] == pair_id)
    poses = predicted_pose_map(frames, initial_pose, record)
    with np.load(args.student_state) as archive:
        state_frames = archive["frames"].copy()
        state_joints = archive["hand_joints_world_m"].copy()
        state_confidence = archive["joint_confidence"].copy()

    output_frames = args.output_root / "frames"
    output_masks = args.output_root / "fill_masks"
    output_arm_masks = args.output_root / "arm_masks"
    for directory in (output_frames, output_masks, output_arm_masks):
        directory.mkdir(parents=True, exist_ok=True)

    totals = {
        "frames": len(frames),
        "filled_components": 0,
        "filled_pixels": 0,
        "frames_with_fill": 0,
        "original_l1_sum": 0.0,
        "repaired_l1_sum": 0.0,
        "original_fill_l1_sum": 0.0,
        "repaired_fill_l1_sum": 0.0,
        "filled_frames_for_region_metric": 0,
        "improved_frames": 0,
        "outside_max_difference": 0,
        "object_guarded_components": 0,
    }
    per_frame = []
    kernel_radius = args.mesh_neighborhood_px
    kernel = cv2.getStructuringElement(
        cv2.MORPH_ELLIPSE, (2 * kernel_radius + 1, 2 * kernel_radius + 1)
    )
    for index, frame in enumerate(frames):
        name = f"{index:06d}.png"
        prediction = np.asarray(Image.open(args.prediction_root / name).convert("RGB"))
        target = np.asarray(
            Image.open(args.sequence_root / "cam4/rgb" / f"{frame:06d}.png")
            .convert("RGB").resize((size, size), Image.Resampling.BILINEAR)
        )
        raw_mask = np.asarray(Image.open(args.model_input_root / "arm_masks" / name)) > 0
        object_mask = np.asarray(Image.open(args.model_input_root / "object_masks" / name)) > 0
        if args.object_guard_px > 0:
            object_mask = cv2.dilate(
                object_mask.astype(np.uint8),
                cv2.getStructuringElement(
                    cv2.MORPH_ELLIPSE,
                    (2 * args.object_guard_px + 1, 2 * args.object_guard_px + 1),
                ),
            ) > 0
        fill = np.zeros_like(raw_mask)
        component_count = 0
        if index > 0:
            joints_world, confidence = interpolate_state(
                state_frames, state_joints, state_confidence, frame
            )
            camera_from_world = np.linalg.inv(poses[frame])
            joints_camera = joints_world @ camera_from_world[:3, :3].T + camera_from_world[:3, 3]
            mesh = project_mesh_mask(joints_camera, confidence >= 0.15, intrinsics, size)
            neighborhood = cv2.dilate(mesh.astype(np.uint8), kernel) > 0
            for component in enclosed_small_components(raw_mask, args.maximum_hole_area):
                if boundary_jaggedness(component) < args.minimum_jaggedness:
                    continue
                if np.any(component & object_mask):
                    totals["object_guarded_components"] += 1
                    continue
                overlap = float((component & neighborhood).sum()) / max(int(component.sum()), 1)
                if overlap >= args.minimum_mesh_overlap:
                    fill |= component
                    component_count += 1
        inpaint_method = (
            cv2.INPAINT_TELEA
            if args.color_completion == "telea"
            else cv2.INPAINT_NS
        )
        repaired = cv2.inpaint(
            prediction, fill.astype(np.uint8) * 255, 3.0, inpaint_method
        )
        repaired_arm = raw_mask | fill
        Image.fromarray(repaired, "RGB").save(output_frames / name)
        Image.fromarray(fill.astype(np.uint8) * 255, "L").save(output_masks / name)
        Image.fromarray(repaired_arm.astype(np.uint8) * 255, "L").save(output_arm_masks / name)

        original_l1 = l1(prediction, target)
        repaired_l1 = l1(repaired, target)
        outside_difference = int(
            np.abs(repaired.astype(np.int16) - prediction.astype(np.int16))[~fill].max()
            if np.any(~fill) else 0
        )
        record = {
            "frame": frame,
            "filled_components": component_count,
            "filled_pixels": int(fill.sum()),
            "original_l1": original_l1,
            "repaired_l1": repaired_l1,
            "outside_max_difference": outside_difference,
        }
        if np.any(fill):
            original_region = l1(prediction, target, fill)
            repaired_region = l1(repaired, target, fill)
            record["original_fill_l1"] = original_region
            record["repaired_fill_l1"] = repaired_region
            totals["original_fill_l1_sum"] += original_region
            totals["repaired_fill_l1_sum"] += repaired_region
            totals["filled_frames_for_region_metric"] += 1
            totals["improved_frames"] += int(repaired_region < original_region)
            totals["frames_with_fill"] += 1
        totals["filled_components"] += component_count
        totals["filled_pixels"] += int(fill.sum())
        totals["original_l1_sum"] += original_l1
        totals["repaired_l1_sum"] += repaired_l1
        totals["outside_max_difference"] = max(totals["outside_max_difference"], outside_difference)
        per_frame.append(record)

    region_count = max(totals["filled_frames_for_region_metric"], 1)
    metrics = {
        "protocol": {
            "maximum_hole_area_px": args.maximum_hole_area,
            "minimum_jaggedness": args.minimum_jaggedness,
            "mesh_neighborhood_px": args.mesh_neighborhood_px,
            "minimum_mesh_overlap": args.minimum_mesh_overlap,
            "object_guard_px": args.object_guard_px,
            "color_completion": (
                f"OpenCV {args.color_completion}, radius 3, authorized pixels only"
            ),
            "future_ego_used_for_inference": False,
        },
        "frames": totals["frames"],
        "frames_with_fill": totals["frames_with_fill"],
        "filled_components": totals["filled_components"],
        "filled_pixels": totals["filled_pixels"],
        "filled_fraction": totals["filled_pixels"] / (len(frames) * size * size),
        "original_l1": totals["original_l1_sum"] / len(frames),
        "repaired_l1": totals["repaired_l1_sum"] / len(frames),
        "relative_full_l1_change": (
            totals["repaired_l1_sum"] / max(totals["original_l1_sum"], 1e-12) - 1.0
        ),
        "original_fill_l1": totals["original_fill_l1_sum"] / region_count,
        "repaired_fill_l1": totals["repaired_fill_l1_sum"] / region_count,
        "relative_fill_l1_change": (
            totals["repaired_fill_l1_sum"] / max(totals["original_fill_l1_sum"], 1e-12) - 1.0
            if totals["filled_frames_for_region_metric"] else 0.0
        ),
        "improved_filled_frames": totals["improved_frames"],
        "outside_max_difference": totals["outside_max_difference"],
        "object_guarded_components": totals["object_guarded_components"],
        "per_frame": per_frame,
    }
    (args.output_root / "metrics.json").write_text(
        json.dumps(metrics, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )
    print(json.dumps({key: value for key, value in metrics.items() if key != "per_frame"}, indent=2))


if __name__ == "__main__":
    main()
