#!/usr/bin/env python3
"""Prepare additional training clips with the complete current final pipeline.

Every accepted clip passes the same ordered stages: deployable four-exo hand
state, four-exo arm state, first-frame hand exclusion, final layered render,
ProPainter proposal, and constrained composition.  Completion is determined
from files and frame counts, so interrupted runs resume without admitting a
partial clip into the training manifest.
"""

from __future__ import annotations

import argparse
import csv
import json
import os
from pathlib import Path
import subprocess
import sys

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

# The public repository may live beside the private datasets/environments
# rather than containing them.  Keep code paths repository-relative while
# resolving large local assets from the containing workspace when necessary.
WORKSPACE_ROOT = (
    PROJECT_ROOT if (PROJECT_ROOT / "datasets").exists() else PROJECT_ROOT.parent
)

from h2o_physics_baseline.protocols import (
    LAYERED_CONTRACT_VERSION,
    validate_layered_manifest,
)


INDEX = WORKSPACE_ROOT / "datasets/H2O/oracle_state/paired_physical_clips.csv"
DETECTION_PYTHON = WORKSPACE_ROOT / "envs/h2o-detection/bin/python"
VIDEO_PYTHON = WORKSPACE_ROOT / "envs/propainter/bin/python"
FACE_MODEL = WORKSPACE_ROOT / "models/mediapipe/face_landmarker.task"
POSE_MODEL = WORKSPACE_ROOT / "models/mediapipe/pose_landmarker_full.task"
HAND_MODEL = WORKSPACE_ROOT / "models/mediapipe/hand_landmarker.task"
RAW_ROOT = WORKSPACE_ROOT / "datasets/H2O/raw"

DEFAULT_PAIRS = (
    "subject1_h1_0_000000_000063_cam0_to_cam4",
    "subject1_h2_3_000480_000543_cam0_to_cam4",
    "subject1_k2_2_000224_000287_cam0_to_cam4",
    "subject1_o2_0_000064_000127_cam0_to_cam4",
    "subject2_h1_5_000416_000479_cam0_to_cam4",
    "subject2_k1_1_000352_000415_cam0_to_cam4",
    "subject2_k2_6_000704_000767_cam0_to_cam4",
    "subject2_o1_2_000096_000159_cam0_to_cam4",
)

DEFAULT_TEST_PAIRS = (
    "subject4_h1_0_000000_000063_cam0_to_cam4",
    "subject4_h2_3_000384_000447_cam0_to_cam4",
    "subject4_k1_4_000160_000223_cam0_to_cam4",
    "subject4_o1_5_000320_000383_cam0_to_cam4",
    "subject4_o2_2_000064_000127_cam0_to_cam4",
    "subject4_k2_4_000160_000223_cam0_to_cam4",
    "subject4_k2_5_000000_000063_cam0_to_cam4",
    "subject4_o1_7_000000_000063_cam0_to_cam4",
)

DEFAULT_VAL_PAIRS = (
    "subject3_h1_0_000000_000063_cam0_to_cam4",
    "subject3_h2_3_000384_000447_cam0_to_cam4",
    "subject3_k1_4_000160_000223_cam0_to_cam4",
    "subject3_o1_5_000320_000383_cam0_to_cam4",
    "subject3_o2_2_000064_000127_cam0_to_cam4",
)


def run(command: list[str], environment: dict[str, str] | None = None) -> None:
    print(json.dumps({"run": command}, ensure_ascii=False), flush=True)
    effective_environment = dict(os.environ)
    if environment is not None:
        effective_environment.update(environment)
    previous_path = effective_environment.get("PYTHONPATH", "")
    effective_environment["PYTHONPATH"] = (
        str(PROJECT_ROOT) + (os.pathsep + previous_path if previous_path else "")
    )
    effective_environment.setdefault("MPLCONFIGDIR", "/tmp/h2o-final-scale-matplotlib")
    subprocess.run(
        command, cwd=PROJECT_ROOT, env=effective_environment, check=True
    )


def npz_has_frames(path: Path, expected: set[int]) -> bool:
    if not path.exists():
        return False
    with np.load(path) as archive:
        return expected.issubset(set(int(value) for value in archive["frames"]))


def png_count(path: Path, digits: int = 6) -> int:
    return sum(
        item.suffix.lower() == ".png" and len(item.stem) == digits
        for item in path.glob("*.png")
    ) if path.is_dir() else 0


def layered_ready(
    model_input: Path, pair_id: str,
    pose_confidence_calibration: Path | None = None,
    head_rotation_scale: float = 0.5,
) -> bool:
    manifest_path = model_input / "manifest.json"
    if not manifest_path.exists() or png_count(model_input / "input_frames") != 64:
        return False
    try:
        manifest = json.loads(manifest_path.read_text())
        validate_layered_manifest(
            manifest, protocol="anchored_gt_mount_annotated_object"
        )
    except (KeyError, TypeError, ValueError):
        return False
    return (
        manifest.get("pair_id") == pair_id
        and manifest["input_contract"].get("version") == LAYERED_CONTRACT_VERSION
        and manifest["input_contract"].get("pose_confidence_calibration")
        == (
            str(pose_confidence_calibration)
            if pose_confidence_calibration is not None else None
        )
        and float(manifest["input_contract"].get("head_rotation_scale", -1.0))
        == head_rotation_scale
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-root", type=Path)
    parser.add_argument(
        "--purpose", choices=("train", "val", "test"), default="train"
    )
    parser.add_argument("--pair-id", action="append", dest="pairs")
    parser.add_argument("--gpu", default="2")
    parser.add_argument("--pose-confidence-calibration", type=Path)
    parser.add_argument("--head-rotation-scale", type=float, default=0.5)
    parser.add_argument("--stop-after", choices=("state", "layers", "propainter"))
    args = parser.parse_args()
    default_roots = {
        "train": WORKSPACE_ROOT / "datasets/H2O/experiments/final_layer_scale_train8",
        "val": WORKSPACE_ROOT / "datasets/H2O/experiments/final_layer_scale_val5",
        "test": WORKSPACE_ROOT / "datasets/H2O/experiments/final_layer_frozen_test8",
    }
    output_root = (args.output_root or default_roots[args.purpose]).resolve()
    state_root = output_root / "state/hand"
    arm_root = output_root / "state/arm"
    head_root = output_root / "state/head"
    mask_root = output_root / "state/initial_hand_masks"
    clip_root = output_root / "clips"
    output_root.mkdir(parents=True, exist_ok=True)

    with INDEX.open(encoding="utf-8") as handle:
        indexed = {row["pair_id"]: row for row in csv.DictReader(handle)}
    default_pairs = {
        "train": DEFAULT_PAIRS,
        "val": DEFAULT_VAL_PAIRS,
        "test": DEFAULT_TEST_PAIRS,
    }
    pairs = tuple(args.pairs or default_pairs[args.purpose])
    missing = [pair for pair in pairs if pair not in indexed]
    if missing:
        raise ValueError(f"Pairs missing from canonical index: {missing}")
    expected_split = args.purpose
    wrong_split = [pair for pair in pairs if indexed[pair]["split"] != expected_split]
    if wrong_split:
        raise ValueError(
            f"purpose={args.purpose} requires split={expected_split}: {wrong_split}"
        )

    completed = []
    video_environment = dict(os.environ, CUDA_VISIBLE_DEVICES=args.gpu)
    for pair in pairs:
        row = indexed[pair]
        sequence = Path(row["sequence"])
        start, end = int(row["start_frame"]), int(row["end_frame"])
        dense_frames = set(range(start, end + 1))
        hand_path = state_root / sequence / "student_state.npz"
        arm_path = arm_root / sequence / "arm_state.npz"
        head_path = head_root / f"{pair}.json"
        mask_path = mask_root / sequence / "initial_hand_masks.npz"

        if not npz_has_frames(hand_path, dense_frames):
            run([
                str(DETECTION_PYTHON),
                "h2o_geometric_baseline/precompute_mediapipe_student_state.py",
                "--pair-id", pair, "--frames-per-clip", "64",
                "--index", str(INDEX), "--model", str(HAND_MODEL),
                "--output-root", str(state_root),
            ])
        if not npz_has_frames(arm_path, dense_frames):
            run([
                str(DETECTION_PYTHON), "h2o_physics_baseline/precompute_exo_arm_state.py",
                "--split", row["split"], "--pair-id", pair,
                "--index", str(INDEX), "--pose-model", str(POSE_MODEL),
                "--frames-per-clip", "64", "--output-root", str(arm_root),
            ])
        head_ready = False
        if head_path.exists():
            head_data = json.loads(head_path.read_text())
            if len(head_data.get("per_clip", [])) == 1:
                head_record = head_data["per_clip"][0]
                head_ready = (
                    head_record.get("pair_id") == pair
                    and set(int(value) for value in head_record.get("frames", [])) == dense_frames
                    and len(head_record.get("future", [])) == 63
                )
        if not head_ready:
            run([
                str(DETECTION_PYTHON),
                "h2o_physics_baseline/audit_exo_face_head_motion.py",
                "--split", row["split"], "--pair-id", pair,
                "--index", str(INDEX),
                "--model", str(FACE_MODEL), "--pose-model", str(POSE_MODEL),
                "--frames-per-clip", "64", "--output", str(head_path),
            ])
        if not npz_has_frames(mask_path, {start}):
            run([
                str(DETECTION_PYTHON),
                "h2o_geometric_baseline/precompute_initial_hand_masks.py",
                "--pair-id", pair, "--student-root", str(state_root),
                "--index", str(INDEX), "--raw-root", str(RAW_ROOT),
                "--output-root", str(mask_root), "--previews", "0",
            ])
        if args.stop_after == "state":
            continue

        destination = clip_root / row["sequence"].replace("/", "_") / f"{start:06d}_{end:06d}"
        model_input = destination / "model_input"
        layers_updated = not layered_ready(
            model_input, pair, args.pose_confidence_calibration,
            args.head_rotation_scale,
        )
        if layers_updated:
            render_command = [
                str(VIDEO_PYTHON),
                "h2o_physics_baseline/render_causal_video_background_split.py",
                "--index", str(INDEX),
                "--pair-id", pair, "--head-summary", str(head_path),
                "--student-state-root", str(state_root),
                "--arm-state-root", str(arm_root),
                "--initial-hand-mask-root", str(mask_root),
                "--object-state-root", str(INDEX.parent),
                "--source-feather-radius", "2", "--source-color-align",
                "--head-rotation-scale", str(args.head_rotation_scale),
                "--causal-motion-masks",
                "--annotated-exo-object", "--annotated-exo-object-mode", "anchor_warp",
                "--object-alpha", "0.25", "--model-input-root", str(model_input),
                "--output", str(destination / "layered.mp4"),
            ]
            if args.pose_confidence_calibration is not None:
                render_command.extend([
                    "--pose-confidence-calibration",
                    str(args.pose_confidence_calibration.resolve()),
                ])
            run(render_command, video_environment)
        if args.stop_after == "layers":
            continue

        proposal_root = destination / "propainter_unknown"
        proposal_frames = proposal_root / "input_frames/frames"
        # A layer rebuild changes the ProPainter inputs even when the stale
        # output directory still happens to contain 64 files.
        if layers_updated or png_count(proposal_frames, digits=4) != 64:
            run([
                str(VIDEO_PYTHON), "third_party/ProPainter/inference_propainter.py",
                "-i", str(model_input / "input_frames"),
                "-m", str(model_input / "repair_masks"),
                "-o", str(proposal_root), "--width", "256", "--height", "256",
                "--save_frames", "--fp16", "--save_fps", "15",
            ], video_environment)
        composed_root = destination / "propainter_composed"
        if layers_updated or png_count(composed_root / "frames") != 64:
            run([
                str(VIDEO_PYTHON), "h2o_physics_baseline/compose_propainter_repair.py",
                "--model-input-root", str(model_input),
                "--propainter-frames", str(proposal_frames),
                "--target-rgb-dir", row["target_rgb_dir"],
                "--output-root", str(composed_root), "--size", "256",
                "--feather-radius", "2", "--fps", "15",
            ], video_environment)
        completed.append({
            "pair_id": pair,
            "root": os.path.relpath(destination, PROJECT_ROOT),
            "frames": 64,
            "protocol": "current_final_full4_anchor_object_propainter",
        })
        (output_root / f"{args.purpose}_clips.json").write_text(
            json.dumps({"purpose": args.purpose, "clips": completed}, indent=2),
            encoding="utf-8"
        )

    print(json.dumps({"completed": len(completed), "root": str(output_root)}), flush=True)


if __name__ == "__main__":
    main()
