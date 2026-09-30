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

import numpy as np


PROJECT_ROOT = Path(__file__).resolve().parents[1]
INDEX = PROJECT_ROOT / "datasets/H2O/oracle_state/paired_physical_clips.csv"
HEAD_SUMMARY = PROJECT_ROOT / "datasets/H2O/experiments/exo_face_head_motion/summary_train32.json"
DETECTION_PYTHON = PROJECT_ROOT / "envs/h2o-detection/bin/python"
VIDEO_PYTHON = PROJECT_ROOT / "envs/propainter/bin/python"

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


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-root", type=Path, default=Path(
        "datasets/H2O/experiments/final_layer_scale_train8"
    ))
    parser.add_argument("--pair-id", action="append", dest="pairs")
    parser.add_argument("--gpu", default="2")
    parser.add_argument("--stop-after", choices=("state", "layers", "propainter"))
    args = parser.parse_args()
    output_root = (PROJECT_ROOT / args.output_root).resolve()
    state_root = output_root / "state/hand"
    arm_root = output_root / "state/arm"
    head_root = output_root / "state/head"
    mask_root = output_root / "state/initial_hand_masks"
    clip_root = output_root / "clips"
    output_root.mkdir(parents=True, exist_ok=True)

    with INDEX.open(encoding="utf-8") as handle:
        indexed = {row["pair_id"]: row for row in csv.DictReader(handle)}
    pairs = tuple(args.pairs or DEFAULT_PAIRS)
    missing = [pair for pair in pairs if pair not in indexed]
    if missing:
        raise ValueError(f"Pairs missing from canonical index: {missing}")

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
                "--output-root", str(state_root),
            ])
        if not npz_has_frames(arm_path, dense_frames):
            run([
                str(DETECTION_PYTHON), "h2o_physics_baseline/precompute_exo_arm_state.py",
                "--split", row["split"], "--pair-id", pair,
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
                "--frames-per-clip", "64", "--output", str(head_path),
            ])
        if not npz_has_frames(mask_path, {start}):
            run([
                str(DETECTION_PYTHON),
                "h2o_geometric_baseline/precompute_initial_hand_masks.py",
                "--pair-id", pair, "--student-root", str(state_root),
                "--output-root", str(mask_root), "--previews", "0",
            ])
        if args.stop_after == "state":
            continue

        destination = clip_root / row["sequence"].replace("/", "_") / f"{start:06d}_{end:06d}"
        model_input = destination / "model_input"
        manifest = model_input / "manifest.json"
        if not manifest.exists() or png_count(model_input / "input_frames") != 64:
            run([
                str(VIDEO_PYTHON),
                "h2o_physics_baseline/render_causal_video_background_split.py",
                "--pair-id", pair, "--head-summary", str(head_path),
                "--student-state-root", str(state_root),
                "--arm-state-root", str(arm_root),
                "--initial-hand-mask-root", str(mask_root),
                "--source-feather-radius", "2", "--source-color-align",
                "--annotated-exo-object", "--annotated-exo-object-mode", "anchor_warp",
                "--object-alpha", "0.25", "--model-input-root", str(model_input),
                "--output", str(destination / "layered.mp4"),
            ], video_environment)
        if args.stop_after == "layers":
            continue

        proposal_root = destination / "propainter_unknown"
        proposal_frames = proposal_root / "input_frames/frames"
        if png_count(proposal_frames, digits=4) != 64:
            run([
                str(VIDEO_PYTHON), "third_party/ProPainter/inference_propainter.py",
                "-i", str(model_input / "input_frames"),
                "-m", str(model_input / "repair_masks"),
                "-o", str(proposal_root), "--width", "256", "--height", "256",
                "--save_frames", "--fp16", "--save_fps", "15",
            ], video_environment)
        composed_root = destination / "propainter_composed"
        if png_count(composed_root / "frames") != 64:
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
            "root": str(destination.relative_to(PROJECT_ROOT)),
            "frames": 64,
            "protocol": "current_final_full4_anchor_object_propainter",
        })
        (output_root / "training_clips.json").write_text(
            json.dumps({"clips": completed}, indent=2), encoding="utf-8"
        )

    print(json.dumps({"completed": len(completed), "root": str(output_root)}), flush=True)


if __name__ == "__main__":
    main()
