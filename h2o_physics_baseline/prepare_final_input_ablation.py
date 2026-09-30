#!/usr/bin/env python3
"""Prepare input-consistent exo/ego-anchor ablations for final validation clips.

The full4+anchor export is reused from the standard final pipeline.  This
script creates the reduced-input variants, including independent cam0-only
hand, arm, head and first-frame hand-mask state so the single-view columns
cannot inherit four-view motion estimates.  It also supports withholding the
initial head-camera mount through a cross-split exo-only pose summary.
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

from h2o_physics_baseline.protocols import LAYERED_CONTRACT_VERSION


INDEX = PROJECT_ROOT / "datasets/H2O/oracle_state/paired_physical_clips.csv"
DETECTION_PYTHON = PROJECT_ROOT / "envs/h2o-detection/bin/python"
VIDEO_PYTHON = PROJECT_ROOT / "envs/propainter/bin/python"

DEFAULT_PAIRS = (
    "subject3_h1_0_000000_000063_cam0_to_cam4",
    "subject3_h2_3_000384_000447_cam0_to_cam4",
    "subject3_k1_4_000160_000223_cam0_to_cam4",
    "subject3_o1_5_000320_000383_cam0_to_cam4",
    "subject3_o2_2_000064_000127_cam0_to_cam4",
)


def run(command: list[str], environment: dict[str, str] | None = None) -> None:
    print(json.dumps({"run": command}, ensure_ascii=False), flush=True)
    effective = dict(os.environ)
    if environment is not None:
        effective.update(environment)
    effective["PYTHONPATH"] = str(PROJECT_ROOT) + (
        os.pathsep + effective["PYTHONPATH"] if effective.get("PYTHONPATH") else ""
    )
    effective.setdefault("MPLCONFIGDIR", "/tmp/h2o-final-input-ablation-matplotlib")
    subprocess.run(command, cwd=PROJECT_ROOT, env=effective, check=True)


def npz_has_frames(path: Path, expected: set[int]) -> bool:
    if not path.exists():
        return False
    with np.load(path) as archive:
        return expected.issubset(set(int(value) for value in archive["frames"]))


def png_count(path: Path, digits: int = 6) -> int:
    if not path.is_dir():
        return 0
    return sum(
        item.suffix.lower() == ".png" and len(item.stem) == digits
        for item in path.glob("*.png")
    )


def head_ready(path: Path, pair: str, frames: set[int]) -> bool:
    if not path.exists():
        return False
    data = json.loads(path.read_text())
    if len(data.get("per_clip", [])) != 1:
        return False
    record = data["per_clip"][0]
    return (
        record.get("pair_id") == pair
        and set(int(value) for value in record.get("frames", [])) == frames
        and len(record.get("future", [])) == len(frames) - 1
    )


def layered_ready(model_input: Path, pair_id: str) -> bool:
    manifest_path = model_input / "manifest.json"
    if png_count(model_input / "input_frames") != 64 or not manifest_path.exists():
        return False
    try:
        manifest = json.loads(manifest_path.read_text())
    except (OSError, json.JSONDecodeError):
        return False
    contract = manifest.get("input_contract", {})
    return (
        manifest.get("pair_id") == pair_id
        and contract.get("version") == LAYERED_CONTRACT_VERSION
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--dataset-root", type=Path,
        default=Path("datasets/H2O/experiments/final_layer_scale_val5"),
    )
    parser.add_argument("--pair-id", action="append", dest="pairs")
    parser.add_argument("--gpu", default="2")
    parser.add_argument(
        "--initial-pose-summary", type=Path,
        default=Path(
            "datasets/H2O/experiments/initial_camera_mount_prior/"
            "exo4_rgbd_robust_train32_to_val8.json"
        ),
    )
    parser.add_argument("--stop-after", choices=("state", "layers", "propainter"))
    args = parser.parse_args()

    dataset_root = (PROJECT_ROOT / args.dataset_root).resolve()
    full_state = dataset_root / "state"
    cam0_state = dataset_root / "state_cam0"
    output_root = dataset_root / "input_ablation"
    pairs = tuple(args.pairs or DEFAULT_PAIRS)
    initial_pose_summary = (PROJECT_ROOT / args.initial_pose_summary).resolve()
    if not initial_pose_summary.exists():
        raise FileNotFoundError(initial_pose_summary)
    pose_targets = {
        record["pair_id"]
        for record in json.loads(initial_pose_summary.read_text())["targets"]
    }
    missing_pose_targets = set(pairs) - pose_targets
    if missing_pose_targets:
        raise ValueError(
            f"Initial-pose summary is missing targets: {sorted(missing_pose_targets)}"
        )
    with INDEX.open(encoding="utf-8") as handle:
        indexed = {row["pair_id"]: row for row in csv.DictReader(handle)}
    missing = [pair for pair in pairs if pair not in indexed]
    if missing:
        raise ValueError(f"Pairs missing from canonical index: {missing}")

    video_environment = dict(os.environ, CUDA_VISIBLE_DEVICES=args.gpu)
    records = []
    for pair in pairs:
        row = indexed[pair]
        sequence = Path(row["sequence"])
        sequence_key = row["sequence"].replace("/", "_")
        start, end = int(row["start_frame"]), int(row["end_frame"])
        dense_frames = set(range(start, end + 1))
        hand_path = cam0_state / "hand" / sequence / "student_state.npz"
        arm_path = cam0_state / "arm" / sequence / "arm_state.npz"
        head_path = cam0_state / "head" / f"{pair}.json"
        mask_path = cam0_state / "initial_hand_masks" / sequence / "initial_hand_masks.npz"

        if not npz_has_frames(hand_path, dense_frames):
            run([
                str(DETECTION_PYTHON),
                "h2o_geometric_baseline/precompute_mediapipe_student_state.py",
                "--pair-id", pair, "--frames-per-clip", "64",
                "--camera-indices", "0", "--output-root", str(cam0_state / "hand"),
            ])
        if not npz_has_frames(arm_path, dense_frames):
            run([
                str(DETECTION_PYTHON),
                "h2o_physics_baseline/precompute_exo_arm_state.py",
                "--split", row["split"], "--pair-id", pair,
                "--frames-per-clip", "64", "--camera-indices", "0",
                "--output-root", str(cam0_state / "arm"),
            ])
        if not head_ready(head_path, pair, dense_frames):
            run([
                str(DETECTION_PYTHON),
                "h2o_physics_baseline/audit_exo_face_head_motion.py",
                "--split", row["split"], "--pair-id", pair,
                "--frames-per-clip", "64", "--camera-indices", "0",
                "--output", str(head_path),
            ])
        if not npz_has_frames(mask_path, {start}):
            run([
                str(DETECTION_PYTHON),
                "h2o_geometric_baseline/precompute_initial_hand_masks.py",
                "--pair-id", pair, "--student-root", str(cam0_state / "hand"),
                "--output-root", str(cam0_state / "initial_hand_masks"),
                "--previews", "0",
            ])
        if args.stop_after == "state":
            continue

        scene_root = output_root / "scenes" / sequence_key / f"{start:06d}_{end:06d}"
        variants = (
            {
                "name": "exo4_no_anchor",
                "head": full_state / "head" / f"{pair}.json",
                "hand": full_state / "hand",
                "arm": full_state / "arm",
                "mask": full_state / "initial_hand_masks",
                "source": "0,1,2,3", "state": "0,1,2,3",
                "anchor": False, "object": False, "complete": False,
                "initial_pose_summary": None,
            },
            {
                "name": "cam0_anchor",
                "head": head_path, "hand": cam0_state / "hand",
                "arm": cam0_state / "arm", "mask": cam0_state / "initial_hand_masks",
                "source": "0", "state": "0",
                "anchor": True, "object": True, "complete": True,
                "initial_pose_summary": None,
            },
            {
                "name": "cam0_no_anchor",
                "head": head_path, "hand": cam0_state / "hand",
                "arm": cam0_state / "arm", "mask": cam0_state / "initial_hand_masks",
                "source": "0", "state": "0",
                "anchor": False, "object": False, "complete": True,
                "initial_pose_summary": None,
            },
            {
                "name": "exo4_no_anchor_no_mount",
                "head": full_state / "head" / f"{pair}.json",
                "hand": full_state / "hand",
                "arm": full_state / "arm",
                "mask": full_state / "initial_hand_masks",
                "source": "0,1,2,3", "state": "0,1,2,3",
                "anchor": False, "object": False, "complete": False,
                "initial_pose_summary": initial_pose_summary,
            },
        )
        for variant in variants:
            destination = scene_root / variant["name"]
            model_input = destination / "model_input"
            if not layered_ready(model_input, pair):
                command = [
                    str(VIDEO_PYTHON),
                    "h2o_physics_baseline/render_causal_video_background_split.py",
                    "--index", str(INDEX),
                    "--pair-id", pair, "--head-summary", str(variant["head"]),
                    "--student-state-root", str(variant["hand"]),
                    "--arm-state-root", str(variant["arm"]),
                    "--initial-hand-mask-root", str(variant["mask"]),
                    "--object-state-root", str(INDEX.parent),
                    "--source-camera-indices", str(variant["source"]),
                    "--state-camera-indices", str(variant["state"]),
                    "--source-feather-radius", "2", "--source-color-align",
                    "--causal-motion-masks",
                    "--model-input-root", str(model_input),
                    "--output", str(destination / "layered.mp4"),
                ]
                if not variant["anchor"]:
                    command.append("--disable-ego-anchor")
                if variant["initial_pose_summary"] is not None:
                    command.extend([
                        "--initial-pose-summary",
                        str(variant["initial_pose_summary"]),
                    ])
                if variant["object"]:
                    command.extend([
                        "--annotated-exo-object", "--annotated-exo-object-mode",
                        "anchor_warp", "--object-alpha", "0.25",
                    ])
                if variant["complete"]:
                    command.append("--complete-single-view-limbs")
                run(command, video_environment)
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
                    str(VIDEO_PYTHON),
                    "h2o_physics_baseline/compose_propainter_repair.py",
                    "--model-input-root", str(model_input),
                    "--propainter-frames", str(proposal_frames),
                    "--target-rgb-dir", row["target_rgb_dir"],
                    "--output-root", str(composed_root), "--size", "256",
                    "--feather-radius", "2", "--fps", "15",
                ], video_environment)
            records.append({"pair_id": pair, "variant": variant["name"], "root": str(destination)})

        (output_root / "ablation_scenes.json").parent.mkdir(parents=True, exist_ok=True)
        (output_root / "ablation_scenes.json").write_text(
            json.dumps({"variants": records}, indent=2), encoding="utf-8"
        )
    print(json.dumps({"completed_variants": len(records), "root": str(output_root)}), flush=True)


if __name__ == "__main__":
    main()
