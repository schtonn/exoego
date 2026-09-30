#!/usr/bin/env python3
"""Extract initial-ego hand masks from exo student joints and the allowed ego anchor."""

from __future__ import annotations

import argparse
import csv
import json
from collections import defaultdict
from pathlib import Path

import cv2
import numpy as np
from PIL import Image, ImageDraw

from h2o_geometric_baseline.mediapipe_multiview_audit import load_flat, load_hand, project_camera
from h2o_geometric_baseline.precompute_mediapipe_student_state import DEFAULT_INDEX, selected_rows


DEFAULT_RAW = Path("datasets/H2O/raw")
DEFAULT_STUDENT = Path("datasets/H2O/student_state_mediapipe_64_32")
DEFAULT_OUTPUT = Path("datasets/H2O/student_initial_hand_masks_64_32")
HAND_EDGES = (
    (0, 1), (1, 2), (2, 3), (3, 4),
    (0, 5), (5, 6), (6, 7), (7, 8),
    (0, 9), (9, 10), (10, 11), (11, 12),
    (0, 13), (13, 14), (14, 15), (15, 16),
    (0, 17), (17, 18), (18, 19), (19, 20),
    (5, 9), (9, 13), (13, 17),
)


def project_world(points: np.ndarray, pose_world: np.ndarray, intrinsics: np.ndarray) -> np.ndarray:
    shape = points.shape[:-1]
    points = points.reshape(-1, 3)
    world_to_camera = np.linalg.inv(pose_world)
    camera = points @ world_to_camera[:3, :3].T + world_to_camera[:3, 3]
    return project_camera(camera, intrinsics)[0].reshape(*shape, 2)


def seeded_grabcut(image: np.ndarray, uv: np.ndarray, confidence: np.ndarray) -> np.ndarray:
    height, width = image.shape[:2]
    valid = (
        np.isfinite(uv).all(axis=1)
        & (confidence > 0)
        & (uv[:, 0] >= 0)
        & (uv[:, 0] < width)
        & (uv[:, 1] >= 0)
        & (uv[:, 1] < height)
    )
    if int(valid.sum()) < 5:
        return np.zeros((height, width), dtype=np.uint8)
    points = np.rint(uv).astype(np.int32)
    seed = np.zeros((height, width), dtype=np.uint8)
    for first, second in HAND_EDGES:
        if valid[first] and valid[second]:
            cv2.line(seed, tuple(points[first]), tuple(points[second]), 255, 7, cv2.LINE_AA)
    for point in points[valid]:
        cv2.circle(seed, tuple(point), 6, 255, -1, cv2.LINE_AA)
    probable = cv2.dilate(seed, np.ones((25, 25), np.uint8))
    ys, xs = np.nonzero(probable)
    if not len(xs):
        return np.zeros((height, width), dtype=np.uint8)
    margin = 25
    x0, x1 = max(0, int(xs.min()) - margin), min(width, int(xs.max()) + margin + 1)
    y0, y1 = max(0, int(ys.min()) - margin), min(height, int(ys.max()) + margin + 1)
    labels = np.full((height, width), cv2.GC_BGD, dtype=np.uint8)
    labels[y0:y1, x0:x1] = cv2.GC_PR_BGD
    labels[probable > 0] = cv2.GC_PR_FGD
    labels[seed > 0] = cv2.GC_FGD
    background = np.zeros((1, 65), dtype=np.float64)
    foreground = np.zeros((1, 65), dtype=np.float64)
    try:
        cv2.grabCut(image, labels, None, background, foreground, 5, cv2.GC_INIT_WITH_MASK)
    except cv2.error:
        return probable
    mask = np.where((labels == cv2.GC_FGD) | (labels == cv2.GC_PR_FGD), 255, 0).astype(np.uint8)
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, np.ones((7, 7), np.uint8))
    # Remove disconnected color regions not attached to the projected skeleton.
    count, components, _, _ = cv2.connectedComponentsWithStats(mask, 8)
    keep = np.zeros_like(mask)
    for component in range(1, count):
        region = components == component
        if np.any(region & (seed > 0)):
            keep[region] = 255
    return keep


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--index", type=Path, default=DEFAULT_INDEX)
    parser.add_argument("--raw-root", type=Path, default=DEFAULT_RAW)
    parser.add_argument("--student-root", type=Path, default=DEFAULT_STUDENT)
    parser.add_argument("--output-root", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--max-train-samples", type=int, default=64)
    parser.add_argument("--max-val-samples", type=int, default=32)
    parser.add_argument("--image-size", type=int, default=64)
    parser.add_argument("--previews", type=int, default=12)
    parser.add_argument("--pair-id", type=str, default=None)
    args = parser.parse_args()

    if args.pair_id is not None:
        with args.index.open(encoding="utf-8") as handle:
            rows = [row for row in csv.DictReader(handle) if row["pair_id"] == args.pair_id]
        if not rows:
            raise ValueError(f"Pair id not found: {args.pair_id}")
    else:
        rows = selected_rows(args.index, "train", args.max_train_samples) + selected_rows(
            args.index, "val", args.max_val_samples
        )
    sequence_frames: dict[str, set[int]] = defaultdict(set)
    for row in rows:
        sequence_frames[row["sequence"]].add(int(row["start_frame"]))

    joint_hits: list[float] = []
    dilated_joint_hits: list[float] = []
    mask_area: list[float] = []
    missing_masks = 0
    total_masks = 0
    preview_count = 0
    per_sequence = []
    for sequence in sorted(sequence_frames):
        sequence_root = args.raw_root / sequence
        camera_root = sequence_root / "cam4"
        intrinsics = load_flat(camera_root / "cam_intrinsics.txt")
        with np.load(args.student_root / sequence / "student_state.npz") as archive:
            student = {name: archive[name] for name in archive.files}
        frames = np.asarray(sorted(sequence_frames[sequence]), dtype=np.int32)
        masks = np.zeros((len(frames), 2, args.image_size, args.image_size), dtype=np.uint8)
        sequence_missing = 0
        for frame_index, frame in enumerate(frames):
            stem = f"{int(frame):06d}"
            student_index = int(np.searchsorted(student["frames"], frame))
            if student_index >= len(student["frames"]) or student["frames"][student_index] != frame:
                raise IndexError(f"Student state missing {sequence} frame {frame}")
            pose = load_flat(camera_root / "cam_pose" / f"{stem}.txt", 16).reshape(4, 4)
            uv = project_world(student["hand_joints_world_m"][student_index], pose, intrinsics)
            image_path = camera_root / "rgb" / f"{stem}.png"
            image_bgr = cv2.imread(str(image_path), cv2.IMREAD_COLOR)
            if image_bgr is None:
                raise FileNotFoundError(image_path)
            full_masks = []
            for hand in range(2):
                mask = seeded_grabcut(
                    image_bgr,
                    uv[hand],
                    student["joint_confidence"][student_index, hand],
                )
                full_masks.append(mask)
                total_masks += 1
                if not np.any(mask):
                    missing_masks += 1
                    sequence_missing += 1
                else:
                    mask_area.append(float(np.mean(mask > 0)))
                masks[frame_index, hand] = cv2.resize(
                    mask, (args.image_size, args.image_size), interpolation=cv2.INTER_AREA
                )

            # Ground truth is read only after mask extraction for point-coverage audit.
            presence, gt_camera = load_hand(camera_root / "hand_pose" / f"{stem}.txt")
            gt_uv, gt_valid = [], []
            for hand in range(2):
                projected, valid = project_camera(gt_camera[hand], intrinsics)
                gt_uv.append(projected)
                gt_valid.append(valid)
                if not presence[hand] or not np.any(valid):
                    continue
                mask = full_masks[hand]
                dilated = cv2.dilate(mask, np.ones((11, 11), np.uint8))
                points = np.rint(projected[valid]).astype(np.int32)
                inside = (
                    (points[:, 0] >= 0) & (points[:, 0] < mask.shape[1])
                    & (points[:, 1] >= 0) & (points[:, 1] < mask.shape[0])
                )
                points = points[inside]
                if len(points):
                    joint_hits.append(float(np.mean(mask[points[:, 1], points[:, 0]] > 0)))
                    dilated_joint_hits.append(float(np.mean(dilated[points[:, 1], points[:, 0]] > 0)))

            if preview_count < args.previews:
                rgb = Image.open(image_path).convert("RGB")
                overlay = Image.new("RGBA", rgb.size, (0, 0, 0, 0))
                colors = ((255, 48, 48, 95), (0, 210, 255, 95))
                for hand, mask in enumerate(full_masks):
                    color = np.zeros((mask.shape[0], mask.shape[1], 4), dtype=np.uint8)
                    color[mask > 0] = colors[hand]
                    overlay.alpha_composite(Image.fromarray(color, mode="RGBA"))
                composed = Image.alpha_composite(rgb.convert("RGBA"), overlay).convert("RGB")
                draw = ImageDraw.Draw(composed)
                for hand in range(2):
                    for x, y in gt_uv[hand][gt_valid[hand]]:
                        draw.ellipse((x - 3, y - 3, x + 3, y + 3), outline="white", width=2)
                destination = args.output_root / "previews" / sequence.replace("/", "_")
                destination.mkdir(parents=True, exist_ok=True)
                composed.resize((640, 360), Image.Resampling.LANCZOS).save(destination / f"{stem}.jpg")
                preview_count += 1
        destination = args.output_root / sequence
        destination.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(destination / "initial_hand_masks.npz", frames=frames, masks=masks)
        per_sequence.append(
            {"sequence": sequence, "frames": int(len(frames)), "missing_hand_masks": sequence_missing}
        )

    report = {
        "contract": {
            "mask_input": "initial ego RGB + exo-only student joints + initial ego calibration",
            "future_ego_information_used": False,
            "ground_truth_use": "post-hoc projected-joint coverage audit only",
        },
        "clips": len(rows),
        "sequences": len(sequence_frames),
        "hand_masks": total_masks,
        "missing_hand_masks": missing_masks,
        "mask_availability": 1.0 - missing_masks / max(total_masks, 1),
        "mask_area_fraction": {
            "mean": float(np.mean(mask_area)),
            "median": float(np.median(mask_area)),
            "p95": float(np.percentile(mask_area, 95)),
        },
        "gt_joint_inside_mask": float(np.mean(joint_hits)),
        "gt_joint_inside_mask_dilated_11px": float(np.mean(dilated_joint_hits)),
        "per_sequence": per_sequence,
    }
    args.output_root.mkdir(parents=True, exist_ok=True)
    (args.output_root / "summary.json").write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
