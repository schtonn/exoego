#!/usr/bin/env python3
"""Fit an exo-hand-only proxy for object center and audit it on held-out clips."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np

from h2o_physics_baseline.dataset import DEFAULT_INDEX, DEFAULT_STATS, H2OPhysicalClipDataset


def _dataset(split: str, maximum: int) -> H2OPhysicalClipDataset:
    return H2OPhysicalClipDataset(
        DEFAULT_INDEX,
        split=split,
        frames_per_clip=4,
        image_size=64,
        stats_path=DEFAULT_STATS,
        source_cameras=("cam0", "cam1", "cam2", "cam3"),
        max_samples=maximum,
        combine_source_cameras=True,
    )


def _fill_hands(joints: np.ndarray, confidence: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    joints = joints.astype(np.float64).copy()
    confidence = np.nan_to_num(confidence.astype(np.float64), nan=0.0)
    valid = np.isfinite(joints).all(axis=-1) & (confidence > 0)
    all_valid_points = joints[valid]
    fallback = np.nanmean(all_valid_points, axis=0) if len(all_valid_points) else np.zeros(3)
    for hand in range(2):
        hand_valid = valid[hand]
        center = joints[hand, hand_valid].mean(axis=0) if np.any(hand_valid) else fallback
        joints[hand, ~hand_valid] = center
    return joints, valid.astype(np.float64)


def _features(joints: np.ndarray, confidence: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    filled, valid = _fill_hands(joints, confidence)
    weights = np.maximum(confidence, 0) * valid
    denominator = weights.sum()
    global_center = (
        (filled * weights[..., None]).sum(axis=(0, 1)) / denominator
        if denominator > 0
        else filled.mean(axis=(0, 1))
    )
    centered = filled - global_center
    hand_centers = []
    for hand in range(2):
        hand_weight = weights[hand]
        if hand_weight.sum() > 0:
            center = (filled[hand] * hand_weight[:, None]).sum(axis=0) / hand_weight.sum()
        else:
            center = global_center
        hand_centers.append(center - global_center)
    feature = np.concatenate(
        (
            centered.reshape(-1),
            np.concatenate(hand_centers),
            np.clip(confidence, 0, 1).reshape(-1),
            valid.reshape(-1),
        )
    )
    return feature, global_center


def _load_rows(
    split: str, maximum: int, student_root: Path
) -> tuple[np.ndarray, np.ndarray, np.ndarray, list[str], np.ndarray, list[str]]:
    features, centers, targets, groups, object_ids, clip_ids = [], [], [], [], [], []
    dataset = _dataset(split, maximum)
    for row in dataset.rows:
        with np.load(row["state_path"]) as archive:
            state = {name: archive[name] for name in archive.files}
        student_path = student_root / row["sequence"] / "student_state.npz"
        with np.load(student_path) as archive:
            student = {name: archive[name] for name in archive.files}
        frame_numbers, indices = dataset._frame_indices(row, state["frames"])
        student_indices = np.searchsorted(student["frames"], frame_numbers)
        for state_index, student_index in zip(indices, student_indices, strict=True):
            feature, center = _features(
                student["hand_joints_world_m"][student_index],
                student["joint_confidence"][student_index],
            )
            target = state["object_center_world_m"][state_index].astype(np.float64)
            if not np.isfinite(target).all():
                continue
            features.append(feature)
            centers.append(center)
            targets.append(target)
            groups.append(row["sequence"])
            object_ids.append(int(state["object_id"][state_index]))
            clip_ids.append(row["clip_id"])
    return (
        np.asarray(features),
        np.asarray(centers),
        np.asarray(targets),
        groups,
        np.asarray(object_ids),
        clip_ids,
    )


def _fit_ridge(x: np.ndarray, y: np.ndarray, alpha: float) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    mean = x.mean(axis=0)
    scale = np.maximum(x.std(axis=0), 1e-6)
    normalized = (x - mean) / scale
    augmented = np.concatenate((normalized, np.ones((len(x), 1))), axis=1)
    penalty = np.eye(augmented.shape[1]) * alpha
    penalty[-1, -1] = 0
    weights = np.linalg.solve(augmented.T @ augmented + penalty, augmented.T @ y)
    return weights, mean, scale


def _predict(x: np.ndarray, weights: np.ndarray, mean: np.ndarray, scale: np.ndarray) -> np.ndarray:
    normalized = (x - mean) / scale
    return np.concatenate((normalized, np.ones((len(x), 1))), axis=1) @ weights


def _summary(error_m: np.ndarray) -> dict[str, float | int]:
    error_mm = error_m * 1000
    return {
        "count": int(len(error_mm)),
        "mean_mm": float(error_mm.mean()),
        "median_mm": float(np.median(error_mm)),
        "p95_mm": float(np.percentile(error_mm, 95)),
        "within_30mm": float(np.mean(error_mm <= 30)),
        "within_50mm": float(np.mean(error_mm <= 50)),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--student-root", type=Path, default=Path("datasets/H2O/student_state_mediapipe_64_32")
    )
    parser.add_argument(
        "--output", type=Path, default=Path("datasets/H2O/student_object_proxy_64_32")
    )
    args = parser.parse_args()

    train_x, train_center, train_target, train_groups, _, train_clip_ids = _load_rows(
        "train", 64, args.student_root
    )
    val_x, val_center, val_target, val_groups, val_object_ids, val_clip_ids = _load_rows(
        "val", 32, args.student_root
    )
    train_y = train_target - train_center

    alphas = (1e-3, 1e-2, 1e-1, 1.0, 10.0, 100.0, 1000.0)
    cv = {}
    folds = np.asarray(
        [int(hashlib.sha1(group.encode()).hexdigest()[:8], 16) % 5 for group in train_groups]
    )
    for alpha in alphas:
        errors = []
        for fold in range(5):
            fit = folds != fold
            test = folds == fold
            weights, mean, scale = _fit_ridge(train_x[fit], train_y[fit], alpha)
            prediction = train_center[test] + _predict(train_x[test], weights, mean, scale)
            errors.extend(np.linalg.norm(prediction - train_target[test], axis=1))
        cv[str(alpha)] = _summary(np.asarray(errors))
    best_alpha = min(alphas, key=lambda value: cv[str(value)]["mean_mm"])
    weights, mean, scale = _fit_ridge(train_x, train_y, best_alpha)
    prediction = val_center + _predict(val_x, weights, mean, scale)

    def motion_data(
        x: np.ndarray,
        center: np.ndarray,
        target: np.ndarray,
        clip_ids: list[str],
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        motion_x, hand_delta, target_delta = [], [], []
        for clip_id in dict.fromkeys(clip_ids):
            indices = np.flatnonzero(np.asarray(clip_ids) == clip_id)
            first = indices[0]
            for index in indices:
                center_delta = center[index] - center[first]
                motion_x.append(np.concatenate((x[index] - x[first], center_delta)))
                hand_delta.append(center_delta)
                target_delta.append(target[index] - target[first])
        return np.asarray(motion_x), np.asarray(hand_delta), np.asarray(target_delta)

    train_motion_x, _, train_motion_target = motion_data(
        train_x, train_center, train_target, train_clip_ids
    )
    val_motion_x, val_hand_delta, val_motion_target = motion_data(
        val_x, val_center, val_target, val_clip_ids
    )
    motion_cv = {}
    for alpha in alphas:
        errors = []
        for fold in range(5):
            fit = folds != fold
            test = folds == fold
            motion_weights, motion_mean, motion_scale = _fit_ridge(
                train_motion_x[fit], train_motion_target[fit], alpha
            )
            motion_prediction = _predict(
                train_motion_x[test], motion_weights, motion_mean, motion_scale
            )
            errors.extend(np.linalg.norm(motion_prediction - train_motion_target[test], axis=1))
        motion_cv[str(alpha)] = _summary(np.asarray(errors))
    best_motion_alpha = min(alphas, key=lambda value: motion_cv[str(value)]["mean_mm"])
    oof_motion_prediction = np.zeros_like(train_motion_target)
    for fold in range(5):
        fit = folds != fold
        test = folds == fold
        fold_weights, fold_mean, fold_scale = _fit_ridge(
            train_motion_x[fit], train_motion_target[fit], best_motion_alpha
        )
        oof_motion_prediction[test] = _predict(
            train_motion_x[test], fold_weights, fold_mean, fold_scale
        )
    gate_grid = {}
    for threshold_mm in (0, 5, 10, 20, 30, 50, 75):
        active = np.linalg.norm(oof_motion_prediction, axis=1) * 1000 >= threshold_mm
        for shrink in (0.25, 0.5, 0.75, 1.0):
            gated = oof_motion_prediction * active[:, None] * shrink
            key = f"threshold_mm={threshold_mm},shrink={shrink}"
            gate_grid[key] = _summary(np.linalg.norm(gated - train_motion_target, axis=1))
    best_gate = min(gate_grid, key=lambda key: gate_grid[key]["mean_mm"])
    gate_parts = dict(part.split("=") for part in best_gate.split(","))
    gate_threshold_mm = float(gate_parts["threshold_mm"])
    gate_shrink = float(gate_parts["shrink"])
    motion_weights, motion_mean, motion_scale = _fit_ridge(
        train_motion_x, train_motion_target, best_motion_alpha
    )
    motion_prediction = _predict(val_motion_x, motion_weights, motion_mean, motion_scale)
    motion_active = np.linalg.norm(motion_prediction, axis=1) * 1000 >= gate_threshold_mm
    gated_motion_prediction = motion_prediction * motion_active[:, None] * gate_shrink
    future_mask = np.ones(len(val_clip_ids), dtype=bool)
    for clip_id in dict.fromkeys(val_clip_ids):
        future_mask[val_clip_ids.index(clip_id)] = False
    motion_magnitude_mm = np.linalg.norm(val_motion_target, axis=1) * 1000

    def motion_comparison(mask: np.ndarray) -> dict[str, dict[str, float | int]]:
        return {
            "zero_motion": _summary(np.linalg.norm(val_motion_target[mask], axis=1)),
            "hand_centroid_delta": _summary(
                np.linalg.norm(val_hand_delta[mask] - val_motion_target[mask], axis=1)
            ),
            "ridge_hand_motion": _summary(
                np.linalg.norm(motion_prediction[mask] - val_motion_target[mask], axis=1)
            ),
            "gated_ridge_hand_motion": _summary(
                np.linalg.norm(gated_motion_prediction[mask] - val_motion_target[mask], axis=1)
            ),
        }

    # Recover simple geometry-only controls from the first 126 centered-coordinate
    # features plus each frame's global center.
    filled = train_x  # Keep the fitted representation documented in the artifact.
    del filled
    val_all_joint = val_center
    results = {
        "contract": {
            "train_target": "oracle object center used only to fit the proxy",
            "test_input": "exo-only triangulated hand joints and confidence",
            "object_identity_used": False,
            "future_ego_information_used": False,
        },
        "train_frames": int(len(train_x)),
        "val_frames": int(len(val_x)),
        "train_sequences": len(set(train_groups)),
        "val_sequences": len(set(val_groups)),
        "alpha_cv": cv,
        "best_alpha": best_alpha,
        "validation": {
            "hand_joint_centroid": _summary(np.linalg.norm(val_all_joint - val_target, axis=1)),
            "ridge_hand_configuration": _summary(np.linalg.norm(prediction - val_target, axis=1)),
        },
        "validation_by_object_id": {
            str(identifier): _summary(
                np.linalg.norm(prediction[val_object_ids == identifier] - val_target[val_object_ids == identifier], axis=1)
            )
            for identifier in np.unique(val_object_ids)
        },
        "motion": {
            "target": "object displacement relative to the first clip frame",
            "alpha_cv": motion_cv,
            "best_alpha": best_motion_alpha,
            "gate_cv": gate_grid,
            "best_gate": {
                "threshold_mm": gate_threshold_mm,
                "shrink": gate_shrink,
            },
            "validation": motion_comparison(np.ones(len(val_motion_target), dtype=bool)),
            "validation_future_frames": motion_comparison(future_mask),
            "validation_target_motion_over_20mm": motion_comparison(motion_magnitude_mm > 20),
            "validation_target_motion_over_50mm": motion_comparison(motion_magnitude_mm > 50),
        },
    }
    args.output.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        args.output / "ridge_model.npz",
        weights=weights.astype(np.float32),
        feature_mean=mean.astype(np.float32),
        feature_scale=scale.astype(np.float32),
        alpha=np.asarray(best_alpha, dtype=np.float32),
        motion_weights=motion_weights.astype(np.float32),
        motion_feature_mean=motion_mean.astype(np.float32),
        motion_feature_scale=motion_scale.astype(np.float32),
        motion_alpha=np.asarray(best_motion_alpha, dtype=np.float32),
    )
    np.savez_compressed(
        args.output / "val_predictions.npz",
        prediction_world_m=prediction.astype(np.float32),
        target_world_m=val_target.astype(np.float32),
        object_id=val_object_ids,
        motion_prediction_m=motion_prediction.astype(np.float32),
        gated_motion_prediction_m=gated_motion_prediction.astype(np.float32),
        motion_target_m=val_motion_target.astype(np.float32),
    )
    (args.output / "summary.json").write_text(json.dumps(results, indent=2) + "\n")
    print(json.dumps(results, indent=2))


if __name__ == "__main__":
    main()
