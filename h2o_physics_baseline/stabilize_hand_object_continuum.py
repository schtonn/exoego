#!/usr/bin/env python3
"""Enforce short-horizon hand/object support continuity on an ego video.

The pass never reads future ego ground truth.  It aligns nearby predicted frames
with optical flow, restores only dynamic pixels supported by another time, and
keeps arm/object classes separate while treating their union as one interaction
continuum for hole and contact diagnostics.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import cv2
import numpy as np
from PIL import Image


def load_rgb(path: Path, size: int) -> np.ndarray:
    return np.asarray(
        Image.open(path).convert("RGB").resize((size, size), Image.Resampling.BILINEAR)
    ).astype(np.float32) / 255.0


def load_mask(path: Path, size: int) -> np.ndarray:
    return np.asarray(
        Image.open(path).convert("L").resize((size, size), Image.Resampling.NEAREST)
    ) > 127


def dilate(mask: np.ndarray, radius: int) -> np.ndarray:
    kernel = cv2.getStructuringElement(
        cv2.MORPH_ELLIPSE, (2 * radius + 1, 2 * radius + 1)
    )
    return cv2.dilate(mask.astype(np.uint8), kernel) > 0


def close(mask: np.ndarray, radius: int) -> np.ndarray:
    kernel = cv2.getStructuringElement(
        cv2.MORPH_ELLIPSE, (2 * radius + 1, 2 * radius + 1)
    )
    return cv2.morphologyEx(mask.astype(np.uint8), cv2.MORPH_CLOSE, kernel) > 0


def flow_sample(
    current_gray: np.ndarray,
    neighbor_gray: np.ndarray,
    neighbor_value: np.ndarray,
    nearest: bool,
) -> np.ndarray:
    """Sample a neighboring frame at correspondences for current-frame pixels."""
    flow = cv2.calcOpticalFlowFarneback(
        current_gray,
        neighbor_gray,
        None,
        pyr_scale=0.5,
        levels=3,
        winsize=21,
        iterations=4,
        poly_n=7,
        poly_sigma=1.5,
        flags=0,
    )
    height, width = current_gray.shape
    yy, xx = np.mgrid[:height, :width].astype(np.float32)
    interpolation = cv2.INTER_NEAREST if nearest else cv2.INTER_LINEAR
    return cv2.remap(
        neighbor_value,
        xx + flow[..., 0],
        yy + flow[..., 1],
        interpolation,
        borderMode=cv2.BORDER_CONSTANT,
        borderValue=0,
    )


def centroid(mask: np.ndarray) -> np.ndarray:
    ys, xs = np.nonzero(mask)
    if len(xs) == 0:
        return np.array([np.nan, np.nan], dtype=np.float64)
    return np.array([xs.mean(), ys.mean()], dtype=np.float64)


def finite_interpolate(values: np.ndarray) -> np.ndarray:
    result = values.copy()
    x = np.arange(len(result))
    for channel in range(result.shape[1]):
        valid = np.isfinite(result[:, channel])
        if np.any(valid):
            result[~valid, channel] = np.interp(
                x[~valid], x[valid], result[valid, channel]
            )
        else:
            result[:, channel] = 0
    return result


def centroid_acceleration(values: np.ndarray) -> float:
    values = finite_interpolate(values)
    if len(values) < 3:
        return 0.0
    return float(np.linalg.norm(np.diff(values, n=2, axis=0), axis=1).mean())


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model-input-root", type=Path, required=True)
    parser.add_argument("--base-frames", type=Path, required=True)
    parser.add_argument("--target-rgb-dir", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--temporal-radius", type=int, default=2)
    parser.add_argument("--support-votes", type=int, default=2)
    parser.add_argument("--search-radius", type=int, default=8)
    args = parser.parse_args()

    manifest = json.loads((args.model_input_root / "manifest.json").read_text())
    size = int(manifest["image_size"])
    records = manifest["frames"]
    names = [f"{int(record['index']):06d}.png" for record in records]
    frames = np.stack([load_rgb(args.base_frames / name, size) for name in names])
    grays = np.stack(
        [cv2.cvtColor(np.rint(frame * 255).astype(np.uint8), cv2.COLOR_RGB2GRAY)
         for frame in frames]
    )
    arm = np.stack(
        [load_mask(args.model_input_root / "arm_masks" / name, size) for name in names]
    )
    objects = np.stack(
        [load_mask(args.model_input_root / "object_masks" / name, size) for name in names]
    )
    raw = arm | objects

    output_root = args.output_root
    directories = {
        name: output_root / name
        for name in (
            "frames", "raw_masks", "stable_masks", "restored_masks",
            "stable_arm_masks", "stable_object_masks",
        )
    }
    for directory in directories.values():
        directory.mkdir(parents=True, exist_ok=True)

    stabilized_frames: list[np.ndarray] = []
    stabilized_masks: list[np.ndarray] = []
    stabilized_arm_masks: list[np.ndarray] = []
    stabilized_object_masks: list[np.ndarray] = []
    restored_masks: list[np.ndarray] = []
    raw_flow_disagreement = []
    stable_flow_disagreement = []

    for index in range(len(frames)):
        # Object support comes from the rigid SE(3) layer and is locked exactly;
        # even a morphological close can move its centroid at this resolution.
        current_object = objects[index].copy()
        current_arm = close(arm[index], 1) & (~current_object)
        current = current_arm | current_object
        votes = current.astype(np.int32)
        arm_votes = current_arm.astype(np.int32)
        object_votes = current_object.astype(np.int32)
        rgb_sum = np.zeros_like(frames[index], dtype=np.float32)
        rgb_weight = np.zeros((size, size), dtype=np.float32)
        warped_previous = None
        for neighbor in range(
            max(0, index - args.temporal_radius),
            min(len(frames), index + args.temporal_radius + 1),
        ):
            if neighbor == index:
                continue
            warped_arm = flow_sample(
                grays[index], grays[neighbor], arm[neighbor].astype(np.uint8), True
            ) > 0
            warped_object = flow_sample(
                grays[index], grays[neighbor], objects[neighbor].astype(np.uint8), True
            ) > 0
            warped_mask = warped_arm | warped_object
            warped_rgb = flow_sample(
                grays[index], grays[neighbor], frames[neighbor], False
            ).astype(np.float32)
            votes += warped_mask.astype(np.int32)
            arm_votes += warped_arm.astype(np.int32)
            object_votes += warped_object.astype(np.int32)
            rgb_sum += warped_rgb * warped_mask[..., None]
            rgb_weight += warped_mask.astype(np.float32)
            if neighbor == index - 1:
                warped_previous = warped_mask

        # The annotated/object-tracked SE(3) layer is the authority for object
        # location. Optical flow may restore a disappearing arm/contact pixel,
        # but must not expand the rigid object and move its centroid.
        temporal = arm_votes >= args.support_votes
        # Restore only near the current interaction body. This closes transient
        # holes without leaving a trail when the hand or object truly moves.
        near_current = dilate(current, args.search_radius)
        restored = (~current) & temporal & near_current

        # If hand and object nearly touch, permit only temporally observed pixels
        # in the narrow contact corridor. No new texture is hallucinated here.
        if np.any(arm[index]) and np.any(objects[index]):
            contact_corridor = (
                dilate(arm[index], 5) & dilate(objects[index], 5) & (~current)
            )
            restored |= contact_corridor & (votes >= 1) & (rgb_weight > 0)

        stable = current | restored
        stable_arm = current_arm | restored
        stable_object = current_object & (~stable_arm)
        output = frames[index].copy()
        supported = restored & (rgb_weight > 0)
        output[supported] = (
            rgb_sum[supported] / np.maximum(rgb_weight[supported, None], 1.0)
        )

        stabilized_frames.append(np.clip(output, 0, 1))
        stabilized_masks.append(stable)
        stabilized_arm_masks.append(stable_arm)
        stabilized_object_masks.append(stable_object)
        restored_masks.append(restored)
        if warped_previous is not None:
            raw_flow_disagreement.append(float(np.logical_xor(current, warped_previous).mean()))

        Image.fromarray(np.rint(output * 255).astype(np.uint8), "RGB").save(
            directories["frames"] / names[index]
        )
        for directory_name, mask in (
            ("raw_masks", raw[index]),
            ("stable_masks", stable),
            ("restored_masks", restored),
            ("stable_arm_masks", stable_arm),
            ("stable_object_masks", stable_object),
        ):
            Image.fromarray(mask.astype(np.uint8) * 255, "L").save(
                directories[directory_name] / names[index]
            )

    stable_stack = np.stack(stabilized_masks)
    stable_arm_stack = np.stack(stabilized_arm_masks)
    stable_object_stack = np.stack(stabilized_object_masks)
    restored_stack = np.stack(restored_masks)
    for index in range(1, len(frames)):
        warped_previous = flow_sample(
            grays[index], grays[index - 1], stable_stack[index - 1].astype(np.uint8), True
        ) > 0
        stable_flow_disagreement.append(
            float(np.logical_xor(stable_stack[index], warped_previous).mean())
        )

    targets = np.stack(
        [
            load_rgb(
                args.target_rgb_dir / f"{int(record['dataset_frame']):06d}.png", size
            )
            for record in records
        ]
    )
    stable_frames = np.stack(stabilized_frames)
    evaluation_region = stable_stack | raw
    baseline_error = np.abs(frames - targets).mean(axis=-1)
    stable_error = np.abs(stable_frames - targets).mean(axis=-1)
    raw_centroids = np.stack([centroid(mask) for mask in raw])
    stable_centroids = np.stack([centroid(mask) for mask in stable_stack])
    raw_arm_centroids = np.stack([centroid(mask) for mask in arm])
    stable_arm_centroids = np.stack([centroid(mask) for mask in stable_arm_stack])
    raw_object_centroids = np.stack([centroid(mask) for mask in objects])
    stable_object_centroids = np.stack(
        [centroid(mask) for mask in stable_object_stack]
    )
    metrics = {
        "protocol": (
            "offline symmetric temporal support; no future ego ground truth used for inference"
        ),
        "raw_interaction_fraction": float(raw.mean()),
        "stabilized_interaction_fraction": float(stable_stack.mean()),
        "restored_fraction": float(restored_stack.mean()),
        "all_l1_before": float(baseline_error.mean()),
        "all_l1_after": float(stable_error.mean()),
        "interaction_union_l1_before": float(baseline_error[evaluation_region].mean()),
        "interaction_union_l1_after": float(stable_error[evaluation_region].mean()),
        "flow_aligned_mask_xor_before": float(np.mean(raw_flow_disagreement)),
        "flow_aligned_mask_xor_after": float(np.mean(stable_flow_disagreement)),
        "centroid_acceleration_px_per_frame2_before": centroid_acceleration(raw_centroids),
        "centroid_acceleration_px_per_frame2_after": centroid_acceleration(stable_centroids),
        "arm_centroid_acceleration_px_per_frame2_before": centroid_acceleration(raw_arm_centroids),
        "arm_centroid_acceleration_px_per_frame2_after": centroid_acceleration(stable_arm_centroids),
        "object_centroid_acceleration_px_per_frame2_before": centroid_acceleration(raw_object_centroids),
        "object_centroid_acceleration_px_per_frame2_after": centroid_acceleration(stable_object_centroids),
        "temporal_delta_l1_before": float(
            np.abs(np.diff(frames, axis=0) - np.diff(targets, axis=0)).mean()
        ),
        "temporal_delta_l1_after": float(
            np.abs(np.diff(stable_frames, axis=0) - np.diff(targets, axis=0)).mean()
        ),
        "parameters": {
            "temporal_radius": args.temporal_radius,
            "support_votes": args.support_votes,
            "search_radius": args.search_radius,
        },
    }
    (output_root / "metrics.json").write_text(
        json.dumps(metrics, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )
    print(json.dumps(metrics, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
