#!/usr/bin/env python3
"""Prepare a limb-only second pass without regenerating an accepted background.

The base is the previous completed video.  Strictly observed arm/object pixels
from the new geometry render are copied onto it, while legacy arm pixels that
are no longer supported and the newly predicted missing-limb envelope are
authorized for a small ProPainter pass.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import cv2
import numpy as np


def read_image(path: Path) -> np.ndarray:
    value = cv2.imread(str(path), cv2.IMREAD_COLOR)
    if value is None:
        raise FileNotFoundError(path)
    return value


def read_mask(path: Path, shape: tuple[int, int]) -> np.ndarray:
    if not path.exists():
        return np.zeros(shape, dtype=bool)
    return read_image(path)[:, :, 0] > 127


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--new-model-input", type=Path, required=True)
    parser.add_argument("--legacy-model-input", type=Path, required=True)
    parser.add_argument("--legacy-completed-frames", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--support-dilation", type=int, default=2)
    args = parser.parse_args()

    frame_root = args.output_root / "input_frames"
    mask_root = args.output_root / "masks"
    frame_root.mkdir(parents=True, exist_ok=True)
    mask_root.mkdir(parents=True, exist_ok=True)

    filenames = sorted((args.new_model_input / "input_frames").glob("*.png"))
    kernel_size = args.support_dilation * 2 + 1
    kernel = np.ones((kernel_size, kernel_size), np.uint8)
    fractions: list[float] = []
    for path in filenames:
        filename = path.name
        new_geometry = read_image(path)
        base = read_image(args.legacy_completed_frames / filename)
        shape = base.shape[:2]
        new_arm = read_mask(args.new_model_input / "arm_masks" / filename, shape)
        new_object = read_mask(args.new_model_input / "object_masks" / filename, shape)
        completion = read_mask(
            args.new_model_input / "dynamic_completion_masks" / filename, shape
        )
        legacy_arm = read_mask(
            args.legacy_model_input / "arm_masks" / filename, shape
        )

        observed = new_arm | new_object
        observed_guard = cv2.dilate(observed.astype(np.uint8), kernel) > 0
        unsupported_legacy_arm = legacy_arm & (~observed_guard)
        authorized = (completion | unsupported_legacy_arm) & (~observed)

        # Give the video model the reliable strict geometry as context while
        # leaving the already accepted background untouched.
        prepared = base.copy()
        prepared[observed] = new_geometry[observed]
        cv2.imwrite(str(frame_root / filename), prepared)
        cv2.imwrite(str(mask_root / filename), authorized.astype(np.uint8) * 255)
        fractions.append(float(authorized.mean()))

    print(f"frames={len(filenames)} mean_authorized_fraction={np.mean(fractions):.6f}")


if __name__ == "__main__":
    main()
