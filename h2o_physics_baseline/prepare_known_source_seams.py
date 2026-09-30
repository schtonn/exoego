#!/usr/bin/env python3
"""Build compact masks for boundaries between observed background sources.

The renderer's raw seam mask includes boundaries around every small projection
island and around unknown pixels.  That is useful as a diagnostic, but too broad
for a video inpainter.  This script locally consolidates the three *observed*
source labels and emits only known-to-known boundaries.  The original dynamic
guard is retained by intersecting with the renderer's seam mask.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import cv2
import numpy as np
from PIL import Image


def load_u8(path: Path) -> np.ndarray:
    return np.asarray(Image.open(path).convert("L"), dtype=np.uint8)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model-input-root", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--majority-size", type=int, default=5)
    parser.add_argument("--seam-width", type=int, default=5)
    parser.add_argument(
        "--minimum-votes",
        type=int,
        default=5,
        help="Minimum local support before changing a known provenance label.",
    )
    args = parser.parse_args()
    if args.majority_size % 2 != 1 or args.seam_width % 2 != 1:
        raise ValueError("majority-size and seam-width must be odd")

    mask_root = args.output_root / "known_seam_masks"
    label_root = args.output_root / "clean_provenance_labels"
    mask_root.mkdir(parents=True, exist_ok=True)
    label_root.mkdir(parents=True, exist_ok=True)

    records = []
    label_paths = sorted((args.model_input_root / "provenance_labels").glob("*.png"))
    kernel = np.ones((args.seam_width, args.seam_width), dtype=np.uint8)
    for label_path in label_paths:
        labels = load_u8(label_path)
        raw_seam = load_u8(args.model_input_root / "seam_masks" / label_path.name) > 0
        known = (labels >= 1) & (labels <= 3)

        # Consolidate only labels 1--3. Unknown label 4 remains unknown and can
        # never become apparent evidence through this cleanup operation.
        counts = np.stack(
            [
                cv2.boxFilter(
                    (labels == value).astype(np.float32),
                    -1,
                    (args.majority_size, args.majority_size),
                    normalize=False,
                )
                for value in (1, 2, 3)
            ]
        )
        winner = np.argmax(counts, axis=0).astype(np.uint8) + 1
        clean = labels.copy()
        replace = known & (counts.max(axis=0) >= args.minimum_votes)
        clean[replace] = winner[replace]

        # Mark only transitions for which both sides contain real observations.
        edge = np.zeros(labels.shape, dtype=bool)
        vertical = (
            (clean[1:, :] != clean[:-1, :])
            & (clean[1:, :] >= 1)
            & (clean[1:, :] <= 3)
            & (clean[:-1, :] >= 1)
            & (clean[:-1, :] <= 3)
        )
        edge[1:, :] |= vertical
        edge[:-1, :] |= vertical
        horizontal = (
            (clean[:, 1:] != clean[:, :-1])
            & (clean[:, 1:] >= 1)
            & (clean[:, 1:] <= 3)
            & (clean[:, :-1] >= 1)
            & (clean[:, :-1] <= 3)
        )
        edge[:, 1:] |= horizontal
        edge[:, :-1] |= horizontal
        seam = cv2.dilate(edge.astype(np.uint8), kernel) > 0
        # The exported raw seam already excludes the enlarged arm/hand support.
        seam &= raw_seam & known

        Image.fromarray(clean, "L").save(label_root / label_path.name)
        Image.fromarray(seam.astype(np.uint8) * 255, "L").save(
            mask_root / label_path.name
        )
        records.append(
            {
                "frame": label_path.stem,
                "raw_seam_fraction": float(raw_seam.mean()),
                "known_seam_fraction": float(seam.mean()),
                "changed_label_fraction": float((clean != labels).mean()),
            }
        )

    summary = {
        "majority_size": args.majority_size,
        "seam_width": args.seam_width,
        "minimum_votes": args.minimum_votes,
        "mean_raw_seam_fraction": float(
            np.mean([record["raw_seam_fraction"] for record in records])
        ),
        "mean_known_seam_fraction": float(
            np.mean([record["known_seam_fraction"] for record in records])
        ),
        "mean_changed_label_fraction": float(
            np.mean([record["changed_label_fraction"] for record in records])
        ),
        "frames": records,
    }
    (args.output_root / "manifest.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(args.output_root / "manifest.json")


if __name__ == "__main__":
    main()
