#!/usr/bin/env python3
"""Cache Depth Anything V2 metric depth for mount-prior calibration frames."""

from __future__ import annotations

import argparse
import csv
import json
from collections import OrderedDict
from pathlib import Path
import sys

import cv2
import numpy as np
import torch


def load_combined_rows(index: Path, split: str, max_samples: int) -> list[dict[str, str]]:
    with index.open(encoding="utf-8") as handle:
        rows = [row for row in csv.DictReader(handle) if row["split"] == split]
    grouped: OrderedDict[str, list[dict[str, str]]] = OrderedDict()
    for row in rows:
        grouped.setdefault(row["clip_id"], []).append(row)
    result = []
    for clip_rows in grouped.values():
        by_camera = {row["source_camera"]: row for row in clip_rows}
        if not all(f"cam{index}" in by_camera for index in range(4)):
            continue
        representative = dict(by_camera["cam0"])
        representative["source_rgb_dirs"] = json.dumps(
            [by_camera[f"cam{index}"]["source_rgb_dir"] for index in range(4)]
        )
        result.append(representative)
    if max_samples < len(result):
        selected = np.linspace(0, len(result) - 1, max_samples, dtype=np.int64)
        result = [result[int(index)] for index in selected]
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--index",
        type=Path,
        default=Path("datasets/H2O/oracle_state/paired_physical_clips.csv"),
    )
    parser.add_argument("--calibration-split", default="train")
    parser.add_argument("--target-split", default="val")
    parser.add_argument("--calibration-samples", type=int, default=32)
    parser.add_argument("--target-max-samples", type=int, default=32)
    parser.add_argument("--target-pair-id", action="append", required=True)
    parser.add_argument("--camera-indices", default="0,1,2,3")
    parser.add_argument("--input-size", type=int, default=392)
    parser.add_argument(
        "--repository", type=Path, default=Path("third_party/Depth-Anything-V2")
    )
    parser.add_argument(
        "--weights",
        type=Path,
        default=Path(
            "models/depth_anything_v2/"
            "depth_anything_v2_metric_hypersim_vits.pth"
        ),
    )
    parser.add_argument(
        "--output-root",
        type=Path,
        default=Path("datasets/H2O/experiments/depth_anything_mount_frames"),
    )
    args = parser.parse_args()
    camera_indices = tuple(
        int(value.strip()) for value in args.camera_indices.split(",") if value.strip()
    )

    sys.path.insert(0, str(args.repository / "metric_depth"))
    from depth_anything_v2.dpt import DepthAnythingV2

    model = DepthAnythingV2(
        encoder="vits", features=64, out_channels=[48, 96, 192, 384], max_depth=20
    )
    model.load_state_dict(torch.load(args.weights, map_location="cpu", weights_only=True))
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = model.to(device).eval()

    calibration = load_combined_rows(
        args.index, args.calibration_split, args.calibration_samples
    )
    targets = load_combined_rows(args.index, args.target_split, args.target_max_samples)
    requested = set(args.target_pair_id)
    targets = [row for row in targets if row["pair_id"] in requested]
    missing = requested - {row["pair_id"] for row in targets}
    if missing:
        raise ValueError(f"Target pair ids not selected: {sorted(missing)}")

    rows = calibration + targets
    records = []
    for row_index, row in enumerate(rows):
        frame = int(row["start_frame"])
        roots = [Path(value).parent for value in json.loads(row["source_rgb_dirs"])]
        for camera_index in camera_indices:
            camera_root = roots[camera_index]
            rgb_path = camera_root / "rgb" / f"{frame:06d}.png"
            relative = Path(row["sequence"]) / f"cam{camera_index}" / f"{frame:06d}.npy"
            output = args.output_root / relative
            output.parent.mkdir(parents=True, exist_ok=True)
            if output.exists():
                depth = np.load(output)
                status = "cached"
            else:
                image = cv2.imread(str(rgb_path))
                if image is None:
                    raise FileNotFoundError(rgb_path)
                with torch.inference_mode():
                    depth = model.infer_image(image, input_size=args.input_size).astype(np.float32)
                np.save(output, depth)
                status = "written"
            records.append(
                {
                    "pair_id": row["pair_id"],
                    "sequence": row["sequence"],
                    "camera_index": camera_index,
                    "frame": frame,
                    "path": str(output),
                    "depth_min_m": float(np.nanmin(depth)),
                    "depth_median_m": float(np.nanmedian(depth)),
                    "depth_max_m": float(np.nanmax(depth)),
                }
            )
            print(
                json.dumps(
                    {
                        "row": row_index,
                        "pair_id": row["pair_id"],
                        "camera": camera_index,
                        "status": status,
                    }
                ),
                flush=True,
            )
    manifest = {
        "model": "Depth Anything V2 Metric Hypersim Small",
        "weights": str(args.weights),
        "input_size": args.input_size,
        "device": device,
        "depth_unit": "metres",
        "calibration_split": args.calibration_split,
        "calibration_samples": len(calibration),
        "target_split": args.target_split,
        "target_pair_ids": sorted(requested),
        "camera_indices": list(camera_indices),
        "records": records,
    }
    args.output_root.mkdir(parents=True, exist_ok=True)
    (args.output_root / "manifest.json").write_text(
        json.dumps(manifest, indent=2, ensure_ascii=False) + "\n"
    )


if __name__ == "__main__":
    main()
