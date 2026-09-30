#!/usr/bin/env python3
"""Render the deprecated low-resolution student branch for diagnostics only.

This is not the current layered physics renderer and must not be presented as a
sample of the final method.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import subprocess

import numpy as np
import torch
from PIL import Image, ImageDraw
from torch.utils.data._utils.collate import default_collate

from h2o_physics_baseline.dataset import H2OPhysicalClipDataset
from h2o_physics_baseline.model import (
    PhysicalEgoVideoPredictor,
    _student_disocclusion_fill,
    _student_flow_warp,
)
from h2o_physics_baseline.train import to_device
from h2o_physics_baseline.visualize_student_warp_fill import (
    INPUT_BORDER,
    INTERMEDIATE_BORDER,
    PREDICTION_BORDER,
    TARGET_BORDER,
    font,
)


DEFAULT_PAIR = "subject3_k2_4_000288_000351_cam0_to_cam4"
DEFAULT_SMALL = Path(
    "datasets/H2O/experiments/student_warp_fill_v2/fill_seed25/"
    "multiview_anchored_student_flow_warp_fill_local_gate31/best.pt"
)
DEFAULT_LARGE_2D = Path(
    "datasets/H2O/experiments/scale_data_256_f8_seed25/"
    "multiview_anchored_student_flow_warp_fill_local_gate31/best.pt"
)
DEFAULT_TEMPORAL = Path(
    "datasets/H2O/experiments/scale_data_256_f8_seed25/"
    "multiview_anchored_student_flow_warp_fill_temporal_small_gate31/best.pt"
)


def load_model(checkpoint_path: Path, device: torch.device) -> PhysicalEgoVideoPredictor:
    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    model = PhysicalEgoVideoPredictor(condition_mode=checkpoint["config"]["condition_mode"])
    model.load_state_dict(checkpoint["model"])
    return model.to(device).eval()


def tensor_image(value: torch.Tensor, size: int) -> Image.Image:
    array = value.detach().float().cpu().numpy()
    array = np.moveaxis(np.clip(array, 0, 1), 0, -1)
    return Image.fromarray(np.rint(array * 255).astype(np.uint8), "RGB").resize(
        (size, size), Image.Resampling.NEAREST
    )


def raw_image(directory: Path, frame: int, size: int) -> Image.Image:
    return Image.open(directory / f"{frame:06d}.png").convert("RGB").resize(
        (size, size), Image.Resampling.LANCZOS
    )


def title_tile(
    image: Image.Image,
    title: str,
    size: int,
    border: tuple[int, int, int],
) -> Image.Image:
    tile = Image.new("RGB", (size, size + 44), "white")
    tile.paste(image.resize((size, size), Image.Resampling.NEAREST), (0, 0))
    draw = ImageDraw.Draw(tile)
    draw.rectangle((0, 0, size - 1, size - 1), outline=border, width=5)
    draw.text((8, size + 5), title, font=font(20), fill=(15, 23, 42))
    return tile


def future_l1(prediction: torch.Tensor, target: torch.Tensor) -> float:
    return float((prediction[:, 1:] - target[:, 1:]).abs().mean())


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--allow-deprecated-student",
        action="store_true",
        help="Explicit acknowledgement that this is not the current method.",
    )
    parser.add_argument("--pair-id", default=DEFAULT_PAIR)
    parser.add_argument(
        "--student-state-root",
        type=Path,
        default=Path("datasets/H2O/student_state_mediapipe_k2_4_full64"),
    )
    parser.add_argument("--small-checkpoint", type=Path, default=DEFAULT_SMALL)
    parser.add_argument("--large-2d-checkpoint", type=Path, default=DEFAULT_LARGE_2D)
    parser.add_argument("--temporal-checkpoint", type=Path, default=DEFAULT_TEMPORAL)
    parser.add_argument(
        "--output",
        type=Path,
        default=Path(
            "datasets/H2O/experiments/deprecated_student_scale_visual/"
            "model_scale_k2_4_full64_diagnostic.mp4"
        ),
    )
    parser.add_argument("--device", default="cuda:0" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--size", type=int, default=256)
    parser.add_argument("--fps", type=int, default=15)
    args = parser.parse_args()
    if not args.allow_deprecated_student:
        raise SystemExit(
            "Refusing to render the obsolete student_warp_fill branch. "
            "Use train_final_layer_fusion.py with final layered exports. "
            "For historical diagnostics only, pass --allow-deprecated-student."
        )

    dataset = H2OPhysicalClipDataset(
        split="val",
        frames_per_clip=64,
        image_size=64,
        source_cameras=("cam0", "cam1", "cam2", "cam3"),
        combine_source_cameras=True,
        student_state_root=args.student_state_root,
        student_flow=True,
    )
    selected = [index for index, row in enumerate(dataset.rows) if row["pair_id"] == args.pair_id]
    if len(selected) != 1:
        raise ValueError(f"Expected one row for {args.pair_id}, found {len(selected)}")
    item = dataset[selected[0]]
    row = dataset.rows[selected[0]]
    cpu_batch = default_collate([item])
    device = torch.device(args.device)
    batch = to_device(cpu_batch, device)

    models = {
        "small": load_model(args.small_checkpoint, device),
        "large_2d": load_model(args.large_2d_checkpoint, device),
        "temporal": load_model(args.temporal_checkpoint, device),
    }
    with torch.inference_mode():
        predictions = {}
        for name, model in models.items():
            predictions[name], _ = model(
                batch["source"],
                batch["physical_vector"],
                batch["physical_maps"],
                batch["coarse_rgb"],
                batch["coarse_depth"],
                batch["visibility_mask"],
                batch["ego_anchor"],
                batch["initial_ego_maps"],
                batch["source_hand_maps"],
                batch["source_joint_uv"],
                batch["source_joint_confidence"],
                batch["target_joint_uv"],
                batch["target_joint_confidence"],
            )
        warped = _student_flow_warp(batch["ego_anchor"], batch["initial_ego_maps"])
        deterministic, _, _ = _student_disocclusion_fill(
            batch["ego_anchor"], warped, batch["initial_ego_maps"]
        )

    source_directories = [Path(value) for value in json.loads(row["source_rgb_dirs"])]
    target_directory = Path(row["target_rgb_dir"])
    frame_numbers = item["frame_numbers"].tolist()
    size, gap, title_height = args.size, 8, 44
    tile_height = size + title_height
    canvas_width = 5 * size + 4 * gap
    canvas_height = 2 * tile_height + gap

    args.output.parent.mkdir(parents=True, exist_ok=True)
    command = [
        "ffmpeg", "-y", "-loglevel", "error", "-f", "rawvideo", "-pix_fmt", "rgb24",
        "-s", f"{canvas_width}x{canvas_height}", "-r", str(args.fps), "-i", "-", "-an",
        "-c:v", "libx264", "-preset", "slow", "-crf", "15", "-pix_fmt", "yuv420p",
        "-movflags", "+faststart", str(args.output),
    ]
    process = subprocess.Popen(command, stdin=subprocess.PIPE)
    assert process.stdin is not None
    last_canvas = None
    for time_index, frame_number in enumerate(frame_numbers):
        panels = [
            title_tile(raw_image(source_directories[view], frame_number, size), f"cam{view}", size, INPUT_BORDER)
            for view in range(4)
        ]
        panels.append(
            title_tile(raw_image(target_directory, frame_numbers[0], size), "ego首帧", size, INPUT_BORDER)
        )
        panels.extend(
            (
                title_tile(tensor_image(deterministic[0, time_index], size), "确定性warp/fill", size, INTERMEDIATE_BORDER),
                title_tile(tensor_image(predictions["small"][0, time_index], size), "64-clip逐帧头", size, PREDICTION_BORDER),
                title_tile(tensor_image(predictions["large_2d"][0, time_index], size), "256-clip逐帧头", size, PREDICTION_BORDER),
                title_tile(tensor_image(predictions["temporal"][0, time_index], size), "256-clip时空头", size, PREDICTION_BORDER),
                title_tile(raw_image(target_directory, frame_number, size), "未来ego真值", size, TARGET_BORDER),
            )
        )
        canvas = Image.new("RGB", (canvas_width, canvas_height), (241, 245, 249))
        for panel_index, panel in enumerate(panels):
            panel_row, column = divmod(panel_index, 5)
            canvas.paste(panel, (column * (size + gap), panel_row * (tile_height + gap)))
        process.stdin.write(np.asarray(canvas, dtype=np.uint8).tobytes())
        last_canvas = canvas
    process.stdin.close()
    if process.wait() != 0:
        raise RuntimeError("ffmpeg failed")
    assert last_canvas is not None
    poster = args.output.with_name(args.output.stem + "_poster.png")
    last_canvas.save(poster)
    metrics = {
        "pair_id": args.pair_id,
        "frames": len(frame_numbers),
        "future_l1": {
            "deterministic": future_l1(deterministic, batch["target"]),
            **{name: future_l1(value, batch["target"]) for name, value in predictions.items()},
        },
    }
    metrics_path = args.output.with_suffix(".json")
    metrics_path.write_text(json.dumps(metrics, indent=2) + "\n", encoding="utf-8")
    print(args.output)
    print(poster)
    print(metrics_path)


if __name__ == "__main__":
    main()
