#!/usr/bin/env python3
"""Learn only the authorized terminal fusion of the final layered pipeline.

This module deliberately does not reconstruct the ego frame from scratch.  It
consumes exports from ``render_causal_video_background_split.py``. A model may
fully edit explicit repair pixels and make smaller reliability-weighted
corrections to transported pixels, while keeping every correction bounded.
"""

from __future__ import annotations

import csv
import json
from dataclasses import dataclass
from pathlib import Path

import numpy as np
from PIL import Image
import torch
from torch import nn
from torch.utils.data import Dataset
import torch.nn.functional as F

from h2o_physics_baseline.protocols import validate_layered_manifest


RGB_STREAMS = (
    "input_frames",
    "proposal_frames",
    "geometry_frames",
    "background_frames",
    "anchor_layer_frames",
    "exo_layer_frames",
    "object_layer_frames",
)
MASK_STREAMS = (
    "repair_masks",
    "unknown_masks",
    "seam_masks",
    "foreground_masks",
    "arm_masks",
    "object_masks",
)
PROVENANCE_CLASSES = 6
INPUT_CHANNELS = len(RGB_STREAMS) * 3 + len(MASK_STREAMS) + PROVENANCE_CLASSES
SOFT_CORRECTION_LIMIT = 0.25


def _rgb(path: Path, size: tuple[int, int] | None = None) -> torch.Tensor:
    image = Image.open(path).convert("RGB")
    if size is not None and image.size != size:
        image = image.resize(size, Image.Resampling.BILINEAR)
    value = np.asarray(image, dtype=np.float32) / 255.0
    return torch.from_numpy(value).permute(2, 0, 1)


def _mask(path: Path) -> torch.Tensor:
    value = np.asarray(Image.open(path).convert("L"), dtype=np.float32) / 255.0
    return torch.from_numpy(value)[None]


@dataclass(frozen=True)
class LayeredClip:
    root: Path
    model_root: Path
    pair_id: str
    target_rgb_dir: Path
    dataset_frames: tuple[int, ...]
    split: str
    pose_confidences: tuple[float, ...]


def discover_clip(
    root: Path,
    index_path: Path,
    require_full_protocol: bool = True,
) -> LayeredClip:
    model_root = root / "model_input"
    if not (model_root / "manifest.json").exists():
        # The first mount-prior exports predate the model_input wrapper and
        # store the same streams directly below the scene root.
        model_root = root
    manifest_path = model_root / "manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    required_protocol = {
        "source_camera_indices": [0, 1, 2, 3],
        "ego_anchor_enabled": True,
        "annotated_exo_object": True,
        "annotated_exo_object_mode": "anchor_warp",
    }
    mismatches = {
        key: {"required": expected, "found": manifest.get(key)}
        for key, expected in required_protocol.items()
        if manifest.get(key) != expected
    }
    if require_full_protocol and mismatches:
        raise ValueError(
            f"{root} is not a current full4+anchor+physical-object export: {mismatches}"
        )
    if require_full_protocol:
        validate_layered_manifest(
            manifest, protocol="anchored_gt_mount_annotated_object"
        )
    elif "input_contract" in manifest:
        validate_layered_manifest(manifest)
    with index_path.open(encoding="utf-8") as handle:
        rows = [row for row in csv.DictReader(handle) if row["pair_id"] == manifest["pair_id"]]
    if len(rows) != 1:
        raise ValueError(f"Expected one index row for {manifest['pair_id']}, found {len(rows)}")
    frames = tuple(int(record["dataset_frame"]) for record in manifest["frames"])
    expected_frames = tuple(
        range(int(rows[0]["start_frame"]), int(rows[0]["end_frame"]) + 1)
    )
    if frames != expected_frames:
        raise ValueError(f"Manifest frames do not match canonical index for {manifest['pair_id']}")
    contract = manifest.get("input_contract", {})
    if contract and contract.get("dataset_split") != rows[0]["split"]:
        raise ValueError(
            "Manifest split does not match canonical index: "
            f"{contract.get('dataset_split')!r} != {rows[0]['split']!r}"
        )
    expected_names = {f"{index:06d}.png" for index in range(len(frames))}
    stream_directories = {
        name: (
            root / "propainter_composed" / "frames"
            if name == "proposal_frames"
            else model_root / name
        )
        for name in RGB_STREAMS
    } | {
        name: model_root / name for name in MASK_STREAMS
    } | {"provenance_labels": model_root / "provenance_labels"}
    incomplete = {
        name: len(expected_names - {path.name for path in directory.glob("*.png")})
        for name, directory in stream_directories.items()
        if {path.name for path in directory.glob("*.png")} != expected_names
    }
    if incomplete:
        raise ValueError(f"Incomplete or stale final-layer export at {root}: {incomplete}")
    return LayeredClip(
        root=root,
        model_root=model_root,
        pair_id=manifest["pair_id"],
        target_rgb_dir=Path(rows[0]["target_rgb_dir"]),
        dataset_frames=frames,
        split=rows[0]["split"],
        pose_confidences=tuple(
            float(record.get("predicted_camera_pose_confidence", 1.0))
            for record in manifest["frames"]
        ),
    )


class FinalLayerWindowDataset(Dataset):
    """Temporal windows from complete final-pipeline exports."""

    def __init__(
        self,
        roots: list[Path],
        index_path: Path,
        window: int = 5,
        stride: int = 2,
        require_full_protocol: bool = True,
    ) -> None:
        if window < 3 or window % 2 != 1:
            raise ValueError("window must be odd and at least 3")
        self.clips = [
            discover_clip(root, index_path, require_full_protocol)
            for root in roots
        ]
        self.window = window
        self.samples: list[tuple[int, int]] = []
        for clip_index, clip in enumerate(self.clips):
            if len(clip.dataset_frames) < window:
                continue
            self.samples.extend(
                (clip_index, start)
                for start in range(0, len(clip.dataset_frames) - window + 1, stride)
            )

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, index: int) -> dict[str, torch.Tensor | str]:
        clip_index, start = self.samples[index]
        clip = self.clips[clip_index]
        time_indices = range(start, start + self.window)
        model_root = clip.model_root
        streams: dict[str, list[torch.Tensor]] = {name: [] for name in RGB_STREAMS}
        masks: dict[str, list[torch.Tensor]] = {name: [] for name in MASK_STREAMS}
        provenance: list[torch.Tensor] = []
        targets: list[torch.Tensor] = []
        for time_index in time_indices:
            filename = f"{time_index:06d}.png"
            for name in RGB_STREAMS:
                path = (
                    clip.root / "propainter_composed" / "frames" / filename
                    if name == "proposal_frames"
                    else model_root / name / filename
                )
                streams[name].append(_rgb(path))
            for name in MASK_STREAMS:
                masks[name].append(_mask(model_root / name / filename))
            labels = torch.from_numpy(
                np.asarray(Image.open(model_root / "provenance_labels" / filename), dtype=np.int64)
            )
            provenance.append(
                F.one_hot(labels.clamp(1, PROVENANCE_CLASSES) - 1, PROVENANCE_CLASSES)
                .permute(2, 0, 1)
                .float()
            )
            target_frame = clip.dataset_frames[time_index]
            spatial_size = (streams["input_frames"][-1].shape[2], streams["input_frames"][-1].shape[1])
            targets.append(
                _rgb(clip.target_rgb_dir / f"{target_frame:06d}.png", spatial_size)
            )

        # Network convention is C,T,H,W.  Preserve named base/masks as well so
        # composition and audits cannot accidentally use inferred channels.
        provenance_tensor = torch.stack(provenance, dim=1)
        input_tensor = torch.cat(
            [torch.stack(streams[name], dim=1) for name in RGB_STREAMS]
            + [torch.stack(masks[name], dim=1) for name in MASK_STREAMS]
            + [provenance_tensor],
            dim=0,
        )
        foreground = torch.stack(masks["foreground_masks"], dim=1)
        physical_object = torch.stack(masks["object_masks"], dim=1)
        repair = torch.stack(masks["repair_masks"], dim=1)
        authorized = repair * (1.0 - torch.maximum(foreground, physical_object))
        # A binary lock makes pose and segmentation mistakes permanent.  Use a
        # conservative reliability prior: current exo transport is more
        # reliable than history, generated pixels are least reliable, and
        # hand/object boundaries plus source seams are explicitly uncertain.
        dynamic = torch.maximum(foreground, physical_object)
        eroded = 1.0 - F.max_pool2d(1.0 - dynamic, kernel_size=5, stride=1, padding=2)
        inner_boundary = (dynamic - eroded).clamp(0.0, 1.0)
        source_reliability = torch.ones_like(dynamic)
        labels = provenance_tensor.argmax(dim=0, keepdim=True) + 1
        for label, reliability in ((1, 1.0), (2, 0.95), (3, 0.85), (4, 0.2)):
            source_reliability = torch.where(
                labels == label,
                reliability * torch.ones_like(source_reliability),
                source_reliability,
            )
        pose_reliability = torch.tensor(
            clip.pose_confidences[start : start + self.window], dtype=torch.float32
        )[None, :, None, None]
        transported = (labels >= 1) & (labels <= 3)
        source_reliability = torch.where(
            transported,
            torch.minimum(source_reliability, pose_reliability),
            source_reliability,
        )
        source_reliability = torch.where(
            dynamic.bool(), 0.8 * torch.ones_like(dynamic), source_reliability
        )
        source_reliability = torch.where(
            inner_boundary.bool(), torch.zeros_like(dynamic), source_reliability
        )
        source_reliability = torch.where(
            torch.stack(masks["seam_masks"], dim=1).bool(),
            torch.zeros_like(source_reliability), source_reliability,
        )
        edit_weight = torch.maximum(
            authorized, (1.0 - source_reliability) * SOFT_CORRECTION_LIMIT
        )
        return {
            "input": input_tensor,
            "base": torch.stack(streams["input_frames"], dim=1),
            "proposal": torch.stack(streams["proposal_frames"], dim=1),
            "target": torch.stack(targets, dim=1),
            "authorized": authorized,
            "edit_weight": edit_weight,
            "source_reliability": source_reliability,
            "repair": repair,
            "foreground": foreground,
            "object": physical_object,
            "pair_id": clip.pair_id,
        }


class Residual3D(nn.Module):
    def __init__(self, channels: int) -> None:
        super().__init__()
        groups = min(8, channels)
        while channels % groups:
            groups -= 1
        self.body = nn.Sequential(
            nn.GroupNorm(groups, channels),
            nn.SiLU(),
            nn.Conv3d(channels, channels, 3, padding=1),
            nn.GroupNorm(groups, channels),
            nn.SiLU(),
            nn.Conv3d(channels, channels, 3, padding=1),
        )

    def forward(self, value: torch.Tensor) -> torch.Tensor:
        return value + self.body(value)


class AuthorizedFinalFusion(nn.Module):
    """Temporal U-Net head with externally bounded per-pixel corrections."""

    def __init__(
        self, width: int = 16, blocks: int = 1,
        residual_limit: float = 0.03, blend_mode: str = "convex",
    ) -> None:
        super().__init__()
        if blend_mode not in {"convex", "legacy_extrapolating"}:
            raise ValueError(f"Unknown blend mode: {blend_mode}")
        self.residual_limit = residual_limit
        self.blend_mode = blend_mode
        self.stem = nn.Conv3d(INPUT_CHANNELS, width, 3, padding=1)
        self.enc = nn.Sequential(*[Residual3D(width) for _ in range(blocks)])
        self.down = nn.Conv3d(width, width * 2, 3, stride=(1, 2, 2), padding=1)
        self.mid = nn.Sequential(*[Residual3D(width * 2) for _ in range(blocks + 1)])
        self.up = nn.Conv3d(width * 2, width, 3, padding=1)
        self.dec = nn.Sequential(*[Residual3D(width) for _ in range(blocks)])
        self.head = nn.Sequential(nn.SiLU(), nn.Conv3d(width, 4, 3, padding=1))
        # A scaled model must start from the current final algorithm, not from a
        # random re-blending of its outputs.  Zero head => exact proposal.
        nn.init.zeros_(self.head[-1].weight)
        nn.init.zeros_(self.head[-1].bias)
        if blend_mode == "convex":
            # sigmoid(-12) is effectively zero: the untrained model reproduces
            # the proposal without permitting extrapolation past either input.
            with torch.no_grad():
                self.head[-1].bias[0] = -12.0

    def forward(
        self,
        inputs: torch.Tensor,
        base: torch.Tensor,
        proposal: torch.Tensor,
        edit_weight: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        skip = self.enc(self.stem(inputs))
        value = self.mid(self.down(skip))
        value = F.interpolate(value, size=skip.shape[-3:], mode="trilinear", align_corners=False)
        raw = self.head(self.dec(self.up(value) + skip))
        if self.blend_mode == "convex":
            blend_control = torch.sigmoid(raw[:, :1])
            candidate = (
                proposal * (1.0 - blend_control) + base * blend_control
            )
        else:
            blend_control = torch.tanh(raw[:, :1])
            candidate = proposal + blend_control * (proposal - base)
        residual = torch.tanh(raw[:, 1:]) * self.residual_limit
        candidate = candidate + residual
        output = proposal + edit_weight * (candidate - proposal)
        return output.clamp(0.0, 1.0), blend_control, residual


def parameter_count(model: nn.Module) -> int:
    return sum(value.numel() for value in model.parameters() if value.requires_grad)
