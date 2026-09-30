#!/usr/bin/env python3
"""Small model that exposes reusable physical condition tokens."""

from __future__ import annotations

import torch
from torch import nn
from torch.nn import functional


def _student_flow_warp(anchor: torch.Tensor, maps: torch.Tensor) -> torch.Tensor:
    """Transport anchor pixels with confidence-tapered initial-to-current hand flow."""
    batch, time, _, height, width = anchor.shape
    ys = (torch.arange(height, device=anchor.device, dtype=anchor.dtype) + 0.5) * 2 / height - 1
    xs = (torch.arange(width, device=anchor.device, dtype=anchor.dtype) + 0.5) * 2 / width - 1
    yy, xx = torch.meshgrid(ys, xs, indexing="ij")
    grid = torch.stack((xx, yy), dim=-1)[None].expand(batch * time, -1, -1, -1)
    flow = maps[:, :, 2:4].reshape(batch * time, 2, height, width).permute(0, 2, 3, 1)
    support = maps[:, :, :2].amax(dim=2).reshape(batch * time, height, width, 1)
    grid = grid - 2.0 * flow * support.square()
    warped = functional.grid_sample(
        anchor.reshape(batch * time, 3, height, width),
        grid,
        mode="bilinear",
        padding_mode="border",
        align_corners=False,
    )
    return warped.reshape(batch, time, 3, height, width)


def _student_disocclusion_fill(
    anchor: torch.Tensor,
    transported: torch.Tensor,
    maps: torch.Tensor,
    use_mask_support: bool = False,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Estimate revealed background and expose old/new supports to the repair head."""
    batch, time, _, height, width = anchor.shape
    if use_mask_support:
        initial_support = maps[:, :1, 4:5].expand(-1, time, -1, -1, -1)
        current_support = maps[:, :, 4:5]
    else:
        initial_support = maps[:, :1, :2].amax(dim=2, keepdim=True).expand(-1, time, -1, -1, -1)
        current_support = maps[:, :, :2].amax(dim=2, keepdim=True)
    old_only = torch.relu(initial_support - current_support)
    new_only = torch.relu(current_support - initial_support)
    old_only = functional.max_pool2d(
        old_only.reshape(batch * time, 1, height, width),
        kernel_size=9,
        stride=1,
        padding=4,
    ).reshape(batch, time, 1, height, width)
    background = functional.avg_pool2d(
        anchor.reshape(batch * time, 3, height, width),
        kernel_size=5,
        stride=1,
        padding=2,
        count_include_pad=False,
    ).reshape(batch, time, 3, height, width)
    filled = transported * (1.0 - old_only) + background * old_only
    return filled, old_only, new_only


class ConvBlock(nn.Module):
    def __init__(self, input_channels: int, output_channels: int, stride: int = 1) -> None:
        super().__init__()
        self.block = nn.Sequential(
            nn.Conv2d(input_channels, output_channels, 3, stride=stride, padding=1),
            nn.GroupNorm(min(8, output_channels), output_channels),
            nn.SiLU(),
            nn.Conv2d(output_channels, output_channels, 3, padding=1),
            nn.GroupNorm(min(8, output_channels), output_channels),
            nn.SiLU(),
        )

    def forward(self, value: torch.Tensor) -> torch.Tensor:
        return self.block(value)


class TemporalResidualBlock(nn.Module):
    """A residual 3-D convolution block for short, non-causal repair windows."""

    def __init__(self, channels: int) -> None:
        super().__init__()
        groups = min(8, channels)
        self.block = nn.Sequential(
            nn.Conv3d(channels, channels, 3, padding=1),
            nn.GroupNorm(groups, channels),
            nn.SiLU(),
            nn.Conv3d(channels, channels, 3, padding=1),
            nn.GroupNorm(groups, channels),
        )

    def forward(self, value: torch.Tensor) -> torch.Tensor:
        return functional.silu(value + self.block(value))


class TemporalMotionHead(nn.Module):
    """Predict gated RGB corrections while exchanging evidence across frames.

    Inputs and outputs retain the public ``[B,T,C,H,W]`` convention.  The final
    projection is zero initialized so that a new run starts exactly at the
    audited deterministic warp/fill rather than perturbing it randomly.
    """

    def __init__(self, input_channels: int, width: int, blocks: int) -> None:
        super().__init__()
        self.stem = nn.Sequential(
            nn.Conv3d(input_channels, width, (1, 3, 3), padding=(0, 1, 1)),
            nn.GroupNorm(min(8, width), width),
            nn.SiLU(),
        )
        self.blocks = nn.Sequential(*(TemporalResidualBlock(width) for _ in range(blocks)))
        self.output = nn.Conv3d(width, 3, (1, 3, 3), padding=(0, 1, 1))
        nn.init.zeros_(self.output.weight)
        nn.init.zeros_(self.output.bias)

    def forward(self, value: torch.Tensor) -> torch.Tensor:
        value = value.permute(0, 2, 1, 3, 4)
        result = torch.tanh(self.output(self.blocks(self.stem(value))))
        return result.permute(0, 2, 1, 3, 4)


class PhysicalConditionEncoder(nn.Module):
    """Encode per-frame 4D state, preserving one token per video frame."""

    def __init__(self, input_dim: int = 317, token_dim: int = 256, layers: int = 3) -> None:
        super().__init__()
        self.input = nn.Sequential(
            nn.Linear(input_dim, token_dim), nn.LayerNorm(token_dim), nn.SiLU(), nn.Linear(token_dim, token_dim)
        )
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=token_dim,
            nhead=8,
            dim_feedforward=token_dim * 4,
            dropout=0.1,
            activation="gelu",
            batch_first=True,
            norm_first=True,
        )
        self.temporal = nn.TransformerEncoder(encoder_layer, num_layers=layers, enable_nested_tensor=False)
        self.output_norm = nn.LayerNorm(token_dim)

    def forward(self, state: torch.Tensor) -> torch.Tensor:
        return self.output_norm(self.temporal(self.input(state)))


class PhysicalEgoVideoPredictor(nn.Module):
    """Proof-of-concept short-video predictor for physical condition ablations.

    The model deliberately avoids pixel-aligned exo-to-ego skips. Exocentric RGB
    contributes a global appearance token, while target-view physical maps define
    spatial layout and the 4D state encoder supplies temporally contextual tokens.
    """

    valid_modes = {
        "rgb_state",
        "rgb_only",
        "state_only",
        "geometry",
        "geometry_state",
        "geometry_state_distance",
        "anchored_rgb",
        "anchored_residual",
        "multiview_rgb",
        "multiview_anchored_residual",
        "anchored_crossview",
        "multiview_anchored_crossview",
        "multiview_anchored_oracle_world_state",
        "multiview_anchored_oracle_local",
        "multiview_anchored_oracle_local_gate15",
        "multiview_anchored_oracle_local_gate31",
        "multiview_anchored_student_local_gate31",
        "multiview_anchored_oracle_hand_local_gate31",
        "multiview_anchored_student_flow_local_gate31",
        "multiview_anchored_student_flow_warp_local_gate31",
        "multiview_anchored_student_flow_warp_fill_local_gate31",
        "multiview_anchored_student_flow_warp_fill_temporal_small_gate31",
        "multiview_anchored_student_flow_warp_fill_temporal_medium_gate31",
        "multiview_anchored_student_mask_flow_warp_fill_local_gate31",
        "multiview_anchored_student_flow_warp_fill_exo_local_gate31",
        "multiview_anchored_student_flow_warp_fill_exo_hand_local_gate31",
        "multiview_anchored_student_flow_warp_fill_exo_joint_local_gate31",
        "multiview_anchored_student_flow_warp_fill_exo_joint_residual_gate31",
    }

    def __init__(
        self,
        physical_dim: int = 317,
        map_channels: int = 5,
        token_dim: int = 256,
        condition_mode: str = "rgb_state",
    ) -> None:
        super().__init__()
        if condition_mode not in self.valid_modes:
            raise ValueError(f"condition_mode must be one of {sorted(self.valid_modes)}")
        self.condition_mode = condition_mode
        self.source_encoder = nn.Sequential(
            ConvBlock(3, 32, 2),
            ConvBlock(32, 64, 2),
            ConvBlock(64, 128, 2),
            ConvBlock(128, token_dim, 2),
            nn.AdaptiveAvgPool2d(1),
        )
        if condition_mode == "multiview_anchored_student_flow_warp_fill_exo_hand_local_gate31":
            self.source_hand_encoder = nn.Sequential(
                ConvBlock(3, 32, 2),
                ConvBlock(32, 64, 2),
            )
            self.source_hand_projection = nn.Sequential(
                nn.Linear(64, token_dim), nn.LayerNorm(token_dim), nn.SiLU()
            )
        if condition_mode in {
            "multiview_anchored_student_flow_warp_fill_exo_joint_local_gate31",
            "multiview_anchored_student_flow_warp_fill_exo_joint_residual_gate31",
        }:
            self.source_joint_encoder = nn.Sequential(
                ConvBlock(3, 32, 2),
                ConvBlock(32, 64, 2),
            )
            self.source_joint_projection = nn.Sequential(
                nn.Linear(64, 8), nn.LayerNorm(8), nn.SiLU()
            )
        if condition_mode == "multiview_anchored_student_flow_warp_fill_exo_joint_residual_gate31":
            self.joint_motion_head = nn.Sequential(
                ConvBlock(17, 32),
                ConvBlock(32, 32),
                nn.Conv2d(32, 3, 3, padding=1),
                nn.Tanh(),
            )
            nn.init.zeros_(self.joint_motion_head[2].weight)
            nn.init.zeros_(self.joint_motion_head[2].bias)
        self.state_encoder = PhysicalConditionEncoder(physical_dim, token_dim)
        self.map_channels = map_channels
        self.map_encoder = nn.Sequential(
            ConvBlock(map_channels + 5, 32, 2),
            ConvBlock(32, 64, 2),
            ConvBlock(64, 128, 2),
            ConvBlock(128, token_dim, 2),
        )
        if condition_mode in {"anchored_crossview", "multiview_anchored_crossview"}:
            self.cross_view_attention = nn.MultiheadAttention(
                token_dim, num_heads=8, dropout=0.0, batch_first=True
            )
            self.cross_view_norm = nn.LayerNorm(token_dim)
            self.cross_view_scale = nn.Parameter(torch.tensor(0.1))
        if condition_mode in {
            "multiview_anchored_oracle_local",
            "multiview_anchored_oracle_local_gate15",
            "multiview_anchored_oracle_local_gate31",
            "multiview_anchored_student_local_gate31",
            "multiview_anchored_oracle_hand_local_gate31",
            "multiview_anchored_student_flow_local_gate31",
            "multiview_anchored_student_flow_warp_local_gate31",
            "multiview_anchored_student_flow_warp_fill_local_gate31",
            "multiview_anchored_student_flow_warp_fill_temporal_small_gate31",
            "multiview_anchored_student_flow_warp_fill_temporal_medium_gate31",
            "multiview_anchored_student_mask_flow_warp_fill_local_gate31",
            "multiview_anchored_student_flow_warp_fill_exo_local_gate31",
            "multiview_anchored_student_flow_warp_fill_exo_hand_local_gate31",
            "multiview_anchored_student_flow_warp_fill_exo_joint_local_gate31",
            "multiview_anchored_student_flow_warp_fill_exo_joint_residual_gate31",
        }:
            local_input_channels = 15 if condition_mode in {
                    "multiview_anchored_student_flow_warp_fill_local_gate31",
                    "multiview_anchored_student_flow_warp_fill_temporal_small_gate31",
                    "multiview_anchored_student_flow_warp_fill_temporal_medium_gate31",
                    "multiview_anchored_student_mask_flow_warp_fill_local_gate31",
                    "multiview_anchored_student_flow_warp_fill_exo_local_gate31",
                    "multiview_anchored_student_flow_warp_fill_exo_hand_local_gate31",
                    "multiview_anchored_student_flow_warp_fill_exo_joint_local_gate31",
                    "multiview_anchored_student_flow_warp_fill_exo_joint_residual_gate31",
                } else 13
            if condition_mode in {
                "multiview_anchored_student_flow_warp_fill_exo_local_gate31",
                "multiview_anchored_student_flow_warp_fill_exo_hand_local_gate31",
            }:
                self.local_token_projection = nn.Sequential(
                    nn.Linear(token_dim, 8), nn.LayerNorm(8), nn.SiLU()
                )
                local_input_channels += 8
            if condition_mode == "multiview_anchored_student_flow_warp_fill_exo_joint_local_gate31":
                # Eight geometry-aligned appearance channels plus their support.
                local_input_channels += 9
            if condition_mode == "multiview_anchored_student_flow_warp_fill_temporal_small_gate31":
                self.local_motion_head = TemporalMotionHead(local_input_channels, width=64, blocks=3)
            elif condition_mode == "multiview_anchored_student_flow_warp_fill_temporal_medium_gate31":
                self.local_motion_head = TemporalMotionHead(local_input_channels, width=96, blocks=4)
            else:
                self.local_motion_head = nn.Sequential(
                    ConvBlock(local_input_channels, 32),
                    ConvBlock(32, 32),
                    nn.Conv2d(32, 3, 3, padding=1),
                    nn.Tanh(),
                )
            if condition_mode in {
                "multiview_anchored_student_flow_warp_local_gate31",
                "multiview_anchored_student_flow_warp_fill_local_gate31",
                "multiview_anchored_student_flow_warp_fill_temporal_small_gate31",
                "multiview_anchored_student_flow_warp_fill_temporal_medium_gate31",
                "multiview_anchored_student_mask_flow_warp_fill_local_gate31",
                "multiview_anchored_student_flow_warp_fill_exo_local_gate31",
                "multiview_anchored_student_flow_warp_fill_exo_hand_local_gate31",
                "multiview_anchored_student_flow_warp_fill_exo_joint_local_gate31",
                "multiview_anchored_student_flow_warp_fill_exo_joint_residual_gate31",
            }:
                # Start exactly at the deterministic physical warp.  The network
                # must earn every learned correction rather than erase the useful
                # directional prior at initialization.
                if isinstance(self.local_motion_head, nn.Sequential):
                    nn.init.zeros_(self.local_motion_head[2].weight)
                    nn.init.zeros_(self.local_motion_head[2].bias)
        self.fusion = nn.Sequential(
            nn.Linear(token_dim * 2, token_dim), nn.LayerNorm(token_dim), nn.SiLU(), nn.Linear(token_dim, token_dim)
        )
        self.film = nn.Linear(token_dim, token_dim * 2)
        self.decoder = nn.Sequential(
            nn.ConvTranspose2d(token_dim, 128, 4, stride=2, padding=1), nn.GroupNorm(8, 128), nn.SiLU(),
            nn.ConvTranspose2d(128, 64, 4, stride=2, padding=1), nn.GroupNorm(8, 64), nn.SiLU(),
            nn.ConvTranspose2d(64, 32, 4, stride=2, padding=1), nn.GroupNorm(8, 32), nn.SiLU(),
            nn.ConvTranspose2d(32, 3, 4, stride=2, padding=1), nn.Sigmoid(),
        )

    def _crossview_joint_feature_maps(
        self,
        source: torch.Tensor,
        source_joint_uv: torch.Tensor,
        source_joint_confidence: torch.Tensor,
        target_joint_uv: torch.Tensor,
        target_joint_confidence: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Lift sampled exo joint features and splat them into initial-ego coordinates."""
        batch, views, time, channels, height, width = source.shape
        joints = source_joint_uv.shape[3]
        features = self.source_joint_encoder(
            source.reshape(batch * views * time, channels, height, width)
        )
        feature_height, feature_width = features.shape[-2:]
        sampled = functional.grid_sample(
            features,
            source_joint_uv.reshape(batch * views * time, joints, 1, 2),
            mode="bilinear",
            padding_mode="zeros",
            align_corners=False,
        ).squeeze(-1).transpose(1, 2)
        sampled = sampled.reshape(batch, views, time, joints, -1)
        source_weight = source_joint_confidence[..., None]
        descriptors = (sampled * source_weight).sum(dim=1) / source_weight.sum(
            dim=1
        ).clamp_min(1e-4)
        descriptors = self.source_joint_projection(descriptors)

        ys = (torch.arange(feature_height, device=source.device, dtype=source.dtype) + 0.5)
        ys = ys * 2.0 / feature_height - 1.0
        xs = (torch.arange(feature_width, device=source.device, dtype=source.dtype) + 0.5)
        xs = xs * 2.0 / feature_width - 1.0
        yy, xx = torch.meshgrid(ys, xs, indexing="ij")
        grid = torch.stack((xx, yy), dim=-1)
        squared_distance = torch.square(
            grid[None, None, None] - target_joint_uv[..., None, None, :]
        ).sum(dim=-1)
        joint_weight = target_joint_confidence[..., None, None] * torch.exp(
            -squared_distance / (2.0 * 0.12**2)
        )
        total_weight = joint_weight.sum(dim=2, keepdim=True)
        descriptor_map = torch.einsum("btjd,btjhw->btdhw", descriptors, joint_weight)
        descriptor_map = descriptor_map / total_weight.clamp_min(1e-4)
        support = 1.0 - torch.exp(-total_weight)
        descriptor_map = functional.interpolate(
            descriptor_map.reshape(batch * time, -1, feature_height, feature_width),
            size=(height, width),
            mode="bilinear",
            align_corners=False,
        ).reshape(batch, time, -1, height, width)
        support = functional.interpolate(
            support.reshape(batch * time, 1, feature_height, feature_width),
            size=(height, width),
            mode="bilinear",
            align_corners=False,
        ).reshape(batch, time, 1, height, width)
        return descriptor_map, support

    def condition_tokens(
        self,
        source: torch.Tensor,
        physical_vector: torch.Tensor,
        ego_anchor: torch.Tensor | None = None,
        source_hand_maps: torch.Tensor | None = None,
    ) -> torch.Tensor:
        if self.condition_mode == "multiview_anchored_student_flow_warp_fill_exo_hand_local_gate31":
            if source_hand_maps is None:
                raise ValueError("exo-hand mode requires source-view hand maps")
            batch, views, time, channels, height, width = source.shape
            features = self.source_hand_encoder(
                source.reshape(batch * views * time, channels, height, width)
            )
            support = source_hand_maps.amax(dim=3).reshape(
                batch * views * time, 1, height, width
            )
            support = functional.max_pool2d(support, kernel_size=15, stride=1, padding=7)
            support = functional.interpolate(
                support, size=features.shape[-2:], mode="bilinear", align_corners=False
            )
            pooled = (features * support).sum(dim=(-2, -1)) / support.sum(
                dim=(-2, -1)
            ).clamp_min(1e-4)
            source_tokens = self.source_hand_projection(pooled).reshape(
                batch, views, time, -1
            ).mean(dim=1)
        elif source.ndim == 6:
            batch, views, time, channels, height, width = source.shape
            encoded = self.source_encoder(
                source.reshape(batch * views * time, channels, height, width)
            ).flatten(1)
            source_tokens = encoded.reshape(batch, views, time, -1).mean(dim=1)
        else:
            batch, time, channels, height, width = source.shape
            source_tokens = self.source_encoder(
                source.reshape(batch * time, channels, height, width)
            ).flatten(1)
            source_tokens = source_tokens.reshape(batch, time, -1)
        if self.condition_mode in {
            "anchored_rgb",
            "anchored_residual",
            "multiview_anchored_residual",
            "anchored_crossview",
            "multiview_anchored_crossview",
            "multiview_anchored_oracle_world_state",
            "multiview_anchored_oracle_local",
            "multiview_anchored_oracle_local_gate15",
            "multiview_anchored_oracle_local_gate31",
            "multiview_anchored_student_local_gate31",
            "multiview_anchored_oracle_hand_local_gate31",
            "multiview_anchored_student_flow_local_gate31",
            "multiview_anchored_student_flow_warp_local_gate31",
            "multiview_anchored_student_flow_warp_fill_local_gate31",
            "multiview_anchored_student_flow_warp_fill_temporal_small_gate31",
            "multiview_anchored_student_flow_warp_fill_temporal_medium_gate31",
            "multiview_anchored_student_mask_flow_warp_fill_local_gate31",
            "multiview_anchored_student_flow_warp_fill_exo_local_gate31",
            "multiview_anchored_student_flow_warp_fill_exo_hand_local_gate31",
            "multiview_anchored_student_flow_warp_fill_exo_joint_local_gate31",
            "multiview_anchored_student_flow_warp_fill_exo_joint_residual_gate31",
        }:
            if ego_anchor is None:
                raise ValueError("anchored_rgb requires the initial ego frame")
            anchor_tokens = self.source_encoder(
                ego_anchor.reshape(batch * time, channels, height, width)
            ).flatten(1)
            anchor_tokens = anchor_tokens.reshape(batch, time, -1)
            return self.fusion(torch.cat((source_tokens, anchor_tokens), dim=-1))
        state_tokens = self.state_encoder(physical_vector)
        state_enabled = self.condition_mode in {
            "rgb_state",
            "state_only",
            "geometry_state",
            "geometry_state_distance",
        }
        if not state_enabled:
            state_tokens = torch.zeros_like(state_tokens)
        elif self.condition_mode == "geometry_state":
            # 273:315 are the 42 continuous joint-to-surface proximity values.
            no_distance = physical_vector.clone()
            no_distance[..., 273:315] = 0
            state_tokens = self.state_encoder(no_distance)
        if self.condition_mode == "state_only":
            source_tokens = torch.zeros_like(source_tokens)
        return self.fusion(torch.cat((source_tokens, state_tokens), dim=-1))

    def forward(
        self,
        source: torch.Tensor,
        physical_vector: torch.Tensor,
        physical_maps: torch.Tensor,
        coarse_rgb: torch.Tensor | None = None,
        coarse_depth: torch.Tensor | None = None,
        visibility_mask: torch.Tensor | None = None,
        ego_anchor: torch.Tensor | None = None,
        initial_ego_maps: torch.Tensor | None = None,
        source_hand_maps: torch.Tensor | None = None,
        source_joint_uv: torch.Tensor | None = None,
        source_joint_confidence: torch.Tensor | None = None,
        target_joint_uv: torch.Tensor | None = None,
        target_joint_confidence: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        if source.ndim == 6:
            batch, _, time, _, height, width = source.shape
        else:
            batch, time, _, height, width = source.shape
        if self.condition_mode in {
            "multiview_anchored_oracle_local",
            "multiview_anchored_oracle_local_gate15",
            "multiview_anchored_oracle_local_gate31",
            "multiview_anchored_student_local_gate31",
            "multiview_anchored_oracle_hand_local_gate31",
            "multiview_anchored_student_flow_local_gate31",
            "multiview_anchored_student_flow_warp_local_gate31",
            "multiview_anchored_student_flow_warp_fill_local_gate31",
            "multiview_anchored_student_flow_warp_fill_temporal_small_gate31",
            "multiview_anchored_student_flow_warp_fill_temporal_medium_gate31",
            "multiview_anchored_student_mask_flow_warp_fill_local_gate31",
            "multiview_anchored_student_flow_warp_fill_exo_joint_local_gate31",
            "multiview_anchored_student_flow_warp_fill_exo_joint_residual_gate31",
        }:
            # This diagnostic isolates whether the physical trajectory itself is
            # useful; RGB appearance remains entirely in the ego anchor.
            tokens = physical_vector.new_zeros((batch, time, self.film.in_features))
        else:
            tokens = self.condition_tokens(source, physical_vector, ego_anchor, source_hand_maps)
        if self.condition_mode in {
            "multiview_anchored_oracle_world_state",
            "multiview_anchored_oracle_local",
            "multiview_anchored_oracle_local_gate15",
            "multiview_anchored_oracle_local_gate31",
            "multiview_anchored_student_local_gate31",
            "multiview_anchored_oracle_hand_local_gate31",
            "multiview_anchored_student_flow_local_gate31",
            "multiview_anchored_student_flow_warp_local_gate31",
            "multiview_anchored_student_flow_warp_fill_local_gate31",
            "multiview_anchored_student_flow_warp_fill_temporal_small_gate31",
            "multiview_anchored_student_flow_warp_fill_temporal_medium_gate31",
            "multiview_anchored_student_mask_flow_warp_fill_local_gate31",
            "multiview_anchored_student_flow_warp_fill_exo_local_gate31",
            "multiview_anchored_student_flow_warp_fill_exo_hand_local_gate31",
            "multiview_anchored_student_flow_warp_fill_exo_joint_local_gate31",
            "multiview_anchored_student_flow_warp_fill_exo_joint_residual_gate31",
        }:
            if initial_ego_maps is None:
                raise ValueError("oracle world-state mode requires initial-ego projected maps")
            state_maps = initial_ego_maps.clone()
        else:
            state_maps = physical_maps.clone()
        if self.condition_mode not in {
            "rgb_state",
            "state_only",
            "geometry_state",
            "geometry_state_distance",
            "multiview_anchored_oracle_world_state",
            "multiview_anchored_oracle_local",
            "multiview_anchored_oracle_local_gate15",
            "multiview_anchored_oracle_local_gate31",
            "multiview_anchored_student_local_gate31",
            "multiview_anchored_oracle_hand_local_gate31",
            "multiview_anchored_student_flow_local_gate31",
            "multiview_anchored_student_flow_warp_local_gate31",
            "multiview_anchored_student_flow_warp_fill_local_gate31",
            "multiview_anchored_student_flow_warp_fill_temporal_small_gate31",
            "multiview_anchored_student_flow_warp_fill_temporal_medium_gate31",
            "multiview_anchored_student_mask_flow_warp_fill_local_gate31",
            "multiview_anchored_student_flow_warp_fill_exo_local_gate31",
            "multiview_anchored_student_flow_warp_fill_exo_hand_local_gate31",
            "multiview_anchored_student_flow_warp_fill_exo_joint_local_gate31",
            "multiview_anchored_student_flow_warp_fill_exo_joint_residual_gate31",
        }:
            state_maps.zero_()
        elif self.condition_mode == "geometry_state":
            state_maps[:, :, 2:4] = 0
        geometry_enabled = self.condition_mode in {
            "geometry",
            "geometry_state",
            "geometry_state_distance",
        }
        if coarse_rgb is None:
            coarse_rgb = torch.zeros(
                (batch, time, 3, height, width), device=source.device, dtype=source.dtype
            )
        if coarse_depth is None:
            coarse_depth = torch.zeros((batch, time, 1, height, width), device=source.device, dtype=source.dtype)
        if visibility_mask is None:
            visibility_mask = torch.zeros_like(coarse_depth)
        anchor_enabled = self.condition_mode in {
            "anchored_rgb",
            "anchored_residual",
            "multiview_anchored_residual",
            "anchored_crossview",
            "multiview_anchored_crossview",
            "multiview_anchored_oracle_world_state",
            "multiview_anchored_oracle_local",
            "multiview_anchored_oracle_local_gate15",
            "multiview_anchored_oracle_local_gate31",
            "multiview_anchored_student_local_gate31",
            "multiview_anchored_oracle_hand_local_gate31",
            "multiview_anchored_student_flow_local_gate31",
            "multiview_anchored_student_flow_warp_local_gate31",
            "multiview_anchored_student_flow_warp_fill_local_gate31",
            "multiview_anchored_student_flow_warp_fill_temporal_small_gate31",
            "multiview_anchored_student_flow_warp_fill_temporal_medium_gate31",
            "multiview_anchored_student_mask_flow_warp_fill_local_gate31",
            "multiview_anchored_student_flow_warp_fill_exo_local_gate31",
            "multiview_anchored_student_flow_warp_fill_exo_hand_local_gate31",
            "multiview_anchored_student_flow_warp_fill_exo_joint_local_gate31",
            "multiview_anchored_student_flow_warp_fill_exo_joint_residual_gate31",
        }
        if anchor_enabled:
            if ego_anchor is None:
                raise ValueError("anchored_rgb requires the initial ego frame")
            coarse_rgb = ego_anchor
            coarse_depth = torch.zeros_like(coarse_depth)
            visibility_mask = torch.zeros_like(visibility_mask)
        elif not geometry_enabled:
            coarse_rgb = torch.zeros_like(coarse_rgb)
            coarse_depth = torch.zeros_like(coarse_depth)
            visibility_mask = torch.zeros_like(visibility_mask)
        if self.condition_mode in {
            "multiview_anchored_oracle_local",
            "multiview_anchored_oracle_local_gate15",
            "multiview_anchored_oracle_local_gate31",
            "multiview_anchored_student_local_gate31",
            "multiview_anchored_oracle_hand_local_gate31",
            "multiview_anchored_student_flow_local_gate31",
            "multiview_anchored_student_flow_warp_local_gate31",
            "multiview_anchored_student_flow_warp_fill_local_gate31",
            "multiview_anchored_student_flow_warp_fill_temporal_small_gate31",
            "multiview_anchored_student_flow_warp_fill_temporal_medium_gate31",
            "multiview_anchored_student_mask_flow_warp_fill_local_gate31",
            "multiview_anchored_student_flow_warp_fill_exo_local_gate31",
            "multiview_anchored_student_flow_warp_fill_exo_hand_local_gate31",
            "multiview_anchored_student_flow_warp_fill_exo_joint_local_gate31",
            "multiview_anchored_student_flow_warp_fill_exo_joint_residual_gate31",
        }:
            initial_maps = state_maps[:, :1].expand_as(state_maps)
            uses_warp = self.condition_mode in {
                "multiview_anchored_student_flow_warp_local_gate31",
                "multiview_anchored_student_flow_warp_fill_local_gate31",
                "multiview_anchored_student_flow_warp_fill_temporal_small_gate31",
                "multiview_anchored_student_flow_warp_fill_temporal_medium_gate31",
                "multiview_anchored_student_mask_flow_warp_fill_local_gate31",
                "multiview_anchored_student_flow_warp_fill_exo_local_gate31",
                "multiview_anchored_student_flow_warp_fill_exo_hand_local_gate31",
                "multiview_anchored_student_flow_warp_fill_exo_joint_local_gate31",
                "multiview_anchored_student_flow_warp_fill_exo_joint_residual_gate31",
            }
            base_rgb = _student_flow_warp(ego_anchor, state_maps) if uses_warp else ego_anchor
            extra_maps: tuple[torch.Tensor, ...] = ()
            if self.condition_mode in {
                "multiview_anchored_student_flow_warp_fill_local_gate31",
                "multiview_anchored_student_flow_warp_fill_temporal_small_gate31",
                "multiview_anchored_student_flow_warp_fill_temporal_medium_gate31",
                "multiview_anchored_student_mask_flow_warp_fill_local_gate31",
                "multiview_anchored_student_flow_warp_fill_exo_local_gate31",
                "multiview_anchored_student_flow_warp_fill_exo_hand_local_gate31",
                "multiview_anchored_student_flow_warp_fill_exo_joint_local_gate31",
                "multiview_anchored_student_flow_warp_fill_exo_joint_residual_gate31",
            }:
                base_rgb, old_only, new_only = _student_disocclusion_fill(
                    ego_anchor,
                    base_rgb,
                    state_maps,
                    use_mask_support=(
                        self.condition_mode
                        == "multiview_anchored_student_mask_flow_warp_fill_local_gate31"
                    ),
                )
                extra_maps = (old_only, new_only)
            if self.condition_mode in {
                "multiview_anchored_student_flow_warp_fill_exo_local_gate31",
                "multiview_anchored_student_flow_warp_fill_exo_hand_local_gate31",
            }:
                token_maps = self.local_token_projection(tokens).reshape(batch, time, 8, 1, 1)
                extra_maps = (*extra_maps, token_maps.expand(-1, -1, -1, height, width))
            joint_map = None
            joint_support = None
            if self.condition_mode in {
                "multiview_anchored_student_flow_warp_fill_exo_joint_local_gate31",
                "multiview_anchored_student_flow_warp_fill_exo_joint_residual_gate31",
            }:
                joint_inputs = (
                    source_joint_uv,
                    source_joint_confidence,
                    target_joint_uv,
                    target_joint_confidence,
                )
                if any(value is None for value in joint_inputs):
                    raise ValueError("exo-joint mode requires calibrated joint coordinates")
                joint_map, joint_support = self._crossview_joint_feature_maps(
                    source,
                    source_joint_uv,
                    source_joint_confidence,
                    target_joint_uv,
                    target_joint_confidence,
                )
                if self.condition_mode == "multiview_anchored_student_flow_warp_fill_exo_joint_local_gate31":
                    extra_maps = (*extra_maps, joint_map, joint_support)
            local_input = torch.cat(
                (base_rgb, state_maps, state_maps - initial_maps, *extra_maps), dim=2
            )
            if isinstance(self.local_motion_head, TemporalMotionHead):
                delta = self.local_motion_head(local_input)
            else:
                delta = self.local_motion_head(
                    local_input.reshape(batch * time, local_input.shape[2], height, width)
                ).reshape(batch, time, 3, height, width)
            if self.condition_mode.endswith("gate15") or self.condition_mode.endswith("gate31"):
                kernel = 15 if self.condition_mode.endswith("gate15") else 31
                gate = torch.maximum(state_maps, initial_maps).amax(dim=2, keepdim=True)
                gate = torch.nn.functional.max_pool2d(
                    gate.reshape(batch * time, 1, height, width),
                    kernel_size=kernel,
                    stride=1,
                    padding=kernel // 2,
                ).reshape(batch, time, 1, height, width)
                delta = delta * gate
            if self.condition_mode == "multiview_anchored_student_flow_warp_fill_exo_joint_residual_gate31":
                if joint_map is None or joint_support is None:
                    raise AssertionError("joint residual features were not constructed")
                joint_input = torch.cat((base_rgb, state_maps, joint_map, joint_support), dim=2)
                joint_delta = self.joint_motion_head(
                    joint_input.reshape(batch * time, 17, height, width)
                ).reshape(batch, time, 3, height, width)
                joint_gate = functional.max_pool2d(
                    joint_support.reshape(batch * time, 1, height, width),
                    kernel_size=7,
                    stride=1,
                    padding=3,
                ).reshape(batch, time, 1, height, width)
                delta = delta + joint_delta * joint_gate
            return torch.clamp(base_rgb + 0.25 * delta, 0.0, 1.0), tokens
        maps = torch.cat((state_maps, coarse_rgb, coarse_depth, visibility_mask), dim=2)
        spatial = self.map_encoder(maps.reshape(batch * time, maps.shape[2], height, width))
        if self.condition_mode in {"anchored_crossview", "multiview_anchored_crossview"}:
            source_backbone = self.source_encoder[:-1]
            if source.ndim == 6:
                _, views, _, channels, _, _ = source.shape
                source_features = source_backbone(
                    source.reshape(batch * views * time, channels, height, width)
                )
                feature_height, feature_width = source_features.shape[-2:]
                source_tokens = (
                    source_features.reshape(
                        batch, views, time, -1, feature_height, feature_width
                    )
                    .permute(0, 2, 1, 4, 5, 3)
                    .reshape(batch * time, views * feature_height * feature_width, -1)
                )
            else:
                channels = source.shape[2]
                source_features = source_backbone(
                    source.reshape(batch * time, channels, height, width)
                )
                feature_height, feature_width = source_features.shape[-2:]
                source_tokens = source_features.flatten(2).transpose(1, 2)
            anchor_features = source_backbone(
                ego_anchor.reshape(batch * time, 3, height, width)
            )
            anchor_tokens = anchor_features.flatten(2).transpose(1, 2)
            attended, _ = self.cross_view_attention(
                anchor_tokens, source_tokens, source_tokens, need_weights=False
            )
            attended = self.cross_view_norm(attended).transpose(1, 2).reshape(
                batch * time, -1, feature_height, feature_width
            )
            if attended.shape[-2:] != spatial.shape[-2:]:
                raise ValueError("Source and anchor feature maps have incompatible sizes")
            spatial = spatial + self.cross_view_scale * attended
        gamma, beta = self.film(tokens.reshape(batch * time, -1)).chunk(2, dim=-1)
        spatial = spatial * (1 + gamma[:, :, None, None]) + beta[:, :, None, None]
        decoded = self.decoder(spatial).reshape(batch, time, 3, height, width)
        if self.condition_mode in {
            "anchored_residual",
            "multiview_anchored_residual",
            "anchored_crossview",
            "multiview_anchored_crossview",
            "multiview_anchored_oracle_world_state",
        }:
            # Identity-preserving parameterization: zero residual is exactly the
            # supplied ego anchor, unlike an absolute encoder-decoder output.
            output = torch.clamp(ego_anchor + 0.25 * (decoded - 0.5), 0.0, 1.0)
        elif geometry_enabled:
            known_refined = torch.clamp(coarse_rgb + 0.25 * (decoded - 0.5), 0.0, 1.0)
            output = visibility_mask * known_refined + (1.0 - visibility_mask) * decoded
        else:
            output = decoded
        return output, tokens
