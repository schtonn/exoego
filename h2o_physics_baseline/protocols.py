"""Explicit test-time information contracts for H2O experiments."""

from __future__ import annotations


MODE_REQUIREMENTS: dict[str, frozenset[str]] = {
    "rgb_only": frozenset({"exo_rgb"}),
    "anchored_rgb": frozenset({"exo_rgb", "initial_ego_anchor"}),
    "anchored_residual": frozenset({"exo_rgb", "initial_ego_anchor"}),
    "multiview_rgb": frozenset({"multi_exo_rgb"}),
    "multiview_anchored_residual": frozenset({"multi_exo_rgb", "initial_ego_anchor"}),
    "anchored_crossview": frozenset({"exo_rgb", "initial_ego_anchor"}),
    "multiview_anchored_crossview": frozenset({"multi_exo_rgb", "initial_ego_anchor"}),
    "multiview_anchored_oracle_world_state": frozenset(
        {"multi_exo_rgb", "initial_ego_anchor", "initial_ego_pose", "oracle_world_hand_object_state"}
    ),
    "multiview_anchored_oracle_local": frozenset(
        {"multi_exo_rgb", "initial_ego_anchor", "initial_ego_pose", "oracle_world_hand_object_state"}
    ),
    "multiview_anchored_oracle_local_gate15": frozenset(
        {"multi_exo_rgb", "initial_ego_anchor", "initial_ego_pose", "oracle_world_hand_object_state"}
    ),
    "multiview_anchored_oracle_local_gate31": frozenset(
        {"multi_exo_rgb", "initial_ego_anchor", "initial_ego_pose", "oracle_world_hand_object_state"}
    ),
    "multiview_anchored_student_local_gate31": frozenset(
        {"multi_exo_rgb", "initial_ego_anchor", "initial_ego_pose", "estimated_world_hand_state"}
    ),
    "multiview_anchored_oracle_hand_local_gate31": frozenset(
        {"multi_exo_rgb", "initial_ego_anchor", "initial_ego_pose", "oracle_world_hand_state"}
    ),
    "multiview_anchored_student_flow_local_gate31": frozenset(
        {"multi_exo_rgb", "initial_ego_anchor", "initial_ego_pose", "estimated_world_hand_state"}
    ),
    "multiview_anchored_student_flow_warp_local_gate31": frozenset(
        {"multi_exo_rgb", "initial_ego_anchor", "initial_ego_pose", "estimated_world_hand_state"}
    ),
    "multiview_anchored_student_flow_warp_fill_local_gate31": frozenset(
        {"multi_exo_rgb", "initial_ego_anchor", "initial_ego_pose", "estimated_world_hand_state"}
    ),
    "multiview_anchored_student_flow_warp_fill_temporal_small_gate31": frozenset(
        {"multi_exo_rgb", "initial_ego_anchor", "initial_ego_pose", "estimated_world_hand_state"}
    ),
    "multiview_anchored_student_flow_warp_fill_temporal_medium_gate31": frozenset(
        {"multi_exo_rgb", "initial_ego_anchor", "initial_ego_pose", "estimated_world_hand_state"}
    ),
    "multiview_anchored_student_mask_flow_warp_fill_local_gate31": frozenset(
        {
            "multi_exo_rgb",
            "initial_ego_anchor",
            "initial_ego_pose",
            "estimated_world_hand_state",
            "estimated_initial_ego_hand_mask",
        }
    ),
    "multiview_anchored_student_flow_warp_fill_exo_local_gate31": frozenset(
        {"multi_exo_rgb", "initial_ego_anchor", "initial_ego_pose", "estimated_world_hand_state"}
    ),
    "multiview_anchored_student_flow_warp_fill_exo_hand_local_gate31": frozenset(
        {"multi_exo_rgb", "initial_ego_anchor", "initial_ego_pose", "estimated_world_hand_state"}
    ),
    "multiview_anchored_student_flow_warp_fill_exo_joint_local_gate31": frozenset(
        {"multi_exo_rgb", "initial_ego_anchor", "initial_ego_pose", "estimated_world_hand_state"}
    ),
    "multiview_anchored_student_flow_warp_fill_exo_joint_residual_gate31": frozenset(
        {"multi_exo_rgb", "initial_ego_anchor", "initial_ego_pose", "estimated_world_hand_state"}
    ),
    "rgb_state": frozenset({"exo_rgb", "target_view_hand_object_state"}),
    "state_only": frozenset({"target_view_hand_object_state"}),
    "geometry": frozenset(
        {"exo_rgb", "exo_metric_depth", "exo_camera_pose", "full_target_camera_trajectory"}
    ),
    "geometry_state": frozenset(
        {
            "exo_rgb",
            "exo_metric_depth",
            "exo_camera_pose",
            "full_target_camera_trajectory",
            "target_view_hand_object_state",
        }
    ),
    "geometry_state_distance": frozenset(
        {
            "exo_rgb",
            "exo_metric_depth",
            "exo_camera_pose",
            "full_target_camera_trajectory",
            "target_view_hand_object_state",
            "oracle_hand_object_surface_distance",
        }
    ),
}


PROTOCOL_ALLOWED: dict[str, frozenset[str]] = {
    "oracle": frozenset().union(*MODE_REQUIREMENTS.values()),
    # Anchored makes its one ego frame explicit; it still forbids every future
    # target pose/frame/state field used by the Oracle geometry modes.
    "anchored": frozenset({"exo_rgb", "target_person_track", "initial_ego_anchor"}),
    "canonical": frozenset({"exo_rgb", "target_person_track", "user_fov"}),
    "multi_exo": frozenset({"multi_exo_rgb", "target_person_track"}),
    "multi_exo_anchored": frozenset(
        {"multi_exo_rgb", "target_person_track", "initial_ego_anchor"}
    ),
    "multi_exo_student_anchored": frozenset(
        {
            "multi_exo_rgb",
            "target_person_track",
            "initial_ego_anchor",
            "initial_ego_pose",
            "estimated_world_hand_state",
            "estimated_initial_ego_hand_mask",
        }
    ),
}


# The layered renderer is newer than the condition-mode experiments above and
# has a different set of actual file reads.  Validate its exported manifest,
# rather than inferring the contract from a model name.
LAYERED_CONTRACT_VERSION = 2
LAYERED_PROTOCOLS: dict[str, dict[str, bool]] = {
    "anchored_gt_mount_annotated_object": {
        "ego_first_rgbd": True,
        "ego_first_pose": True,
        "annotated_object_pose": True,
        "future_ego_rgb_for_inference": False,
    },
    "anchored_gt_mount_no_object": {
        "ego_first_rgbd": True,
        "ego_first_pose": True,
        "annotated_object_pose": False,
        "future_ego_rgb_for_inference": False,
    },
    "exo_only_gt_mount_annotated_object": {
        "ego_first_rgbd": False,
        "ego_first_pose": True,
        "annotated_object_pose": True,
        "future_ego_rgb_for_inference": False,
    },
    "exo_only_gt_mount_no_object": {
        "ego_first_rgbd": False,
        "ego_first_pose": True,
        "annotated_object_pose": False,
        "future_ego_rgb_for_inference": False,
    },
    "exo_only_estimated_mount_annotated_object": {
        "ego_first_rgbd": False,
        "ego_first_pose": False,
        "annotated_object_pose": True,
        "future_ego_rgb_for_inference": False,
    },
    "exo_only_estimated_mount": {
        "ego_first_rgbd": False,
        "ego_first_pose": False,
        "annotated_object_pose": False,
        "future_ego_rgb_for_inference": False,
    },
}


def validate_layered_manifest(manifest: dict, protocol: str | None = None) -> dict:
    """Check the renderer's recorded file-level input contract.

    This does not prove that an arbitrary Python edit cannot read another file;
    it makes the current renderer/fusion hand-off fail closed when required
    provenance fields are absent or inconsistent.
    """
    contract = manifest.get("input_contract")
    if not isinstance(contract, dict):
        raise ValueError("Layered export has no input_contract; regenerate it")
    if contract.get("version") != LAYERED_CONTRACT_VERSION:
        raise ValueError(
            "Unsupported layered input contract version: "
            f"{contract.get('version')!r}"
        )
    actual_protocol = manifest.get("protocol_name")
    expected_protocol = protocol or actual_protocol
    if expected_protocol not in LAYERED_PROTOCOLS:
        raise ValueError(f"Unknown layered protocol: {expected_protocol!r}")
    if actual_protocol != expected_protocol:
        raise ValueError(
            f"Expected layered protocol {expected_protocol!r}, found {actual_protocol!r}"
        )
    mismatches = {
        key: {"required": expected, "found": contract.get(key)}
        for key, expected in LAYERED_PROTOCOLS[expected_protocol].items()
        if contract.get(key) is not expected
    }
    if mismatches:
        raise ValueError(f"Layered input contract mismatch: {mismatches}")
    return contract


def validate_protocol(condition_mode: str, protocol: str) -> frozenset[str]:
    if condition_mode not in MODE_REQUIREMENTS:
        raise ValueError(f"Unknown condition mode: {condition_mode}")
    if protocol not in PROTOCOL_ALLOWED:
        raise ValueError(f"Unknown protocol: {protocol}")
    required = MODE_REQUIREMENTS[condition_mode]
    missing = required - PROTOCOL_ALLOWED[protocol]
    if missing:
        fields = ", ".join(sorted(missing))
        raise ValueError(
            f"condition_mode={condition_mode!r} violates protocol={protocol!r}; "
            f"unavailable test-time inputs: {fields}"
        )
    return required
