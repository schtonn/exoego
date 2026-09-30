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
