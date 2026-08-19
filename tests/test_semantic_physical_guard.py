from __future__ import annotations

import math
from pathlib import Path

from scene_graph.captioning.evidence import crop_evidence_manifest
from scene_graph.captioning.physical_guard import (
    assess_guard_visible_form_recovery,
    assess_physical_form,
    assess_physical_recovery,
)


def _hard(
    verdict: str = "contradiction",
    *,
    visible_form: str = "pipe frame",
    topology: str = "standalone",
    complete_bounded: bool = True,
) -> dict:
    return {
        "hard_veto": True,
        "verdict": verdict,
        "model_guard": {
            "verdict": verdict,
            "decision": "keep",
            "visible_form_category": visible_form,
            "visible_form_confidence": 0.97,
            "topology": topology,
            "complete_bounded": complete_bounded,
        },
    }


def _recovery(
    category: str,
    *,
    source: str = "physical_form_recovery",
    views: tuple[int, ...] = (1, 2, 3),
    bounded: bool = True,
    confidence: float = 0.95,
    fingerprint: str | None = None,
) -> dict:
    poses = {}
    crops = []
    for image_id in views:
        angle = math.radians(float(image_id) * 20.0)
        center = [10.0 * math.cos(angle), 10.0 * math.sin(angle), 0.0]
        poses[image_id] = {
            "camera_center_world_m": center,
            "camera_forward_world": [-center[0] / 10.0, -center[1] / 10.0, 0.0],
        }
        crops.append((Path(f"img_{image_id:06d}_det_0001.npz"), b"jpeg"))
    manifest = crop_evidence_manifest(
        7,
        crops,
        frame_pose_index=poses,
        object_position_world_m=[0.0, 0.0, 0.0],
    )
    return {
        "source": source,
        "event_id": f"object:7:{source}",
        "category": category,
        "description": category,
        "attributes": ["standalone_bounded"] if bounded else ["component_only"],
        "standalone_bounded": bounded,
        "component_only": not bounded,
        "confidence": confidence,
        "decision": "keep" if bounded else "unknown",
        "confirmation_eligible": False,
        **manifest,
        "evidence_fingerprint_sha256": fingerprint or (
            manifest["evidence_fingerprint_sha256"]
        ),
    }


def _verification(category: str, **kwargs) -> dict:
    kwargs.setdefault("views", (4, 5, 6))
    return _recovery(
        category,
        source="physical_form_recovery_verification",
        **kwargs,
    )


def test_volumetric_cylinder_cannot_confirm_as_manhole_cover() -> None:
    result = assess_physical_form("manhole cover", [0.338, 0.248, 0.600])
    assert result["hard_veto"] is True
    assert result["confirmation_allowed"] is False
    assert "planar_noun_has_strongly_volumetric_metric_shape" in result["reason_codes"]


def test_tall_pipe_frame_cannot_confirm_as_workbench() -> None:
    result = assess_physical_form("workbench", [1.126, 0.502, 1.963])
    assert result["hard_veto"] is True
    assert "horizontal_support_noun_has_dominant_vertical_extent" in result["reason_codes"]


def test_thin_component_shape_cannot_confirm_as_bulky_vehicle() -> None:
    result = assess_physical_form("vehicle", [2.134, 0.197, 0.727])
    assert result["hard_veto"] is True
    assert result["verdict"] == "part_only"


def test_missing_metric_dimensions_fail_closed_without_relabeling() -> None:
    result = assess_physical_form("cabinet", None)
    assert result["hard_veto"] is False
    assert result["confirmation_allowed"] is False
    assert result["category"] == "cabinet"


def test_candidate_conditioned_guard_is_veto_only() -> None:
    result = assess_physical_form(
        "cart",
        [1.0, 0.6, 0.8],
        model_guard={
            "verdict": "part_only",
            "confidence": 0.96,
            "supported_view_count": 3,
        },
    )
    assert result["hard_veto"] is True
    assert result["category"] == "cart"
    assert "paired_visual_guard_part_only" in result["reason_codes"]


def test_compatible_guard_never_promotes_or_changes_category() -> None:
    result = assess_physical_form(
        "cabinet",
        [1.0, 0.5, 1.8],
        model_guard={
            "verdict": "compatible",
            "proposed_category": "cabinet",
            "confidence": 0.99,
            "supported_view_count": 3,
        },
    )
    assert result["hard_veto"] is False
    assert result["confirmation_allowed"] is True
    assert result["category"] == "cabinet"


def test_insufficient_visual_guard_withholds_confirmation() -> None:
    result = assess_physical_form(
        "cabinet",
        [1.0, 0.5, 1.8],
        model_guard={
            "verdict": "insufficient",
            "proposed_category": "cabinet",
            "confidence": 0.99,
            "supported_view_count": 3,
        },
    )
    assert result["hard_veto"] is False
    assert result["confirmation_allowed"] is False


def test_volumetric_cover_recovery_accepts_bounded_cylindrical_container() -> None:
    result = assess_physical_recovery(
        _recovery("cylindrical container"),
        _hard("contradiction", visible_form="cylindrical container"),
        verification=_verification("cylindrical container"),
        rejected_category="manhole cover",
        metric_dimensions_m=[0.338, 0.248, 0.600],
    )
    assert result["accepted"] is True
    assert result["category"] == "cylindrical container"
    assert result["maximum_semantic_tier"] == "probable"


def test_tall_work_surface_recovery_accepts_pipe_frame() -> None:
    result = assess_physical_recovery(
        _recovery("pipe frame"),
        _hard("contradiction"),
        verification=_verification("pipe frame"),
        rejected_category="workbench",
        metric_dimensions_m=[1.126, 0.502, 1.963],
    )
    assert result["accepted"] is True
    assert result["category"] == "pipe frame"


def test_false_standalone_part_is_suppressed_by_second_event() -> None:
    result = assess_physical_recovery(
        _recovery("metal frame"),
        _hard("part_only"),
        verification=_verification("metal frame", bounded=False),
        rejected_category="vehicle",
        metric_dimensions_m=[2.134, 0.197, 0.727],
    )
    assert result["accepted"] is False
    assert result["action"] == "suppress_geometry_only"
    assert result["maximum_semantic_tier"] == "geometry_only"


def test_vehicle_part_may_recover_as_bounded_visible_panel() -> None:
    result = assess_physical_recovery(
        _recovery("metal panel"),
        _hard("part_only", visible_form="panel"),
        verification=_verification("metal panel"),
        rejected_category="vehicle",
        metric_dimensions_m=[2.134, 0.197, 0.727],
    )
    assert result["accepted"] is True
    assert result["category"] == "panel"
    assert result["maximum_semantic_tier"] == "probable"


def test_recovery_cannot_repeat_rejected_noun_or_use_low_confidence() -> None:
    repeated = assess_physical_recovery(
        _recovery("workbench"),
        _hard(),
        verification=_verification("workbench"),
        rejected_category="workbench",
        metric_dimensions_m=[1.0, 0.8, 0.7],
    )
    weak = assess_physical_recovery(
        _recovery("pipe frame", confidence=0.84),
        _hard(),
        verification=_verification("pipe frame"),
        rejected_category="workbench",
        metric_dimensions_m=[1.0, 0.8, 1.8],
    )
    assert repeated["accepted"] is False
    assert "recovery_primary_repeats_rejected_category" in repeated["reason_codes"]
    assert "recovery_verification_repeats_rejected_category" in repeated["reason_codes"]
    assert weak["accepted"] is False
    assert "recovery_primary_confidence_below_threshold" in weak["reason_codes"]


def test_recovery_form_disagreement_fails_closed() -> None:
    result = assess_physical_recovery(
        _recovery("pipe frame"),
        _hard(),
        verification=_verification("ventilation fan"),
        rejected_category="workbench",
        metric_dimensions_m=[1.0, 0.8, 1.8],
    )
    assert result["accepted"] is False
    assert "recovery_visible_form_disagreement" in result["reason_codes"]


def test_recovery_missing_or_reused_views_fails_closed() -> None:
    result = assess_physical_recovery(
        _recovery("pipe frame", views=(1, 2)),
        _hard(),
        verification=_verification("pipe frame", views=(1, 2)),
        rejected_category="workbench",
        metric_dimensions_m=[1.0, 0.8, 1.8],
    )
    assert result["accepted"] is False
    assert "recovery_primary_insufficient_unique_views" in result["reason_codes"]
    assert "recovery_verification_insufficient_unique_views" in result["reason_codes"]
    assert "recovery_view_overlap_above_threshold" in result["reason_codes"]


def test_recovery_requires_metric_compatibility() -> None:
    result = assess_physical_recovery(
        _recovery("cover plate"),
        _hard(visible_form="cover plate"),
        verification=_verification("cover plate"),
        rejected_category="workbench",
        metric_dimensions_m=[0.5, 0.4, 0.8],
    )
    assert result["accepted"] is False
    assert "recovery_agreed_noun_metric_contradiction" in result["reason_codes"]


def test_exact_functional_noun_agreement_is_not_neutral_form_evidence() -> None:
    result = assess_physical_recovery(
        _recovery("pallet"),
        _hard(visible_form="slatted platform"),
        verification=_verification("pallet"),
        rejected_category="scaffold",
        metric_dimensions_m=[4.2, 1.3, 0.3],
    )
    assert result["accepted"] is False
    assert "recovery_agreed_noun_not_conservative_visible_form" in result["reason_codes"]


def test_candidate_conditioned_guard_never_introduces_replacement_noun() -> None:
    complete = assess_guard_visible_form_recovery(
        _hard(visible_form="metal wheeled platform"), [1.2, 0.8, 0.3]
    )
    partial = assess_guard_visible_form_recovery(
        _hard(
            visible_form="lattice structure",
            topology="partial_unbounded",
            complete_bounded=False,
        ),
        [2.0, 0.5, 4.0],
    )
    assert complete["accepted"] is False
    assert complete["category"] == "unresolved object"
    assert complete["maximum_semantic_tier"] == "geometry_only"
    assert "candidate_conditioned_guard_veto_only" in complete["reason_codes"]
    assert partial["accepted"] is False
    assert "candidate_conditioned_guard_veto_only" in partial["reason_codes"]


def test_recovery_rejects_disjoint_ids_from_near_duplicate_camera_poses() -> None:
    primary = _recovery("pipe frame", views=(1, 2, 3))
    verification = _verification("pipe frame", views=(4, 5, 6))
    for event, base in ((primary, 0.0), (verification, 1.0)):
        pose_index = {}
        for offset, image_id in enumerate(event["crop_image_ids"]):
            angle = math.radians(base + 0.2 * offset)
            center = [10.0 * math.cos(angle), 10.0 * math.sin(angle), 0.0]
            pose_index[image_id] = {
                "camera_center_world_m": center,
                "camera_forward_world": [-center[0] / 10.0, -center[1] / 10.0, 0.0],
            }
        event.update(crop_evidence_manifest(
            7,
            [(Path(f"img_{value:06d}_det_0001.npz"), b"jpeg") for value in event["crop_image_ids"]],
            frame_pose_index=pose_index,
            object_position_world_m=[0.0, 0.0, 0.0],
        ))
    result = assess_physical_recovery(
        primary,
        _hard(visible_form="pipe frame"),
        verification=verification,
        rejected_category="workbench",
        metric_dimensions_m=[1.1, 0.5, 2.0],
    )
    assert result["accepted"] is False
    assert "recovery_viewpoint_separation_below_threshold" in result["reason_codes"]
