from __future__ import annotations

import copy
import sys

import numpy as np
import pytest

from farm_runtime.obb_release_evidence import (
    OBBReleaseEvidencePolicy,
    evaluate_obb_release_evidence,
    validate_serialized_obb_release_evidence,
)
from scripts.geometry.refine_farm_object_geometry import (
    _independent_timestamp_projection_evidence,
    main as refine_geometry,
)


def _evaluate(**overrides: object) -> dict:
    values = {
        "geometry_fit_passed": True,
        "independent_physical_timestamps": 5,
        "timestamp_median_projected_box_iou": 0.70,
        "timestamp_q25_projected_box_iou": 0.60,
        "orientation_materiality": 0.50,
        "orientation_confidence": 0.80,
    }
    values.update(overrides)
    return evaluate_obb_release_evidence(**values)


def test_release_gate_accepts_only_complete_consistent_evidence() -> None:
    result = _evaluate()
    assert result["passed"] is True
    assert result["route"] == "release"
    assert result["reasons"] == []
    assert validate_serialized_obb_release_evidence(result) == []


@pytest.mark.parametrize(
    ("overrides", "expected_reason"),
    [
        (
            {"independent_physical_timestamps": 3},
            "insufficient_independent_physical_timestamps",
        ),
        (
            {
                "timestamp_median_projected_box_iou": 0.368,
                "timestamp_q25_projected_box_iou": 0.356,
            },
            "timestamp_median_projected_box_iou_below_release_limit",
        ),
        (
            {"orientation_materiality": 0.26, "orientation_confidence": 0.232},
            "material_orientation_confidence_below_release_limit",
        ),
    ],
)
def test_more_evidence_failures_route_to_view_rescue(
    overrides: dict[str, object], expected_reason: str
) -> None:
    result = _evaluate(**overrides)
    assert result["passed"] is False
    assert result["route"] == "view_rescue"
    assert expected_reason in result["reasons"]


def test_orientation_gate_is_skipped_when_yaw_is_not_material() -> None:
    result = _evaluate(
        orientation_materiality=0.10,
        orientation_confidence=0.01,
    )
    assert result["passed"] is True
    assert result["orientation_required"] is False


@pytest.mark.parametrize("materiality", [0.10, None])
def test_serialized_roundtrip_allows_absent_nonmaterial_orientation_confidence(
    materiality: float | None,
) -> None:
    result = _evaluate(
        orientation_materiality=materiality,
        orientation_confidence=None,
    )

    assert result["passed"] is True
    assert result["orientation_required"] is False
    assert validate_serialized_obb_release_evidence(result) == []


def test_serialized_material_orientation_still_requires_confidence() -> None:
    result = _evaluate()
    result["orientation_confidence"] = None

    assert validate_serialized_obb_release_evidence(result) == [
        "geometry_release_evidence_claim_contradicted"
    ]


def test_failed_provisional_fit_routes_to_refit_when_view_evidence_is_good() -> None:
    result = _evaluate(geometry_fit_passed=False)
    assert result["passed"] is False
    assert result["route"] == "obb_refit"
    assert result["reasons"] == ["geometry_fit_gate_not_passed"]


def test_serialized_pass_claim_is_recomputed_fail_closed() -> None:
    result = _evaluate()
    tampered = copy.deepcopy(result)
    tampered["independent_physical_timestamps"] = 2
    assert validate_serialized_obb_release_evidence(tampered) == [
        "geometry_release_evidence_claim_contradicted"
    ]
    assert validate_serialized_obb_release_evidence({}) == [
        "geometry_release_evidence_missing_or_invalid"
    ]

    weakened = copy.deepcopy(result)
    weakened["policy"]["minimum_independent_physical_timestamps"] = 2
    assert validate_serialized_obb_release_evidence(weakened) == [
        "geometry_release_evidence_policy_weaker_than_required"
    ]

    out_of_range = copy.deepcopy(result)
    out_of_range["timestamp_median_projected_box_iou"] = 1.1
    assert validate_serialized_obb_release_evidence(out_of_range) == [
        "geometry_release_evidence_claim_contradicted"
    ]


def _observation(timestamp: str, image_id: int, bbox: list[float]) -> dict:
    return {
        "physical_timestamp": timestamp,
        "image_id": image_id,
        "pose": np.eye(4, dtype=np.float64),
        "K": np.asarray(
            [[100.0, 0.0, 50.0], [0.0, 100.0, 50.0], [0.0, 0.0, 1.0]],
            dtype=np.float64,
        ),
        "raw_bbox": np.asarray(bbox, dtype=np.float64),
    }


def test_timestamp_evidence_dedupes_cameras_and_ignores_partial_sibling() -> None:
    center = np.asarray([0.0, 0.0, 5.0], dtype=np.float64)
    corners = np.asarray(
        [[x, y, z] for x in (-1.0, 1.0) for y in (-1.0, 1.0) for z in (4.0, 6.0)],
        dtype=np.float64,
    )
    observations = [
        _observation("100", 1, [20.0, 20.0, 80.0, 80.0]),
        _observation("100", 2, [47.0, 47.0, 53.0, 53.0]),
        _observation("200", 3, [22.0, 22.0, 78.0, 78.0]),
        _observation("invalid", 4, [20.0, 20.0, 80.0, 80.0]),
    ]
    evidence = _independent_timestamp_projection_evidence(
        center, corners, observations, support_iou=0.20
    )
    assert evidence["independent_physical_timestamps"] == 2
    assert evidence["missing_or_invalid_physical_timestamp_views"] == 1
    assert [row["image_id"] for row in evidence["timestamp_rows"]] == [1, 3]
    assert evidence["timestamp_q25_projected_box_iou"] > 0.60


def test_policy_rejects_invalid_thresholds() -> None:
    with pytest.raises(ValueError, match="thresholds"):
        _evaluate(
            policy=OBBReleaseEvidencePolicy(minimum_timestamp_q25_projected_box_iou=1.5)
        )


@pytest.mark.parametrize(
    ("option", "value"),
    [
        ("--release-min-independent-physical-timestamps", "3"),
        ("--release-min-timestamp-median-projected-box-iou", "0.49"),
        ("--release-min-timestamp-q25-projected-box-iou", "0.39"),
        ("--release-min-material-orientation-confidence", "0.34"),
        ("--orientation-materiality-threshold", "0.26"),
    ],
)
def test_cli_cannot_weaken_release_policy(
    monkeypatch, capsys, tmp_path, option: str, value: str
) -> None:
    argv = ["refine_farm_object_geometry.py"]
    for name in (
        "scene-state",
        "frames-json",
        "mask-dir",
        "output-state",
        "output-report",
    ):
        argv.extend([f"--{name}", str(tmp_path / name)])
    argv.extend([option, value])
    monkeypatch.setattr(sys, "argv", argv)

    with pytest.raises(SystemExit) as exc_info:
        refine_geometry()

    assert exc_info.value.code == 2
    assert "may be strengthened but not weakened" in capsys.readouterr().err
    assert list(tmp_path.iterdir()) == []
