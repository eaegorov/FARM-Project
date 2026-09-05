from __future__ import annotations

import json
import sys
from pathlib import Path

import cv2
import numpy as np
import pytest
import torch


SCRIPTS = Path(__file__).resolve().parents[1] / "scripts"
_INSERTED_SCRIPTS = str(SCRIPTS) not in sys.path
if _INSERTED_SCRIPTS:
    sys.path.insert(0, str(SCRIPTS))

from scripts.evaluation import analyze_farm_scene_quality as quality  # noqa: E402
from scripts.geometry.build_farm_object_assemblies import _fit_gravity_obb  # noqa: E402
from scripts.geometry.farm_geometry_axes import (  # noqa: E402
    horizontal_plane_basis,
    normalize_up_vector,
    resolve_up_policy,
    write_state_up_policy,
)
from scripts.geometry.refine_farm_compound_geometry import _fit_floor_obb  # noqa: E402
from scripts.geometry.refine_farm_object_geometry import (  # noqa: E402
    _anchored_voxel_component,
    _fit_robust_obb,
)

if _INSERTED_SCRIPTS:
    sys.path.remove(str(SCRIPTS))


def _box_points(up_vector: object) -> np.ndarray:
    up = normalize_up_vector(up_vector)
    plane = horizontal_plane_basis(up)
    basis = np.column_stack((plane[:, 0], plane[:, 1], up))
    axes = (
        np.linspace(-2.0, 2.0, 9),
        np.linspace(-0.8, 0.8, 7),
        np.linspace(-0.35, 0.35, 5),
    )
    local = np.stack(np.meshgrid(*axes, indexing="ij"), axis=-1).reshape(-1, 3)
    return (local @ basis.T + np.asarray([0.25, -0.5, 1.5])).astype(np.float32)


def _all_fits(points: np.ndarray, up: np.ndarray) -> list[dict]:
    return [
        _fit_robust_obb(
            points, 0.02, orientation_mode="gravity_yaw", up_vector=up
        ),
        _fit_gravity_obb(points, 0.02, up_vector=up),
        _fit_floor_obb(points, 0.02, up_vector=up),
    ]


@pytest.mark.parametrize("up", ([1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]))
def test_all_geometry_fitters_honor_each_axis(up: list[float]) -> None:
    up_np = np.asarray(up, dtype=np.float64)
    for result in _all_fits(_box_points(up_np), up_np):
        fitted = np.asarray(result["rotation_matrix"], dtype=np.float64)[:, 2]
        np.testing.assert_allclose(fitted, up_np, atol=1.0e-7, rtol=0.0)
        assert float(result.get("gravity_tilt_degrees", 0.0)) < 1.0e-5


def test_voxel_component_selection_anchors_to_rgbd_and_ignores_larger_floater() -> None:
    voxel_size = 0.05
    core_grid = np.stack(
        np.meshgrid(np.arange(-2, 3), np.arange(-2, 3), np.arange(-2, 3)),
        axis=-1,
    ).reshape(-1, 3)
    background_grid = np.stack(
        np.meshgrid(np.arange(6), np.arange(6), np.arange(6)),
        axis=-1,
    ).reshape(-1, 3)
    core = (core_grid + 0.5) * voxel_size
    background = (background_grid + 0.5) * voxel_size + np.asarray([3.0, 0.0, 0.0])
    base = {
        "center": np.zeros(3, dtype=np.float32),
        "dimensions": np.full(3, 0.35, dtype=np.float32),
        "rotation_matrix": np.eye(3, dtype=np.float32),
    }
    selected, diagnostics = _anchored_voxel_component(
        np.vstack((core, background)),
        base,
        voxel_size_m=voxel_size,
    )
    assert diagnostics["component_count"] == 2
    assert diagnostics["largest_component_fraction"] > 0.5
    assert diagnostics["selected_component_points"] == core.shape[0]
    assert diagnostics["selected_component_anchor_points"] > 0
    assert float(np.max(np.abs(selected[:, 0]))) < 0.2


def test_oblique_up_vector_is_exact_and_persisted_without_axis_rounding() -> None:
    up = normalize_up_vector([1.0, 2.0, 3.0])
    policy = resolve_up_policy(up_vector=up)
    assert policy.axis is None
    state: dict = {}
    write_state_up_policy(state, policy)
    assert state["object_geometry_up_axis"] == "vector"
    np.testing.assert_allclose(state["object_geometry_up_vector"], up, atol=1.0e-12)
    for result in _all_fits(_box_points(up), up):
        fitted = np.asarray(result["rotation_matrix"], dtype=np.float64)[:, 2]
        np.testing.assert_allclose(fitted, up, atol=1.0e-7, rtol=0.0)


def test_legacy_y_up_and_explicit_vector_y_are_bitwise_identical() -> None:
    points = _box_points([0.0, 1.0, 0.0])
    pairs = [
        (
            _fit_robust_obb(points, 0.02, orientation_mode="gravity_yaw", up_axis_index=1),
            _fit_robust_obb(
                points, 0.02, orientation_mode="gravity_yaw", up_axis_index=1,
                up_vector=[0.0, 1.0, 0.0],
            ),
        ),
        (
            _fit_gravity_obb(points, 0.02, up_axis_index=1),
            _fit_gravity_obb(points, 0.02, up_axis_index=1, up_vector=[0.0, 1.0, 0.0]),
        ),
        (
            _fit_floor_obb(points, 0.02, up_axis_index=1),
            _fit_floor_obb(points, 0.02, up_axis_index=1, up_vector=[0.0, 1.0, 0.0]),
        ),
    ]
    for implicit, explicit in pairs:
        for key in ("center", "dimensions", "rotation_matrix", "wxyz", "up_vector"):
            np.testing.assert_array_equal(implicit[key], explicit[key])


@pytest.mark.parametrize("invalid", ([0.0, 0.0, 0.0], [1.0, np.nan, 0.0], "1,2"))
def test_invalid_up_vectors_fail_closed(invalid: object) -> None:
    with pytest.raises(ValueError):
        normalize_up_vector(invalid)


def _valid_state() -> dict:
    return {
        "means": torch.tensor([[1.0, 2.0, 3.0]], dtype=torch.float32),
        "cov6": torch.tensor([[0.1, 0.0, 0.0, 0.1, 0.0, 0.1]], dtype=torch.float32),
        "active": torch.tensor([True]),
        "count": torch.tensor([3]),
        "object_id": torch.tensor([7]),
        "object_category": ["machine"],
        "object_supercategory": ["equipment"],
        "object_caption": ["compact industrial machine"],
        "object_qwen3_vl_embedding": [torch.ones(4)],
        "object_siglip2_embedding": [torch.ones(3)],
        "object_detection_category_conf": [{"machine": 0.9}],
        "loser_object_ids": [[]],
        "image_positions": [torch.tensor([0.0, 0.0, 0.0]), torch.tensor([1.0, 0.0, 1.0])],
    }


def test_quality_helpers_support_generic_and_legacy_selection() -> None:
    generic = quality.selection_summary(
        {"selection": {"selected_observations": 5, "selected_views": 10, "residual_flagged_edges": 0}}
    )
    legacy = quality.selection_summary(
        {"selection": {"selected_timestamps": 4, "selected_views": 8, "residual_flagged_edges": 0}}
    )
    assert generic["observations"] == 5
    assert generic["observation_source"] == "selected_observations"
    assert legacy["observations"] == 4
    assert legacy["observation_source"] == "selected_timestamps"


def test_quality_gates_use_configured_alignment_and_fail_on_missing_or_nan() -> None:
    valid = quality.alignment_summary(
        {
            "alignment_qa": {
                "passed": True,
                "thresholds": {"min_valid_depth_ratio": 0.80},
                "aggregate": {"min_valid_depth_ratio": 0.85},
            }
        }
    )
    assert valid["passed"] is True
    assert quality.alignment_summary({})["passed"] is False
    invalid = quality.alignment_summary(
        {
            "alignment_qa": {
                "passed": True,
                "thresholds": {"min_valid_depth_ratio": 0.80},
                "aggregate": {"min_valid_depth_ratio": float("nan")},
            }
        }
    )
    assert invalid["passed"] is False
    required_metric = quality.metric_summary(
        {"metric_scale": {"required": True, "passed": True, "meters_per_unit": float("nan")}},
        {},
        None,
    )
    assert required_metric["passed"] is False
    assert quality.metric_summary({}, {}, False)["passed"] is True


def test_mapping_state_health_rejects_missing_and_nonfinite_geometry() -> None:
    assert quality.mapping_state_health(_valid_state())["finite_geometry"] is True
    assert quality.mapping_state_health({})["structurally_valid"] is False
    state = _valid_state()
    state["means"][0, 0] = float("inf")
    health = quality.mapping_state_health(state)
    assert health["structurally_valid"] is True
    assert health["finite_geometry"] is False


def test_quality_records_use_refined_obb_centers() -> None:
    state = _valid_state()
    state["object_box_centers_m"] = torch.tensor([[4.0, 5.0, 6.0]])
    records, invariants = quality.object_records(state)
    assert records[0]["position_world_m"] == [4.0, 5.0, 6.0]
    assert invariants["nonfinite_object_box_centers"] == 0


def test_generic_quality_main_writes_source_backed_index_and_dashboard(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    state_path = tmp_path / "state.pt"
    frames_path = tmp_path / "frames.json"
    selection_path = tmp_path / "selection.json"
    prep_path = tmp_path / "prep.json"
    timing_path = tmp_path / "timing.json"
    resources_path = tmp_path / "resources.json"
    presentation_path = tmp_path / "presentation.json"
    output = tmp_path / "quality"
    torch.save({"state": _valid_state()}, state_path)
    frames_path.write_text(json.dumps({"frames": [{"image": "view.png"}]}), encoding="utf-8")
    selection_path.write_text(
        json.dumps(
            {
                "scene_id": "synthetic",
                "selection": {
                    "selected_observations": 3,
                    "selected_views": 6,
                    "residual_flagged_edges": 0,
                },
                "inputs": {"registered_images": 6},
            }
        ),
        encoding="utf-8",
    )
    prep_path.write_text(
        json.dumps(
            {
                "alignment_qa": {
                    "passed": True,
                    "thresholds": {"min_valid_depth_ratio": 0.80},
                    "aggregate": {"min_valid_depth_ratio": 0.85},
                },
                "metric_scale": {
                    "required": True,
                    "passed": True,
                    "meters_per_unit": 1.0,
                },
            }
        ),
        encoding="utf-8",
    )
    timing_path.write_text(
        json.dumps(
            {
                "total_seconds": 5.0,
                "stages": {
                    "selection": {"duration_seconds": 2.0},
                    "mapping": {"duration_seconds": 3.0},
                },
            }
        ),
        encoding="utf-8",
    )
    resources_path.write_text(
        json.dumps(
            {
                "measurement_scope": "whole_selected_device_not_pid_attributed",
                "device_peak_used_gib": 24.0,
                "incremental_peak_delta_gib": 20.0,
                "stages": {
                    "selection": {
                        "measurement_scope": "whole_selected_device_not_pid_attributed",
                        "device_peak_used_gib": 2.0,
                        "incremental_peak_delta_gib": 1.0,
                    },
                    "mapping": {
                        "measurement_scope": "whole_selected_device_not_pid_attributed",
                        "device_peak_used_gib": 24.0,
                        "incremental_peak_delta_gib": 20.0,
                    },
                },
            }
        ),
        encoding="utf-8",
    )
    presentation_path.write_text(json.dumps([{"id": 7}]), encoding="utf-8")
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "analyze_farm_scene_quality.py",
            "--pt", str(state_path),
            "--frames-json", str(frames_path),
            "--selection-manifest", str(selection_path),
            "--prep-summary", str(prep_path),
            "--timing-summary", str(timing_path),
            "--resource-summary", str(resources_path),
            "--presentation-catalog", str(presentation_path),
            "--output-dir", str(output),
            "--no-require-semantics",
        ],
    )
    quality.main()
    report = json.loads((output / "quality_summary.json").read_text(encoding="utf-8"))
    index = json.loads((output / "quality_index.json").read_text(encoding="utf-8"))
    assert report["schema"] == "farm.scene-quality.v2"
    assert report["qa_status"] == "pass"
    assert index["metrics"]["total_time_seconds"]["value"] == 5.0
    assert index["metrics"]["device_peak_used_gib"]["value"] == 24.0
    assert index["metrics"]["incremental_peak_delta_gib"]["value"] == 20.0
    assert (
        index["metrics"]["gpu_memory_measurement_scope"]["value"]
        == "whole_selected_device_not_pid_attributed"
    )
    assert "peak_vram_gib" not in index["metrics"]
    assert index["metrics"]["presentation_visible_objects"]["value"] == 1
    assert index["metrics"]["presentation_visible_objects"]["source"].endswith(
        "presentation.json#$"
    )
    assert (
        report["run_profile"]["memory_measurement_scope"]
        == "whole_selected_device_not_pid_attributed"
    )
    assert index["metrics"]["selected_views"]["source"].endswith(
        "selection.json#$.selection.selected_views"
    )
    assert (output / "01_evidence_tier_dashboard_4k.jpg").stat().st_size > 10_000
    dashboard = cv2.imread(str(output / "01_evidence_tier_dashboard_4k.jpg"))
    assert dashboard is not None and dashboard.shape[:2] == (2160, 3840)
    readme = (output / "README.md").read_text(encoding="utf-8").lower()
    assert "factory" not in readme
    assert "not process-attributed" in readme


def test_chart_stage_rows_are_bounded_and_keep_highest_values() -> None:
    rows = [
        {"stage": f"stage_{index}", "duration_seconds": float(index)}
        for index in range(1, 17)
    ]
    selected, total = quality.select_chart_stage_rows(rows, "duration_seconds", 8)
    assert total == 16
    assert len(selected) == 8
    assert {row["stage"] for row in selected} == {
        f"stage_{index}" for index in range(9, 17)
    }


def test_acceptance_quality_check_is_fail_closed_and_count_bound() -> None:
    valid = {
        "status": "WARN",
        "counts": {
            "presentation_after": 12,
            "remaining_auto_clusters": 0,
            "remaining_blocking_clusters": 0,
            "label_hard_errors": 0,
        },
    }
    passed, detail = quality.acceptance_release_check(valid, 12)
    assert passed is True and detail["reasons"] == []
    invalid = {**valid, "counts": {**valid["counts"], "remaining_blocking_clusters": 1}}
    passed, detail = quality.acceptance_release_check(invalid, 11)
    assert passed is False
    assert set(detail["reasons"]) == {
        "remaining_blocking_clusters_nonzero",
        "presentation_after_catalog_mismatch",
    }


def test_duplicate_summary_applies_semantic_visibility_contract() -> None:
    state = {
        "active": torch.tensor([True, True, True]),
        "object_display_status": ["canonical", "canonical", "duplicate_suppressed"],
        "object_duplicate_canonical_id": torch.tensor([-1, -1, 1]),
        "object_geometry_status": ["geometry_pass"] * 3,
        "object_semantic_tier": ["confirmed", "geometry_only", "confirmed"],
        "object_compound_boxes": [[], [], []],
    }
    derived = quality.duplicate_summary(state)
    authoritative = quality.duplicate_summary(state, presentation_count=7)
    assert derived["presentation_visible"] == 1
    assert authoritative["presentation_visible"] == 7
    assert authoritative["presentation_count_authoritative"] is True


def test_quality_profile_reads_legacy_peak_without_claiming_process_scope(
    tmp_path: Path,
) -> None:
    resource_path = tmp_path / "legacy-resource.json"
    profile = quality.run_profile(
        {},
        None,
        {"peak_vram_gib": 12.0},
        resource_path,
    )
    assert profile["device_peak_used_gib"] == 12.0
    assert profile["incremental_peak_delta_gib"] is None
    assert (
        profile["memory_measurement_scope"]
        == "legacy_unspecified_not_assumed_process_attributed"
    )
    assert (
        profile["memory_measurement_scope_source"]
        == "derived:legacy GPU-memory key without scope"
    )
