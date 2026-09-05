from __future__ import annotations

import importlib.util
from pathlib import Path

import numpy as np
import pytest


ROOT = Path(__file__).resolve().parents[1]


def _load():
    spec = importlib.util.spec_from_file_location(
        "farm_scene_relation_graph",
        ROOT / "scripts/geometry/build_farm_scene_relation_graph.py",
    )
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _geometry(object_id: int, center=(0.0, 0.0, 0.0), dimensions=(1.0, 1.0, 1.0)):
    return {
        "id": object_id,
        "center_m": list(center),
        "dimensions_m": list(dimensions),
        "wxyz": [1.0, 0.0, 0.0, 0.0],
    }


def _event(category: str, *, eligible=True, whole=True):
    return {
        "confirmation_eligible": eligible,
        "label_contract": {
            "category": category,
            "topology": "standalone_whole" if whole else "partial_unbounded",
            "complete_bounded": whole,
            "context_sufficient": True,
            "visible_target_coverage": 0.98 if whole else 0.6,
            "description": category,
        },
    }


def _semantic(object_id: int, category="cabinet", description="cabinet on casters"):
    return {
        "id": object_id,
        "category": category,
        "description": description,
        "semantic_evidence": [_event(category), _event(category)],
    }


def _heldout(object_id: int, *, timestamps=3, precision=0.8, recall=0.8, iou=0.75):
    def metric(value):
        return {"minimum": value, "median": value}
    return {
        "object_id": object_id,
        "status": "verified",
        "heldout_timestamps": timestamps,
        "good_timestamps": timestamps,
        "provisional_gaussians": 1000,
        "summaries": {
            "iou": metric(iou),
            "precision": metric(precision),
            "recall": metric(recall),
            "largest_component_fraction": metric(1.0),
        },
    }


def test_quaternion_identity_produces_expected_corners() -> None:
    module = _load()
    corners = module.obb_corners(_geometry(1, center=(1, 2, 3), dimensions=(2, 4, 6)))
    assert corners.min(axis=0).tolist() == [0.0, 0.0, 0.0]
    assert corners.max(axis=0).tolist() == [2.0, 4.0, 6.0]


def test_pair_geometry_separates_near_and_overlap() -> None:
    module = _load()
    overlap = module.pair_geometry(_geometry(1), _geometry(2, center=(0.5, 0, 0)))
    near = module.pair_geometry(_geometry(1), _geometry(2, center=(1.1, 0, 0)))
    assert overlap["aabb_smaller_overlap"] == 0.5
    assert near["aabb_smaller_overlap"] == 0.0
    assert np.isclose(near["aabb_gap_m"], 0.1)


def test_alias_normalization_prevents_palet_typo_conflict() -> None:
    module = _load()
    row = _semantic(1, category="palet")
    row["semantic_evidence"] = [_event("palet"), _event("pallet")]
    summary = module.semantic_summary(row)
    assert summary["final_category"] == "pallet"
    assert summary["eligible_categories"] == ["pallet"]
    assert summary["normalization_applied"] is True


def test_strong_standalone_object_routes_to_relation_verification() -> None:
    module = _load()
    graph = module.build_graph(
        [_semantic(1)], [_geometry(1)], [_heldout(1)], [], {1},
    )
    assert graph["queues"]["relation_verification_ids"] == [1]
    assert graph["queues"]["mask_refinement_ids"] == []


def test_worst_view_recall_routes_to_mask_refinement() -> None:
    module = _load()
    graph = module.build_graph(
        [_semantic(1)], [_geometry(1)], [_heldout(1, recall=0.4)], [], {1},
    )
    assert graph["queues"]["mask_refinement_ids"] == [1]
    assert "worst_view_recall_below_threshold" in graph["nodes"][0]["decision_reasons"]


def test_final_label_mismatch_routes_to_refinement() -> None:
    module = _load()
    row = _semantic(1, category="chiller")
    row["semantic_evidence"] = [_event("cabinet"), _event("cabinet")]
    graph = module.build_graph([row], [_geometry(1)], [_heldout(1)], [], {1})
    assert graph["queues"]["semantic_verification_ids"] == [1]
    assert graph["queues"]["mask_refinement_ids"] == []
    assert "final_category_not_supported_by_eligible_events" in graph["nodes"][0]["decision_reasons"]


def test_explicit_wall_attachment_fails_closed() -> None:
    module = _load()
    row = _semantic(1, description="cabinet mounted on a wall")
    graph = module.build_graph([row], [_geometry(1)], [_heldout(1)], [], {1})
    assert graph["queues"]["excluded_attached_ids"] == [1]
    assert graph["queues"]["inpainting_exclusion_ids"] == [1]
    assert "explicit_wall_or_ceiling_attachment" in graph["nodes"][0]["decision_reasons"]


def test_floor_attachment_is_reviewed_instead_of_dropped() -> None:
    module = _load()
    row = _semantic(1, description="industrial chiller floor-mounted next to a cabinet")
    graph = module.build_graph([row], [_geometry(1)], [_heldout(1)], [], {1})
    assert graph["queues"]["mask_refinement_ids"] == []
    assert graph["queues"]["relation_verification_ids"] == [1]
    assert graph["queues"]["inpainting_exclusion_ids"] == [1]
    assert "floor_attachment_requires_relation_review" in graph["nodes"][0]["decision_reasons"]


def test_ambiguous_cleaner_routes_to_semantics_without_mask_work() -> None:
    module = _load()
    row = _semantic(1, category="cleaner")
    graph = module.build_graph([row], [_geometry(1)], [_heldout(1)], [], {1})
    assert graph["queues"]["semantic_verification_ids"] == [1]
    assert graph["queues"]["mask_geometry_refinement_ids"] == []
    assert graph["queues"]["full_colmap_processing_ids"] == []
    assert "category_is_not_specific_enough_for_publication" in (
        graph["nodes"][0]["decision_reasons"]
    )


def test_previous_audit_can_require_independent_evidence() -> None:
    module = _load()
    graph = module.build_graph(
        [_semantic(1)], [_geometry(1)], [_heldout(1)],
        [{"object_id": 1, "status": "needs_independent_evidence"}], {1},
    )
    assert graph["queues"]["independent_verification_ids"] == [1]

def test_candidate_id_file_is_exact_and_comment_tolerant(tmp_path: Path) -> None:
    module = _load()
    candidate_file = tmp_path / "candidate_ids.txt"
    candidate_file.write_text("# audit queue\n185\n16\n", encoding="utf-8")

    assert module.load_candidate_ids(None, candidate_file) == {16, 185}


@pytest.mark.parametrize("payload", ["", "16\n16\n", "-1\n", "object-16\n"])
def test_candidate_id_file_fails_closed_on_invalid_input(
    tmp_path: Path, payload: str
) -> None:
    module = _load()
    candidate_file = tmp_path / "candidate_ids.txt"
    candidate_file.write_text(payload, encoding="utf-8")

    with pytest.raises(ValueError):
        module.load_candidate_ids(None, candidate_file)
