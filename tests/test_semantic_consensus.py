from __future__ import annotations

import ast
import importlib.util
import json
import math
import sys
from pathlib import Path

from scene_graph.captioning.evidence import crop_evidence_manifest


ROOT = Path(__file__).resolve().parents[1]


def _load_script(module_name: str, relative: str):
    scripts = str(ROOT / "scripts")
    if scripts not in sys.path:
        sys.path.insert(0, scripts)
    path = ROOT / relative
    spec = importlib.util.spec_from_file_location(module_name, path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _load():
    return _load_script(
        "farm_semantic_finalizer", "scripts/finalize_farm_semantic_consensus.py"
    )


def _evidence(
    source: str,
    category: str,
    *,
    decision: str = "keep",
    event_id: str | None = None,
    views: list[int] | None = None,
) -> dict:
    row = {
        "source": source,
        "category": category,
        "description": category,
        "attributes": [],
        "confidence": 0.95,
        "decision": decision,
    }
    if event_id is not None:
        row["event_id"] = event_id
    if views is not None:
        pose_index = {}
        crops = []
        for image_id in views:
            angle = math.radians(float(image_id) * 20.0)
            center = [10.0 * math.cos(angle), 10.0 * math.sin(angle), 0.0]
            pose_index[image_id] = {
                "camera_center_world_m": center,
                "camera_forward_world": [-center[0] / 10.0, -center[1] / 10.0, 0.0],
            }
            crops.append((Path(f"img_{image_id:06d}_det_0001.npz"), b"jpeg"))
        row.update(crop_evidence_manifest(
            7,
            crops,
            frame_pose_index=pose_index,
            object_position_world_m=[0.0, 0.0, 0.0],
        ))
        row["confirmation_eligible"] = True
    return row


def _row(category: str, reason: str = "") -> dict:
    return {
        "category": category,
        "description": category,
        "semantic_gate_reason": reason,
        "candidates": [],
    }


def test_two_new_channels_may_replace_current_run_farm_vote() -> None:
    semantic = _load()
    category, reason = semantic.choose_category(
        _row("bed"), {}, _row("cabinet"), _row("cabinet"), {}
    )
    assert category == "cabinet"
    assert reason == "ensemble_and_blind_paired_crop_agree"


def test_neutral_fallback_cannot_replace_current_run_two_pass_vote() -> None:
    semantic = _load()
    category, reason = semantic.choose_category(
        _row("server"), {},
        _row("cabinet", "geometry_valid_neutral_last_resort"),
        _row("unknown"), {},
    )
    assert category == "server"
    assert reason == "two_pass_vote_protected_from_neutral_fallback"


def test_three_distinct_nonfallback_passes_recover_specific_category() -> None:
    semantic = _load()
    ensemble = _row("cabinet", "geometry_valid_neutral_last_resort")
    ensemble["candidates"] = [
        {"source": "blind_pass_a", "category": "server", "decision": "keep"},
        {"source": "blind_pass_b", "category": "server", "decision": "keep"},
        {"source": "multi_source_fusion", "category": "server", "decision": "keep"},
        {"source": "neutral_form_fallback", "category": "cabinet", "decision": "keep"},
    ]
    category, reason = semantic.choose_category({}, {}, ensemble, _row("unknown"), {})
    assert category == "server"
    assert reason == "three_distinct_model_passes_agree"


def test_review_catalog_accepts_only_kept_current_run_votes(tmp_path: Path) -> None:
    semantic = _load()
    path = tmp_path / "review.json"
    path.write_text(json.dumps({"objects": [
        {
            "id": 4,
            "review_category": "poster",
            "review_description": "printed safety sheet",
            "review_attributes": ["printed"],
            "review_confidence": 0.96,
            "review_decision": "keep",
        },
        {
            "id": 5,
            "review_category": "cabinet",
            "review_description": "uncertain enclosure",
            "review_confidence": 0.2,
            "review_decision": "unknown",
        },
    ]}), encoding="utf-8")

    rows = semantic.review_rows(path)
    assert rows[4]["category"] == "poster"
    assert rows[4]["description"] == "printed safety sheet"
    assert rows[5]["category"] == ""


def test_consensus_json_is_atomically_replaceable(tmp_path: Path) -> None:
    semantic = _load()
    path = tmp_path / "consensus.json"
    semantic.write_json_atomic(path, {"attempt": 1})
    semantic.write_json_atomic(path, {"attempt": 2})

    assert json.loads(path.read_text(encoding="utf-8")) == {"attempt": 2}
    assert not (tmp_path / ".consensus.json.tmp").exists()


def test_review_and_ensemble_aliases_are_one_inference_event() -> None:
    semantic = _load()
    prior = {"evidence": [
        _evidence(
            "initial_blind_review",
            "cabinet",
            event_id="object:8:initial_blind_review",
        )
    ]}
    ensemble = {"candidates": [_evidence("blind_pass_a", "cabinet")]}
    evidence = semantic.collect_semantic_evidence(prior, {}, ensemble, {})
    tier, supporting = semantic.semantic_tier_for("cabinet", evidence)
    assert len(evidence) == 1
    assert tier == "probable" and len(supporting) == 1


def test_genuinely_new_blind_event_may_confirm_alias_deduplicated_vote() -> None:
    semantic = _load()
    prior = {"evidence": [
        _evidence("initial_blind_review", "cabinet", views=[1, 2, 3])
    ]}
    ensemble = {"candidates": [
        _evidence("blind_pass_a", "cabinet", views=[1, 2, 3]),
        _evidence("blind_pass_b", "cabinet", views=[4, 5, 6]),
    ]}
    evidence = semantic.collect_semantic_evidence(prior, {}, ensemble, {})
    tier, supporting = semantic.semantic_tier_for("cabinet", evidence)
    assert len(evidence) == 2
    assert tier == "confirmed" and len(supporting) == 2


def test_nonkeep_and_derived_rows_are_not_positive_evidence() -> None:
    semantic = _load()
    ensemble = {"candidates": [
        _evidence("verification_pass", "cabinet", decision="unknown"),
        _evidence("multi_source_fusion", "cabinet"),
        _evidence("blind_pass_b", "cabinet"),
    ]}
    evidence = semantic.collect_semantic_evidence({}, {}, ensemble, {})
    tier, _ = semantic.semantic_tier_for("cabinet", evidence)
    assert [row["source"] for row in evidence] == ["blind_pass_b"]
    assert tier == "probable"


def test_single_paired_blind_vote_is_probable_not_confirmed() -> None:
    semantic = _load()
    blind = {"reconciliation_evidence": [
        _evidence("paired_crop_blind_adjudicator", "poster")
    ]}
    evidence = semantic.collect_semantic_evidence({}, {}, {}, blind)
    tier, supporting = semantic.semantic_tier_for("poster", evidence)
    assert tier == "probable" and len(supporting) == 1


def test_reconciler_filters_nonkeep_and_uses_provenance_for_tiers() -> None:
    semantic = _load_script(
        "farm_semantic_reconciler", "scripts/reconcile_farm_semantics.py"
    )
    keep = _evidence("blind_pass_b", "poster")
    rejected = _evidence("verification_pass", "sign", decision="unknown")
    evidence = semantic._candidate_evidence(
        {"id": 3, "candidates": [keep, rejected]}, {}, 0.80
    )
    assert [row["category"] for row in evidence] == ["poster"]
    one = semantic._final_row(
        {"id": 3}, keep, evidence, {"poster": 1.0}, "test"
    )
    assert one["semantic_tier"] == "probable"
    second = _evidence("blind_pass_c", "poster")
    two = semantic._final_row(
        {"id": 3}, keep, evidence + [second], {"poster": 2.0}, "test"
    )
    assert two["semantic_tier"] == "probable"
    none = semantic._final_row(
        {"id": 3}, keep, [], {"poster": 0.0}, "test"
    )
    assert none["semantic_tier"] == "geometry_only"
    assert none["category"] == "unresolved object"


def test_same_views_with_different_event_ids_cannot_confirm() -> None:
    semantic = _load()
    evidence = [
        _evidence("blind_pass_b", "cabinet", views=[1, 2, 3]),
        _evidence("blind_pass_c", "cabinet", views=[1, 2, 3]),
    ]
    tier, supporting, assessment = semantic.semantic_tier_assessment(
        "cabinet", evidence
    )
    assert tier == "probable"
    assert len(supporting) == 1
    assert assessment["confirmation_ready"] is False
    assert assessment["reason"] == "correlated_view_evidence"


def test_disjoint_six_view_support_can_confirm() -> None:
    semantic = _load()
    evidence = [
        _evidence("blind_pass_b", "cabinet", views=[1, 2, 3]),
        _evidence("blind_pass_c", "cabinet", views=[4, 5, 6]),
    ]
    tier, supporting, assessment = semantic.semantic_tier_assessment(
        "cabinet", evidence
    )
    assert tier == "confirmed"
    assert len(supporting) == 2
    assert assessment["selected_unique_view_count"] == 6


def test_tied_independent_category_is_fail_closed() -> None:
    semantic = _load()
    evidence = [
        _evidence("blind_pass_b", "cart", views=[1, 2, 3]),
        _evidence("blind_pass_c", "cabinet", views=[4, 5, 6]),
    ]
    tier, supporting, assessment = semantic.semantic_tier_assessment(
        "cart", evidence
    )
    assert tier == "geometry_only"
    assert supporting == []
    assert assessment["outvoted_or_tied"] is True


def test_physical_recovery_override_replaces_rejected_noun_but_never_confirms() -> None:
    semantic = _load()
    recovery = _evidence(
        "physical_form_recovery",
        "cylindrical container",
        views=[1, 2, 3],
    )
    recovery.update({
        "attributes": ["standalone_bounded"],
        "standalone_bounded": True,
        "confirmation_eligible": False,
    })
    verification = _evidence(
        "physical_form_recovery_verification",
        "cylindrical container",
        views=[4, 5, 6],
    )
    verification.update({
        "attributes": ["standalone_bounded"],
        "standalone_bounded": True,
        "confirmation_eligible": False,
    })
    blind = {
        "metric_dimensions_m": [0.338, 0.248, 0.600],
        "physical_form_rejected_semantics": {"category": "manhole cover"},
            "physical_form_rejected_assessment": {
                "hard_veto": True,
                "verdict": "contradiction",
                "model_guard": {
                    "verdict": "contradiction",
                    "decision": "keep",
                    "visible_form_category": "cylindrical container",
                    # Below the one-event guard-fallback threshold; the two
                    # diverse blind recovery events remain the tested path.
                    "visible_form_confidence": 0.89,
                    "topology": "standalone",
                    "complete_bounded": True,
                },
            },
        "physical_form_recovery": recovery,
        "physical_form_recovery_verification": verification,
    }
    override = semantic.physical_recovery_override(blind)
    assert override is not None
    assert override["action"] == "recover"
    assert override["category"] == "cylindrical container"
    assert "guard_fallback" not in override

    tier, _, assessment = semantic.semantic_tier_assessment(
        override["category"],
        [override["recovery"], override["verification"]],
    )
    assert tier == "probable"
    assert assessment["confirmation_ready"] is False
    evidence = semantic.collect_semantic_evidence({}, {}, {}, blind)
    assert {
        event["source"] for event in evidence
    } == {
        "physical_form_recovery",
        "physical_form_recovery_verification",
    }


def test_failed_part_only_recovery_forces_suppression_override() -> None:
    semantic = _load()
    blind = {
        "physical_form_rejected_semantics": {"category": "vehicle"},
        "physical_form_rejected_assessment": {
            "hard_veto": True,
            "verdict": "part_only",
        },
        "physical_form_recovery": {
            "source": "physical_form_recovery",
            "category": "component",
            "attributes": ["component_only"],
            "component_only": True,
            "confidence": 0.96,
            "decision": "unknown",
            "confirmation_eligible": False,
        },
    }
    override = semantic.physical_recovery_override(blind)
    assert override is not None
    assert override["action"] == "suppress"
    assert override["category"] == "unresolved object"


def test_standard_semantics_does_not_require_literal_category_equality() -> None:
    source = (ROOT / "scripts/farm_standard_stage.py").read_text(encoding="utf-8")
    start = source.index("    def semantics(self)")
    end = source.index("\n    def ", start + 8)
    block = source[start:end]
    assert '"--verify-kept"' not in block
    assert '"--require-category-consensus"' not in block
    assert '"--blind-adjudication"' not in block
    assert '"--enable-recovery-calls"' not in block


def test_every_standard_crop_review_receives_metric_frame_poses() -> None:
    source = (ROOT / "scripts/farm_standard_stage.py").read_text(encoding="utf-8")
    tree = ast.parse(source)
    invocations: list[ast.List] = []
    for node in ast.walk(tree):
        if not isinstance(node, ast.Call) or len(node.args) < 2:
            continue
        script = node.args[0]
        arguments = node.args[1]
        if not (
            isinstance(script, ast.Constant)
            and script.value == "review_farm_object_crops.py"
            and isinstance(arguments, ast.List)
        ):
            continue
        invocations.append(arguments)

    assert len(invocations) == 3
    for arguments in invocations:
        index = next(
            position
            for position, value in enumerate(arguments.elts)
            if isinstance(value, ast.Constant) and value.value == "--frames-json"
        )
        assert index + 1 < len(arguments.elts)
        frame_value = arguments.elts[index + 1]
        if isinstance(frame_value, ast.Constant):
            assert frame_value.value in {
                "/farm-run/rgbd/frames.json",
                "/farm-run/qa/full_colmap_rescue/combined/frames.json",
            }
        else:
            assert isinstance(frame_value, ast.Call)
            assert isinstance(frame_value.func, ast.Attribute)
            assert frame_value.func.attr == "direct_frames"
