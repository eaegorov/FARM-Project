from __future__ import annotations

import importlib.util
import math
import sys
from pathlib import Path

from scene_graph.captioning.evidence import crop_evidence_manifest


ROOT = Path(__file__).resolve().parents[1]


def _load():
    scripts = str(ROOT / "scripts")
    if scripts not in sys.path:
        sys.path.insert(0, scripts)
    spec = importlib.util.spec_from_file_location(
        "farm_adjudicate_upright_folds",
        ROOT / "scripts/semantics/adjudicate_farm_semantics.py",
    )
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _manifest(source: str, image_ids: tuple[int, ...]) -> dict:
    return {
        "source": source,
        "event_id": f"object:7:{source}",
        **crop_evidence_manifest(
            7,
            [
                (Path(f"img_{image_id:06d}_det_0000.npz"), b"jpeg")
                for image_id in image_ids
            ],
        ),
    }


def _rolled_pose_index(turns_clockwise: int) -> dict:
    angle = math.radians(90.0 * turns_clockwise)
    return {
        9: {
            "source_image": "cam00_000009.png",
            "camera_center_world_m": [0.0, 0.0, 0.0],
            "camera_forward_world": [0.0, 0.0, 1.0],
            "camera_right_world": [math.cos(angle), math.sin(angle), 0.0],
            "camera_down_world": [math.sin(angle), -math.cos(angle), 0.0],
        }
    }


def _source_index(*image_ids: int) -> dict:
    return {
        int(image_id): {"source_image": f"camera_{image_id:06d}.png"}
        for image_id in image_ids
    }


def test_preferred_initial_subset_overlap_blocks_all_mask_alternate() -> None:
    adjudicate = _load()
    # The initial review may have selected a preferred source subset, while the
    # adjudicator sees a larger mask bank. Reusing even one physical image is a
    # hard failure; silently dropping that image would change the fixed fold.
    initial = _manifest("initial_blind_review", (2, 8, 14))
    alternate = _manifest("blind_pass_c", (3, 8, 15))

    audit = adjudicate.alternate_blind_fold_audit(
        initial,
        alternate,
        frame_pose_index=_source_index(2, 3, 8, 14, 15),
    )
    gated = adjudicate.apply_alternate_fold_gate(
        {"request_alternate": True, "reason": "unresolved_or_low_confidence"},
        audit,
    )

    assert audit["eligible"] is False
    assert audit["overlap_image_ids"] == [8]
    assert "blind_fold_source_image_overlap" in audit["reason_codes"]
    assert gated["request_alternate_before_fold_gate"] is True
    assert gated["request_alternate"] is False
    assert gated["reason"] == "alternate_blind_fold_not_provably_disjoint"


def test_contextual_dual_panel_event_retains_initial_blind_provenance() -> None:
    adjudicate = _load()
    event = _manifest("contextual_dual_panel_review", (2, 8, 14))
    row = {
        "semantic_evidence": [event],
        "review_initial": {"category": "legacy_without_crop_provenance"},
    }

    selected = adjudicate.candidate_from_review("initial", row)

    assert selected == event
    assert selected["crop_image_ids_complete"] is True


def test_provenance_incomplete_or_empty_alternate_fails_closed() -> None:
    adjudicate = _load()
    initial = {
        "source": "blind_pass_a",
        "event_id": "legacy",
        "crop_image_ids": [1, 2, 3],
    }
    alternate = _manifest("blind_pass_c", ())

    audit = adjudicate.alternate_blind_fold_audit(initial, alternate)

    assert audit["status"] == "unavailable"
    assert "initial_crop_provenance_incomplete" in audit["reason_codes"]
    assert "alternate_crop_image_ids_empty" in audit["reason_codes"]


def test_disjoint_complete_blind_folds_remain_eligible() -> None:
    adjudicate = _load()
    initial = _manifest("initial_blind_review", (1, 4, 7))
    alternate = _manifest("blind_pass_c", (2, 5, 8))

    audit = adjudicate.alternate_blind_fold_audit(
        initial,
        alternate,
        frame_pose_index=_source_index(1, 2, 4, 5, 7, 8),
    )
    gated = adjudicate.apply_alternate_fold_gate(
        {"request_alternate": True, "reason": "unresolved_or_low_confidence"},
        audit,
    )

    assert audit["eligible"] is True
    assert audit["status"] == "disjoint"
    assert audit["overlap_image_ids"] == []
    assert gated["request_alternate"] is True


def test_different_mapping_ids_for_same_physical_image_are_rejected() -> None:
    adjudicate = _load()
    initial = _manifest("initial_blind_review", (1, 4, 7))
    alternate = _manifest("blind_pass_c", (2, 5, 8))
    source_index = _source_index(1, 2, 4, 5, 7, 8)
    source_index[8]["source_image"] = source_index[7]["source_image"]

    audit = adjudicate.alternate_blind_fold_audit(
        initial,
        alternate,
        frame_pose_index=source_index,
    )

    assert audit["eligible"] is False
    assert audit["overlap_image_ids"] == []
    assert audit["overlap_source_images"] == ["camera_000007.png"]
    assert "blind_fold_physical_source_overlap" in audit["reason_codes"]


def test_blind_pass_c_event_contains_gravity_upright_provenance(monkeypatch) -> None:
    adjudicate = _load()

    def fake_request(*_args, **_kwargs):
        return {
            "review_category": "pallet",
            "review_description": "bounded slotted platform",
            "review_attributes": ["slotted base"],
            "review_confidence": 0.96,
            "review_decision": "keep",
            "review_raw_response": "{}",
            "review_label_contract": {"contract_valid": True},
        }

    monkeypatch.setattr(adjudicate, "request_review", fake_request)
    crops = [(Path("img_000009_det_0000.npz"), b"jpeg")]
    event = adjudicate.request_candidate(
        "http://unused",
        "qwen",
        {"id": 7, "position_world_m": [0.0, 0.0, 2.0]},
        crops,
        adjudicate.TIEBREAKER_PROMPT,
        "blind_pass_c",
        frame_pose_index=_rolled_pose_index(2),
        world_up_vector=[0.0, 1.0, 0.0],
        world_up_source="geometry_report:$.up.vector",
    )

    upright = event["gravity_upright_normalization"]
    assert upright["schema"] == "farm.gravity-upright-normalization.v1"
    assert upright["provenance_complete"] is True
    assert len(upright["fingerprint_sha256"]) == 64
    assert upright["crops"][0]["source_orientation"] == "upside_down"
    assert upright["crops"][0]["applied_quarter_turns_ccw"] == 2
    assert upright["crops"][0]["source_image"] == "cam00_000009.png"
    assert event["source"] == "blind_pass_c"
    assert event["confirmation_eligible"] is True
