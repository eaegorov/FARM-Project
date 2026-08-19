from __future__ import annotations

import importlib.util
import math
import sys
from pathlib import Path

import cv2
import numpy as np


ROOT = Path(__file__).resolve().parents[1]


def _pose_index(image_ids) -> dict:
    result = {}
    for image_id in image_ids:
        angle = math.radians(float(image_id) * 20.0)
        center = [10.0 * math.cos(angle), 10.0 * math.sin(angle), 0.0]
        result[int(image_id)] = {
            "camera_center_world_m": center,
            "camera_forward_world": [-center[0] / 10.0, -center[1] / 10.0, 0.0],
        }
    return result


def _load(name: str, relative: str):
    scripts = str(ROOT / "scripts")
    if scripts not in sys.path:
        sys.path.insert(0, scripts)
    spec = importlib.util.spec_from_file_location(name, ROOT / relative)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _candidate(source: str, category: str) -> dict:
    return {
        "source": source,
        "category": category,
        "description": category,
        "attributes": [],
        "confidence": 0.95,
        "decision": "keep",
    }


def _crop_row(
    image_id: int,
    detail: int,
    *,
    member: int | None = None,
) -> tuple[Path, bytes]:
    member_token = f"_member_{member:06d}" if member is not None else ""
    return (
        Path(
            f"/run/masks/object_000007/img_{image_id:06d}"
            f"{member_token}_det_0000.npz"
        ),
        b"x" * detail,
    )


def _crop_ids(rows) -> set[int]:
    return {
        int(row[0].name.split("img_", 1)[1].split("_", 1)[0])
        for row in rows
    }


def test_crop_candidates_isolate_saved_mask_on_neutral_background(tmp_path: Path) -> None:
    review = _load("farm_mask_grounding", "scripts/review_farm_object_crops.py")
    object_dir = tmp_path / "object_000051"
    object_dir.mkdir()
    image = np.full((20, 20, 3), 245, dtype=np.uint8)
    image[8:12, 8:12] = np.asarray([10, 20, 230], dtype=np.uint8)
    ok, encoded = cv2.imencode(".jpg", image)
    assert ok
    raw = np.ones((4, 4), dtype=np.uint8)
    np.savez_compressed(
        object_dir / "img_000001_det_0001.npz",
        crop_jpeg_bytes=np.asarray(encoded, dtype=np.uint8),
        raw_bits=np.packbits(raw.reshape(-1), bitorder="little"),
        raw_shape=np.asarray(raw.shape, dtype=np.int32),
        crop_bbox_xyxy=np.asarray([8, 8, 12, 12], dtype=np.int32),
    )

    rows = review.crop_candidates([object_dir / "img_000001_det_0001.npz"])
    assert len(rows) == 1
    grounded = cv2.imdecode(np.frombuffer(rows[0][1], dtype=np.uint8), cv2.IMREAD_COLOR)
    assert grounded is not None
    assert float(grounded[:5, :5].mean()) < 60.0
    assert int(grounded[9:11, 9:11, 2].mean()) > 150


def test_crop_candidates_do_not_expand_into_neighbouring_source_frame(tmp_path: Path) -> None:
    review = _load("farm_tight_semantic_crop", "scripts/review_farm_object_crops.py")
    object_dir = tmp_path / "mapping" / "masks" / "object_000007"
    object_dir.mkdir(parents=True)
    image = np.full((20, 24, 3), 190, dtype=np.uint8)
    ok, encoded = cv2.imencode(".jpg", image)
    assert ok
    mask = np.ones((10, 12), dtype=np.uint8)
    np.savez_compressed(
        object_dir / "img_000000_det_0000.npz",
        crop_jpeg_bytes=np.asarray(encoded, dtype=np.uint8),
        raw_bits=np.packbits(mask.reshape(-1), bitorder="little"),
        raw_shape=np.asarray(mask.shape, dtype=np.int32),
        crop_bbox_xyxy=np.asarray([6, 5, 18, 15], dtype=np.int32),
        raw_bbox_xyxy=np.asarray([30, 30, 42, 40], dtype=np.int32),
    )

    rows = review.crop_candidates([object_dir / "img_000000_det_0000.npz"])
    grounded = cv2.imdecode(np.frombuffer(rows[0][1], dtype=np.uint8), cv2.IMREAD_COLOR)
    assert grounded is not None
    assert grounded.shape[:2] == image.shape[:2]


def test_metric_dimensions_are_added_as_scale_evidence_without_class_hint() -> None:
    review = _load("farm_metric_semantic_evidence", "scripts/review_farm_object_crops.py")
    prompt = review.prompt_with_metric_evidence(
        "Identify the object.", {"metric_dimensions_m": [0.76163, 0.48061, 1.04]}
    )
    assert "L × W × H" in prompt
    assert "0.762 × 0.481 × 1.040 metres" in prompt
    assert "third value is the gravity-aligned height" in prompt
    assert "physical-scale sanity check" in prompt
    assert "cabinet" not in prompt and "book" not in prompt


def test_candidate_conditioned_verification_may_refine_open_vocabulary_noun() -> None:
    review = _load(
        "farm_verification_refinement", "scripts/review_farm_object_crops.py"
    )
    reason = review.verification_gate_reason(
        {"category": "sign"}, {"review_category": "poster"}
    )
    assert reason == "candidate_conditioned_verification_revised_category"


def test_invalid_metric_dimensions_are_not_added_to_prompt() -> None:
    review = _load("farm_invalid_metric_semantic_evidence", "scripts/review_farm_object_crops.py")
    assert review.prompt_with_metric_evidence(
        "Identify the object.", {"metric_dimensions_m": [1.0, 0.0, 2.0]}
    ) == "Identify the object."


def test_initial_crop_selection_uses_pose_diversity_additively() -> None:
    review = _load(
        "farm_pose_aware_initial_selection",
        "scripts/review_farm_object_crops.py",
    )
    candidates = [
        _crop_row(0, 100),
        _crop_row(1, 100),
        _crop_row(2, 100),
        _crop_row(3, 80),
        _crop_row(4, 80),
    ]
    poses = _pose_index((0, 1, 2, 3, 4))

    selected, partition = review.choose_evidence_crops(
        candidates,
        "initial_blind_review",
        3,
        frame_pose_index=poses,
        object_position_world_m=[0.0, 0.0, 0.0],
        pose_aware=True,
    )
    unchanged, unchanged_partition = review.choose_evidence_crops(
        candidates,
        "independent_verification",
        3,
        frame_pose_index=poses,
        object_position_world_m=[0.0, 0.0, 0.0],
    )

    assert _crop_ids(selected) == {0, 1, 2}
    assert partition["selection_diagnostics"]["method"] == "pose_aware_exact"
    assert partition["selection_diagnostics"]["diversity_gate_met"] is True
    assert unchanged == review.choose_crops(
        sorted(candidates, key=lambda item: len(item[1]), reverse=True),
        3,
    )
    assert "selection_diagnostics" not in unchanged_partition


def test_pose_selection_preserves_assembly_member_balancing() -> None:
    review = _load(
        "farm_pose_aware_assembly_selection",
        "scripts/review_farm_object_crops.py",
    )
    candidates = [
        _crop_row(0, 100, member=1),
        _crop_row(1, 95, member=1),
        _crop_row(2, 90, member=1),
        _crop_row(3, 85, member=2),
        _crop_row(4, 80, member=2),
    ]
    plain, _ = review.choose_evidence_crops(
        candidates,
        "initial_blind_review",
        3,
    )
    selected, partition = review.choose_evidence_crops(
        candidates,
        "initial_blind_review",
        3,
        frame_pose_index=_pose_index((0, 1, 2, 3, 4)),
        object_position_world_m=[0.0, 0.0, 0.0],
        pose_aware=True,
    )

    assert selected == plain
    assert partition["selection_diagnostics"]["method"] == "temporal_fallback"
    assert (
        partition["selection_diagnostics"]["reason"]
        == "assembly_member_balancing_preserved"
    )


def test_negative_review_cannot_be_overridden_by_one_later_guess() -> None:
    semantic = _load("farm_semantic_negative", "scripts/adjudicate_farm_semantics.py")
    tier, chosen, reason = semantic.independent_evidence_decision(
        [_candidate("blind_pass_b", "tool")], ["strict_two_pass"], 0.9
    )
    assert tier == "geometry_only"
    assert chosen is None
    assert reason == "negative_independent_evidence"


def test_same_model_repeated_votes_do_not_override_unknown_evidence() -> None:
    semantic = _load("farm_semantic_consensus", "scripts/adjudicate_farm_semantics.py")
    tier, chosen, reason = semantic.independent_evidence_decision(
        [
            _candidate("blind_pass_a", "cabinet"),
            _candidate("blind_pass_c", "cabinet"),
        ],
        ["verification_pass"],
        0.9,
    )
    assert tier == "geometry_only"
    assert chosen is None
    assert reason == "negative_independent_evidence"


def test_two_matching_votes_without_negative_evidence_are_confirmed() -> None:
    semantic = _load("farm_semantic_positive_consensus", "scripts/adjudicate_farm_semantics.py")
    tier, chosen, reason = semantic.independent_evidence_decision(
        [_candidate("blind_pass_a", "cabinet"), _candidate("blind_pass_b", "cabinet")],
        [],
        0.9,
    )
    assert tier == "probable"
    assert chosen is not None and chosen["category"] == "cabinet"
    assert reason == "correlated_or_single_high_confidence_evidence"


def test_conflicting_labels_are_not_confirmed_without_independent_majority() -> None:
    semantic = _load("farm_semantic_conflict", "scripts/adjudicate_farm_semantics.py")
    tier, chosen, reason = semantic.independent_evidence_decision(
        [
            _candidate("blind_pass_a", "cabinet"),
            _candidate("blind_pass_b", "bed"),
        ],
        [],
        0.9,
    )
    assert tier == "geometry_only"
    assert chosen is None
    assert reason == "conflicting_independent_evidence"


def test_neutral_fallback_cannot_override_any_informative_vote() -> None:
    semantic = _load("farm_neutral_last_resort", "scripts/adjudicate_farm_semantics.py")
    assert semantic.neutral_fallback_allowed([]) is True
    assert semantic.neutral_fallback_allowed([_candidate("blind_pass_a", "server")]) is False


def test_physical_recovery_request_is_blind_provenance_complete_and_ineligible(
    tmp_path: Path,
    monkeypatch,
) -> None:
    semantic = _load(
        "farm_physical_recovery", "scripts/reconcile_farm_semantics.py"
    )
    captured: dict = {}

    def fake_request(url, model, row, crops, prompt, **kwargs):
        captured.update({"row": row, "prompt": prompt, "crops": crops})
        return {
            "review_category": "pipe frame",
            "review_description": "bounded frame holding visible pipes",
            "review_attributes": ["standalone_bounded", "metal frame"],
            "review_confidence": 0.93,
            "review_decision": "keep",
            "review_raw_response": "{}",
        }

    monkeypatch.setattr(semantic, "request_review", fake_request)
    crops = [
        (tmp_path / f"img_{value:06d}_det_0001.npz", b"jpeg")
        for value in (4, 7, 10)
    ]
    event = semantic.physical_recovery_event(
        "http://unused",
        "model",
        {"id": 1, "category": "workbench", "metric_dimensions_m": [1, 1, 2], "position_world_m": [0, 0, 0]},
        crops,
        {
            "partition_id": "view-fold-2-of-3",
            "partition_index": 2,
            "partition_count": 3,
        },
        _pose_index((4, 7, 10)),
    )

    assert event["source"] == "physical_form_recovery"
    assert event["confirmation_eligible"] is False
    assert event["crop_image_ids"] == [4, 7, 10]
    assert event["crop_partition_id"] == "view-fold-2-of-3"
    assert event["standalone_bounded"] is True
    assert event["camera_pose_provenance_complete"] is True
    assert set(captured["row"]) == {"metric_dimensions_m"}
    assert "workbench" not in captured["prompt"].lower()

    verification = semantic.physical_recovery_event(
        "http://unused",
        "model",
        {"id": 1, "category": "workbench", "metric_dimensions_m": [1, 1, 2], "position_world_m": [0, 0, 0]},
        [
            (tmp_path / f"img_{value:06d}_det_0001.npz", b"jpeg")
            for value in (5, 8, 11)
        ],
        {
            "partition_id": "view-fold-1-of-3",
            "partition_index": 1,
            "partition_count": 3,
        },
        _pose_index((5, 8, 11)),
        source="physical_form_recovery_verification",
    )
    assert verification["source"] == "physical_form_recovery_verification"
    assert verification["event_id"].endswith(
        ":physical_form_recovery_verification"
    )
    assert verification["crop_image_ids"] == [5, 8, 11]
    assert verification["confirmation_eligible"] is False
    assert set(captured["row"]) == {"metric_dimensions_m"}
    assert "workbench" not in captured["prompt"].lower()


def test_recovery_preflight_rejects_reused_views_before_second_request(
    tmp_path: Path,
) -> None:
    semantic = _load(
        "farm_physical_recovery_preflight", "scripts/reconcile_farm_semantics.py"
    )
    primary = [
        (tmp_path / f"img_{value:06d}_det_0001.npz", b"jpeg")
        for value in (1, 2, 3)
    ]
    independent = [
        (tmp_path / f"img_{value:06d}_det_0001.npz", b"jpeg")
        for value in (4, 5, 6)
    ]
    reused = list(primary)
    first_partition = {
        "partition_id": "view-fold-0-of-2",
        "partition_index": 0,
        "partition_count": 2,
    }
    second_partition = {
        "partition_id": "view-fold-1-of-2",
        "partition_index": 1,
        "partition_count": 2,
    }

    poses = _pose_index((1, 2, 3, 4, 5, 6))
    accepted = semantic.independent_recovery_crop_contract(
        7, primary, first_partition, independent, second_partition,
        poses, [0.0, 0.0, 0.0],
    )
    rejected = semantic.independent_recovery_crop_contract(
        7, primary, first_partition, reused, first_partition,
        poses, [0.0, 0.0, 0.0],
    )

    assert accepted["eligible"] is True
    assert accepted["unique_view_count"] == 6
    assert accepted["view_overlap_coefficient"] == 0.0
    assert rejected["eligible"] is False
    assert "recovery_view_overlap_above_threshold" in rejected["reason_codes"]
    assert "recovery_same_evidence_fingerprint" in rejected["reason_codes"]
