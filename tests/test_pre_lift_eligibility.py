from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest

from farm_runtime.obb_release_evidence import (
    evaluate_obb_release_evidence,
)
from farm_runtime.pre_lift_eligibility import (
    IDS_FILENAME,
    REPORT_FILENAME,
    SELECTION_SCHEMA,
    build_pre_lift_selection,
    publish_pre_lift_selection,
    sha256_file,
)
from scripts.evaluation.select_farm_pre_lift_eligible_objects import parse_args


def _json(path: Path, payload: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload), encoding="utf-8")


def _view(root: Path, source: str, timestamp: int, suffix: str) -> dict:
    relative = Path("masks") / f"{suffix}.npz"
    path = root / relative
    path.parent.mkdir(parents=True, exist_ok=True)
    crop = np.ones((4, 5), dtype=bool)
    np.savez_compressed(
        path,
        image_shape=np.asarray([9, 13], dtype=np.int32),
        raw_bits=np.packbits(crop.reshape(-1), bitorder="little"),
        raw_shape=np.asarray(crop.shape, dtype=np.int32),
        raw_bbox_xyxy=np.asarray([3, 2, 8, 6], dtype=np.int32),
    )
    return {
        "source_image": source,
        "physical_timestamp_ns": timestamp,
        "accepted": True,
        "quality_gates": {"depth": True, "bounded_seed_area_expansion": True},
        "rejection_reasons": [],
        "mask_relative": relative.as_posix(),
        "mask_sha256": sha256_file(path),
        "stored_mask_pixels": int(crop.sum()),
    }


def _object(object_id: int, accepted: bool, views: list[dict]) -> dict:
    return {
        "object_id": object_id,
        "accepted": accepted,
        "accepted_views": len(views),
        "views": views,
    }


def _geometry(object_id: int, status: str) -> dict:
    passed = status == "geometry_pass"
    evidence = evaluate_obb_release_evidence(
        geometry_fit_passed=passed,
        independent_physical_timestamps=4 if passed else 2,
        timestamp_median_projected_box_iou=0.75 if passed else 0.25,
        timestamp_q25_projected_box_iou=0.65 if passed else 0.20,
        orientation_materiality=0.50,
        orientation_confidence=0.80,
    )
    return {
        "object_id": object_id,
        "status": status,
        "geometry_fit_passed": passed,
        "geometry_release_ready": passed,
        "train_geometry_evidence": evidence,
    }


def _refinement(
    root: Path, role: str, objects: list[dict], *, state_fit_authorized: bool
) -> Path:
    heldout = role == "heldout_reference_only"
    payload = {
        "schema": "farm.full-colmap-mask-refinement.v1",
        "status": "PASS",
        "refinement_role": role,
        "input_view_contract": {
            "fold_contract": "farm.full-colmap-rgbd-fold.v1",
            "active_split": "heldout" if heldout else "train",
            "state_fit_authorized": state_fit_authorized,
            "plan_view_field": "heldout_views" if heldout else "train_views",
            "requested_view_role": "heldout-reference" if heldout else "train",
        },
        "policy": {
            "view_role": "heldout-reference" if heldout else "train",
            "state_fit_authorized": state_fit_authorized,
            "merge_apply_forbidden": heldout,
            "heldout_reference_is_evaluation_only": heldout,
            "reference_masks_are_pseudo_labels_not_ground_truth": heldout,
        },
        "accepted_objects": sum(row["accepted"] is True for row in objects),
        "accepted_object_ids": sorted(
            row["object_id"] for row in objects if row["accepted"] is True
        ),
        "objects": objects,
    }
    path = root / "result.json"
    _json(path, payload)
    return path


def _fixture(tmp_path: Path) -> dict[str, object]:
    train_root = tmp_path / "train"
    train = _refinement(
        train_root,
        "train_fit",
        [
            _object(
                7,
                True,
                [
                    _view(train_root, "train_100.png", 100, "7_100"),
                    _view(train_root, "train_110.png", 110, "7_110"),
                ],
            ),
            _object(8, False, []),
            _object(9, True, [_view(train_root, "train_120.png", 120, "9_120")]),
            _object(130, True, [_view(train_root, "train_130.png", 130, "130_130")]),
        ],
        state_fit_authorized=True,
    )
    geometry = tmp_path / "geometry.json"
    _json(
        geometry,
        {
            "schema": "farm.object-geometry-audit.v3",
            "objects": [
                _geometry(7, "geometry_pass"),
                _geometry(8, "geometry_pass"),
                _geometry(9, "geometry_rejected"),
                _geometry(130, "geometry_pass"),
            ],
        },
    )
    first_root = tmp_path / "heldout-a"
    first = _refinement(
        first_root,
        "heldout_reference_only",
        [
            _object(7, False, [_view(first_root, "held_200.png", 200, "7_200")]),
            _object(
                8,
                True,
                [
                    _view(first_root, "held_210.png", 210, "8_210"),
                    _view(first_root, "held_220.png", 220, "8_220"),
                ],
            ),
            _object(
                9,
                True,
                [
                    _view(first_root, "held_230.png", 230, "9_230"),
                    _view(first_root, "held_240.png", 240, "9_240"),
                ],
            ),
            _object(
                130,
                False,
                [_view(first_root, "held_250.png", 250, "130_250")],
            ),
        ],
        state_fit_authorized=False,
    )
    second_root = tmp_path / "heldout-b"
    second = _refinement(
        second_root,
        "heldout_reference_only",
        [
            _object(
                7,
                True,
                [
                    _view(second_root, "held_200.png", 200, "7_200_duplicate"),
                    _view(second_root, "held_300.png", 300, "7_300"),
                ],
            ),
            _object(
                130,
                False,
                [_view(second_root, "held_250.png", 250, "130_250_duplicate")],
            ),
        ],
        state_fit_authorized=False,
    )
    return {"train": train, "geometry": geometry, "heldout": [first, second]}


def _build(fixture: dict[str, object], *, minimum: int = 2) -> dict:
    return build_pre_lift_selection(
        fixture["train"],
        fixture["geometry"],
        fixture["heldout"],
        minimum_independent_physical_timestamps=minimum,
    )


def test_unions_topups_dedupes_exact_views_and_applies_all_three_gates(
    tmp_path: Path,
) -> None:
    selection = _build(_fixture(tmp_path))
    assert selection["schema"] == SELECTION_SCHEMA
    assert selection["eligible_object_ids"] == [7]
    assert selection["gaussian_mask_improvement_proven"] is False
    rows = {row["object_id"]: row for row in selection["objects"]}
    assert rows[7]["heldout_observations_after_exact_dedupe"] == 2
    assert rows[7]["heldout_physical_timestamps"] == ["200", "300"]
    assert "train_refinement_not_accepted" in rows[8]["reasons"]
    assert "geometry_status_not_geometry_pass" in rows[9]["reasons"]
    assert rows[130]["independent_heldout_physical_timestamps"] == 1
    assert (
        "insufficient_independent_heldout_physical_timestamps" in rows[130]["reasons"]
    )
    assert selection["policy"]["post_lift_gaussian_mask_qc_required"] is True
    assert len(selection["provenance"]["inputs"]) == 4


def test_publishes_exact_ids_and_sha_provenance_atomically(tmp_path: Path) -> None:
    selection = _build(_fixture(tmp_path))
    output = tmp_path / "published"
    summary = publish_pre_lift_selection(output, selection)
    ids_path = output / IDS_FILENAME
    report_path = output / REPORT_FILENAME
    assert ids_path.read_text(encoding="utf-8") == "7\n"
    report = json.loads(report_path.read_text(encoding="utf-8"))
    assert report["outputs"]["exact_object_ids"]["sha256"] == sha256_file(ids_path)
    assert summary["selection_json_sha256"] == sha256_file(report_path)
    assert summary["exact_ids_txt_sha256"] == sha256_file(ids_path)
    with pytest.raises(FileExistsError, match="refusing to overwrite"):
        publish_pre_lift_selection(output, selection)


def test_configurable_minimum_remains_pre_lift_only(tmp_path: Path) -> None:
    selection = _build(_fixture(tmp_path), minimum=3)
    assert selection["eligible_object_ids"] == []
    assert selection["selection_kind"] == "pre_lift_eligibility_only"
    assert selection["policy"][
        "pre_lift_eligibility_is_not_gaussian_mask_improvement_proof"
    ]


def test_legacy_geometry_pass_without_release_evidence_is_not_publishable(
    tmp_path: Path,
) -> None:
    fixture = _fixture(tmp_path)
    geometry_path = fixture["geometry"]
    payload = json.loads(geometry_path.read_text(encoding="utf-8"))
    del payload["objects"][0]["train_geometry_evidence"]
    _json(geometry_path, payload)

    selection = _build(fixture)
    row = next(row for row in selection["objects"] if row["object_id"] == 7)
    assert row["eligible"] is False
    assert "geometry_release_evidence_missing_or_invalid" in row["reasons"]


def test_rejects_heldout_fit_authorization_before_selection(tmp_path: Path) -> None:
    fixture = _fixture(tmp_path)
    path = fixture["heldout"][0]
    payload = json.loads(path.read_text(encoding="utf-8"))
    payload["input_view_contract"]["state_fit_authorized"] = True
    _json(path, payload)
    with pytest.raises(ValueError, match="input_view_contract/policy"):
        _build(fixture)


def test_rejects_any_train_heldout_physical_timestamp_overlap(tmp_path: Path) -> None:
    fixture = _fixture(tmp_path)
    path = fixture["heldout"][0]
    payload = json.loads(path.read_text(encoding="utf-8"))
    payload["objects"][0]["views"][0]["source_image"] = "held_overlap_100.png"
    payload["objects"][0]["views"][0]["physical_timestamp_ns"] = 100
    _json(path, payload)
    with pytest.raises(ValueError, match="timestamp overlap"):
        _build(fixture)


def test_rejects_conflicting_timestamp_for_same_source_across_topups(
    tmp_path: Path,
) -> None:
    fixture = _fixture(tmp_path)
    path = fixture["heldout"][1]
    payload = json.loads(path.read_text(encoding="utf-8"))
    payload["objects"][0]["views"][1]["source_image"] = "held_200.png"
    _json(path, payload)
    with pytest.raises(ValueError, match="conflicting timestamps"):
        _build(fixture)


def test_rejects_invalid_accepted_mask_or_hard_gates(tmp_path: Path) -> None:
    fixture = _fixture(tmp_path)
    path = fixture["heldout"][0]
    payload = json.loads(path.read_text(encoding="utf-8"))
    payload["objects"][0]["views"][0]["quality_gates"]["depth"] = False
    _json(path, payload)
    with pytest.raises(ValueError, match="hard quality gates"):
        _build(fixture)


def test_publish_revalidates_inputs_and_cleans_partial_output(tmp_path: Path) -> None:
    fixture = _fixture(tmp_path)
    selection = _build(fixture)
    train_path = fixture["train"]
    train_path.write_text(
        train_path.read_text(encoding="utf-8") + " ", encoding="utf-8"
    )
    output = tmp_path / "must-not-exist"
    with pytest.raises(RuntimeError, match="input (size|hash) changed"):
        publish_pre_lift_selection(output, selection)
    assert not output.exists()
    assert not list(tmp_path.glob(".must-not-exist.*.tmp"))


def test_cli_default_requires_two_independent_timestamps(tmp_path: Path) -> None:
    args = parse_args(
        [
            "--train-refinement-result",
            str(tmp_path / "train.json"),
            "--geometry-audit",
            str(tmp_path / "geometry.json"),
            "--heldout-reference-result",
            str(tmp_path / "heldout.json"),
            "--output-dir",
            str(tmp_path / "output"),
        ]
    )
    assert args.minimum_independent_physical_timestamps == 2
