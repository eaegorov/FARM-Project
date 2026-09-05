from __future__ import annotations

import copy
import json
import sys
from pathlib import Path

import numpy as np
import pytest

from farm_runtime.full_colmap_refinement_contract import (
    HELDOUT_REFINEMENT_ROLE,
    TRAIN_REFINEMENT_ROLE,
    prepare_plan_for_view_role,
    require_train_fit_refinement,
)
from scripts.geometry.apply_farm_full_colmap_rescue import main as apply_rescue
from scripts.geometry.merge_farm_full_colmap_rescue import main as merge_rescue
from scripts.geometry.refine_farm_full_colmap_masks import (
    add_projected_voxel_fallbacks,
    main as refine_masks,
)


def _views(prefix: str, split: str, timestamps: tuple[str, ...]) -> list[dict]:
    return [
        {
            "name": f"{prefix}_{timestamp}.png",
            "physical_timestamp": timestamp,
            "split": split,
        }
        for timestamp in timestamps
    ]


def _plan() -> dict:
    train = _views("train", "train", ("10", "20"))
    heldout = _views("heldout", "heldout", ("30", "40"))
    return {
        "schema": "farm.full-colmap-rescue-plan.v1",
        "train_names": [row["name"] for row in train],
        "heldout_names": [row["name"] for row in heldout],
        "objects": [
            {
                "object_id": 7,
                "selected_views": copy.deepcopy(train),
                "train_views": train,
                "heldout_views": heldout,
            }
        ],
    }


def _frames(split: str, *, state_fit_authorized: bool) -> dict:
    timestamps = ("10", "20") if split == "train" else ("30", "40")
    prefix = split
    return {
        "full_colmap_fold": {
            "schema": "farm.full-colmap-rgbd-fold.v1",
            "role": split,
            "state_fit_authorized": state_fit_authorized,
            "physical_timestamp_count": len(timestamps),
        },
        "selection_contract": {"split": split},
        "frames": [
            {
                "source_image": f"{prefix}_{timestamp}.png",
                "rgb_path": f"rgb/{prefix}_{timestamp}.jpg",
                "physical_timestamp": timestamp,
                "full_colmap_split": split,
            }
            for timestamp in timestamps
        ],
    }


def test_heldout_reference_selects_only_heldout_views() -> None:
    scoped, contract = prepare_plan_for_view_role(
        _plan(), _frames("heldout", state_fit_authorized=False), "heldout-reference"
    )
    assert contract["active_split"] == "heldout"
    assert contract["state_fit_authorized"] is False
    assert contract["plan_view_field"] == "heldout_views"
    assert [row["name"] for row in scoped["objects"][0]["selected_views"]] == [
        "heldout_30.png",
        "heldout_40.png",
    ]
    assert not any(
        row["name"].startswith("train_")
        for row in scoped["objects"][0]["selected_views"]
    )


def test_heldout_reference_only_fold_is_refinable_but_never_mutable() -> None:
    frames = _frames("heldout", state_fit_authorized=False)
    frames["full_colmap_fold"].update(
        {
            "role": "heldout_reference_only",
            "geometry_merge_authorized": False,
            "semantic_mutation_authorized": False,
        }
    )

    scoped, contract = prepare_plan_for_view_role(
        _plan(), frames, "heldout-reference"
    )

    assert contract["fold_role"] == "heldout_reference_only"
    assert contract["state_fit_authorized"] is False
    assert [
        row["name"] for row in scoped["objects"][0]["selected_views"]
    ] == ["heldout_30.png", "heldout_40.png"]


def test_heldout_reference_only_fold_requires_explicit_non_mutation_flags() -> None:
    frames = _frames("heldout", state_fit_authorized=False)
    frames["full_colmap_fold"]["role"] = "heldout_reference_only"

    with pytest.raises(ValueError, match="mutation authorization=false"):
        prepare_plan_for_view_role(_plan(), frames, "heldout-reference")


@pytest.mark.parametrize(
    ("requested_role", "frames_split", "fit"),
    [
        ("train", "heldout", False),
        ("heldout-reference", "train", True),
        ("heldout-reference", "heldout", True),
    ],
)
def test_rejects_train_heldout_role_or_authorization_mismatch(
    requested_role: str, frames_split: str, fit: bool
) -> None:
    with pytest.raises(
        ValueError, match="requested|state_fit_authorized|input contains"
    ):
        prepare_plan_for_view_role(
            _plan(), _frames(frames_split, state_fit_authorized=fit), requested_role
        )


def test_refine_cli_defaults_to_train_and_rejects_heldout_before_output(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    plan_path = tmp_path / "plan.json"
    frames_path = tmp_path / "frames.json"
    plan_path.write_text(json.dumps(_plan()), encoding="utf-8")
    frames_path.write_text(
        json.dumps(_frames("heldout", state_fit_authorized=False)), encoding="utf-8"
    )
    missing = tmp_path / "must-not-be-opened"
    output = tmp_path / "refinement-output"
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "refine_farm_full_colmap_masks.py",
            "--plan",
            str(plan_path),
            "--source-state",
            str(missing),
            "--rescue-state",
            str(missing),
            "--rescue-frames-json",
            str(frames_path),
            "--rescue-mask-root",
            str(missing),
            "--model",
            str(missing),
            "--output-dir",
            str(output),
        ],
    )
    with pytest.raises(ValueError, match="train input contains"):
        refine_masks()
    assert not output.exists()


def test_rejects_cross_fold_physical_timestamp_leakage() -> None:
    plan = _plan()
    plan["objects"][0]["heldout_views"][0]["physical_timestamp"] = "10"
    frames = _frames("heldout", state_fit_authorized=False)
    frames["frames"][0]["physical_timestamp"] = "10"
    with pytest.raises(ValueError, match="train/heldout leakage"):
        prepare_plan_for_view_role(plan, frames, "heldout-reference")


def test_rejects_duplicate_physical_timestamp_within_object() -> None:
    plan = _plan()
    plan["objects"][0]["heldout_views"][1]["physical_timestamp"] = "30"
    frames = _frames("heldout", state_fit_authorized=False)
    frames["frames"][1]["physical_timestamp"] = "30"
    with pytest.raises(ValueError, match="repeats source/physical timestamp"):
        prepare_plan_for_view_role(plan, frames, "heldout-reference")


def test_heldout_completion_never_consumes_selected_train_views(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    scoped, _ = prepare_plan_for_view_role(
        _plan(), _frames("heldout", state_fit_authorized=False), "heldout-reference"
    )
    source = {
        "object_id": np.asarray([7], dtype=np.int64),
        "object_box_centers_m": np.asarray([[0.0, 0.0, 2.0]], dtype=np.float32),
        "object_box_dimensions_m": np.ones((1, 3), dtype=np.float32),
        "object_box_wxyz": np.asarray([[1.0, 0.0, 0.0, 0.0]], dtype=np.float32),
    }
    rescue = {
        "images": [
            {"source_ref": "prepared/heldout_30.png"},
            {"source_ref": "prepared/heldout_40.png"},
            {"source_ref": "prepared/train_10.png"},
        ]
    }
    frames = {
        "heldout_30": {
            "rgb_path": "prepared/heldout_30.png",
            "source_image": "heldout_30.png",
        },
        "heldout_40": {
            "rgb_path": "prepared/heldout_40.png",
            "source_image": "heldout_40.png",
        },
        "train_10": {
            "rgb_path": "prepared/train_10.png",
            "source_image": "train_10.png",
        },
    }
    monkeypatch.setattr(
        "scripts.geometry.refine_farm_full_colmap_masks._object_voxel_points",
        lambda state, index: np.zeros((30, 3), dtype=np.float32),
    )
    rows = add_projected_voxel_fallbacks(
        scoped,
        source,
        rescue,
        frames,
        [],
        minimum_views=2,
        view_split="heldout",
    )
    assert [view["source_image"] for view in rows[0]["matched_views"]] == [
        "heldout_30.png",
        "heldout_40.png",
    ]


def _write_heldout_result(tmp_path: Path) -> tuple[Path, Path]:
    frames_path = tmp_path / "heldout-frames.json"
    frames_path.write_text(
        json.dumps(_frames("heldout", state_fit_authorized=False)), encoding="utf-8"
    )
    result_path = tmp_path / "heldout-result.json"
    result_path.write_text(
        json.dumps(
            {
                "schema": "farm.full-colmap-mask-refinement.v1",
                "status": "PASS",
                "refinement_role": HELDOUT_REFINEMENT_ROLE,
                "rescue_frames_json": str(frames_path),
                "input_view_contract": {
                    "active_split": "heldout",
                    "state_fit_authorized": False,
                },
                "policy": {"state_fit_authorized": False},
                "objects": [],
            }
        ),
        encoding="utf-8",
    )
    return result_path, frames_path


def test_shared_guard_rejects_heldout_reference_and_legacy_heldout(
    tmp_path: Path,
) -> None:
    result_path, frames_path = _write_heldout_result(tmp_path)
    payload = json.loads(result_path.read_text(encoding="utf-8"))
    with pytest.raises(ValueError, match="evaluation-only"):
        require_train_fit_refinement(payload, refinement_path=result_path)
    payload.pop("refinement_role")
    with pytest.raises(ValueError, match="requested train|input contains"):
        require_train_fit_refinement(payload, refinement_path=result_path)


def test_merge_and_apply_reject_heldout_before_writing(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    result_path, frames_path = _write_heldout_result(tmp_path)
    missing = tmp_path / "must-not-be-opened"
    merge_output = tmp_path / "merge-output"
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "merge_farm_full_colmap_rescue.py",
            "--source-state",
            str(missing),
            "--source-frames",
            str(missing),
            "--source-mask-root",
            str(missing),
            "--rescue-state",
            str(missing),
            "--rescue-frames",
            str(frames_path),
            "--refinement",
            str(result_path),
            "--prior-catalog",
            str(missing),
            "--output-dir",
            str(merge_output),
        ],
    )
    with pytest.raises(ValueError, match="evaluation-only"):
        merge_rescue()
    assert not merge_output.exists()

    apply_outputs = [
        tmp_path / name for name in ("state.pt", "catalog.json", "report.json")
    ]
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "apply_farm_full_colmap_rescue.py",
            "--source-state",
            str(missing),
            "--refined-state",
            str(missing),
            "--refinement",
            str(result_path),
            "--geometry-report",
            str(missing),
            "--review-report",
            str(missing),
            "--prior-catalog",
            str(missing),
            "--output-state",
            str(apply_outputs[0]),
            "--output-catalog",
            str(apply_outputs[1]),
            "--output-report",
            str(apply_outputs[2]),
        ],
    )
    with pytest.raises(ValueError, match="evaluation-only"):
        apply_rescue()
    assert not any(path.exists() for path in apply_outputs)


def test_explicit_train_fit_contract_is_accepted(tmp_path: Path) -> None:
    result_path = tmp_path / "train-result.json"
    payload = {
        "schema": "farm.full-colmap-mask-refinement.v1",
        "status": "PASS",
        "refinement_role": TRAIN_REFINEMENT_ROLE,
        "policy": {"state_fit_authorized": True},
        "input_view_contract": {
            "active_split": "train",
            "state_fit_authorized": True,
        },
    }
    result_path.write_text(json.dumps(payload), encoding="utf-8")
    require_train_fit_refinement(payload, refinement_path=result_path)
