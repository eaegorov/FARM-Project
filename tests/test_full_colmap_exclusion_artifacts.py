from __future__ import annotations

import hashlib
import json
from pathlib import Path

import pytest

from scripts.geometry.plan_farm_full_colmap_rescue import (
    choose_heldout_reference_views,
    finalize_planned_object_views,
    load_exclusion_frame_artifacts,
    main as plan_main,
)
from farm_runtime.full_colmap_view_planner import physical_timestamp_fold


IDENTITY_REGEX = (
    r"^(?P<camera>cam\d+)_(?P<timestamp>\d+)_"
    r"(?P<view>center|left)\.png$"
)


def _write_frames(
    path: Path,
    names: list[str],
    *,
    scale: float = 1.25,
    regex: str = IDENTITY_REGEX,
) -> Path:
    payload = {
        "meters_per_scene_unit": scale,
        "identity_contract": {"regex": regex},
        "frames": [{"source_image": name} for name in names],
    }
    path.write_text(json.dumps(payload, sort_keys=True), encoding="utf-8")
    return path


def test_exclusion_union_deduplicates_names_and_physical_timestamps(
    tmp_path: Path,
) -> None:
    primary = _write_frames(
        tmp_path / "primary.json",
        ["cam00_000001_center.png", "cam00_000002_center.png"],
    )
    additional = _write_frames(
        tmp_path / "additional.json",
        [
            "cam00_000001_center.png",
            "cam00_000001_left.png",
            "cam01_000003_center.png",
        ],
    )

    result = load_exclusion_frame_artifacts(primary, [additional])

    assert result["existing_names"] == {
        "cam00_000001_center.png",
        "cam00_000001_left.png",
        "cam00_000002_center.png",
        "cam01_000003_center.png",
    }
    assert result["existing_timestamps"] == {"000001", "000002", "000003"}
    provenance = result["provenance"]
    assert provenance["artifacts"][0]["unique_names"] == 2
    assert provenance["artifacts"][0]["unique_physical_timestamps"] == 2
    assert provenance["artifacts"][1]["unique_names"] == 3
    assert provenance["artifacts"][1]["unique_physical_timestamps"] == 2
    assert provenance["union"] == {
        "artifact_count": 2,
        "frame_rows": 5,
        "unique_names": 4,
        "unique_physical_timestamps": 3,
    }


@pytest.mark.parametrize(
    ("scale", "regex", "message"),
    (
        (2.0, IDENTITY_REGEX, "incompatible meters_per_scene_unit"),
        (
            1.25,
            r"^(?P<timestamp>\d+)_(?P<camera>cam\d+)\.png$",
            "incompatible identity_contract.regex",
        ),
    ),
)
def test_exclusion_artifacts_fail_closed_on_incompatible_contract(
    tmp_path: Path, scale: float, regex: str, message: str
) -> None:
    primary = _write_frames(
        tmp_path / "primary.json", ["cam00_000001_center.png"]
    )
    additional = _write_frames(
        tmp_path / "additional.json",
        ["cam00_000002_center.png"],
        scale=scale,
        regex=regex,
    )

    with pytest.raises(ValueError, match=message):
        load_exclusion_frame_artifacts(primary, [additional])


def test_exclusion_artifacts_fail_closed_on_missing_file_or_frames(
    tmp_path: Path,
) -> None:
    primary = _write_frames(
        tmp_path / "primary.json", ["cam00_000001_center.png"]
    )
    missing = tmp_path / "missing.json"
    with pytest.raises(FileNotFoundError):
        load_exclusion_frame_artifacts(primary, [missing])

    malformed = tmp_path / "malformed.json"
    malformed.write_text(
        json.dumps(
            {
                "meters_per_scene_unit": 1.25,
                "identity_contract": {"regex": IDENTITY_REGEX},
            }
        ),
        encoding="utf-8",
    )
    with pytest.raises(ValueError, match="no frames list"):
        load_exclusion_frame_artifacts(primary, [malformed])


def test_exclusion_provenance_hashes_and_paths_are_deterministic(
    tmp_path: Path,
) -> None:
    primary = _write_frames(
        tmp_path / "primary.json", ["cam00_000001_center.png"]
    )
    additional = _write_frames(
        tmp_path / "additional.json", ["cam00_000002_center.png"]
    )

    first = load_exclusion_frame_artifacts(primary, [additional])["provenance"]
    second = load_exclusion_frame_artifacts(primary, [additional])["provenance"]

    assert first == second
    for artifact, path in zip(
        first["artifacts"], (primary, additional), strict=True
    ):
        content = path.read_bytes()
        assert artifact["path"] == str(path.resolve())
        assert artifact["bytes"] == len(content)
        assert artifact["sha256"] == hashlib.sha256(content).hexdigest()


def _heldout_candidates(count: int) -> list[dict[str, object]]:
    seed = "heldout-reference-test"
    timestamps: list[str] = []
    candidate = 0
    while len(timestamps) < count:
        timestamp = f"{candidate:06d}"
        if physical_timestamp_fold(
            timestamp, heldout_fraction=0.25, split_seed=seed
        ) == "heldout":
            timestamps.append(timestamp)
        candidate += 1
    return [
        {
            "name": f"cam00_{timestamp}_center.png",
            "physical_timestamp": timestamp,
            "score": float(100 - index),
            "area_ratio": 0.1,
            "margin_ratio": 0.1,
            "view_direction": [float(index == 0), float(index == 1), 1.0],
        }
        for index, timestamp in enumerate(timestamps)
    ]


def test_heldout_reference_selection_and_object_rows_never_contain_train() -> None:
    candidates = _heldout_candidates(3)
    heldout = choose_heldout_reference_views(
        candidates,
        heldout_limit=2,
        min_view_angle_degrees=5.0,
        heldout_fraction=0.25,
        split_seed="heldout-reference-test",
    )
    row = finalize_planned_object_views(
        {"object_id": 2},
        [],
        heldout,
        heldout_reference_only=True,
        minimum_train_views=0,
        minimum_heldout_views=2,
    )

    assert len(heldout) == 2
    assert all(view["split"] == "heldout" for view in heldout)
    assert row["planning_status"] == "planned_heldout_reference_only"
    assert row["selected_count"] == 0
    assert row["selected_views"] == []
    assert row["train_views"] == []
    assert row["heldout_count"] == 2


def test_object_13_with_zero_heldout_references_fails_closed() -> None:
    row = finalize_planned_object_views(
        {"object_id": 13},
        [],
        [],
        heldout_reference_only=True,
        minimum_train_views=0,
        minimum_heldout_views=2,
    )

    assert row["planning_status"] == "insufficient_independent_views"
    assert row["planning_rejection_reasons"] == [
        "insufficient_heldout_physical_timestamps"
    ]
    assert row["train_views"] == []
    assert row["heldout_views"] == []


def test_heldout_reference_selection_does_not_fill_with_duplicate_direction() -> None:
    candidates = _heldout_candidates(3)
    for row in candidates:
        row["view_direction"] = [0.0, 0.0, 1.0]

    heldout = choose_heldout_reference_views(
        candidates,
        heldout_limit=2,
        min_view_angle_degrees=5.0,
        heldout_fraction=0.25,
        split_seed="heldout-reference-test",
    )
    result = finalize_planned_object_views(
        {"object_id": 13},
        [],
        heldout,
        heldout_reference_only=True,
        minimum_train_views=0,
        minimum_heldout_views=2,
    )

    assert len(heldout) == 1
    assert result["planning_status"] == "insufficient_independent_views"
    assert result["planning_rejection_reasons"] == [
        "insufficient_heldout_physical_timestamps"
    ]


def test_heldout_reference_finalization_rejects_any_train_leakage() -> None:
    with pytest.raises(ValueError, match="produced train views"):
        finalize_planned_object_views(
            {"object_id": 2},
            [{"name": "train.png", "physical_timestamp": "1"}],
            _heldout_candidates(2),
            heldout_reference_only=True,
            minimum_train_views=0,
            minimum_heldout_views=2,
        )


@pytest.mark.parametrize(
    "unsafe_args",
    (
        [],
        ["--object-id-file", "ids.txt", "--views-per-object", "1"],
        [
            "--object-id-file",
            "ids.txt",
            "--minimum-train-views-per-object",
            "1",
        ],
    ),
)
def test_heldout_reference_cli_rejects_adaptive_or_train_demand(
    monkeypatch, unsafe_args: list[str]
) -> None:
    argv = [
        "plan_farm_full_colmap_rescue.py",
        "--scene-state",
        "state.pt",
        "--colmap-model",
        "colmap",
        "--image-root",
        "images",
        "--frames-json",
        "frames.json",
        "--output",
        "plan.json",
        "--heldout-reference-only",
        *unsafe_args,
    ]
    monkeypatch.setattr("sys.argv", argv)
    with pytest.raises(SystemExit) as error:
        plan_main()
    assert error.value.code == 2
