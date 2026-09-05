from __future__ import annotations

import copy
import hashlib
import json
from argparse import Namespace
from dataclasses import replace
from pathlib import Path

import numpy as np
import pytest

from farm_runtime.obb_release_evidence import (
    evaluate_obb_release_evidence,
)
from tools.farm_shaper_bridge import run_gaussian_lift as lift_runner
from tools.farm_shaper_bridge.common import FarmObject, RunData
from tools.farm_shaper_bridge.gaussian_lift import (
    assert_gaussian_owner_scope,
    execute,
    release_scope_can_publish,
    restrict_release_scope,
    validate_release_object_scope,
)


def _sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _object(object_id: int) -> FarmObject:
    return FarmObject(
        object_id=object_id,
        category="object",
        description="",
        center_m=np.zeros(3),
        dimensions_m=np.ones(3),
        rotation=np.eye(3),
        observations=(),
    )


def _run(root: Path, object_ids: tuple[int, ...] = (7, 8, 9, 42)) -> RunData:
    return RunData(
        run_dir=root,
        scene_id="scene",
        frames_doc={},
        frames=(),
        objects=tuple(_object(value) for value in object_ids),
        meters_per_scene_unit=1.0,
        rgbd_config={},
        success_sha256="a" * 64,
        acceptance_sha256="b" * 64,
        resource_preflight={},
        scene_preflight={},
        legacy=False,
    )


def _write_json(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, sort_keys=True) + "\n", encoding="utf-8")


def _artifact(role: str, index: int, path: Path) -> dict[str, object]:
    return {
        "role": role,
        "input_index": index,
        "path": str(path.resolve()),
        "bytes": path.stat().st_size,
        "sha256": _sha(path),
    }


def _release_evidence() -> dict[str, object]:
    return evaluate_obb_release_evidence(
        geometry_fit_passed=True,
        independent_physical_timestamps=5,
        timestamp_median_projected_box_iou=0.75,
        timestamp_q25_projected_box_iou=0.65,
        orientation_materiality=0.50,
        orientation_confidence=0.80,
    )


def _fixture(tmp_path: Path) -> dict[str, object]:
    sources = tmp_path / "sources"
    train = sources / "train.json"
    geometry = sources / "geometry.json"
    heldout = sources / "heldout.json"
    _write_json(
        train,
        {
            "schema": "farm.full-colmap-mask-refinement.v1",
            "status": "PASS",
            "scene_id": "scene",
            "refinement_role": "train_fit",
            "objects": [
                {"object_id": 7},
                {"object_id": 8},
                {"object_id": 9},
                {"object_id": 130},
            ],
        },
    )
    _write_json(
        geometry,
        {
            "schema": "farm.object-geometry-audit.v3",
            "scene_id": "scene",
            "objects": [
                {
                    "object_id": value,
                    "status": "geometry_pass",
                    "train_geometry_evidence": _release_evidence(),
                }
                for value in (7, 8, 9, 130)
            ],
        },
    )
    _write_json(
        heldout,
        {
            "schema": "farm.full-colmap-mask-refinement.v1",
            "status": "PASS",
            "scene_id": "scene",
            "refinement_role": "heldout_reference_only",
            "objects": [
                {"object_id": 7},
                {"object_id": 8},
                {"object_id": 9},
                {"object_id": 130},
            ],
        },
    )
    inputs = [
        _artifact("train_refinement_result", 0, train),
        _artifact("geometry_audit", 0, geometry),
        _artifact("heldout_reference_result", 0, heldout),
    ]
    consumed = sorted(copy.deepcopy(inputs), key=lambda row: (row["role"], row["path"]))

    selection_dir = tmp_path / "selection"
    selection_dir.mkdir()
    ids = selection_dir / "pre_lift_eligible_object_ids.txt"
    ids.write_text("7\n", encoding="utf-8")
    selection = {
        "schema": "farm.pre-lift-eligibility-selection.v1",
        "status": "PASS",
        "selection_kind": "pre_lift_eligibility_only",
        "gaussian_mask_improvement_proven": False,
        "eligible_object_ids": [7],
        "rejected_object_ids": [8, 9, 130],
        "ignored_heldout_only_object_ids": [],
        "objects": [
            {
                "object_id": value,
                "eligible": value == 7,
                "reasons": [] if value == 7 else ["test_rejected"],
                "geometry_status": "geometry_pass",
                "geometry_release_evidence": _release_evidence(),
            }
            for value in (7, 8, 9, 130)
        ],
        "integrity": {
            "physical_timestamp_sets_disjoint": True,
            "train_heldout_physical_timestamp_overlap": [],
        },
        "policy": {
            "geometry_release_evidence_required": True,
            "legacy_geometry_pass_without_release_evidence_is_not_publishable": True,
            "heldout_state_fit_authorized": False,
            "train_heldout_physical_timestamp_overlap_forbidden": True,
            "post_lift_gaussian_mask_qc_required": True,
            "pre_lift_eligibility_is_not_gaussian_mask_improvement_proof": True,
        },
        "provenance": {
            "inputs": inputs,
            "consumed_artifacts": consumed,
        },
        "outputs": {
            "exact_object_ids": {
                "relative_path": ids.name,
                "format": "one_decimal_object_id_per_line",
                "bytes": ids.stat().st_size,
                "sha256": _sha(ids),
            }
        },
    }
    report = selection_dir / "selection.json"
    _write_json(report, selection)
    mounted = [Path(row["path"]) for row in consumed]
    return {
        "run": _run(tmp_path / "run"),
        "selection": report,
        "ids": ids,
        "sources": {"train": train, "geometry": geometry, "heldout": heldout},
        "mounted": mounted,
    }


def _validate(fixture: dict[str, object]) -> dict[str, object]:
    return validate_release_object_scope(
        fixture["run"],
        fixture["selection"],
        fixture["ids"],
        fixture["mounted"],
    )


def _edit_selection(fixture: dict[str, object], edit) -> None:
    path = fixture["selection"]
    payload = json.loads(path.read_text(encoding="utf-8"))
    edit(payload)
    _write_json(path, payload)


def test_valid_release_allowlist_binds_artifacts_universe_and_owner_scope(
    tmp_path: Path,
) -> None:
    fixture = _fixture(tmp_path)
    scope = _validate(fixture)
    assert scope["status"] == "PASS"
    assert scope["active_object_ids"] == [7]
    assert scope["selection_object_ids"] == [7, 8, 9, 130]
    assert scope["source_active_object_ids"] == [7, 8, 9, 42]
    assert scope["excluded_active_object_ids"] == [8, 9, 42]
    assert scope["inactive_rejected_object_ids"] == [130]
    assert scope["active_ids_not_in_selection"] == [42]
    assert scope["forced_exclusion_audit"] == [{
        "object_id": 42,
        "eligible": False,
        "reason": "missing_from_full_train_refinement_universe",
    }]
    assert scope["artifacts"]["object_id_file"]["sha256"] == _sha(fixture["ids"])
    assert scope["artifacts"]["pre_lift_selection"]["sha256"] == _sha(
        fixture["selection"]
    )

    split = {
        "objects": [
            {"object_id": 7},
            {"object_id": 8},
            {"object_id": 9},
            {"object_id": 42},
        ],
        "build_timestamps": ["1"],
        "heldout_timestamps": ["2"],
    }
    scoped_run, scoped_split = restrict_release_scope(
        fixture["run"], split, scope
    )
    assert [obj.object_id for obj in scoped_run.objects] == [7]
    assert [row["object_id"] for row in scoped_split["objects"]] == [7]
    assert_gaussian_owner_scope(
        np.asarray([7, -1, 7], dtype=np.int32), [7], label="test"
    )
    with pytest.raises(AssertionError, match="excluded Gaussian owners"):
        assert_gaussian_owner_scope(
            np.asarray([7, 8, -1], dtype=np.int32), [7], label="test"
        )


def test_legacy_selection_filters_for_evaluation_but_cannot_release(
    tmp_path: Path,
) -> None:
    fixture = _fixture(tmp_path)
    legacy_run = replace(fixture["run"], legacy=True)
    fixture["run"] = legacy_run
    scope = _validate(fixture)
    split = {
        "objects": [{"object_id": 7}, {"object_id": 8}, {"object_id": 9}],
        "build_timestamps": ["1"],
        "heldout_timestamps": ["2"],
    }
    scoped_run, scoped_split = restrict_release_scope(legacy_run, split, scope)
    assert [obj.object_id for obj in scoped_run.objects] == [7]
    assert [row["object_id"] for row in scoped_split["objects"]] == [7]
    assert (
        release_scope_can_publish(
            scoped_run, {"status": "PASS"}, scope, smoke=False
        )
        is False
    )


def test_public_runner_preflights_and_plans_exact_read_only_mounts(
    tmp_path: Path,
) -> None:
    fixture = _fixture(tmp_path)
    prepared = lift_runner.prepare_release_selection_inputs(
        fixture["ids"], fixture["selection"]
    )
    assert prepared is not None
    assert prepared["container_object_ids"].endswith(
        "/pre_lift_eligible_object_ids.txt"
    )
    assert prepared["container_selection"].endswith("/selection.json")
    assert len(prepared["provenance_mounts"]) == 3
    assert all(
        target == f"/farm/pre_lift_provenance/{index:06d}"
        for index, (_path, target) in enumerate(prepared["provenance_mounts"])
    )
    parsed = lift_runner.parse_args([
        "--run", str(tmp_path / "run"),
        "--ply", str(tmp_path / "scene.ply"),
        "--output", str(tmp_path / "out"),
        "--object-id-file", str(fixture["ids"]),
        "--pre-lift-selection", str(fixture["selection"]),
    ])
    assert parsed.object_id_file == fixture["ids"]
    assert parsed.pre_lift_selection == fixture["selection"]


def test_runner_command_forwards_release_allowlist_into_signed_core(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    fixture = _fixture(tmp_path)
    run = tmp_path / "canonical-run"
    (run / "input").mkdir(parents=True)
    image_id = "sha256:" + "a" * 64
    _write_json(
        run / "input" / "resource_preflight.json",
        {
            "checks": [{
                "name": "runtime:prep",
                "status": "pass",
                "metrics": {
                    "image": "rest3d:pinned",
                    "expected_image_id": image_id,
                    "actual_image_id": image_id,
                    "python": "/opt/conda/envs/rest3d/bin/python",
                },
            }],
        },
    )
    _write_json(
        run / "manifest.json",
        {
            "schema": "farm.pipeline-run.v1",
            "status": "success",
            "git": {"commit": "c" * 40, "dirty": False},
            "source_snapshot": {"tree_sha256": "b" * 64},
        },
    )
    snapshot = tmp_path / "snapshot"
    for relative in lift_runner.REQUIRED_LIFT_SOURCE:
        path = snapshot / relative
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text("# pinned\n", encoding="utf-8")
    config = tmp_path / "lift.yaml"
    config.write_bytes((snapshot / "configs/gaussian_lift.v1.yaml").read_bytes())
    ply = tmp_path / "scene.ply"
    ply.write_bytes(b"ply")

    import farm_runtime.source_snapshot as snapshot_module

    monkeypatch.setattr(
        snapshot_module,
        "validated_source_snapshot_project_root",
        lambda *_args, **_kwargs: snapshot,
    )
    monkeypatch.setattr(lift_runner, "_docker_image_id", lambda _tag: image_id)
    monkeypatch.setattr(lift_runner, "_git_snapshot", lambda _repo: ("d" * 40, False))
    command, metadata = lift_runner.build_command(Namespace(
        run=run,
        ply=ply,
        output=tmp_path / "lift-output",
        config=config,
        gpu="0",
        plan_only=True,
        smoke_object_id=[],
        allow_legacy_run=False,
        object_id_file=fixture["ids"],
        pre_lift_selection=fixture["selection"],
        full_colmap_fold_manifest=None,
        allow_rebuilt_image_nonrelease=False,
        print_command=False,
    ))
    assert command[command.index("--object-id-file") + 1] == (
        "/farm/pre_lift_selection/pre_lift_eligible_object_ids.txt"
    )
    assert command[command.index("--pre-lift-selection") + 1] == (
        "/farm/pre_lift_selection/selection.json"
    )
    assert command.count("--pre-lift-provenance-artifact") == 3
    mounts = [
        command[index + 1]
        for index, value in enumerate(command[:-1])
        if value == "--mount"
    ]
    assert (
        f"type=bind,src={fixture['selection'].parent},"
        "dst=/farm/pre_lift_selection,readonly"
    ) in mounts
    assert all(
        any(f"dst=/farm/pre_lift_provenance/{index:06d},readonly" in mount for mount in mounts)
        for index in range(3)
    )
    assert metadata["release_object_allowlist"]["artifacts"][
        "object_id_file"
    ]["sha256"] == _sha(fixture["ids"])


@pytest.mark.parametrize(
    ("field", "value", "message"),
    [
        ("schema", "other", "schema"),
        ("status", "WARN", "status must be PASS"),
        ("eligible_object_ids", [7, 7], "sorted and unique"),
        ("eligible_object_ids", [8, 7], "sorted and unique"),
    ],
)
def test_release_allowlist_rejects_bad_selection_authority(
    tmp_path: Path, field: str, value: object, message: str
) -> None:
    fixture = _fixture(tmp_path)
    _edit_selection(fixture, lambda payload: payload.__setitem__(field, value))
    with pytest.raises(ValueError, match=message):
        _validate(fixture)


def test_release_allowlist_rejects_id_artifact_drift(tmp_path: Path) -> None:
    fixture = _fixture(tmp_path)
    fixture["ids"].write_text("7\n8\n", encoding="utf-8")
    with pytest.raises(ValueError, match="exactly equal eligible_object_ids"):
        _validate(fixture)


def test_release_allowlist_rejects_source_hash_drift(tmp_path: Path) -> None:
    fixture = _fixture(tmp_path)
    train = fixture["sources"]["train"]
    train.write_text(train.read_text(encoding="utf-8") + " ", encoding="utf-8")
    with pytest.raises(ValueError, match="source input size mismatch"):
        _validate(fixture)
    with pytest.raises(ValueError, match="size mismatch"):
        lift_runner.prepare_release_selection_inputs(
            fixture["ids"], fixture["selection"]
        )


def test_release_allowlist_rejects_same_size_source_sha_drift(
    tmp_path: Path,
) -> None:
    fixture = _fixture(tmp_path)
    train = fixture["sources"]["train"]
    content = train.read_bytes()
    train.write_bytes(bytes([content[0] ^ 1]) + content[1:])
    with pytest.raises(ValueError, match="source input SHA256 mismatch"):
        _validate(fixture)
    with pytest.raises(ValueError, match="SHA256 mismatch"):
        lift_runner.prepare_release_selection_inputs(
            fixture["ids"], fixture["selection"]
        )


def test_release_allowlist_rejects_inactive_or_absent_eligible_objects(
    tmp_path: Path,
) -> None:
    fixture = _fixture(tmp_path)
    fixture["run"] = _run(tmp_path / "run", (8, 9, 42))
    with pytest.raises(ValueError, match="eligible object IDs are inactive/absent"):
        _validate(fixture)


def test_release_allowlist_rejects_train_universe_or_scene_mismatch(
    tmp_path: Path,
) -> None:
    fixture = _fixture(tmp_path)
    train = fixture["sources"]["train"]
    payload = json.loads(train.read_text(encoding="utf-8"))
    payload["objects"] = [{"object_id": 7}]
    _write_json(train, payload)
    selection = json.loads(fixture["selection"].read_text(encoding="utf-8"))
    for spec in selection["provenance"]["inputs"]:
        if spec["role"] == "train_refinement_result":
            spec["bytes"] = train.stat().st_size
            spec["sha256"] = _sha(train)
    for spec in selection["provenance"]["consumed_artifacts"]:
        if spec["role"] == "train_refinement_result":
            spec["bytes"] = train.stat().st_size
            spec["sha256"] = _sha(train)
    _write_json(fixture["selection"], selection)
    with pytest.raises(ValueError, match="object universe"):
        _validate(fixture)

    fixture = _fixture(tmp_path / "scene-mismatch")
    geometry = fixture["sources"]["geometry"]
    payload = json.loads(geometry.read_text(encoding="utf-8"))
    payload["scene_id"] = "other"
    _write_json(geometry, payload)
    selection = json.loads(fixture["selection"].read_text(encoding="utf-8"))
    for group in ("inputs", "consumed_artifacts"):
        for spec in selection["provenance"][group]:
            if spec["role"] == "geometry_audit":
                spec["bytes"] = geometry.stat().st_size
                spec["sha256"] = _sha(geometry)
    _write_json(fixture["selection"], selection)
    with pytest.raises(ValueError, match="scene_id mismatch"):
        _validate(fixture)


def test_release_allowlist_binds_eligible_row_to_geometry_evidence(
    tmp_path: Path,
) -> None:
    fixture = _fixture(tmp_path)
    geometry = fixture["sources"]["geometry"]
    payload = json.loads(geometry.read_text(encoding="utf-8"))
    payload["objects"][0]["train_geometry_evidence"][
        "timestamp_median_projected_box_iou"
    ] = 0.74
    _write_json(geometry, payload)

    selection = json.loads(fixture["selection"].read_text(encoding="utf-8"))
    for group in ("inputs", "consumed_artifacts"):
        for spec in selection["provenance"][group]:
            if spec["role"] == "geometry_audit":
                spec["bytes"] = geometry.stat().st_size
                spec["sha256"] = _sha(geometry)
    _write_json(fixture["selection"], selection)

    with pytest.raises(ValueError, match="not bound to passed geometry audit evidence"):
        _validate(fixture)


def test_release_allowlist_and_smoke_are_mutually_exclusive_before_gpu(
    tmp_path: Path,
) -> None:
    fixture = _fixture(tmp_path)
    args = Namespace(
        config=tmp_path / "missing.yaml",
        run=tmp_path / "missing-run",
        allow_legacy_run=False,
        smoke_object_id=[7],
        object_id_file=fixture["ids"],
        pre_lift_selection=fixture["selection"],
        pre_lift_provenance_artifact=fixture["mounted"],
    )
    with pytest.raises(ValueError, match="incompatible"):
        execute(args)
