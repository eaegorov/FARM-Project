from __future__ import annotations

import hashlib
import json
from argparse import Namespace
from pathlib import Path

import numpy as np
import pytest
import yaml

from tools.farm_shaper_bridge.common import (
    UNKNOWN_ID,
    FarmObject,
    RunData,
    binary_mask_metrics,
    build_verified_csr,
    canonical_presentation_ids,
    load_mask_pair,
    open_graphdeco_ply,
    resolve_sparse_claims,
    sh_dc_to_rgb,
    solve_global_timestamp_split,
    validate_acceptance_contract,
    validate_physical_timestamp_bijection,
    validate_run_artifact_bundle,
    write_full_labeled_ply,
)
from farm_runtime.integrity import directory_tree_descriptor
from tools.farm_shaper_bridge.gaussian_lift import (
    finalize_dense,
    load_config,
    restrict_smoke_scope,
    summarize_heldout_by_timestamp,
)
from tools.farm_shaper_bridge import run_gaussian_lift as lift_runner
from tools.farm_shaper_bridge.run_gaussian_lift import extract_prep_runtime


def test_checked_in_lift_config_is_frozen_after_two_scene_calibration() -> None:
    config_path = Path(__file__).resolve().parents[1] / "configs/gaussian_lift.v1.yaml"
    config, digest = load_config(config_path)
    assert len(digest) == 64
    assert config["refinement"]["enabled"] is True
    assert config["release"] == {
        "calibrated": True,
        "minimum_verified_objects": 1,
        "minimum_verified_strict_fraction": 0.50,
    }


@pytest.mark.parametrize(
    ("section", "key", "value"),
    [("release", "minimum_verified_strict_fraction", float("nan")),
     ("heldout", "minimum_median_precision", float("inf")),
     ("build", "maximum_object_fraction", 1.1)],
)
def test_lift_config_rejects_nonfinite_or_out_of_range_thresholds(
    tmp_path: Path, section: str, key: str, value: float
) -> None:
    source = Path(__file__).resolve().parents[1] / "configs/gaussian_lift.v1.yaml"
    payload = yaml.safe_load(source.read_text(encoding="utf-8"))
    payload[section][key] = value
    path = tmp_path / "bad.yaml"
    path.write_text(yaml.safe_dump(payload), encoding="utf-8")
    with pytest.raises(ValueError, match=f"{section}.{key}"):
        load_config(path)


def test_physical_timestamp_identity_is_strictly_bijective() -> None:
    by_id, by_ns = validate_physical_timestamp_bijection(
        [("174", 174_000), ("174", 174_000), ("175", 175_000)]
    )
    assert by_id == {"174": 174_000, "175": 175_000}
    assert by_ns == {174_000: "174", 175_000: "175"}
    with pytest.raises(ValueError, match="multiple timestamp_ns"):
        validate_physical_timestamp_bijection([("174", 1), ("174", 2)])
    with pytest.raises(ValueError, match="multiple frame_id"):
        validate_physical_timestamp_bijection([("174", 1), ("0174", 1)])


def test_heldout_summaries_weight_physical_timestamps_not_virtual_views() -> None:
    def row(timestamp: int, iou: float) -> dict[str, object]:
        return {
            "physical_timestamp_ns": timestamp,
            "iou": iou,
            "precision": iou,
            "recall": iou,
            "largest_component_fraction": iou,
            "area_ratio": iou,
            "soft_iou": iou,
            "depth_error_median": 1.0 - iou,
            "depth_error_p90": 1.0 - iou,
            "depth_pixels": 10,
        }
    summary = summarize_heldout_by_timestamp(
        [row(100, 0.2) for _ in range(5)] + [row(200, 0.8)]
    )
    assert summary["summaries"]["iou"]["median"] == pytest.approx(0.5)
    assert [item["view_count"] for item in summary["timestamp_rows"]] == [5, 1]


def _write_bound_run_bundle(run: Path) -> None:
    canonical = {
        "scene_state": "final/scene_state.pt",
        "catalog": "final/catalog.json",
        "presentation_catalog": "final/presentation_catalog.json",
        "rgbd": "rgbd",
        "mapping": "mapping",
        "scene_preflight": "input/scene_preflight.json",
        "resource_preflight": "input/resource_preflight.json",
        "final_acceptance": "qa/acceptance/result.json",
    }
    for name, relative in canonical.items():
        path = run / relative
        if name in {"rgbd", "mapping"}:
            path.mkdir(parents=True)
            (path / "payload.bin").write_bytes(name.encode())
        else:
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_bytes(name.encode())
    viewer = run / "viewer"
    viewer.mkdir()
    artifacts = {name: "../" + relative for name, relative in canonical.items()}
    integrity = {}
    for name, relative in canonical.items():
        path = run / relative
        if path.is_dir():
            integrity[name] = {
                "kind": "directory", "path": artifacts[name],
                **directory_tree_descriptor(path),
            }
        else:
            raw = path.read_bytes()
            integrity[name] = {
                "kind": "file", "path": artifacts[name], "bytes": len(raw),
                "sha256": hashlib.sha256(raw).hexdigest(),
            }
    bundle = {
        "schema": "farm.viewer-bundle.v1", "scene_id": "scene", "run_id": run.name,
        "artifacts": artifacts, "artifact_integrity": integrity, "configured": False,
    }
    (viewer / "bundle.json").write_text(json.dumps(bundle), encoding="utf-8")
    success = {
        "schema": "farm.pipeline-success.v1", "status": "success",
        "scene_id": "scene", "run_id": run.name, "viewer_bundle": "viewer/bundle.json",
        "viewer_bundle_sha256": hashlib.sha256((viewer / "bundle.json").read_bytes()).hexdigest(),
    }
    (run / "_SUCCESS.json").write_text(json.dumps(success), encoding="utf-8")


@pytest.mark.parametrize("directory", ["rgbd", "mapping"])
def test_root_bound_directory_inventory_rejects_post_success_mutation(
    tmp_path: Path, directory: str
) -> None:
    run = tmp_path / "run-v3"
    run.mkdir()
    _write_bound_run_bundle(run)
    assert validate_run_artifact_bundle(run, allow_legacy=False) is not None
    (run / directory / "payload.bin").write_bytes(b"mutated-after-success")
    with pytest.raises(ValueError, match="directory differs"):
        validate_run_artifact_bundle(run, allow_legacy=False)


def test_unbound_run_bundle_requires_explicit_legacy_override(tmp_path: Path) -> None:
    run = tmp_path / "legacy-v7"
    run.mkdir()
    success = {
        "schema": "farm.pipeline-success.v1", "status": "success",
        "scene_id": "factory", "run_id": run.name,
        "viewer_bundle": "viewer/bundle.json",
    }
    (run / "_SUCCESS.json").write_text(json.dumps(success), encoding="utf-8")
    with pytest.raises(ValueError, match="root-bound"):
        validate_run_artifact_bundle(run, allow_legacy=False)
    assert validate_run_artifact_bundle(run, allow_legacy=True) is None


def _write_graphdeco_ply(path: Path, count: int = 4) -> np.ndarray:
    fields = [
        ("x", "<f4"), ("y", "<f4"), ("z", "<f4"),
        ("f_dc_0", "<f4"), ("f_dc_1", "<f4"), ("f_dc_2", "<f4"),
        ("f_rest_0", "<f4"), ("opacity", "<f4"),
        ("scale_0", "<f4"), ("scale_1", "<f4"), ("scale_2", "<f4"),
        ("rot_0", "<f4"), ("rot_1", "<f4"), ("rot_2", "<f4"), ("rot_3", "<f4"),
    ]
    rows = np.zeros(count, dtype=np.dtype(fields, align=False))
    rows["x"] = np.arange(count, dtype=np.float32) + 0.125
    rows["y"] = -np.arange(count, dtype=np.float32)
    rows["z"] = np.linspace(1.0, 2.0, count, dtype=np.float32)
    rows["f_dc_0"] = np.linspace(-0.2, 0.2, count, dtype=np.float32)
    rows["f_dc_1"] = np.linspace(0.3, -0.1, count, dtype=np.float32)
    rows["f_dc_2"] = 0.05
    rows["f_rest_0"] = np.arange(count, dtype=np.float32) * 0.01
    rows["opacity"] = 2.0
    rows["scale_0"] = rows["scale_1"] = rows["scale_2"] = -3.0
    rows["rot_0"] = 1.0
    header = [
        "ply",
        "format binary_little_endian 1.0",
        "comment synthetic contract fixture",
        f"element vertex {count}",
    ]
    header.extend(f"property float {name}" for name in rows.dtype.names)
    header.append("end_header")
    path.write_bytes(("\n".join(header) + "\n").encode("ascii") + rows.tobytes())
    return rows


def _split_policy() -> dict[str, object]:
    return {
        "target_heldout_fraction": 0.25,
        "minimum_heldout_fraction": 0.20,
        "maximum_heldout_fraction": 0.35,
        "temporal_bins": 4,
        "require_early_late_heldout": True,
        "strict_minimum_timestamps": 5,
        "strict_minimum_build_timestamps": 3,
        "strict_minimum_heldout_timestamps": 2,
        "evidence_limited_minimum_timestamps": 3,
        "solver_time_limit_seconds": 10,
    }


def test_full_labeled_ply_preserves_source_rows_and_order(tmp_path: Path) -> None:
    source_path = tmp_path / "source.ply"
    expected = _write_graphdeco_ply(source_path)
    source = open_graphdeco_ply(source_path)
    labels = np.asarray([UNKNOWN_ID, 5, 5, UNKNOWN_ID], dtype=np.int32)
    confidence = np.asarray([0.0, 0.75, 0.9, 0.0], dtype=np.float32)
    support = np.asarray([0, 2, 3, 0], dtype=np.uint16)

    manifest = write_full_labeled_ply(
        source, tmp_path / "labeled.ply", labels, confidence, support, chunk_rows=2
    )
    labeled = open_graphdeco_ply(tmp_path / "labeled.ply")

    assert manifest["vertex_count"] == len(expected)
    assert manifest["source_order_preserved"] is True
    assert manifest["original_fields_bitwise_preserved"] is True
    assert manifest["rows_deleted"] == 0
    for name in expected.dtype.names:
        assert np.array_equal(labeled.data[name], expected[name])
    assert np.array_equal(labeled.data["farm_instance_id"], labels)
    assert np.array_equal(labeled.data["farm_instance_confidence"], confidence)
    assert np.array_equal(labeled.data["farm_timestamp_support"], support)
    original_rgb = sh_dc_to_rgb(np.column_stack([
        expected["f_dc_0"], expected["f_dc_1"], expected["f_dc_2"]
    ]))
    assert np.array_equal(
        np.asarray([labeled.data["red"][0], labeled.data["green"][0], labeled.data["blue"][0]]),
        original_rgb[0],
    )
    assert source_path.read_bytes().endswith(expected.tobytes())


def _write_mask(path: Path, *, malformed: bool = False) -> tuple[np.ndarray, np.ndarray]:
    image_shape = (6, 8)
    raw_local = np.asarray([[1, 0, 1], [1, 1, 0]], dtype=np.uint8)
    inlier_local = np.asarray([[1, 0], [0, 1]], dtype=np.uint8)
    np.savez(
        path,
        image_shape=np.asarray(image_shape, dtype=np.int32),
        raw_shape=np.asarray(raw_local.shape, dtype=np.int32),
        raw_bbox_xyxy=np.asarray([2, 1, 6 if malformed else 5, 3], dtype=np.int32),
        raw_bits=np.packbits(raw_local.ravel(), bitorder="little"),
        inlier_shape=np.asarray(inlier_local.shape, dtype=np.int32),
        inlier_bbox_xyxy=np.asarray([2, 1, 4, 3], dtype=np.int32),
        inlier_bits=np.packbits(inlier_local.ravel(), bitorder="little"),
    )
    return raw_local.astype(bool), inlier_local.astype(bool)


def test_mask_npz_requires_native_shape_and_exclusive_bbox(tmp_path: Path) -> None:
    path = tmp_path / "mask.npz"
    raw_local, inlier_local = _write_mask(path)
    raw, inlier, digest = load_mask_pair(path, (6, 8))
    assert len(digest) == 64
    assert np.array_equal(raw[1:3, 2:5], raw_local)
    assert np.array_equal(inlier[1:3, 2:4], inlier_local)
    with pytest.raises(ValueError, match="image_shape"):
        load_mask_pair(path, (3, 4))

    malformed = tmp_path / "malformed.npz"
    _write_mask(malformed, malformed=True)
    with pytest.raises(ValueError, match="shape/bbox"):
        load_mask_pair(malformed, (6, 8))


def test_global_split_is_deterministic_disjoint_and_object_safe() -> None:
    timestamps = [f"t{index:02d}" for index in range(16)]
    objects = {
        1: timestamps,
        2: timestamps[2:14],
        3: timestamps[5:11],
    }
    first = solve_global_timestamp_split(timestamps, objects, _split_policy(), seed=19)
    second = solve_global_timestamp_split(timestamps, objects, _split_policy(), seed=19)
    assert first["heldout_timestamps"] == second["heldout_timestamps"]
    build, heldout = set(first["build_timestamps"]), set(first["heldout_timestamps"])
    assert build.isdisjoint(heldout)
    assert build | heldout == set(timestamps)
    assert 4 <= len(heldout) <= 5
    for row in first["objects"]:
        assert len(row["build_timestamps"]) >= 3
        assert len(row["heldout_timestamps"]) >= 2
        values = row["all_timestamps"]
        midpoint = len(values) // 2
        assert heldout.intersection(values[:midpoint])
        assert heldout.intersection(values[midpoint:])


def test_global_split_rejects_unknown_timestamp_cleanly() -> None:
    with pytest.raises(ValueError, match="unknown timestamp"):
        solve_global_timestamp_split(
            ["t0", "t1", "t2"], {4: ["t0", "not-present", "t2"]}, _split_policy(), seed=1
        )


def test_sparse_conflicts_remain_unknown_instead_of_forced_assignment() -> None:
    claims = [
        {
            "object_id": 10,
            "indices": np.asarray([0, 1, 2]),
            "scores": np.asarray([3.0, 2.0, 1.0]),
            "confidence": np.asarray([0.9, 0.8, 0.7]),
            "support": np.asarray([3, 2, 2], dtype=np.uint16),
        },
        {
            "object_id": 20,
            "indices": np.asarray([1, 2, 3]),
            "scores": np.asarray([1.0, 0.95, 5.0]),
            "confidence": np.asarray([0.6, 0.7, 0.95]),
            "support": np.asarray([2, 2, 4], dtype=np.uint16),
        },
    ]
    labels, confidence, support, audit = resolve_sparse_claims(
        claims, 5, minimum_margin=0.15, minimum_ratio=1.05
    )
    assert labels.tolist() == [10, 10, UNKNOWN_ID, 20, UNKNOWN_ID]
    assert confidence[2] == 0
    assert support[2] == 0
    assert audit["ambiguous"] == 1


def test_verified_csr_uses_original_row_indices_and_stable_object_order() -> None:
    labels = np.asarray([UNKNOWN_ID, 7, 3, 7, UNKNOWN_ID, 3], dtype=np.int32)
    confidence = np.asarray([0, .7, .8, .9, 0, .6], dtype=np.float32)
    support = np.asarray([0, 2, 3, 4, 0, 2], dtype=np.uint16)
    bank = build_verified_csr(labels, confidence, support)
    assert bank["object_ids"].tolist() == [3, 7]
    assert bank["indptr"].tolist() == [0, 2, 4]
    assert bank["indices"].tolist() == [2, 5, 1, 3]
    assert np.array_equal(bank["confidence"], confidence[bank["indices"]])
    assert np.array_equal(bank["timestamp_support"], support[bank["indices"]])


def test_finalize_dense_exports_verified_only_without_runner_up_promotion() -> None:
    provisional = np.asarray([1, 1, 2, 3, UNKNOWN_ID], dtype=np.int32)
    confidence = np.asarray([.9, .8, .7, .6, 0], dtype=np.float32)
    support = np.asarray([3, 2, 2, 1, 0], dtype=np.uint16)
    labels, final_confidence, final_support, status = finalize_dense(
        provisional,
        confidence,
        support,
        [
            {"object_id": 1, "status": "verified"},
            {"object_id": 2, "status": "rejected"},
            {"object_id": 3, "status": "provisional"},
        ],
    )
    assert labels.tolist() == [1, 1, UNKNOWN_ID, UNKNOWN_ID, UNKNOWN_ID]
    assert status.tolist() == [2, 2, 3, 1, 0]
    assert final_confidence.tolist()[:2] == pytest.approx([.9, .8])
    assert np.all(final_confidence[2:] == 0)
    assert final_support.tolist() == [3, 2, 0, 0, 0]


def test_binary_mask_qc_reports_leakage_and_fragmentation() -> None:
    target = np.zeros((8, 8), dtype=bool)
    target[1:5, 1:5] = True
    predicted = target.copy()
    predicted[6, 6] = True
    metrics = binary_mask_metrics(predicted, target, predicted.astype(np.float32))
    assert metrics["iou"] == pytest.approx(16 / 17)
    assert metrics["precision"] == pytest.approx(16 / 17)
    assert metrics["recall"] == 1
    assert metrics["largest_component_fraction"] == pytest.approx(16 / 17)
    assert metrics["soft_iou"] == pytest.approx(16 / 17)


def test_prep_runtime_contract_requires_one_passing_matching_digest() -> None:
    image_id = "sha256:" + "a" * 64
    payload = {
        "checks": [{
            "name": "runtime:prep",
            "status": "pass",
            "metrics": {
                "image": "rest3d:pinned",
                "expected_image_id": image_id,
                "actual_image_id": image_id,
                "python": "/opt/conda/envs/rest3d/bin/python",
            },
        }]
    }
    assert extract_prep_runtime(payload) == (
        "rest3d:pinned", image_id, "/opt/conda/envs/rest3d/bin/python", False
    )
    payload["checks"][0]["metrics"]["actual_image_id"] = "sha256:" + "b" * 64
    with pytest.raises(ValueError, match="pin"):
        extract_prep_runtime(payload)
    payload["checks"][0]["metrics"]["actual_image_id"] = image_id
    payload["checks"][0]["metrics"]["python"] = "/usr/bin/python3"
    with pytest.raises(ValueError, match="interpreter"):
        extract_prep_runtime(payload)


def test_gaussian_lift_standalone_bootstrap_includes_src_runtime() -> None:
    source = (
        Path(__file__).resolve().parents[1]
        / "tools/farm_shaper_bridge/gaussian_lift.py"
    ).read_text(encoding="utf-8")
    assert 'sys.path.insert(0, str(_ROOT / "src"))' in source


def test_prep_runtime_legacy_fallback_is_explicit_and_missing_only() -> None:
    image_id = "sha256:" + "a" * 64
    payload = {
        "checks": [{
            "name": "runtime:prep",
            "status": "pass",
            "metrics": {
                "image": "rest3d:pinned",
                "expected_image_id": image_id,
                "actual_image_id": image_id,
            },
        }]
    }
    with pytest.raises(ValueError, match="does not pin the prep interpreter"):
        extract_prep_runtime(payload)
    assert extract_prep_runtime(
        payload, allow_legacy_interpreter_fallback=True
    ) == (
        "rest3d:pinned", image_id, "/opt/conda/envs/rest3d/bin/python", True
    )
    payload["checks"][0]["metrics"]["python"] = "/usr/bin/python3"
    with pytest.raises(ValueError, match="unsupported prep interpreter"):
        extract_prep_runtime(payload, allow_legacy_interpreter_fallback=True)
    payload["checks"][0]["metrics"].pop("python")
    payload["checks"][0]["metrics"]["actual_image_id"] = "sha256:" + "b" * 64
    with pytest.raises(ValueError, match="image ID"):
        extract_prep_runtime(payload, allow_legacy_interpreter_fallback=True)


def test_acceptance_and_catalog_reconciliation_is_fail_closed() -> None:
    state = {
        "object_id": np.asarray([3, 7, 9], dtype=np.int64),
        "active": np.asarray([True, True, False]),
        "object_semantic_tier": ["confirmed", "probable", "geometry_only"],
        "object_display_status": ["canonical", "duplicate_suppressed", "canonical"],
        "object_geometry_status": ["verified", "verified", "compound_geometry_probable"],
        "object_compound_boxes": [[], [], [{"box": 1}, {"box": 2}]],
    }
    catalog_ids = canonical_presentation_ids(state)
    assert catalog_ids == {3, 9}
    acceptance = {
        "schema": "farm.final-acceptance.v1",
        "mode": "apply",
        "scene_id": "factory",
        "status": "WARN",
        "errors": [],
        "counts": {
            "presentation_after": 2,
            "remaining_auto_clusters": 0,
            "remaining_blocking_clusters": 0,
            "label_hard_errors": 0,
        },
    }
    validate_acceptance_contract(acceptance, scene_id="factory", catalog_ids=catalog_ids)
    for field, value, message in (
        ("mode", "verify", "applied"),
        ("scene_id", "other", "scene_id"),
    ):
        broken = {**acceptance, field: value}
        with pytest.raises(ValueError, match=message):
            validate_acceptance_contract(broken, scene_id="factory", catalog_ids=catalog_ids)
    broken = {**acceptance, "counts": {**acceptance["counts"], "label_hard_errors": 1}}
    with pytest.raises(ValueError, match="blockers"):
        validate_acceptance_contract(broken, scene_id="factory", catalog_ids=catalog_ids)
    broken = {**acceptance, "counts": {**acceptance["counts"], "presentation_after": 3}}
    with pytest.raises(ValueError, match="count mismatch"):
        validate_acceptance_contract(broken, scene_id="factory", catalog_ids=catalog_ids)


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


def test_smoke_scope_reuses_full_run_split_instead_of_resolving_subset(tmp_path: Path) -> None:
    run = RunData(
        run_dir=tmp_path,
        scene_id="scene",
        frames_doc={},
        frames=(),
        objects=(_object(1), _object(2)),
        meters_per_scene_unit=1.0,
        rgbd_config={},
        success_sha256="a" * 64,
        acceptance_sha256="b" * 64,
        resource_preflight={},
        scene_preflight={},
        legacy=False,
    )
    full_split = {
        "physical_timestamp_count": 16,
        "build_timestamps": [f"t{i}" for i in range(12)],
        "heldout_timestamps": [f"t{i}" for i in range(12, 16)],
        "objects": [
            {"object_id": 1, "build_timestamps": ["t0"], "heldout_timestamps": ["t12"]},
            {"object_id": 2, "build_timestamps": ["t1"], "heldout_timestamps": ["t13"]},
        ],
    }
    scoped_run, scoped_split = restrict_smoke_scope(run, full_split, [1])
    assert [obj.object_id for obj in scoped_run.objects] == [1]
    assert scoped_split["build_timestamps"] == full_split["build_timestamps"]
    assert scoped_split["heldout_timestamps"] == full_split["heldout_timestamps"]
    assert [row["object_id"] for row in scoped_split["objects"]] == [1]
    assert scoped_split["scope"] == "smoke_subset_of_canonical_full_split"


def test_runner_mounts_signed_snapshot_and_only_exact_output_rw(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    run = tmp_path / "run"
    (run / "input").mkdir(parents=True)
    image_id = "sha256:" + "a" * 64
    (run / "input" / "resource_preflight.json").write_text(json.dumps({
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
    }), encoding="utf-8")
    tree = "b" * 64
    (run / "manifest.json").write_text(json.dumps({
        "schema": "farm.pipeline-run.v1",
        "status": "success",
        "git": {"commit": "c" * 40, "dirty": False},
        "source_snapshot": {"tree_sha256": tree},
    }), encoding="utf-8")
    snapshot = tmp_path / "snapshot"
    for relative in lift_runner.REQUIRED_LIFT_SOURCE:
        path = snapshot / relative
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text("# pinned\n", encoding="utf-8")
    ply = tmp_path / "source.ply"
    config = tmp_path / "lift.yaml"
    ply.write_bytes(b"ply")
    config.write_bytes((snapshot / "configs/gaussian_lift.v1.yaml").read_bytes())
    output = tmp_path / "outputs" / "lift-v1"

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
        output=output,
        config=config,
        gpu="0",
        plan_only=False,
        smoke_object_id=[],
        allow_legacy_run=False,
        print_command=False,
    ))
    mounts = [command[index + 1] for index, value in enumerate(command[:-1]) if value == "--mount"]
    assert f"type=bind,src={snapshot},dst=/opt/farm-src,readonly" in mounts
    assert f"type=bind,src={run},dst=/farm/runs/{run.name},readonly" in mounts
    assert f"type=bind,src={output},dst=/farm/output" in mounts
    assert not any(f"src={output.parent},dst=/farm/output" in value for value in mounts)
    output_flag = command.index("--output")
    assert command[output_flag + 1] == "/farm/output"
    assert command[command.index("--entrypoint") + 1] == "/opt/conda/envs/rest3d/bin/python"
    assert metadata["python"] == "/opt/conda/envs/rest3d/bin/python"
    assert metadata["container_run"] == f"/farm/runs/{run.name}"
    assert command[command.index("--run") + 1] == f"/farm/runs/{run.name}"
    assert metadata["runtime_interpreter_fallback"] is False
    assert metadata["nonrelease_warnings"] == []
    assert "FARM_BRIDGE_RUNTIME_INTERPRETER_FALLBACK=false" in command
    assert metadata["execution_source"] == {
        "kind": "farm_signed_source_snapshot",
        "commit": "c" * 40,
        "tree_sha256": tree,
        "validated": True,
        "dirty": False,
    }
    rebuilt_image_id = "sha256:" + "e" * 64
    monkeypatch.setattr(lift_runner, "_docker_image_id", lambda _tag: rebuilt_image_id)
    with pytest.raises(RuntimeError, match="prep image drift"):
        lift_runner.build_command(Namespace(
            run=run, ply=ply, output=output, config=config, gpu="0",
            plan_only=False, smoke_object_id=[], allow_legacy_run=False,
            print_command=False,
        ))
    rebuilt_command, rebuilt_metadata = lift_runner.build_command(Namespace(
        run=run, ply=ply, output=output, config=config, gpu="0",
        plan_only=False, smoke_object_id=[], allow_legacy_run=False,
        allow_rebuilt_image_nonrelease=True, print_command=False,
    ))
    assert rebuilt_command[rebuilt_command.index("--entrypoint") + 2] == rebuilt_image_id
    assert f"FARM_BRIDGE_IMAGE_ID={rebuilt_image_id}" in rebuilt_command
    assert "FARM_BRIDGE_SOURCE_DIRTY=true" in rebuilt_command
    assert rebuilt_metadata["source_run_image_id"] == image_id
    assert rebuilt_metadata["image_id"] == rebuilt_image_id
    assert rebuilt_metadata["runtime_image_drift"] is True
    assert rebuilt_metadata["nonrelease_warnings"] == [
        "explicit_rebuilt_image_nonrelease_mode",
        "rebuilt_prep_image_differs_from_source_run",
    ]
    monkeypatch.setattr(lift_runner, "_docker_image_id", lambda _tag: image_id)

    config.write_text("# drifted\n", encoding="utf-8")
    with pytest.raises(ValueError, match="byte-for-byte"):
        lift_runner.build_command(Namespace(
            run=run, ply=ply, output=output, config=config, gpu="0",
            plan_only=True, smoke_object_id=[], allow_legacy_run=False,
            print_command=False,
        ))


def test_runner_legacy_missing_interpreter_uses_audited_nonrelease_fallback(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    run = tmp_path / "factory-generic-v7"
    (run / "input").mkdir(parents=True)
    image_id = "sha256:" + "a" * 64
    (run / "input/resource_preflight.json").write_text(json.dumps({
        "checks": [{
            "name": "runtime:prep",
            "status": "pass",
            "metrics": {
                "image": "rest3d:pinned",
                "expected_image_id": image_id,
                "actual_image_id": image_id,
            },
        }],
    }), encoding="utf-8")
    (run / "manifest.json").write_text(json.dumps({
        "schema": "farm.pipeline-run.v1",
        "status": "success",
        "source_snapshot": None,
    }), encoding="utf-8")
    ply = tmp_path / "factory_scene_last.ply"
    config = tmp_path / "gaussian_lift.v1.yaml"
    ply.write_bytes(b"ply")
    config.write_text("schema_version: fixture\n", encoding="utf-8")
    monkeypatch.setattr(lift_runner, "_docker_image_id", lambda _tag: image_id)
    monkeypatch.setattr(lift_runner, "_git_snapshot", lambda _repo: ("d" * 40, True))

    command, metadata = lift_runner.build_command(Namespace(
        run=run,
        ply=ply,
        output=tmp_path / "lift-plan",
        config=config,
        gpu="0",
        plan_only=True,
        smoke_object_id=[],
        allow_legacy_run=True,
        print_command=True,
    ))
    assert metadata["run"].endswith("/factory-generic-v7")
    assert metadata["container_run"] == "/farm/runs/factory-generic-v7"
    assert command[command.index("--run") + 1] == "/farm/runs/factory-generic-v7"
    assert metadata["runtime_interpreter_fallback"] is True
    assert metadata["nonrelease_warnings"] == [
        "legacy_resource_preflight_missing_prep_interpreter"
    ]
    assert metadata["python"] == "/opt/conda/envs/rest3d/bin/python"
    assert "FARM_BRIDGE_RUNTIME_INTERPRETER_FALLBACK=true" in command
    assert command[command.index("--entrypoint") + 1] == "/opt/conda/envs/rest3d/bin/python"
    assert "--allow-legacy-run" in command
