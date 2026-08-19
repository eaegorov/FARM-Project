from __future__ import annotations

import hashlib
import json
import threading
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
import yaml

from farm_runtime.config import canonical_json
from farm_runtime.unified_viewer import (
    SceneValidationFailure,
    UnifiedViserViewer,
    UnifiedViewerError,
    _canonical_sampled_sha256,
    _markdown_text,
    camera_presets,
    focus_pose,
    is_loopback_host,
    load_bridge_instances,
    load_dense_gaussian_splats,
    load_dense_lift,
    load_farm_preview,
    load_gaussian_splat_arrays,
    load_registry,
    open_binary_ply,
    sha256_file,
    validate_registry,
    viewer_exposure_warning,
)


def _write_json(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2) + "\n", encoding="utf-8")


def _write_source_ply(path: Path, count: int = 4) -> np.dtype:
    dtype = np.dtype([
        ("x", "<f4"), ("y", "<f4"), ("z", "<f4"),
        ("f_dc_0", "<f4"), ("f_dc_1", "<f4"), ("f_dc_2", "<f4"),
        ("opacity", "<f4"),
        ("scale_0", "<f4"), ("scale_1", "<f4"), ("scale_2", "<f4"),
        ("rot_0", "<f4"), ("rot_1", "<f4"), ("rot_2", "<f4"), ("rot_3", "<f4"),
    ])
    rows = np.zeros(count, dtype=dtype)
    rows["x"] = np.arange(count, dtype=np.float32)
    rows["y"] = np.arange(count, dtype=np.float32) + 10
    rows["z"] = -np.arange(count, dtype=np.float32)
    rows["f_dc_0"] = 0.1
    rows["f_dc_1"] = 0.2
    rows["f_dc_2"] = 0.3
    rows["opacity"] = 1.0
    rows["scale_0"] = rows["scale_1"] = rows["scale_2"] = -2.0
    rows["rot_0"] = 1.0
    header = [
        "ply", "format binary_little_endian 1.0", f"element vertex {count}",
        *[f"property float {name}" for name in dtype.names or ()],
        "end_header", "",
    ]
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("wb") as handle:
        handle.write("\n".join(header).encode("ascii"))
        rows.tofile(handle)
    return dtype


def _make_run(tmp_path: Path, *, scene_id: str = "fixture") -> tuple[Path, Path]:
    source = tmp_path / "source.ply"
    _write_source_ply(source)
    source_sha = sha256_file(source)
    run = tmp_path / "farm" / "latest"
    catalog = [{
        "id": 2,
        "category": "cabinet",
        "description": "fixture object",
        "center_m": [1.0, 2.0, 3.0],
        "dimensions_m": [2.0, 1.0, 0.5],
        "wxyz": [1.0, 0.0, 0.0, 0.0],
        "evidence_tier": "robust",
        "semantic_tier": "confirmed",
    }]
    _write_json(run / "final" / "catalog.json", catalog)
    _write_json(run / "final" / "presentation_catalog.json", catalog)
    (run / "final").mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        run / "final" / "cloud.npz",
        xyz=np.asarray([[0, 0, 0], [1, 2, 3], [2, 3, 4]], dtype=np.float32),
        rgb=np.asarray([[10, 20, 30], [40, 50, 60], [70, 80, 90]], dtype=np.uint8),
    )
    config_sha = hashlib.sha256(b"scene-config").hexdigest()
    success = {
        "schema": "farm.pipeline-success.v1",
        "status": "success",
        "scene_id": scene_id,
        "run_id": "fixture-run-v1",
        "config_sha256": config_sha,
        "viewer_bundle": "viewer/bundle.json",
    }
    _write_json(run / "manifest.json", {
        "schema": "farm.pipeline-run.v1",
        "status": "success",
        "scene_id": scene_id,
        "run_id": "fixture-run-v1",
        "config_sha256": config_sha,
    })
    _write_json(run / "final" / "result.json", {
        "schema": "farm.standard-stage.v1", "status": "PASS", "scene_id": scene_id,
    })
    _write_json(run / "viewer" / "result.json", {
        "schema": "farm.viewer-result.v1",
        "status": "PASS",
        "scene_id": scene_id,
        "artifacts": {"catalog": "../final/catalog.json", "cloud": "../final/cloud.npz"},
    })
    _write_json(run / "input" / "resolved_context.json", {
        "schema": "farm.standard-resolved-context.v1",
        "resolved_up": [0.0, -1.0, 0.0],
        "meters_per_scene_unit": 2.0,
    })
    _write_json(run / "input" / "scene_preflight.json", {
        "checks": [{
            "name": "fingerprints",
            "status": "pass",
            "metrics": {"files": [{
                "path": str(source),
                "size_bytes": source.stat().st_size,
                "algorithm": "sha256",
                "digest": source_sha,
                "sampled_offsets": [],
                "chunk_bytes": 1024,
            }]},
        }],
    })
    _write_json(run / "qa" / "acceptance" / "result.json", {
        "schema_version": "farm.final-acceptance.v1",
        "status": "PASS",
        "scene_id": scene_id,
    })
    bundle = {
        "schema": "farm.viewer-bundle.v1",
        "scene_id": scene_id,
        "run_id": "fixture-run-v1",
        "configured": True,
        "artifacts": {
            "presentation_catalog": "../final/presentation_catalog.json",
            "cloud": "../final/cloud.npz",
            "resolved_context": "../input/resolved_context.json",
        },
        "artifact_integrity": {},
    }
    for name, relative in bundle["artifacts"].items():
        path = (run / "viewer" / relative).resolve()
        bundle["artifact_integrity"][name] = {
            "kind": "file",
            "path": relative,
            "bytes": path.stat().st_size,
            "sha256": sha256_file(path),
        }
    _write_json(run / "viewer" / "bundle.json", bundle)
    success["viewer_bundle_sha256"] = sha256_file(run / "viewer" / "bundle.json")
    _write_json(run / "_SUCCESS.json", success)
    return run, source


def _write_registry(
    tmp_path: Path,
    run: Path,
    source: Path,
    *,
    bridge_roots: list[Path] | None = None,
    dense_roots: list[Path] | None = None,
    shaper_roots: list[Path] | None = None,
) -> Path:
    registry = tmp_path / "viewer.yaml"
    registry.write_text(yaml.safe_dump({
        "schema_version": "farm.unified-viewer-registry.v1",
        "scenes": [{
            "scene_id": "fixture",
            "label": "Fixture",
            "farm_run": str(run),
            "source_ply": str(source),
            "source_fingerprint": {
                "algorithm": "sha256",
                "digest": sha256_file(source),
                "size_bytes": source.stat().st_size,
                "chunk_bytes": 1024,
                "sampled_offsets": [],
            },
            "bridge_roots": [str(path) for path in bridge_roots or []],
            "dense_lift_roots": [str(path) for path in dense_roots or []],
            "shaper_roots": [str(path) for path in shaper_roots or []],
        }],
    }, sort_keys=False), encoding="utf-8")
    return registry


def _set_registry_source_fingerprint(
    registry: Path,
    source: Path,
    *,
    algorithm: str,
    chunk_bytes: int = 8,
) -> None:
    raw = yaml.safe_load(registry.read_text(encoding="utf-8"))
    if algorithm == "sha256":
        digest = sha256_file(source)
        offsets: tuple[int, ...] = ()
    else:
        digest, offsets = _canonical_sampled_sha256(source, chunk_bytes=chunk_bytes)
    raw["scenes"][0]["source_fingerprint"] = {
        "algorithm": algorithm,
        "digest": digest,
        "size_bytes": source.stat().st_size,
        "chunk_bytes": chunk_bytes,
        "sampled_offsets": list(offsets),
    }
    registry.write_text(yaml.safe_dump(raw, sort_keys=False), encoding="utf-8")


def _preflight_source_record(run: Path, source: Path) -> tuple[dict[str, object], dict[str, object]]:
    path = run / "input" / "scene_preflight.json"
    report = json.loads(path.read_text(encoding="utf-8"))
    rows = [
        row
        for check in report["checks"]
        if check["name"] == "fingerprints"
        for row in check["metrics"]["files"]
        if Path(str(row["path"])).resolve() == source.resolve()
    ]
    assert len(rows) == 1
    return report, rows[0]


def _artifact(path: Path) -> dict[str, object]:
    return {"path": path.name, "bytes": path.stat().st_size, "sha256": sha256_file(path)}


def _resign_dense(root: Path) -> None:
    result_path = root / "result.json"
    result = json.loads(result_path.read_text(encoding="utf-8"))
    final = result["artifacts"]["final"]
    for record in final.values():
        path = root / str(record["path"])
        record["bytes"] = path.stat().st_size
        record["sha256"] = sha256_file(path)
    _write_json(result_path, result)
    marker_path = root / "_SUCCESS.json"
    marker = json.loads(marker_path.read_text(encoding="utf-8"))
    marker["result_sha256"] = sha256_file(result_path)
    _write_json(marker_path, marker)


def _make_dense_lift(root: Path, run: Path, source: Path) -> None:
    root.mkdir(parents=True)
    object_id = np.asarray([2, -1, 2, 7], dtype=np.int32)
    confidence = np.asarray([0.9, 0.0, 0.8, 0.95], dtype=np.float32)
    support = np.asarray([3, 0, 2, 4], dtype=np.uint16)
    status = np.asarray([2, 0, 2, 2], dtype=np.uint8)
    np.save(root / "per_gaussian_object_id.npy", object_id, allow_pickle=False)
    np.save(root / "per_gaussian_confidence.npy", confidence, allow_pickle=False)
    np.save(root / "per_gaussian_timestamp_support.npy", support, allow_pickle=False)
    np.save(root / "per_gaussian_status.npy", status, allow_pickle=False)
    bank = root / "verified_instance_bank.npz"
    np.savez_compressed(
        bank,
        object_ids=np.asarray([2, 7], dtype=np.int32),
        indptr=np.asarray([0, 2, 3], dtype=np.int64),
        indices=np.asarray([0, 2, 3], dtype=np.int64),
        confidence=np.asarray([0.9, 0.8, 0.95], dtype=np.float32),
        timestamp_support=np.asarray([3, 2, 4], dtype=np.uint16),
    )
    source_dtype = _write_source_ply(source)
    labeled_dtype = np.dtype(list(source_dtype.descr) + [
        ("farm_instance_id", "<i4"),
        ("farm_instance_confidence", "<f4"),
        ("farm_timestamp_support", "<u2"),
        ("red", "u1"), ("green", "u1"), ("blue", "u1"),
    ])
    labeled_rows = np.zeros(4, dtype=labeled_dtype)
    original = np.memmap(
        source,
        dtype=source_dtype,
        mode="r",
        offset=source.stat().st_size - 4 * source_dtype.itemsize,
        shape=(4,),
    )
    for name in source_dtype.names or ():
        labeled_rows[name] = original[name]
    labeled_rows["farm_instance_id"] = object_id
    labeled_rows["farm_instance_confidence"] = confidence
    labeled_rows["farm_timestamp_support"] = support
    labeled_rows["red"] = [255, 10, 255, 0]
    labeled_rows["green"] = [0, 20, 0, 255]
    labeled_rows["blue"] = [0, 30, 0, 0]
    labeled = root / "instance_labeled_full.ply"
    header = [
        "ply", "format binary_little_endian 1.0", "element vertex 4",
        *[f"property float {name}" for name in source_dtype.names or ()],
        "property int farm_instance_id",
        "property float farm_instance_confidence",
        "property ushort farm_timestamp_support",
        "property uchar red", "property uchar green", "property uchar blue",
        "end_header", "",
    ]
    with labeled.open("wb") as handle:
        handle.write("\n".join(header).encode("ascii"))
        labeled_rows.tofile(handle)
    labeled_manifest = root / "labeled_ply_manifest.json"
    _write_json(labeled_manifest, {
        "schema_version": "farm.gaussian-lift.labeled-ply.v1",
        "source_path": str(source),
        "source_bytes": source.stat().st_size,
        "source_body_sha256": "2" * 64,
        "source_full_sha256": sha256_file(source),
        "output_path": str(labeled),
        "output_bytes": labeled.stat().st_size,
        "output_sha256": sha256_file(labeled),
        "vertex_count": 4,
        "original_properties": list(source_dtype.names or ()),
        "appended_properties": [
            "farm_instance_id", "farm_instance_confidence", "farm_timestamp_support",
            "red", "green", "blue",
        ],
        "original_row_bytes": source_dtype.itemsize,
        "output_row_bytes": labeled_dtype.itemsize,
        "original_fields_bitwise_preserved": True,
        "source_order_preserved": True,
        "rows_deleted": 0,
    })
    final_artifacts = {
        "object_id": _artifact(root / "per_gaussian_object_id.npy"),
        "confidence": _artifact(root / "per_gaussian_confidence.npy"),
        "timestamp_support": _artifact(root / "per_gaussian_timestamp_support.npy"),
        "status": _artifact(root / "per_gaussian_status.npy"),
        "verified_instance_bank": _artifact(bank),
        "instance_labeled_full": _artifact(labeled),
        "labeled_ply_manifest": _artifact(labeled_manifest),
    }
    result = {
        "schema_version": "farm.gaussian-lift.result.v1",
        "status": "PASS",
        "quality_status": "PASS",
        "release_eligible": True,
        "scene_id": "fixture",
        "inputs": {
            "farm_run": str(run),
            "farm_success_sha256": sha256_file(run / "_SUCCESS.json"),
            "farm_acceptance_sha256": sha256_file(run / "qa" / "acceptance" / "result.json"),
            "source_ply": str(source),
            "source_ply_sha256": sha256_file(source),
            "source_gaussian_count": 4,
            "config": "fixture.yaml",
            "config_sha256": "1" * 64,
        },
        "contracts": {
            "exact_pinned_gsplat_contributor_vjp": True,
            "native_frame_resolution": True,
            "global_physical_timestamp_holdout": True,
            "dense_arrays_in_source_ply_order": True,
            "verified_only_canonical_labels": True,
            "unknown_instance_id": -1,
            "runner_up_promotion": False,
            "source_ply_mutated": False,
            "gaussians_deleted": 0,
            "labeled_ply_full_graphdeco_schema": True,
            "clean_versioned_source_snapshot": True,
        },
        "counts": {
            "source_gaussians": 4,
            "verified_gaussians": 3,
            "unknown_gaussians": 1,
            "verified_objects": 2,
            "provisional_objects": 0,
            "rejected_objects": 0,
        },
        "artifacts": {"final": final_artifacts},
    }
    result_path = root / "result.json"
    _write_json(result_path, result)
    _write_json(root / "_SUCCESS.json", {
        "schema_version": "farm.gaussian-lift.success.v1",
        "status": "success",
        "release_eligible": True,
        "scene_id": "fixture",
        "result": "result.json",
        "result_sha256": sha256_file(result_path),
        "source_ply_sha256": sha256_file(source),
        "config_sha256": "1" * 64,
    })


def _make_legacy_bridge(root: Path, run: Path, source: Path) -> None:
    cloud_dir = root / "instances"
    cloud_dir.mkdir(parents=True)
    np.savez_compressed(
        cloud_dir / "instance_visual_cloud.npz",
        xyz=np.asarray([[1, 2, 3], [4, 5, 6]], dtype=np.float32),
        rgb=np.asarray([[255, 0, 0], [0, 255, 0]], dtype=np.uint8),
        source_rgb=np.zeros((2, 3), dtype=np.uint8),
        object_id=np.asarray([2, -1], dtype=np.int32),
        confidence=np.ones(2, dtype=np.float16),
        source_index=np.arange(2, dtype=np.int64),
    )

    def legacy(path: Path) -> dict[str, object]:
        stat = path.stat()
        digest = hashlib.sha256()
        with path.open("rb") as handle:
            digest.update(handle.read(1 << 20))
            if stat.st_size > (2 << 20):
                handle.seek(max(0, stat.st_size - (1 << 20)))
                digest.update(handle.read(1 << 20))
        return {
            "path": str(path.resolve()), "bytes": stat.st_size,
            "mtime_ns": stat.st_mtime_ns, "sample_sha256": digest.hexdigest(),
        }

    _write_json(cloud_dir / "segmentation_report.json", {
        "schema_version": "farm-shaper-bridge.instance-segmentation.v1",
        "status": "PASS",
        "inputs": {"farm_run": legacy(run / "_SUCCESS.json"), "source_ply": legacy(source)},
        "objects": {"rows": [{
            "id": 2, "final_gaussians": 1, "observations": 3,
            "median_final_confidence": 0.9,
        }]},
    })
    _write_json(root / "result.json", {
        "schema": "farm-shaper-bridge.result.v1",
        "status": "REVIEW",
        "source_run": str((run / "_SUCCESS.json").resolve()),
    })


def _add_signed_snapshot(run: Path) -> str:
    snapshot_root = run / "config" / "source_snapshot" / "FARM-Project"
    source_file = snapshot_root / "README.md"
    source_file.parent.mkdir(parents=True, exist_ok=True)
    source_file.write_text("signed fixture source\n", encoding="utf-8")
    source_file.chmod(0o644)
    record = {
        "path": "README.md",
        "size": source_file.stat().st_size,
        "mode": 0o644,
        "executable": False,
        "sha256": sha256_file(source_file),
    }
    tree = hashlib.sha256(canonical_json({
        "schema": "farm.source-snapshot-tree.v1",
        "files": [record],
    }).encode("utf-8")).hexdigest()
    _write_json(run / "config" / "source_snapshot" / "manifest.json", {
        "schema": "farm.source-snapshot.v1",
        "project_relative_path": "FARM-Project",
        "tree_schema": "farm.source-snapshot-tree.v1",
        "tree_sha256": tree,
        "file_count": 1,
        "files": [record],
    })
    manifest_path = run / "manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    manifest["source_snapshot"] = {
        "schema": "farm.source-snapshot.v1",
        "relative_path": "config/source_snapshot/FARM-Project",
        "manifest_relative_path": "config/source_snapshot/manifest.json",
        "tree_sha256": tree,
        "file_count": 1,
    }
    _write_json(manifest_path, manifest)
    return tree


def _artifact_from(path: Path, root: Path) -> dict[str, object]:
    return {
        "path": path.resolve().relative_to(root.resolve()).as_posix(),
        "bytes": path.stat().st_size,
        "sha256": sha256_file(path),
    }


def _make_canonical_shaper(root: Path, run: Path, source: Path, dense: Path, tree: str) -> None:
    root.mkdir(parents=True)
    batch = root.parent / "batch"
    batch.mkdir(parents=True)
    batch_result = batch / "result.json"
    _write_json(batch_result, {"schema_version": "farm.shaper-batch.result.v1", "status": "PASS"})
    source_glb = batch / "object_000002.glb"
    source_glb.write_bytes(b"fixture source GLB")
    source_glb_record = _artifact_from(source_glb, batch)

    vertices = np.asarray([[0, 0, 0], [1, 0, 0], [0, 1, 0]], dtype=np.float32)
    faces = np.asarray([[0, 1, 2]], dtype=np.int32)
    object_ids = np.asarray([2, 2, 2], dtype=np.int32)
    colours = np.asarray([[30, 180, 220]] * 3, dtype=np.uint8)
    combined_npz = root / "combined_metric_objects.npz"
    np.savez_compressed(
        combined_npz,
        vertices=vertices,
        faces=faces,
        object_id=object_ids,
        colors=colours,
    )
    combined_glb = root / "combined_metric_objects.glb"
    combined_glb.write_bytes(b"fixture combined GLB")
    npz_record = _artifact_from(combined_npz, root)
    glb_record = _artifact_from(combined_glb, root)
    dense_result_sha = sha256_file(dense / "result.json")
    bank_sha = sha256_file(dense / "verified_instance_bank.npz")
    farm_success = json.loads((run / "_SUCCESS.json").read_text(encoding="utf-8"))
    farm_run_id = str(farm_success["run_id"])
    chain = {
        "farm_success_sha256": sha256_file(run / "_SUCCESS.json"),
        "farm_acceptance_sha256": sha256_file(run / "qa" / "acceptance" / "result.json"),
        "farm_viewer_bundle_sha256": farm_success["viewer_bundle_sha256"],
        "resolved_context_sha256": sha256_file(run / "input" / "resolved_context.json"),
        "shaper_batch": str(batch.resolve()),
        "shaper_batch_result_sha256": sha256_file(batch_result),
        "shaper_inputs_result_sha256": "3" * 64,
        "gaussian_lift_result_sha256": dense_result_sha,
        "verified_instance_bank_sha256": bank_sha,
        "source_ply_sha256": sha256_file(source),
    }
    object_row = {
        "object_id": 2,
        "category": "cabinet",
        "included": True,
        "source_glb": source_glb_record,
        "vertices": 3,
        "faces": 1,
        "bounds_m": [[0, 0, 0], [1, 1, 0]],
        "geometry_qa": {"status": "PASS", "vertices": 3, "faces": 1, "center_error_m": 0.0},
    }
    scene_manifest = {
        "schema_version": "farm.shaper-scene.v2",
        "status": "PASS",
        "quality_status": "PASS",
        "release_eligible": True,
        "scene_id": "fixture",
        "coordinate_units": "metres",
        "coordinate_frame": "FARM metric world",
        "inputs": {"farm_run": str(run.resolve()), "farm_run_id": farm_run_id, **chain},
        "source_authority": {"signed": True, "tree_sha256": tree},
        "contracts": {
            "qa_pass_meshes_only": True,
            "metric_scale_preserved": True,
            "lift_labels_mutated": False,
            "source_ply_mutated": False,
        },
        "objects": {"prepared": 1, "mesh_qa_passed": 1, "excluded": 0, "rows": [object_row]},
        "combined": {
            "vertices": 3,
            "faces": 1,
            "bounds_m": [[0, 0, 0], [1, 1, 0]],
            "glb": glb_record,
            "glb_coordinate_frame": "glTF metric world (+Y up)",
            "npz": npz_record,
            "npz_coordinate_frame": "FARM metric world",
        },
    }
    scene_path = root / "scene_manifest.json"
    _write_json(scene_path, scene_manifest)
    viewer_assets = {
        "schema_version": "farm.shaper-viewer-assets.v1",
        "status": "PASS",
        "scene_id": "fixture",
        "farm_run_id": farm_run_id,
        "source_scene_manifest_sha256": sha256_file(scene_path),
        "source_shaper_batch": str(batch.resolve()),
        "source_shaper_batch_result_sha256": chain["shaper_batch_result_sha256"],
        "shaper_inputs_result_sha256": chain["shaper_inputs_result_sha256"],
        "farm_success_sha256": chain["farm_success_sha256"],
        "farm_acceptance_sha256": chain["farm_acceptance_sha256"],
        "farm_viewer_bundle_sha256": chain["farm_viewer_bundle_sha256"],
        "gaussian_lift_result_sha256": chain["gaussian_lift_result_sha256"],
        "verified_instance_bank_sha256": chain["verified_instance_bank_sha256"],
        "source_ply_sha256": chain["source_ply_sha256"],
        "layers": {
            "combined_gltf": glb_record,
            "combined_farm_metric_npz": npz_record,
        },
        "objects": [{
            "object_id": 2,
            "category": "cabinet",
            "source_glb": source_glb_record,
            "geometry_qa_status": "PASS",
        }],
    }
    assets_path = root / "viewer_assets.json"
    _write_json(assets_path, viewer_assets)
    _write_json(root / "_SUCCESS.json", {
        "schema_version": "farm.shaper-scene.success.v2",
        "status": "success",
        "release_eligible": True,
        "scene_id": "fixture",
        "farm_run_id": farm_run_id,
        "result": "scene_manifest.json",
        "result_sha256": sha256_file(scene_path),
        "viewer_assets": "viewer_assets.json",
        "viewer_assets_sha256": sha256_file(assets_path),
    })


def test_registry_validates_required_farm_source_and_reports_optional_layers(tmp_path: Path) -> None:
    run, source = _make_run(tmp_path)
    registry = _write_registry(tmp_path, run, source)
    validation = validate_registry(registry)
    assert validation.ok
    scene = validation.ready_scenes[0]
    assert scene.farm.run_id == "fixture-run-v1"
    assert scene.source_table.count == 4
    assert scene.layers["farm_preview"].ready
    assert not scene.layers["bridge_instances"].ready
    assert not scene.layers["dense_lift"].ready
    assert scene.layers["source_gaussians"].ready
    assert "all 4 source rows; Viser float16/uint8 quantized DC preview" in (
        scene.layers["source_gaussians"].reason
    )


def test_full_sha_producer_binds_to_legacy_sampled_registry_pin(tmp_path: Path) -> None:
    run, source = _make_run(tmp_path)
    registry = _write_registry(tmp_path, run, source)
    _set_registry_source_fingerprint(
        registry, source, algorithm="sha256-sampled-v1", chunk_bytes=8
    )

    validation = validate_registry(registry)

    assert validation.ok
    assert validation.ready_scenes[0].source_table.count == 4


def test_producer_source_binding_allows_content_verified_mount_relocation(
    tmp_path: Path,
) -> None:
    run, host_source = _make_run(tmp_path / "host")
    mounted_source = tmp_path / "workspace" / "data" / "source.ply"
    mounted_source.parent.mkdir(parents=True)
    mounted_source.write_bytes(host_source.read_bytes())
    registry = _write_registry(tmp_path, run, mounted_source)

    validation = validate_registry(registry)

    assert validation.ok
    assert validation.ready_scenes[0].source_table.path == mounted_source.resolve()


def test_legacy_sampled_producer_binding_still_passes(tmp_path: Path) -> None:
    run, source = _make_run(tmp_path)
    registry = _write_registry(tmp_path, run, source)
    _set_registry_source_fingerprint(
        registry, source, algorithm="sha256-sampled-v1", chunk_bytes=8
    )
    report, row = _preflight_source_record(run, source)
    digest, offsets = _canonical_sampled_sha256(source, chunk_bytes=8)
    row.update({
        "algorithm": "sha256-sampled-v1",
        "digest": digest,
        "chunk_bytes": 8,
        "sampled_offsets": list(offsets),
    })
    _write_json(run / "input" / "scene_preflight.json", report)

    assert validate_registry(registry).ok


def test_full_sha_producer_detects_mutation_outside_registry_sample(tmp_path: Path) -> None:
    run, source = _make_run(tmp_path)
    registry = _write_registry(tmp_path, run, source)
    chunk_bytes = 8
    _set_registry_source_fingerprint(
        registry, source, algorithm="sha256-sampled-v1", chunk_bytes=chunk_bytes
    )
    sampled_before, offsets = _canonical_sampled_sha256(source, chunk_bytes=chunk_bytes)
    covered = {
        index
        for offset in offsets
        for index in range(offset, min(source.stat().st_size, offset + chunk_bytes))
    }
    mutation_index = next(index for index in range(source.stat().st_size) if index not in covered)
    payload = bytearray(source.read_bytes())
    payload[mutation_index] ^= 1
    source.write_bytes(payload)
    assert _canonical_sampled_sha256(source, chunk_bytes=chunk_bytes)[0] == sampled_before

    validation = validate_registry(registry)

    assert not validation.ok
    assert "source PLY sha256 mismatch" in validation.scenes[0].reason


@pytest.mark.parametrize("failure", ["digest", "size", "count"])
def test_producer_source_binding_rejects_wrong_record_semantics(
    tmp_path: Path,
    failure: str,
) -> None:
    run, source = _make_run(tmp_path)
    registry = _write_registry(tmp_path, run, source)
    report, row = _preflight_source_record(run, source)
    if failure == "digest":
        row["digest"] = "0" * 64
    elif failure == "size":
        row["size_bytes"] = source.stat().st_size + 1
    else:
        report["checks"][0]["metrics"]["files"].append(dict(row))
    _write_json(run / "input" / "scene_preflight.json", report)

    validation = validate_registry(registry)

    assert not validation.ok
    if failure == "count":
        assert "exactly one absolute source PLY fingerprint record" in validation.scenes[0].reason
    else:
        assert "source PLY" in validation.scenes[0].reason


def test_registry_fails_closed_for_source_mutation_and_path_escape(tmp_path: Path) -> None:
    run, source = _make_run(tmp_path)
    registry = _write_registry(tmp_path, run, source)
    payload = bytearray(source.read_bytes())
    payload[-1] ^= 1
    source.write_bytes(payload)
    validation = validate_registry(registry)
    assert not validation.ok
    assert isinstance(validation.scenes[0], SceneValidationFailure)
    assert "sha256 mismatch" in validation.scenes[0].reason

    run, source = _make_run(tmp_path / "escape")
    registry = _write_registry(tmp_path / "escape", run, source)
    viewer_result = json.loads((run / "viewer" / "result.json").read_text())
    viewer_result["artifacts"]["catalog"] = "../../outside.json"
    _write_json(run / "viewer" / "result.json", viewer_result)
    validation = validate_registry(registry)
    assert not validation.ok
    assert "escapes" in validation.scenes[0].reason


def test_legacy_bridge_is_read_only_review_layer_bound_to_run_and_source(tmp_path: Path) -> None:
    run, source = _make_run(tmp_path)
    bridge = tmp_path / "bridge"
    _make_legacy_bridge(bridge, run, source)
    registry = _write_registry(tmp_path, run, source, bridge_roots=[bridge])
    scene = validate_registry(registry).ready_scenes[0]
    assert scene.layers["bridge_instances"].ready
    assert scene.layers["bridge_instances"].warning
    assert not scene.layers["shaper_meshes"].ready
    points, colors, labels = load_bridge_instances(scene, max_points=10)
    assert points.shape == colors.shape == (1, 3)
    assert labels.tolist() == [2]


def test_canonical_dense_lift_requires_exact_run_source_and_original_indices(tmp_path: Path) -> None:
    run, source = _make_run(tmp_path)
    dense = tmp_path / "dense"
    _make_dense_lift(dense, run, source)
    registry = _write_registry(tmp_path, run, source, dense_roots=[dense])
    validation = validate_registry(registry, verify_full_source=True)
    assert validation.ok
    scene = validation.ready_scenes[0]
    assert scene.layers["dense_lift"].ready
    points, colors, labels = load_dense_lift(scene, max_points=10)
    assert labels.tolist() == [2, 2, 7]
    np.testing.assert_allclose(points[:, 0], [0.0, 4.0, 6.0])
    np.testing.assert_allclose(points[:, 1], [20.0, 24.0, 26.0])
    assert colors.shape == (3, 3)

    result = json.loads((dense / "result.json").read_text())
    result["inputs"]["farm_success_sha256"] = "f" * 64
    _write_json(dense / "result.json", result)
    marker = json.loads((dense / "_SUCCESS.json").read_text())
    marker["result_sha256"] = sha256_file(dense / "result.json")
    _write_json(dense / "_SUCCESS.json", marker)
    scene = validate_registry(registry).ready_scenes[0]
    assert not scene.layers["dense_lift"].ready
    assert "different FARM" in scene.layers["dense_lift"].reason


@pytest.mark.parametrize(
    ("case", "message"),
    [
        ("fields", "fields differ"),
        ("dtype", "dtypes differ"),
        ("object_order", "object IDs must be sorted"),
        ("index_order", "strictly source-order sorted"),
        ("confidence", "confidence must be finite"),
        ("support", "support must be positive"),
    ],
)
def test_dense_verified_csr_fails_closed_on_noncanonical_bank(
    tmp_path: Path,
    case: str,
    message: str,
) -> None:
    run, source = _make_run(tmp_path)
    dense = tmp_path / "dense"
    _make_dense_lift(dense, run, source)
    bank_path = dense / "verified_instance_bank.npz"
    with np.load(bank_path, allow_pickle=False) as archive:
        arrays = {name: np.asarray(archive[name]) for name in archive.files}
    if case == "fields":
        arrays["unexpected"] = np.asarray([1], dtype=np.int8)
    elif case == "dtype":
        arrays["object_ids"] = arrays["object_ids"].astype(np.int64)
    elif case == "object_order":
        arrays["object_ids"] = arrays["object_ids"][::-1].copy()
    elif case == "index_order":
        arrays["indices"][:2] = arrays["indices"][:2][::-1]
    elif case == "confidence":
        arrays["confidence"][0] = np.nan
    elif case == "support":
        arrays["timestamp_support"][0] = 0
    else:  # pragma: no cover - guarded by parametrization
        raise AssertionError(case)
    np.savez_compressed(bank_path, **arrays)
    _resign_dense(dense)
    registry = _write_registry(tmp_path, run, source, dense_roots=[dense])
    scene = validate_registry(registry).ready_scenes[0]
    assert not scene.layers["dense_lift"].ready
    assert message in scene.layers["dense_lift"].reason


@pytest.mark.parametrize(
    ("artifact", "replacement", "message"),
    [
        ("object_id", 7, "labels differ"),
        ("confidence", 0.7, "confidence differs"),
        ("timestamp_support", 9, "support differs"),
        ("status", 1, "not verified"),
    ],
)
def test_dense_verified_csr_cross_checks_hashed_source_order_sidecars(
    tmp_path: Path,
    artifact: str,
    replacement: int | float,
    message: str,
) -> None:
    run, source = _make_run(tmp_path)
    dense = tmp_path / "dense"
    _make_dense_lift(dense, run, source)
    result = json.loads((dense / "result.json").read_text(encoding="utf-8"))
    record = result["artifacts"]["final"][artifact]
    path = dense / str(record["path"])
    value = np.load(path, allow_pickle=False)
    value[0] = replacement
    np.save(path, value, allow_pickle=False)
    _resign_dense(dense)
    registry = _write_registry(tmp_path, run, source, dense_roots=[dense])
    scene = validate_registry(registry).ready_scenes[0]
    assert not scene.layers["dense_lift"].ready
    assert message in scene.layers["dense_lift"].reason


def test_registry_schema_camera_presets_and_focus_are_deterministic(tmp_path: Path) -> None:
    run, source = _make_run(tmp_path)
    registry = _write_registry(tmp_path, run, source)
    loaded = load_registry(registry)
    assert loaded.scenes[0].scene_id == "fixture"
    points = np.asarray([
        [-2, 0, -1], [-1, 0, 1], [0, 0, 0], [1, 0, -1], [2, 0, 1],
    ], dtype=np.float32)
    first = camera_presets(points, [0, -1, 0])
    second = camera_presets(points, [0, -1, 0])
    for name in ("overview", "front", "side", "top"):
        for left, right in zip(first[name], second[name]):
            np.testing.assert_allclose(left, right)
    top_direction = first["top"][1] - first["top"][0]
    assert abs(float(np.dot(top_direction / np.linalg.norm(top_direction), first["top"][2]))) < 1e-6
    focused = focus_pose(
        np.asarray([[0, 0, 0], [2, 1, 1]], dtype=np.float32),
        [5, -2, 5],
        [0, 0, 0],
        [0, -1, 0],
    )
    np.testing.assert_allclose(focused[1], [1, 0.5, 0.5])
    assert np.isfinite(np.concatenate(focused)).all()


def test_registry_rejects_unknown_fields(tmp_path: Path) -> None:
    path = tmp_path / "bad.yaml"
    path.write_text(yaml.safe_dump({
        "schema_version": "farm.unified-viewer-registry.v1",
        "unexpected": True,
        "scenes": [],
    }), encoding="utf-8")
    with pytest.raises(UnifiedViewerError, match="unknown viewer registry fields"):
        load_registry(path)


def test_gaussian_decoder_preserves_all_rows_metric_covariance_opacity_and_dc(tmp_path: Path) -> None:
    source = tmp_path / "tiny.ply"
    _write_source_ply(source, count=2)
    table = open_binary_ply(source)
    rows = np.memmap(
        source,
        dtype=table.dtype,
        mode="r+",
        offset=table.header_bytes,
        shape=(table.count,),
    )
    rows[1]["scale_0"] = np.log(1.0)
    rows[1]["scale_1"] = np.log(2.0)
    rows[1]["scale_2"] = np.log(3.0)
    rows[1]["rot_0"] = np.sqrt(0.5) * 7.0
    rows[1]["rot_1"] = 0.0
    rows[1]["rot_2"] = 0.0
    rows[1]["rot_3"] = np.sqrt(0.5) * 7.0
    rows.flush()
    centers, covariances, rgbs, opacities = load_gaussian_splat_arrays(
        table,
        meters_per_scene_unit=2.0,
        chunk_rows=1,
    )
    assert centers.shape == (2, 3)
    assert covariances.shape == (2, 3, 3)
    assert rgbs.shape == (2, 3)
    assert opacities.shape == (2, 1)
    np.testing.assert_allclose(centers[1], [2.0, 22.0, -2.0])
    np.testing.assert_allclose(np.diag(covariances[1]), [16.0, 4.0, 36.0], atol=2e-5)
    np.testing.assert_allclose(covariances[1] - np.diag(np.diag(covariances[1])), 0.0, atol=2e-5)
    np.testing.assert_allclose(rgbs[0], 0.5 + 0.28209479177387814 * np.asarray([0.1, 0.2, 0.3]), rtol=1e-6)
    np.testing.assert_allclose(opacities[:, 0], 1.0 / (1.0 + np.exp(-1.0)), rtol=1e-6)
    assert all(value.dtype == np.float32 and value.flags.c_contiguous for value in (centers, covariances, rgbs, opacities))


def test_farm_inventory_rejects_post_success_and_post_validation_mutations(tmp_path: Path) -> None:
    run, source = _make_run(tmp_path / "before")
    registry = _write_registry(tmp_path / "before", run, source)
    catalog = run / "final" / "presentation_catalog.json"
    payload = catalog.read_bytes()
    assert b"cabinet" in payload
    catalog.write_bytes(payload.replace(b"cabinet", b"mutated", 1))
    validation = validate_registry(registry)
    assert not validation.ok
    assert "SHA-256 changed" in validation.scenes[0].reason

    run, source = _make_run(tmp_path / "lazy")
    registry = _write_registry(tmp_path / "lazy", run, source)
    scene = validate_registry(registry).ready_scenes[0]
    cloud = run / "final" / "cloud.npz"
    payload = bytearray(cloud.read_bytes())
    payload[-1] ^= 1
    cloud.write_bytes(payload)
    with pytest.raises(UnifiedViewerError, match="changed after producer completion"):
        load_farm_preview(scene, max_points=10)


def test_legacy_bridge_exact_path_never_bypasses_missing_fingerprint(tmp_path: Path) -> None:
    run, source = _make_run(tmp_path)
    bridge = tmp_path / "bridge"
    _make_legacy_bridge(bridge, run, source)
    report_path = bridge / "instances" / "segmentation_report.json"
    report = json.loads(report_path.read_text(encoding="utf-8"))
    report["inputs"]["farm_run"] = {"path": str((run / "_SUCCESS.json").resolve())}
    _write_json(report_path, report)
    registry = _write_registry(tmp_path, run, source, bridge_roots=[bridge])
    scene = validate_registry(registry).ready_scenes[0]
    assert not scene.layers["bridge_instances"].ready
    assert "different FARM" in scene.layers["bridge_instances"].reason


def test_dense_rehashes_labeled_full_ply_at_validation_and_lazy_load(tmp_path: Path) -> None:
    run, source = _make_run(tmp_path / "lazy")
    dense = tmp_path / "lazy" / "dense"
    _make_dense_lift(dense, run, source)
    registry = _write_registry(tmp_path / "lazy", run, source, dense_roots=[dense])
    scene = validate_registry(registry).ready_scenes[0]
    labeled = dense / "instance_labeled_full.ply"
    payload = bytearray(labeled.read_bytes())
    payload[-1] ^= 1
    labeled.write_bytes(payload)
    with pytest.raises(UnifiedViewerError, match=r"dense lift artifact\["):
        load_dense_lift(scene, max_points=10)

    run, source = _make_run(tmp_path / "startup")
    dense = tmp_path / "startup" / "dense"
    _make_dense_lift(dense, run, source)
    labeled = dense / "instance_labeled_full.ply"
    payload = bytearray(labeled.read_bytes())
    payload[-1] ^= 1
    labeled.write_bytes(payload)
    registry = _write_registry(tmp_path / "startup", run, source, dense_roots=[dense])
    scene = validate_registry(registry).ready_scenes[0]
    assert not scene.layers["dense_lift"].ready
    assert "instance_labeled_full PLY SHA-256 mismatch" in scene.layers["dense_lift"].reason


def test_dense_all_verified_splats_and_canonical_shaper_v2_exact_assets(tmp_path: Path) -> None:
    run, source = _make_run(tmp_path)
    tree = _add_signed_snapshot(run)
    dense = tmp_path / "dense"
    _make_dense_lift(dense, run, source)
    shaper = tmp_path / "assembly"
    _make_canonical_shaper(shaper, run, source, dense, tree)
    registry = _write_registry(
        tmp_path,
        run,
        source,
        dense_roots=[dense],
        shaper_roots=[shaper],
    )
    scene = validate_registry(registry, verify_full_source=True).ready_scenes[0]
    assert scene.layers["dense_lift"].ready
    assert scene.layers["shaper_meshes"].ready
    centers, covariances, rgbs, opacities, labels = load_dense_gaussian_splats(scene)
    assert len(centers) == scene.dense_lift.verified_gaussians == 3
    assert labels.tolist() == [2, 2, 7]
    assert covariances.shape == (3, 3, 3)
    assert rgbs.shape == (3, 3)
    assert opacities.shape == (3, 1)

    combined = shaper / "combined_metric_objects.npz"
    payload = bytearray(combined.read_bytes())
    payload[-1] ^= 1
    combined.write_bytes(payload)
    scene = validate_registry(registry).ready_scenes[0]
    assert not scene.layers["shaper_meshes"].ready
    assert "SHA-256 changed" in scene.layers["shaper_meshes"].reason


def test_shaper_chain_allows_cryptographically_bound_farm_run_mount_relocation(
    tmp_path: Path,
) -> None:
    run, source = _make_run(tmp_path)
    tree = _add_signed_snapshot(run)
    dense = tmp_path / "dense"
    _make_dense_lift(dense, run, source)
    shaper = tmp_path / "assembly"
    _make_canonical_shaper(shaper, run, source, dense, tree)
    scene_path = shaper / "scene_manifest.json"
    scene_manifest = json.loads(scene_path.read_text(encoding="utf-8"))
    scene_manifest["inputs"]["farm_run"] = str((tmp_path / "other" / run.name).resolve())
    _write_json(scene_path, scene_manifest)
    assets_path = shaper / "viewer_assets.json"
    assets = json.loads(assets_path.read_text(encoding="utf-8"))
    assets["source_scene_manifest_sha256"] = sha256_file(scene_path)
    _write_json(assets_path, assets)
    marker_path = shaper / "_SUCCESS.json"
    marker = json.loads(marker_path.read_text(encoding="utf-8"))
    marker["result_sha256"] = sha256_file(scene_path)
    marker["viewer_assets_sha256"] = sha256_file(assets_path)
    _write_json(marker_path, marker)
    registry = _write_registry(
        tmp_path, run, source, dense_roots=[dense], shaper_roots=[shaper]
    )
    scene = validate_registry(registry).ready_scenes[0]
    assert scene.layers["shaper_meshes"].ready


@pytest.mark.parametrize("field", ["farm_run_id", "farm_viewer_bundle_sha256"])
def test_shaper_relocated_chain_rejects_wrong_run_identity(
    tmp_path: Path,
    field: str,
) -> None:
    run, source = _make_run(tmp_path)
    tree = _add_signed_snapshot(run)
    dense = tmp_path / "dense"
    _make_dense_lift(dense, run, source)
    shaper = tmp_path / "assembly"
    _make_canonical_shaper(shaper, run, source, dense, tree)
    scene_path = shaper / "scene_manifest.json"
    scene_manifest = json.loads(scene_path.read_text(encoding="utf-8"))
    scene_manifest["inputs"]["farm_run"] = "/host/output/farm/fixture-run-v1"
    scene_manifest["inputs"][field] = "0" * 64 if field.endswith("sha256") else "wrong-run"
    _write_json(scene_path, scene_manifest)
    assets_path = shaper / "viewer_assets.json"
    assets = json.loads(assets_path.read_text(encoding="utf-8"))
    assets["source_scene_manifest_sha256"] = sha256_file(scene_path)
    if field == "farm_viewer_bundle_sha256":
        assets[field] = scene_manifest["inputs"][field]
    _write_json(assets_path, assets)
    marker_path = shaper / "_SUCCESS.json"
    marker = json.loads(marker_path.read_text(encoding="utf-8"))
    marker["result_sha256"] = sha256_file(scene_path)
    marker["viewer_assets_sha256"] = sha256_file(assets_path)
    _write_json(marker_path, marker)
    registry = _write_registry(
        tmp_path, run, source, dense_roots=[dense], shaper_roots=[shaper]
    )

    scene = validate_registry(registry).ready_scenes[0]

    assert not scene.layers["shaper_meshes"].ready
    assert "different FARM/lift/source chain" in scene.layers["shaper_meshes"].reason


def test_shaper_uniform_color_contract_and_pinned_viser_call_shapes(tmp_path: Path) -> None:
    run, source = _make_run(tmp_path)
    tree = _add_signed_snapshot(run)
    dense = tmp_path / "dense"
    _make_dense_lift(dense, run, source)
    shaper = tmp_path / "assembly"
    _make_canonical_shaper(shaper, run, source, dense, tree)
    registry = _write_registry(
        tmp_path, run, source, dense_roots=[dense], shaper_roots=[shaper]
    )
    scene = validate_registry(registry, verify_full_source=True).ready_scenes[0]

    class Handle:
        def on_click(self, callback: object) -> object:
            self.callback = callback
            return callback

    class PinnedViserScene:
        def __init__(self) -> None:
            self.mesh_calls: list[dict[str, object]] = []
            self.gaussian_calls: list[dict[str, object]] = []

        def add_mesh_simple(
            self,
            name: str,
            *,
            vertices: np.ndarray,
            faces: np.ndarray,
            color: tuple[int, int, int],
            material: str,
            flat_shading: bool,
            side: str,
            cast_shadow: bool,
            receive_shadow: bool,
        ) -> Handle:
            assert isinstance(color, tuple) and len(color) == 3
            assert all(isinstance(channel, int) and 0 <= channel <= 255 for channel in color)
            self.mesh_calls.append({"name": name, "vertices": vertices, "faces": faces, "color": color})
            return Handle()

        def add_gaussian_splats(
            self,
            name: str,
            *,
            centers: np.ndarray,
            covariances: np.ndarray,
            rgbs: np.ndarray,
            opacities: np.ndarray,
        ) -> Handle:
            assert centers.shape == (4, 3)
            assert covariances.shape == (4, 3, 3)
            assert rgbs.shape == (4, 3)
            assert opacities.shape == (4, 1)
            self.gaussian_calls.append({"name": name, "count": len(centers)})
            return Handle()

    api = PinnedViserScene()
    viewer = UnifiedViserViewer.__new__(UnifiedViserViewer)
    viewer.server = SimpleNamespace(scene=api)
    viewer._mutation_lock = threading.RLock()
    runtime = SimpleNamespace(scene=scene, bounds_by_id={}, mesh_rows_by_id={})
    mesh_layer = SimpleNamespace(handles=[])
    source_layer = SimpleNamespace(handles=[])
    viewer._load_shaper_layer(runtime, mesh_layer)
    viewer._load_source_layer(runtime, source_layer)
    assert api.mesh_calls[0]["color"] == (30, 180, 220)
    assert api.gaussian_calls[0]["count"] == 4
    assert len(mesh_layer.handles) == len(source_layer.handles) == 1


def test_shaper_rejects_nonuniform_vertex_colors_even_when_resigned(tmp_path: Path) -> None:
    run, source = _make_run(tmp_path)
    tree = _add_signed_snapshot(run)
    dense = tmp_path / "dense"
    _make_dense_lift(dense, run, source)
    shaper = tmp_path / "assembly"
    _make_canonical_shaper(shaper, run, source, dense, tree)
    combined = shaper / "combined_metric_objects.npz"
    with np.load(combined, allow_pickle=False) as archive:
        arrays = {name: np.asarray(archive[name]) for name in archive.files}
    arrays["colors"][1] = [255, 0, 0]
    np.savez_compressed(combined, **arrays)
    record = _artifact_from(combined, shaper)
    scene_path = shaper / "scene_manifest.json"
    scene_manifest = json.loads(scene_path.read_text(encoding="utf-8"))
    scene_manifest["combined"]["npz"] = record
    _write_json(scene_path, scene_manifest)
    assets_path = shaper / "viewer_assets.json"
    assets = json.loads(assets_path.read_text(encoding="utf-8"))
    assets["layers"]["combined_farm_metric_npz"] = record
    assets["source_scene_manifest_sha256"] = sha256_file(scene_path)
    _write_json(assets_path, assets)
    marker_path = shaper / "_SUCCESS.json"
    marker = json.loads(marker_path.read_text(encoding="utf-8"))
    marker["result_sha256"] = sha256_file(scene_path)
    marker["viewer_assets_sha256"] = sha256_file(assets_path)
    _write_json(marker_path, marker)
    registry = _write_registry(
        tmp_path, run, source, dense_roots=[dense], shaper_roots=[shaper]
    )
    scene = validate_registry(registry).ready_scenes[0]
    assert not scene.layers["shaper_meshes"].ready
    assert "one uniform RGB color" in scene.layers["shaper_meshes"].reason


def test_markdown_text_neutralises_link_image_and_code_injection() -> None:
    rendered = _markdown_text("![click](javascript:alert(1)) `break` <img src=x>")
    assert "![" not in rendered
    assert "](javascript" not in rendered
    assert "`" not in rendered
    assert "<img" not in rendered


def test_ui_redacts_absolute_paths_and_warns_for_unauthenticated_nonloopback() -> None:
    rendered = _markdown_text("failed to load /home/user/private/scene.ply")
    assert "/home/user" not in rendered
    assert "local path" in rendered
    assert is_loopback_host("127.0.0.1")
    assert is_loopback_host("::1")
    assert is_loopback_host("localhost")
    assert viewer_exposure_warning("127.0.0.1") is None
    warning = viewer_exposure_warning("0.0.0.0")
    assert warning is not None
    assert "single-user" in warning and "no authentication" in warning
