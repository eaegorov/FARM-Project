#!/usr/bin/env python3
"""Assemble only QA-passing ShapeR meshes into a metric scene derivative."""

from __future__ import annotations

import argparse
import json
import os
import sys
import tempfile
import time
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np
import trimesh

if __package__ in (None, ""):
    ROOT = Path(__file__).resolve().parents[2]
    sys.path.insert(0, str(ROOT))
    sys.path.insert(0, str(ROOT / "src"))

from tools.farm_shaper_bridge.common import (  # noqa: E402
    atomic_json,
    load_run,
    revalidate_run_integrity,
    sha256_file,
    utc_now,
)
from tools.farm_shaper_bridge.shaper_contracts import (  # noqa: E402
    SHAPER_BATCH_SCHEMA,
    artifact_record,
    require_current_sources_match_authority,
    resolve_farm_source_authority,
    safe_name,
    validate_success_result,
)


SOURCE_FILES = (
    "tools/farm_shaper_bridge/assemble_scene.py",
    "tools/farm_shaper_bridge/shaper_contracts.py",
    "tools/farm_shaper_bridge/common.py",
)
PALETTE = np.asarray([
    [59, 130, 246, 255], [236, 72, 153, 255], [37, 194, 160, 255],
    [250, 173, 20, 255], [139, 92, 246, 255], [238, 85, 76, 255],
    [39, 174, 224, 255], [151, 191, 62, 255], [244, 114, 182, 255],
    [52, 211, 153, 255], [96, 165, 250, 255], [251, 191, 36, 255],
], dtype=np.uint8)


def validate_batch_farm_binding(batch: Mapping[str, Any], run: Any) -> None:
    batch_inputs = batch.get("inputs")
    if not isinstance(batch_inputs, Mapping):
        raise ValueError("ShapeR batch input provenance is absent")
    if (
        str(batch.get("scene_id") or "") != run.scene_id
        or batch_inputs.get("farm_success_sha256") != run.success_sha256
        or batch_inputs.get("farm_acceptance_sha256") != run.acceptance_sha256
    ):
        raise ValueError("ShapeR batch is not bound to this exact FARM run/scene/acceptance")


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run", type=Path, required=True)
    parser.add_argument("--shaper-outputs", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--allow-unsigned-source", action="store_true")
    return parser.parse_args(argv)


def _normalise(value: Any, *, label: str) -> np.ndarray:
    vector = np.asarray(value, dtype=np.float64).reshape(3)
    norm = float(np.linalg.norm(vector))
    if not np.isfinite(vector).all() or not np.isfinite(norm) or norm <= 1.0e-9:
        raise ValueError(f"{label} must be one finite non-zero 3-vector")
    return vector / norm


def rotation_from_to(source: Any, target: Any) -> np.ndarray:
    """Return a proper homogeneous rotation mapping source onto target."""

    left = _normalise(source, label="source up")
    right = _normalise(target, label="target up")
    cosine = float(np.clip(np.dot(left, right), -1.0, 1.0))
    if cosine > 1.0 - 1.0e-10:
        rotation = np.eye(3)
    elif cosine < -1.0 + 1.0e-10:
        axes = np.eye(3)
        seed = axes[int(np.argmin(np.abs(axes @ left)))]
        axis = seed - np.dot(seed, left) * left
        axis /= np.linalg.norm(axis)
        rotation = 2.0 * np.outer(axis, axis) - np.eye(3)
    else:
        cross = np.cross(left, right)
        skew = np.asarray([
            [0.0, -cross[2], cross[1]],
            [cross[2], 0.0, -cross[0]],
            [-cross[1], cross[0], 0.0],
        ])
        rotation = np.eye(3) + skew + skew @ skew * ((1.0 - cosine) / np.dot(cross, cross))
    transform = np.eye(4)
    transform[:3, :3] = rotation
    if not np.allclose(rotation @ left, right, atol=1.0e-7) or not np.isclose(
        np.linalg.det(rotation), 1.0, atol=1.0e-7
    ):
        raise RuntimeError("failed to construct a proper source-up rotation")
    return transform


def _single_mesh(path: Path) -> trimesh.Trimesh:
    loaded = trimesh.load(path, process=False)
    if isinstance(loaded, trimesh.Scene):
        meshes = [value for value in loaded.geometry.values() if isinstance(value, trimesh.Trimesh)]
        if not meshes:
            raise ValueError(f"GLB contains no triangle mesh: {path}")
        return trimesh.util.concatenate(meshes)
    if not isinstance(loaded, trimesh.Trimesh):
        raise TypeError(f"unsupported geometry in {path}: {type(loaded).__name__}")
    return loaded


def _atomic_bytes(path: Path, payload: bytes) -> None:
    fd, temporary = tempfile.mkstemp(prefix=f".{path.name}.", suffix=".tmp", dir=path.parent)
    try:
        with os.fdopen(fd, "wb") as stream:
            stream.write(payload)
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary, path)
    finally:
        try:
            os.unlink(temporary)
        except FileNotFoundError:
            pass


def _atomic_npz(path: Path, **arrays: np.ndarray) -> None:
    fd, temporary = tempfile.mkstemp(prefix=f".{path.name}.", suffix=".tmp", dir=path.parent)
    try:
        with os.fdopen(fd, "wb") as stream:
            np.savez_compressed(stream, **arrays)
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary, path)
    finally:
        try:
            os.unlink(temporary)
        except FileNotFoundError:
            pass


def _validated_mesh_rows(source: Path, batch: Mapping[str, Any]) -> list[tuple[Mapping[str, Any], Path]]:
    result: list[tuple[Mapping[str, Any], Path]] = []
    for row in (batch.get("objects") or {}).get("rows") or []:
        if not isinstance(row, Mapping) or row.get("status") != "PASS":
            continue
        record = row.get("glb")
        if not isinstance(record, Mapping):
            raise ValueError("QA-passing ShapeR row has no GLB artifact")
        relative = str(record.get("path") or "")
        if Path(relative).name != relative or not relative.endswith(".glb"):
            raise ValueError("unsafe ShapeR GLB artifact path")
        path = (source / relative).resolve(strict=True)
        if not path.is_file() or not path.is_relative_to(source):
            raise ValueError("ShapeR GLB artifact escapes its immutable batch directory")
        if path.stat().st_size != int(record.get("bytes", -1)) or sha256_file(path) != record.get("sha256"):
            raise ValueError(f"ShapeR GLB artifact fingerprint mismatch: {relative}")
        result.append((row, path))
    if len(result) != int((batch.get("objects") or {}).get("passed", -1)) or not result:
        raise ValueError("ShapeR PASS count/GLB artifacts mismatch")
    return result


def execute(args: argparse.Namespace) -> dict[str, Any]:
    started = time.perf_counter()
    project_root = Path(__file__).resolve().parents[2]
    run_dir = args.run.expanduser().resolve(strict=True)
    run = load_run(run_dir, allow_legacy=bool(args.allow_unsigned_source))
    source = args.shaper_outputs.expanduser().resolve(strict=True)
    batch, marker_name, _ = validate_success_result(
        source, result_name="result.json", expected_schema=SHAPER_BATCH_SCHEMA
    )
    recorded_fallback = (batch.get("source_authority") or {}).get("fallback")
    legacy_nonrelease_fallback = bool(
        run.legacy
        and args.allow_unsigned_source
        and marker_name == "_NONRELEASE_SUCCESS.json"
        and not batch.get("release_eligible")
        and isinstance(recorded_fallback, Mapping)
        and recorded_fallback.get("warning")
        == "legacy_unsigned_fallback_from_incomplete_signed_snapshot"
    )
    authority = resolve_farm_source_authority(
        run_dir,
        project_root,
        required_files=SOURCE_FILES,
        allow_unsigned=bool(args.allow_unsigned_source),
        allow_incomplete_snapshot_fallback=legacy_nonrelease_fallback,
    )
    require_current_sources_match_authority(authority, project_root, SOURCE_FILES)
    if authority.fallback is not None:
        if (
            not isinstance(recorded_fallback, Mapping)
            or recorded_fallback.get("incomplete_snapshot_tree_sha256")
            != authority.fallback.get("incomplete_snapshot_tree_sha256")
        ):
            raise ValueError("ShapeR batch and legacy source fallback evidence differ")
        print(
            "WARNING: assembling ShapeR from current versioned FARM source in explicit "
            "unsigned/nonrelease legacy mode",
            file=sys.stderr,
            flush=True,
        )
    validate_batch_farm_binding(batch, run)
    if str((batch.get("runtime") or {}).get("farm_source_tree") or "") != str(authority.tree_sha256 or ""):
        raise ValueError("ShapeR batch and selected FARM source authority differ")
    mesh_rows = _validated_mesh_rows(source, batch)
    resolved_context_path = run_dir / "input" / "resolved_context.json"
    resolved_context = json.loads(resolved_context_path.read_text(encoding="utf-8"))
    if not isinstance(resolved_context, Mapping) or "resolved_up" not in resolved_context:
        raise ValueError("FARM resolved_context.json misses resolved_up")
    source_up = _normalise(resolved_context["resolved_up"], label="resolved_up")
    gltf_up = np.asarray([0.0, 1.0, 0.0])
    gltf_from_farm = rotation_from_to(source_up, gltf_up)
    output = args.output.expanduser().resolve()
    if output.exists() and (not output.is_dir() or any(output.iterdir())):
        raise FileExistsError(f"refusing non-empty ShapeR scene output: {output}")
    output.mkdir(parents=True, exist_ok=True)

    scene = trimesh.Scene(base_frame="gltf_metric_world_y_up")
    catalog: list[dict[str, Any]] = []
    packed_vertices: list[np.ndarray] = []
    packed_faces: list[np.ndarray] = []
    packed_ids: list[np.ndarray] = []
    packed_colours: list[np.ndarray] = []
    face_offset = 0
    for row, mesh_path in mesh_rows:
        mesh = _single_mesh(mesh_path)
        vertices = np.asarray(mesh.vertices, dtype=np.float32)
        faces = np.asarray(mesh.faces, dtype=np.int32)
        if (
            vertices.ndim != 2 or vertices.shape[1] != 3
            or faces.ndim != 2 or faces.shape[1] != 3
            or not len(vertices) or not len(faces)
            or not np.isfinite(vertices).all()
        ):
            raise ValueError(f"invalid QA-passing triangle mesh: {mesh_path}")
        object_id = int(row["object_id"])
        category = str(row.get("category") or "object")
        colour = PALETTE[object_id % len(PALETTE)]
        mesh.visual.vertex_colors = np.tile(colour, (len(vertices), 1))
        node_name = f"object_{object_id:06d}__{safe_name(category, fallback='object')}"
        gltf_mesh = mesh.copy()
        gltf_mesh.apply_transform(gltf_from_farm)
        scene.add_geometry(gltf_mesh, node_name=node_name, geom_name=node_name)
        packed_vertices.append(vertices)
        packed_faces.append(faces + face_offset)
        packed_ids.append(np.full(len(vertices), object_id, dtype=np.int32))
        packed_colours.append(np.tile(colour[:3], (len(vertices), 1)))
        face_offset += len(vertices)
        catalog.append({
            "object_id": object_id,
            "category": category,
            "included": True,
            "source_glb": artifact_record(mesh_path, root=source),
            "vertices": int(len(vertices)),
            "faces": int(len(faces)),
            "bounds_m": np.asarray(mesh.bounds, dtype=float).tolist(),
            "geometry_qa": row["geometry_qa"],
        })

    vertices = np.concatenate(packed_vertices, axis=0)
    faces = np.concatenate(packed_faces, axis=0)
    object_ids = np.concatenate(packed_ids, axis=0)
    colours = np.concatenate(packed_colours, axis=0)
    glb_path = output / "combined_metric_objects.glb"
    exported = scene.export(file_type="glb")
    if not isinstance(exported, (bytes, bytearray)):
        raise RuntimeError("trimesh did not return a binary combined GLB")
    _atomic_bytes(glb_path, bytes(exported))
    npz_path = output / "combined_metric_objects.npz"
    _atomic_npz(
        npz_path,
        vertices=vertices,
        faces=faces,
        object_id=object_ids,
        colors=colours,
    )
    release_eligible = bool(
        authority.signed
        and not run.legacy
        and batch.get("release_eligible")
        and marker_name == "_SUCCESS.json"
    )
    result = {
        "schema_version": "farm.shaper-scene.v2",
        "status": "PASS",
        "quality_status": batch.get("quality_status", "WARN"),
        "release_eligible": release_eligible,
        "scene_id": run.scene_id,
        "created_utc": utc_now(),
        "coordinate_units": "metres",
        "coordinate_frame": "FARM metric world",
        "inputs": {
            "farm_run": str(run_dir),
            "farm_run_id": run_dir.name,
            "farm_success_sha256": run.success_sha256,
            "farm_acceptance_sha256": run.acceptance_sha256,
            "farm_viewer_bundle_sha256": (
                run.integrity.get("viewer_bundle_sha256")
                if run.integrity is not None else None
            ),
            "resolved_context_sha256": sha256_file(resolved_context_path),
            "shaper_batch": str(source),
            "shaper_batch_result_sha256": sha256_file(source / "result.json"),
            "shaper_inputs_result_sha256": batch["inputs"]["shaper_inputs_result_sha256"],
            "gaussian_lift_result_sha256": batch["inputs"]["gaussian_lift_result_sha256"],
            "verified_instance_bank_sha256": batch["inputs"]["verified_instance_bank_sha256"],
            "source_ply_sha256": batch["inputs"]["source_ply_sha256"],
        },
        "source_authority": {
            "signed": authority.signed,
            "tree_sha256": authority.tree_sha256,
            "project_root": str(authority.project_root),
            "fallback": authority.fallback,
        },
        "contracts": {
            "qa_pass_meshes_only": True,
            "metric_scale_preserved": True,
            "lift_labels_mutated": False,
            "source_ply_mutated": False,
        },
        "objects": {
            "prepared": int(batch["objects"]["requested"]),
            "mesh_qa_passed": len(catalog),
            "excluded": int(batch["objects"]["requested"]) - len(catalog),
            "rows": catalog,
        },
        "combined": {
            "vertices": int(len(vertices)),
            "faces": int(len(faces)),
            "bounds_m": [vertices.min(axis=0).tolist(), vertices.max(axis=0).tolist()],
            "glb": artifact_record(glb_path, root=output),
            "glb_coordinate_frame": "glTF metric world (+Y up)",
            "gltf_from_farm_world": gltf_from_farm.tolist(),
            "npz": artifact_record(npz_path, root=output),
            "npz_coordinate_frame": "FARM metric world",
        },
        "timing_seconds": {"total": time.perf_counter() - started},
        "limitations": [
            "ShapeR surfaces remain generative hypotheses even after geometry QA.",
            "Objects rejected by input or mesh QA remain absent rather than being silently promoted.",
        ],
    }
    atomic_json(output / "scene_manifest.json", result)
    viewer_assets = {
        "schema_version": "farm.shaper-viewer-assets.v1",
        "status": "PASS",
        "scene_id": run.scene_id,
        "source_scene_manifest_sha256": sha256_file(output / "scene_manifest.json"),
        "source_shaper_batch": str(source),
        "source_shaper_batch_result_sha256": sha256_file(source / "result.json"),
        "shaper_inputs_result_sha256": batch["inputs"]["shaper_inputs_result_sha256"],
        "farm_run_id": run_dir.name,
        "farm_success_sha256": run.success_sha256,
        "farm_acceptance_sha256": run.acceptance_sha256,
        "farm_viewer_bundle_sha256": (
            run.integrity.get("viewer_bundle_sha256")
            if run.integrity is not None else None
        ),
        "gaussian_lift_result_sha256": batch["inputs"]["gaussian_lift_result_sha256"],
        "verified_instance_bank_sha256": batch["inputs"]["verified_instance_bank_sha256"],
        "source_ply_sha256": batch["inputs"]["source_ply_sha256"],
        "layers": {
            "combined_gltf": artifact_record(glb_path, root=output),
            "combined_farm_metric_npz": artifact_record(npz_path, root=output),
        },
        "objects": [
            {
                "object_id": row["object_id"],
                "category": row["category"],
                "source_glb": row["source_glb"],
                "geometry_qa_status": row["geometry_qa"]["status"],
            }
            for row in catalog
        ],
    }
    atomic_json(output / "viewer_assets.json", viewer_assets)
    revalidate_run_integrity(run)
    marker_name = "_SUCCESS.json" if release_eligible else "_NONRELEASE_SUCCESS.json"
    atomic_json(output / marker_name, {
        "schema_version": "farm.shaper-scene.success.v2",
        "status": "success",
        "release_eligible": release_eligible,
        "scene_id": run.scene_id,
        "farm_run_id": run_dir.name,
        "farm_success_sha256": run.success_sha256,
        "farm_acceptance_sha256": run.acceptance_sha256,
        "result": "scene_manifest.json",
        "result_sha256": sha256_file(output / "scene_manifest.json"),
        "viewer_assets": "viewer_assets.json",
        "viewer_assets_sha256": sha256_file(output / "viewer_assets.json"),
    })
    print(json.dumps({
        "status": "PASS",
        "quality_status": result["quality_status"],
        "release_eligible": release_eligible,
        "included_meshes": len(catalog),
        "vertices": len(vertices),
        "faces": len(faces),
    }, indent=2))
    return result


def main(argv: Sequence[str] | None = None) -> int:
    try:
        execute(parse_args(argv))
        return 0
    except Exception as exc:
        print(f"ERROR: {type(exc).__name__}: {exc}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
