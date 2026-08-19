#!/usr/bin/env python3
"""Run one pinned ShapeR model load over validated FARM object PKLs.

The script runs inside the ShapeR container against a clean archived upstream
commit.  Its pinhole ingestion adapter is versioned here, so no dirty ShapeR
worktree file is runtime authority.
"""

from __future__ import annotations

import argparse
import gc
import json
import math
import os
import pickle
import random
import re
import sys
import tempfile
import time
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np
import torch
import trimesh

if __package__ in (None, ""):
    ROOT = Path(__file__).resolve().parents[2]
    sys.path.insert(0, str(ROOT))
    sys.path.insert(0, str(ROOT / "src"))

from tools.farm_shaper_bridge.common import atomic_json, sha256_file, utc_now  # noqa: E402
from tools.farm_shaper_bridge.shaper_contracts import (  # noqa: E402
    SHAPER_BATCH_SCHEMA,
    SHAPER_INPUT_SCHEMA,
    artifact_record,
    load_shaper_config,
    validate_success_result,
)


PALETTE = np.asarray([
    [59, 130, 246, 255], [236, 72, 153, 255], [37, 194, 160, 255],
    [250, 173, 20, 255], [139, 92, 246, 255], [238, 85, 76, 255],
    [39, 174, 224, 255], [151, 191, 62, 255], [244, 114, 182, 255],
    [52, 211, 153, 255], [96, 165, 250, 255], [251, 191, 36, 255],
], dtype=np.uint8)


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--profile", choices=("speed", "balance", "quality"), required=True)
    return parser.parse_args(argv)


def _atomic_bytes(path: Path, payload: bytes) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
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


def install_versioned_pinhole_adapter():
    """Patch both clean and previously patched upstream loaders in memory."""

    cwd = str(Path.cwd())
    if cwd not in sys.path:
        sys.path.insert(0, cwd)
    import dataset.image_processor as image_processor
    import dataset.shaper_dataset as shaper_dataset

    original_dispatch = shaper_dataset.get_image_data_based_on_strategy

    def pinhole(pkl_sample, num_views, scale, is_rgb, strategy="cluster"):
        if strategy == "cluster":
            selected = image_processor.cluster_and_select_images(
                pkl_sample, num_views, scale, is_rgb
            )
        elif strategy == "last_n":
            selected = image_processor.last_n_view_selection(
                pkl_sample, num_views, scale, is_rgb
            )
        elif strategy == "view_angle":
            selected = image_processor.view_angle_based_strategy(
                pkl_sample, num_views, scale, is_rgb
            )
        else:
            raise ValueError(f"unknown ShapeR view strategy: {strategy}")
        if not selected:
            raise ValueError("no pinhole views survived ShapeR selection")
        if len(selected) < num_views:
            selected = selected + random.choices(selected, k=num_views - len(selected))
        images = np.stack([row[1] for row in selected], axis=0)
        masks = np.stack([row[2] for row in selected], axis=0)
        points = [np.where(masks[index] > 0) for index in range(len(masks))]
        point_masks = [
            np.stack((np.full_like(value[1], index), value[1], value[0]), axis=1)
            for index, value in enumerate(points)
        ]
        intrinsics = np.stack([np.asarray(row[3]) for row in selected], axis=0)
        if intrinsics.ndim == 3 and intrinsics.shape[1:] == (3, 3):
            intrinsics = image_processor.convert_to_4x4(intrinsics)
        elif not (intrinsics.ndim == 3 and intrinsics.shape[1:] == (4, 4)):
            raise ValueError(f"invalid pinhole intrinsic array: {intrinsics.shape}")
        camera_to_world = np.stack([row[4] for row in selected], axis=0)
        return (
            images,
            point_masks,
            intrinsics.astype(np.float32),
            camera_to_world.astype(np.float32),
        )

    def dispatch(pkl_sample, num_views, scale, is_rgb, strategy="cluster"):
        if pkl_sample.get("already_rectified_pinhole", False):
            return pinhole(pkl_sample, num_views, scale, is_rgb, strategy)
        return original_dispatch(pkl_sample, num_views, scale, is_rgb, strategy)

    shaper_dataset.get_image_data_based_on_strategy = dispatch
    # A local development checkout may already contain the branch. Replace it
    # too so behavior still comes from this signed FARM source file.
    shaper_dataset.get_image_data_pinhole_preprocessed = pinhole
    return shaper_dataset.InferenceDataset


def _safe_input_artifacts(input_dir: Path, result: Mapping[str, Any]) -> list[tuple[Path, Mapping[str, Any]]]:
    rows = list(((result.get("objects") or {}).get("rows") or []))
    artifacts: list[tuple[Path, Mapping[str, Any]]] = []
    seen: set[str] = set()
    for row in rows:
        if not isinstance(row, Mapping) or row.get("status") != "PASS":
            continue
        record = row.get("artifact")
        if not isinstance(record, Mapping):
            raise ValueError("prepared ShapeR object misses its PKL artifact record")
        relative = str(record.get("path") or "")
        if Path(relative).name != relative or relative in seen or not relative.endswith(".pkl"):
            raise ValueError("unsafe or duplicate ShapeR PKL artifact path")
        seen.add(relative)
        path = (input_dir / relative).resolve(strict=True)
        if not path.is_file() or not path.is_relative_to(input_dir):
            raise ValueError("ShapeR PKL artifact escapes its immutable input directory")
        if path.stat().st_size != int(record.get("bytes", -1)) or sha256_file(path) != record.get("sha256"):
            raise ValueError(f"ShapeR PKL artifact fingerprint mismatch: {relative}")
        artifacts.append((path, row))
    if len(artifacts) != int((result.get("objects") or {}).get("prepared", -1)) or not artifacts:
        raise ValueError("prepared ShapeR object count/artifacts mismatch")
    return artifacts


def mesh_qa(mesh: trimesh.Trimesh, points_world: np.ndarray, policy: Mapping[str, Any]) -> dict[str, Any]:
    vertices = np.asarray(mesh.vertices, dtype=np.float64)
    faces = np.asarray(mesh.faces)
    if vertices.size == 0 or faces.size == 0 or not np.isfinite(vertices).all() or not np.isfinite(faces).all():
        return {"status": "REVIEW", "reason": "empty_or_nonfinite_mesh"}
    mesh_bounds = np.asarray(mesh.bounds, dtype=np.float64)
    point_bounds = np.quantile(np.asarray(points_world, dtype=np.float64), [0.01, 0.99], axis=0)
    mesh_extent = np.maximum(mesh_bounds[1] - mesh_bounds[0], 1.0e-6)
    point_extent = np.maximum(point_bounds[1] - point_bounds[0], 1.0e-6)
    extent_ratio = mesh_extent / point_extent
    center_error = float(np.linalg.norm(np.mean(mesh_bounds, axis=0) - np.mean(point_bounds, axis=0)))
    diagonal = float(np.linalg.norm(point_extent))
    components = mesh.split(only_watertight=False)
    passed = bool(
        len(vertices) >= int(policy["minimum_mesh_vertices"])
        and len(faces) >= int(policy["minimum_mesh_faces"])
        and np.all(extent_ratio >= float(policy["minimum_extent_ratio"]))
        and np.all(extent_ratio <= float(policy["maximum_extent_ratio"]))
        and center_error <= max(
            float(policy["maximum_center_error_m"]),
            float(policy["maximum_center_error_diagonal_fraction"]) * diagonal,
        )
    )
    return {
        "status": "PASS" if passed else "REVIEW",
        "vertices": int(len(vertices)),
        "faces": int(len(faces)),
        "components": int(len(components)),
        "watertight": bool(mesh.is_watertight),
        "mesh_bounds_m": mesh_bounds.tolist(),
        "instance_q01_q99_bounds_m": point_bounds.tolist(),
        "mesh_extent_m": mesh_extent.tolist(),
        "instance_extent_m": point_extent.tolist(),
        "extent_ratio": extent_ratio.tolist(),
        "center_error_m": center_error,
    }


def _runtime_contract(config: Mapping[str, Any], config_sha: str) -> dict[str, Any]:
    runtime = config["runtime"]
    observed = {
        "image_id": os.environ.get("FARM_SHAPER_IMAGE_ID"),
        "shaper_commit": os.environ.get("FARM_SHAPER_COMMIT"),
        "shaper_tree": os.environ.get("FARM_SHAPER_TREE"),
        "farm_source_tree": os.environ.get("FARM_SOURCE_TREE") or None,
        "config_sha256": os.environ.get("FARM_SHAPER_CONFIG_SHA256"),
        "runtime_pin_sha256": os.environ.get("FARM_SHAPER_RUNTIME_PIN_SHA256"),
    }
    if observed["image_id"] != runtime["image_id"]:
        raise RuntimeError("ShapeR Docker image ID differs from the pinned config")
    if observed["shaper_commit"] != runtime["shaper_commit"]:
        raise RuntimeError("ShapeR source commit differs from the pinned config")
    if observed["config_sha256"] != config_sha:
        raise RuntimeError("ShapeR config digest differs from the host-validated config")
    if not re.fullmatch(r"[0-9a-f]{40}", str(observed["shaper_tree"] or "")):
        raise RuntimeError("missing immutable runtime identity: shaper_tree")
    if not re.fullmatch(r"[0-9a-f]{64}", str(observed["runtime_pin_sha256"] or "")):
        raise RuntimeError("missing immutable runtime identity: runtime_pin_sha256")
    if observed["farm_source_tree"] is not None and not re.fullmatch(
        r"[0-9a-f]{64}", str(observed["farm_source_tree"])
    ):
        raise RuntimeError("invalid signed FARM source tree identity")
    return observed


def execute(args: argparse.Namespace) -> dict[str, Any]:
    started_total = time.perf_counter()
    input_dir = args.input_dir.resolve(strict=True)
    output = args.output_dir.resolve()
    config, config_sha = load_shaper_config(args.config)
    input_result, input_marker_name, input_marker = validate_success_result(
        input_dir,
        result_name="result.json",
        expected_schema=SHAPER_INPUT_SCHEMA,
    )
    if input_result.get("inputs", {}).get("config_sha256") != config_sha:
        raise RuntimeError("ShapeR input config differs from the pinned inference config")
    artifacts = _safe_input_artifacts(input_dir, input_result)
    runtime_identity = _runtime_contract(config, config_sha)
    if input_result.get("source_authority", {}).get("tree_sha256") != runtime_identity["farm_source_tree"]:
        raise RuntimeError("mounted FARM source snapshot differs from ShapeR input provenance")
    profile = config["runtime"]["profiles"].get(args.profile)
    if not isinstance(profile, Mapping):
        raise ValueError(f"profile absent from pinned config: {args.profile}")
    if output.exists() and (not output.is_dir() or any(output.iterdir())):
        raise FileExistsError(f"refusing non-empty ShapeR output directory: {output}")
    output.mkdir(parents=True, exist_ok=True)

    InferenceDataset = install_versioned_pinhole_adapter()
    import omegaconf
    from model.flow_matching.shaper_denoiser import ShapeRDenoiser
    from model.text.hf_embedder import TextFeatureExtractor
    from model.vae3d.autoencoder import MichelangeloLikeAutoencoderWrapper
    from postprocessing.helper import remove_floating_geometry

    seed = int(config.get("seed", 20260819))
    device = torch.device("cuda")
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required for ShapeR inference")
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.cuda.reset_peak_memory_stats(device)
    load_started = time.perf_counter()
    state_dict = torch.load("checkpoints/019-0-bfloat16.ckpt", map_location=device, weights_only=False)
    model_config = omegaconf.OmegaConf.load("checkpoints/config.yaml")
    model = ShapeRDenoiser(model_config).to(device)
    model.convert_to_bfloat16()
    model.load_state_dict(state_dict, strict=False)
    del state_dict
    vae = MichelangeloLikeAutoencoderWrapper("checkpoints/vae-088-0-bfloat16.ckpt", device)
    text = TextFeatureExtractor(device=device).to(torch.bfloat16)
    model = torch.compile(model, fullgraph=True).eval()
    vae.model.use_udf_extraction = True
    vae.model.udf_iso = 0.375
    scales = vae.model.get_token_scales()
    scale_probability = np.zeros_like(scales)
    scale_probability[6] = 1.0
    vae.model.set_inference_scale_probabilities(scale_probability)
    token_count = int(scales[np.argmax(scale_probability)].item()) * int(profile["token_multiplier"])
    token_shape = (1, token_count, vae.get_embed_dim())
    shifted = getattr(model_config.fm_transformer, "time_sampler", "lognorm") == "flux"
    load_seconds = time.perf_counter() - load_started

    rows: list[dict[str, Any]] = []
    for order, (pkl_path, input_row) in enumerate(artifacts):
        object_id = int(input_row["object_id"])
        object_seed = seed + object_id * 1009
        torch.manual_seed(object_seed)
        torch.cuda.manual_seed_all(object_seed)
        np.random.seed(object_seed % (2**32))
        random.seed(object_seed)
        torch.cuda.reset_peak_memory_stats(device)
        started = time.perf_counter()
        try:
            dataset = InferenceDataset(
                model_config,
                paths=[str(pkl_path)],
                override_num_views=int(profile["views"]),
            )
            loader = torch.utils.data.DataLoader(
                dataset,
                batch_size=1,
                shuffle=False,
                drop_last=False,
                num_workers=0,
                collate_fn=dataset.custom_collate,
            )
            with torch.no_grad():
                batch = next(iter(loader))
                batch = InferenceDataset.move_batch_to_device(
                    batch, device, dtype=torch.bfloat16
                )
                latent = model.infer_latents(
                    batch,
                    token_shape=token_shape,
                    text_feature_extractor=text,
                    num_steps=int(profile["diffusion_steps"]),
                    use_shifted_sampling=shifted,
                )
                mesh = vae.infer_mesh_from_latents(latent)[0]
                mesh = remove_floating_geometry(mesh)
                maximum_faces = int(config["shape"]["maximum_mesh_faces"])
                if len(mesh.faces) > maximum_faces:
                    mesh = mesh.simplify_quadric_decimation(face_count=maximum_faces)
                mesh = dataset.rescale_back(batch["index"][0], mesh, True)
            # Normalize backend-specific mesh attributes through OBJ in tmpfs.
            temporary_obj = Path("/tmp") / f"farm-shaper-{os.getpid()}-{object_id}.obj"
            mesh.export(temporary_obj)
            mesh = trimesh.load(temporary_obj, force="mesh")
            temporary_obj.unlink(missing_ok=True)
            with pkl_path.open("rb") as stream:
                source = pickle.load(stream)
            world_from_model = np.linalg.inv(np.asarray(source["T_model_world"], dtype=np.float64))
            points_model = np.asarray(source["points_model"], dtype=np.float64)
            points_world = points_model @ world_from_model[:3, :3].T + world_from_model[:3, 3]
            qa = mesh_qa(mesh, points_world, config["shape"])
            glb_record = None
            if qa["status"] == "PASS":
                palette_index = object_id % len(PALETTE)
                mesh.visual.vertex_colors = np.tile(PALETTE[palette_index], (len(mesh.vertices), 1))
                glb_path = output / f"{pkl_path.stem}.glb"
                exported = mesh.export(file_type="glb", include_normals=True)
                if not isinstance(exported, (bytes, bytearray)):
                    raise RuntimeError("trimesh did not return a binary GLB payload")
                _atomic_bytes(glb_path, bytes(exported))
                glb_record = artifact_record(glb_path, root=output)
            row = {
                "name": pkl_path.stem,
                "object_id": object_id,
                "category": str(input_row.get("category") or "object"),
                "status": qa["status"],
                "profile": args.profile,
                "seed": object_seed,
                "runtime_seconds": time.perf_counter() - started,
                "peak_vram_bytes": int(torch.cuda.max_memory_allocated(device)),
                "source_pkl": artifact_record(pkl_path, root=input_dir),
                "glb": glb_record,
                "geometry_qa": qa,
            }
        except Exception as exc:
            row = {
                "name": pkl_path.stem,
                "object_id": object_id,
                "category": str(input_row.get("category") or "object"),
                "status": "FAIL",
                "profile": args.profile,
                "seed": object_seed,
                "runtime_seconds": time.perf_counter() - started,
                "source_pkl": artifact_record(pkl_path, root=input_dir),
                "error": f"{type(exc).__name__}: {exc}",
            }
        row_path = output / f"{pkl_path.stem}.json"
        atomic_json(row_path, row)
        row["report_artifact"] = artifact_record(row_path, root=output)
        rows.append(row)
        print(
            f"[{order + 1:03d}/{len(artifacts):03d}] {pkl_path.stem}: "
            f"{row['status']} {row['runtime_seconds']:.1f}s",
            flush=True,
        )
        gc.collect()
        torch.cuda.empty_cache()

    passed = sum(row["status"] == "PASS" for row in rows)
    if passed == 0:
        atomic_json(output / "_FAILED.json", {
            "schema_version": "farm.shaper-batch.failure.v1",
            "status": "failed",
            "reason": "no_mesh_passed_geometry_qa",
            "objects": rows,
        })
        raise RuntimeError("no ShapeR mesh passed geometry QA")
    release_eligible = bool(
        input_result.get("release_eligible")
        and input_marker_name == "_SUCCESS.json"
        and runtime_identity.get("farm_source_tree")
    )
    result = {
        "schema_version": SHAPER_BATCH_SCHEMA,
        "status": "PASS",
        "quality_status": "PASS" if passed == len(rows) else "WARN",
        "release_eligible": release_eligible,
        "scene_id": input_result["scene_id"],
        "created_utc": utc_now(),
        "profile": args.profile,
        "profile_contract": dict(profile),
        "inputs": {
            "shaper_inputs": str(input_dir),
            "shaper_inputs_result_sha256": sha256_file(input_dir / "result.json"),
            "farm_success_sha256": input_result["inputs"]["farm_success_sha256"],
            "farm_acceptance_sha256": input_result["inputs"]["farm_acceptance_sha256"],
            "gaussian_lift_result_sha256": input_result["inputs"]["gaussian_lift_result_sha256"],
            "verified_instance_bank_sha256": input_result["inputs"]["verified_instance_bank_sha256"],
            "source_ply_sha256": input_result["inputs"]["source_ply_sha256"],
            "config_sha256": config_sha,
        },
        "runtime": runtime_identity,
        "source_authority": input_result.get("source_authority"),
        "contracts": {
            "clean_archived_upstream_commit": True,
            "versioned_pinhole_adapter": True,
            "single_model_load": True,
            "network_disabled": os.environ.get("HF_HUB_OFFLINE") == "1",
            "lift_labels_mutated": False,
        },
        "objects": {
            "requested": len(rows),
            "passed": passed,
            "review": sum(row["status"] == "REVIEW" for row in rows),
            "failed": sum(row["status"] == "FAIL" for row in rows),
            "rows": rows,
        },
        "resources": {
            "peak_vram_bytes": max([int(row.get("peak_vram_bytes", 0)) for row in rows] or [0]),
        },
        "timing_seconds": {
            "model_load": load_seconds,
            "objects_sum": sum(float(row["runtime_seconds"]) for row in rows),
            "total": time.perf_counter() - started_total,
        },
        "limitations": [
            "ShapeR meshes are generative hypotheses, not scanned ground truth.",
            "Only meshes passing metric extent/centre geometry QA are exported for assembly.",
        ],
    }
    atomic_json(output / "result.json", result)
    marker_name = "_SUCCESS.json" if release_eligible else "_NONRELEASE_SUCCESS.json"
    atomic_json(output / marker_name, {
        "schema_version": "farm.shaper-batch.success.v1",
        "status": "success",
        "release_eligible": release_eligible,
        "scene_id": result["scene_id"],
        "result": "result.json",
        "result_sha256": sha256_file(output / "result.json"),
        "runtime_pin_sha256": runtime_identity["runtime_pin_sha256"],
    })
    print(json.dumps({
        "status": result["status"],
        "quality_status": result["quality_status"],
        "passed": passed,
        "review": result["objects"]["review"],
        "failed": result["objects"]["failed"],
        "model_load_seconds": load_seconds,
        "total_seconds": result["timing_seconds"]["total"],
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
