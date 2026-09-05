#!/usr/bin/env python3
"""Render frozen source/candidate Gaussian instance masks on heldout cameras.

This GPU process has no CLI argument for a SAM3 target, YOLOE mask, RGB image,
or observed scene depth.  It renders only the immutable Gaussian PLY/label
pairs on camera calibrations carried by a sanitised projection request.
"""

from __future__ import annotations

import argparse
import json
import os
import platform
import resource
import shutil
import sys
import tempfile
import time
from collections import defaultdict
from pathlib import Path
from types import SimpleNamespace
from typing import Any, Mapping, Sequence

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
if __package__ in (None, ""):
    sys.path.insert(0, str(ROOT))
    sys.path.insert(0, str(ROOT / "src"))

from farm_runtime.frozen_heldout_evidence import PROJECTION_SCHEMA  # noqa: E402
from farm_runtime.frozen_heldout_projection import (  # noqa: E402
    validate_renderer_inputs,
)
from farm_runtime.frozen_heldout_qc import sha256_file  # noqa: E402
from tools.farm_shaper_bridge.common import (  # noqa: E402
    Frame,
    atomic_json,
    atomic_npy,
    open_graphdeco_ply,
)
from tools.farm_shaper_bridge.gaussian_lift import (  # noqa: E402
    load_gaussians,
    reverse_render,
)


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--request", type=Path, required=True)
    parser.add_argument("--heldout-frames", type=Path, required=True)
    parser.add_argument("--rgbd-render-manifest", type=Path, required=True)
    parser.add_argument("--union-frames", type=Path, required=True)
    parser.add_argument("--source-ply", type=Path, required=True)
    parser.add_argument("--source-labels", type=Path, required=True)
    parser.add_argument("--candidate-ply", type=Path, required=True)
    parser.add_argument("--candidate-labels", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--maximum-object-channels", type=int, default=15)
    return parser.parse_args(argv)


def _artifact(path: Path, root: Path, **extra: object) -> dict[str, Any]:
    return {
        "path": os.path.relpath(path, root),
        "bytes": path.stat().st_size,
        "sha256": sha256_file(path),
        **extra,
    }


def _snapshot_renderer_sources(output: Path) -> dict[str, dict[str, Any]]:
    """Copy every local Python source that participates in projection."""

    relatives = (
        "tools/farm_shaper_bridge/frozen_heldout_projection.py",
        "tools/farm_shaper_bridge/gaussian_lift.py",
        "tools/farm_shaper_bridge/common.py",
        "src/farm_runtime/frozen_heldout_projection.py",
        "src/farm_runtime/frozen_heldout_evidence.py",
        "src/farm_runtime/frozen_heldout_qc.py",
    )
    snapshot_root = output / "renderer_sources"
    result: dict[str, dict[str, Any]] = {}
    for relative in relatives:
        source = (ROOT / relative).resolve(strict=True)
        destination = snapshot_root / relative
        destination.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(source, destination)
        result[relative] = _artifact(destination, output)
    return result


def _labels(path: Path, expected_count: int) -> np.ndarray:
    suffix = path.suffix.lower()
    if suffix == ".npy":
        value = np.load(path, allow_pickle=False)
    elif suffix == ".npz":
        with np.load(path, allow_pickle=False) as archive:
            if len(archive.files) != 1:
                raise ValueError("label NPZ must contain exactly one array")
            value = archive[archive.files[0]]
    else:
        raise ValueError("labels must be NPY or single-array NPZ")
    result = np.asarray(value).squeeze()
    if (
        result.shape != (expected_count,)
        or not np.issubdtype(result.dtype, np.integer)
        or np.any(result < -1)
    ):
        raise ValueError("labels do not align with PLY rows or contain invalid IDs")
    return result.astype(np.int64, copy=False)


def _frame(source: str, raw: Mapping[str, Any]) -> Frame:
    return Frame(
        image_id=0,
        frame_id=str(raw["frame_id"]),
        timestamp_ns=int(raw["timestamp_ns"]),
        camera=str(raw.get("camera") or "heldout"),
        sensor="heldout",
        family="heldout",
        source_image=source,
        depth_size=tuple(int(value) for value in raw["depth_size"]),
        K=np.asarray(raw["K"], dtype=np.float64),
        T_world_cam=np.asarray(raw["T_world_cam"], dtype=np.float64),
        rgb_path=Path("/unmounted/rgb"),
        depth_path=Path("/unmounted/depth"),
    )


def _safe_stem(source: str) -> str:
    return "".join(
        character if character.isalnum() or character in "_.-" else "_"
        for character in source
    )


def _render_variant(
    *,
    variant: str,
    ply_path: Path,
    labels_path: Path,
    request: Mapping[str, Any],
    frame_rows: Mapping[str, Mapping[str, Any]],
    output: Path,
    maximum_channels: int,
) -> dict[tuple[int, str], dict[str, Any]]:
    import torch

    table = open_graphdeco_ply(ply_path)
    labels = _labels(labels_path, table.count)
    object_ids = sorted({int(row["object_id"]) for row in request["observations"]})
    indices_by_id = {
        object_id: np.flatnonzero(labels == object_id).astype(np.int64, copy=False)
        for object_id in object_ids
    }
    config = request["renderer_config"]
    run = SimpleNamespace(
        meters_per_scene_unit=float(request["meters_per_scene_unit"]),
        rgbd_config={
            "depth_min_m": float(config["depth_min_m"]),
            "depth_max_m": float(config["depth_max_m"]),
            "radius_clip": float(config["radius_clip"]),
        },
    )
    grouped: dict[str, list[int]] = defaultdict(list)
    for row in request["observations"]:
        grouped[str(row["source_image"])].append(int(row["object_id"]))
    gaussians = load_gaussians(table, run.meters_per_scene_unit)
    threshold = float(config["object_alpha_threshold"])
    root = output / variant
    mask_root = root / "masks"
    depth_root = root / "depth"
    mask_root.mkdir(parents=True)
    depth_root.mkdir(parents=True)
    rendered_rows: dict[tuple[int, str], dict[str, Any]] = {}
    try:
        for source in sorted(grouped):
            frame = _frame(source, frame_rows[source])
            wanted = sorted(set(grouped[source]))
            for start in range(0, len(wanted), maximum_channels):
                batch = wanted[start : start + maximum_channels]
                mass, _share, depth, _scene_depth, _alpha = reverse_render(
                    gaussians, run, frame, batch, indices_by_id
                )
                for column, object_id in enumerate(batch):
                    mask = np.asarray(mass[..., column] >= threshold, dtype=bool)
                    object_depth = np.asarray(depth[..., column], dtype=np.float32)
                    valid = mask & np.isfinite(object_depth) & (object_depth > 0.0)
                    object_depth = np.where(valid, object_depth, 0.0).astype(np.float32)
                    stem = f"object_{object_id:06d}__{_safe_stem(source)}"
                    mask_path = mask_root / f"{stem}.npy"
                    depth_path = depth_root / f"{stem}.npy"
                    atomic_npy(mask_path, mask)
                    atomic_npy(depth_path, object_depth)
                    rendered_rows[(object_id, source)] = {
                        "mask": _artifact(mask_path, output),
                        "depth": _artifact(depth_path, output),
                        "mask_pixels": int(np.count_nonzero(mask)),
                        "depth_pixels": int(np.count_nonzero(valid)),
                        "soft_mass_max": float(np.max(mass[..., column])),
                        "labelled_gaussians": int(indices_by_id[object_id].size),
                    }
                del mass, depth, _share, _scene_depth, _alpha
            torch.cuda.empty_cache()
    finally:
        del gaussians
        torch.cuda.empty_cache()
    return rendered_rows


def _runtime() -> dict[str, Any]:
    from importlib.metadata import version

    import gsplat
    import torch

    return {
        "python": platform.python_version(),
        "torch": torch.__version__,
        "cuda_runtime": torch.version.cuda,
        "gsplat": getattr(gsplat, "__version__", None) or version("gsplat"),
        "gpu": torch.cuda.get_device_name(0),
        "container_image": os.environ.get("FARM_PROJECTION_IMAGE"),
        "container_image_id": os.environ.get("FARM_PROJECTION_IMAGE_ID"),
        "source_commit": os.environ.get("FARM_PROJECTION_SOURCE_COMMIT"),
        "source_dirty": os.environ.get("FARM_PROJECTION_SOURCE_DIRTY") == "true",
        "runtime_image_drift_nonrelease": os.environ.get("FARM_PROJECTION_IMAGE_DRIFT")
        == "true",
    }


def run(args: argparse.Namespace) -> Path:
    import torch

    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required for exact gsplat heldout projection")
    if args.maximum_object_channels < 1 or 2 * args.maximum_object_channels + 1 > 32:
        raise ValueError("maximum-object-channels must satisfy 1 <= K and 2*K+1 <= 32")
    destination = args.output_dir.expanduser().resolve()
    if destination.exists():
        raise FileExistsError(f"refusing to overwrite output: {destination}")
    destination.parent.mkdir(parents=True, exist_ok=True)
    request, frames, input_hashes = validate_renderer_inputs(
        args.request,
        args.heldout_frames,
        args.rgbd_render_manifest,
        args.union_frames,
        args.source_ply,
        args.source_labels,
        args.candidate_ply,
        args.candidate_labels,
        verify_rgbd_transitive_artifacts=False,
    )
    temporary = Path(
        tempfile.mkdtemp(
            prefix=f".{destination.name}.", suffix=".tmp", dir=destination.parent
        )
    )
    try:
        request_copy = temporary / "projection_request.json"
        implementation_copy = temporary / "renderer_implementation.py"
        shutil.copyfile(args.request.resolve(), request_copy)
        shutil.copyfile(Path(__file__).resolve(), implementation_copy)
        renderer_sources = _snapshot_renderer_sources(temporary)
        started = time.perf_counter()
        torch.cuda.reset_peak_memory_stats()
        source = _render_variant(
            variant="source",
            ply_path=args.source_ply.resolve(),
            labels_path=args.source_labels.resolve(),
            request=request,
            frame_rows=frames,
            output=temporary,
            maximum_channels=args.maximum_object_channels,
        )
        candidate = _render_variant(
            variant="candidate",
            ply_path=args.candidate_ply.resolve(),
            labels_path=args.candidate_labels.resolve(),
            request=request,
            frame_rows=frames,
            output=temporary,
            maximum_channels=args.maximum_object_channels,
        )
        torch.cuda.synchronize()
        elapsed = time.perf_counter() - started
        observations: list[dict[str, Any]] = []
        for row in request["observations"]:
            key = (int(row["object_id"]), str(row["source_image"]))
            source_row = source[key]
            candidate_row = candidate[key]
            observations.append(
                {
                    **row,
                    "prediction_mask": candidate_row["mask"],
                    "prediction_depth": candidate_row["depth"],
                    "source_prediction_mask": source_row["mask"],
                    "source_prediction_depth": source_row["depth"],
                    "diagnostics": {
                        "source": {
                            key: value
                            for key, value in source_row.items()
                            if key not in {"mask", "depth"}
                        },
                        "candidate": {
                            key: value
                            for key, value in candidate_row.items()
                            if key not in {"mask", "depth"}
                        },
                    },
                }
            )
        manifest = {
            "schema": PROJECTION_SCHEMA,
            "status": "frozen",
            "prediction_kind": "post_lift_gaussian_projection",
            "depth_unit": "meters",
            "fold_manifest_sha256": request["fold_manifest_sha256"],
            "candidate_manifest_sha256": request["candidate_manifest_sha256"],
            "heldout_frames_sha256": request["heldout_frames_sha256"],
            "heldout_reference_result_sha256": request[
                "heldout_reference_result_sha256"
            ],
            "rgbd_render_manifest_sha256": request["rgbd_render_manifest_sha256"],
            "union_frames_sha256": request["union_frames_sha256"],
            "source_ply_sha256": request["source_ply_sha256"],
            "source_labels_sha256": request["source_labels_sha256"],
            "candidate_ply_sha256": request["candidate_ply_sha256"],
            "candidate_labels_sha256": request["candidate_labels_sha256"],
            "rgbd_transitive_artifact_contract": request.get(
                "rgbd_transitive_artifact_contract"
            ),
            "projection_request_sha256": input_hashes["request"],
            "projection_request": _artifact(request_copy, temporary),
            "renderer_implementation": _artifact(implementation_copy, temporary),
            "renderer_sources": renderer_sources,
            "renderer_config": request["renderer_config"],
            "renderer_contract": {
                "read_only": True,
                "candidate_frozen_before_projection": True,
                "candidate_ply_mutated": False,
                "candidate_labels_mutated": False,
                "source_ply_mutated": False,
                "source_labels_mutated": False,
                "heldout_updates_candidate": False,
                "renders_only_precommitted_heldout_views": True,
                "prediction_generated_without_target_masks": True,
                "targets_not_consumed_by_renderer": True,
                "target_artifacts_mounted": False,
                "rgbd_transitive_artifacts_mounted": False,
                "rgbd_transitive_artifacts_opened_by_renderer": False,
                "rgbd_transitive_declarations_request_bound": True,
                "prediction_depth_in_meters": True,
                "source_and_candidate_rendered_from_respective_frozen_labels": True,
                "geometry_or_appearance_optimization_performed": False,
            },
            "runtime": _runtime(),
            "resources": {
                "cuda_peak_allocated_bytes": int(torch.cuda.max_memory_allocated()),
                "cuda_peak_reserved_bytes": int(torch.cuda.max_memory_reserved()),
                "max_rss_kib": int(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss),
            },
            "timing_seconds": {"total": elapsed},
            "observations": observations,
        }
        changed = {
            name: {
                "before": digest,
                "after": sha256_file(
                    Path(
                        getattr(args, name) if name != "request" else args.request
                    ).resolve()
                ),
            }
            for name, digest in input_hashes.items()
            if name
            in {
                "request",
                "heldout_frames",
                "rgbd_render_manifest",
                "union_frames",
                "source_ply",
                "source_labels",
                "candidate_ply",
                "candidate_labels",
            }
            and sha256_file(
                Path(
                    getattr(args, name) if name != "request" else args.request
                ).resolve()
            )
            != digest
        }
        if changed:
            raise RuntimeError(
                f"frozen renderer inputs changed during projection: {sorted(changed)}"
            )
        changed_sources = [
            relative
            for relative, spec in renderer_sources.items()
            if sha256_file((ROOT / relative).resolve(strict=True)) != spec["sha256"]
        ]
        if changed_sources:
            raise RuntimeError(
                "renderer source changed during projection: "
                f"{sorted(changed_sources)}"
            )
        manifest_path = temporary / "manifest.json"
        atomic_json(manifest_path, manifest)
        atomic_json(
            temporary / "_SUCCESS.json",
            {
                "schema": "farm.frozen-heldout-gaussian-projection-success.v1",
                "status": "success",
                "manifest": "manifest.json",
                "manifest_sha256": sha256_file(manifest_path),
                "observations": len(observations),
            },
        )
        os.replace(temporary, destination)
    except BaseException:
        shutil.rmtree(temporary, ignore_errors=True)
        raise
    return destination / "manifest.json"


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    manifest = run(args)
    payload = json.loads(manifest.read_text(encoding="utf-8"))
    print(
        json.dumps(
            {
                "status": payload["status"],
                "manifest": str(manifest),
                "observations": len(payload["observations"]),
                "target_artifacts_mounted": False,
                "timing_seconds": payload["timing_seconds"]["total"],
                "cuda_peak_allocated_bytes": payload["resources"][
                    "cuda_peak_allocated_bytes"
                ],
            },
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
