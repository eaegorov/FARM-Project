#!/usr/bin/env python3
"""Run one camera-pure SAM3 concept/video episode with short identity masks.

The output is deliberately a proposal artifact.  Numeric IDs are local to the
episode; global 3D association and heldout checks are required before FARM may
publish an object or change an OBB.
"""

from __future__ import annotations

# Allow direct execution from the source checkout.
import sys as _entry_sys
from pathlib import Path as _EntryPath
_entry_root = _EntryPath(__file__).resolve().parents[2]
_entry_sys.path[:0] = [str(_entry_root), str(_entry_root / "src")]

import argparse
import hashlib
import importlib.metadata
import json
import resource
import time
from pathlib import Path
from typing import Any, Iterable


import numpy as np


IMAGE_SUFFIXES = {".jpg", ".jpeg", ".png"}
SCHEMA = "farm.sam3-concept-episode.v1"
SAM3_KERNELS_PACKAGE_VERSION = "0.16.1"
SAM3_CV_UTILS_REPOSITORY = "kernels-community/cv-utils"
SAM3_CV_UTILS_REVISION = "8bc5d583ad41502e37647763437b4881b87e5110"


def _load_pinned_cv_utils_kernel(
    *,
    package_version_resolver: Any | None = None,
    kernel_loader: Any | None = None,
    modeling_module: Any | None = None,
) -> dict[str, Any]:
    """Load the exact official SAM3 mask-QC kernel from local cache.

    Transformers otherwise resolves the moving ``v1`` Hub ref lazily and
    silently falls back to weaker post-processing when loading fails. FARM
    pins both the Python resolver and kernel commit, fails closed, and assigns
    the loaded module to the official Transformers integration point.
    """

    resolver = package_version_resolver or importlib.metadata.version
    installed = str(resolver("kernels"))
    if installed != SAM3_KERNELS_PACKAGE_VERSION:
        raise RuntimeError(
            "SAM3 requires kernels=="
            f"{SAM3_KERNELS_PACKAGE_VERSION}; found {installed}"
        )
    if kernel_loader is None:
        from kernels import load_kernel

        kernel_loader = load_kernel
    if modeling_module is None:
        from transformers.models.sam3_video import modeling_sam3_video

        modeling_module = modeling_sam3_video
    kernel = kernel_loader(
        SAM3_CV_UTILS_REPOSITORY,
        lockfile=None,
        revision=SAM3_CV_UTILS_REVISION,
        backend="cuda",
    )
    required = ("generic_nms", "cc_2d")
    missing = [name for name in required if not callable(getattr(kernel, name, None))]
    if missing:
        raise RuntimeError("pinned cv-utils kernel lacks: " + ", ".join(missing))
    modeling_module.cv_utils_kernel = kernel
    return {
        "resolver_package": "kernels",
        "resolver_version": installed,
        "repository": SAM3_CV_UTILS_REPOSITORY,
        "revision": SAM3_CV_UTILS_REVISION,
        "backend": "cuda",
        "required_functions": list(required),
        "load_mode": "exact-revision-local-cache-fail-closed",
    }


def _sha256(path: Path, chunk_size: int = 8 * 1024 * 1024) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while True:
            chunk = handle.read(chunk_size)
            if not chunk:
                break
            digest.update(chunk)
    return digest.hexdigest()


def _read_lines(path: Path, *, label: str) -> list[str]:
    values = [line.strip() for line in path.read_text(encoding="utf-8").splitlines()]
    values = [value for value in values if value and not value.startswith("#")]
    if not values:
        raise ValueError(f"{label} is empty")
    if len(values) != len(set(value.casefold() for value in values)):
        raise ValueError(f"{label} contains duplicate values")
    return values


def _validate_images(
    image_dir: Path,
    *,
    expected_names: Iterable[str] | None,
) -> list[Path]:
    if not image_dir.is_dir():
        raise ValueError(f"image directory does not exist: {image_dir}")
    entries = sorted(image_dir.iterdir(), key=lambda value: value.name)
    if not entries:
        raise ValueError("image directory is empty")
    invalid = [
        value.name
        for value in entries
        if not value.is_file() or value.suffix.lower() not in IMAGE_SUFFIXES
    ]
    if invalid:
        raise ValueError("episode contains non-image entries: " + ", ".join(invalid[:8]))
    if expected_names is not None:
        expected = list(expected_names)
        if any(Path(value).name != value for value in expected):
            raise ValueError("frame list must contain basenames")
        if [value.name for value in entries] != sorted(expected):
            raise ValueError("materialized episode does not match frame list")
    return entries


def _compose_short_id_mask(
    *,
    object_ids: np.ndarray,
    scores: np.ndarray,
    masks: np.ndarray,
    local_id_by_object_id: dict[int, int],
    minimum_score: float,
    maximum_local_id: int = 254,
) -> tuple[np.ndarray, dict[str, Any]]:
    """Resolve overlapping SAM3 proposals by score into an 8-bit ID map."""

    object_ids = np.asarray(object_ids, dtype=np.int64).reshape(-1)
    scores = np.asarray(scores, dtype=np.float32).reshape(-1)
    masks = np.asarray(masks, dtype=bool)
    if masks.ndim != 3 or masks.shape[0] != len(object_ids) or len(scores) != len(object_ids):
        raise ValueError("SAM3 object IDs, scores and masks have inconsistent shapes")
    height, width = masks.shape[-2:]
    keep = np.flatnonzero(np.isfinite(scores) & (scores >= float(minimum_score)))
    canvas = np.zeros((height, width), dtype=np.uint8)
    if keep.size == 0:
        return canvas, {
            "input_objects": int(len(object_ids)),
            "accepted_objects": 0,
            "claimed_pixels": 0,
            "overlap_pixels_before_resolution": 0,
            "overlap_fraction_before_resolution": 0.0,
        }

    accepted_masks = masks[keep]
    accepted_scores = scores[keep]
    claim_count = accepted_masks.sum(axis=0)
    union = claim_count > 0
    overlap = claim_count > 1
    score_volume = np.where(
        accepted_masks,
        accepted_scores[:, None, None],
        np.float32(-np.inf),
    )
    winner = np.argmax(score_volume, axis=0)
    for local_index, source_index in enumerate(keep.tolist()):
        object_id = int(object_ids[source_index])
        if object_id not in local_id_by_object_id:
            next_id = len(local_id_by_object_id) + 1
            if next_id > int(maximum_local_id):
                raise ValueError("SAM3 episode exceeds the 8-bit local identity budget")
            local_id_by_object_id[object_id] = next_id
        selected = union & (winner == local_index)
        canvas[selected] = np.uint8(local_id_by_object_id[object_id])
    union_pixels = int(union.sum())
    overlap_pixels = int(overlap.sum())
    return canvas, {
        "input_objects": int(len(object_ids)),
        "accepted_objects": int(len(keep)),
        "claimed_pixels": union_pixels,
        "overlap_pixels_before_resolution": overlap_pixels,
        "overlap_fraction_before_resolution": (
            float(overlap_pixels / union_pixels) if union_pixels else 0.0
        ),
    }


def _phase_start(torch_module: Any) -> tuple[float, int, int]:
    torch_module.cuda.synchronize()
    torch_module.cuda.reset_peak_memory_stats()
    return (
        time.perf_counter(),
        int(torch_module.cuda.memory_allocated()),
        int(torch_module.cuda.memory_reserved()),
    )


def _phase_end(torch_module: Any, started: tuple[float, int, int]) -> dict[str, float]:
    torch_module.cuda.synchronize()
    start_time, start_allocated, start_reserved = started
    divisor = 1024.0 * 1024.0
    return {
        "seconds": float(time.perf_counter() - start_time),
        "start_allocated_mib": float(start_allocated / divisor),
        "start_reserved_mib": float(start_reserved / divisor),
        "peak_allocated_mib": float(torch_module.cuda.max_memory_allocated() / divisor),
        "peak_reserved_mib": float(torch_module.cuda.max_memory_reserved() / divisor),
    }


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", required=True, type=Path)
    parser.add_argument("--image-dir", required=True, type=Path)
    parser.add_argument("--frame-list", type=Path)
    parser.add_argument("--prompt-file", required=True, type=Path)
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument("--measurement-json", type=Path)
    parser.add_argument("--run-id", required=True)
    parser.add_argument("--planner-manifest-sha256")
    parser.add_argument("--minimum-score", type=float, default=0.45)
    parser.add_argument("--maximum-objects", type=int, default=254)
    parser.add_argument("--vision-cache-frames", type=int, default=1)
    parser.add_argument(
        "--inference-state-device", choices=("cuda", "cpu"), default="cuda"
    )
    parser.add_argument(
        "--video-storage-device", choices=("cuda", "cpu"), default="cpu"
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    import torch
    from PIL import Image
    from transformers import Sam3VideoModel, Sam3VideoProcessor

    args = _parser().parse_args(argv)
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required for the SAM3 benchmark")
    if not 0.0 <= float(args.minimum_score) <= 1.0:
        raise ValueError("minimum score must be in [0, 1]")
    if not 1 <= int(args.maximum_objects) <= 254:
        raise ValueError("maximum objects must be in [1, 254]")
    if int(args.vision_cache_frames) < 1:
        raise ValueError("vision cache size must be positive")

    model_path = args.model.expanduser().resolve(strict=True)
    image_dir = args.image_dir.expanduser().resolve(strict=True)
    prompt_file = args.prompt_file.expanduser().resolve(strict=True)
    expected = _read_lines(args.frame_list.expanduser().resolve(strict=True), label="frame list") if args.frame_list else None
    prompts = _read_lines(prompt_file, label="prompt file")
    image_paths = _validate_images(image_dir, expected_names=expected)
    output_dir = args.output_dir.expanduser().resolve()
    if output_dir.exists() and any(output_dir.iterdir()):
        raise ValueError(f"output directory is not empty: {output_dir}")
    output_dir.mkdir(parents=True, exist_ok=True)
    mask_dir = output_dir / "Annotations"
    mask_dir.mkdir()
    measurement_path = (
        args.measurement_json.expanduser().resolve()
        if args.measurement_json
        else output_dir / "measurement.json"
    )

    torch.manual_seed(42)
    torch.cuda.manual_seed_all(42)
    torch.backends.cudnn.benchmark = False
    wall_started = time.perf_counter()
    stages: dict[str, dict[str, float]] = {}

    started = _phase_start(torch)
    kernel_runtime = _load_pinned_cv_utils_kernel()
    processor = Sam3VideoProcessor.from_pretrained(
        str(model_path), local_files_only=True
    )
    model = Sam3VideoModel.from_pretrained(
        str(model_path), local_files_only=True, dtype=torch.bfloat16
    ).eval().to("cuda")
    stages["sam3_model_load"] = _phase_end(torch, started)

    started = _phase_start(torch)
    frames = []
    for image_path in image_paths:
        with Image.open(image_path) as image:
            frames.append(image.convert("RGB").copy())
    session = processor.init_video_session(
        video=[frames],
        inference_device="cuda",
        inference_state_device=args.inference_state_device,
        processing_device="cpu",
        video_storage_device=args.video_storage_device,
        max_vision_features_cache_size=int(args.vision_cache_frames),
        dtype=torch.bfloat16,
    )
    processor.add_text_prompt(session, prompts)
    stages["video_preprocess_and_prompt_encoding"] = _phase_end(torch, started)

    frame_rows: list[dict[str, Any]] = []
    local_id_by_object_id: dict[int, int] = {}
    prompt_by_object_id: dict[int, str] = {}
    started = _phase_start(torch)
    with torch.inference_mode():
        for frame_index, image_path in enumerate(image_paths):
            model_output = model(session, frame_idx=frame_index)
            output = processor.postprocess_outputs(session, model_output)
            object_ids = output["object_ids"].detach().cpu().numpy()
            scores = output["scores"].detach().float().cpu().numpy()
            masks = output["masks"].detach().cpu().numpy()
            for prompt, ids in output["prompt_to_obj_ids"].items():
                for object_id in ids:
                    prompt_by_object_id[int(object_id)] = str(prompt)
            index_mask, composition = _compose_short_id_mask(
                object_ids=object_ids,
                scores=scores,
                masks=masks,
                local_id_by_object_id=local_id_by_object_id,
                minimum_score=float(args.minimum_score),
                maximum_local_id=int(args.maximum_objects),
            )
            Image.fromarray(index_mask, mode="L").save(mask_dir / f"{image_path.stem}.png")
            frame_rows.append(
                {
                    "frame_index": int(frame_index),
                    "frame_name": image_path.name,
                    **composition,
                    "objects": [
                        {
                            "sam3_object_id": int(object_id),
                            "local_id": local_id_by_object_id.get(int(object_id)),
                            "prompt": prompt_by_object_id.get(int(object_id)),
                            "score": float(score),
                            "area_px": int(mask.sum()),
                        }
                        for object_id, score, mask in zip(object_ids, scores, masks)
                    ],
                }
            )
            del model_output, output, masks
    stages["concept_detection_and_tracking"] = _phase_end(torch, started)

    inference_seconds = stages["concept_detection_and_tracking"]["seconds"]
    model_file = model_path / "model.safetensors"
    config_file = model_path / "config.json"
    checkpoint = {
        "directory": str(model_path),
        "model_bytes": model_file.stat().st_size if model_file.is_file() else None,
        "model_sha256": _sha256(model_file) if model_file.is_file() else None,
        "config_sha256": _sha256(config_file) if config_file.is_file() else None,
    }
    identity_rows = [
        {
            "sam3_object_id": int(object_id),
            "local_id": int(local_id),
            "prompt": prompt_by_object_id.get(int(object_id)),
        }
        for object_id, local_id in sorted(local_id_by_object_id.items(), key=lambda item: item[1])
    ]
    peaks_allocated = [stage["peak_allocated_mib"] for stage in stages.values()]
    peaks_reserved = [stage["peak_reserved_mib"] for stage in stages.values()]
    report = {
        "schema": SCHEMA,
        "run_id": str(args.run_id),
        "status": "pass",
        "implementation": {
            "model": "SAM3 concept prompting and video tracking",
            "output_semantics": "episode-local short identities",
            "cv_utils_kernel": kernel_runtime,
        },
        "input": {
            "image_directory": str(image_dir),
            "frame_count": len(image_paths),
            "first_frame": image_paths[0].name,
            "last_frame": image_paths[-1].name,
            "ordered_frame_names_sha256": hashlib.sha256(
                ("\n".join(path.name for path in image_paths) + "\n").encode("utf-8")
            ).hexdigest(),
            "planner_manifest_sha256": args.planner_manifest_sha256,
            "prompts": prompts,
            "prompt_file_sha256": _sha256(prompt_file),
        },
        "checkpoint": checkpoint,
        "configuration": {
            "dtype": "bfloat16",
            "minimum_score": float(args.minimum_score),
            "maximum_objects": int(args.maximum_objects),
            "inference_state_device": args.inference_state_device,
            "video_storage_device": args.video_storage_device,
            "vision_cache_frames": int(args.vision_cache_frames),
            "overlap_resolution": "highest_object_score",
        },
        "measurement": {
            "wall_seconds": float(time.perf_counter() - wall_started),
            "inference_fps": float(len(image_paths) / inference_seconds),
            "peak_cuda_allocated_mib": max(peaks_allocated),
            "peak_cuda_reserved_mib": max(peaks_reserved),
            "peak_cpu_rss_mib": float(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024.0),
            "stages": stages,
        },
        "device": {
            "name": torch.cuda.get_device_name(0),
            "cuda_runtime": torch.version.cuda,
            "torch": torch.__version__,
        },
        "output": {
            "directory": str(output_dir),
            "mask_count": len(list(mask_dir.glob("*.png"))),
            "local_identity_count": len(identity_rows),
            "identities": identity_rows,
            "frames": frame_rows,
        },
        "limitations": [
            "Prompt-conditioned recall depends on the Qwen-derived scene vocabulary.",
            "IDs are episode-local and require geometric association across streams.",
            "Masks and boxes are proposals; depth, exclusivity and heldout silhouette gates remain mandatory.",
        ],
    }
    measurement_path.parent.mkdir(parents=True, exist_ok=True)
    measurement_path.write_text(
        json.dumps(report, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(report["measurement"], indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
