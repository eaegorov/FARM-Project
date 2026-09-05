#!/usr/bin/env python3
"""Run one camera-pure SAM1+DEVA episode with auditable measurements.

This is a thin, measured adaptation of DEVA's ``demo/demo_automatic.py``.
It intentionally runs the automatic segmentation only once: the resulting
short-ID PNGs are the training masks, while coloured previews can be derived
from those PNGs without a second model invocation.

The script expects the DEVA and Segment Anything repositories on PYTHONPATH.
It does not download checkpoints and refuses mixed/non-image input folders.
"""

from __future__ import annotations

# Allow direct execution from the source checkout.
import sys as _entry_sys
from pathlib import Path as _EntryPath
_entry_root = _EntryPath(__file__).resolve().parents[2]
_entry_sys.path[:0] = [str(_entry_root), str(_entry_root / "src")]

import argparse
import hashlib
import json
import os
import resource
import sys
import time
from pathlib import Path
from typing import Any, Iterable


IMAGE_SUFFIXES = {".jpg", ".jpeg", ".png"}
SCHEMA = "farm.deva-automatic-episode.v1"
TORCHVISION_BACKBONES = {
    "resnet50-19c8e357.pth": "19c8e3572231adff6824a2da93fd67b5986919a2e65f8b6007eab4edee220097",
    "resnet18-5c106cde.pth": "5c106cde386e87d4033832f2996f5493238eda96ccf559d1d62760c4de0613f8",
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


def _read_expected_frames(path: Path | None) -> list[str] | None:
    if path is None:
        return None
    values = [line.strip() for line in path.read_text(encoding="utf-8").splitlines()]
    values = [value for value in values if value and not value.startswith("#")]
    if not values:
        raise ValueError("frame list is empty")
    if any(Path(value).name != value for value in values):
        raise ValueError("frame list must contain basenames, not paths")
    if len(values) != len(set(values)):
        raise ValueError("frame list contains duplicate names")
    return values


def _validate_image_directory(
    image_dir: Path,
    *,
    expected_frames: Iterable[str] | None,
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
        raise ValueError(
            "image directory must contain only image files/symlinks; invalid: "
            + ", ".join(invalid[:8])
        )
    names = [value.name for value in entries]
    expected = None if expected_frames is None else list(expected_frames)
    if expected is not None and names != sorted(expected):
        missing = sorted(set(expected).difference(names))
        unexpected = sorted(set(names).difference(expected))
        raise ValueError(
            "materialized episode does not match frame list; "
            f"missing={missing[:8]}, unexpected={unexpected[:8]}"
        )
    return entries


def _prepare_output(path: Path) -> None:
    if path.exists() and any(path.iterdir()):
        raise ValueError(f"output directory is not empty: {path}")
    path.mkdir(parents=True, exist_ok=True)


def _phase_start(torch_module: Any) -> tuple[float, int, int]:
    torch_module.cuda.synchronize()
    torch_module.cuda.reset_peak_memory_stats()
    return (
        time.perf_counter(),
        int(torch_module.cuda.memory_allocated()),
        int(torch_module.cuda.memory_reserved()),
    )


def _phase_end(
    torch_module: Any,
    started: tuple[float, int, int],
) -> dict[str, float]:
    torch_module.cuda.synchronize()
    start_time, start_allocated, start_reserved = started
    divisor = 1024.0 * 1024.0
    return {
        "seconds": float(time.perf_counter() - start_time),
        "start_allocated_mib": float(start_allocated / divisor),
        "start_reserved_mib": float(start_reserved / divisor),
        "peak_allocated_mib": float(
            torch_module.cuda.max_memory_allocated() / divisor
        ),
        "peak_reserved_mib": float(
            torch_module.cuda.max_memory_reserved() / divisor
        ),
    }


def _summarize_predictions(path: Path) -> dict[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    annotations = payload.get("annotations")
    if not isinstance(annotations, list):
        raise ValueError("DEVA pred.json has no annotations list")
    identities: set[int] = set()
    observations = 0
    areas: list[int] = []
    for annotation in annotations:
        for segment in annotation.get("segments_info", []):
            identities.add(int(segment["id"]))
            observations += 1
            areas.append(int(segment.get("area", 0)))
    return {
        "annotated_frames": len(annotations),
        "unique_local_identities": len(identities),
        "identity_observations": observations,
        "minimum_observation_area_px": min(areas) if areas else 0,
        "maximum_observation_area_px": max(areas) if areas else 0,
    }


def _build_parser() -> argparse.ArgumentParser:
    # Importing these modules is intentionally deferred so CPU-only tests can
    # exercise input validation without a CUDA/DEVA environment.
    from deva.ext.ext_eval_args import add_auto_default_args, add_ext_eval_args
    from deva.inference.eval_args import add_common_eval_args

    parser = argparse.ArgumentParser(
        description="Measured, one-pass SAM1+DEVA automatic episode runner"
    )
    add_common_eval_args(parser)
    add_ext_eval_args(parser)
    add_auto_default_args(parser)
    parser.add_argument("--frame-list", type=Path)
    parser.add_argument("--measurement-json", type=Path)
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument(
        "--torch-hub-dir",
        required=True,
        type=Path,
        help="Pre-populated torch hub directory; network downloads are forbidden.",
    )
    parser.add_argument(
        "--no-suppress-small-objects",
        dest="suppress_small_objects",
        action="store_false",
        help="Ablation only: preserve small SAM proposals.",
    )
    parser.add_argument(
        "--no-amp",
        dest="amp",
        action="store_false",
        help="Ablation only: disable mixed precision.",
    )
    parser.add_argument("--run-id", required=True)
    parser.add_argument("--planner-manifest-sha256")
    parser.set_defaults(
        amp=True,
        chunk_size=4,
        detection_every=5,
        num_voting_frames=3,
        size=480,
        max_num_objects=200,
        temporal_setting="semionline",
        use_short_id=True,
        suppress_small_objects=True,
        SAM_PRED_IOU_THRESHOLD=0.70,
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    import numpy as np
    import torch
    from torch.utils.data import DataLoader

    from deva.ext.automatic_processor import process_frame_automatic
    from deva.ext.automatic_sam import get_sam_model
    from deva.inference.data.simple_video_reader import SimpleVideoReader, no_collate
    from deva.inference.demo_utils import flush_buffer
    from deva.inference.inference_core import DEVAInferenceCore
    from deva.inference.result_utils import ResultSaver
    from deva.model.network import DEVA

    args = _build_parser().parse_args(argv)
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required for the DEVA benchmark")
    if args.temporal_setting.lower() != "semionline":
        raise ValueError("only the audited semionline mode is supported")
    if not args.use_short_id:
        raise ValueError("use_short_id is required for Gaussian Grouping masks")
    if int(args.num_workers) < 0:
        raise ValueError("num_workers must be non-negative")
    if int(args.max_num_objects) > 254:
        raise ValueError("8-bit Gaussian Grouping masks support at most 254 objects")

    image_dir = Path(args.img_path).resolve()
    output_dir = Path(args.output).resolve() if args.output else None
    if output_dir is None:
        raise ValueError("--output is required")
    deva_checkpoint = Path(args.model).resolve() if args.model else None
    sam_checkpoint = Path(args.SAM_CHECKPOINT_PATH).resolve()
    if deva_checkpoint is None or not deva_checkpoint.is_file():
        raise ValueError(f"DEVA checkpoint does not exist: {deva_checkpoint}")
    if not sam_checkpoint.is_file():
        raise ValueError(f"SAM checkpoint does not exist: {sam_checkpoint}")
    torch_hub_dir = args.torch_hub_dir.expanduser().resolve(strict=True)
    backbone_rows = []
    for filename, expected_sha256 in TORCHVISION_BACKBONES.items():
        backbone = torch_hub_dir / "checkpoints" / filename
        if not backbone.is_file():
            raise ValueError(
                f"required offline torchvision backbone does not exist: {backbone}"
            )
        actual_sha256 = _sha256(backbone)
        if actual_sha256 != expected_sha256:
            raise ValueError(f"torchvision backbone checksum mismatch: {backbone}")
        backbone_rows.append(
            {
                "path": str(backbone),
                "bytes": backbone.stat().st_size,
                "sha256": actual_sha256,
            }
        )
    torch.hub.set_dir(str(torch_hub_dir))

    expected = _read_expected_frames(args.frame_list)
    image_paths = _validate_image_directory(
        image_dir,
        expected_frames=expected,
    )
    _prepare_output(output_dir)
    measurement_path = (
        args.measurement_json.resolve()
        if args.measurement_json is not None
        else output_dir / "measurement.json"
    )

    torch.autograd.set_grad_enabled(False)
    torch.manual_seed(42)
    torch.cuda.manual_seed_all(42)
    np.random.seed(42)
    torch.backends.cudnn.benchmark = False

    cfg = vars(args).copy()
    cfg["enable_long_term"] = not bool(cfg["disable_long_term"])
    cfg["temporal_setting"] = str(cfg["temporal_setting"]).lower()
    cfg["img_path"] = str(image_dir)
    cfg["output"] = str(output_dir)
    cfg["model"] = str(deva_checkpoint)
    cfg["SAM_CHECKPOINT_PATH"] = str(sam_checkpoint)

    wall_started = time.perf_counter()
    stages: dict[str, dict[str, float]] = {}

    started = _phase_start(torch)
    deva_model = DEVA(cfg).cuda().eval()
    model_weights = torch.load(
        str(deva_checkpoint), map_location="cuda", weights_only=True
    )
    deva_model.load_weights(model_weights)
    del model_weights
    stages["deva_model_load"] = _phase_end(torch, started)

    started = _phase_start(torch)
    sam_model = get_sam_model(cfg, "cuda")
    stages["sam1_model_load"] = _phase_end(torch, started)

    reader = SimpleVideoReader(str(image_dir))
    loader = DataLoader(
        reader,
        batch_size=None,
        collate_fn=no_collate,
        num_workers=min(int(args.num_workers), len(reader)),
    )
    cfg["enable_long_term_count_usage"] = (
        cfg["enable_long_term"]
        and (
            len(loader)
            / (cfg["max_mid_term_frames"] - cfg["min_mid_term_frames"])
            * cfg["num_prototypes"]
        )
        >= cfg["max_long_term_elements"]
    )

    deva = DEVAInferenceCore(deva_model, config=cfg)
    deva.next_voting_frame = int(args.num_voting_frames) - 1
    saver = ResultSaver(
        str(output_dir),
        None,
        dataset="demo",
        object_manager=deva.object_manager,
    )

    started = _phase_start(torch)
    with torch.amp.autocast("cuda", enabled=bool(args.amp)):
        for index, (frame, image_path) in enumerate(loader):
            process_frame_automatic(
                deva,
                sam_model,
                image_path,
                saver,
                index,
                image_np=frame,
            )
        flush_buffer(deva, saver)
    saver.end()
    stages["automatic_segmentation_and_tracking"] = _phase_end(torch, started)

    prediction_path = output_dir / "pred.json"
    prediction_path.write_text(
        json.dumps(saver.video_json, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    prediction_summary = _summarize_predictions(prediction_path)
    mask_paths = sorted((output_dir / "Annotations").glob("*.png"))
    if len(mask_paths) != len(image_paths):
        raise RuntimeError(
            f"DEVA emitted {len(mask_paths)} masks for {len(image_paths)} frames"
        )

    inference_seconds = stages["automatic_segmentation_and_tracking"]["seconds"]
    peaks_allocated = [stage["peak_allocated_mib"] for stage in stages.values()]
    peaks_reserved = [stage["peak_reserved_mib"] for stage in stages.values()]
    report = {
        "schema": SCHEMA,
        "run_id": str(args.run_id),
        "status": "pass",
        "implementation": {
            "proposal_model": "SAM1 automatic mask generator",
            "temporal_model": "DEVA",
            "execution": "one-pass-short-id",
            "upstream_reference": "Tracking-Anything-with-DEVA/demo/demo_automatic.py",
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
        },
        "checkpoints": {
            "deva": {
                "path": str(deva_checkpoint),
                "bytes": deva_checkpoint.stat().st_size,
                "sha256": _sha256(deva_checkpoint),
            },
            "sam1": {
                "path": str(sam_checkpoint),
                "bytes": sam_checkpoint.stat().st_size,
                "sha256": _sha256(sam_checkpoint),
                "encoder": str(args.SAM_ENCODER_VERSION),
            },
            "torchvision_backbones": backbone_rows,
        },
        "configuration": {
            "size": int(args.size),
            "amp": bool(args.amp),
            "chunk_size": int(args.chunk_size),
            "detection_every": int(args.detection_every),
            "num_voting_frames": int(args.num_voting_frames),
            "max_num_objects": int(args.max_num_objects),
            "sam_points_per_side": int(args.SAM_NUM_POINTS_PER_SIDE),
            "sam_points_per_batch": int(args.SAM_NUM_POINTS_PER_BATCH),
            "sam_pred_iou_threshold": float(args.SAM_PRED_IOU_THRESHOLD),
            "sam_overlap_threshold": float(args.SAM_OVERLAP_THRESHOLD),
            "suppress_small_objects": bool(args.suppress_small_objects),
            "use_short_id": bool(args.use_short_id),
            "temporal_setting": str(args.temporal_setting),
            "num_workers": int(args.num_workers),
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
            "mask_count": len(mask_paths),
            "pred_json": str(prediction_path),
            **prediction_summary,
        },
        "limitations": [
            "IDs are episode-local and must not be merged across streams by numeric value.",
            "This stage supplies 2D identity proposals; global 3D association and heldout QC remain mandatory.",
            "DEVA/Gaussian Grouping licensing must be reviewed before commercial use.",
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
