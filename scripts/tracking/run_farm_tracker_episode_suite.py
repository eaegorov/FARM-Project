#!/usr/bin/env python3
"""Run a verified materialized tracker suite while loading the model once.

Backend-specific arguments are passed through to the existing single-episode
runner parser.  Per-episode paths/run IDs are owned by this orchestrator and
must not be supplied by the caller.

Examples::

  python scripts/tracking/run_farm_tracker_episode_suite.py --backend deva \
    --episodes-root EPISODES --output-root OUTPUT --run-id RUN \
    --model DEVA.pth --SAM_CHECKPOINT_PATH SAM.pth --torch-hub-dir HUB

  python scripts/tracking/run_farm_tracker_episode_suite.py --backend sam3 \
    --episodes-root EPISODES --output-root OUTPUT --run-id RUN \
    --model SAM3_DIR --prompt-file prompts.txt
"""

from __future__ import annotations

# Allow direct execution from the source checkout.
import sys as _entry_sys
from pathlib import Path as _EntryPath
_entry_root = _EntryPath(__file__).resolve().parents[2]
_entry_sys.path[:0] = [str(_entry_root), str(_entry_root / "src")]

import argparse
import gc
import hashlib
import json
import os
import shutil
import sys
import time
from pathlib import Path
from typing import Any, Sequence

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from farm_runtime.tracker_episode_suite import (  # noqa: E402
    EpisodeInput,
    PeakRssSampler,
    atomic_output_directory,
    audit_8bit_masks,
    load_materialized_suite,
    sha256_file,
)

SCHEMA = "farm.tracker-episode-suite.v1"


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
    started_at, allocated, reserved = started
    divisor = 1024.0 * 1024.0
    return {
        "seconds": float(time.perf_counter() - started_at),
        "start_allocated_mib": float(allocated / divisor),
        "start_reserved_mib": float(reserved / divisor),
        "peak_allocated_mib": float(torch_module.cuda.max_memory_allocated() / divisor),
        "peak_reserved_mib": float(torch_module.cuda.max_memory_reserved() / divisor),
    }


def _contains_option(arguments: Sequence[str], option: str) -> bool:
    return any(value == option or value.startswith(option + "=") for value in arguments)


def _reject_owned_options(arguments: Sequence[str], options: Sequence[str]) -> None:
    conflicts = [option for option in options if _contains_option(arguments, option)]
    if conflicts:
        raise ValueError(
            "suite runner owns per-episode options; remove: " + ", ".join(conflicts)
        )


def _write_json(path: Path, payload: dict[str, Any]) -> None:
    path.write_text(
        json.dumps(
            payload, ensure_ascii=False, indent=2, sort_keys=True, allow_nan=False
        )
        + "\n",
        encoding="utf-8",
    )


def _validate_episode_report(
    output_dir: Path, report: dict[str, Any], episode: EpisodeInput
) -> str:
    if report.get("status") != "pass":
        raise RuntimeError(f"episode did not report pass: {episode.episode_id}")
    if report.get("input", {}).get("episode_id") != episode.episode_id:
        raise RuntimeError(f"episode report provenance mismatch: {episode.episode_id}")
    if int(report.get("output", {}).get("mask_count", -1)) != episode.frame_count:
        raise RuntimeError(f"episode report mask count mismatch: {episode.episode_id}")
    path = output_dir / "measurement.json"
    if not path.is_file():
        raise RuntimeError(f"episode measurement is missing: {episode.episode_id}")
    stored = json.loads(path.read_text(encoding="utf-8"))
    if stored != report:
        raise RuntimeError(
            f"episode measurement content mismatch: {episode.episode_id}"
        )
    return sha256_file(path)


def _checkpoint_file(path: Path) -> dict[str, Any]:
    path = path.expanduser().resolve(strict=True)
    if not path.is_file():
        raise ValueError(f"checkpoint is not a file: {path}")
    return {
        "path": str(path),
        "bytes": path.stat().st_size,
        "sha256": sha256_file(path),
    }


def _checkpoint_directory(path: Path) -> dict[str, Any]:
    path = path.expanduser().resolve(strict=True)
    if not path.is_dir():
        raise ValueError(f"checkpoint is not a directory: {path}")
    candidates = sorted(
        file
        for file in path.rglob("*")
        if file.is_file()
        and (
            file.suffix in {".safetensors", ".json"}
            or file.name.endswith(".safetensors.index.json")
        )
    )
    if not candidates or not any(file.suffix == ".safetensors" for file in candidates):
        raise ValueError("SAM3 checkpoint directory has no safetensors weights")
    rows = [
        {
            "relative_path": str(file.relative_to(path)),
            "bytes": file.stat().st_size,
            "sha256": sha256_file(file),
        }
        for file in candidates
    ]
    aggregate = hashlib.sha256(
        "".join(
            f"{row['relative_path']}\0{row['bytes']}\0{row['sha256']}\n" for row in rows
        ).encode("utf-8")
    ).hexdigest()
    return {"directory": str(path), "files": rows, "aggregate_sha256": aggregate}


def _compact_deva_long_id_outputs(
    annotation_root: Path,
    video_json: dict[str, Any],
    *,
    maximum_local_id: int,
) -> tuple[dict[str, Any], list[dict[str, int]]]:
    """Convert lossless DEVA 24-bit IDs to collision-free uint8 episode IDs."""

    from PIL import Image

    annotations = video_json.get("annotations")
    if not isinstance(annotations, list):
        raise ValueError("DEVA video JSON has no annotations list")
    ordered_original_ids: list[int] = []
    seen: set[int] = set()
    for annotation in annotations:
        for segment in annotation.get("segments_info", []):
            original_id = int(segment["id"])
            if not 256 <= original_id < 256**3:
                raise ValueError(
                    "DEVA long ID is outside the 24-bit non-background range"
                )
            if original_id not in seen:
                seen.add(original_id)
                ordered_original_ids.append(original_id)
    if len(ordered_original_ids) > int(maximum_local_id):
        raise ValueError("DEVA episode exceeds the compact local identity budget")
    mapping = {
        original_id: local_id
        for local_id, original_id in enumerate(ordered_original_ids, start=1)
    }
    lookup = np.zeros(256**3, dtype=np.uint8)
    for original_id, local_id in mapping.items():
        lookup[original_id] = np.uint8(local_id)

    compact_root = annotation_root.with_name(annotation_root.name + ".compact")
    if compact_root.exists():
        raise FileExistsError(f"temporary compact mask directory exists: {compact_root}")
    compact_root.mkdir()
    observed: set[int] = set()
    try:
        paths = sorted(annotation_root.glob("*.png"), key=lambda path: path.name)
        if not paths:
            raise ValueError("DEVA emitted no long-ID masks")
        for path in paths:
            with Image.open(path) as image:
                array = np.asarray(image)
            if array.dtype != np.uint8 or array.ndim != 3 or array.shape[2] != 3:
                raise ValueError("DEVA long-ID mask must be an RGB uint8 PNG")
            decoded = (
                array[..., 0].astype(np.uint32)
                + (array[..., 1].astype(np.uint32) << 8)
                + (array[..., 2].astype(np.uint32) << 16)
            )
            frame_ids = set(int(value) for value in np.unique(decoded)) - {0}
            unknown = frame_ids.difference(mapping)
            if unknown:
                raise ValueError(
                    "DEVA RGB mask contains IDs absent from pred.json: "
                    + ", ".join(str(value) for value in sorted(unknown)[:8])
                )
            observed.update(frame_ids)
            compact = lookup[decoded]
            temporary = compact_root / (path.name + ".tmp")
            Image.fromarray(compact, mode="L").save(temporary, format="PNG")
            os.replace(temporary, compact_root / path.name)
        missing = set(mapping).difference(observed)
        if missing:
            raise ValueError(
                "DEVA pred.json IDs are absent from RGB masks: "
                + ", ".join(str(value) for value in sorted(missing)[:8])
            )
        shutil.rmtree(annotation_root)
        os.replace(compact_root, annotation_root)
    except BaseException:
        shutil.rmtree(compact_root, ignore_errors=True)
        raise

    compact_json = json.loads(json.dumps(video_json))
    for annotation in compact_json["annotations"]:
        for segment in annotation.get("segments_info", []):
            original_id = int(segment["id"])
            segment["deva_original_id"] = original_id
            segment["id"] = mapping[original_id]
    rows = [
        {"deva_original_id": original_id, "local_id": mapping[original_id]}
        for original_id in ordered_original_ids
    ]
    return compact_json, rows


def _episode_provenance(episode: EpisodeInput) -> dict[str, Any]:
    return {
        "episode_id": episode.episode_id,
        "camera": episode.camera,
        "view_family": episode.view_family,
        "split": episode.split,
        "frame_count": episode.frame_count,
        "episode_json": str(episode.episode_json),
        "episode_json_sha256": episode.episode_json_sha256,
        "ordered_frame_names_sha256": episode.ordered_frame_names_sha256,
        "ordered_frame_content_sha256": episode.ordered_frame_content_sha256,
        "first_source_frame": episode.source_names[0],
        "last_source_frame": episode.source_names[-1],
        "physical_timestamps": list(episode.physical_timestamps),
    }


class DevaBackend:
    name = "sam1_deva_automatic"

    def __init__(self, backend_argv: list[str]) -> None:
        import torch
        from torch.utils.data import DataLoader

        from deva.ext.automatic_processor import process_frame_automatic
        from deva.ext.automatic_sam import get_sam_model
        from deva.inference.data.simple_video_reader import (
            SimpleVideoReader,
            no_collate,
        )
        from deva.inference.demo_utils import flush_buffer
        from deva.inference.inference_core import DEVAInferenceCore
        from deva.inference.result_utils import ResultSaver
        from deva.model.network import DEVA
        from scripts.tracking.run_farm_deva_automatic_episode import (
            TORCHVISION_BACKBONES,
            _build_parser,
            _summarize_predictions,
        )

        _reject_owned_options(
            backend_argv,
            (
                "--img_path",
                "--frame-list",
                "--output",
                "--measurement-json",
                "--run-id",
                "--planner-manifest-sha256",
            ),
        )
        injected = [
            "--img_path",
            "/suite/placeholder/frames",
            "--output",
            "/suite/placeholder/output",
            "--run-id",
            "suite-placeholder",
        ]
        self.args = _build_parser().parse_args([*backend_argv, *injected])
        if not torch.cuda.is_available():
            raise RuntimeError("CUDA is required for the DEVA suite")
        if str(self.args.temporal_setting).lower() != "semionline":
            raise ValueError("only the audited semionline DEVA mode is supported")
        if not bool(self.args.use_short_id):
            raise ValueError("use_short_id is required for Gaussian Grouping masks")
        if not 1 <= int(self.args.max_num_objects) <= 254:
            raise ValueError("DEVA max_num_objects must be in [1, 254]")
        if int(self.args.num_workers) < 0:
            raise ValueError("num_workers must be non-negative")

        deva_checkpoint = Path(self.args.model).expanduser().resolve(strict=True)
        sam_checkpoint = (
            Path(self.args.SAM_CHECKPOINT_PATH).expanduser().resolve(strict=True)
        )
        hub = self.args.torch_hub_dir.expanduser().resolve(strict=True)
        backbone_rows: list[dict[str, Any]] = []
        for filename, expected_hash in TORCHVISION_BACKBONES.items():
            path = hub / "checkpoints" / filename
            row = _checkpoint_file(path)
            if row["sha256"] != expected_hash:
                raise ValueError(f"torchvision backbone checksum mismatch: {path}")
            backbone_rows.append(row)
        torch.hub.set_dir(str(hub))
        self.checkpoints = {
            "deva": _checkpoint_file(deva_checkpoint),
            "sam1": {
                **_checkpoint_file(sam_checkpoint),
                "encoder": self.args.SAM_ENCODER_VERSION,
            },
            "torchvision_backbones": backbone_rows,
        }

        torch.autograd.set_grad_enabled(False)
        torch.manual_seed(42)
        torch.cuda.manual_seed_all(42)
        np.random.seed(42)
        torch.backends.cudnn.benchmark = False
        self.torch = torch
        self.DataLoader = DataLoader
        self.SimpleVideoReader = SimpleVideoReader
        self.no_collate = no_collate
        self.process_frame = process_frame_automatic
        self.flush_buffer = flush_buffer
        self.InferenceCore = DEVAInferenceCore
        self.ResultSaver = ResultSaver
        self._summarize_predictions = _summarize_predictions

        self.cfg = vars(self.args).copy()
        self.cfg["enable_long_term"] = not bool(self.cfg["disable_long_term"])
        self.cfg["temporal_setting"] = str(self.cfg["temporal_setting"]).lower()
        self.cfg["model"] = str(deva_checkpoint)
        self.cfg["SAM_CHECKPOINT_PATH"] = str(sam_checkpoint)
        started = _phase_start(torch)
        with PeakRssSampler() as rss:
            self.deva_model = DEVA(self.cfg).cuda().eval()
            weights = torch.load(
                str(deva_checkpoint), map_location="cuda", weights_only=True
            )
            self.deva_model.load_weights(weights)
            del weights
        self.model_load = {"deva": {**_phase_end(torch, started), "rss": rss.report()}}
        started = _phase_start(torch)
        with PeakRssSampler() as rss:
            self.sam_model = get_sam_model(self.cfg, "cuda")
        self.model_load["sam1"] = {**_phase_end(torch, started), "rss": rss.report()}

    def configuration(self) -> dict[str, Any]:
        args = self.args
        return {
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
            "temporal_setting": str(args.temporal_setting),
            "num_workers": int(args.num_workers),
            "intermediate_identity_encoding": "lossless_deva_24bit_rgb",
            "published_identity_encoding": (
                "deterministic_episode_local_uint8_1_to_254"
            ),
        }

    def run_episode(
        self,
        episode: EpisodeInput,
        output_dir: Path,
        final_dir: Path,
        run_id: str,
    ) -> dict[str, Any]:
        torch = self.torch
        torch.manual_seed(42)
        torch.cuda.manual_seed_all(42)
        np.random.seed(42)
        output_dir.mkdir()
        cfg = self.cfg.copy()
        cfg["img_path"] = str(episode.frame_root)
        cfg["output"] = str(output_dir)
        reader = self.SimpleVideoReader(str(episode.frame_root))
        if len(reader) != episode.frame_count:
            raise RuntimeError(
                "DEVA reader frame count disagrees with episode manifest"
            )
        loader = self.DataLoader(
            reader,
            batch_size=None,
            collate_fn=self.no_collate,
            num_workers=min(int(self.args.num_workers), len(reader)),
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
        deva = self.InferenceCore(self.deva_model, config=cfg)
        deva.next_voting_frame = int(self.args.num_voting_frames) - 1
        # DEVA's nominal short-ID path can emit IDs above 255 after consensus
        # voting and silently wrap them while saving uint8. Preserve exact IDs
        # in RGB, then compact once the complete episode is available.
        deva.enabled_long_id()
        saver = self.ResultSaver(
            str(output_dir), None, dataset="demo", object_manager=deva.object_manager
        )
        # Long-ID RGB is an intermediate integrity format, not a request for
        # DEVA's extra blended JPEG visualizations.
        saver.visualize = False
        episode_started = time.perf_counter()
        started = _phase_start(torch)
        with PeakRssSampler() as rss:
            with torch.amp.autocast("cuda", enabled=bool(self.args.amp)):
                for index, (frame, image_path) in enumerate(loader):
                    self.process_frame(
                        deva,
                        self.sam_model,
                        image_path,
                        saver,
                        index,
                        image_np=frame,
                    )
                self.flush_buffer(deva, saver)
            saver.end()
        inference = {**_phase_end(torch, started), "rss": rss.report()}
        prediction_path = output_dir / "pred.json"
        compact_json, identity_mapping = _compact_deva_long_id_outputs(
            output_dir / "Annotations",
            saver.video_json,
            maximum_local_id=min(int(self.args.max_num_objects), 254),
        )
        _write_json(prediction_path, compact_json)
        prediction_summary = self._summarize_predictions(prediction_path)
        masks = audit_8bit_masks(
            output_dir / "Annotations",
            episode.frame_names,
            reference_frame_root=episode.frame_root,
        )
        if masks["local_identity_count"] > int(self.args.max_num_objects):
            raise RuntimeError("DEVA output exceeds configured local identity limit")
        if (
            prediction_summary["unique_local_identities"]
            != masks["local_identity_count"]
        ):
            raise RuntimeError(
                "DEVA pred.json identities disagree with emitted PNG masks"
            )
        report = {
            "schema": "farm.deva-automatic-episode.v1",
            "run_id": f"{run_id}:{episode.episode_id}",
            "status": "pass",
            "execution": "multi-episode-suite; shared model; fresh DEVAInferenceCore",
            "input": _episode_provenance(episode),
            "checkpoints": self.checkpoints,
            "configuration": self.configuration(),
            "measurement": {
                "wall_seconds": float(time.perf_counter() - episode_started),
                "inference_fps": float(episode.frame_count / inference["seconds"]),
                "peak_cuda_allocated_mib": inference["peak_allocated_mib"],
                "peak_cuda_reserved_mib": inference["peak_reserved_mib"],
                "peak_cpu_rss_mib": inference["rss"]["peak_rss_mib"],
                "stages": {"automatic_segmentation_and_tracking": inference},
                "shared_model_load_excluded": True,
            },
            "output": {
                "directory": str(final_dir),
                "pred_json": str(final_dir / "pred.json"),
                "pred_json_sha256": sha256_file(prediction_path),
                **prediction_summary,
                **masks,
                "identity_mapping": identity_mapping,
            },
            "limitations": [
                "IDs are episode-local and must not be joined by numeric value.",
                "Global 3D association and heldout QC remain mandatory.",
                "DEVA/Gaussian Grouping licensing must be reviewed before commercial use.",
            ],
        }
        _write_json(output_dir / "measurement.json", report)
        del saver, deva, loader, reader
        gc.collect()
        torch.cuda.empty_cache()
        return report


class Sam3Backend:
    name = "sam3_concept_video"

    def __init__(self, backend_argv: list[str]) -> None:
        import torch
        from PIL import Image
        from transformers import Sam3VideoModel, Sam3VideoProcessor
        from scripts.tracking.run_farm_sam3_concept_episode import (
            _compose_short_id_mask,
            _load_pinned_cv_utils_kernel,
            _parser,
            _read_lines,
        )

        _reject_owned_options(
            backend_argv,
            (
                "--image-dir",
                "--frame-list",
                "--output-dir",
                "--measurement-json",
                "--run-id",
                "--planner-manifest-sha256",
            ),
        )
        injected = [
            "--image-dir",
            "/suite/placeholder/frames",
            "--output-dir",
            "/suite/placeholder/output",
            "--run-id",
            "suite-placeholder",
        ]
        self.args = _parser().parse_args([*backend_argv, *injected])
        if not torch.cuda.is_available():
            raise RuntimeError("CUDA is required for the SAM3 suite")
        if not 0.0 <= float(self.args.minimum_score) <= 1.0:
            raise ValueError("minimum score must be in [0, 1]")
        if not 1 <= int(self.args.maximum_objects) <= 254:
            raise ValueError("SAM3 maximum objects must be in [1, 254]")
        if int(self.args.vision_cache_frames) < 1:
            raise ValueError("vision cache size must be positive")
        self.model_path = self.args.model.expanduser().resolve(strict=True)
        self.prompt_file = self.args.prompt_file.expanduser().resolve(strict=True)
        self.prompts = _read_lines(self.prompt_file, label="prompt file")
        self.checkpoints = {"sam3": _checkpoint_directory(self.model_path)}
        self.prompt_provenance = {
            "path": str(self.prompt_file),
            "bytes": self.prompt_file.stat().st_size,
            "sha256": sha256_file(self.prompt_file),
            "prompts": self.prompts,
        }
        self.torch = torch
        self.Image = Image
        self._compose_short_id_mask = _compose_short_id_mask
        torch.manual_seed(42)
        torch.cuda.manual_seed_all(42)
        torch.backends.cudnn.benchmark = False
        started = _phase_start(torch)
        with PeakRssSampler() as rss:
            self.kernel_runtime = _load_pinned_cv_utils_kernel()
            self.processor = Sam3VideoProcessor.from_pretrained(
                str(self.model_path), local_files_only=True
            )
            self.model = (
                Sam3VideoModel.from_pretrained(
                    str(self.model_path), local_files_only=True, dtype=torch.bfloat16
                )
                .eval()
                .to("cuda")
            )
        self.model_load = {"sam3": {**_phase_end(torch, started), "rss": rss.report()}}

    def configuration(self) -> dict[str, Any]:
        return {
            "dtype": "bfloat16",
            "minimum_score": float(self.args.minimum_score),
            "maximum_objects": int(self.args.maximum_objects),
            "inference_state_device": self.args.inference_state_device,
            "video_storage_device": self.args.video_storage_device,
            "vision_cache_frames": int(self.args.vision_cache_frames),
            "overlap_resolution": "highest_object_score",
            "cv_utils_kernel": self.kernel_runtime,
        }

    def run_episode(
        self,
        episode: EpisodeInput,
        output_dir: Path,
        final_dir: Path,
        run_id: str,
    ) -> dict[str, Any]:
        torch = self.torch
        torch.manual_seed(42)
        torch.cuda.manual_seed_all(42)
        output_dir.mkdir()
        mask_dir = output_dir / "Annotations"
        mask_dir.mkdir()
        episode_started = time.perf_counter()
        started = _phase_start(torch)
        with PeakRssSampler() as preprocess_rss:
            frames = []
            for image_path in sorted(
                episode.frame_root.iterdir(), key=lambda path: path.name
            ):
                with self.Image.open(image_path) as image:
                    frames.append(image.convert("RGB").copy())
            session = self.processor.init_video_session(
                video=[frames],
                inference_device="cuda",
                inference_state_device=self.args.inference_state_device,
                processing_device="cpu",
                video_storage_device=self.args.video_storage_device,
                max_vision_features_cache_size=int(self.args.vision_cache_frames),
                dtype=torch.bfloat16,
            )
            self.processor.add_text_prompt(session, self.prompts)
        preprocessing = {**_phase_end(torch, started), "rss": preprocess_rss.report()}
        frame_rows: list[dict[str, Any]] = []
        local_ids: dict[int, int] = {}
        prompt_by_id: dict[int, str] = {}
        started = _phase_start(torch)
        with PeakRssSampler() as inference_rss:
            with torch.inference_mode():
                for frame_index, (image_path, output_name) in enumerate(
                    zip(
                        sorted(
                            episode.frame_root.iterdir(), key=lambda path: path.name
                        ),
                        episode.frame_names,
                    )
                ):
                    model_output = self.model(session, frame_idx=frame_index)
                    output = self.processor.postprocess_outputs(session, model_output)
                    object_ids = output["object_ids"].detach().cpu().numpy()
                    scores = output["scores"].detach().float().cpu().numpy()
                    proposal_masks = output["masks"].detach().cpu().numpy()
                    for prompt, ids in output["prompt_to_obj_ids"].items():
                        for object_id in ids:
                            prompt_by_id[int(object_id)] = str(prompt)
                    index_mask, composition = self._compose_short_id_mask(
                        object_ids=object_ids,
                        scores=scores,
                        masks=proposal_masks,
                        local_id_by_object_id=local_ids,
                        minimum_score=float(self.args.minimum_score),
                        maximum_local_id=int(self.args.maximum_objects),
                    )
                    self.Image.fromarray(index_mask).save(
                        mask_dir / f"{Path(output_name).stem}.png"
                    )
                    frame_rows.append(
                        {
                            "frame_index": frame_index,
                            "frame_name": output_name,
                            **composition,
                            "objects": [
                                {
                                    "sam3_object_id": int(object_id),
                                    "local_id": local_ids.get(int(object_id)),
                                    "prompt": prompt_by_id.get(int(object_id)),
                                    "score": float(score),
                                    "area_px": int(mask.sum()),
                                }
                                for object_id, score, mask in zip(
                                    object_ids, scores, proposal_masks
                                )
                            ],
                        }
                    )
                    del model_output, output, proposal_masks
        inference = {**_phase_end(torch, started), "rss": inference_rss.report()}
        masks = audit_8bit_masks(
            mask_dir,
            episode.frame_names,
            reference_frame_root=episode.frame_root,
        )
        identity_rows = [
            {
                "sam3_object_id": int(object_id),
                "local_id": int(local_id),
                "prompt": prompt_by_id.get(int(object_id)),
            }
            for object_id, local_id in sorted(
                local_ids.items(), key=lambda item: item[1]
            )
        ]
        report = {
            "schema": "farm.sam3-concept-episode.v1",
            "run_id": f"{run_id}:{episode.episode_id}",
            "status": "pass",
            "execution": "multi-episode-suite; shared model; fresh video session",
            "input": {**_episode_provenance(episode), "prompt": self.prompt_provenance},
            "checkpoint": self.checkpoints["sam3"],
            "configuration": self.configuration(),
            "measurement": {
                "wall_seconds": float(time.perf_counter() - episode_started),
                "inference_fps": float(episode.frame_count / inference["seconds"]),
                "peak_cuda_allocated_mib": max(
                    preprocessing["peak_allocated_mib"], inference["peak_allocated_mib"]
                ),
                "peak_cuda_reserved_mib": max(
                    preprocessing["peak_reserved_mib"], inference["peak_reserved_mib"]
                ),
                "peak_cpu_rss_mib": max(
                    preprocessing["rss"]["peak_rss_mib"],
                    inference["rss"]["peak_rss_mib"],
                ),
                "stages": {
                    "video_preprocess_and_prompt_encoding": preprocessing,
                    "concept_detection_and_tracking": inference,
                },
                "shared_model_load_excluded": True,
            },
            "output": {
                "directory": str(final_dir),
                **masks,
                "identities": identity_rows,
                "frames": frame_rows,
            },
            "limitations": [
                "Prompt-conditioned recall depends on the Qwen-derived vocabulary.",
                "IDs are episode-local and require geometric association across streams.",
                "Depth, exclusivity and heldout silhouette gates remain mandatory.",
            ],
        }
        _write_json(output_dir / "measurement.json", report)
        del session, frames, frame_rows, local_ids, prompt_by_id
        gc.collect()
        torch.cuda.empty_cache()
        return report


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__, allow_abbrev=False)
    parser.add_argument("--backend", required=True, choices=("deva", "sam3"))
    parser.add_argument("--episodes-root", required=True, type=Path)
    parser.add_argument("--output-root", required=True, type=Path)
    parser.add_argument("--run-id", required=True)
    parser.add_argument("--episode-id", action="append")
    parser.add_argument("--split", action="append", choices=("train", "heldout"))
    return parser


def main(argv: list[str] | None = None) -> int:
    suite_started = time.perf_counter()
    args, backend_argv = _parser().parse_known_args(argv)
    if backend_argv and backend_argv[0] == "--":
        backend_argv = backend_argv[1:]
    input_started = time.perf_counter()
    suite = load_materialized_suite(
        args.episodes_root,
        episode_ids=set(args.episode_id or ()),
        splits=set(args.split or ()),
        verify_frame_hashes=True,
    )
    input_seconds = time.perf_counter() - input_started
    destination = args.output_root.expanduser().resolve()
    if destination.exists():
        raise FileExistsError(f"refusing to overwrite suite output: {destination}")

    import torch

    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required for tracker suite execution")
    with PeakRssSampler() as suite_rss:
        model_started = time.perf_counter()
        backend = (
            DevaBackend(backend_argv)
            if args.backend == "deva"
            else Sam3Backend(backend_argv)
        )
        model_wall_seconds = time.perf_counter() - model_started
        episode_reports: list[dict[str, Any]] = []
        with atomic_output_directory(destination) as staging:
            episode_root = staging / "episodes"
            episode_root.mkdir()
            episode_report_hashes: list[str] = []
            for episode in suite.episodes:
                output_dir = episode_root / episode.episode_id
                report = backend.run_episode(
                    episode,
                    output_dir,
                    destination / "episodes" / episode.episode_id,
                    str(args.run_id),
                )
                episode_report_hashes.append(
                    _validate_episode_report(output_dir, report, episode)
                )
                episode_reports.append(report)

            episode_measurements = [report["measurement"] for report in episode_reports]
            frame_count = sum(episode.frame_count for episode in suite.episodes)
            inference_seconds = sum(
                report["measurement"]["stages"][
                    (
                        "automatic_segmentation_and_tracking"
                        if args.backend == "deva"
                        else "concept_detection_and_tracking"
                    )
                ]["seconds"]
                for report in episode_reports
            )
            load_stages = list(backend.model_load.values())
            suite_report = {
                "schema": SCHEMA,
                "run_id": str(args.run_id),
                "status": "pass",
                "backend": backend.name,
                "execution": {
                    "model_loads": 1,
                    "episode_state": "fresh/reset per episode",
                    "publication": "atomic whole-suite rename",
                    "network_required": False,
                },
                "input": {
                    "materialized_root": str(suite.root),
                    "materialized_manifest": str(suite.manifest_path),
                    "materialized_manifest_sha256": suite.manifest_sha256,
                    "source_plan": str(suite.source_plan),
                    "source_plan_sha256": suite.source_plan_sha256,
                    "exact_frame_hashes_verified": True,
                    "episode_count": len(suite.episodes),
                    "frame_count": frame_count,
                    "episode_count_by_split": {
                        split: sum(1 for ep in suite.episodes if ep.split == split)
                        for split in sorted({ep.split for ep in suite.episodes})
                    },
                    "frame_count_by_split": {
                        split: sum(
                            ep.frame_count for ep in suite.episodes if ep.split == split
                        )
                        for split in sorted({ep.split for ep in suite.episodes})
                    },
                    "episodes": [_episode_provenance(ep) for ep in suite.episodes],
                },
                "checkpoints": backend.checkpoints,
                "configuration": backend.configuration(),
                "measurement": {
                    "input_validation_seconds": input_seconds,
                    "model_load_wall_seconds": model_wall_seconds,
                    "model_load_stages": backend.model_load,
                    "episode_inference_seconds": inference_seconds,
                    "episode_inference_fps": float(frame_count / inference_seconds),
                    "suite_wall_seconds": float(time.perf_counter() - suite_started),
                    "model_load_amortized_seconds_per_episode": float(
                        model_wall_seconds / len(suite.episodes)
                    ),
                    "peak_cuda_allocated_mib": max(
                        [stage["peak_allocated_mib"] for stage in load_stages]
                        + [
                            row["peak_cuda_allocated_mib"]
                            for row in episode_measurements
                        ]
                    ),
                    "peak_cuda_reserved_mib": max(
                        [stage["peak_reserved_mib"] for stage in load_stages]
                        + [
                            row["peak_cuda_reserved_mib"]
                            for row in episode_measurements
                        ]
                    ),
                    "per_episode": [
                        {
                            "episode_id": episode.episode_id,
                            "split": episode.split,
                            **report["measurement"],
                        }
                        for episode, report in zip(suite.episodes, episode_reports)
                    ],
                },
                "device": {
                    "name": torch.cuda.get_device_name(0),
                    "cuda_runtime": torch.version.cuda,
                    "torch": torch.__version__,
                },
                "output": {
                    "directory": str(destination),
                    "episode_count": len(episode_reports),
                    "mask_count": sum(
                        int(report["output"]["mask_count"])
                        for report in episode_reports
                    ),
                    "episode_reports": [
                        {
                            "episode_id": episode.episode_id,
                            "path": str(
                                destination
                                / "episodes"
                                / episode.episode_id
                                / "measurement.json"
                            ),
                            "sha256": digest,
                        }
                        for episode, digest in zip(
                            suite.episodes, episode_report_hashes
                        )
                    ],
                },
                "limitations": [
                    "All numeric identities remain episode-local.",
                    "The suite output is proposal evidence, not publishable FARM geometry.",
                    "Cross-episode 3D association and heldout gates are separate stages.",
                ],
            }
            # Snapshot after every episode; the sampler remains active through
            # serialization and the atomic directory publication.
            suite_report["measurement"]["suite_cpu_rss"] = suite_rss.report()
            _write_json(staging / "suite_measurement.json", suite_report)
    print(
        json.dumps(
            {
                "status": "pass",
                "backend": backend.name,
                "episodes": len(suite.episodes),
                "frames": sum(ep.frame_count for ep in suite.episodes),
                "output": str(destination),
            },
            indent=2,
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
