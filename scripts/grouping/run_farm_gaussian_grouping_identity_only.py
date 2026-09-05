#!/usr/bin/env python3
"""Train only Gaussian Grouping identity features on a frozen Factory 3DGS PLY.

This adapter deliberately does not optimize XYZ, SH appearance, opacity, scale,
rotation, or point count.  It saves learned identity features as a sidecar, so
the multi-gigabyte source PLY is neither copied nor silently changed.
"""

from __future__ import annotations

# Allow direct execution from the source checkout.
import sys as _entry_sys
from pathlib import Path as _EntryPath
_entry_root = _EntryPath(__file__).resolve().parents[2]
_entry_sys.path[:0] = [str(_entry_root), str(_entry_root / "src")]

import argparse
import json
import math
import os
import platform
import random
import resource
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from types import SimpleNamespace
from typing import Any, Iterable

import numpy as np
from PIL import Image

ROOT = Path(__file__).resolve().parents[2]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from farm_runtime.gaussian_grouping_dataset import (  # noqa: E402
    CameraRecord,
    ImageRecord,
    sha256_file,
)
from farm_runtime.gaussian_grouping_identity import (  # noqa: E402
    preflight_identity_only,
)
from farm_runtime.gaussian_grouping_visualization import (  # noqa: E402
    build_contact_sheet,
    export_heldout_frame,
    write_export_manifest,
)


def _atomic_json(path: Path, payload: Any) -> None:
    destination = Path(path)
    temporary = destination.with_name(
        f".{destination.name}.tmp-{os.getpid()}-{time.time_ns()}"
    )
    temporary.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    os.replace(temporary, destination)


def _read_filtered_colmap(
    dataset_root: Path,
) -> tuple[dict[int, CameraRecord], list[ImageRecord]]:
    cameras: dict[int, CameraRecord] = {}
    cameras_path = dataset_root / "sparse/0/cameras.txt"
    for line_number, line in enumerate(
        cameras_path.read_text(encoding="utf-8").splitlines(), 1
    ):
        stripped = line.strip()
        if not stripped or stripped.startswith("#"):
            continue
        parts = stripped.split()
        if len(parts) < 5:
            raise ValueError(f"{cameras_path}:{line_number}: malformed camera")
        camera_id = int(parts[0])
        record = CameraRecord(
            camera_id=camera_id,
            model=parts[1],
            width=int(parts[2]),
            height=int(parts[3]),
            params=tuple(float(value) for value in parts[4:]),
        )
        if camera_id in cameras:
            raise ValueError(f"duplicate camera ID {camera_id}")
        cameras[camera_id] = record
    images: list[ImageRecord] = []
    seen_ids: set[int] = set()
    seen_names: set[str] = set()
    images_path = dataset_root / "sparse/0/images.txt"
    for line_number, line in enumerate(
        images_path.read_text(encoding="utf-8").splitlines(), 1
    ):
        stripped = line.strip()
        if not stripped or stripped.startswith("#"):
            continue
        parts = stripped.split(maxsplit=9)
        if len(parts) != 10:
            raise ValueError(f"{images_path}:{line_number}: POINTS2D are not expected")
        record = ImageRecord(
            image_id=int(parts[0]),
            qvec=tuple(float(value) for value in parts[1:5]),  # type: ignore[arg-type]
            tvec=tuple(float(value) for value in parts[5:8]),  # type: ignore[arg-type]
            camera_id=int(parts[8]),
            name=parts[9],
        )
        if record.image_id in seen_ids or record.name in seen_names:
            raise ValueError("duplicate filtered COLMAP image ID/name")
        if record.camera_id not in cameras:
            raise ValueError(f"image references missing camera {record.camera_id}")
        seen_ids.add(record.image_id)
        seen_names.add(record.name)
        images.append(record)
    if not cameras or not images:
        raise ValueError("filtered COLMAP model is empty")
    return cameras, images


def _split_images(
    dataset_root: Path, images: Iterable[ImageRecord]
) -> tuple[list[ImageRecord], list[ImageRecord]]:
    by_name = {image.name: image for image in images}
    result: dict[str, list[ImageRecord]] = {}
    for split in ("train", "heldout"):
        names = {
            path.name
            for path in (dataset_root / f"images_{split}").iterdir()
            if path.is_symlink()
        }
        missing = sorted(names.difference(by_name))
        if missing:
            raise ValueError(
                f"images_{split} contains names absent from COLMAP: {missing[:8]}"
            )
        result[split] = [by_name[name] for name in sorted(names)]
    return result["train"], result["heldout"]


def _class_weights(
    dataset_root: Path, names: Iterable[str], num_classes: int
) -> np.ndarray:
    counts = np.zeros(num_classes, dtype=np.int64)
    for name in names:
        with Image.open(
            dataset_root / "object_mask" / f"{Path(name).stem}.png"
        ) as image:
            values = np.asarray(image, dtype=np.uint8)
        counts += np.bincount(values.reshape(-1), minlength=num_classes)[:num_classes]
    if np.any(counts <= 0):
        missing = np.flatnonzero(counts <= 0).tolist()
        raise ValueError(
            f"training masks have no pixels for declared classes: {missing}"
        )
    inverse_sqrt = np.sqrt(
        float(counts.sum()) / (float(num_classes) * counts.astype(np.float64))
    )
    weights = np.clip(inverse_sqrt, 0.25, 4.0)
    weights /= weights.mean()
    return weights.astype(np.float32)


def _load_frozen_gaussians(source_ply: Path, *, seed: int):
    import torch
    from plyfile import PlyData
    from scene.gaussian_model import GaussianModel

    ply = PlyData.read(str(source_ply), mmap="r")
    vertices = ply.elements[0]
    names = {prop.name for prop in vertices.properties}
    gaussian = GaussianModel(3)

    def values(columns: list[str]) -> "torch.Tensor":
        missing = sorted(set(columns).difference(names))
        if missing:
            raise ValueError("source PLY is missing: " + ", ".join(missing))
        array = np.stack(
            [np.asarray(vertices[column], dtype=np.float32) for column in columns],
            axis=1,
        )
        return torch.from_numpy(np.ascontiguousarray(array)).to(
            device="cuda", non_blocking=False
        )

    xyz = values(["x", "y", "z"])
    features_dc = values([f"f_dc_{index}" for index in range(3)])[:, None, :]
    # ``values`` already returns a CUDA tensor; reshape/permute without a host copy.
    features_rest = (
        values([f"f_rest_{index}" for index in range(45)])
        .reshape(-1, 3, 15)
        .transpose(1, 2)
        .contiguous()
    )
    opacity = values(["opacity"])
    scaling = values([f"scale_{index}" for index in range(3)])
    rotation = values([f"rot_{index}" for index in range(4)])
    point_count = int(xyz.shape[0])
    generator = torch.Generator(device="cuda")
    generator.manual_seed(int(seed))
    objects = torch.empty(
        (point_count, 1, gaussian.num_objects), dtype=torch.float32, device="cuda"
    )
    objects.normal_(mean=0.0, std=0.01, generator=generator)
    gaussian._xyz = torch.nn.Parameter(xyz, requires_grad=False)
    gaussian._features_dc = torch.nn.Parameter(features_dc, requires_grad=False)
    gaussian._features_rest = torch.nn.Parameter(
        features_rest.contiguous(), requires_grad=False
    )
    gaussian._opacity = torch.nn.Parameter(opacity, requires_grad=False)
    gaussian._scaling = torch.nn.Parameter(scaling, requires_grad=False)
    gaussian._rotation = torch.nn.Parameter(rotation, requires_grad=False)
    gaussian._objects_dc = torch.nn.Parameter(objects, requires_grad=True)
    gaussian.active_sh_degree = gaussian.max_sh_degree
    return gaussian


def _camera_and_target(
    dataset_root: Path,
    camera_record: CameraRecord,
    image_record: ImageRecord,
    *,
    resolution_factor: int,
    torch,
    camera_class,
    qvec2rotmat,
):
    width = max(1, int(round(camera_record.width / resolution_factor)))
    height = max(1, int(round(camera_record.height / resolution_factor)))
    if camera_record.model == "PINHOLE":
        fx, fy = camera_record.params[:2]
    elif camera_record.model == "SIMPLE_PINHOLE":
        fx = fy = camera_record.params[0]
    else:
        raise ValueError(
            f"unsupported identity-only camera model {camera_record.model}"
        )
    fovx = 2.0 * math.atan(camera_record.width / (2.0 * fx))
    fovy = 2.0 * math.atan(camera_record.height / (2.0 * fy))
    blank = torch.zeros((3, height, width), dtype=torch.float32, device="cpu")
    camera = camera_class(
        colmap_id=camera_record.camera_id,
        R=np.transpose(qvec2rotmat(np.asarray(image_record.qvec, dtype=np.float64))),
        T=np.asarray(image_record.tvec, dtype=np.float64),
        FoVx=fovx,
        FoVy=fovy,
        image=blank,
        gt_alpha_mask=None,
        image_name=Path(image_record.name).stem,
        uid=image_record.image_id,
        data_device="cpu",
        objects=None,
    )
    mask_path = dataset_root / "object_mask" / f"{Path(image_record.name).stem}.png"
    with Image.open(mask_path) as image:
        target_image = image.convert("L").resize(
            (width, height), resample=Image.Resampling.NEAREST
        )
        target = (
            torch.from_numpy(np.asarray(target_image, dtype=np.uint8).copy())
            .long()
            .cuda()
        )
    return camera, target


def _heldout_metrics(
    *,
    dataset_root: Path,
    cameras: dict[int, CameraRecord],
    images: list[ImageRecord],
    resolution_factor: int,
    gaussians,
    classifier,
    renderer,
    pipeline,
    background,
    torch,
    camera_class,
    qvec2rotmat,
    num_classes: int,
) -> dict[str, Any]:
    confusion = torch.zeros((num_classes, num_classes), dtype=torch.int64, device="cpu")
    with torch.no_grad():
        for image in images:
            camera, target = _camera_and_target(
                dataset_root,
                cameras[image.camera_id],
                image,
                resolution_factor=resolution_factor,
                torch=torch,
                camera_class=camera_class,
                qvec2rotmat=qvec2rotmat,
            )
            objects = renderer(camera, gaussians, pipeline, background)["render_object"]
            prediction = classifier(objects).argmax(dim=0)
            encoded = (target.reshape(-1) * num_classes + prediction.reshape(-1)).cpu()
            confusion += torch.bincount(
                encoded, minlength=num_classes * num_classes
            ).reshape(num_classes, num_classes)
            del camera, target, objects, prediction, encoded
    true_positive = confusion.diag().to(torch.float64)
    union = confusion.sum(0) + confusion.sum(1) - confusion.diag()
    iou = torch.where(union > 0, true_positive / union, torch.nan)
    total = int(confusion.sum().item())
    return {
        "frames": len(images),
        "pixel_accuracy": float(true_positive.sum().item() / total) if total else None,
        "foreground_mean_iou": float(torch.nanmean(iou[1:]).item()),
        "per_class_iou": [
            None if torch.isnan(value) else float(value) for value in iou
        ],
        "confusion": confusion.tolist(),
    }


def _final_heldout_export(
    *,
    output_root: Path,
    dataset_root: Path,
    cameras: dict[int, CameraRecord],
    images: list[ImageRecord],
    resolution_factor: int,
    gaussians,
    classifier,
    renderer,
    pipeline,
    background,
    torch,
    camera_class,
    qvec2rotmat,
    maximum_id: int,
) -> dict[str, Any]:
    """Render every frozen heldout view once without updating model state."""

    if not images:
        raise ValueError("final heldout export requires at least one frozen frame")
    stems = [Path(image.name).stem for image in images]
    if len(stems) != len(set(stems)):
        raise ValueError("heldout image names must have unique PNG output stems")
    export_root = output_root / "heldout_export"
    rows: list[dict[str, Any]] = []
    classifier_training = bool(classifier.training)
    classifier.eval()
    try:
        with torch.no_grad():
            for image_record in images:
                camera, target = _camera_and_target(
                    dataset_root,
                    cameras[image_record.camera_id],
                    image_record,
                    resolution_factor=resolution_factor,
                    torch=torch,
                    camera_class=camera_class,
                    qvec2rotmat=qvec2rotmat,
                )
                rendered = renderer(camera, gaussians, pipeline, background)
                prediction = classifier(rendered["render_object"]).argmax(dim=0)
                target_np = target.detach().cpu().numpy()
                prediction_np = prediction.detach().cpu().numpy()
                height, width = target_np.shape
                source_path = dataset_root / "images" / image_record.name
                with Image.open(source_path) as source:
                    rgb = np.asarray(
                        source.convert("RGB").resize(
                            (width, height), resample=Image.Resampling.LANCZOS
                        ),
                        dtype=np.uint8,
                    ).copy()
                rows.append(
                    export_heldout_frame(
                        export_root,
                        source_name=image_record.name,
                        source_sha256=sha256_file(source_path),
                        rgb=rgb,
                        target=target_np,
                        prediction=prediction_np,
                        maximum_id=maximum_id,
                    )
                )
                del camera, target, rendered, prediction
    finally:
        classifier.train(classifier_training)
    if len(rows) != len(images):
        raise RuntimeError("final heldout export did not cover every frozen frame")
    contact_sheet = build_contact_sheet(export_root, rows)
    return write_export_manifest(
        export_root,
        rows,
        contact_sheet,
        maximum_id=maximum_id,
        resolution_factor=resolution_factor,
    )


def run_training(
    args: argparse.Namespace, preflight: dict[str, Any], output_root: Path
) -> dict[str, Any]:
    repository = Path(preflight["gaussian_grouping_repo"])
    sys.path.insert(0, str(repository))
    import torch
    import torch.nn.functional as functional
    from gaussian_renderer import render
    from scene.cameras import Camera
    from scene.colmap_loader import qvec2rotmat
    from utils.loss_utils import loss_cls_3d

    if not torch.cuda.is_available():
        raise RuntimeError("identity-only Gaussian Grouping requires CUDA")
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)
    torch.cuda.reset_peak_memory_stats()
    dataset_root = Path(preflight["dataset_root"])
    cameras, images = _read_filtered_colmap(dataset_root)
    train_images, heldout_images = _split_images(dataset_root, images)
    num_classes = int(preflight["num_classes"])
    weights_np = _class_weights(
        dataset_root, (image.name for image in train_images), num_classes
    )
    weights = torch.from_numpy(weights_np).cuda()
    stage_times: dict[str, float] = {}
    started = time.perf_counter()
    stage = time.perf_counter()
    gaussians = _load_frozen_gaussians(
        Path(preflight["source_gaussians"]["path"]), seed=args.seed
    )
    torch.cuda.synchronize()
    stage_times["load_source_gaussians"] = time.perf_counter() - stage
    classifier = torch.nn.Conv2d(
        gaussians.num_objects, num_classes, kernel_size=1
    ).cuda()
    object_optimizer = torch.optim.Adam(
        [gaussians._objects_dc], lr=args.identity_lr, eps=1e-15
    )
    classifier_optimizer = torch.optim.Adam(
        classifier.parameters(), lr=args.classifier_lr
    )
    pipeline = SimpleNamespace(
        convert_SHs_python=False, compute_cov3D_python=False, debug=False
    )
    background = torch.zeros(3, dtype=torch.float32, device="cuda")
    history: list[dict[str, Any]] = []
    evaluation: list[dict[str, Any]] = []
    train_started = time.perf_counter()
    for iteration in range(1, args.iterations + 1):
        image = train_images[(iteration - 1) % len(train_images)]
        camera, target = _camera_and_target(
            dataset_root,
            cameras[image.camera_id],
            image,
            resolution_factor=args.resolution_factor,
            torch=torch,
            camera_class=Camera,
            qvec2rotmat=qvec2rotmat,
        )
        rendered = render(camera, gaussians, pipeline, background)
        logits = classifier(rendered["render_object"])
        loss_2d = functional.cross_entropy(
            logits.unsqueeze(0), target.unsqueeze(0), weight=weights
        )
        loss_2d = loss_2d / torch.log(torch.tensor(float(num_classes), device="cuda"))
        loss_3d = torch.zeros((), device="cuda")
        if args.reg3d_interval > 0 and iteration % args.reg3d_interval == 0:
            logits_3d = classifier(gaussians._objects_dc.permute(2, 0, 1))
            probabilities = torch.softmax(logits_3d, dim=0).squeeze().permute(1, 0)
            loss_3d = loss_cls_3d(
                gaussians._xyz.detach(),
                probabilities,
                args.reg3d_k,
                args.reg3d_lambda,
                args.reg3d_max_points,
                args.reg3d_sample_size,
            )
        loss = loss_2d + loss_3d
        loss.backward()
        object_optimizer.step()
        classifier_optimizer.step()
        object_optimizer.zero_grad(set_to_none=True)
        classifier_optimizer.zero_grad(set_to_none=True)
        if (
            iteration == 1
            or iteration % args.log_every == 0
            or iteration == args.iterations
        ):
            history.append(
                {
                    "iteration": iteration,
                    "image": image.name,
                    "loss_total": float(loss.detach().cpu()),
                    "loss_2d": float(loss_2d.detach().cpu()),
                    "loss_3d": float(loss_3d.detach().cpu()),
                }
            )
        del camera, target, rendered, logits, loss, loss_2d, loss_3d
        if args.eval_every > 0 and (
            iteration % args.eval_every == 0 or iteration == args.iterations
        ):
            metrics = _heldout_metrics(
                dataset_root=dataset_root,
                cameras=cameras,
                images=heldout_images,
                resolution_factor=args.resolution_factor,
                gaussians=gaussians,
                classifier=classifier,
                renderer=render,
                pipeline=pipeline,
                background=background,
                torch=torch,
                camera_class=Camera,
                qvec2rotmat=qvec2rotmat,
                num_classes=num_classes,
            )
            metrics["iteration"] = iteration
            evaluation.append(metrics)
    torch.cuda.synchronize()
    stage_times["identity_training_and_eval"] = time.perf_counter() - train_started
    save_started = time.perf_counter()
    sidecar_path = output_root / "identity_features.pt"
    classifier_path = output_root / "classifier.pth"
    torch.save(
        {
            "schema": "farm.gaussian-grouping-identity-sidecar.v1",
            "source_gaussians_sha256": preflight["source_gaussians"]["sha256"],
            "source_gaussian_count": preflight["source_gaussians"]["vertex_count"],
            "dataset_manifest_sha256": preflight["dataset_manifest"]["sha256"],
            "identity_dimension": gaussians.num_objects,
            "objects_dc": gaussians._objects_dc.detach().cpu(),
        },
        sidecar_path,
    )
    torch.save(classifier.state_dict(), classifier_path)
    stage_times["save_sidecars"] = time.perf_counter() - save_started
    export_started = time.perf_counter()
    heldout_export = _final_heldout_export(
        output_root=output_root,
        dataset_root=dataset_root,
        cameras=cameras,
        images=heldout_images,
        resolution_factor=args.resolution_factor,
        gaussians=gaussians,
        classifier=classifier,
        renderer=render,
        pipeline=pipeline,
        background=background,
        torch=torch,
        camera_class=Camera,
        qvec2rotmat=qvec2rotmat,
        maximum_id=int(preflight["maximum_global_id"]),
    )
    torch.cuda.synchronize()
    stage_times["final_read_only_heldout_export"] = time.perf_counter() - export_started
    source_hash_after = sha256_file(Path(preflight["source_gaussians"]["path"]))
    if source_hash_after != preflight["source_gaussians"]["sha256"]:
        raise RuntimeError("source Gaussian PLY changed during identity-only training")
    torch.cuda.synchronize()
    return {
        "schema": "farm.gaussian-grouping-identity-run.v1",
        "status": "complete",
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "mode": "identity-only-frozen-source-gaussians",
        "preflight": preflight,
        "configuration": {
            key: value
            for key, value in vars(args).items()
            if isinstance(value, (str, int, float, bool, type(None)))
        },
        "class_weights": weights_np.tolist(),
        "training_history": history,
        "heldout_evaluation": evaluation,
        "outputs": {
            "identity_sidecar": {
                "path": str(sidecar_path),
                "bytes": sidecar_path.stat().st_size,
                "sha256": sha256_file(sidecar_path),
            },
            "classifier": {
                "path": str(classifier_path),
                "bytes": classifier_path.stat().st_size,
                "sha256": sha256_file(classifier_path),
            },
            "heldout_export": heldout_export,
            "combined_ply": None,
        },
        "invariants": {
            "geometry_parameters_in_optimizer": False,
            "appearance_parameters_in_optimizer": False,
            "opacity_parameters_in_optimizer": False,
            "densification_or_pruning": False,
            "source_ply_hash_unchanged": True,
            "source_order_preserved": True,
            "heldout_export_updates_training": False,
            "heldout_export_updates_geometry": False,
            "all_heldout_frames_exported": (
                int(heldout_export["frame_count"]) == len(heldout_images)
            ),
        },
        "measurement": {
            "wall_seconds": time.perf_counter() - started,
            "stages_seconds": stage_times,
            "peak_vram_allocated_mib": torch.cuda.max_memory_allocated() / 1024**2,
            "peak_vram_reserved_mib": torch.cuda.max_memory_reserved() / 1024**2,
            "peak_cpu_rss_mib": resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
            / 1024.0,
        },
        "runtime": {
            "python": platform.python_version(),
            "torch": torch.__version__,
            "cuda": torch.version.cuda,
            "gpu": torch.cuda.get_device_name(0),
        },
        "license_note": preflight["license_note"],
    }


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset-root", required=True, type=Path)
    parser.add_argument("--gaussian-grouping-repo", required=True, type=Path)
    parser.add_argument("--output-root", required=True, type=Path)
    parser.add_argument("--iterations", type=int, default=5_000)
    parser.add_argument(
        "--resolution-factor", type=int, default=4, choices=[1, 2, 4, 8]
    )
    parser.add_argument("--identity-lr", type=float, default=0.0025)
    parser.add_argument("--classifier-lr", type=float, default=0.0005)
    parser.add_argument("--reg3d-interval", type=int, default=2)
    parser.add_argument("--reg3d-k", type=int, default=5)
    parser.add_argument("--reg3d-lambda", type=float, default=2.0)
    parser.add_argument("--reg3d-max-points", type=int, default=200_000)
    parser.add_argument("--reg3d-sample-size", type=int, default=800)
    parser.add_argument("--eval-every", type=int, default=500)
    parser.add_argument("--log-every", type=int, default=25)
    parser.add_argument("--seed", type=int, default=20260903)
    parser.add_argument("--preflight-only", action="store_true")
    args = parser.parse_args(argv)
    if args.iterations <= 0 or args.log_every <= 0:
        parser.error("--iterations and --log-every must be positive")
    if args.reg3d_interval < 0 or args.eval_every < 0:
        parser.error("--reg3d-interval and --eval-every must be non-negative")
    for name in ("identity_lr", "classifier_lr", "reg3d_lambda"):
        if getattr(args, name) <= 0:
            parser.error(f"--{name.replace('_', '-')} must be positive")
    return args


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    output_root = args.output_root.expanduser().resolve()
    if output_root.exists():
        raise FileExistsError(
            f"refusing to overwrite identity-only output: {output_root}"
        )
    output_root.mkdir(parents=True)
    try:
        preflight = preflight_identity_only(
            args.dataset_root,
            args.gaussian_grouping_repo,
            resolution_factor=args.resolution_factor,
        )
        _atomic_json(output_root / "preflight.json", preflight)
        if args.preflight_only:
            report = {
                "schema": "farm.gaussian-grouping-identity-run.v1",
                "status": "preflight_passed_training_not_run",
                "created_utc": datetime.now(timezone.utc).isoformat(),
                "preflight": preflight,
            }
        else:
            report = run_training(args, preflight, output_root)
        _atomic_json(output_root / "run.json", report)
        print(
            json.dumps(
                {"status": report["status"], "output_root": str(output_root)}, indent=2
            )
        )
        return 0
    except BaseException as exc:
        _atomic_json(
            output_root / "failure.json",
            {
                "schema": "farm.gaussian-grouping-identity-failure.v1",
                "status": "failed",
                "error_type": type(exc).__name__,
                "error": str(exc),
                "created_utc": datetime.now(timezone.utc).isoformat(),
            },
        )
        raise


if __name__ == "__main__":
    raise SystemExit(main())
