"""Measured angular/upright ablations using the existing FARM YOLOE checkpoint.

Run with python -m farm_runtime.quality.discovery. No 3D assignment or identity
changes occur here. Outputs are proposals with soft mask logits, never gold.
"""

from __future__ import annotations

import argparse
from collections import Counter, defaultdict
import json
from pathlib import Path
import resource
import time

import numpy as np
from PIL import Image, ImageDraw

from farm_runtime.angular_discovery import (
    balanced_view_selection,
    rotate_image,
    upright_quarter_turns,
)
from farm_runtime.quality_baseline import describe_file, json_digest, write_json
from farm_runtime.quality_benchmark import verify_packet


def decode_soft_masks(proto, coefficients, boxes, shape):
    """Match the pinned native binary decoder, retaining pre-threshold logits."""
    import torch
    from ultralytics.utils import ops

    c, h, w = proto.shape
    logits = (coefficients @ proto.float().view(c, -1)).view(-1, h, w)
    logits = ops.scale_masks(logits[None], shape)[0]
    inside_box = ops.crop_mask(torch.ones_like(logits), boxes).bool()
    return logits.masked_fill(~inside_box, -20.0)


def soft_mask_predictor():
    """Pin to the repository YOLOE predictor; preserve logits before >0."""
    import torch
    from ultralytics.engine.results import Results
    from ultralytics.models.yolo.segment.predict import SegmentationPredictor
    from ultralytics.utils import ops

    class SoftMaskPredictor(SegmentationPredictor):
        def postprocess(self, preds, img, orig_imgs):
            predictions = ops.non_max_suppression(
                preds[0],
                self.args.conf,
                self.args.iou,
                agnostic=self.args.agnostic_nms,
                max_det=self.args.max_det,
                nc=len(self.model.names),
                classes=self.args.classes,
            )
            if not isinstance(orig_imgs, list):
                orig_imgs = ops.convert_torch2numpy_batch(orig_imgs)
            proto = preds[1][-1] if isinstance(preds[1], tuple) else preds[1]
            results = []
            for i, (pred, original, path) in enumerate(
                zip(predictions, orig_imgs, self.batch[0])
            ):
                logits = None
                if len(pred):
                    pred[:, :4] = ops.scale_boxes(
                        img.shape[2:], pred[:, :4], original.shape
                    )
                    logits = decode_soft_masks(
                        proto[i], pred[:, 6:], pred[:, :4], original.shape[:2]
                    )
                result = Results(
                    original,
                    path=path,
                    names=self.model.names,
                    boxes=pred[:, :6],
                    masks=None if logits is None else logits > 0,
                )
                result.farm_mask_logits = logits
                results.append(result)
            return results

    return SoftMaskPredictor


def build_ablation(packet, colmap, *, sampling, timestamp_count, image_root):
    import pycolmap
    from scripts.select_colmap_keyframes import (
        load_registered_views,
        DEFAULT_NAME_REGEX,
    )

    errors = verify_packet(packet)
    if errors:
        raise ValueError(errors)
    reconstruction = pycolmap.Reconstruction(str(colmap))
    views = load_registered_views(reconstruction, DEFAULT_NAME_REGEX, "default")
    allowed = {
        ts
        for ts, role in packet["split_by_timestamp"].items()
        if role in ("train", "dev")
    }
    if sampling == "pilot":
        timestamps = sorted(
            {
                r["physical_timestamp"]
                for r in packet["observations"]
                if r["split"] != "test"
            }
        )
    else:
        available = sorted(allowed)
        count = min(timestamp_count, len(available))
        timestamps = [
            available[i]
            for i in np.linspace(0, len(available) - 1, count).round().astype(int)
        ]
    if set(timestamps) - allowed:
        raise ValueError("test timestamp requested")
    by_timestamp = defaultdict(list)
    for view in views:
        if view.timestamp in timestamps:
            by_timestamp[view.timestamp].append(view)
    sensors = sorted({v.sensor for rows in by_timestamp.values() for v in rows})
    families = sorted({v.family for rows in by_timestamp.values() for v in rows})
    if "center" not in families:
        raise ValueError("this A/B requires an explicit center family baseline")
    center = [v for ts in timestamps for v in by_timestamp[ts] if v.family == "center"]
    balanced, angular = balanced_view_selection(
        by_timestamp, timestamps, sensors, families
    )
    all_views = [v for ts in timestamps for v in by_timestamp[ts]]
    if len(center) != len(balanced):
        raise ValueError("center and balanced must have the same source-view budget")
    lookup = {}
    for v in all_views:
        camera = reconstruction.cameras[v.camera_id]
        lookup[v.name] = {
            "name": v.name,
            "timestamp": v.timestamp,
            "sensor": v.sensor,
            "family": v.family,
            "shape_hw": [int(camera.height), int(camera.width)],
            "camera_from_world_rotation": v.rotation.tolist(),
            "source_image": describe_file(image_root / v.name),
        }
    variants = [
        {
            "name": "center_raw",
            "views": [v.name for v in center],
            "upright": False,
            "resolution": 640,
        },
        {
            "name": "balanced_raw",
            "views": [v.name for v in balanced],
            "upright": False,
            "resolution": 640,
        },
        {
            "name": "balanced_upright",
            "views": [v.name for v in balanced],
            "upright": True,
            "resolution": 640,
        },
        {
            "name": "all_lowres_upright",
            "views": [v.name for v in all_views],
            "upright": True,
            "resolution": 288,
        },
    ]
    return {
        "schema": "farm.discovery-ablation-plan.v1",
        "packet_sha256": json_digest(packet),
        "sampling": sampling,
        "timestamps": timestamps,
        "sources": lookup,
        "variants": variants,
        "angular_selection": angular,
        "test_opened": False,
        "comparison": "center/balanced preserve timestamp and source pixel budget; lowres uses more decoded pixels and is separately costed",
        "quality_reference": "user scope plus assistant visual review; no manual pixel gold",
    }


def predictor_parity(model, plan, *, resolution=640):
    """Compare the custom soft decoder to the pinned native YOLOE predictor."""
    import torch
    from ultralytics.models.yolo.segment.predict import SegmentationPredictor

    rows = []
    for name in plan["variants"][0]["views"][:2]:
        with Image.open(plan["sources"][name]["source_image"]["path"]) as im:
            image = im.convert("RGB")
            image.thumbnail((resolution, resolution))
        options = dict(
            source=image,
            imgsz=resolution,
            conf=0.4,
            iou=0.5,
            agnostic_nms=True,
            max_det=200,
            device=0,
            retina_masks=True,
            verbose=False,
            save=False,
        )
        model.predictor = None
        native = model.predict(**options, predictor=SegmentationPredictor)[0]
        native_boxes = native.boxes.data.detach().clone()
        native_masks = (
            None if native.masks is None else native.masks.data.detach().clone().bool()
        )
        model.predictor = None
        soft = model.predict(**options, predictor=soft_mask_predictor())[0]
        box_equal = native_boxes.shape == soft.boxes.data.shape and torch.allclose(
            native_boxes, soft.boxes.data, atol=1e-4, rtol=1e-5
        )
        if native_masks is None:
            mask_equal = soft.masks is None
        else:
            mask_equal = soft.masks is not None and torch.equal(
                native_masks, soft.masks.data.bool()
            )
        rows.append(
            dict(
                source=name,
                boxes_scores_classes_equal=bool(box_equal),
                masks_equal=bool(mask_equal),
                detections=len(native_boxes),
            )
        )
        if not box_equal or not mask_equal:
            raise ValueError("custom decoder differs from native YOLOE predictions")
    model.predictor = None
    return rows


def infer(plan, output, *, model_root, vocabulary, world_up):
    import torch
    from scene_graph.segmentation.yoloe import YOLOESegmenter

    torch.manual_seed(20260905)
    started = time.monotonic()
    segmenter = YOLOESegmenter(
        vocab_file=vocabulary, conf_thres=0.4, use_dino_features=False
    )
    model = segmenter.model
    # Dedicated predictor retains the same checkpoint/vocabulary/NMS semantics.
    model.predictor = None
    predictor = soft_mask_predictor()
    torch.cuda.synchronize()
    setup_seconds = time.monotonic() - started
    setup_peak_allocated_mib = torch.cuda.max_memory_allocated() / 2**20
    parity = predictor_parity(model, plan)
    results = []
    for variant in plan["variants"]:
        dest = output / variant["name"]
        dest.mkdir()
        (dest / "masks").mkdir()
        # First-call predictor/backend setup can allocate much more than steady
        # inference. Report it separately so variant order does not bias A/B.
        warm_started = time.monotonic()
        torch.cuda.reset_peak_memory_stats()
        model.predict(
            source=Image.new("RGB", (variant["resolution"], variant["resolution"])),
            imgsz=variant["resolution"],
            conf=0.4,
            iou=0.5,
            agnostic_nms=True,
            max_det=200,
            device=0,
            retina_masks=True,
            verbose=False,
            save=False,
            predictor=predictor,
        )
        torch.cuda.synchronize()
        warmup = {
            "seconds": time.monotonic() - warm_started,
            "peak_allocated_mib": torch.cuda.max_memory_allocated() / 2**20,
        }
        torch.cuda.empty_cache()
        before = time.monotonic()
        torch.cuda.reset_peak_memory_stats()
        observations = []
        decoded_pixels = 0
        model_pixels = 0
        canvas = Image.new("RGB", (1600, 5 * 330 + 50), "#111827")
        draw = ImageDraw.Draw(canvas)
        draw.text((12, 12), variant["name"] + " / proposal masks / no GT", fill="white")
        for index, name in enumerate(variant["views"]):
            row = plan["sources"][name]
            with Image.open(row["source_image"]["path"]) as image:
                rgb = np.asarray(image.convert("RGB"))
            h, w = rgb.shape[:2]
            if [h, w] != row["shape_hw"]:
                raise ValueError("source dimensions changed")
            decoded_pixels += h * w
            orientation = upright_quarter_turns(
                np.asarray(row["camera_from_world_rotation"]), world_up
            )
            turns = (
                orientation["applied_quarter_turns_ccw"] if variant["upright"] else 0
            )
            rgb = rotate_image(rgb, turns)
            rh, rw = rgb.shape[:2]
            size = variant["resolution"]
            scale = min(1.0, size / max(rh, rw))
            small = np.asarray(
                Image.fromarray(rgb).resize(
                    (round(rw * scale), round(rh * scale)), Image.Resampling.BILINEAR
                )
            )
            model_pixels += size * size  # padded YOLO tensor, batch=1
            pred = model.predict(
                source=Image.fromarray(small),
                imgsz=size,
                conf=0.4,
                iou=0.5,
                agnostic_nms=True,
                max_det=200,
                device=0,
                retina_masks=True,
                verbose=False,
                save=False,
                predictor=predictor,
            )[0]
            masks = pred.farm_mask_logits
            detections = []
            arrays = {}
            overlay = rotate_image(small, -turns).copy()
            oh, ow = overlay.shape[:2]
            for j, box in enumerate(pred.boxes):
                logits = rotate_image(masks[j].detach().float().cpu().numpy(), -turns)
                ys, xs = np.where(logits > 0)
                if not len(xs):
                    continue
                # Soft tiles are stored in the original orientation at the
                # declared detector grid; zero padding is not a probability.
                x0, x1, y0, y1 = (
                    int(xs.min()),
                    int(xs.max()) + 1,
                    int(ys.min()),
                    int(ys.max()) + 1,
                )
                margin = 2
                x0, x1, y0, y1 = (
                    max(0, x0 - margin),
                    min(ow, x1 + margin),
                    max(0, y0 - margin),
                    min(oh, y1 + margin),
                )
                key = f"logits_{j:03d}"
                arrays[key] = logits[y0:y1, x0:x1].astype(np.float16)
                label = pred.names[int(box.cls.item())]
                detections.append(
                    {
                        "index": j,
                        "label": label,
                        "score": float(box.conf.item()),
                        "logit_key": key,
                        "grid_window_xyxy": [x0, y0, x1, y1],
                        "source_bbox_xyxy": [
                            x0 * w / ow,
                            y0 * h / oh,
                            x1 * w / ow,
                            y1 * h / oh,
                        ],
                        "positive_grid_pixels": len(xs),
                    }
                )
                colour = np.array(
                    [
                        (j * 67 + 71) % 220 + 20,
                        (j * 113 + 47) % 220 + 20,
                        (j * 31 + 89) % 220 + 20,
                    ]
                )
                mask = logits > 0
                overlay[mask] = np.clip(
                    overlay[mask] * 0.55 + colour * 0.45, 0, 255
                ).astype(np.uint8)
            mask_path = dest / "masks" / (Path(name).stem + ".npz")
            np.savez_compressed(mask_path, **arrays)
            observations.append(
                {
                    **row,
                    "orientation": orientation,
                    "applied_quarter_turns": turns,
                    "grid_shape_hw": [oh, ow],
                    "detections": detections,
                    "mask_artifact": describe_file(mask_path),
                    "logit_semantics": "uncalibrated prototype logits; foreground threshold 0; FP16 soft tiles in source orientation",
                }
            )
            if index < 20:
                tile = Image.fromarray(overlay).resize(
                    (390, 290), Image.Resampling.BILINEAR
                )
                x = (index % 4) * 400
                y = (index // 4) * 330 + 50
                canvas.paste(tile, (x, y))
                draw.text((x + 3, y + 292), name, fill="white")
                draw.text(
                    (x + 3, y + 307),
                    ", ".join(d["label"] for d in detections[:5]),
                    fill="#fbbf24",
                )
            if (index + 1) % 20 == 0:
                print(
                    json.dumps(
                        {
                            "variant": variant["name"],
                            "done": index + 1,
                            "total": len(variant["views"]),
                        }
                    ),
                    flush=True,
                )
        torch.cuda.synchronize()
        summary = {
            "name": variant["name"],
            "observations": observations,
            "warmup": warmup,
            "decoded_source_pixels": decoded_pixels,
            "model_tensor_pixels": model_pixels,
            "wall_seconds": time.monotonic() - before,
            "peak_allocated_mib": torch.cuda.max_memory_allocated() / 2**20,
            "peak_reserved_mib": torch.cuda.max_memory_reserved() / 2**20,
            "source_frames": len(observations),
            "detections": sum(len(o["detections"]) for o in observations),
            "labels": dict(
                Counter(d["label"] for o in observations for d in o["detections"])
            ),
            "test_opened": False,
            "status": "PROPOSALS_REQUIRE_VISUAL_AND_GEOMETRY_REVIEW",
        }
        write_json(dest / "predictions.json", summary)
        canvas.save(dest / "overview.jpg", quality=90)
        results.append({k: v for k, v in summary.items() if k != "observations"})
    report = {
        "schema": "farm.discovery-ablation-results.v1",
        "plan_sha256": json_digest(plan),
        "model": describe_file(model_root / "yoloe/yoloe-v8l-seg-pf.pt"),
        "vocabulary": describe_file(vocabulary),
        "setup_seconds": setup_seconds,
        "setup_peak_allocated_mib": setup_peak_allocated_mib,
        "native_predictor_parity": parity,
        "variants": results,
        "peak_process_rss_kib": resource.getrusage(resource.RUSAGE_SELF).ru_maxrss,
        "test_opened": False,
        "manual_recall_measured": False,
    }
    write_json(output / "results.json", report)
    print(json.dumps(report), flush=True)


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--packet", type=Path, required=True)
    p.add_argument("--colmap", type=Path, required=True)
    p.add_argument("--images", type=Path, required=True)
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--model-root", type=Path, required=True)
    p.add_argument("--vocabulary", type=Path, required=True)
    p.add_argument("--sampling", choices=("pilot", "scene"), default="scene")
    p.add_argument("--timestamps", type=int, default=48)
    p.add_argument("--world-up", nargs=3, type=float, required=True)
    p.add_argument("--plan-only", action="store_true")
    p.add_argument(
        "--variant",
        action="append",
        choices=(
            "center_raw",
            "balanced_raw",
            "balanced_upright",
            "all_lowres_upright",
        ),
    )
    args = p.parse_args(argv)
    if args.output.exists() or args.timestamps < 2:
        raise ValueError("output must be new and timestamps >= 2")
    packet = json.loads(args.packet.read_text())
    plan = build_ablation(
        packet,
        args.colmap,
        sampling=args.sampling,
        timestamp_count=args.timestamps,
        image_root=args.images,
    )
    if args.variant:
        plan["variants"] = [v for v in plan["variants"] if v["name"] in args.variant]
    args.output.mkdir(parents=True)
    write_json(args.output / "plan.json", plan)
    if not args.plan_only:
        infer(
            plan,
            args.output,
            model_root=args.model_root,
            vocabulary=args.vocabulary,
            world_up=args.world_up,
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
