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


def predictor_parity(
    model,
    plan,
    *,
    resolution=640,
    predictor_factory=soft_mask_predictor,
    confidence=0.4,
    extra_options=None,
    predict_call=None,
):
    """Compare the custom soft decoder to the pinned native YOLOE predictor."""
    import torch
    from ultralytics.models.yolo.segment.predict import SegmentationPredictor

    predict_call = predict_call or model.predict
    rows = []
    for name in plan["variants"][0]["views"][:2]:
        with Image.open(plan["sources"][name]["source_image"]["path"]) as im:
            image = im.convert("RGB")
            image.thumbnail((resolution, resolution))
        options = dict(
            source=image,
            imgsz=resolution,
            conf=confidence,
            iou=0.5,
            agnostic_nms=True,
            max_det=200,
            device=0,
            retina_masks=True,
            verbose=False,
            save=False,
        )
        options.update(extra_options or {})
        model.predictor = None
        native = predict_call(**options, predictor=SegmentationPredictor)[0]
        native_boxes = native.boxes.data.detach().clone()
        native_masks = (
            None if native.masks is None else native.masks.data.detach().clone().bool()
        )
        model.predictor = None
        soft = predict_call(**options, predictor=predictor_factory())[0]
        if not hasattr(soft, "farm_mask_logits"):
            raise ValueError("custom predictor was not used; soft logits missing")
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


def validate_inference_plan(plan):
    """Reject changed inputs or mixed evaluation roles before GPU initialization."""
    import re
    from farm_runtime.quality.mask_refinement import checked_file

    if plan.get("test_opened") is not False or not plan.get("variants"):
        raise ValueError("explicit nonempty development plan required")
    names = [v["name"] for v in plan["variants"]]
    if len(set(names)) != len(names) or any(
        not re.fullmatch(r"[a-z0-9_]+", n) for n in names
    ):
        raise ValueError("unique safe variant names required")
    seen = set()
    stems = set()
    for variant in plan["variants"]:
        views = variant["views"]
        if not views or len(set(views)) != len(views):
            raise ValueError("nonempty unique views per variant required")
        resolution = variant["resolution"]
        if type(resolution) is not int or not 32 <= resolution <= 2048:
            raise ValueError("resolution must be in 32..2048")
        for name in views:
            row = plan["sources"][name]
            if row.get("name") != name or not str(row.get("timestamp", "")):
                raise ValueError("source name and capture identity required")
            if name not in seen:
                stem = Path(name).stem
                if stem in stems:
                    raise ValueError("ambiguous source image basenames")
                stems.add(stem)
                checked_file(row["source_image"])
                seen.add(name)


def infer(
    plan, output, *, model_root, vocabulary, world_up, checkpoint=None, confidence=0.4
):
    import torch

    if plan.get("test_opened") is not False or not 0 < confidence < 1:
        raise ValueError("development plan and confidence in (0,1) required")
    validate_inference_plan(plan)
    torch.manual_seed(20260905)
    started = time.monotonic()
    extra_options = {}
    if checkpoint is None:
        from scene_graph.segmentation.yoloe import YOLOESegmenter
        from ultralytics.nn.text_model import _resolve_mobileclip_checkpoint

        segmenter = YOLOESegmenter(
            vocab_file=vocabulary, conf_thres=confidence, use_dino_features=False
        )
        model = segmenter.model
        model_descriptor = describe_file(model_root / "yoloe/yoloe-v8l-seg-pf.pt")
        components = {
            "vocabulary_head_checkpoint": describe_file(segmenter._base_ckpt),
            "text_encoder_checkpoint": describe_file(
                _resolve_mobileclip_checkpoint("blt")
            ),
        }
        predictor_factory = soft_mask_predictor
    else:
        from farm_runtime.quality.yoloe_modern import load_text_model
        from farm_runtime.quality.yoloe_modern import (
            soft_mask_predictor as modern_predictor,
        )

        model, components = load_text_model(checkpoint, vocabulary, model_root)
        model_descriptor = components["detector_checkpoint"]
        predictor_factory = modern_predictor
        # Explicitly use dense predictions + native NMS, matching the comparison.
        extra_options = {"nms": None}
    predict_call = model.predict
    if checkpoint is not None:
        from functools import partial
        from ultralytics.engine.model import Model

        # YOLOE.predict consumes the predictor argument as a visual-prompt
        # selector and does not forward it for text prompts in Ultralytics 8.4.
        # Dispatch through the ordinary model API to retain our soft decoder.
        predict_call = partial(Model.predict, model)
    model.predictor = None
    predictor = predictor_factory()
    torch.cuda.synchronize()
    setup_seconds = time.monotonic() - started
    setup_peak_allocated_mib = torch.cuda.max_memory_allocated() / 2**20
    parity = predictor_parity(
        model,
        plan,
        predictor_factory=predictor_factory,
        confidence=confidence,
        extra_options=extra_options,
        predict_call=predict_call,
    )
    results = []
    for variant in plan["variants"]:
        dest = output / variant["name"]
        dest.mkdir()
        (dest / "masks").mkdir()
        # First-call predictor/backend setup can allocate much more than steady
        # inference. Report it separately so variant order does not bias A/B.
        warm_started = time.monotonic()
        torch.cuda.reset_peak_memory_stats()
        predict_call(
            source=Image.new("RGB", (variant["resolution"], variant["resolution"])),
            imgsz=variant["resolution"],
            conf=confidence,
            iou=0.5,
            agnostic_nms=True,
            max_det=200,
            device=0,
            retina_masks=True,
            verbose=False,
            save=False,
            predictor=predictor,
            **extra_options,
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
            scale = size / max(rh, rw)
            if not variant.get("allow_upscale", False):
                scale = min(1.0, scale)
            small = np.asarray(
                Image.fromarray(rgb).resize(
                    (round(rw * scale), round(rh * scale)), Image.Resampling.BILINEAR
                )
            )
            model_pixels += size * size  # padded YOLO tensor, batch=1
            pred = predict_call(
                source=Image.fromarray(small),
                imgsz=size,
                conf=confidence,
                iou=0.5,
                agnostic_nms=True,
                max_det=200,
                device=0,
                retina_masks=True,
                verbose=False,
                save=False,
                predictor=predictor,
                **extra_options,
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
            "person_query_present": "person" in model.names.values(),
            "detections": sum(len(o["detections"]) for o in observations),
            "labels": dict(
                Counter(d["label"] for o in observations for d in o["detections"])
            ),
            "test_opened": False,
            "status": "PROPOSALS_REQUIRE_VISUAL_AND_GEOMETRY_REVIEW",
        }
        write_json(dest / "predictions.json", summary)
        if "person" in model.names.values():
            write_json(
                dest / "transients.json",
                dict(
                    summary,
                    observations=[
                        dict(
                            row,
                            queries=[{"prompt": "person"}],
                            detections=[
                                d for d in row["detections"] if d["label"] == "person"
                            ],
                        )
                        for row in observations
                    ],
                    role="person-only exclusion proposals from this detector; worn parts not merged",
                ),
            )
        canvas.save(dest / "overview.jpg", quality=90)
        results.append({k: v for k, v in summary.items() if k != "observations"})
    report = {
        "schema": "farm.discovery-ablation-results.v1",
        "plan_sha256": json_digest(plan),
        "model": model_descriptor,
        "model_components": components,
        "mode": "legacy-vocabulary" if checkpoint is None else "text",
        "confidence_threshold": confidence,
        "extra_predict_options": extra_options,
        "code": describe_file(Path(__file__)),
        "modern_adapter_code": (
            describe_file(Path(__file__).with_name("yoloe_modern.py"))
            if checkpoint is not None
            else None
        ),
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


def primary_profile(args):
    """Reuse the measured YOLOE decoder in the common scene pipeline."""
    from farm_runtime.quality.mask_refinement import checked_file

    source = json.loads(args.plan.read_text())
    if source.get("test_opened") is not False or not 1 <= args.views <= 24:
        raise ValueError("development plan and 1..24 primary views required")
    if not 0 < args.confidence < 1:
        raise ValueError("confidence must be in (0,1)")
    up = np.asarray(args.world_up, float)
    if up.shape != (3,) or not np.isfinite(up).all() or np.linalg.norm(up) < 1e-8:
        raise ValueError("finite nonzero world-up required")
    variants = [v for v in source["variants"] if v["name"] == "balanced_upright"]
    if len(variants) != 1:
        raise ValueError("one balanced_upright source variant required")
    names = variants[0]["views"]
    if len(names) != len(set(names)):
        raise ValueError("unique source views required")
    selected = (
        [
            names[i]
            for i in np.unique(
                np.linspace(0, len(names) - 1, min(args.views, len(names)))
                .round()
                .astype(int)
            )
        ]
        if names
        else []
    )
    prompts = [s.strip() for s in args.vocabulary.read_text().splitlines() if s.strip()]
    if "person" not in prompts or len(prompts) != len(set(prompts)):
        raise ValueError("unique primary vocabulary must include person")
    plan = dict(
        schema="farm.primary-yoloe-source-plan.v1",
        source_plan=describe_file(args.plan),
        sources={n: source["sources"][n] for n in selected},
        variants=[
            dict(
                name="balanced_upright",
                views=selected,
                upright=True,
                resolution=640,
                allow_upscale=True,
            )
        ],
        test_opened=False,
    )
    if selected:
        validate_inference_plan(plan)
    weights = args.checkpoint or args.model_root / "yoloe/yoloe-v8l-seg-pf.pt"
    text_weights = args.model_root / (
        "yoloe/mobileclip2_b.ts" if args.checkpoint else "mobileclip/mobileclip_blt.pt"
    )
    config = dict(
        detector=describe_file(weights),
        text_encoder=describe_file(text_weights),
        vocabulary=describe_file(args.vocabulary),
        confidence=args.confidence,
        mode="modern-text" if args.checkpoint else "legacy-vocabulary",
        resolution=640,
        allow_upscale=True,
        world_up=args.world_up,
        iou=0.5,
        agnostic_nms=True,
        max_det=200,
    )
    if args.checkpoint is None:
        config["vocabulary_head"] = describe_file(
            args.model_root / "yoloe/yoloe-v8l-seg.pt"
        )
        # The legacy loader uses environment lookup. Verify it before loading.
        from scene_graph.runtime_paths import find_model_file
        from ultralytics.nn.text_model import _resolve_mobileclip_checkpoint

        for item in (config["detector"], config["vocabulary_head"]):
            resolved = find_model_file(Path(item["path"]).name, "yoloe")
            if resolved is None or resolved.resolve() != Path(item["path"]).resolve():
                raise ValueError("runtime YOLOE checkpoint differs from model root")
        if (
            Path(_resolve_mobileclip_checkpoint("blt")).resolve()
            != text_weights.resolve()
        ):
            raise ValueError("runtime MobileCLIP checkpoint differs from model root")
    args.output.mkdir(parents=True)
    write_json(args.output / "plan.json", plan)
    write_json(args.output / "config.json", config)
    started = time.monotonic()
    observations = []
    result = None
    if selected:
        infer(
            plan,
            args.output,
            model_root=args.model_root,
            vocabulary=args.vocabulary,
            world_up=args.world_up,
            checkpoint=args.checkpoint,
            confidence=args.confidence,
        )
        result = json.loads((args.output / "results.json").read_text())
        prediction = json.loads(
            (args.output / "balanced_upright/predictions.json").read_text()
        )
        if prediction.get("person_query_present") is not True:
            raise ValueError("primary model did not run the required person query")
        actual_text = result["model_components"]["text_encoder_checkpoint"]
        if actual_text["sha256"] != config["text_encoder"]["sha256"]:
            raise ValueError(
                "loaded text encoder differs from compiled primary profile"
            )
        if result["model"]["sha256"] != config["detector"]["sha256"]:
            raise ValueError("loaded detector differs from compiled primary profile")
        if args.checkpoint is None and (
            result["model_components"]["vocabulary_head_checkpoint"]["sha256"]
            != config["vocabulary_head"]["sha256"]
        ):
            raise ValueError(
                "loaded vocabulary head differs from compiled primary profile"
            )
        for row in prediction["observations"]:
            checked_file(row["mask_artifact"])
            counts = Counter(d["label"] for d in row["detections"])
            if set(counts) - set(prompts):
                raise ValueError("detector labels differ from the compiled vocabulary")
            observations.append(
                dict(
                    row, queries=[dict(prompt=p, detections=counts[p]) for p in prompts]
                )
            )
    manifest = dict(
        schema="farm.primary-yoloe-discovery.v1",
        source_plan=describe_file(args.plan),
        plan=describe_file(args.output / "plan.json"),
        model_config=describe_file(args.output / "config.json"),
        model_weights=config["detector"],
        vocabulary=config["vocabulary"],
        observations=observations,
        source_image_encoder_calls=len(observations),
        model_components=result["model_components"] if result else None,
        total_seconds=time.monotonic() - started,
        no_inference_reason=None if selected else "no_additional_views",
        person_query_present=True,
        test_opened=False,
        release_eligible=False,
        labels_are_detector_hypotheses=True,
        visual_features_for_legacy_association_exported=False,
    )
    write_json(args.output / "manifest.json", manifest)
    write_json(
        args.output / "transients.json",
        dict(
            manifest,
            observations=[
                dict(
                    row,
                    queries=[q for q in row["queries"] if q["prompt"] == "person"],
                    detections=[d for d in row["detections"] if d["label"] == "person"],
                )
                for row in observations
            ],
            role="person-only exclusions",
        ),
    )
    return 0


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--plan", type=Path, help="Prepared development source plan")
    p.add_argument(
        "--primary-profile",
        action="store_true",
        help="Emit common pipeline proposals from bounded source views",
    )
    p.add_argument("--views", type=int, default=12)
    p.add_argument("--packet", type=Path)
    p.add_argument("--colmap", type=Path)
    p.add_argument("--images", type=Path)
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--model-root", type=Path, required=True)
    p.add_argument("--vocabulary", type=Path, required=True)
    p.add_argument(
        "--checkpoint",
        type=Path,
        help="Opt-in modern text checkpoint; separate Ultralytics runtime required",
    )
    p.add_argument("--confidence", type=float, default=0.4)
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
    if args.primary_profile:
        if (
            args.plan is None
            or args.plan_only
            or args.variant
            or any(x is not None for x in (args.packet, args.colmap, args.images))
        ):
            p.error("primary profile requires --plan and no ablation selectors")
        if args.output.exists():
            raise ValueError("new output required")
        return primary_profile(args)
    if args.output.exists() or args.timestamps < 2:
        raise ValueError("output must be new and timestamps >= 2")
    if args.plan is not None:
        if any(x is not None for x in (args.packet, args.colmap, args.images)):
            p.error("--plan cannot be combined with --packet/--colmap/--images")
        plan = json.loads(args.plan.read_text())
    else:
        if any(x is None for x in (args.packet, args.colmap, args.images)):
            p.error("provide --plan or all of --packet/--colmap/--images")
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
    validate_inference_plan(plan)
    args.output.mkdir(parents=True)
    write_json(args.output / "plan.json", plan)
    if not args.plan_only:
        infer(
            plan,
            args.output,
            model_root=args.model_root,
            vocabulary=args.vocabulary,
            world_up=args.world_up,
            checkpoint=args.checkpoint,
            confidence=args.confidence,
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
