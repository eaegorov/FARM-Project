"""Reproducible image-level detector comparison, independent of 3D assignment.

Use separate processes/runtimes for the legacy fork and current Ultralytics.
Prediction counts and model agreement are diagnostics, never ground-truth recall.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import time


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def write_json(path, value):
    Path(path).write_text(json.dumps(value, indent=2, ensure_ascii=False) + "\n")


def make_plan(scenes, vocabulary, captures=6):
    """Freeze evenly spaced capture groups, retaining every camera in each group."""
    from farm_runtime.frame_identity import normalize_capture_timestamps

    if captures < 2:
        raise ValueError("at least two capture groups required")
    rows = []
    seen_scenes = set()
    for scene in scenes:
        name = scene["name"]
        if name in seen_scenes:
            raise ValueError("duplicate scene name")
        seen_scenes.add(name)
        path = Path(scene["frames"]).resolve(strict=True)
        frames = normalize_capture_timestamps(json.loads(path.read_text())["frames"])
        groups = {}
        for frame in frames:
            groups.setdefault(frame["physical_timestamp"], []).append(frame)
        keys = sorted(groups, key=int)
        count = min(captures, len(keys))
        if count < 2:
            raise ValueError("scene has fewer than two captures")
        selected = [
            keys[round(i * (len(keys) - 1) / (count - 1))] for i in range(count)
        ]
        for key in selected:
            for frame in sorted(groups[key], key=lambda f: f["camera"]):
                rgb = (path.parent / frame["rgb_path"]).resolve(strict=True)
                rows.append(
                    dict(
                        key=f"{name}_{len(rows):03d}",
                        scene=name,
                        capture=key,
                        camera=frame["camera"],
                        rgb_path=str(rgb),
                        rgb_sha256=digest(rgb),
                        T_world_cam=frame["T_world_cam"],
                        world_up=scene["world_up"],
                        source_frames=str(path),
                        source_frames_sha256=digest(path),
                    )
                )
    return dict(
        schema="farm.detector-comparison-plan.v1",
        frames=rows,
        selection="evenly_spaced_captures_all_cameras_development_only",
        vocabulary_path=str(Path(vocabulary).resolve(strict=True)),
        vocabulary_sha256=digest(vocabulary),
        settings=dict(
            imgsz=640,
            conf=0.25,
            iou=0.5,
            max_det=200,
            agnostic_nms=True,
            retina_masks=True,
            upright=True,
        ),
        ground_truth=False,
        independent_heldout=False,
    )


def run(plan_path, output, *, mode, checkpoint=None):
    import numpy as np
    from PIL import Image
    import torch
    import ultralytics
    from farm_runtime.angular_discovery import upright_quarter_turns, rotate_image

    plan = json.loads(plan_path.read_text())
    if plan.get("schema") != "farm.detector-comparison-plan.v1":
        raise ValueError("unsupported comparison plan")
    vocabulary = Path(plan["vocabulary_path"])
    if digest(vocabulary) != plan["vocabulary_sha256"]:
        raise ValueError("vocabulary changed")
    names = [s.strip() for s in vocabulary.read_text().splitlines() if s.strip()]
    if not names:
        raise ValueError("empty vocabulary")
    for frame in plan["frames"]:
        if digest(frame["rgb_path"]) != frame["rgb_sha256"]:
            raise ValueError("source image changed")
    output.mkdir(parents=True, exist_ok=False)
    (output / "masks").mkdir()
    (output / "benchmark_source.py").write_bytes(Path(__file__).read_bytes())
    torch.manual_seed(20260909)
    torch.set_num_threads(4)
    settings = {k: v for k, v in plan["settings"].items() if k != "upright"}
    before = time.monotonic()
    if mode == "legacy-vocabulary":
        from scene_graph.segmentation.yoloe import YOLOESegmenter

        segmenter = YOLOESegmenter(vocab_file=vocabulary, conf_thres=settings["conf"])
        model = segmenter.model
        checkpoint = segmenter._base_ckpt
    else:
        from ultralytics import YOLOE

        model = YOLOE(str(checkpoint)).to("cuda:0")
        model.set_classes(names)
        # Compare the dense branch used for the paper's cross-generation AP.
        # This avoids silently comparing legacy NMS with YOLO26's default E2E.
        settings["nms"] = None
    model.eval()
    torch.cuda.synchronize()
    setup_seconds = time.monotonic() - before
    before = time.monotonic()
    model.predict(
        Image.new("RGB", (640, 640)), **settings, device=0, verbose=False, save=False
    )
    torch.cuda.synchronize()
    warmup_seconds = time.monotonic() - before
    torch.cuda.reset_peak_memory_stats()
    rows = []
    for frame in plan["frames"]:
        image = np.asarray(Image.open(frame["rgb_path"]).convert("RGB"))
        orientation = upright_quarter_turns(
            np.asarray(frame["T_world_cam"])[:3, :3].T, frame["world_up"]
        )
        turns = orientation["applied_quarter_turns_ccw"]
        image = rotate_image(image, turns)
        before = time.monotonic()
        pred = model.predict(
            Image.fromarray(image), **settings, device=0, verbose=False, save=False
        )[0]
        torch.cuda.synchronize()
        seconds = time.monotonic() - before
        masks = (
            np.zeros((0, *image.shape[:2]), dtype=bool)
            if pred.masks is None
            else pred.masks.data.cpu().numpy().astype(bool)
        )
        boxes = pred.boxes.data.cpu().numpy()
        if len(masks) != len(boxes) or masks.shape[1:] != image.shape[:2]:
            raise ValueError("native mask output is not aligned with source image")
        # Store masks in original camera pixels for later geometric evaluation.
        masks = (
            np.stack([rotate_image(m, -turns) for m in masks])
            if len(masks)
            else np.zeros(
                (0, *np.asarray(Image.open(frame["rgb_path"])).shape[:2]), dtype=bool
            )
        )
        np.savez_compressed(output / "masks" / f"{frame['key']}.npz", masks=masks)
        detections = []
        for i, (box, mask) in enumerate(zip(boxes, masks)):
            yy, xx = np.where(mask)
            detections.append(
                dict(
                    index=i,
                    label=pred.names[int(box[5])],
                    class_id=int(box[5]),
                    confidence=float(box[4]),
                    model_bbox_upright_xyxy=box[:4].tolist(),
                    mask_bbox_xyxy=(
                        None
                        if not len(xx)
                        else [
                            int(xx.min()),
                            int(yy.min()),
                            int(xx.max() + 1),
                            int(yy.max() + 1),
                        ]
                    ),
                    pixels=int(mask.sum()),
                )
            )
        row = dict(
            key=frame["key"],
            scene=frame["scene"],
            seconds=seconds,
            orientation=orientation,
            detections=detections,
            native_speed_ms=pred.speed,
        )
        rows.append(row)
        print(
            json.dumps(
                dict(
                    frame=row["key"],
                    detections=len(detections),
                    seconds=round(seconds, 3),
                )
            ),
            flush=True,
        )
    result = dict(
        schema="farm.detector-comparison-result.v1",
        plan_path=str(plan_path),
        plan_sha256=digest(plan_path),
        code_sha256=digest(__file__),
        code_path=str(output / "benchmark_source.py"),
        mode=mode,
        checkpoint=str(checkpoint),
        checkpoint_sha256=digest(checkpoint),
        ultralytics_version=ultralytics.__version__,
        torch_version=torch.__version__,
        ultralytics_path=ultralytics.__file__,
        settings=settings,
        setup_seconds=setup_seconds,
        warmup_seconds=warmup_seconds,
        inference_seconds=sum(r["seconds"] for r in rows),
        peak_allocated_mib=torch.cuda.max_memory_allocated() / 2**20,
        rows=rows,
        ground_truth=False,
        quality_accepted=False,
        legacy_note="Uses legacy detector weights/vocabulary with native predictor; downstream depth/erosion/association are excluded.",
    )
    write_json(output / "results.json", result)


def render_comparison(plan_path, runs, output):
    """Render every frozen frame, with source RGB and unfiltered predictions."""
    import colorsys
    import cv2
    import numpy as np
    from PIL import Image, ImageDraw, ImageFont
    from farm_runtime.angular_discovery import rotate_image

    plan = json.loads(plan_path.read_text())
    models = [json.loads((path / "results.json").read_text()) for path in runs]
    if len(models) != 3 or any(m["plan_sha256"] != digest(plan_path) for m in models):
        raise ValueError("three runs on the exact same frozen plan required")
    if any(
        [r["key"] for r in m["rows"]] != [r["key"] for r in plan["frames"]]
        for m in models
    ):
        raise ValueError("incomplete or differently ordered comparison cohort")
    font = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf", 19)
    title = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf", 25)
    output.mkdir(parents=True, exist_ok=False)
    for index, frame in enumerate(plan["frames"]):
        rgb = np.asarray(Image.open(frame["rgb_path"]).convert("RGB"))
        turns = models[0]["rows"][index]["orientation"]["applied_quarter_turns_ccw"]
        counts = [len(m["rows"][index]["detections"]) for m in models]
        legend_lines = (max(counts) + 1) // 2
        width, photo = 900, 864
        height = photo + 100 + legend_lines * 25
        canvas = Image.new("RGB", (2 * width, 2 * height), "#111827")
        draw = ImageDraw.Draw(canvas)
        for panel in range(4):
            x0, y0 = (panel % 2) * width, (panel // 2) * height
            array = rgb.copy()
            labels = []
            if panel:
                model = models[panel - 1]
                row = model["rows"][index]
                if row["orientation"]["applied_quarter_turns_ccw"] != turns:
                    raise ValueError("comparison orientation differs")
                masks = np.load(runs[panel - 1] / "masks" / f"{frame['key']}.npz")[
                    "masks"
                ]
                for d, mask in zip(row["detections"], masks):
                    hue = (
                        int(hashlib.sha256(d["label"].encode()).hexdigest()[:6], 16)
                        / 0xFFFFFF
                    )
                    color = tuple(
                        round(c * 255) for c in colorsys.hsv_to_rgb(hue, 0.70, 1)
                    )
                    array[mask] = np.rint(
                        0.60 * array[mask] + 0.40 * np.array(color)
                    ).astype(np.uint8)
                    contours, _ = cv2.findContours(
                        mask.astype(np.uint8),
                        cv2.RETR_EXTERNAL,
                        cv2.CHAIN_APPROX_SIMPLE,
                    )
                    cv2.drawContours(array, contours, -1, color, 2)
                    labels.append(
                        (color, f"{d['index']+1}: {d['label']} {d['confidence']:.2f}")
                    )
                label = f"{runs[panel-1].name} | {len(labels)} detections"
            else:
                label = f"Source RGB | {frame['key']}"
            array = rotate_image(array, turns)
            im = Image.fromarray(array)
            im.thumbnail((photo, photo))
            canvas.paste(im, (x0 + 18, y0 + 50))
            draw.text((x0 + 18, y0 + 13), label, font=title, fill="white")
            if panel:
                masks = [rotate_image(m, turns) for m in masks]
                for d, mask in zip(row["detections"], masks):
                    dist = cv2.distanceTransform(mask.astype(np.uint8), cv2.DIST_L2, 3)
                    yy, xx = np.unravel_index(dist.argmax(), dist.shape)
                    tx, ty = x0 + 18 + round(
                        xx * im.width / mask.shape[1]
                    ), y0 + 50 + round(yy * im.height / mask.shape[0])
                    draw.text(
                        (tx, ty),
                        str(d["index"] + 1),
                        font=font,
                        fill="white",
                        stroke_width=2,
                        stroke_fill="black",
                    )
            for j, (color, text) in enumerate(labels):
                draw.text(
                    (x0 + 18 + (j % 2) * 440, y0 + photo + 65 + (j // 2) * 25),
                    text,
                    font=font,
                    fill=color,
                )
            if not panel:
                draw.text(
                    (x0 + 18, y0 + photo + 65),
                    "Same source, vocabulary, 640 px input and threshold 0.25",
                    font=font,
                    fill="#CBD5E1",
                )
                draw.text(
                    (x0 + 18, y0 + photo + 90),
                    "Raw model proposals; labels and masks are not ground truth.",
                    font=font,
                    fill="#CBD5E1",
                )
        canvas.save(output / f"{frame['key']}.jpg", quality=90)
    write_json(
        output / "manifest.json",
        dict(
            plan_sha256=digest(plan_path),
            runs=[
                dict(path=str(p / "results.json"), sha256=digest(p / "results.json"))
                for p in runs
            ],
            code_sha256=digest(__file__),
            frames=len(plan["frames"]),
            all_proposals_shown=True,
        ),
    )


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="action", required=True)
    p = sub.add_parser("plan")
    p.add_argument("--scenes", type=Path, required=True)
    p.add_argument("--vocabulary", type=Path, required=True)
    p.add_argument("--captures", type=int, default=6)
    p.add_argument("--output", type=Path, required=True)
    p = sub.add_parser("run")
    p.add_argument("--plan", type=Path, required=True)
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--mode", choices=["legacy-vocabulary", "text"], required=True)
    p.add_argument("--checkpoint", type=Path)
    p = sub.add_parser("render")
    p.add_argument("--plan", type=Path, required=True)
    p.add_argument("--runs", type=Path, nargs=3, required=True)
    p.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.action == "plan":
        if args.output.exists():
            raise ValueError("refusing to overwrite a frozen comparison plan")
        args.output.parent.mkdir(parents=True, exist_ok=True)
        write_json(
            args.output,
            make_plan(
                json.loads(args.scenes.read_text()), args.vocabulary, args.captures
            ),
        )
    elif args.action == "render":
        render_comparison(args.plan, args.runs, args.output)
    else:
        if args.mode == "text" and args.checkpoint is None:
            parser.error("text mode requires --checkpoint")
        run(args.plan, args.output, mode=args.mode, checkpoint=args.checkpoint)


if __name__ == "__main__":
    main()
