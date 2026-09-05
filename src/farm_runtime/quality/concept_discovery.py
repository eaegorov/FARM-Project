"""Bounded SAM3 concept-discovery experiment with cached image/text encoders."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import time

import numpy as np
from PIL import Image, ImageDraw

from farm_runtime.angular_discovery import rotate_image, upright_quarter_turns
from farm_runtime.quality_baseline import describe_file, write_json


def main(argv=None):
    import torch
    from farm_runtime.concept_segmentation import load_sam3_detector
    from transformers.utils import logging

    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("plan", "vocabulary", "model", "output"):
        parser.add_argument("--" + name, type=Path, required=True)
    parser.add_argument("--views", type=int, default=12)
    parser.add_argument("--world-up", type=float, nargs=3, required=True)
    args = parser.parse_args(argv)
    if args.output.exists() or not 1 <= args.views <= 24:
        raise ValueError("new output and 1..24 views required")
    plan = json.loads(args.plan.read_text())
    if plan.get("test_opened") is not False:
        raise ValueError("development-only input plan required")
    names = next(
        v["views"] for v in plan["variants"] if v["name"] == "balanced_upright"
    )
    selected = [
        names[i]
        for i in np.unique(
            np.linspace(0, len(names) - 1, min(args.views, len(names)))
            .round()
            .astype(int)
        )
    ]
    prompts = list(
        dict.fromkeys(
            x.strip() for x in args.vocabulary.read_text().splitlines() if x.strip()
        )
    )
    if not prompts or len(prompts) > 80:
        raise ValueError("one to 80 concept prompts required")
    args.output.mkdir(parents=True)
    (args.output / "masks").mkdir()
    (args.output / "visuals").mkdir()
    logging.set_verbosity_error()
    logging.disable_progress_bar()
    started = time.monotonic()
    model, processor, loading = load_sam3_detector(args.model)
    unexpected = loading.get("unexpected_keys", [])
    torch.cuda.synchronize()
    load_seconds = time.monotonic() - started
    text_started = time.monotonic()
    text_cache = []
    with torch.inference_mode():
        for prompt in prompts:
            inputs = processor(text=prompt, return_tensors="pt").to("cuda")
            embedding = model.get_text_features(**inputs)
            text_cache.append((embedding, inputs.attention_mask))
    torch.cuda.synchronize()
    text_seconds = time.monotonic() - text_started
    observations = []
    for name in selected:
        row = plan["sources"][name]
        path = Path(row["source_image"]["path"])
        if describe_file(path)["sha256"] != row["source_image"]["sha256"]:
            raise ValueError("source RGB changed")
        orientation = upright_quarter_turns(
            np.asarray(row["camera_from_world_rotation"]), args.world_up
        )
        turns = orientation["applied_quarter_turns_ccw"]
        with Image.open(path) as im:
            original = np.asarray(im.convert("RGB"))
        upright = Image.fromarray(rotate_image(original, turns))
        scale = 640 / max(original.shape[:2])
        grid = (round(upright.height * scale), round(upright.width * scale))
        frame_started = time.monotonic()
        images = processor(images=upright, return_tensors="pt").to("cuda")
        with torch.inference_mode():
            vision = model.get_vision_features(pixel_values=images.pixel_values)
        torch.cuda.synchronize()
        encoder_seconds = time.monotonic() - frame_started
        arrays, detections, query_rows = {}, [], []
        for prompt_index, (prompt, (embedding, attention)) in enumerate(
            zip(prompts, text_cache)
        ):
            before = time.monotonic()
            with torch.inference_mode():
                output = model(
                    vision_embeds=vision,
                    text_embeds=embedding,
                    attention_mask=attention,
                )
                scores = output.pred_logits[0].sigmoid()
                if output.presence_logits is not None:
                    scores = scores * output.presence_logits[0].sigmoid()
                keep = scores > 0.5
                probabilities = output.pred_masks[0, keep].sigmoid()
                if len(probabilities):
                    probabilities = torch.nn.functional.interpolate(
                        probabilities[None],
                        size=grid,
                        mode="bilinear",
                        align_corners=False,
                    )[0]
                # Check our retained soft probabilities against the installed
                # official processor before using them as discovery evidence.
                if prompt_index == 0:
                    native = processor.post_process_instance_segmentation(
                        output, threshold=0.5, mask_threshold=0.5, target_sizes=[grid]
                    )[0]
                    if not torch.equal(probabilities > 0.5, native["masks"].bool()):
                        raise ValueError(
                            "soft SAM3 masks differ from native postprocessor"
                        )
            torch.cuda.synchronize()
            query_rows.append(
                dict(
                    prompt=prompt,
                    detections=int(keep.sum()),
                    seconds=time.monotonic() - before,
                )
            )
            for probability, score in zip(probabilities, scores[keep]):
                probability = rotate_image(probability.cpu().numpy(), -turns)
                yy, xx = np.where(probability > 0.5)
                if not len(xx):
                    continue
                x0, y0 = max(0, int(xx.min()) - 4), max(0, int(yy.min()) - 4)
                x1, y1 = min(probability.shape[1], int(xx.max()) + 5), min(
                    probability.shape[0], int(yy.max()) + 5
                )
                tile = np.clip(probability[y0:y1, x0:x1], 1e-6, 1 - 1e-6)
                key = f"logits_{len(detections):04d}"
                arrays[key] = np.log(tile / (1 - tile)).astype(np.float16)
                detections.append(
                    dict(
                        label=prompt,
                        score=float(score),
                        logit_key=key,
                        grid_window_xyxy=[x0, y0, x1, y1],
                        positive_grid_pixels=len(xx),
                    )
                )
            del output, probabilities
        dest = args.output / "masks" / (Path(name).stem + ".npz")
        np.savez_compressed(dest, **arrays)
        # One panel per detected concept retains competing scopes visibly.
        active = [q for q in query_rows if q["detections"]]
        grid_source = Image.fromarray(original).resize(
            (round(original.shape[1] * scale), round(original.shape[0] * scale)),
            Image.Resampling.BILINEAR,
        )
        for offset in range(0, len(active), 6):
            chunk = active[offset : offset + 6]
            sheet = Image.new("RGB", (960, 350 * ((len(chunk) + 2) // 3)), "#111827")
            draw = ImageDraw.Draw(sheet)
            for index, query in enumerate(chunk):
                rgb = np.array(grid_source)
                for detection in detections:
                    if detection["label"] != query["prompt"]:
                        continue
                    x0, y0, x1, y1 = detection["grid_window_xyxy"]
                    m = arrays[detection["logit_key"]] > 0
                    rgb[y0:y1, x0:x1][m] = (
                        rgb[y0:y1, x0:x1][m] * 0.6 + np.array([30, 210, 255]) * 0.4
                    ).astype(np.uint8)
                tile = Image.fromarray(rotate_image(rgb, turns))
                tile.thumbnail((312, 312))
                x, y = (index % 3) * 320, (index // 3) * 350
                sheet.paste(tile, (x + (312 - tile.width) // 2, y + 30))
                draw.text(
                    (x + 5, y + 5),
                    f"{query['prompt']} ({query['detections']})",
                    fill="white",
                )
            sheet.save(
                args.output / "visuals" / f"{Path(name).stem}_{offset//6:02d}.jpg",
                quality=94,
            )
        observation = dict(
            **row,
            orientation=orientation,
            applied_quarter_turns=turns,
            grid_shape_hw=list(np.asarray(grid_source).shape[:2]),
            detections=detections,
            mask_artifact=describe_file(dest),
            queries=query_rows,
            image_encoder_seconds=encoder_seconds,
            frame_seconds=time.monotonic() - frame_started,
        )
        observations.append(observation)
        write_json(args.output / (Path(name).stem + ".json"), observation)
        print(
            json.dumps(
                dict(
                    source=name,
                    detections=len(detections),
                    concepts=len(active),
                    seconds=observation["frame_seconds"],
                )
            ),
            flush=True,
        )
        del vision, images
    write_json(
        args.output / "manifest.json",
        dict(
            schema="farm.concept-discovery.v1",
            observations=observations,
            plan=describe_file(args.plan),
            vocabulary=describe_file(args.vocabulary),
            model_config=describe_file(args.model / "config.json"),
            model_path=str(args.model),
            model_weights=describe_file(args.model / "model.safetensors"),
            detector_weights_complete=True,
            ignored_tracker_tensors=len(unexpected),
            source_image_encoder_calls=len(selected),
            text_encoder_calls=len(prompts),
            load_seconds=load_seconds,
            text_seconds=text_seconds,
            total_seconds=time.monotonic() - started,
            peak_allocated_mib=torch.cuda.max_memory_allocated() / 2**20,
            dtype="float32",
            model_resolution=1008,
            mask_export_long_side=640,
            confidence_threshold=0.5,
            mask_threshold=0.5,
            cross_prompt_duplicates_retained=True,
            counts_are_not_recall=True,
            test_opened=False,
            release_eligible=False,
        ),
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
