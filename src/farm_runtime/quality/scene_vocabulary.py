"""Build a scene vocabulary from distributed RGB, independent of object labels."""

from __future__ import annotations

import argparse
from collections import Counter
import json
from pathlib import Path
import time

import numpy as np
from PIL import Image

from farm_runtime.angular_discovery import upright_quarter_turns, rotate_image
from farm_runtime.quality_baseline import describe_file, write_json
from farm_runtime.semantic_refinement import LocalObjectReviewer

PROMPT = """Inspect these photographs of one real scene. For each image, list the categories of physical objects actually visible. Use concise singular English noun phrases, including small objects when recognizable. Use a broad physical category when exact function cannot be identified. Do not invent objects typical of this environment but absent from the image. A partially occluded or image-border object still counts if it is recognizable.
Separate movable or installed object instances (equipment, enclosures, furniture, signs, containers, tools and fixtures) from large structural surfaces, and from transient people/animals. Count a whole sign as an object, not each symbol printed on it. Do not list brand names, colors alone, abstractions, or multiple synonyms for the same category. Do not split ordinary integral parts into standalone objects.
Return only JSON:
{"views":[{"image_id":123,"objects":["object category"],"structures":["structural category"],"transient":["transient category"]}]}.
At most 25 object categories per image. Image IDs are identifiers, not category hints.
"""


def validate_vocabulary(text, ids):
    text = text.strip()
    if text.startswith("```"):
        text = text.split("\n", 1)[1].rsplit("```", 1)[0].strip()
    result = json.loads(text)
    rows = result.get("views")
    if (
        not isinstance(rows, list)
        or len(rows) != len(ids)
        or {r.get("image_id") for r in rows} != set(ids)
    ):
        raise ValueError("wrong vocabulary image IDs")
    for row in rows:
        for field in ("objects", "structures", "transient"):
            terms = row.get(field)
            if (
                not isinstance(terms, list)
                or len(terms) > 25
                or not all(
                    isinstance(x, str) and 0 < len(x.strip()) <= 80 for x in terms
                )
            ):
                raise ValueError("invalid category list")
            row[field] = sorted({" ".join(x.lower().split()) for x in terms})
    return result


def main(argv=None):
    import torch

    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("plan", "model", "output"):
        parser.add_argument("--" + name, type=Path, required=True)
    parser.add_argument("--views", type=int, default=12)
    parser.add_argument("--world-up", type=float, nargs=3, required=True)
    args = parser.parse_args(argv)
    if args.output.exists() or not 2 <= args.views <= 24:
        raise ValueError("new output and 2..24 context views required")
    plan = json.loads(args.plan.read_text())
    if plan.get("test_opened") is not False:
        raise ValueError("development-only discovery plan required")
    source_names = next(
        v["views"] for v in plan["variants"] if v["name"] == "balanced_upright"
    )
    positions = np.unique(
        np.linspace(0, len(source_names) - 1, min(args.views, len(source_names)))
        .round()
        .astype(int)
    )
    args.output.mkdir(parents=True)
    (args.output / "rgb").mkdir()
    (args.output / "prompt.txt").write_text(PROMPT)
    rows = []
    for i in positions:
        row = plan["sources"][source_names[i]]
        path = Path(row["source_image"]["path"])
        if describe_file(path)["sha256"] != row["source_image"]["sha256"]:
            raise ValueError("RGB changed since plan")
        turns = upright_quarter_turns(
            np.asarray(row["camera_from_world_rotation"]), args.world_up
        )["applied_quarter_turns_ccw"]
        with Image.open(path) as im:
            image = Image.fromarray(rotate_image(np.asarray(im.convert("RGB")), turns))
            image.thumbnail((896, 896))
        dest = args.output / "rgb" / (Path(source_names[i]).stem + ".jpg")
        image.save(dest, quality=94)
        rows.append(
            dict(
                image_id=int(i),
                source=row["source_image"],
                image=describe_file(dest),
                turns=turns,
            )
        )
    started = time.monotonic()
    model = LocalObjectReviewer(args.model)
    load_seconds = time.monotonic() - started
    results = []
    for offset in range(0, len(rows), 2):
        chunk = rows[offset : offset + 2]
        images = [Image.open(r["image"]["path"]).convert("RGB") for r in chunk]
        response = model.ask(
            PROMPT,
            images,
            [r["image_id"] for r in chunk],
            validate_vocabulary,
            max_new_tokens=700,
        )
        results.append(dict(inputs=chunk, **response))
        write_json(args.output / f"batch_{offset//2:02d}.json", results[-1])
        print(
            json.dumps(
                dict(
                    batch=offset // 2,
                    parsed=response["parsed"],
                    seconds=response["seconds"],
                )
            ),
            flush=True,
        )
    counts = {field: Counter() for field in ("objects", "structures", "transient")}
    for result in results:
        if result["parsed"]:
            for row in result["parsed"]["views"]:
                for field in counts:
                    counts[field].update(row[field])
    if not counts["objects"]:
        raise ValueError("VLM produced no valid object vocabulary")
    vocabulary = args.output / "object_vocabulary.txt"
    vocabulary.write_text("\n".join(sorted(counts["objects"])) + "\n")
    write_json(
        args.output / "manifest.json",
        dict(
            schema="farm.scene-vocabulary.v1",
            batches=results,
            categories={key: dict(value) for key, value in counts.items()},
            vocabulary=describe_file(vocabulary),
            plan=describe_file(args.plan),
            model_config=describe_file(args.model / "config.json"),
            prompt=describe_file(args.output / "prompt.txt"),
            load_seconds=load_seconds,
            total_seconds=time.monotonic() - started,
            peak_allocated_mib=torch.cuda.max_memory_allocated() / 2**20,
            world_up=args.world_up,
            failed_batches=sum(r["parsed"] is None for r in results),
            known_object_labels_in_prompt=False,
            reserved_test_opened=False,
            release_eligible=False,
        ),
    )
