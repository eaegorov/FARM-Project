"""Bounded complementary YOLOE proposals on the primary detector's exact views."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import time

import numpy as np

from farm_runtime.angular_discovery import upright_quarter_turns
from farm_runtime.quality.proposal_geometry import read_observations
from farm_runtime.quality_baseline import describe_file, write_json
from farm_runtime.quality.mask_refinement import checked_file


def source_plan(primary, world_up):
    """Validate the development cohort before loading models or decoding RGB."""
    _, rows = read_observations(primary)
    if len(rows) > 48:
        raise ValueError("complementary discovery supports at most 48 existing views")
    up = np.asarray(world_up, float)
    if up.shape != (3,) or not np.isfinite(up).all() or np.linalg.norm(up) < 1e-8:
        raise ValueError("finite nonzero world-up required")
    if len({Path(r["name"]).stem for r in rows}) != len(rows):
        raise ValueError("ambiguous primary image basenames")
    sources = {}
    fields = (
        "name",
        "timestamp",
        "shape_hw",
        "source_image",
        "camera_from_world_rotation",
    )
    for row in rows:
        if not any(q["prompt"] == "person" for q in row["queries"]):
            raise ValueError("primary view needs explicit person exclusion query")
        checked_file(row["source_image"])
        shape = np.asarray(row["shape_hw"])
        rotation = np.asarray(row["camera_from_world_rotation"], float)
        if (
            shape.shape != (2,)
            or not np.issubdtype(shape.dtype, np.integer)
            or (shape <= 0).any()
            or rotation.shape != (3, 3)
            or not np.isfinite(rotation).all()
        ):
            raise ValueError("invalid primary source shape or rotation")
        grid = [round(int(v) * 640 / int(shape.max())) for v in shape]
        turns = upright_quarter_turns(rotation, up)["applied_quarter_turns_ccw"]
        if grid != row["grid_shape_hw"] or turns != row["applied_quarter_turns"]:
            raise ValueError("primary detector grid/orientation differs")
        sources[row["name"]] = {k: row[k] for k in fields}
    return dict(
        schema="farm.complementary-source-plan.v1",
        primary=describe_file(primary),
        sources=sources,
        variants=[
            dict(
                name="yoloe",
                views=list(sources),
                upright=True,
                resolution=640,
                allow_upscale=True,
            )
        ],
        test_opened=False,
    )


def run(primary, model_root, vocabulary, world_up, output):
    if output.exists():
        raise ValueError("new output required")
    started = time.monotonic()
    plan = source_plan(primary, world_up)
    vocabulary_descriptor = describe_file(vocabulary)
    weights = model_root / "yoloe" / "yoloe-v8l-seg-pf.pt"
    weights_descriptor = describe_file(weights)
    if plan["sources"]:
        # The existing backend resolves local weights through runtime_paths.
        # Refuse a mismatch instead of attributing another checkpoint to this run.
        from scene_graph.runtime_paths import find_model_file

        resolved = find_model_file(weights.name, "yoloe")
        if resolved is None or resolved.resolve() != weights.resolve():
            raise ValueError("runtime YOLOE checkpoint differs from --model-root")
    output.mkdir(parents=True)
    write_json(output / "plan.json", plan)
    dest = output / "yoloe"
    components = {}
    if plan["sources"]:
        from farm_runtime.quality.discovery import infer

        infer(
            plan,
            output,
            model_root=model_root,
            vocabulary=vocabulary,
            world_up=world_up,
        )
        components = json.loads((output / "results.json").read_text())[
            "model_components"
        ]
        predictions = json.loads((dest / "predictions.json").read_text())
        observations = predictions["observations"]
    else:
        dest.mkdir()
        (dest / "masks").mkdir()
        observations = []
    write_json(
        dest / "manifest.json",
        dict(
            schema="farm.complementary-discovery.v1",
            primary=describe_file(primary),
            plan=describe_file(output / "plan.json"),
            model_weights=weights_descriptor,
            model_components=components,
            vocabulary=vocabulary_descriptor,
            observations=observations,
            source_image_count=len(observations),
            timing_semantics="Includes source validation, setup, predictor parity and inference.",
            total_seconds=time.monotonic() - started,
            no_inference_reason=None if observations else "no_primary_views",
            person_exclusion_authority="primary detector only",
            scores_calibrated_against_primary=False,
            test_opened=False,
            release_eligible=False,
        ),
    )
    return dest / "manifest.json"


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("primary", "model-root", "vocabulary", "output"):
        parser.add_argument("--" + name, type=Path, required=True)
    parser.add_argument("--world-up", type=float, nargs=3, required=True)
    args = parser.parse_args(argv)
    run(args.primary, args.model_root, args.vocabulary, args.world_up, args.output)
    return 0
