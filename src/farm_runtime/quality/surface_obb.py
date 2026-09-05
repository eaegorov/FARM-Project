"""Compare observed FARM surface envelopes and fixed learned OBB proposals."""

from __future__ import annotations

import argparse
from collections import Counter
import json
from pathlib import Path
import time

import numpy as np
from PIL import Image, ImageDraw

from farm_runtime.angular_discovery import rotate_image
from farm_runtime.obb_proposals import (
    fit_surface_envelope,
    metric_box_corners,
    project_world,
    surface_coverage,
)
from farm_runtime.quality.mask_refinement import checked_file
from farm_runtime.quality_baseline import describe_file, write_json
from scripts.geometry.refine_farm_object_geometry import _fit_robust_obb


def support(nodes, clouds):
    counts = Counter(n["timestamp"] for n in nodes)
    points, weights = [], []
    for node in nodes:
        cloud = clouds[f"node_{node['id']:04d}"]
        points.append(cloud)
        weights.append(
            np.full(len(cloud), 1 / (counts[node["timestamp"]] * len(cloud)))
        )
    return np.concatenate(points), np.concatenate(weights)


def fit_choices(points, weights, predictions, up):
    # Existing FARM orientation baseline; use only its rotation. Its historical
    # 8 cm minimum size and display padding are not physical measurements.
    legacy = _fit_robust_obb(
        points, 0.005, orientation_mode="gravity_yaw", up_vector=up
    )
    baseline = fit_surface_envelope(
        points, weights, legacy["rotation_matrix"], "existing FARM gravity-PCA"
    )
    candidates = []
    for row in predictions:
        box = fit_surface_envelope(
            points,
            weights,
            row["rotation"],
            f"Boxer node {row['node_id']} orientation only",
        )
        candidates.append(
            dict(
                node_id=row["node_id"],
                box=box,
                volume=float(np.prod(box["dimensions_m"])),
            )
        )
    best = (
        min(candidates, key=lambda c: (c["volume"], c["node_id"]))
        if candidates
        else None
    )
    return baseline, best


def coverage(points, box):
    return surface_coverage(
        points, box["center_m"], box["rotation_world_from_box"], box["dimensions_m"]
    )


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("boxer", "output"):
        parser.add_argument("--" + name, type=Path, required=True)
    args = parser.parse_args(argv)
    if args.output.exists():
        raise ValueError("output must be new")
    started = time.monotonic()
    boxer = json.loads(args.boxer.read_text())
    if boxer.get("closed_test_opened") is not False:
        raise ValueError("development-only OBB predictions required")
    scope = json.loads(checked_file(boxer["source_evidence"]).read_text())
    geometry = json.loads(checked_file(scope["source_groups"]).read_text())
    observations = json.loads(
        checked_file(geometry["inputs"]["proposals"]).read_text()
    )["observations"]
    observations = {r["name"]: r for r in observations}
    frames = json.loads(checked_file(geometry["inputs"]["frames"]).read_text())[
        "frames"
    ]
    frames = {r["source_image"]: r for r in frames}
    with np.load(
        checked_file(geometry["surface_artifact"]), allow_pickle=False
    ) as data:
        clouds = {k: data[k] for k in data.files}
    nodes = {n["id"]: n for n in geometry["nodes"]}
    args.output.mkdir(parents=True)
    (args.output / "visuals").mkdir()
    groups = []
    rgb_cache = {}
    for group in scope["groups"]:
        members = [nodes[i] for i in group["members"]]
        predictions = [r for r in boxer["observations"] if r["group_id"] == group["id"]]
        if not predictions:
            continue
        points, weights = support(members, clouds)
        baseline, best = fit_choices(
            points, weights, predictions, boxer["source_world_up"]
        )
        raw_support = [
            dict(
                node_id=r["node_id"],
                model_score=r["model_score"],
                **surface_coverage(
                    points, r["center_m"], r["rotation"], r["dimensions_m"]
                ),
            )
            for r in predictions
        ]
        validation = []
        for timestamp in sorted({n["timestamp"] for n in members}):
            train = [n for n in members if n["timestamp"] != timestamp]
            held = [n for n in members if n["timestamp"] == timestamp]
            ptrain, wtrain = support(train, clouds)
            pheld, _ = support(held, clouds)
            ptr = [r for r in predictions if r["timestamp"] != timestamp]
            base_train, best_train = fit_choices(
                ptrain, wtrain, ptr, boxer["source_world_up"]
            )
            validation.append(
                dict(
                    timestamp=timestamp,
                    training_timestamps=sorted({n["timestamp"] for n in train}),
                    orientation_query_timestamps=sorted({r["timestamp"] for r in ptr}),
                    farm_gravity_pca=coverage(pheld, base_train),
                    boxer_orientation=(
                        coverage(pheld, best_train["box"]) if best_train else None
                    ),
                )
            )
        row = dict(
            group_id=group["id"],
            source_points=len(points),
            independent_timestamps=len({n["timestamp"] for n in members}),
            farm_gravity_pca=baseline,
            boxer_orientation=best,
            raw_boxer_surface_support=raw_support,
            leave_one_timestamp_out=validation,
            physical_extent_validated=False,
            native_gaussian_ownership_assigned=False,
        )
        groups.append(row)
        for evidence in [
            r for r in scope["observations"] if r["object_id"] == group["id"]
        ]:
            node = nodes[evidence["node_id"]]
            name = node["frame"]
            obs = observations[name]
            frame = frames[name]
            if name not in rgb_cache:
                with Image.open(checked_file(obs["source_image"])) as im:
                    rgb_cache[name] = im.convert("RGB").resize((960, 960))
            image = rgb_cache[name].copy()
            draw = ImageDraw.Draw(image)
            # Original RGB and metric K are in the same source orientation.
            K = np.asarray(frame["K"], float).copy()
            K[0] *= 960 / frame["depth_size"][1]
            K[1] *= 960 / frame["depth_size"][0]
            pose = np.asarray(frame["T_world_cam"], float)
            for box, color in [(baseline, "#ffb72e"), (best["box"], "#22ff88")]:
                corners = metric_box_corners(
                    box["center_m"], box["rotation_world_from_box"], box["dimensions_m"]
                )
                uv, z = project_world(corners, K, pose)
                for a in range(8):
                    for b in range(a + 1, 8):
                        if (a ^ b) in (1, 2, 4) and min(z[a], z[b]) > 0:
                            draw.line([tuple(uv[a]), tuple(uv[b])], fill=color, width=3)
            image = Image.fromarray(
                rotate_image(np.asarray(image), obs["applied_quarter_turns"])
            )
            ImageDraw.Draw(image).text(
                (8, 8),
                f"group {group['id']} | orange FARM PCA | green Boxer yaw + SAME observed support",
                fill="white",
            )
            path = (
                args.output
                / "visuals"
                / f"group_{group['id']:04d}_node_{node['id']:04d}.jpg"
            )
            image.save(path, quality=94)
            row.setdefault("visuals", []).append(describe_file(path))
    write_json(
        args.output / "manifest.json",
        dict(
            schema="farm.surface-obb-comparison.v1",
            boxer=describe_file(args.boxer),
            source_geometry=describe_file(checked_file(scope["source_groups"])),
            source_surface=geometry["surface_artifact"],
            groups=groups,
            total_seconds=time.monotonic() - started,
            interpretation="Observed partial surface envelopes; same-support consistency is by construction, not independent physical accuracy. Leave-one-timestamp-out excludes its points AND learned orientations from fitting.",
            candidate_selection="Minimum observed-support volume among available Boxer orientations; unlearned FARM PCA is a separate fixed baseline.",
            physical_extent_validated=False,
            closed_test_opened=False,
            release_eligible=False,
        ),
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
