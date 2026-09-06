"""Compose the measured quality stages on registered metric RGBD input.

GPU work stays in the configured inference runtimes. The existing FARM run
orchestrator owns execution, snapshots, timing and failure reporting.
"""

from __future__ import annotations

import argparse
import copy
import json
from pathlib import Path
import shutil

import numpy as np
from PIL import Image

from farm_runtime.quality.mask_refinement import checked_file
from farm_runtime.quality_baseline import describe_file, write_json


def read(path):
    return json.loads(Path(path).read_text())


def bounded_groups(geometry, budget):
    if type(budget) is not int or not 1 <= budget <= 128:
        raise ValueError("group budget must be in 1..128")
    if geometry.get("test_opened") is not False:
        raise ValueError("development geometry required")
    groups = [g for g in geometry["groups"] if g["independent_timestamps"] >= 2]
    ordered = sorted(groups, key=lambda g: (-g["independent_timestamps"], g["id"]))
    return [g["id"] for g in ordered[:budget]], [g["id"] for g in ordered[budget:]]


def prepare_inputs(rgbd, output):
    index, prep = read(rgbd / "frames.json"), read(rgbd / "prep_summary.json")
    if (
        prep.get("status") != "complete"
        or index.get("depth_units") != "metres"
        or index.get("pose_translation_units") != "metres"
    ):
        raise ValueError("completed metric registered RGBD required")
    trusted = {
        n
        for g in index["camera_registration"]["groups"]
        if g["trusted"]
        for n in g["source_images"]
    }
    sources = {}
    for frame in index["frames"]:
        name = frame["source_image"]
        if name not in trusted:
            continue
        if name in sources:
            raise ValueError("unique registered source images required")
        path = Path(prep["inputs"]["image_root"]) / name
        with Image.open(path) as image:
            shape = [image.height, image.width]
        sources[name] = dict(
            name=name,
            timestamp=str(frame["frame_id"]),
            camera=frame["camera"],
            shape_hw=shape,
            source_image=describe_file(path),
            camera_from_world_rotation=np.asarray(frame["T_world_cam"])[
                :3, :3
            ].T.tolist(),
        )
    if len({r["timestamp"] for r in sources.values()}) < 2:
        raise ValueError("two trusted independent timestamps required")
    output.mkdir(parents=True)
    # Same physical-timestamp order as registration, interleaving sibling views.
    write_json(
        output / "plan.json",
        dict(
            schema="farm.quality-source-plan.v1",
            scene_id=index["scene_id"],
            sources=sources,
            variants=[dict(name="balanced_upright", views=list(sources))],
            timestamps=sorted({r["timestamp"] for r in sources.values()}),
            frames=describe_file(rgbd / "frames.json"),
            prep_summary=describe_file(rgbd / "prep_summary.json"),
            test_opened=False,
        ),
    )


def union_vocabulary(scene, core, output):
    document = read(scene)
    if document.get("reserved_test_opened") is not False:
        raise ValueError("development scene vocabulary required")
    terms = list(
        dict.fromkeys(
            x.strip().lower() for x in core.read_text().splitlines() if x.strip()
        )
    )
    if "person" not in terms or len(terms) > 80:
        raise ValueError("bounded core vocabulary must include person")
    counts = document["categories"]["objects"]
    extras = sorted(
        (x for x in counts if x not in terms), key=lambda x: (-counts[x], x)
    )
    omitted = extras[80 - len(terms) :]
    terms = sorted(terms + extras[: 80 - len(terms)])
    output.mkdir(parents=True)
    path = output / "vocabulary.txt"
    path.write_text("\n".join(terms) + "\n")
    write_json(
        output / "manifest.json",
        dict(
            schema="farm.quality-vocabulary.v1",
            scene=describe_file(scene),
            core=describe_file(core),
            vocabulary=describe_file(path),
            omitted_scene_terms=omitted,
            terms=len(terms),
            test_opened=False,
            interpretation="Core categories are queries, never assertions of presence.",
        ),
    )


def merge_proposals(paths, output):
    documents = [read(p) for p in paths]
    if not documents or any(d.get("test_opened") is not False for d in documents):
        raise ValueError("development proposal batches required")
    for field in ("vocabulary", "model_config", "model_weights"):
        if len({d[field]["sha256"] for d in documents}) != 1:
            raise ValueError("proposal model/vocabulary differs across batches")
    rows, names = [], set()
    output.mkdir(parents=True)
    (output / "masks").mkdir()
    for path, doc in zip(paths, documents):
        for original in doc["observations"]:
            if original["name"] in names:
                raise ValueError("proposal batches must have disjoint source images")
            if not any(q["prompt"] == "person" for q in original["queries"]):
                raise ValueError("every view needs an explicit person query")
            names.add(original["name"])
            checked_file(original["source_image"])
            source = checked_file(original["mask_artifact"])
            target = output / "masks" / source.name
            if target.exists():
                raise ValueError("mask artifact filename collision")
            shutil.copyfile(source, target)
            rows.append(dict(original, mask_artifact=describe_file(target)))
    combined = dict(
        documents[0],
        observations=rows,
        source_manifests=[describe_file(p) for p in paths],
        source_image_encoder_calls=len(rows),
        total_seconds=sum(d["total_seconds"] for d in documents),
        timing_semantics="Sum of recorded segmentation batches; includes their model loads.",
    )
    write_json(output / "manifest.json", combined)
    transients = copy.deepcopy(combined)
    for row in transients["observations"]:
        row["detections"] = [d for d in row["detections"] if d["label"] == "person"]
        row["queries"] = [q for q in row["queries"] if q["prompt"] == "person"]
    transients["role"] = "person-only exclusions"
    write_json(output / "transients.json", transients)


def segment(args):
    from farm_runtime.quality import concept_discovery

    plan = read(args.plan)
    names = next(
        v["views"] for v in plan["variants"] if v["name"] == "balanced_upright"
    )
    if names:
        return concept_discovery.main(
            [
                "--plan",
                str(args.plan),
                "--vocabulary",
                str(args.vocabulary),
                "--model",
                str(args.model),
                "--views",
                str(args.views),
                "--world-up",
                *map(str, args.world_up),
                "--output",
                str(args.output),
            ]
        )
    # An exhausted coverage schedule must not load a GPU model or repeat old views.
    if plan.get("test_opened") is not False:
        raise ValueError("development-only source plan required")
    args.output.mkdir(parents=True)
    (args.output / "masks").mkdir()
    write_json(
        args.output / "manifest.json",
        dict(
            schema="farm.concept-discovery.v1",
            observations=[],
            plan=describe_file(args.plan),
            vocabulary=describe_file(args.vocabulary),
            model_config=describe_file(args.model / "config.json"),
            model_weights=describe_file(args.model / "model.safetensors"),
            model_path=str(args.model),
            source_image_encoder_calls=0,
            text_encoder_calls=0,
            total_seconds=0,
            no_inference_reason="no_additional_views",
            test_opened=False,
            release_eligible=False,
        ),
    )


def native(args):
    from farm_runtime.quality import native_observations

    chosen, deferred = bounded_groups(read(args.geometry), args.groups)
    if not chosen:
        raise ValueError(
            "no multi-timestamp groups; native quality output is unavailable"
        )
    native_observations.main(
        [
            "--geometry",
            str(args.geometry),
            "--ply",
            str(args.ply),
            "--config",
            str(args.config),
            "--world-up",
            *map(str, args.world_up),
            "--scope-alternative-budget",
            str(args.alternatives),
            "--mode",
            "exclusions_on",
            "--output",
            str(args.output),
            *[x for g in chosen for x in ("--group-id", str(g))],
        ]
    )
    write_json(
        args.output / "selection.json",
        dict(
            selected_group_ids=chosen,
            deferred_group_ids=deferred,
            source_geometry=describe_file(args.geometry),
            policy="Independent timestamps descending, stable group ID tie break",
            candidate_budget=args.groups,
            scene_completeness_claimed=False,
        ),
    )


def semantics(args):
    from farm_runtime.quality import scope_evidence, refinement

    chosen, deferred = bounded_groups(read(args.geometry), args.groups)
    if not chosen:
        raise ValueError("no multi-timestamp appearance candidates")
    args.output.mkdir(parents=True)
    scope_evidence.main(
        [
            "--groups",
            str(args.geometry),
            "--views",
            "2",
            "--output",
            str(args.output / "evidence"),
            *[x for g in chosen for x in ("--group-id", str(g))],
        ]
    )
    refinement.main(
        [
            "vlm",
            "--proposals",
            str(args.output / "evidence/manifest.json"),
            "--model",
            str(args.model),
            "--compact-semantics",
            "--views",
            "2",
            "--output",
            str(args.output / "appearance"),
        ]
    )
    write_json(
        args.output / "manifest.json",
        dict(
            schema="farm.quality-appearance-stage.v1",
            appearance=describe_file(args.output / "appearance/manifest.json"),
            source_geometry=describe_file(args.geometry),
            selected_group_ids=chosen,
            deferred_group_ids=deferred,
            candidate_budget=args.groups,
            labels_are_model_proposals=True,
            physical_scope_assessed=False,
        ),
    )


def compile_plan(args):
    """Return a normal FARM DAG; no additional execution framework."""
    budgets = dict(
        initial=args.initial_views,
        adaptive=args.adaptive_views,
        vocabulary=args.vocabulary_views,
        native=args.native_groups,
        semantics=args.semantic_groups,
        alternatives=args.alternatives,
    )
    for key, maximum in dict(
        initial=24,
        adaptive=24,
        vocabulary=24,
        native=128,
        semantics=128,
        alternatives=16,
    ).items():
        if type(budgets[key]) is not int or not 1 <= budgets[key] <= maximum:
            raise ValueError(f"invalid {key} budget")
    if budgets["vocabulary"] < 2:
        raise ValueError("vocabulary needs at least two views")
    up = np.asarray(args.world_up, float)
    if up.shape != (3,) or not np.isfinite(up).all() or np.linalg.norm(up) < 1e-8:
        raise ValueError("finite nonzero world-up required")
    runtimes = (
        read(args.runtimes)
        if args.runtimes
        else {"main": ["${python_executable}"], "geometry": ["${python_executable}"]}
    )
    if set(runtimes) != {"main", "geometry"} or any(
        not isinstance(v, list)
        or not v
        or any(not isinstance(x, str) or not x for x in v)
        for v in runtimes.values()
    ):
        raise ValueError("main and geometry Python command prefixes required")
    root = "${execution_project_root}"
    q = "${run_dir}/quality"
    stages = []
    up_args = ["--world-up", *map(str, args.world_up)]

    def add(name, action, arguments, output, inputs, runtime="main"):
        previous = stages[-1]["id"] if stages else None
        stages.append(
            dict(
                id=name,
                needs=[previous] if previous else [],
                command=[
                    *runtimes[runtime],
                    "-m",
                    "farm_runtime.cli",
                    "quality",
                    action,
                    *map(str, arguments),
                ],
                cwd="${project_root}",
                env={
                    "PYTHONDONTWRITEBYTECODE": "1",
                    "PYTHONPATH": f"{root}/src:{root}",
                },
                inputs=list(map(str, inputs)),
                outputs=[output],
                fingerprint_inputs=[
                    *map(str, inputs),
                    "${project_root}/src/farm_runtime/quality",
                    "${project_root}/configs/quality",
                ],
                fingerprint_mode="content",
                timeout_seconds=900,
            )
        )

    add(
        "quality_inputs",
        "scene-profile",
        ["inputs", "--rgbd", args.rgbd, "--output", f"{q}/input"],
        f"{q}/input/plan.json",
        [args.rgbd / "frames.json", args.rgbd / "prep_summary.json"],
    )
    add(
        "scene_vocabulary",
        "scene-vocabulary",
        [
            "--plan",
            f"{q}/input/plan.json",
            "--model",
            args.vlm_model,
            "--views",
            args.vocabulary_views,
            *up_args,
            "--output",
            f"{q}/scene_vocabulary",
        ],
        f"{q}/scene_vocabulary/manifest.json",
        [f"{q}/input/plan.json"],
    )
    add(
        "union_vocabulary",
        "scene-profile",
        [
            "vocabulary",
            "--scene",
            f"{q}/scene_vocabulary/manifest.json",
            "--core",
            f"{root}/configs/quality/core_vocabulary.txt",
            "--output",
            f"{q}/vocabulary",
        ],
        f"{q}/vocabulary/manifest.json",
        [f"{q}/scene_vocabulary/manifest.json"],
    )

    def sam(name, plan, views):
        add(
            name,
            "scene-profile",
            [
                "segment",
                "--plan",
                plan,
                "--vocabulary",
                f"{q}/vocabulary/vocabulary.txt",
                "--model",
                args.sam_model,
                "--views",
                views,
                *up_args,
                "--output",
                f"{q}/{name}",
            ],
            f"{q}/{name}/manifest.json",
            [plan, f"{q}/vocabulary/vocabulary.txt"],
        )

    def merge(name, sources):
        add(
            name,
            "scene-profile",
            [
                "merge",
                *[v for s in sources for v in ("--source", s)],
                "--output",
                f"{q}/{name}",
            ],
            f"{q}/{name}/manifest.json",
            sources,
        )

    def geometry(name, proposals):
        add(
            name,
            "proposal-geometry",
            [
                "--proposals",
                f"{q}/{proposals}/manifest.json",
                "--rgbd",
                args.rgbd,
                "--transients",
                f"{q}/{proposals}/transients.json",
                "--output",
                f"{q}/{name}",
            ],
            f"{q}/{name}/manifest.json",
            [f"{q}/{proposals}/manifest.json", f"{q}/{proposals}/transients.json"],
        )

    sam("initial_segmentation", f"{q}/input/plan.json", args.initial_views)
    merge("initial_proposals", [f"{q}/initial_segmentation/manifest.json"])
    geometry("initial_geometry", "initial_proposals")
    add(
        "coverage",
        "discovery-coverage",
        [
            "--geometry",
            f"{q}/initial_geometry/manifest.json",
            "--plan",
            f"{q}/input/plan.json",
            "--views",
            args.adaptive_views,
            "--output",
            f"{q}/coverage",
        ],
        f"{q}/coverage/plan.json",
        [f"{q}/initial_geometry/manifest.json", f"{q}/input/plan.json"],
    )
    sam("adaptive_segmentation", f"{q}/coverage/plan.json", args.adaptive_views)
    merge(
        "combined_proposals",
        [
            f"{q}/initial_segmentation/manifest.json",
            f"{q}/adaptive_segmentation/manifest.json",
        ],
    )
    geometry("geometry", "combined_proposals")
    add(
        "native",
        "scene-profile",
        [
            "native",
            "--geometry",
            f"{q}/geometry/manifest.json",
            "--ply",
            args.ply,
            "--config",
            f"{root}/configs/quality/native_rendered_v1.json",
            "--groups",
            args.native_groups,
            "--alternatives",
            args.alternatives,
            *up_args,
            "--output",
            f"{q}/native",
        ],
        f"{q}/native/manifest.json",
        [f"{q}/geometry/manifest.json", args.ply],
        "geometry",
    )
    add(
        "appearance",
        "scene-profile",
        [
            "semantics",
            "--geometry",
            f"{q}/geometry/manifest.json",
            "--model",
            args.vlm_model,
            "--groups",
            args.semantic_groups,
            "--output",
            f"{q}/semantics",
        ],
        f"{q}/semantics/manifest.json",
        [f"{q}/geometry/manifest.json"],
    )
    add(
        "catalog",
        "scene-catalog",
        [
            "--native",
            f"{q}/native/manifest.json",
            "--semantics",
            f"{q}/semantics/manifest.json",
            "--output",
            f"{q}/catalog",
        ],
        f"{q}/catalog/catalog.json",
        [f"{q}/native/manifest.json", f"{q}/semantics/manifest.json"],
    )
    return dict(
        schema_version=1,
        project_root=str(args.project_root),
        scene=dict(id=args.scene_id),
        output=dict(root=str(args.output_root)),
        pipeline=dict(stages=stages),
        artifacts=dict(
            catalog=f"{q}/catalog/catalog.json",
            native_masks=f"{q}/catalog/object_masks.npz",
        ),
        quality_profile=dict(
            schema="farm.bounded-quality-profile.v1",
            budgets=budgets,
            start="completed registered metric RGBD from existing FARM ingress",
            refinement="Targeted crop refinement remains a separate measured stage; not enabled here",
            scope_resolution="preserve nested alternatives; no automatic physical merging",
            release_eligible=False,
        ),
    )


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__)
    sub = p.add_subparsers(dest="phase", required=True)

    def command(name, paths=()):
        q = sub.add_parser(name)
        for field in paths:
            q.add_argument("--" + field, type=Path, required=True)
        q.add_argument("--output", type=Path, required=True)
        return q

    command("inputs", ["rgbd"])
    command("vocabulary", ["scene", "core"])
    q = command("merge")
    q.add_argument("--source", type=Path, action="append", required=True)
    q = command("segment", ["plan", "vocabulary", "model"])
    q.add_argument("--views", type=int, required=True)
    q.add_argument("--world-up", type=float, nargs=3, required=True)
    q = command("native", ["geometry", "ply", "config"])
    q.add_argument("--groups", type=int, default=128)
    q.add_argument("--alternatives", type=int, default=16)
    q.add_argument("--world-up", type=float, nargs=3, required=True)
    q = command("semantics", ["geometry", "model"])
    q.add_argument("--groups", type=int, default=64)
    q = command(
        "plan", ["rgbd", "ply", "sam-model", "vlm-model", "project-root", "output-root"]
    )
    q.add_argument("--runtimes", type=Path)
    q.add_argument("--scene-id", required=True)
    q.add_argument("--world-up", type=float, nargs=3, required=True)
    for field, value in dict(
        initial_views=12,
        adaptive_views=8,
        vocabulary_views=8,
        native_groups=128,
        semantic_groups=64,
        alternatives=16,
    ).items():
        q.add_argument("--" + field.replace("_", "-"), type=int, default=value)
    args = p.parse_args(argv)
    if args.output.exists():
        raise ValueError("new output required")
    if args.phase == "inputs":
        prepare_inputs(args.rgbd, args.output)
    elif args.phase == "vocabulary":
        union_vocabulary(args.scene, args.core, args.output)
    elif args.phase == "merge":
        merge_proposals(args.source, args.output)
    elif args.phase == "segment":
        segment(args)
    elif args.phase == "native":
        native(args)
    elif args.phase == "semantics":
        semantics(args)
    elif args.phase == "plan":
        write_json(args.output, compile_plan(args))
    return 0
