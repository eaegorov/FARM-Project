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


def bounded_groups(geometry, budget, timestamp_counts=None):
    if type(budget) is not int or not 1 <= budget <= 128:
        raise ValueError("group budget must be in 1..128")
    if geometry.get("test_opened") is not False:
        raise ValueError("development geometry required")
    counts = (
        {g["id"]: g["independent_timestamps"] for g in geometry["groups"]}
        if timestamp_counts is None
        else timestamp_counts
    )
    groups = [g for g in geometry["groups"] if counts.get(g["id"], 0) >= 2]
    ordered = sorted(groups, key=lambda g: (-counts[g["id"]], g["id"]))
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


def union_proposals(primary, supplements, output, *, maximum_primary_iou=None):
    """Combine complementary detectors on identical pixels, without score fusion.

    Primary near-duplicate masks keep priority. Supplementary detector scores
    are never interpreted as calibrated probabilities against the primary model.
    The primary person queries remain the exclusion authority.
    """
    from farm_runtime.quality.proposal_geometry import read_masks, read_observations

    if maximum_primary_iou is not None and not 0 < maximum_primary_iou <= 1:
        raise ValueError("maximum primary IoU must be in (0,1]")
    from farm_runtime.proposal_geometry import mask_overlap

    paths = [primary, *supplements]
    if not supplements or len(set(paths)) != len(paths) or output.exists():
        raise ValueError(
            "distinct primary/supplement manifests and new output required"
        )
    documents, sources = [], []
    for path in paths:
        doc, rows = read_observations(path)
        documents.append(doc)
        sources.append({r["name"]: r for r in rows})
    base = sources[0]
    if any(set(rows) - set(base) for rows in sources[1:]):
        raise ValueError("supplement views must belong to primary observations")
    fields = ("timestamp", "source_image", "grid_shape_hw", "applied_quarter_turns")
    filenames = [Path(name).stem + ".npz" for name in base]
    if len(set(filenames)) != len(filenames):
        raise ValueError("ambiguous mask artifact basenames")
    for name, first in base.items():
        if not any(q["prompt"] == "person" for q in first["queries"]):
            raise ValueError("primary view needs an explicit person query")
        checked_file(first["source_image"])
        for source in sources:
            if name in source and any(source[name][k] != first[k] for k in fields):
                raise ValueError("proposal source identity/grid differs")
    output.mkdir(parents=True)
    (output / "masks").mkdir()
    combined_rows, transient_rows, suppressed = [], [], []
    for name, first in base.items():
        arrays, detections, people = {}, [], []
        primary_masks = read_masks(first, primary.parent)
        for priority, (path, source) in enumerate(zip(paths, sources)):
            if name not in source:
                continue
            row = source[name]
            masks = read_masks(row, path.parent)  # Validate finite tiles/grids.
            with np.load(
                checked_file(row["mask_artifact"]), allow_pickle=False
            ) as archive:
                for index, detection in enumerate(row["detections"]):
                    if priority and maximum_primary_iou is not None:
                        overlaps = [
                            (mask_overlap(masks[index], mask)[0], i)
                            for i, mask in enumerate(primary_masks)
                            if first["detections"][i]["label"] != "person"
                        ]
                        best, primary_index = max(overlaps, default=(0.0, None))
                        if best >= maximum_primary_iou:
                            suppressed.append(
                                dict(
                                    name=name,
                                    source_priority=priority,
                                    source_detection_index=index,
                                    primary_detection_index=primary_index,
                                    primary_iou=best,
                                    reason="overlapping_primary_proposal",
                                )
                            )
                            continue
                    key = f"source_{priority:03d}_mask_{index:04d}"
                    arrays[key] = archive[detection["logit_key"]].copy()
                    entry = dict(
                        detection,
                        index=len(detections),
                        logit_key=key,
                        source_priority=priority,
                        source_detection_index=index,
                        source_logit_semantics=row.get("logit_semantics"),
                    )
                    detections.append(entry)
                    if priority == 0 and detection["label"] == "person":
                        people.append(entry)
        target = output / "masks" / (Path(name).stem + ".npz")
        np.savez_compressed(target, **arrays)
        row = dict(
            first,
            detections=detections,
            mask_artifact=describe_file(target),
            logit_semantics="Uncalibrated per-source logits; >0 foreground; see source manifests.",
        )
        combined_rows.append(row)
        transient_rows.append(
            dict(
                row,
                detections=people,
                queries=[q for q in first["queries"] if q["prompt"] == "person"],
            )
        )
    common = dict(
        schema="farm.complementary-proposals.v1",
        source_manifests=[describe_file(p) for p in paths],
        policy="Primary source first; confidence ranks only within a source; duplicate geometry masks preserve primary priority.",
        observations=combined_rows,
        test_opened=False,
        release_eligible=False,
        source_image_encoder_calls=0,
        source_observation_count=sum(len(rows) for rows in sources),
        maximum_primary_iou=maximum_primary_iou,
        suppressed_proposals=suppressed,
        model_scores_calibrated_across_sources=False,
        primary_transients_preserved=True,
    )
    write_json(output / "manifest.json", common)
    write_json(
        output / "transients.json",
        dict(
            common, observations=transient_rows, role="primary person-only exclusions"
        ),
    )


def validated_cohort(validation_path, group_id, output):
    """Reuse accepted observations in a later recovery iteration, without inference.

    Original logits and person exclusions are copied with exact source bindings.
    The cohort does not assert a new association or promote model labels.
    """
    from farm_runtime.quality.proposal_geometry import read_masks, read_observations
    from farm_runtime.quality.surface_evidence import SurfaceInputs

    validation = read(validation_path)
    if (
        output.exists()
        or validation.get("closed_test_opened") is not False
        or validation.get("release_eligible") is not False
    ):
        raise ValueError("new output and development validation required")
    audit = read(checked_file(validation["source_audit"]))
    inputs = SurfaceInputs(checked_file(audit["source_geometry"]))
    groups = {g["id"]: g for g in inputs.geometry["groups"]}
    rows = {g["group_id"]: g for g in validation["groups"]}
    if group_id not in groups or group_id not in rows:
        raise ValueError("group absent from validation/geometry")
    fields = ("timestamp", "source_image", "grid_shape_hw", "applied_quarter_turns")
    extras = {}
    for descriptor in [
        validation["source_proposals"],
        *validation.get("supplements", []),
    ]:
        path = checked_file(descriptor)
        _, observations = read_observations(path)
        for obs in observations:
            name = obs["name"]
            if name in extras and any(obs[k] != extras[name][0][1][k] for k in fields):
                raise ValueError("cohort supplement identity/grid mismatch")
            extras.setdefault(name, []).append((path, obs))
    selected = {}
    for node_id in groups[group_id]["members"]:
        node = inputs.nodes[node_id]
        name = node["frame"]
        obs = inputs.observations[name]
        transient = inputs.transients[name]
        people = [
            (inputs.transients_path, transient, i)
            for i, d in enumerate(transient["detections"])
            if d["label"] == "person"
        ]
        if name in selected:
            raise ValueError("duplicate source group/frame")
        selected[name] = (
            obs,
            [(inputs.proposals_path, obs, node["representative_detection"]), *people],
        )
    for match in rows[group_id]["extra_matches"]:
        index = match["selected_detection"]
        if index is None:
            continue
        eligible = {c["detection_index"] for c in match["candidates"] if c["eligible"]}
        if match["decision"] != "matched_static_surface" or index not in eligible:
            raise ValueError("cohort requires geometrically accepted observations")
        name = match["name"]
        if name in selected or name not in extras:
            raise ValueError("duplicate or absent additional group/frame")
        bindings = [
            (path, obs, i)
            for path, obs in extras[name]
            for i in range(len(obs["detections"]))
        ]
        if (
            not 0 <= index < len(bindings)
            or bindings[index][1]["detections"][bindings[index][2]]["label"] == "person"
        ):
            raise ValueError("invalid additional object mask index")
        people = [b for b in bindings if b[1]["detections"][b[2]]["label"] == "person"]
        selected[name] = (extras[name][0][1], [bindings[index], *people])
    if len({obs["timestamp"] for obs, _ in selected.values()}) < 2:
        raise ValueError("cohort requires at least two independent timestamps")
    output.mkdir(parents=True)
    (output / "masks").mkdir()
    observations, transients = [], []
    for number, (name, (base, bindings)) in enumerate(selected.items()):
        if not any(q["prompt"] == "person" for q in base["queries"]):
            raise ValueError("explicit person exclusion query required")
        checked_file(base["source_image"])
        arrays, detections = {}, []
        for j, (path, obs, index) in enumerate(bindings):
            if any(obs[k] != base[k] for k in fields):
                raise ValueError("cohort mask source identity/grid mismatch")
            read_masks(obs, path.parent)
            detection = obs["detections"][index]
            key = f"mask_{j:04d}"
            with np.load(
                checked_file(obs["mask_artifact"]), allow_pickle=False
            ) as archive:
                arrays[key] = archive[detection["logit_key"]].copy()
            detections.append(
                dict(
                    detection,
                    index=j,
                    logit_key=key,
                    cohort_source=dict(
                        manifest=describe_file(path),
                        detection_index=index,
                        source_logit_semantics=obs.get("logit_semantics"),
                    ),
                )
            )
        target = output / "masks" / f"view_{number:04d}.npz"
        np.savez_compressed(target, **arrays)
        row = dict(
            base,
            detections=detections,
            mask_artifact=describe_file(target),
            logit_semantics="Original uncalibrated source logits copied unchanged; >0 foreground; see cohort_source.",
        )
        observations.append(row)
        transients.append(
            dict(
                row,
                detections=detections[1:],
                queries=[q for q in base["queries"] if q["prompt"] == "person"],
            )
        )
    common = dict(
        schema="farm.validated-observation-cohort.v1",
        source_validation=describe_file(validation_path),
        source_geometry=describe_file(inputs.geometry_path),
        original_group_id=group_id,
        observations=observations,
        inference_calls=0,
        test_opened=False,
        closed_test_opened=False,
        release_eligible=False,
    )
    write_json(output / "manifest.json", common)
    write_json(
        output / "transients.json",
        dict(
            common,
            observations=transients,
            role="unchanged validation person exclusions",
        ),
    )


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
    recovery_validation = getattr(args, "recovery_validation", None)
    recovered = (
        native_observations.confirmed_recovery_groups(
            args.geometry, recovery_validation
        )
        if recovery_validation
        else []
    )
    added = sorted(set(recovered) - set(chosen))
    chosen += added
    deferred = [g for g in deferred if g not in chosen]
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
            *(
                ["--recovery-validation", str(recovery_validation)]
                if recovery_validation
                else []
            ),
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
            recovery_added_group_ids=added,
            recovery_additional_budget_limit=16,
            source_recovery_validation=(
                describe_file(recovery_validation) if recovery_validation else None
            ),
            scene_completeness_claimed=False,
        ),
    )


def apply_refinement(args):
    """Rebuild only after accepted replacements or an observation quarantine."""
    from farm_runtime.quality import native_observations

    native = read(args.native)
    selection = read(args.selection)
    if (
        native.get("closed_test_opened") is not False
        or selection.get("closed_test_opened") is not False
    ):
        raise ValueError("development native refinement required")
    for key in ("input", "config"):
        checked_file(native[key])
        checked_file(selection[key])
        if native[key]["sha256"] != selection[key]["sha256"]:
            raise ValueError("refinement does not belong to this native build")
    changed = sum(bool(r["selected"]) for r in selection["observations"])
    quarantined = len(selection["quarantined_observations"])
    if changed or quarantined:
        native_observations.main(
            [
                "--input",
                str(checked_file(selection["output_input"])),
                "--ply",
                str(checked_file(native["source_ply"])),
                "--config",
                str(checked_file(native["config"])),
                "--mode",
                "exclusions_on",
                "--scope-alternative-budget",
                str(args.alternatives),
                "--output",
                str(args.output),
            ]
        )
        result = read(args.output / "manifest.json")
    else:
        args.output.mkdir(parents=True)
        result = dict(native)
    result["refinement_application"] = dict(
        original_native=describe_file(args.native),
        selection=describe_file(args.selection),
        replaced_observations=changed,
        quarantined_observations=quarantined,
        native_rebuilt=bool(changed or quarantined),
    )
    write_json(args.output / "manifest.json", result)


def semantics(args):
    from farm_runtime.quality import scope_evidence, refinement

    geometry = read(args.geometry)
    native_input = checked_file(read(args.native)["input"]) if args.native else None
    counts = None
    if native_input:
        from farm_runtime.quality.native_observations import retained_timestamp_counts

        counts = retained_timestamp_counts(native_input, args.geometry)
    chosen, deferred = bounded_groups(geometry, args.groups, counts)
    recovered = (
        sorted(
            obj["object_id"]
            for obj in read(native_input)["objects"]
            if counts.get(obj["object_id"], 0) >= 2
            and any(
                row.get("source_kind") == "validated_additional_view"
                for row in obj["masks"]
            )
        )
        if native_input
        else []
    )
    if len(recovered) > 16:
        raise ValueError("semantic recovery cohort must contain at most 16 groups")
    added = sorted(set(recovered) - set(chosen))
    chosen += added
    deferred = [g for g in deferred if g not in chosen]
    if not chosen:
        raise ValueError("no multi-timestamp appearance candidates")
    deferred = sorted(
        set(deferred)
        | {
            g["id"]
            for g in read(args.geometry)["groups"]
            if g["independent_timestamps"] >= 2 and g["id"] not in chosen
        }
    )
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
            *(["--native-input", str(native_input)] if native_input else []),
        ]
    )
    previous_stage = getattr(args, "retain_stage", None)
    if previous_stage:
        from farm_runtime.quality.semantic_refresh import refresh_appearance

        refresh_appearance(
            previous_stage,
            args.output / "evidence/manifest.json",
            args.model,
            args.output / "appearance",
        )
    else:
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
            source_native_input=describe_file(native_input) if native_input else None,
            selected_group_ids=chosen,
            deferred_group_ids=deferred,
            candidate_budget=args.groups,
            recovery_added_group_ids=added,
            recovery_additional_budget_limit=16,
            labels_are_model_proposals=True,
            physical_scope_assessed=False,
        ),
    )


def compile_plan(args):
    """Return a normal FARM DAG; no additional execution framework."""
    complementary_root = getattr(args, "complementary_model_root", None)
    complementary_vocabulary = getattr(args, "complementary_vocabulary", None)
    partial_view = getattr(args, "partial_view_association", False)
    if complementary_vocabulary and not complementary_root:
        raise ValueError("complementary vocabulary requires a model root")
    crop_budget = getattr(args, "refinement_crops", 0)
    if type(crop_budget) is not int or not 0 <= crop_budget <= 32:
        raise ValueError("refinement crop budget must be in 0..32")
    budgets = dict(
        refinement_crops=crop_budget,
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
                *(["--partial-view-association"] if partial_view else []),
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
    final_proposals = "combined_proposals"
    if complementary_root:
        vocabulary = complementary_vocabulary or f"{root}/configs/yoloe_vocabulary.txt"
        primary = f"{q}/combined_proposals/manifest.json"
        supplement = f"{q}/complementary_segmentation/yoloe/manifest.json"
        add(
            "complementary_segmentation",
            "complementary-discovery",
            [
                "--primary",
                primary,
                "--model-root",
                complementary_root,
                "--vocabulary",
                vocabulary,
                *up_args,
                "--output",
                f"{q}/complementary_segmentation",
            ],
            supplement,
            [
                primary,
                vocabulary,
                complementary_root / "yoloe/yoloe-v8l-seg-pf.pt",
                complementary_root / "yoloe/yoloe-v8l-seg.pt",
                complementary_root / "mobileclip/mobileclip_blt.pt",
            ],
        )
        final_proposals = "complementary_proposals"
        add(
            final_proposals,
            "scene-profile",
            [
                "union",
                "--primary",
                primary,
                "--supplement",
                supplement,
                "--max-primary-iou",
                0.5,
                "--output",
                f"{q}/{final_proposals}",
            ],
            f"{q}/{final_proposals}/manifest.json",
            [primary, supplement],
        )
    geometry("geometry", final_proposals)
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
    final_native = f"{q}/native/manifest.json"
    if crop_budget:
        schedule = f"{q}/refinement_schedule"
        selection = f"{q}/refinement_selection"
        add(
            "refinement_schedule",
            "refinement-schedule",
            [
                "--native",
                final_native,
                "--crop-budget",
                crop_budget,
                *(
                    ["--balance-error-modes"]
                    if getattr(args, "balance_refinement_errors", False)
                    else []
                ),
                *up_args,
                "--output",
                schedule,
            ],
            f"{schedule}/manifest.json",
            [final_native],
            "geometry",
        )
        for backend in ("tracker", "concept"):
            name = f"refinement_{backend}"
            add(
                name,
                "refinement",
                [
                    "sam",
                    "--evidence",
                    f"{schedule}/manifest.json",
                    "--model",
                    args.sam_model,
                    "--backend",
                    backend,
                    *[
                        x
                        for p in (getattr(args, "refinement_reuse", []) or [])
                        for x in ("--reuse-proposals", p)
                    ],
                    *(
                        ["--prompts", f"{schedule}/concepts.json"]
                        if backend == "concept"
                        else []
                    ),
                    "--output",
                    f"{q}/{name}",
                ],
                f"{q}/{name}/manifest.json",
                [
                    f"{schedule}/manifest.json",
                    *(getattr(args, "refinement_reuse", []) or []),
                ],
            )
        proposal_paths = [
            f"{q}/refinement_{b}/manifest.json" for b in ("tracker", "concept")
        ]
        add(
            "refinement_selection",
            "native-refinement",
            [
                "--input",
                f"{q}/native/input/manifest.json",
                "--config",
                f"{q}/native/config.json",
                "--ply",
                args.ply,
                *[x for p in proposal_paths for x in ("--proposals", p)],
                "--output",
                selection,
            ],
            f"{selection}/report.json",
            [final_native, *proposal_paths],
            "geometry",
        )
        add(
            "refined_native",
            "scene-profile",
            [
                "apply-refinement",
                "--native",
                final_native,
                "--selection",
                f"{selection}/report.json",
                "--alternatives",
                args.alternatives,
                "--output",
                f"{q}/refined_native",
            ],
            f"{q}/refined_native/manifest.json",
            [final_native, f"{selection}/report.json"],
            "geometry",
        )
        final_native = f"{q}/refined_native/manifest.json"
    add(
        "appearance",
        "scene-profile",
        [
            "semantics",
            "--native",
            final_native,
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
        [f"{q}/geometry/manifest.json", final_native],
    )
    add(
        "catalog",
        "scene-catalog",
        [
            "--native",
            final_native,
            "--semantics",
            f"{q}/semantics/manifest.json",
            "--output",
            f"{q}/catalog",
        ],
        f"{q}/catalog/catalog.json",
        [final_native, f"{q}/semantics/manifest.json"],
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
            refinement=(
                "bounded other-timestamp crop refinement"
                if crop_budget
                else "disabled by crop budget 0"
            ),
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
    q = command("union", ["primary"])
    q.add_argument("--supplement", type=Path, action="append", required=True)
    q.add_argument("--max-primary-iou", type=float, default=None)
    q = command("cohort", ["validation"])
    q.add_argument("--group-id", type=int, required=True)
    q = command("segment", ["plan", "vocabulary", "model"])
    q.add_argument("--views", type=int, required=True)
    q.add_argument("--world-up", type=float, nargs=3, required=True)
    q = command("native", ["geometry", "ply", "config"])
    q.add_argument("--recovery-validation", type=Path)
    q.add_argument("--groups", type=int, default=128)
    q.add_argument("--alternatives", type=int, default=16)
    q.add_argument("--world-up", type=float, nargs=3, required=True)
    q = command("apply-refinement", ["native", "selection"])
    q.add_argument("--alternatives", type=int, default=16)
    q = command("semantics", ["geometry", "model"])
    q.add_argument(
        "--retain-stage",
        type=Path,
        help="Preserve previous annotations on identical evidence; review changed objects only",
    )
    q.add_argument("--native", type=Path)
    q.add_argument("--groups", type=int, default=64)
    q = command(
        "plan", ["rgbd", "ply", "sam-model", "vlm-model", "project-root", "output-root"]
    )
    q.add_argument("--runtimes", type=Path)
    q.add_argument("--scene-id", required=True)
    q.add_argument(
        "--partial-view-association",
        action="store_true",
        help="Use co-visible surface agreement for observations clipped by the image boundary",
    )
    q.add_argument(
        "--complementary-model-root",
        type=Path,
        help="Opt in to YOLOE supplements on the same selected RGB; runtime model root must match",
    )
    q.add_argument(
        "--complementary-vocabulary",
        type=Path,
        help="YOLOE vocabulary; defaults to the repository YOLOE vocabulary",
    )

    q.add_argument("--world-up", type=float, nargs=3, required=True)
    for field, value in dict(
        refinement_crops=0,
        initial_views=12,
        adaptive_views=8,
        vocabulary_views=8,
        native_groups=128,
        semantic_groups=64,
        alternatives=16,
    ).items():
        q.add_argument("--" + field.replace("_", "-"), type=int, default=value)
    q.add_argument(
        "--balance-refinement-errors",
        action="store_true",
        help="Opt in to foreground/background error diversity within the existing crop budget",
    )
    q.add_argument(
        "--refinement-reuse",
        type=Path,
        action="append",
        default=[],
        help="Compatible proposal manifests reused by refinement before new inference",
    )
    args = p.parse_args(argv)
    if args.output.exists():
        raise ValueError("new output required")
    if args.phase == "inputs":
        prepare_inputs(args.rgbd, args.output)
    elif args.phase == "vocabulary":
        union_vocabulary(args.scene, args.core, args.output)
    elif args.phase == "merge":
        merge_proposals(args.source, args.output)
    elif args.phase == "union":
        union_proposals(
            args.primary,
            args.supplement,
            args.output,
            maximum_primary_iou=args.max_primary_iou,
        )
    elif args.phase == "cohort":
        validated_cohort(args.validation, args.group_id, args.output)
    elif args.phase == "segment":
        segment(args)
    elif args.phase == "native":
        native(args)
    elif args.phase == "semantics":
        semantics(args)
    elif args.phase == "apply-refinement":
        apply_refinement(args)
    elif args.phase == "plan":
        write_json(args.output, compile_plan(args))
    return 0
