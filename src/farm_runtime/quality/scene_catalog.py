"""Bind native masks, observed geometry and appearance proposals into one catalog."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import time

import numpy as np

from farm_runtime.instance_bank import filter_instance_bank
from farm_runtime.native_object_geometry import fit_native_obb
from farm_runtime.quality.mask_refinement import checked_file
from farm_runtime.quality_baseline import describe_file, write_json


def read(path):
    return json.loads(Path(path).read_text())


def bound_appearance(stage_path, geometry_record, native_input_record=None):
    """Reject a same-number object from any other geometry namespace."""
    stage = read(stage_path)
    if stage.get("schema") != "farm.quality-appearance-stage.v1":
        raise ValueError("quality appearance stage required")
    checked_file(stage["source_geometry"])
    if stage["source_geometry"]["sha256"] != geometry_record["sha256"]:
        raise ValueError("appearance belongs to a different geometry namespace")
    appearance_path = checked_file(stage["appearance"])
    appearance = read(appearance_path)
    if (
        appearance.get("schema") != "farm.local-object-appearance.v1"
        or appearance.get("physical_scope_assessed") is not False
        or appearance.get("reserved_test_opened") is not False
    ):
        raise ValueError("development compact appearance required")
    evidence = read(checked_file(appearance["proposals"]))
    checked_file(evidence["source_groups"])
    if evidence["source_groups"]["sha256"] != geometry_record["sha256"]:
        raise ValueError("appearance pixels belong to a different geometry namespace")
    if native_input_record is not None:
        for document in (stage, evidence):
            binding = document.get("source_native_input")
            if not binding or binding["sha256"] != native_input_record["sha256"]:
                raise ValueError("appearance masks differ from effective native input")
            checked_file(binding)
    ids = [r["object_id"] for r in appearance["objects"]]
    if len(ids) != len(set(ids)) or set(ids) != set(stage["selected_group_ids"]):
        raise ValueError("appearance group selection mismatch")
    return {r["object_id"]: r for r in appearance["objects"]}, stage


def geometry_status(geometry, *, competing_scopes):
    flags = []
    if geometry is None:
        flags.append("no_accepted_native_mask")
    elif geometry["extent_stability"]["sensitive_axes"]:
        flags.append("observed_extent_tail_sensitive")
    if competing_scopes:
        flags.append("unresolved_nested_scope")
    return dict(
        flags=flags,
        observed_obb_available=geometry is not None,
        physical_dimensions_m=None,
        physical_size_validated=False,
        mask_completeness_validated=False,
        physical_scope_resolved=False,
        interpretation="Observed reconstruction support is not a physical measurement of the complete object.",
    )


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__)
    for name in ("native", "semantics", "output"):
        p.add_argument("--" + name, type=Path, required=True)
    p.add_argument("--native-selection", type=Path)
    args = p.parse_args(argv)
    if args.output.exists():
        raise ValueError("new catalog output required")
    started = time.monotonic()
    manifest = read(args.native)
    if manifest.get("closed_test_opened") is not False:
        raise ValueError("development native output required")
    from farm_runtime.quality.native_observations import load_prepared
    from tools.farm_shaper_bridge.common import open_graphdeco_ply

    input_path = checked_file(manifest["input"])
    run, _, inputs = load_prepared(input_path)
    record = inputs.get("source_geometry")
    if not record:
        raise ValueError("native bank must be bound to geometric group namespace")
    geometry = read(checked_file(record))
    require_binding = bool(
        inputs.get("parent_input")
        or inputs.get("source_validation")
        or read(args.semantics).get("source_native_input")
    )
    appearances, semantics = bound_appearance(
        args.semantics, record, describe_file(input_path) if require_binding else None
    )
    source_path = checked_file(manifest["source_ply"])
    if inputs["source_ply"]["sha256"] != manifest["source_ply"]["sha256"]:
        raise ValueError("native source PLY mismatch")
    source = open_graphdeco_ply(source_path)
    reports = [read(checked_file(r)) for r in manifest["reports"]]
    matching = [r for r in reports if r["mode"] == "exclusions_on"]
    if len(matching) != 1 or matching[0]["source_gaussian_count"] != source.count:
        raise ValueError("one source-matched exclusion-aware native bank required")
    report = matching[0]
    with np.load(checked_file(report["bank"]), allow_pickle=False) as data:
        bank = filter_instance_bank(
            data, data["object_ids"].tolist(), source_count=source.count
        )
    groups = {g["id"]: g for g in geometry["groups"]}
    selected = {o.object_id: o for o in run.objects}
    context = None
    if args.native_selection:
        from farm_runtime.quality.scene_roles import bind_context_selection
        context = bind_context_selection(args.native_selection, record, selected)
    context_ids = {row["object_id"] for row in context or []}
    native_counts = {
        oid: len({run.frame(o.image_id).physical_timestamp for o in obj.observations})
        for oid, obj in selected.items()
    }
    if not set(selected) <= groups.keys() or not set(appearances) <= groups.keys():
        raise ValueError("unknown group in native or appearance input")
    alternative_manifest = (
        read(checked_file(report["scope_alternatives"]))
        if report.get("scope_alternatives")
        else None
    )
    relations = []
    alternatives = {}
    if alternative_manifest:
        if alternative_manifest["source_geometry"]["sha256"] != record["sha256"]:
            raise ValueError("scope alternatives belong to another geometry")
        relations = alternative_manifest["relations"]
        alternatives = {r["object_id"]: r for r in alternative_manifest["objects"]}
    competing = {oid: set() for oid in selected}
    for relation in relations:
        if relation["status"] == "multiview_containment":
            a, b = relation["outer_group"], relation["inner_group"]
            if a in selected and b in selected:
                competing[a].add(b)
                competing[b].add(a)
    args.output.mkdir(parents=True)
    (args.output / "scope_alternatives").mkdir()
    bank_path = args.output / "object_masks.npz"
    np.savez_compressed(bank_path, **bank)
    primary_geometry = {
        r["object_id"]: r["geometry"] for r in report["native_geometry"]
    }
    counts = {
        int(oid): int(bank["indptr"][i + 1] - bank["indptr"][i])
        for i, oid in enumerate(bank["object_ids"])
    }
    rows = []
    for oid, obj in sorted(selected.items()):
        extra = None
        if oid in alternatives:
            item = alternatives[oid]
            with np.load(checked_file(item["bank"]), allow_pickle=False) as raw:
                if not set(raw["object_ids"].tolist()) <= {oid}:
                    raise ValueError("scope alternative contains unrelated objects")
                candidate = filter_instance_bank(
                    raw, raw["object_ids"].tolist(), source_count=source.count
                )
            ids = candidate["indices"]
            observed = None
            if len(ids):
                table = source.data[ids]
                points = (
                    np.column_stack([table[k] for k in ("x", "y", "z")])
                    * run.meters_per_scene_unit
                )
                opacity = 1 / (
                    1 + np.exp(-np.clip(table["opacity"].astype(float), -30, 30))
                )
                observed = fit_native_obb(
                    points, opacity * candidate["confidence"], obj.rotation
                )
            path = args.output / "scope_alternatives" / f"object_{oid:06d}.npz"
            np.savez_compressed(path, **candidate)
            extra = dict(
                bank=dict(describe_file(path), path=str(path.relative_to(args.output))),
                native_gaussians=len(ids),
                observed_obb=observed,
                excluded_scope_competitors=item["excluded_scope_competitors"],
                quality=geometry_status(observed, competing_scopes=competing[oid]),
                exclusive_with_primary=False,
                physical_relation="unresolved",
            )
        appearance = appearances.get(oid)
        parsed = appearance["parsed"] if appearance else None
        rows.append(
            dict(
                object_id=oid,
                candidate_labels=groups[oid]["candidate_labels"],
                independent_timestamps=groups[oid]["independent_timestamps"],
                native_independent_timestamps=native_counts[oid],
                recovered_from_single_timestamp=(
                    groups[oid]["independent_timestamps"] < 2
                    and native_counts[oid] >= 2
                ),
                label=parsed["label"] if parsed else None,
                caption=parsed["caption"] if parsed else None,
                semantic_status=(
                    "unverified_model_proposal" if parsed else "unavailable"
                ),
                appearance_evidence=appearance,
                native_gaussians=counts.get(oid, 0),
                observed_obb=primary_geometry.get(oid),
                quality=geometry_status(
                    primary_geometry.get(oid), competing_scopes=competing[oid]
                ),
                competing_scope_ids=sorted(competing[oid]),
                alternative_scope=extra,
                source_group=dict(namespace_sha256=record["sha256"], id=oid),
            )
        )
    catalog = dict(
        schema="farm.quality-scene-catalog.v1",
        scene_id=run.scene_id,
        objects=rows,
        **(dict(
            scene_context=context,
            scene_context_selection=describe_file(args.native_selection),
            scene_context_masks_source=record,
            scene_context_native_masks_assigned=False,
        ) if context is not None else {}),
        source_ply=manifest["source_ply"],
        source_gaussian_count=source.count,
        meters_per_scene_unit=run.meters_per_scene_unit,
        object_id_namespace=dict(
            geometry=record, interpretation="Geometry group IDs; never legacy FARM IDs"
        ),
        native_mask_bank=dict(describe_file(bank_path), path=bank_path.name),
        native_output=describe_file(args.native),
        appearance_stage=describe_file(args.semantics),
        scope_relations=relations,
        deferred_native_group_ids=sorted(
            g
            for g in groups
            if groups[g]["independent_timestamps"] >= 2 and g not in selected
            and g not in context_ids
        ),
        unconfirmed_single_timestamp_group_ids=sorted(
            g
            for g in groups
            if native_counts.get(g, groups[g]["independent_timestamps"]) < 2
        ),
        deferred_appearance_group_ids=semantics["deferred_group_ids"],
        deferred_scope_alternative_ids=(
            alternative_manifest["deferred_object_ids"] if alternative_manifest else []
        ),
        status="DEVELOPMENT_CANDIDATES",
        primary_bank_exclusive=True,
        alternative_banks_may_overlap=True,
        source_row_order_preserved=True,
        physical_ownership_assigned=False,
        scene_completeness_validated=False,
        automatic_semantic_accuracy_validated=False,
        closed_test_opened=False,
        release_eligible=False,
        total_seconds=time.monotonic() - started,
    )
    write_json(args.output / "catalog.json", catalog)
    print(
        json.dumps(
            dict(
                objects=len(rows),
                primary_masks=len(counts),
                alternatives=len(alternatives),
                seconds=catalog["total_seconds"],
            )
        ),
        flush=True,
    )
    return 0
