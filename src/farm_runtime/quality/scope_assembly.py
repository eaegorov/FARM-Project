"""Resolve confirmed whole-object scopes before exclusive Gaussian ownership.

Families reuse a single original evidence row per Gaussian. They never sum
physical timestamps, fill a bounding-box volume, or bypass native lift gates.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import time

import numpy as np

from farm_runtime.quality.mask_refinement import checked_file
from farm_runtime.quality.scene_catalog import geometry_status
from farm_runtime.quality_baseline import describe_file, write_json
from tools.farm_shaper_bridge import gaussian_lift as lift

FIELDS = ("indices", "radius_m", "positive_weight", "negative_weight",
          "visible_weight", "positive_timestamps", "negative_timestamps")


def select_families(review, known_ids):
    """Compose directed integral relations without joining competing roots.

    Multiple paths from the same root are compatible. Membership reachable from
    several maximal roots remains unresolved; an explicit separate-object veto
    anywhere on that root's hierarchy wins over an integral proposal.
    """
    if review.get("schema") != "farm.scope-family-review.v1":
        raise ValueError("bound scope-family review required")
    known = set(known_ids)
    proposed, separate, seen = {}, {}, set()
    for row in review["decisions"]:
        parent = row["parent_id"]
        if type(parent) is not int or parent not in known or parent in seen:
            raise ValueError("unique known parent IDs required")
        seen.add(parent)
        for key in ("integral_ids", "separate_ids"):
            values = row.get(key, [])
            if any(type(g) is not int for g in values) or len(values) != len(set(values)):
                raise ValueError("unique integer child IDs required")
            if parent in values or not set(values) <= known:
                raise ValueError("known distinct child IDs required")
        separate[parent] = set(row.get("separate_ids", []))
        if row.get("accepted") is True and row.get("parent_scope") == "single_whole_object":
            proposed[parent] = set(row.get("integral_ids", [])) - separate[parent]
    incoming = {}
    for parent, children in proposed.items():
        for child in children:
            incoming.setdefault(child, set()).add(parent)
    nodes = set(proposed) | set(incoming)
    roots = nodes - set(incoming)
    # Kahn's algorithm also defers all descendants of cyclic relations.
    indegree = {g: len(incoming.get(g, ())) for g in nodes}
    queue = sorted(roots)
    root_sets = {g: {g} for g in roots}
    visited = []
    while queue:
        parent = queue.pop(0)
        visited.append(parent)
        for child in sorted(proposed.get(parent, ())):
            root_sets.setdefault(child, set()).update(root_sets[parent])
            indegree[child] -= 1
            if indegree[child] == 0:
                queue.append(child)
    cyclic = nodes - set(visited)
    ambiguous = {g for g, rs in root_sets.items() if len(rs) > 1} | cyclic
    candidates = {
        root: {g for g, rs in root_sets.items() if rs == {root} and g not in cyclic}
        for root in sorted(roots)
    }
    vetoed = set()
    families = {}
    for root, members in candidates.items():
        # Do not inherit descendants through a member removed by a veto.
        veto = set().union(*(separate.get(g, set()) for g in members))
        vetoed.update(members & veto)
        if root in veto:
            # An explicit descendant/ancestor contradiction cannot retain the
            # root merely because traversal is initialized with that root.
            continue
        reachable, pending = {root}, [root]
        while pending:
            parent = pending.pop()
            for child in proposed.get(parent, ()):
                if child in members and child not in veto and child not in reachable:
                    reachable.add(child)
                    pending.append(child)
        if len(reachable) > 1:
            families[root] = sorted(reachable)
    flat = [g for members in families.values() for g in members]
    if len(flat) != len(set(flat)):
        raise AssertionError("families must be disjoint")
    return families, dict(
        ambiguous_child_ids=sorted(ambiguous - cyclic),
        cyclic_or_downstream_ids_deferred=sorted(cyclic),
        separate_veto_ids=sorted(vetoed),
        nested_parent_ids_composed=sorted(set(proposed) & (set(flat) - set(families))),
        policy="unique_maximal_root_with_separate_veto")


def remap_scope_relations(relations, canonical):
    """Deduplicate aliased endpoints without manufacturing independent support.

    These are original proposal-mask observations mapped onto object families,
    not newly measured containment of the assembled Gaussian masks.
    """
    grouped = {}
    for index, row in enumerate(relations):
        source_outer, source_inner = row["outer_group"], row["inner_group"]
        outer = canonical.get(source_outer, source_outer)
        inner = canonical.get(source_inner, source_inner)
        if outer == inner:
            continue
        group = grouped.setdefault((outer, inner), dict(sources=[], evidence={}, timestamps=set()))
        group["sources"].append(dict(source_relation_index=index,
                                     outer_group=source_outer, inner_group=source_inner,
                                     status=row["status"], relation=row.get("relation")))
        group["timestamps"].update(str(value) for value in row.get("timestamps", []))
        for original in row.get("evidence", []):
            evidence = dict(original, source_outer_group=source_outer,
                            source_inner_group=source_inner)
            key = json.dumps(evidence, sort_keys=True, separators=(",", ":"))
            group["evidence"][key] = evidence
            if evidence.get("timestamp") is not None:
                group["timestamps"].add(str(evidence["timestamp"]))
    result = []
    for (outer, inner), group in sorted(grouped.items()):
        sources = group["sources"]
        timestamps = sorted(group["timestamps"])
        statuses = {row["status"] for row in sources}
        if (inner, outer) in grouped or "conflicting_directions" in statuses:
            status = "conflicting_directions"
        elif "multiview_containment" in statuses and len(timestamps) >= 2:
            status = "multiview_containment"
        elif len(timestamps) == 1:
            status = "single_timestamp"
        else:
            # Distinct partial hypotheses do not create a new confirmed relation.
            status = "unresolved_after_family_mapping"
        kinds = {row["relation"] for row in sources if row["relation"] is not None}
        result.append(dict(
            outer_group=outer, inner_group=inner,
            independent_timestamps=len(timestamps), timestamps=timestamps,
            status=status,
            relation=next(iter(kinds)) if len(kinds) == 1 else "mapped_source_relations",
            physical_relation="unresolved",
            evidence=[group["evidence"][key] for key in sorted(group["evidence"])],
            source_relations=sources,
            source_relation_namespace="source_catalog.scope_relations",
            relation_basis="original_proposal_masks_remapped_to_object_families",
            assembled_mask_containment_remeasured=False,
            timestamp_support_summed=False,
        ))
    return result


def refresh_object_quality(objects, relations):
    """Quality flags refer to current OBBs and surviving external relationships."""
    competing = {row["object_id"]: set() for row in objects}
    for relation in relations:
        if relation["status"] not in ("multiview_containment", "conflicting_directions"):
            continue
        outer, inner = relation["outer_group"], relation["inner_group"]
        if outer not in competing or inner not in competing:
            raise ValueError("remapped relation references an absent object")
        competing[outer].add(inner)
        competing[inner].add(outer)
    for row in objects:
        neighbors = sorted(competing[row["object_id"]])
        row["competing_scope_ids"] = neighbors
        row["quality"] = geometry_status(row["observed_obb"], competing_scopes=neighbors)
    return objects


def growth_evidence_eligible(item, config):
    """Existing refinement's evidence gates, before spatial/ownership checks."""
    p = config["refinement"]
    purity = item.positive_weight / np.maximum(
        item.positive_weight + item.negative_weight, item.positive_weight + 1e-8)
    visible = item.positive_weight / np.maximum(item.visible_weight, item.positive_weight + 1e-8)
    return (
        (item.positive_timestamps >= int(p["minimum_positive_timestamps"]))
        & (item.negative_timestamps <= int(p["maximum_negative_timestamps"]))
        & (item.positive_timestamps.astype(np.int32) >=
           item.negative_timestamps.astype(np.int32) + int(p["minimum_timestamp_margin"]))
        & (item.positive_weight >= float(p["minimum_positive_weight"]))
        & (purity >= float(p["minimum_global_purity"]))
        & (visible >= max(float(p["minimum_visible_share"]),
                          float(config["build"].get("minimum_visible_share", 0.0))))
    )


def collapse_family_evidence(evidence, families, config, timestamp_sets):
    """Choose an intact member row: strong > growth eligible > other, then score.

    Among strong claims this is exactly the maximum native score. Multiple
    observations/parts cannot artificially increase independent support.
    """
    flat = [g for members in families.values() for g in members]
    if len(flat) != len(set(flat)):
        raise ValueError("overlapping families forbidden")
    if not set(flat) <= evidence.keys() or any(p not in ids for p, ids in families.items()):
        raise ValueError("known parent-containing families required")
    claims, _ = lift.build_sparse_claims(evidence, config)
    strong = {r["object_id"]: np.asarray(r["indices"]) for r in claims}
    result = {g: item for g, item in evidence.items() if g not in flat}
    audit = []
    for parent, member_ids in sorted(families.items()):
        member_ids = sorted(member_ids)
        arrays = {f: np.concatenate([getattr(evidence[g], f) for g in member_ids]) for f in FIELDS}
        origins = np.concatenate([np.full(len(evidence[g].indices), g, np.int64) for g in member_ids])
        tiers = np.concatenate([
            np.where(np.isin(evidence[g].indices, strong[g]), 2,
                     growth_evidence_eligible(evidence[g], config).astype(np.int8))
            for g in member_ids])
        scores = arrays["positive_timestamps"].astype(np.float32) + float(
            config["build"]["score_log_weight"]) * np.log1p(arrays["positive_weight"])
        order = np.lexsort((origins, -scores, -tiers, arrays["indices"]))
        sorted_ids = arrays["indices"][order]
        selected = order[np.r_[True, sorted_ids[1:] != sorted_ids[:-1]]] if len(order) else order
        timestamps = set().union(*(timestamp_sets[g] for g in member_ids))
        if not timestamps:
            raise ValueError("physical timestamp provenance required")
        chosen = {f: values[selected].copy() for f, values in arrays.items()}
        result[parent] = lift.ObjectEvidence(
            obj=evidence[parent].obj, build_timestamp_count=len(timestamps), **chosen)
        audit.append(dict(parent_id=parent, member_ids=member_ids,
                          candidate_gaussians=len(selected),
                          independent_build_timestamps=sorted(timestamps),
                          strong_source_rows=int(np.count_nonzero(tiers[selected] == 2)),
                          growth_source_rows=int(np.count_nonzero(tiers[selected] == 1)),
                          original_member_row_preserved=True,
                          timestamps_summed=False))
    return result, audit


def resolve(evidence, gs, config):
    from tools.farm_shaper_bridge.lift_refinement import refine_connected_claims, apply_provisional_geometry_gate
    from tools.farm_shaper_bridge.common import build_verified_csr
    owner, confidence, support, rows, conflicts, blocked = lift.make_claims(evidence, gs.count, config)
    owner, confidence, support, refinement = refine_connected_claims(
        gs.means_m, gs.radius_m, evidence, owner, confidence, support, blocked, config)
    owner, confidence, support, geometry = apply_provisional_geometry_gate(
        owner, confidence, support, rows, gs.count, config)
    return build_verified_csr(owner, confidence, support), rows, dict(
        conflicts=conflicts, refinement=refinement, geometry_gate=geometry)


def members_from_bank(bank):
    return {int(g): bank["indices"][bank["indptr"][i]:bank["indptr"][i+1]].copy()
            for i, g in enumerate(bank["object_ids"])}


def cache_evidence(path, evidence):
    arrays = {"object_ids": np.array(sorted(evidence), np.int32)}
    for g, item in evidence.items():
        arrays.update({f"{g}_{f}": getattr(item, f) for f in FIELDS})
        arrays[f"{g}_timestamps"] = np.array(item.build_timestamp_count, np.int32)
    np.savez_compressed(path, **arrays)


def load_evidence_cache(path, objects):
    with np.load(path, allow_pickle=False) as z:
        if set(map(int, z["object_ids"])) != set(objects):
            raise ValueError("cached object set changed")
        return {int(g): lift.ObjectEvidence(
            obj=objects[int(g)], build_timestamp_count=int(z[f"{g}_timestamps"]),
            **{f: z[f"{g}_{f}"].copy() for f in FIELDS}) for g in z["object_ids"]}


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("catalog", "families", "output"):
        parser.add_argument("--" + name, type=Path, required=True)
    parser.add_argument("--evidence-cache", type=Path,
                        help="manifest from a prior source/config-bound assembly run")
    args = parser.parse_args(argv)
    if args.output.exists():
        raise ValueError("new output directory required")
    started = time.monotonic()
    stage_started = started
    stage_seconds = {}

    def finish_stage(name, **details):
        nonlocal stage_started
        now = time.monotonic()
        stage_seconds[name] = now - stage_started
        print(json.dumps(dict(stage=name, stage_seconds=stage_seconds[name],
                              elapsed_seconds=now - started, **details)), flush=True)
        stage_started = now

    print(json.dumps(dict(stage="scope_assembly_started", evidence_cache=bool(args.evidence_cache))), flush=True)
    catalog = json.loads(args.catalog.read_text())
    if catalog.get("schema") != "farm.quality-scene-catalog.v1":
        raise ValueError("unassembled native scene catalog required")
    review = json.loads(args.families.read_text())
    if review.get("closed_test_opened") is not False:
        raise ValueError("development-only family review required")
    if review["source_catalog"]["sha256"] != describe_file(args.catalog)["sha256"]:
        raise ValueError("family review belongs to a different catalog")
    checked_file(review["source_catalog"])
    native_path = checked_file(catalog["native_output"])
    native = json.loads(native_path.read_text())
    from farm_runtime.quality.native_observations import load_prepared
    from farm_runtime.quality.contributor_recovery import recover_contributor_candidates
    from tools.farm_shaper_bridge.common import open_graphdeco_ply
    from farm_runtime.native_object_geometry import fit_native_obb
    run, split, inputs = load_prepared(checked_file(native["input"]))
    config, _ = lift.load_config(checked_file(native["config"]))
    objects = {o.object_id: o for o in run.objects}
    families, selection_audit = select_families(review, objects)
    finish_stage("prepare_inputs", objects=len(objects), families=len(families))
    source_ply = checked_file(catalog["source_ply"])
    gs = lift.load_gaussians(open_graphdeco_ply(source_ply), run.meters_per_scene_unit)
    finish_stage("load_gaussians", gaussian_count=gs.count)
    args.output.mkdir(parents=True)
    source = dict(catalog=describe_file(args.catalog), native_input=native["input"],
                  native_config=native["config"], source_ply=catalog["source_ply"],
                  measurement_code={name: describe_file(Path(lift.__file__).parent / name)["sha256"]
                                    for name in ("gaussian_lift.py", "lift_candidates.py", "common.py")},
                  recovery_code=describe_file(Path(__file__).with_name("contributor_recovery.py"))["sha256"])
    if args.evidence_cache:
        cache = json.loads(args.evidence_cache.read_text())
        if cache["sources"] != source:
            raise ValueError("cached evidence source/config changed")
        evidence = load_evidence_cache(checked_file(cache["cache"]), objects)
    else:
        evidence, _ = lift.build_candidates(run, gs, split, config)
        lift.accumulate_build_evidence(gs, run, split, evidence, config)
        _, _, _, rows, _, _ = lift.make_claims(evidence, gs.count, config)
        if config["candidate"].get("rendered_recovery_budget", 0):
            recover_contributor_candidates(run, gs, split, evidence, config, rows)
    finish_stage("load_or_measure_evidence", cached=bool(args.evidence_cache))
    old, old_rows, old_audit = resolve(evidence, gs, config)
    source_bank = dict(catalog["native_mask_bank"])
    source_bank["path"] = str(args.catalog.parent / source_bank["path"])
    with np.load(checked_file(source_bank), allow_pickle=False) as z:
        saved = members_from_bank(z)
        saved_arrays = {key: z[key].copy() for key in z.files}
    before = members_from_bank(old)
    if set(saved) != set(before) or any(not np.array_equal(saved[g], before[g]) for g in saved):
        raise ValueError("native baseline replay differs; family experiment cannot proceed")
    for key in ("object_ids", "indptr", "indices", "timestamp_support"):
        if not np.array_equal(saved_arrays[key], old[key]):
            raise ValueError(f"native baseline replay differs for {key}")
    confidence_error = float(np.max(np.abs(saved_arrays["confidence"]-old["confidence"]), initial=0))
    if not np.allclose(saved_arrays["confidence"], old["confidence"], rtol=1e-5, atol=1e-7):
        raise ValueError("native baseline confidence replay differs")
    finish_stage("replay_baseline", masks=len(before), confidence_max_abs_error=confidence_error)
    cache_path = args.output / "evidence_cache.npz"
    if args.evidence_cache:
        cache_record = cache["cache"]
    else:
        cache_evidence(cache_path, evidence)
        cache_record = describe_file(cache_path)
    write_json(args.output / "evidence_cache.json", dict(sources=source, cache=cache_record))
    finish_stage("persist_evidence_cache")
    timestamp_sets = {g: {run.frame(o.image_id).physical_timestamp for o in obj.observations}
                      & set(split["build_timestamps"]) for g, obj in objects.items()}
    assembled, pool_audit = collapse_family_evidence(evidence, families, config, timestamp_sets)
    bank, rows, resolved_audit = resolve(assembled, gs, config)
    finish_stage("assemble_ownership", families=len(families), masks=len(bank["object_ids"]))
    bank_path = args.output / "object_masks.npz"
    np.savez_compressed(bank_path, **bank)
    after = members_from_bank(bank)
    by_id = {o["object_id"]: o for o in catalog["objects"]}
    decision_by_parent = {r["parent_id"]: r for r in review["decisions"]}
    selected_objects = []
    positions = {int(g): i for i, g in enumerate(bank["object_ids"])}
    for g in sorted(assembled):
        row = dict(by_id[g])
        indices = after.get(g, np.array([], np.int64))
        obb = None
        if len(indices):
            j = positions[g];weights = bank["confidence"][bank["indptr"][j]:bank["indptr"][j+1]]
            obb = fit_native_obb(gs.means_m[indices], gs.opacity_cpu[indices]*weights, objects[g].rotation)
        row.update(native_gaussians=len(indices), observed_obb=obb)
        if g in families:
            decision = decision_by_parent[g]
            row.update(object_family=dict(member_ids=families[g], policy="reviewed_whole_object_scopes",
                                          same_object_partial_scope_ids=decision.get("same_object_partial_scope_ids", []),
                                          family_review=describe_file(args.families)),
                       prior_label=row.get("label"), prior_caption=row.get("caption"),
                       label=decision.get("label") or row.get("label"),
                       caption=decision.get("caption") or row.get("caption"),
                       semantic_status="unverified_whole_object_proposal")
        # Previous alternative banks and their OBBs have different competitors.
        row["alternative_scope"] = None
        selected_objects.append(row)
    result = dict(catalog, schema="farm.quality-object-assembly.v1",
                  objects=selected_objects,
                  native_mask_bank=dict(describe_file(bank_path), path=bank_path.name),
                  source_catalog=describe_file(args.catalog),
                  family_review=describe_file(args.families),
                  object_families=[dict(parent_id=p,member_ids=m) for p,m in sorted(families.items())],
                  part_records_preserved_in=describe_file(args.catalog),
                  primary_bank_exclusive=True, physical_ownership_assigned=False,
                  closed_test_opened=False, release_eligible=False)
    result["source_native_output"] = result.pop("native_output")
    result["native_bank_stage"] = "scope_assembly"
    result["native_bank_source"] = describe_file(bank_path)
    result["source_native_input"] = native["input"]
    canonical = {g: p for p, members in families.items() for g in members}
    result["scope_relations"] = remap_scope_relations(catalog.get("scope_relations", []), canonical)
    refresh_object_quality(selected_objects, result["scope_relations"])
    result["alternative_banks_may_overlap"] = False
    for key in ("deferred_native_group_ids", "unconfirmed_single_timestamp_group_ids",
                "deferred_appearance_group_ids", "deferred_scope_alternative_ids"):
        result.pop(key, None)
    result["status"] = "DEVELOPMENT_WHOLE_OBJECT_ASSEMBLY"
    member_flat = {g for members in families.values() for g in members}
    changes = []
    for g in sorted(assembled):
        baseline = np.unique(np.concatenate([before.get(m,np.array([],np.int64)) for m in families.get(g,[g])]))
        current = after.get(g,np.array([],np.int64))
        changes.append(dict(object_id=g,member_ids=families.get(g,[g]),
                            before_union_gaussians=len(baseline),after_gaussians=len(current),
                            added=int(len(np.setdiff1d(current,baseline))),
                            removed=int(len(np.setdiff1d(baseline,current)))))
    write_json(args.output / "catalog.json", result)
    finish_stage("export_catalog", objects=len(selected_objects))
    write_json(args.output / "manifest.json", dict(
        schema="farm.scope-assembly-run.v1", sources=source, family_review=describe_file(args.families),
        catalog=describe_file(args.output / "catalog.json"), bank=describe_file(bank_path),
        families=families, family_selection=selection_audit, pooling=pool_audit,
        baseline_replay_exact=True, baseline_confidence_max_abs_error=confidence_error, baseline_object_rows=old_rows,
        baseline_audit=old_audit, object_rows=rows, assembled_audit=resolved_audit,
        changes=changes, stage_seconds=stage_seconds, original_candidates=len(catalog["objects"]),
        assembled_candidates=len(selected_objects),
        source_gaussian_count=gs.count,
        nonempty_before=len(before),nonempty_after=len(after),
        unchanged_unrelated=sum(c["added"]==0 and c["removed"]==0 for c in changes if c["object_id"] not in member_flat),
        unrelated_total=sum(c["object_id"] not in member_flat for c in changes),
        total_seconds=time.monotonic()-started, closed_test_opened=False, release_eligible=False))
    print(json.dumps(dict(families=families,candidates=len(selected_objects),nonempty=len(after),
                          seconds=time.monotonic()-started)),flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
