"""Preserve nested scope hypotheses without asserting physical part ownership."""

from __future__ import annotations

import json
import time

import numpy as np

from farm_runtime.object_scope import aggregate_scope_containment
from farm_runtime.quality.mask_refinement import checked_file
from farm_runtime.quality_baseline import describe_file, write_json
from tools.farm_shaper_bridge import gaussian_lift as lift
from tools.farm_shaper_bridge.common import build_verified_csr
from tools.farm_shaper_bridge.lift_refinement import (
    apply_provisional_geometry_gate,
    refine_connected_claims,
)


def competing_scopes(groups, nodes, pairs, active_ids):
    """Return direct, repeatedly observed nesting, never inferred part-of."""
    active = set(active_ids)
    neighbors = {}
    relations = aggregate_scope_containment(groups, nodes, pairs)
    for row in relations:
        a, b = row["outer_group"], row["inner_group"]
        if row["status"] == "multiview_containment" and {a, b} <= active:
            neighbors.setdefault(a, set()).add(b)
            neighbors.setdefault(b, set()).add(a)
    return neighbors, relations


def current_scope_containment(run, split, active_ids):
    """Measure nesting from masks actually used by the lift, including recovery.

    Only build observations participate. Same-timestamp virtual/sensor views
    provide one vote; transient exclusions never supply containment evidence.
    """
    from collections import defaultdict
    from itertools import combinations

    from tools.farm_shaper_bridge.common import (
        load_mask_pair,
        load_observation_exclusion,
        resolve_mask_path,
    )

    active = set(active_ids)
    objects = {obj.object_id: obj for obj in run.objects}
    if len(objects) != len(run.objects) or not active <= objects.keys():
        raise ValueError("unique known active object IDs required")
    allowed = set(split["build_timestamps"])
    if allowed & set(split["heldout_timestamps"]):
        raise ValueError("build and heldout timestamps must be disjoint")
    object_build = {
        row["object_id"]: set(row["build_timestamps"]) for row in split["objects"]
    }
    by_frame = defaultdict(list)
    for oid in sorted(active):
        seen = set()
        for obs in objects[oid].observations:
            frame = run.frame(obs.image_id)
            if obs.image_id in seen or obs.object_id != oid:
                raise ValueError("unique aligned object/frame observations required")
            seen.add(obs.image_id)
            if (
                frame.physical_timestamp in allowed
                and frame.physical_timestamp in object_build[oid]
            ):
                by_frame[obs.image_id].append(obs)
    members = {oid: [] for oid in sorted(active)}
    nodes, pairs = [], []
    for image_id, observations in sorted(by_frame.items()):
        if len(observations) < 2:
            continue
        frame = run.frame(image_id)
        excluded, exclusion_source = load_observation_exclusion(run, frame)
        masks = []
        for obs in observations:
            path = resolve_mask_path(run, obs)
            mask, _, digest = load_mask_pair(path, frame.depth_size)
            if excluded is not None:
                mask &= ~excluded
            yy, xx = np.nonzero(mask)
            if not len(xx):
                continue
            node_id = len(nodes)
            nodes.append(
                dict(
                    id=node_id,
                    frame=frame.source_image,
                    timestamp=frame.physical_timestamp,
                    object_id=obs.object_id,
                    image_id=image_id,
                    mask=dict(path=str(path), sha256=digest),
                    exclusion=exclusion_source,
                )
            )
            members[obs.object_id].append(node_id)
            masks.append(
                (
                    node_id,
                    mask,
                    len(xx),
                    (
                        int(xx.min()),
                        int(yy.min()),
                        int(xx.max()) + 1,
                        int(yy.max()) + 1,
                    ),
                )
            )
        for (a, ma, na, ba), (b, mb, nb, bb) in combinations(masks, 2):
            x0, y0 = max(ba[0], bb[0]), max(ba[1], bb[1])
            x1, y1 = min(ba[2], bb[2]), min(ba[3], bb[3])
            if x0 >= x1 or y0 >= y1:
                continue
            overlap = np.count_nonzero(ma[y0:y1, x0:x1] & mb[y0:y1, x0:x1])
            ca, cb = overlap / na, overlap / nb
            if (ca >= 0.85 and cb < 0.80) or (cb >= 0.85 and ca < 0.80):
                pairs.append(dict(a=a, b=b, containments=[ca, cb]))
    groups = [dict(id=oid, members=ids) for oid, ids in members.items()]
    neighbors, relations = competing_scopes(groups, nodes, pairs, active)
    return neighbors, relations, nodes


def resolve_scope_alternative(evidence, target, excluded, means, radii, config):
    """Rerun ownership on cached VJP, retaining every unrelated competitor."""
    if (
        target not in evidence
        or target in excluded
        or not set(excluded) <= evidence.keys()
    ):
        raise ValueError("known target and distinct known excluded scopes required")
    subset = {oid: value for oid, value in evidence.items() if oid not in excluded}
    count = len(means)
    owner, confidence, support, rows, conflicts, blocked = lift.make_claims(
        subset, count, config
    )
    owner, confidence, support, refinement = refine_connected_claims(
        means, radii, subset, owner, confidence, support, blocked, config
    )
    owner, confidence, support, geometry = apply_provisional_geometry_gate(
        owner, confidence, support, rows, count, config
    )
    # A separate one-target bank. It is not a replacement exclusive scene bank.
    other = owner != target
    owner[other], confidence[other], support[other] = -1, 0, 0
    bank = build_verified_csr(owner, confidence, support)
    return bank, dict(
        object=next(row for row in rows if row["object_id"] == target),
        conflicts=conflicts,
        refinement=refinement,
        geometry_gate=geometry,
    )


def preserve_scope_alternatives(
    input_manifest,
    evidence,
    gaussians,
    config,
    primary_bank,
    object_rows,
    output,
    budget,
    *,
    run,
    split,
):
    """A bounded CPU stage; all costly rendered contributions are reused."""
    if type(budget) is not int or not 1 <= budget <= 16:
        raise ValueError("scope alternative budget must be from 1 to 16")
    started = time.monotonic()
    geometry_path = checked_file(input_manifest["source_geometry"])
    geometry = json.loads(geometry_path.read_text())
    if geometry.get("test_opened") is not False:
        raise ValueError("development geometry required")
    neighbors, relations, observation_nodes = current_scope_containment(
        run, split, evidence.keys()
    )
    rows_by_id = {row["object_id"]: row for row in object_rows}
    lost = {
        oid: rows_by_id[oid]["supported_claims_before_conflicts"]
        - rows_by_id[oid]["resolved_core_gaussians"]
        for oid in neighbors
    }
    scheduled = sorted(
        (oid for oid in neighbors if lost[oid] > 0),
        key=lambda oid: (-lost[oid], oid),
    )
    output.mkdir()
    results = []
    for oid in scheduled[:budget]:
        tick = time.monotonic()
        bank, audit = resolve_scope_alternative(
            evidence, oid, neighbors[oid], gaussians.means_m, gaussians.radius_m, config
        )
        positions = np.flatnonzero(primary_bank["object_ids"] == oid)
        old = (
            primary_bank["indices"][
                primary_bank["indptr"][positions[0]] : primary_bank["indptr"][
                    positions[0] + 1
                ]
            ]
            if len(positions)
            else np.zeros(0, np.int64)
        )
        target_path = output / f"object_{oid:06d}.npz"
        np.savez_compressed(target_path, **bank)
        results.append(
            dict(
                object_id=oid,
                excluded_scope_competitors=sorted(neighbors[oid]),
                primary_count=len(old),
                alternative_count=len(bank["indices"]),
                added=len(np.setdiff1d(bank["indices"], old)),
                removed=len(np.setdiff1d(old, bank["indices"])),
                bank=describe_file(target_path),
                audit=audit,
                seconds=time.monotonic() - tick,
            )
        )
    manifest = dict(
        schema="farm.native-scope-alternatives.v1",
        source_geometry=describe_file(geometry_path),
        relations=relations,
        relation_node_namespace="current_native_observations",
        observation_nodes=observation_nodes,
        objects=results,
        deferred_object_ids=scheduled[budget:],
        maximum_alternatives=budget,
        total_seconds=time.monotonic() - started,
        rendered_evidence_reused=True,
        additional_vjp_calls=0,
        physical_ownership_assigned=False,
        primary_scene_bank_replaced=False,
        interpretation=(
            "Mutually alternative candidate scopes may overlap in source Gaussian IDs. "
            "Other objects still compete normally. Pixel evidence is frozen from the selected "
            "mask policy and original cohort, not recomputed after removing a competitor. "
            "Containment is not physical part-of; "
            "select or refine scope before publishing an exclusive scene bank."
        ),
        closed_test_opened=False,
        release_eligible=False,
    )
    write_json(output / "manifest.json", manifest)
    return describe_file(output / "manifest.json")
