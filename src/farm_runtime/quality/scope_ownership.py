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
):
    """A bounded CPU stage; all costly rendered contributions are reused."""
    if type(budget) is not int or not 1 <= budget <= 16:
        raise ValueError("scope alternative budget must be from 1 to 16")
    started = time.monotonic()
    geometry_path = checked_file(input_manifest["source_geometry"])
    geometry = json.loads(geometry_path.read_text())
    if geometry.get("test_opened") is not False:
        raise ValueError("development geometry required")
    pair_evidence = json.loads(checked_file(geometry["evidence_artifact"]).read_text())
    neighbors, relations = competing_scopes(
        geometry["groups"],
        geometry["nodes"],
        pair_evidence["scope_alternatives"],
        evidence.keys(),
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
