"""Keep continuous scene-surface hypotheses separate from inventory candidates.

Exact detector categories are a conservative routing prior, not physical truth.
Mixed categories, fixtures, movable panels and unknown observations stay eligible
as objects. Source geometry remains the authority for the context masks.
"""

from __future__ import annotations

import json
from pathlib import Path

from farm_runtime.quality.mask_refinement import checked_file

SURFACE_LABELS = frozenset({"ceiling", "floor", "wall", "ground", "roof"})
POLICY = "unanimous_continuous_surface_labels_at_independent_times.v1"


def surface_context(geometry):
    if geometry.get("test_opened") is not False:
        raise ValueError("development geometry required")
    nodes = {n["id"]: n for n in geometry["nodes"]}
    if len(nodes) != len(geometry["nodes"]):
        raise ValueError("unique geometry node IDs required")
    rows, seen = [], set()
    for group in geometry["groups"]:
        oid = group["id"]
        if oid in seen:
            raise ValueError("unique geometry group IDs required")
        seen.add(oid)
        members = [nodes[n] for n in group["members"]]
        timestamps = {n["timestamp"] for n in members}
        if len(timestamps) != group["independent_timestamps"]:
            raise ValueError("context evidence timestamp count differs")
        labels = set()
        valid = bool(members)
        for node in members:
            values = node.get("labels")
            if (
                not isinstance(values, list)
                or not values
                or any(not isinstance(v, str) or not v.strip() for v in values)
            ):
                valid = False
                continue
            labels.update(v.strip().casefold() for v in values)
        if not valid or len(timestamps) < 2:
            continue
        declared = {v.strip().casefold() for v in group["candidate_labels"]}
        if labels != declared:
            raise ValueError("context categories differ from observation evidence")
        if labels and labels <= SURFACE_LABELS:
            rows.append(
                dict(
                    object_id=oid,
                    category_hypotheses=sorted(labels),
                    observation_node_ids=group["members"],
                    physical_timestamps=sorted(timestamps),
                    role="scene_surface_hypothesis",
                    physical_role_validated=False,
                )
            )
    return sorted(rows, key=lambda r: r["object_id"])


def bind_context_selection(path, geometry_record, object_ids):
    """Bind retained context proposals to the same geometry as the native bank."""
    selection = json.loads(Path(path).read_text())
    source = selection.get("source_geometry")
    if not source or source["sha256"] != geometry_record["sha256"]:
        raise ValueError("context selection belongs to another geometry")
    geometry = json.loads(checked_file(source).read_text())
    if selection.get("context_policy") != POLICY:
        raise ValueError("explicit scene context routing policy required")
    available = {r["object_id"]: r for r in surface_context(geometry)}
    rows = selection["context_groups"]
    ids = [r["object_id"] for r in rows]
    if (
        len(ids) != len(set(ids))
        or any(r != available.get(r["object_id"]) for r in rows)
        or set(ids) & set(object_ids)
        or not set(object_ids) <= set(selection["selected_group_ids"])
    ):
        raise ValueError("context selection and native object cohort differ")
    return rows
