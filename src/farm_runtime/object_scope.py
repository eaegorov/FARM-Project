"""Observed scope relationships, kept separate from physical object ownership."""

from collections import defaultdict


def aggregate_scope_containment(groups, nodes, pairs):
    """Count directed containment across timestamps without inferring part-of.

    A poster inside a wall silhouette is not an integral part of that wall.
    Opposite directions remain an explicit conflict; no identity is merged.
    """
    membership = {}
    for group in groups:
        for member in group["members"]:
            if member in membership:
                raise ValueError("a proposal node belongs to multiple groups")
            membership[member] = group["id"]
    by_id = {n["id"]: n for n in nodes}
    edges = defaultdict(list)
    for pair in pairs:
        a, b = pair["a"], pair["b"]
        if by_id[a]["frame"] != by_id[b]["frame"]:
            raise ValueError("2D containment requires the same source image")
        if by_id[a]["timestamp"] != by_id[b]["timestamp"]:
            raise ValueError("a source image must have one physical timestamp")
        ca, cb = pair["containments"]
        if ca >= 0.85 and cb < 0.80:
            child, parent = a, b
        elif cb >= 0.85 and ca < 0.80:
            child, parent = b, a
        else:
            continue  # Similar-size overlaps do not establish a direction.
        child_group, parent_group = membership[child], membership[parent]
        if child_group == parent_group:
            raise ValueError("distinct nested scopes were assigned one identity")
        edges[parent_group, child_group].append(
            {
                "frame": by_id[a]["frame"],
                "timestamp": by_id[a]["timestamp"],
                "parent_node": parent,
                "child_node": child,
                "child_coverage": max(ca, cb),
                "parent_coverage": min(ca, cb),
            }
        )
    rows = []
    for (parent, child), evidence in sorted(edges.items()):
        timestamps = sorted({r["timestamp"] for r in evidence})
        conflict = (child, parent) in edges
        rows.append(
            {
                "outer_group": parent,
                "inner_group": child,
                "independent_timestamps": len(timestamps),
                "timestamps": timestamps,
                "status": (
                    "conflicting_directions"
                    if conflict
                    else (
                        "multiview_containment"
                        if len(timestamps) >= 2
                        else "single_timestamp"
                    )
                ),
                "relation": "observed_mask_containment",
                "physical_relation": "unresolved",
                "evidence": evidence,
            }
        )
    return rows
