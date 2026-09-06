"""Bound recovery work across detector sources and object image sizes."""

from collections import defaultdict


def recovery_candidates(geometry, budget, allowed_names):
    if type(budget) is not int or not 1 <= budget <= 128:
        raise ValueError("recovery candidate budget must be in 1..128")
    nodes = {n["id"]: n for n in geometry["nodes"]}
    by_source = defaultdict(list)
    for group in geometry["groups"]:
        if group["independent_timestamps"] != 1:
            continue
        members = [nodes[i] for i in group["members"]]
        if any(n["frame"] not in allowed_names for n in members):
            continue
        supported = [
            n
            for n in members
            if n["representative_detection"] is not None
            and n["eligible_surface_pixels"] >= 20
            and n["sampled_points"] >= 20
            and n["mask_pixels"] > 0
        ]
        if not supported:
            continue
        best = max(
            supported,
            key=lambda n: (n["eligible_surface_pixels"] / n["mask_pixels"], -n["id"]),
        )
        priority = best.get("source_priority", 0)
        by_source[priority].append(
            dict(
                group_id=group["id"],
                source_priority=priority,
                mask_pixels=best["mask_pixels"],
                eligible_surface_fraction=best["eligible_surface_pixels"]
                / best["mask_pixels"],
            )
        )
    strata = {}
    for priority, rows in sorted(by_source.items()):
        # Relative size thirds keep a scene dominated by large surfaces from
        # consuming every recovery slot. Detector scores/labels are not compared.
        rows.sort(key=lambda r: (r["mask_pixels"], r["group_id"]))
        for index, row in enumerate(rows):
            band = min(2, 3 * index // len(rows))
            row["size_band"] = band
            strata.setdefault((priority, band), []).append(row)
    for rows in strata.values():
        rows.sort(
            key=lambda r: (
                -r["eligible_surface_fraction"],
                r["mask_pixels"],
                r["group_id"],
            )
        )
    ordered = []
    for index in range(max(map(len, strata.values()), default=0)):
        for key in sorted(strata):
            if index < len(strata[key]):
                ordered.append(strata[key][index])
    return ordered[:budget], len(ordered)


def budgeted_views(selected, used_names, view_budget):
    """Keep independent selected timestamps within the shared RGB budget."""
    used = set(used_names)
    kept = []
    for row in selected:
        if row["name"] in used or len(used) < view_budget:
            kept.append(row)
            used.add(row["name"])
    return kept
