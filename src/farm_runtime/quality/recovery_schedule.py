"""Prioritize new observed regions within bounded source/size recovery strata."""

from collections import defaultdict

import numpy as np


def _novelty_reader(geometry, nodes, allowed_names, mask_reader):
    # Multiview association is only a coverage hint, not accepted object identity.
    # Never exclude candidates on this evidence, including nested object scopes.
    references = defaultdict(list)
    for group in geometry["groups"]:
        if group["independent_timestamps"] < 2 or any(
            nodes[i]["frame"] not in allowed_names for i in group["members"]
        ):
            continue
        for nid in group["members"]:
            node = nodes[nid]
            if node["representative_detection"] is not None:
                references[node["frame"]].append((group["id"], node))
    for rows in references.values():
        rows.sort(key=lambda row: (row[0], row[1]["id"]))
    cache = {}

    def region(node):
        if node["id"] not in cache:
            mask = np.asarray(mask_reader(node))
            if mask.ndim != 2 or mask.dtype != np.bool_:
                raise ValueError("novelty requires two-dimensional boolean masks")
            y, x = np.nonzero(mask)
            box = (
                (int(x.min()), int(y.min()), int(x.max()) + 1, int(y.max()) + 1)
                if len(x)
                else (0, 0, 0, 0)
            )
            cache[node["id"]] = (mask, box, len(x))
        return cache[node["id"]]

    def overlap(node):
        best = dict(
            iou=0.0,
            group_id=None,
            node_id=None,
            candidate_coverage=0.0,
            reference_coverage=0.0,
        )
        if not references[node["frame"]]:
            return best
        mask, box, area = region(node)
        for gid, refnode in references[node["frame"]]:
            ref, rb, ref_area = region(refnode)
            if mask.shape != ref.shape:
                raise ValueError("same-frame novelty masks must have identical shapes")
            x0, y0 = max(box[0], rb[0]), max(box[1], rb[1])
            x1, y1 = min(box[2], rb[2]), min(box[3], rb[3])
            if x0 >= x1 or y0 >= y1:
                continue
            intersection = int(np.count_nonzero(mask[y0:y1, x0:x1] & ref[y0:y1, x0:x1]))
            iou = intersection / (area + ref_area - intersection)
            if iou > best["iou"]:
                best = dict(
                    iou=iou,
                    group_id=gid,
                    node_id=refnode["id"],
                    candidate_coverage=intersection / area,
                    reference_coverage=intersection / ref_area,
                )
        return best

    return overlap


def recovery_candidates(geometry, budget, allowed_names, *, mask_reader=None):
    """Rank support times image novelty; omitted mask_reader retains support-only.

    Maximum same-frame IoU softly deprioritizes repeated regions. A small object
    inside a large surface still has low IoU. This never merges or rejects groups,
    changes geometry gates, or treats view selection as new identity evidence.
    """
    if type(budget) is not int or not 1 <= budget <= 128:
        raise ValueError("recovery candidate budget must be in 1..128")
    nodes = {n["id"]: n for n in geometry["nodes"]}
    overlap = (
        _novelty_reader(geometry, nodes, allowed_names, mask_reader)
        if mask_reader is not None
        else None
    )
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
        novelty = overlap(best) if overlap is not None else None
        support = best["eligible_surface_pixels"] / best["mask_pixels"]
        priority = best.get("source_priority", 0)
        by_source[priority].append(
            dict(
                group_id=group["id"],
                source_priority=priority,
                mask_pixels=best["mask_pixels"],
                eligible_surface_fraction=support,
                source_node_id=best["id"],
                multiview_overlap=novelty,
                novel_support_score=(
                    support * (1 - novelty["iou"]) if novelty else support
                ),
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
                -r["novel_support_score"],
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
