"""Conservative, metric surface evidence for associating fixed 2D proposals.

Rendered depth is evidence about the static reconstruction, not a measurement
of every RGB foreground object. Exclusions and unobserved pixels stay unknown.
This module neither assigns native Gaussian ownership nor estimates full OBBs.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy.ndimage import (
    binary_dilation,
    binary_erosion,
    maximum_filter,
    minimum_filter,
)
from scipy.spatial import cKDTree


@dataclass(frozen=True)
class GeometryPolicy:
    duplicate_iou: float = 0.90
    min_points: int = 20
    max_points: int = 1536
    depth_absolute_m: float = 0.04
    depth_relative: float = 0.015
    min_surface_agreement: float = 0.30
    min_visible_fraction: float = 0.15
    min_mask_agreement: float = 0.80
    negative_mask_fraction: float = 0.35
    ambiguity_margin: float = 0.08
    partial_view_association: bool = False
    depth_consistent_proximity: bool = False


def mask_overlap(a, b):
    """Return IoU and both directed containments on the same pixel grid."""
    intersection = int(np.count_nonzero(a & b))
    na, nb = int(np.count_nonzero(a)), int(np.count_nonzero(b))
    return (
        intersection / max(1, na + nb - intersection),
        intersection / max(1, na),
        intersection / max(1, nb),
    )


def equivalent_masks(masks, scores, threshold=0.90, *, priorities=None):
    """Group near-identical masks; every pair must agree, not only a chain.

    The highest scoring representative preserves original pixels. Alternative
    labels are retained by the caller, never used as identity evidence.
    """
    priorities = [0] * len(masks) if priorities is None else list(priorities)
    if len(priorities) != len(masks) or any(
        type(p) is not int or p < 0 for p in priorities
    ):
        raise ValueError("one nonnegative source priority per mask required")
    groups = []
    for index in sorted(
        range(len(masks)), key=lambda i: (priorities[i], -scores[i], i)
    ):
        for group in groups:
            if all(mask_overlap(masks[index], masks[j])[0] >= threshold for j in group):
                group.append(index)
                break
        else:
            groups.append([index])
    return groups


def stable_depth(depth, policy=GeometryPolicy()):
    valid = np.isfinite(depth) & (depth > 0)
    low = minimum_filter(np.where(valid, depth, np.inf), size=3)
    high = maximum_filter(np.where(valid, depth, -np.inf), size=3)
    tolerance = np.maximum(policy.depth_absolute_m, policy.depth_relative * depth)
    return valid & (high - low <= tolerance)


def surface_points(
    mask, depth, K, T_world_cam, excluded, policy=GeometryPolicy(), *, valid_depth=None
):
    """Sample the eroded mask on stable, non-excluded depth in metres."""
    mask = np.asarray(mask, dtype=bool)
    if mask.shape != depth.shape or excluded.shape != depth.shape:
        raise ValueError("mask, depth and exclusion grids must match")
    K, pose = np.asarray(K, float), np.asarray(T_world_cam, float)
    if K.shape != (3, 3) or pose.shape != (4, 4):
        raise ValueError("3x3 K and 4x4 camera-to-world pose required")
    if not np.isfinite(K).all() or not np.isfinite(pose).all():
        raise ValueError("finite camera matrices required")
    if K[0, 0] <= 0 or K[1, 1] <= 0:
        raise ValueError("positive focal lengths required")
    valid_depth = stable_depth(depth, policy) if valid_depth is None else valid_depth
    eligible = binary_erosion(mask) & valid_depth & ~excluded
    yy, xx = np.where(eligible)
    count = len(xx)
    if count > policy.max_points:
        chosen = np.linspace(0, count - 1, policy.max_points).round().astype(int)
        yy, xx = yy[chosen], xx[chosen]
    rays = np.column_stack((xx, yy, np.ones(len(xx)))) @ np.linalg.inv(K).T
    camera = rays * depth[yy, xx, None]
    world = camera @ pose[:3, :3].T + pose[:3, 3]
    pixel_footprint = (
        float(np.median(depth[yy, xx]) / min(K[0, 0], K[1, 1])) if count else 0.0
    )
    return world.astype(np.float32), {
        "mask_pixels": int(mask.sum()),
        "excluded_pixels": int((mask & excluded).sum()),
        "eligible_surface_pixels": count,
        "sampled_points": len(world),
        "pixel_footprint_m": pixel_footprint,
    }


def project_evidence(
    points,
    target_mask,
    depth,
    K,
    T_world_cam,
    excluded,
    policy=GeometryPolicy(),
    *,
    mask_is_dilated=False,
    include_point_indices=False,
):
    """Evaluate only surface-visible points; occlusion/missing depth are unknown."""
    pose = np.asarray(T_world_cam)
    camera = (points - pose[:3, 3]) @ pose[:3, :3]
    positive = camera[:, 2] > 1e-6
    projected = camera @ np.asarray(K).T
    uv = np.zeros((len(points), 2), dtype=int)
    uv[positive] = np.rint(
        projected[positive, :2] / projected[positive, 2, None]
    ).astype(int)
    h, w = depth.shape
    inside = (
        positive & (uv[:, 0] >= 0) & (uv[:, 0] < w) & (uv[:, 1] >= 0) & (uv[:, 1] < h)
    )
    chosen = np.flatnonzero(inside)
    x, y = uv[chosen].T
    known = np.isfinite(depth[y, x]) & (depth[y, x] > 0) & ~excluded[y, x]
    chosen, x, y = chosen[known], x[known], y[known]
    z = camera[chosen, 2]
    surface_depth = depth[y, x]
    tolerance = np.maximum(
        policy.depth_absolute_m, policy.depth_relative * surface_depth
    )
    visible = np.abs(z - surface_depth) <= tolerance
    foreground = z < surface_depth - tolerance
    mask = (
        target_mask if mask_is_dilated else binary_dilation(target_mask, iterations=1)
    )
    supported = int(mask[y[visible], x[visible]].sum())
    total = int(visible.sum())
    result = {
        "source_points": len(points),
        "in_frame": int(inside.sum()),
        "known_depth": len(chosen),
        "surface_visible": total,
        "occluded": int((z > surface_depth + tolerance).sum()),
        "in_front_of_surface": int(foreground.sum()),
        "mask_supported": supported,
        "mask_opposed": total - supported,
        "visible_fraction": total / max(1, len(points)),
        "mask_agreement": supported / max(1, total),
    }

    if include_point_indices:
        result.update(
            out_of_frame_sides=[
                int((positive & (uv[:, 0] < 0)).sum()),
                int((positive & (uv[:, 0] >= w)).sum()),
                int((positive & (uv[:, 1] < 0)).sum()),
                int((positive & (uv[:, 1] >= h)).sum()),
            ],
            surface_indices=chosen[visible],
            surface_pixels_xy=uv[chosen[visible]],
            occluded_indices=chosen[z > surface_depth + tolerance],
            in_front_of_surface_indices=chosen[foreground],
        )
    return result


def depth_proximity_radius(node, frames, policy):
    """Use the same depth tolerance as projection, in prepared scene units."""
    pose = np.asarray(frames[node["frame"]]["T_world_cam"])
    camera = (node["points"] - pose[:3, 3]) @ pose[:3, :3]
    median_depth = float(np.median(camera[:, 2])) if len(camera) else 0.0
    return max(policy.depth_absolute_m, policy.depth_relative * median_depth)


def compare_surfaces(a, b, frames, policy=GeometryPolicy()):
    """Geometry-only pair evidence; no category or appearance similarity."""
    pa, pb = a["points"], b["points"]
    if min(len(pa), len(pb)) < policy.min_points:
        return {"decision": "unknown", "reason": "insufficient_surface"}
    radius = max(0.025, 2 * max(a["pixel_footprint_m"], b["pixel_footprint_m"]))
    if policy.depth_consistent_proximity:
        # Independently sampled depth surfaces can differ within the allowed
        # projection tolerance. Keep bidirectional visibility, mask agreement,
        # ambiguity checks and cannot-links as separate requirements.
        tolerances = [
            (
                node["_depth_proximity_radius_m"]
                if "_depth_proximity_radius_m" in node
                else depth_proximity_radius(node, frames, policy)
            )
            for node in (a, b)
        ]
        radius = max(radius, *tolerances)
    # Reject remote surfaces cheaply before either projection or tree query.
    gap = np.maximum(a["bounds"][0] - b["bounds"][1], b["bounds"][0] - a["bounds"][1])
    if np.any(gap > radius):
        return {"decision": "separate", "reason": "disjoint_surface_bounds"}
    near = [
        b["tree"].query(pa, k=1)[0] <= radius,
        a["tree"].query(pb, k=1)[0] <= radius,
    ]
    ab, ba = [float(np.mean(values)) for values in near]
    directions = []
    for source, target in ((a, b), (b, a)):
        f = frames[target["frame"]]
        directions.append(
            project_evidence(
                source["points"],
                target["projection_mask"],
                f["depth"],
                f["K"],
                f["T_world_cam"],
                f["excluded"],
                policy,
                mask_is_dilated=True,
                include_point_indices=policy.partial_view_association,
            )
        )
    enough = [
        d["surface_visible"] >= policy.min_points
        and d["visible_fraction"] >= policy.min_visible_fraction
        for d in directions
    ]
    negative = any(
        ok and d["mask_agreement"] < 1 - policy.negative_mask_fraction
        for ok, d in zip(enough, directions)
    )
    positive = (
        all(enough)
        and min(ab, ba) >= policy.min_surface_agreement
        and all(d["mask_agreement"] >= policy.min_mask_agreement for d in directions)
    )
    # A cropped image cannot contradict the part of a surface outside its FOV.
    # Enable the alternate denominator only where the target mask reaches the
    # same edge that projected source points cross. Occluded/excluded points
    # remain unknown; they never become mask support.
    partial, effective_agreement, partial_enough = [], [ab, ba], list(enough)
    if policy.partial_view_association:
        for i, (d, target) in enumerate(zip(directions, (b, a))):
            mask = target["mask"]
            touches = [
                mask[:, 0].any(),
                mask[:, -1].any(),
                mask[0].any(),
                mask[-1].any(),
            ]
            eligible = any(
                touch and count > 0
                for touch, count in zip(touches, d["out_of_frame_sides"])
            )
            # A border-clipped part nested in a larger same-image proposal
            # cannot establish the physical extent of an object by itself.
            containers = target.get("scope_containers", [])
            if eligible and containers:
                d["partial_scope_ambiguous_with"] = list(containers)
                eligible = False
            indices = d.pop("surface_indices")
            for key in (
                "surface_pixels_xy",
                "occluded_indices",
                "in_front_of_surface_indices",
            ):
                d.pop(key)
            partial.append(bool(eligible))
            if eligible:
                partial_enough[i] = (
                    d["surface_visible"] >= policy.min_points
                    and d["surface_visible"] / max(1, d["in_frame"])
                    >= policy.min_visible_fraction
                )
                effective_agreement[i] = (
                    float(np.mean(near[i][indices])) if len(indices) else 0.0
                )
        partial_negative = any(
            ok and d["mask_agreement"] < 1 - policy.negative_mask_fraction
            for ok, d in zip(partial_enough, directions)
        )
        partial_positive = (
            any(partial)
            and all(partial_enough)
            and min(effective_agreement) >= policy.min_surface_agreement
            and all(
                d["mask_agreement"] >= policy.min_mask_agreement for d in directions
            )
        )
    else:
        partial_negative = partial_positive = False
    original_positive = positive
    negative = negative or partial_negative
    positive = positive or partial_positive
    extra = {}
    if any(partial):
        extra = {
            "partial_view_directions": partial,
            "partial_surface_agreement": effective_agreement,
            "partial_visible_sufficient": partial_enough,
        }
    score_agreement = (
        effective_agreement if positive and not original_positive else [ab, ba]
    )
    return {
        **extra,
        "decision": "separate" if negative else "match" if positive else "unknown",
        "reason": (
            "visible_mask_conflict"
            if negative
            else (
                (
                    "mutual_surface_support"
                    if original_positive
                    else "partial_view_surface_support"
                )
                if positive
                else "partial_or_ambiguous_support"
            )
        ),
        "surface_agreement": [ab, ba],
        "radius_m": radius,
        "directions": directions,
        "score": min(score_agreement) * min(d["mask_agreement"] for d in directions),
    }


def associate(nodes, frames, policy=GeometryPolicy()):
    """Use mutual per-view matches and propagate contradictions through unions.

    Distinct non-equivalent proposals in one image cannot become one identity.
    Nested proposals remain scope alternatives instead of merging assemblies.
    """
    from scene_graph.map_update.union_find import find_object_correspondence
    import torch

    for node in nodes:
        if policy.depth_consistent_proximity:
            node["_depth_proximity_radius_m"] = depth_proximity_radius(
                node, frames, policy
            )
        node["tree"] = cKDTree(node["points"])
        node["bounds"] = (
            (node["points"].min(0), node["points"].max(0))
            if len(node["points"])
            else (np.zeros(3), np.zeros(3))
        )
        node["projection_mask"] = binary_dilation(node["mask"], iterations=1)
    blocked, matches, diagnostics, scopes = set(), [], [], []
    same_frame_overlaps = {}
    by_frame = {}
    for i, node in enumerate(nodes):
        node["scope_containers"] = []
        by_frame.setdefault(node["frame"], []).append(i)
    for indices in by_frame.values():
        for offset, i in enumerate(indices):
            for j in indices[offset + 1 :]:
                iou, ca, cb = mask_overlap(nodes[i]["mask"], nodes[j]["mask"])
                same_frame_overlaps[i, j] = (iou, ca, cb)
                if ca >= 0.85 > cb and nodes[j].get("source_priority", 0) <= nodes[
                    i
                ].get("source_priority", 0):
                    nodes[i]["scope_containers"].append(j)
                if cb >= 0.85 > ca and nodes[i].get("source_priority", 0) <= nodes[
                    j
                ].get("source_priority", 0):
                    nodes[j]["scope_containers"].append(i)
    for i, a in enumerate(nodes):
        for j in range(i + 1, len(nodes)):
            b = nodes[j]
            if a["frame"] == b["frame"]:
                blocked.add((i, j))
                iou, ca, cb = same_frame_overlaps[i, j]
                if max(ca, cb) >= 0.85:
                    scopes.append(
                        {
                            "a": i,
                            "b": j,
                            "iou": iou,
                            "containments": [ca, cb],
                            "relation": "scope_alternative",
                        }
                    )
                continue
            evidence = compare_surfaces(a, b, frames, policy)
            if evidence["reason"] == "disjoint_surface_bounds":
                continue
            row = dict(a=i, b=j, **evidence)
            diagnostics.append(row)
            if evidence["decision"] == "separate":
                blocked.add((i, j))
            elif evidence["decision"] == "match":
                matches.append(row)
    choices = {}
    for edge in matches:
        for source, target in ((edge["a"], edge["b"]), (edge["b"], edge["a"])):
            choices.setdefault((source, nodes[target]["frame"]), []).append(
                (edge["score"], target, nodes[target].get("source_priority", 0))
            )
    unique = {}
    for key, candidates in choices.items():
        # A complementary detector cannot displace an available primary match.
        # Geometry scores still arbitrate within the best available source.
        priority = min(c[2] for c in candidates)
        ranked = sorted(
            (c for c in candidates if c[2] == priority), key=lambda x: (-x[0], x[1])
        )
        if len(ranked) == 1 or ranked[0][0] - ranked[1][0] >= policy.ambiguity_margin:
            unique[key] = ranked[0][1]
    accepted = []
    for edge in sorted(
        matches,
        key=lambda e: (
            max(
                nodes[e["a"]].get("source_priority", 0),
                nodes[e["b"]].get("source_priority", 0),
            ),
            -e["score"],
            e["a"],
            e["b"],
        ),
    ):
        i, j = edge["a"], edge["b"]
        if (
            unique.get((i, nodes[j]["frame"])) == j
            and unique.get((j, nodes[i]["frame"])) == i
        ):
            accepted.append(edge)
    _, roots = find_object_correspondence(
        [torch.tensor([e["a"], e["b"]]) for e in accepted],
        len(nodes),
        cannot_link_pairs=blocked,
    )
    groups = {}
    for index, root in enumerate(roots.tolist()):
        groups.setdefault(root, []).append(index)
    for edge in accepted:
        edge["component_merge_allowed"] = int(roots[edge["a"]]) == int(roots[edge["b"]])
    return list(groups.values()), {
        "candidate_pairs": diagnostics,
        "mutual_edges": accepted,
        "scope_alternatives": scopes,
        "cannot_link_pairs": sorted(blocked),
        "ambiguous_direction_count": len(choices) - len(unique),
    }
