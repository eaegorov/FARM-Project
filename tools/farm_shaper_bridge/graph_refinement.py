"""Experimental local Gaussian graph cut using build-only evidence.

Inspired by GaussianCut (arXiv:2411.07555), independently implemented. This
prune-only pilot preserves strong multiview support and all source row IDs.
"""

from __future__ import annotations

import time
import numpy as np
from scipy.spatial import cKDTree
from scipy.sparse import coo_matrix
from scipy.sparse.csgraph import maximum_flow, breadth_first_order


def binary_graph_cut(
    probability, edges, edge_weights, *, force_foreground=None, force_background=None
):
    """Minimize Bernoulli unary NLL plus a symmetric Potts disagreement cost."""
    p = np.asarray(probability, np.float64)
    edges = np.asarray(edges, np.int64).reshape(-1, 2)
    weights = np.asarray(edge_weights, np.float64)
    n = len(p)
    if p.ndim != 1 or not np.isfinite(p).all() or np.any((p < 0) | (p > 1)):
        raise ValueError("finite probabilities in [0,1] required")
    if (
        weights.shape != (len(edges),)
        or not np.isfinite(weights).all()
        or np.any(weights < 0)
    ):
        raise ValueError("nonnegative finite pairwise weights required")
    if np.any((edges < 0) | (edges >= n)):
        raise ValueError("graph edge outside candidate rows")
    fg = (
        np.zeros(n, bool)
        if force_foreground is None
        else np.asarray(force_foreground, bool)
    )
    bg = (
        np.zeros(n, bool)
        if force_background is None
        else np.asarray(force_background, bool)
    )
    if fg.shape != (n,) or bg.shape != (n,) or np.any(fg & bg):
        raise ValueError("inconsistent terminal constraints")
    eps = 1e-6
    # Source side is foreground: source->node pays the background cost.
    source_cost = -np.log(np.clip(1 - p, eps, 1))
    sink_cost = -np.log(np.clip(p, eps, 1))
    incident = np.bincount(
        edges.reshape(-1), weights=np.repeat(weights, 2), minlength=n
    )
    infinity = source_cost + sink_cost + incident + 1.0
    source_cost[fg] = infinity[fg]
    sink_cost[bg] = infinity[bg]
    rows = np.concatenate((np.full(n, n), np.arange(n), edges[:, 0], edges[:, 1]))
    cols = np.concatenate((np.arange(n), np.full(n, n + 1), edges[:, 1], edges[:, 0]))
    values = np.concatenate((source_cost, sink_cost, weights, weights))
    # SciPy max-flow uses integer capacities; 1e-4 energy precision is enough
    # for this bounded engineering pilot. Keep terminal capacities in int64.
    capacities = coo_matrix(
        (np.rint(values * 10000).astype(np.int64), (rows, cols)), shape=(n + 2, n + 2)
    ).tocsr()
    capacities.eliminate_zeros()
    flow = maximum_flow(capacities, n, n + 1, method="dinic")
    residual = capacities - flow.flow
    residual.data = (residual.data > 0).astype(np.int64)
    residual.eliminate_zeros()
    reached = breadth_first_order(residual, n, directed=True, return_predecessors=False)
    result = np.zeros(n, bool)
    result[reached[reached < n]] = True
    return result


def prune_weak_claims(
    means, colors, radius, evidence, owner, confidence, support, *, smoothness=0.35
):
    started = time.monotonic()
    owner, confidence, support = owner.copy(), confidence.copy(), support.copy()
    rows = []
    for object_id, item in sorted(evidence.items()):
        ids = item.indices
        observed = (item.positive_weight + item.negative_weight) > 1e-6
        # Unobserved points cannot become foreground. Include observed local
        # background to anchor cuts around uncertain current claims.
        local = np.flatnonzero(observed)
        current = owner[ids[local]] == object_id
        if current.sum() < 8:
            continue
        points = means[ids[local]]
        pweight, nweight = item.positive_weight[local], item.negative_weight[local]
        purity = pweight / np.maximum(pweight + nweight, 1e-8)
        positive_ts = item.positive_timestamps[local]
        negative_ts = item.negative_timestamps[local]
        # Calibrated only as a ranking energy, not a probability claim. A
        # one-timestamp pure observation has weak odds instead of a hard label.
        reliability = np.minimum(positive_ts.astype(float) / 3, 1)
        probability = 0.5 + (purity - 0.5) * reliability
        strong = (
            current
            & (positive_ts >= 3)
            & (purity >= 0.9)
            & (positive_ts >= negative_ts + 2)
        )
        # Lack of additional positive viewpoints is not negative evidence:
        # a real thin/occluded surface may be visible at only one timestamp.
        # Only independently contradicted points may be removed by this pilot.
        strong |= current & (negative_ts < 2)
        background = (~current) | (negative_ts >= positive_ts + 2)
        background &= ~strong
        tree = cKDTree(points)
        distance, neighbor = tree.query(points, k=min(9, len(points)), workers=1)
        a = np.repeat(np.arange(len(points)), neighbor.shape[1] - 1)
        b = neighbor[:, 1:].reshape(-1)
        dist = distance[:, 1:].reshape(-1)
        limit = np.minimum(
            0.06, np.maximum(0.008, 2 * (radius[ids[local]][a] + radius[ids[local]][b]))
        )
        valid = (a != b) & (dist <= limit)
        pairs = np.sort(np.column_stack((a[valid], b[valid])), axis=1)
        edges, first = np.unique(pairs, axis=0, return_index=True)
        dist, limit = dist[valid][first], limit[valid][first]
        rgb = colors[ids[local]]
        dc = np.linalg.norm(rgb[edges[:, 0]] - rgb[edges[:, 1]], axis=1)
        pair = smoothness * (
            0.7 * np.exp(-2 * (dist / limit) ** 2)
            + 0.3 * np.exp(-0.5 * (dc / 0.12) ** 2)
        )
        kept = binary_graph_cut(
            probability,
            edges,
            pair,
            force_foreground=strong,
            force_background=background,
        )
        removed = ids[local[current & ~kept]]
        owner[removed] = -1
        confidence[removed] = 0
        support[removed] = 0
        rows.append(
            dict(
                object_id=object_id,
                candidates=len(local),
                edges=len(edges),
                before=int(current.sum()),
                strong_preserved=int(strong.sum()),
                removed=len(removed),
                after=int(current.sum()) - len(removed),
            )
        )
    return (
        owner,
        confidence,
        support,
        dict(
            schema="farm.local-graph-pruning.v1",
            objects=rows,
            seconds=time.monotonic() - started,
            smoothness=smoothness,
            prune_only=True,
            heldout_used=False,
            source_order_preserved=True,
        ),
    )
