"""Contributor support remains available before mutually exclusive ownership."""

from pathlib import Path
from types import SimpleNamespace

import numpy as np

from tools.farm_shaper_bridge.gaussian_lift import (
    build_sparse_claims,
    load_config,
    make_claims,
)


def config():
    value, _ = load_config(Path("configs/gaussian_lift.v1.yaml"))
    value["refinement"]["enabled"] = False
    value["build"].update(
        minimum_object_gaussians=1,
        maximum_object_fraction=0.9,
        minimum_gaussian_timestamps=2,
        minimum_timestamp_margin=1,
        minimum_global_purity=0.58,
    )
    return value


def evidence(oid, ids, *, positive=2, negative=0):
    count = len(ids)
    return SimpleNamespace(
        obj=SimpleNamespace(object_id=oid, category="object"),
        indices=np.asarray(ids, np.int64),
        positive_weight=np.ones(count, np.float32),
        negative_weight=np.zeros(count, np.float32),
        visible_weight=np.ones(count, np.float32),
        positive_timestamps=np.full(count, positive, np.uint16),
        negative_timestamps=np.full(count, negative, np.uint16),
        build_timestamp_count=max(2, positive + negative),
    )


def test_supported_claims_keep_multiview_purity_and_visible_share_gates():
    policy = config()
    policy["build"]["minimum_visible_share"] = 0.5
    item = evidence(7, [0, 1, 2, 3, 4, 5])
    # Independently reject a single-view contribution, contradictory timestamps,
    # mostly-background contribution, and a small positive tail on a visible splat.
    item.positive_timestamps[1] = 1
    item.negative_timestamps[2] = 2
    item.negative_weight[3] = 2
    item.visible_weight[4] = 4
    # Exact threshold boundary remains accepted.
    item.visible_weight[5] = 2
    claims, rows = build_sparse_claims({7: item}, policy)
    assert claims[0]["indices"].tolist() == [0, 5]
    assert claims[0]["support"].tolist() == [2, 2]
    assert rows[0]["supported_claims_before_conflicts"] == 2
    assert rows[0]["claims_rejected_visible_share"] == 1
    # Omitted optional visibility guard preserves the historical input policy.
    del policy["build"]["minimum_visible_share"]
    unguarded, _ = build_sparse_claims({7: item}, policy)
    assert unguarded[0]["indices"].tolist() == [0, 4, 5]


def test_nested_claims_survive_extraction_but_exclusive_ties_stay_unknown():
    policy = config()
    objects = {
        1: evidence(1, [0, 1, 2, 3]),
        2: evidence(2, [0, 1]),
        3: evidence(3, [2, 4], positive=3),
    }
    claims, rows = build_sparse_claims(objects, policy)
    assert [r["indices"].tolist() for r in claims] == [[0, 1, 2, 3], [0, 1], [2, 4]]
    labels, _, _, resolved_rows, conflicts, blocked = make_claims(objects, 10, policy)
    # Equal object/part evidence does not pick an arbitrary low ID. The third
    # object's stronger independent evidence still wins the unrelated conflict.
    assert labels[:5].tolist() == [-1, -1, 3, 1, 3]
    assert blocked.tolist() == [0, 1]
    assert conflicts["ambiguous"] == 2
    assert [r["supported_claims_before_conflicts"] for r in rows] == [4, 2, 2]
    assert [r["resolved_core_gaussians"] for r in resolved_rows] == [1, 0, 2]
    assert [r["indices"].tolist() for r in claims] == [[0, 1, 2, 3], [0, 1], [2, 4]]


def test_claim_extraction_is_order_stable_and_does_not_mutate_evidence():
    policy = config()
    objects = {8: evidence(8, [2, 4]), 2: evidence(2, [0, 1])}
    before = {
        oid: {key: value.copy() for key, value in vars(item).items() if isinstance(value, np.ndarray)}
        for oid, item in objects.items()
    }
    claims, rows = build_sparse_claims(objects, policy)
    repeated, repeated_rows = build_sparse_claims(dict(reversed(list(objects.items()))), policy)
    assert [r["object_id"] for r in claims] == [2, 8]
    assert rows == repeated_rows
    for first, second in zip(claims, repeated):
        for key in ("indices", "scores", "confidence", "support"):
            np.testing.assert_array_equal(first[key], second[key])
    # A scope selector can manipulate returned arrays without corrupting the
    # evidence that unrelated ownership and alternate hypotheses still require.
    for claim in claims:
        for key in ("indices", "scores", "confidence", "support"):
            claim[key][:] = 0
    for oid, arrays in before.items():
        for key, original in arrays.items():
            np.testing.assert_array_equal(getattr(objects[oid], key), original)
