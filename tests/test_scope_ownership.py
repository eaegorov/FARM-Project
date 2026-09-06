from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

from farm_runtime.quality.scope_ownership import (
    competing_scopes,
    resolve_scope_alternative,
)
from tools.farm_shaper_bridge.gaussian_lift import load_config, make_claims


def item(oid, ids):
    n = len(ids)
    return SimpleNamespace(
        indices=np.asarray(ids, np.int64),
        positive_weight=np.ones(n, np.float32),
        negative_weight=np.zeros(n, np.float32),
        visible_weight=np.ones(n, np.float32),
        positive_timestamps=np.full(n, 2, np.uint16),
        negative_timestamps=np.zeros(n, np.uint16),
        build_timestamp_count=2,
        obj=SimpleNamespace(object_id=oid, category="object"),
    )


def test_nested_hypothesis_recovers_surface_without_stealing_unrelated_conflict():
    config, _ = load_config(Path("configs/gaussian_lift.v1.yaml"))
    config["refinement"]["enabled"] = False
    config["build"]["minimum_object_gaussians"] = 1
    config["build"]["maximum_object_fraction"] = 0.9
    evidence = {1: item(1, [0, 1, 2, 3]), 2: item(2, [0, 1]), 3: item(3, [2, 4])}
    original = {oid: row.positive_weight.copy() for oid, row in evidence.items()}
    primary = make_claims(evidence, 10, config)[0]
    assert primary[:5].tolist() == [-1, -1, -1, 1, 3]
    bank, audit = resolve_scope_alternative(
        evidence, 1, {2}, np.zeros((10, 3)), np.ones(10), config
    )
    assert bank["object_ids"].tolist() == [1]
    assert bank["indices"].tolist() == [0, 1, 3]
    # Unrelated object 3 still contests Gaussian 2; a nested exclusion is not carte blanche.
    assert 2 not in bank["indices"]
    for oid, value in evidence.items():
        np.testing.assert_array_equal(value.positive_weight, original[oid])
    child, _ = resolve_scope_alternative(
        evidence, 2, {1}, np.zeros((10, 3)), np.ones(10), config
    )
    assert child["indices"].tolist() == [0, 1]
    assert audit["conflicts"]["ambiguous"] == 1


def test_scope_alternatives_require_repeated_same_frame_nesting():
    groups = [dict(id=1, members=[0, 2]), dict(id=2, members=[1, 3])]
    nodes = [dict(id=i, frame=str(i // 2), timestamp=str(i // 2)) for i in range(4)]
    pairs = [dict(a=i, b=i + 1, containments=[0.4, 1.0]) for i in [0, 2]]
    neighbors, rows = competing_scopes(groups, nodes, pairs, {1, 2})
    assert neighbors == {1: {2}, 2: {1}}
    assert all(row["physical_relation"] == "unresolved" for row in rows)
    assert not competing_scopes(groups, nodes, pairs[:1], {1, 2})[0]
    assert not competing_scopes(groups, nodes, pairs, {1})[0]
    pairs[1]["containments"] = [1.0, 0.4]
    assert not competing_scopes(groups, nodes, pairs, {1, 2})[0]


def test_target_cannot_exclude_itself():
    with pytest.raises(ValueError, match="distinct"):
        resolve_scope_alternative(
            {1: item(1, [0])}, 1, {1}, np.zeros((2, 3)), np.ones(2), {}
        )
