import itertools
from types import SimpleNamespace
import numpy as np
import pytest
from tools.farm_shaper_bridge.graph_refinement import (
    binary_graph_cut,
    prune_weak_claims,
)


def test_cut_matches_brute_force_energy():
    p = np.array([0.95, 0.55, 0.25, 0.1])
    edges = np.array([[0, 1], [1, 2], [2, 3]])
    weights = np.array([0.2, 0.6, 0.2])

    def energy(labels):
        labels = np.array(labels, bool)
        return (
            -np.log(np.where(labels, p, 1 - p)).sum()
            + weights[labels[edges[:, 0]] != labels[edges[:, 1]]].sum()
        )

    result = binary_graph_cut(p, edges, weights)
    optimum = min(energy(y) for y in itertools.product((False, True), repeat=4))
    assert energy(result) == pytest.approx(optimum)


def test_hard_terminals_survive_opposite_unary_and_large_graph():
    n = 20000
    p = np.full(n, 0.99)
    bg = np.ones(n, bool)
    bg[0] = False
    fg = np.zeros(n, bool)
    fg[0] = True
    p[0] = 0.01
    edges = np.column_stack((np.arange(n - 1), np.arange(1, n)))
    cut = binary_graph_cut(
        p, edges, np.ones(n - 1), force_foreground=fg, force_background=bg
    )
    assert np.array_equal(cut, fg)


def test_graph_pruning_cannot_add_or_reassign_other_object_rows():
    x = np.arange(16) * 0.003
    points = np.column_stack((x, np.zeros_like(x), np.zeros_like(x)))
    item = SimpleNamespace(
        indices=np.arange(16),
        positive_weight=np.ones(16),
        negative_weight=np.zeros(16),
        positive_timestamps=np.full(16, 4),
        negative_timestamps=np.zeros(16),
    )
    owner = np.array([7] * 8 + [9] * 8)
    confidence = np.full(16, 0.9)
    support = np.full(16, 4)
    labels, _, _, report = prune_weak_claims(
        points,
        np.zeros((16, 3)),
        np.full(16, 0.01),
        {7: item},
        owner,
        confidence,
        support,
    )
    assert np.array_equal(labels, owner)
    assert report["objects"][0]["strong_preserved"] == 8


def test_inconsistent_constraints_rejected():
    with pytest.raises(ValueError):
        binary_graph_cut(
            [0.5], [], [], force_foreground=[True], force_background=[True]
        )


def test_sparse_visibility_without_negative_evidence_is_preserved():
    points = np.column_stack((np.arange(10) * 0.006, np.zeros(10), np.zeros(10)))
    item = SimpleNamespace(
        indices=np.arange(10),
        positive_weight=np.ones(10),
        negative_weight=np.zeros(10),
        positive_timestamps=np.ones(10),
        negative_timestamps=np.zeros(10),
    )
    owner = np.full(10, 7)
    labels, _, _, _ = prune_weak_claims(
        points,
        np.zeros((10, 3)),
        np.full(10, 0.01),
        {7: item},
        owner,
        np.full(10, 0.7),
        np.ones(10),
    )
    assert np.array_equal(labels, owner)
