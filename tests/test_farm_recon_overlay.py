from __future__ import annotations

import importlib.util
from pathlib import Path

import sys
import numpy as np
import pytest


SCRIPT = Path(__file__).parents[1] / "scripts/serve_farm_recon_overlay.py"
SPEC = importlib.util.spec_from_file_location("serve_farm_recon_overlay", SCRIPT)
assert SPEC is not None and SPEC.loader is not None
MODULE = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = MODULE
SPEC.loader.exec_module(MODULE)


def test_stable_stratified_indices_preserve_all_objects_and_budget() -> None:
    labels = np.repeat(np.asarray([4, 9, 21], dtype=np.int32), [10, 100, 1000])
    first = MODULE.stable_stratified_indices(labels, 111, minimum_per_object=8)
    second = MODULE.stable_stratified_indices(labels, 111, minimum_per_object=8)
    assert np.array_equal(first, second)
    assert len(first) == 111
    assert len(np.unique(first)) == 111
    assert set(labels[first].tolist()) == {4, 9, 21}
    assert np.all(first[:-1] < first[1:])


def test_stable_stratified_indices_rejects_dropped_object_budget() -> None:
    labels = np.asarray([1, 2, 3], dtype=np.int32)
    with pytest.raises(MODULE.UnifiedViewerError, match="at least one row"):
        MODULE.stable_stratified_indices(labels, 2)


def test_stable_stratified_indices_rejects_negative_ids() -> None:
    with pytest.raises(MODULE.UnifiedViewerError, match="invalid or missing"):
        MODULE.stable_stratified_indices(np.asarray([-1, 2], dtype=np.int32), 2)


def test_instance_palette_is_only_deterministic_object_id_colour() -> None:
    ids = np.asarray([17, 17, 445, 445], dtype=np.int32)
    colours = np.asarray([MODULE._instance_color(int(value)) for value in ids])
    assert np.array_equal(colours[0], colours[1])
    assert np.array_equal(colours[2], colours[3])
    assert not np.array_equal(colours[0], colours[2])
