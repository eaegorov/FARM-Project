from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

import numpy as np
import pytest


SCRIPT = Path(__file__).parents[1] / "scripts/refine_farm_recon_orientation.py"
SPEC = importlib.util.spec_from_file_location("refine_farm_recon_orientation", SCRIPT)
assert SPEC is not None and SPEC.loader is not None
MODULE = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = MODULE
SPEC.loader.exec_module(MODULE)


@pytest.mark.parametrize(
    ("source", "target"),
    [
        ([0.0, 0.0, 1.0], [0.0, 1.0, 0.0]),
        ([1.0, 0.0, 0.0], [-1.0, 0.0, 0.0]),
        ([0.2, -0.4, 0.9], [-0.1, 0.95, 0.2]),
    ],
)
def test_minimal_row_rotation_maps_directions_and_preserves_handedness(
    source: list[float], target: list[float]
) -> None:
    rotation = MODULE.minimal_row_rotation(source, target)
    actual = MODULE._unit(np.asarray(source) @ rotation, name="actual")
    expected = MODULE._unit(target, name="expected")
    assert np.allclose(actual, expected, atol=1.0e-7)
    assert np.allclose(rotation.T @ rotation, np.eye(3), atol=1.0e-8)
    assert np.linalg.det(rotation) == pytest.approx(1.0, abs=1.0e-8)


def test_fractional_rotation_has_expected_half_angle() -> None:
    full = MODULE.minimal_row_rotation([0, 0, 1], [0, 1, 0])
    half = MODULE.fractional_row_rotation(full, 0.5)
    direction = np.asarray([0.0, 0.0, 1.0]) @ half
    assert MODULE.vector_angle_degrees(direction, [0, 1, 0]) == pytest.approx(45.0)


def test_choose_strongest_mask_supported_correction() -> None:
    fractions = np.asarray([0.0, 0.5, 0.8, 1.0])
    scores = [0.60, 0.66, 0.67, 0.666]
    assert MODULE.choose_correction_fraction(
        fractions,
        scores,
        angle_degrees=40.0,
        tolerance=0.005,
        minimum_angle_degrees=3.0,
    ) == 1.0


def test_choose_keeps_small_tilt_even_with_numeric_score_noise() -> None:
    assert MODULE.choose_correction_fraction(
        [0.0, 0.5, 1.0],
        [0.60, 0.61, 0.62],
        angle_degrees=2.9,
        tolerance=0.005,
        minimum_angle_degrees=3.0,
    ) == 0.0


def test_binary_iou_is_exact() -> None:
    first = np.asarray([[1, 1], [0, 0]], dtype=bool)
    second = np.asarray([[1, 0], [1, 0]], dtype=bool)
    assert MODULE.binary_iou(first, second) == pytest.approx(1.0 / 3.0)
