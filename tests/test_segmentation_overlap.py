from __future__ import annotations

import pytest
import torch

import importlib.util
from pathlib import Path

MODULE_PATH = (
    Path(__file__).parents[1] / "src/scene_graph/segmentation/overlap.py"
)
SPEC = importlib.util.spec_from_file_location("farm_segmentation_overlap", MODULE_PATH)
assert SPEC is not None and SPEC.loader is not None
MODULE = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(MODULE)
suppress_contained_masks = MODULE.suppress_contained_masks


def test_suppresses_lower_score_part_inside_complete_object() -> None:
    masks = torch.zeros((3, 64, 64), dtype=torch.bool)
    masks[0, 8:56, 8:56] = True
    masks[1, 20:34, 20:34] = True
    masks[2, 2:14, 50:62] = True
    keep = suppress_contained_masks(masks, torch.tensor([0.8, 0.6, 0.5]))
    assert keep.tolist() == [0, 2]


def test_preserves_adjacent_and_partially_overlapping_objects() -> None:
    masks = torch.zeros((3, 80, 80), dtype=torch.bool)
    masks[0, 5:35, 5:35] = True
    masks[1, 30:60, 30:60] = True
    masks[2, 5:35, 45:75] = True
    keep = suppress_contained_masks(masks, torch.tensor([0.9, 0.8, 0.7]))
    assert keep.tolist() == [0, 1, 2]


def test_preserves_similar_area_duplicate_for_box_nms_to_handle() -> None:
    masks = torch.zeros((2, 64, 64), dtype=torch.bool)
    masks[0, 5:55, 5:55] = True
    masks[1, 8:54, 8:54] = True
    keep = suppress_contained_masks(masks, torch.tensor([0.9, 0.8]))
    assert keep.tolist() == [0, 1]


def test_rejects_misaligned_contract() -> None:
    with pytest.raises(ValueError, match="row-aligned"):
        suppress_contained_masks(
            torch.zeros((2, 16, 16), dtype=torch.bool), torch.ones(1)
        )
