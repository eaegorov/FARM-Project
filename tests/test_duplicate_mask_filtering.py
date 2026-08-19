import torch

from scene_graph.map_update.filtering import (
    filter_detections_duplicates_iou,
    filter_duplicate_masks_iou,
)


def _nested_masks(*, small_first: bool = False) -> torch.Tensor:
    large = torch.zeros((8, 8), dtype=torch.bool)
    large[1:7, 1:7] = True
    small = torch.zeros((8, 8), dtype=torch.bool)
    small[2:6, 2:6] = True
    separate = torch.zeros((8, 8), dtype=torch.bool)
    separate[0:2, 6:8] = True
    ordered = [small, large, separate] if small_first else [large, small, separate]
    return torch.stack(ordered)


def test_containment_suppression_keeps_larger_mask() -> None:
    keep = filter_duplicate_masks_iou(
        _nested_masks(small_first=True),
        min_iou=0.9,
        min_containment=0.98,
    )
    assert keep.tolist() == [False, True, True]


def test_containment_can_be_disabled_for_legacy_iou_only_mode() -> None:
    keep = filter_duplicate_masks_iou(
        _nested_masks(),
        min_iou=0.9,
        min_containment=1.0,
    )
    assert keep.tolist() == [True, True, True]


def test_detection_filter_preserves_interleaved_batch_order() -> None:
    masks = _nested_masks()
    seg_outputs = {
        "masks": [masks[0], masks[2], masks[1]],
        "batch_ids": torch.tensor([0, 1, 0]),
        "scores": torch.tensor([0.9, 0.8, 0.7]),
        "class_ids": torch.tensor([10, 20, 30]),
    }
    filtered = filter_detections_duplicates_iou(
        seg_outputs,
        min_iou=0.9,
        min_containment=0.98,
    )
    assert filtered["class_ids"].tolist() == [10, 20]
