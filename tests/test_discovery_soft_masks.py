import sys
from pathlib import Path

import pytest
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "third_party/yoloe"))
from ultralytics.utils import ops
from farm_runtime.quality.discovery import decode_soft_masks


@pytest.mark.parametrize("shape", [(31, 63), (64, 32), (53, 53)])
def test_soft_masks_match_pinned_native_decoder_on_non_square_and_fractional_boxes(
    shape,
):
    generator = torch.Generator().manual_seed(91)
    proto = torch.randn(4, 8, 8, generator=generator)
    coefficients = torch.randn(3, 4, generator=generator)
    h, w = shape
    boxes = torch.tensor(
        [[1.2, 2.8, w - 2.3, h - 1.5], [0.0, 0.0, w, h], [w / 3, h / 3, w / 2, h / 2]]
    )
    expected = ops.process_mask_native(proto, coefficients, boxes, shape)
    actual = decode_soft_masks(proto, coefficients, boxes, shape)
    assert torch.equal(actual > 0, expected.bool())
    assert torch.any((actual > -20) & (actual < 0))
    assert torch.any(actual > 0)
