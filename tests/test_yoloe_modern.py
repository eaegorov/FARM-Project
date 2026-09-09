from types import SimpleNamespace

import numpy as np
from packaging.version import Version
import pytest
import torch
import ultralytics

from farm_runtime.quality.yoloe_modern import load_text_model, soft_mask_predictor


def test_modern_checkpoint_refuses_legacy_runtime_before_loading(tmp_path, monkeypatch):
    monkeypatch.setattr(ultralytics, "__version__", "8.3.39")
    with pytest.raises(ValueError, match="separate Ultralytics"):
        load_text_model(tmp_path / "missing.pt", tmp_path / "vocab", tmp_path)


@pytest.mark.skipif(
    Version(ultralytics.__version__) < Version("8.4.0"),
    reason="native parity runs in the isolated modern runtime",
)
@pytest.mark.parametrize("shape", [(31, 63), (64, 32), (53, 53)])
@pytest.mark.parametrize("empty", [False, True])
def test_modern_soft_result_preserves_native_masks_boxes_and_empty_filter(shape, empty):
    from ultralytics.models.yolo.segment.predict import SegmentationPredictor

    h, w = shape
    # Constant prototypes ensure one detection has a wholly empty mask, while
    # fractional boxes and non-square resizing exercise native crop semantics.
    proto = torch.ones((1, 8, 8))
    pred = torch.tensor(
        [[1.2, 2.8, 29.7, 29.2, 0.9, 0, 1.0], [0.0, 0.0, 32.0, 32.0, 0.8, 0, -1.0]]
    )
    if empty:
        pred = pred[:0]
    original = np.zeros((h, w, 3), np.uint8)
    image = torch.zeros((1, 3, 32, 32))
    outputs = []
    for cls in (SegmentationPredictor, soft_mask_predictor()):
        predictor = object.__new__(cls)
        predictor.args = SimpleNamespace(retina_masks=True)
        predictor.model = SimpleNamespace(names={0: "object"})
        outputs.append(
            predictor.construct_result(pred.clone(), image, original, "test.jpg", proto)
        )
    native, soft = outputs
    assert torch.equal(native.boxes.data, soft.boxes.data)
    if empty:
        assert native.masks is soft.masks is soft.farm_mask_logits is None
    else:
        assert len(soft.boxes) == 1
        assert torch.equal(native.masks.data.bool(), soft.masks.data.bool())
        assert torch.equal(soft.farm_mask_logits > 0, native.masks.data.bool())
        assert torch.isfinite(soft.farm_mask_logits).all()
