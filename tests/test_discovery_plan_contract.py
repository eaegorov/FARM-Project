from types import SimpleNamespace

from PIL import Image
import pytest

from farm_runtime.quality.discovery import predictor_parity, validate_inference_plan
from farm_runtime.quality_baseline import describe_file


def source_plan(tmp_path):
    image = tmp_path / "source.jpg"
    Image.new("RGB", (32, 32)).save(image)
    return dict(
        test_opened=False,
        variants=[dict(name="balanced_upright", views=["source.jpg"], resolution=640)],
        sources={
            "source.jpg": dict(
                name="source.jpg", timestamp="1", source_image=describe_file(image)
            )
        },
    )


def test_prepared_detector_plan_rejects_changed_image_and_heldout_input(tmp_path):
    plan = source_plan(tmp_path)
    validate_inference_plan(plan)
    heldout = dict(plan, test_opened=True)
    with pytest.raises(ValueError, match="development"):
        validate_inference_plan(heldout)
    (tmp_path / "source.jpg").write_bytes(b"changed")
    with pytest.raises(ValueError, match="hash mismatch"):
        validate_inference_plan(plan)


def test_prepared_detector_plan_cannot_overwrite_a_different_camera_mask(tmp_path):
    plan = source_plan(tmp_path)
    for name in ["cam0/frame.jpg", "cam1/frame.jpg"]:
        plan["sources"][name] = dict(plan["sources"]["source.jpg"], name=name)
    plan["variants"][0]["views"] = ["cam0/frame.jpg", "cam1/frame.jpg"]
    with pytest.raises(ValueError, match="basenames"):
        validate_inference_plan(plan)


def test_native_parity_rejects_a_runtime_ignoring_the_custom_predictor(tmp_path):
    plan = source_plan(tmp_path)

    class IgnoredPredictor:
        def predict(self, **options):
            # Simulates YOLOE.predict swallowing the explicit predictor option.
            import torch

            return [
                SimpleNamespace(
                    boxes=SimpleNamespace(data=torch.zeros((0, 6))), masks=None
                )
            ]

    with pytest.raises(ValueError, match="custom predictor was not used"):
        predictor_parity(IgnoredPredictor(), plan)
