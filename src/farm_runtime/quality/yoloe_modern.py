"""Explicit modern YOLOE text runtime for the existing proposal geometry stages.

The legacy feature/association backend remains separate. This adapter retains
native NMS and extracts mask logits without replacing them with binary scores.
"""

from __future__ import annotations

from pathlib import Path


def soft_mask_predictor():
    from ultralytics.engine.results import Results
    from ultralytics.models.yolo.segment.predict import SegmentationPredictor
    from ultralytics.utils import ops
    from farm_runtime.quality.discovery import decode_soft_masks

    class ModernSoftMaskPredictor(SegmentationPredictor):
        def construct_result(self, pred, img, original, path, proto):
            if not self.args.retina_masks:
                raise ValueError("source-aligned retina masks required")
            logits = None
            if len(pred):
                pred[:, :4] = ops.scale_boxes(
                    img.shape[2:], pred[:, :4], original.shape
                )
                logits = decode_soft_masks(
                    proto, pred[:, 6:], pred[:, :4], original.shape[:2]
                )
                # Match native removal of detections whose decoded mask is empty.
                keep = (logits > 0).flatten(1).any(1)
                pred, logits = pred[keep], logits[keep]
            result = Results(
                original,
                path=path,
                names=self.model.names,
                boxes=pred[:, :6],
                masks=None if logits is None else logits > 0,
            )
            result.farm_mask_logits = logits
            return result

    return ModernSoftMaskPredictor


def load_text_model(checkpoint, vocabulary, model_root):
    """Resolve both image and text weights explicitly; never download implicitly."""
    import gc
    import torch
    import ultralytics
    from packaging.version import Version
    from farm_runtime.quality_baseline import describe_file

    if Version(ultralytics.__version__) < Version("8.4.0"):
        raise ValueError("modern YOLOE requires a separate Ultralytics >=8.4 runtime")
    from ultralytics import YOLOE
    from ultralytics.nn.text_model import MobileCLIPTS

    checkpoint = Path(checkpoint).resolve(strict=True)
    names = [s.strip() for s in Path(vocabulary).read_text().splitlines() if s.strip()]
    if not names:
        raise ValueError("nonempty YOLOE vocabulary required")
    model = YOLOE(str(checkpoint)).to("cuda:0")
    if model.task != "segment":
        raise ValueError("YOLOE segmentation checkpoint required")
    text_variant = getattr(model.model, "text_model", "mobileclip:blt")
    encoders = {
        "mobileclip2:b": "mobileclip2_b.ts",
        "mobileclip:blt": "mobileclip_blt.ts",
    }
    if text_variant not in encoders:
        raise ValueError(f"unsupported explicit text encoder: {text_variant}")
    text_path = (Path(model_root) / "yoloe" / encoders[text_variant]).resolve(
        strict=True
    )
    with torch.inference_mode():
        model.model.clip_model = MobileCLIPTS(torch.device("cuda:0"), str(text_path))
        embeddings = model.model.get_text_pe(names, cache_clip_model=True)
        model.set_classes(names, embeddings)
    del model.model.clip_model
    gc.collect()
    model.eval()
    return model, dict(
        detector_checkpoint=describe_file(checkpoint),
        text_encoder_checkpoint=describe_file(text_path),
        text_variant=text_variant,
        ultralytics_version=ultralytics.__version__,
        text_backend_source=describe_file(Path(ultralytics.nn.text_model.__file__)),
    )
