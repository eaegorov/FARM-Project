"""Local SAM3 concept masks with image and text encoder reuse."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np


def load_sam3_detector(model_path):
    import torch
    from transformers import Sam3Config, Sam3Model, Sam3Processor

    model_path = Path(model_path)
    raw = json.loads((model_path / "config.json").read_text())
    model, loading = Sam3Model.from_pretrained(
        model_path,
        config=Sam3Config(**raw.get("detector_config", raw)),
        local_files_only=True,
        dtype=torch.float32,
        device_map="cuda",
        output_loading_info=True,
    )
    if any(loading.get(k) for k in ("missing_keys", "mismatched_keys", "error_msgs")):
        raise ValueError(
            "SAM3 detector weights did not load completely: " + str(loading)
        )
    unexpected = loading.get("unexpected_keys", [])
    if any(not k.startswith(("tracker_model.", "tracker_neck.")) for k in unexpected):
        raise ValueError("unrecognized checkpoint tensors outside detector")
    model.eval()
    processor = Sam3Processor.from_pretrained(model_path, local_files_only=True)
    return model, processor, loading


class CachedSAMConceptRefiner:
    """Competing text concepts; seed geometry selects the relevant instance."""

    def __init__(self, model_path):
        self.model, self.processor, self.loading = load_sam3_detector(model_path)
        self.text_cache = {}

    def predict(self, image, variants):
        import torch

        with torch.inference_mode():
            images = self.processor(images=image, return_tensors="pt").to("cuda")
            vision = self.model.get_vision_features(pixel_values=images.pixel_values)
            results = []
            for variant in variants:
                text = variant["text"]
                if text not in self.text_cache:
                    prompt = self.processor(text=text, return_tensors="pt").to("cuda")
                    self.text_cache[text] = (
                        self.model.get_text_features(**prompt),
                        prompt.attention_mask,
                    )
                embedding, attention = self.text_cache[text]
                output = self.model(
                    vision_embeds=vision,
                    text_embeds=embedding,
                    attention_mask=attention,
                )
                score = output.pred_logits[0].sigmoid()
                if output.presence_logits is not None:
                    score = score * output.presence_logits[0].sigmoid()
                keep = score > 0.5
                probabilities = output.pred_masks[0, keep].sigmoid()
                if len(probabilities):
                    probabilities = torch.nn.functional.interpolate(
                        probabilities[None],
                        size=(image.height, image.width),
                        mode="bilinear",
                        align_corners=False,
                    )[0]
                native = self.processor.post_process_instance_segmentation(
                    output,
                    threshold=0.5,
                    mask_threshold=0.5,
                    target_sizes=[(image.height, image.width)],
                )[0]
                if not torch.equal(probabilities > 0.5, native["masks"].bool()):
                    raise ValueError("SAM3 soft masks differ from official decoder")
                for probability, confidence in zip(probabilities, score[keep]):
                    probability = probability.clamp(1e-6, 1 - 1e-6)
                    results.append(
                        dict(
                            variant=variant["name"],
                            prompt=text,
                            score_kind="sam3_presence_instance",
                            model_score=float(confidence),
                            logits=(probability / (1 - probability))
                            .log()
                            .cpu()
                            .numpy(),
                        )
                    )
                if not len(probabilities):
                    results.append(
                        dict(
                            variant=variant["name"],
                            prompt=text,
                            score_kind="sam3_presence_instance",
                            model_score=0.0,
                            logits=np.full(
                                (image.height, image.width), -16.0, np.float32
                            ),
                            abstention="no_concept_instance",
                        )
                    )
        return results
