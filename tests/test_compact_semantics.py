import json
from types import SimpleNamespace

import numpy as np
import pytest
from PIL import Image

from farm_runtime.quality import refinement
from farm_runtime.quality_baseline import describe_file
from farm_runtime.semantic_refinement import APPEARANCE_PROMPT, validate_appearance


def appearance():
    return dict(
        label="enclosure",
        caption="A dark rectangular enclosure.",
        uncertainty="Its function is unclear.",
        observed_image_ids=[7, 8],
    )


@pytest.mark.parametrize(
    "patch",
    [
        {"uncertainty": None},
        {"caption": []},
        {"label": " "},
        {"integral_parts": []},
        {"observed_image_ids": [8, 7]},
        {"observed_image_ids": [7]},
        {"observed_image_ids": [7, True]},
    ],
)
def test_appearance_rejects_untyped_or_unbound_evidence(patch):
    with pytest.raises(ValueError):
        validate_appearance(json.dumps(dict(appearance(), **patch)), [7, 8])


def test_appearance_accepts_uncertain_identity_without_assessing_ownership():
    value = appearance()
    assert validate_appearance(json.dumps(value), [7, 8]) == value
    value["uncertainty"] = ""
    assert validate_appearance(json.dumps(value), [7, 8]) == value


def test_compact_cli_preserves_two_timestamp_evidence_and_context(
    tmp_path, monkeypatch
):
    import torch
    import farm_runtime.semantic_refinement as semantics

    source = tmp_path / "source.png"
    crop = tmp_path / "crop.png"
    masks = tmp_path / "masks.npz"
    Image.new("RGB", (100, 80), (20, 40, 60)).save(source)
    Image.new("RGB", (40, 40), (20, 40, 60)).save(crop)
    mask = np.zeros((40, 40), bool)
    mask[8:32, 12:28] = True
    np.savez_compressed(masks, baseline=mask)
    observations = [
        dict(
            object_id=4,
            image_id=i,
            timestamp=str(i),
            source_image=describe_file(source),
            crop=describe_file(crop),
            masks=describe_file(masks),
            crop_source_xyxy=[20, 10, 60, 50],
            turns=0,
        )
        for i in [7, 8]
    ]
    proposals = tmp_path / "evidence.json"
    proposals.write_text(
        json.dumps(
            dict(
                schema="farm.object-scope-evidence.v1",
                reserved_test_opened=False,
                observations=observations,
            )
        )
    )
    model = tmp_path / "model"
    model.mkdir()
    (model / "config.json").write_text("{}")
    calls = []

    class Reviewer:
        def __init__(self, path):
            assert path == model

        def ask(self, prompt, images, image_ids, validator, *, max_new_tokens):
            calls.append(image_ids)
            assert prompt == APPEARANCE_PROMPT
            assert validator is validate_appearance
            assert max_new_tokens == 180
            assert [im.size for im in images] == [(1560, 550)] * 2
            assert images[0].getpixel((774, 286)) == (20, 40, 60)
            raw = json.dumps(appearance())
            return dict(
                raw=raw,
                parsed=validator(raw, image_ids),
                validation_error=None,
                seconds=0,
            )

    monkeypatch.setattr(semantics, "LocalObjectReviewer", Reviewer)
    monkeypatch.setattr(torch.cuda, "synchronize", lambda: None)
    monkeypatch.setattr(torch.cuda, "reset_peak_memory_stats", lambda: None)
    monkeypatch.setattr(torch.cuda, "max_memory_allocated", lambda: 0)
    output = tmp_path / "output"
    args = [
        "vlm",
        "--proposals",
        str(proposals),
        "--model",
        str(model),
        "--compact-semantics",
        "--output",
        str(output),
    ]
    refinement.main(args)
    result = json.loads((output / "manifest.json").read_text())
    assert calls == [[7, 8]]
    assert result["schema"] == "farm.local-object-appearance.v1"
    assert result["scene_context_included"] is True
    assert result["physical_scope_assessed"] is False
    assert result["objects"][0]["evidence_timestamps"] == ["7", "8"]
    # A repeated timestamp must fail before invoking the model again.
    observations[1]["timestamp"] = "7"
    proposals.write_text(
        json.dumps(
            dict(
                schema="farm.object-scope-evidence.v1",
                reserved_test_opened=False,
                observations=observations,
            )
        )
    )
    args[-1] = str(tmp_path / "invalid")
    with pytest.raises(ValueError, match="distinct timestamps"):
        refinement.main(args)
    assert calls == [[7, 8]]


def test_empty_sam_batch_does_not_depend_on_vlm_arguments(tmp_path, monkeypatch):
    import torch
    import farm_runtime.segmentation_refinement as segmentation

    evidence = tmp_path / "evidence.json"
    evidence.write_text(
        json.dumps(
            dict(reserved_test_opened=False, legacy_heldout_used=False, observations=[])
        )
    )
    model = tmp_path / "model"
    model.mkdir()
    (model / "config.json").write_text("{}")
    (model / "model.safetensors").write_bytes(b"test model description")
    monkeypatch.setattr(segmentation, "CachedSAMRefiner", lambda path: object())
    monkeypatch.setattr(torch.cuda, "synchronize", lambda: None)
    output = tmp_path / "output"
    refinement.sam(
        SimpleNamespace(
            evidence=evidence,
            model=model,
            output=output,
            backend="tracker",
            prompts=None,
            limit=None,
        )
    )
    result = json.loads((output / "manifest.json").read_text())
    assert result["schema"] == "farm.sam-refinement-proposals.v1"
    assert result["observations"] == []
    assert "scene_context_included" not in result
