import json
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
from PIL import Image

from farm_runtime.quality import refinement
from farm_runtime.quality_baseline import describe_file


def fixture(tmp_path, monkeypatch, backend="tracker"):
    import torch
    from farm_runtime import segmentation_refinement, concept_segmentation

    calls = []

    class Model:
        def __init__(self, _):
            calls.append("load")

        def predict(self, image, variants):
            calls.append("encode")
            logits = np.full((image.height, image.width), -5.0, np.float32)
            logits[8:25, 10:30] = 5
            return [
                dict(logits=logits.copy(), variant=v["name"], predicted_iou=0.95)
                for v in variants
            ]

    for module, name in [
        (segmentation_refinement, "CachedSAMRefiner"),
        (concept_segmentation, "CachedSAMConceptRefiner"),
    ]:
        monkeypatch.setattr(module, name, Model)
    monkeypatch.setattr(torch.cuda, "synchronize", lambda: None)
    monkeypatch.setattr(torch.cuda, "reset_peak_memory_stats", lambda: None)
    monkeypatch.setattr(torch.cuda, "max_memory_allocated", lambda: 0)

    def write(path, value):
        path.write_text(json.dumps(value))
        return path

    model = tmp_path / "model"
    model.mkdir()
    write(model / "config.json", {})
    (model / "model.safetensors").write_bytes(b"fixed model")
    image = tmp_path / "000002_000004.png"
    Image.new("RGB", (40, 32), "gray").save(image)
    probability = np.zeros((32, 40), np.float16)
    probability[8:25, 10:30] = 0.9
    seeds = tmp_path / "seed.npz"
    np.savez_compressed(seeds, probability=probability)
    row = dict(
        object_id=2,
        image_id=4,
        timestamp="t1",
        physical_timestamp_ns=100,
        crop=describe_file(image),
        seeds=describe_file(seeds),
        source_image=describe_file(image),
        grid_shape_hw=[64, 80],
        crop_grid_xyxy=[0, 0, 40, 32],
        crop_source_xyxy=[0, 0, 40, 32],
        turns=0,
        foreground_pixels=340,
        priority=0.2,
    )
    evidence = write(
        tmp_path / "evidence.json",
        dict(observations=[row], reserved_test_opened=False, legacy_heldout_used=False),
    )
    prompts = (
        write(tmp_path / "prompts.json", {"2": ["object"]})
        if backend == "concept"
        else None
    )
    args = SimpleNamespace(
        evidence=evidence,
        model=model,
        output=tmp_path / "fresh",
        backend=backend,
        prompts=prompts,
        limit=0,
        reuse_proposals=[],
    )
    refinement.sam(args)
    source = args.output / "manifest.json"
    assert calls == ["load", "encode"]
    args.output = tmp_path / "reused"
    args.reuse_proposals = [source]
    return args, calls, source


@pytest.mark.parametrize("backend", ["tracker", "concept"])
def test_identical_crop_reuses_logits_on_cpu_and_keeps_current_binding(
    tmp_path, monkeypatch, backend
):
    import torch

    args, calls, source = fixture(tmp_path, monkeypatch, backend)
    document = json.loads(args.evidence.read_text())
    document["observations"][0]["priority"] = 0.99
    new_evidence = tmp_path / "current_evidence.json"
    new_evidence.write_text(json.dumps(document))
    args.evidence = new_evidence

    def fail(*a, **kw):
        raise AssertionError("reuse must not initialize CUDA")

    monkeypatch.setattr(torch.cuda, "synchronize", fail)
    monkeypatch.setattr(torch.cuda, "reset_peak_memory_stats", fail)
    refinement.sam(args)
    reused = json.loads((args.output / "manifest.json").read_text())
    fresh = json.loads(source.read_text())
    assert calls == ["load", "encode"]
    assert reused["evidence"] == describe_file(new_evidence)
    assert reused["observations"][0]["priority"] == 0.99
    assert (
        reused["reused_observations"] == 1 and reused["source_image_encoder_calls"] == 0
    )
    assert (
        reused["model_load_seconds"] == 0
        and reused["no_inference_reason"] == "all_requests_reused"
    )
    assert (
        reused["observations"][0]["masks"]["sha256"]
        == fresh["observations"][0]["masks"]["sha256"]
    )
    assert Path(reused["observations"][0]["masks"]["path"]).is_file()


@pytest.mark.parametrize(
    "change", ["seed", "crop", "model", "processor", "implementation", "legacy"]
)
def test_changed_input_or_unfingerprinted_cache_requires_fresh_inference(
    tmp_path, monkeypatch, change
):
    args, calls, source = fixture(tmp_path, monkeypatch)
    evidence = json.loads(args.evidence.read_text())
    row = evidence["observations"][0]
    if change == "seed":
        path = tmp_path / "new_seed.npz"
        probability = np.zeros((32, 40), np.float16)
        probability[7:26, 9:31] = 0.9
        np.savez_compressed(path, probability=probability)
        row["seeds"] = describe_file(path)
    elif change == "crop":
        path = tmp_path / "new_crop.png"
        Image.new("RGB", (40, 32), "blue").save(path)
        row["crop"] = describe_file(path)
    elif change == "model":
        (args.model / "model.safetensors").write_bytes(b"changed weights")
    elif change == "processor":
        (args.model / "processor_config.json").write_text('{"size": 12}')
    else:
        cached = json.loads(source.read_text())
        if change == "legacy":
            cached.pop("inference_contract")
        else:
            cached["inference_contract"]["implementation"][
                "segmentation_refinement.py"
            ] = "different"
        source.write_text(json.dumps(cached))
    # Old evidence remains immutable; altered current requests get a new file.
    args.evidence = tmp_path / "changed_evidence.json"
    args.evidence.write_text(json.dumps(evidence))
    refinement.sam(args)
    result = json.loads((args.output / "manifest.json").read_text())
    assert calls == ["load", "encode", "load", "encode"]
    assert (
        result["reused_observations"] == 0 and result["source_image_encoder_calls"] == 1
    )


def test_changed_concept_text_is_not_a_cache_hit(tmp_path, monkeypatch):
    args, calls, _ = fixture(tmp_path, monkeypatch, "concept")
    args.prompts = tmp_path / "changed_prompts.json"
    args.prompts.write_text(json.dumps({"2": ["a different object"]}))
    refinement.sam(args)
    assert calls == ["load", "encode", "load", "encode"]


def test_corrupt_cached_archive_stops_before_inference(tmp_path, monkeypatch):
    args, calls, source = fixture(tmp_path, monkeypatch)
    row = json.loads(source.read_text())["observations"][0]
    Path(row["masks"]["path"]).write_bytes(b"corrupt")
    with pytest.raises(ValueError, match="hash"):
        refinement.sam(args)
    assert calls == ["load", "encode"]


def test_geometry_and_refinement_helpers_do_not_import_inference_libraries():
    import os
    import subprocess
    import sys

    result = subprocess.run(
        [
            sys.executable,
            "-c",
            """
import sys
from farm_runtime.quality import refinement
from farm_runtime.angular_discovery import upright_quarter_turns
from scene_graph.captioning.evidence import gravity_upright_orientation
assert 'torch' not in sys.modules
assert 'transformers' not in sys.modules
assert 'scene_graph.captioning.worker' not in sys.modules
assert 'scene_graph.captioning.services' not in sys.modules
""",
        ],
        env=dict(os.environ, PYTHONDONTWRITEBYTECODE="1"),
        capture_output=True,
        text=True,
        check=False,
    )
    assert result.returncode == 0, result.stderr


def test_lazy_captioning_exports_keep_the_public_class_identities():
    import scene_graph.captioning as public
    from scene_graph.captioning.models import (
        ObjectCaptionTask,
        ObjectCaptionResult,
        StructuredCaption,
    )
    from scene_graph.captioning.services import CaptionManager
    from scene_graph.captioning.worker import CaptionWorker

    expected = dict(
        CaptionManager=CaptionManager,
        CaptionWorker=CaptionWorker,
        ObjectCaptionTask=ObjectCaptionTask,
        ObjectCaptionResult=ObjectCaptionResult,
        StructuredCaption=StructuredCaption,
    )
    assert set(public.__all__) == set(expected)
    for name, cls in expected.items():
        assert getattr(public, name) is cls and name in dir(public)
    with pytest.raises(AttributeError):
        getattr(public, "not_a_captioning_export")
