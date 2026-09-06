import json
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
from PIL import Image

from farm_runtime.quality import refinement, native_observations, scope_evidence
from farm_runtime.quality.scene_profile import apply_refinement
from farm_runtime.quality.scene_catalog import bound_appearance
from farm_runtime.quality_baseline import describe_file
from farm_runtime.semantic_refinement import select_review_views


def write(path, doc):
    path.write_text(json.dumps(doc))
    return path


def test_empty_sam_schedule_never_initializes_cuda_or_models(tmp_path, monkeypatch):
    import torch
    from farm_runtime import segmentation_refinement, concept_segmentation

    def fail(*a, **kw):
        raise AssertionError("empty schedule loaded GPU/model")

    monkeypatch.setattr(torch.cuda, "synchronize", fail)
    monkeypatch.setattr(segmentation_refinement, "CachedSAMRefiner", fail)
    monkeypatch.setattr(concept_segmentation, "CachedSAMConceptRefiner", fail)
    evidence = write(
        tmp_path / "evidence.json",
        dict(observations=[], reserved_test_opened=False, legacy_heldout_used=False),
    )
    prompts = write(tmp_path / "prompts.json", {})
    model = tmp_path / "model"
    model.mkdir()
    write(model / "config.json", {})
    (model / "model.safetensors").write_bytes(b"checkpoint descriptor only")
    for backend in ["tracker", "concept"]:
        out = tmp_path / backend
        refinement.sam(
            SimpleNamespace(
                evidence=evidence,
                model=model,
                output=out,
                prompts=prompts,
                backend=backend,
                limit=0,
            )
        )
        d = json.loads((out / "manifest.json").read_text())
        assert not d["observations"]
        assert d["model_load_seconds"] == 0
        assert d["no_inference_reason"] == "empty_crop_schedule"


def test_no_change_reuses_exact_bank_and_input_without_native_rebuild(
    tmp_path, monkeypatch
):
    def fail(*a, **kw):
        raise AssertionError("no changes must not rebuild native bank")

    monkeypatch.setattr(native_observations, "main", fail)
    inp = write(tmp_path / "input.json", {})
    cfg = write(tmp_path / "config.json", {})
    native = dict(
        input=describe_file(inp),
        config=describe_file(cfg),
        reports=[dict(bank="original")],
        closed_test_opened=False,
    )
    original = write(tmp_path / "native.json", native)
    report = dict(
        input=describe_file(inp),
        config=describe_file(cfg),
        observations=[dict(selected=None)],
        quarantined_observations=[],
        closed_test_opened=False,
    )
    selection = write(tmp_path / "selection.json", report)
    out = tmp_path / "unchanged"
    apply_refinement(
        SimpleNamespace(
            native=original, selection=selection, alternatives=16, output=out
        )
    )
    result = json.loads((out / "manifest.json").read_text())
    assert result["input"] == native["input"] and result["reports"] == native["reports"]
    assert result["refinement_application"]["native_rebuilt"] is False
    different = write(tmp_path / "another_input.json", dict(scene="another"))
    report["input"] = describe_file(different)
    write(selection, report)
    with pytest.raises(ValueError, match="this native build"):
        apply_refinement(
            SimpleNamespace(
                native=original,
                selection=selection,
                alternatives=16,
                output=tmp_path / "wrong",
            )
        )


def test_review_selection_does_not_prefer_a_larger_mask_grid():
    rows = [
        dict(
            image_id=i,
            timestamp=str(i),
            foreground_pixels=p,
            foreground_fraction=f,
            clipped=False,
            source_image=dict(path=f"/cam0_{i}.png"),
        )
        for i, p, f in [(0, 400, 0.4), (1, 900, 0.3)]
    ]
    assert select_review_views(rows, 1)[0]["image_id"] == 0
    del rows[0]["foreground_fraction"]
    with pytest.raises(ValueError, match="consistent visibility"):
        select_review_views(rows, 1)


def test_scope_uses_replaced_mask_omits_quarantine_and_keeps_additional_view(
    tmp_path, monkeypatch
):
    # Native frame 0 is image c, while geometric frame 0 is a. Frame a is
    # quarantined; b's mask shrinks, and c was added after geometric association.
    mask_dir = tmp_path / "masks"
    mask_dir.mkdir()
    originals, frames = [], []
    for i, name in enumerate(["a.png", "b.png", "c.png"]):
        rgb = tmp_path / name
        Image.new("RGB", (80, 80), (50 + 50 * i, 80, 100)).save(rgb)
        frames.append(
            dict(
                image_id=2 - i,
                name=name,
                source=describe_file(rgb),
                applied_quarter_turns=0,
            )
        )
        if name != "c.png":
            path = mask_dir / f"{name}.npz"
            values = np.full((20, 20), -1.0, dtype=np.float32)
            values[2:18, 2:18] = 1
            np.savez_compressed(path, logits=values)
            originals.append(
                dict(
                    name=name,
                    source_image=describe_file(rgb),
                    applied_quarter_turns=0,
                    mask_artifact=describe_file(path),
                    grid_shape_hw=[20, 20],
                    detections=[
                        dict(grid_window_xyxy=[0, 0, 20, 20], logit_key="logits")
                    ],
                )
            )
    proposals = write(
        tmp_path / "proposals.json", dict(test_opened=False, observations=originals)
    )
    ev = write(tmp_path / "evidence.json", dict(scope_alternatives=[]))
    geometry = write(
        tmp_path / "geometry.json",
        dict(
            test_opened=False,
            inputs=dict(proposals=describe_file(proposals)),
            evidence_artifact=describe_file(ev),
            groups=[
                dict(
                    id=7,
                    members=[0, 1],
                    independent_timestamps=2,
                    candidate_labels=["object"],
                )
            ],
            nodes=[
                dict(id=i, frame=name, timestamp=str(i), representative_detection=0)
                for i, name in enumerate(["a.png", "b.png"])
            ],
        ),
    )
    changed = np.zeros((40, 40), bool)
    changed[10:30, 20:30] = True
    mask_path = tmp_path / "changed.npz"
    descriptor = native_observations.write_native_mask(mask_path, changed, (40, 40), 0)
    masks = [
        dict(
            image_id=i,
            source_name=name,
            mask=descriptor,
            source_grid_hw=[40, 40],
            physical_timestamp_ns=2 - i,
            **({"parent_mask": dict(path="old")} if name == "b.png" else {}),
        )
        for i, name in [(1, "b.png"), (0, "c.png")]
    ]
    inp = write(
        tmp_path / "native_input.json",
        dict(
            source_geometry=describe_file(geometry),
            frames=frames,
            objects=[dict(object_id=7, masks=masks)],
        ),
    )

    def load(path):
        return (
            SimpleNamespace(
                frame=lambda i: SimpleNamespace(
                    depth_size=(40, 40), frame_id=str(2 - i)
                )
            ),
            {},
            json.loads(path.read_text()),
        )

    monkeypatch.setattr(native_observations, "load_prepared", load)
    out = tmp_path / "scope"
    scope_evidence.main(
        ["--groups", str(geometry), "--native-input", str(inp), "--output", str(out)]
    )
    result = json.loads((out / "manifest.json").read_text())
    rows = {r["source_name"]: r for r in result["observations"]}
    assert set(rows) == {"b.png", "c.png"}
    assert rows["b.png"]["image_id"] == 1 and rows["c.png"]["image_id"] == 2
    assert rows["b.png"]["mask_source"] == "refined_native_raw"
    assert rows["b.png"]["foreground_fraction"] == 0.125
    assert rows["c.png"]["node_id"] is None
    assert rows["c.png"]["mask_source"] == "additional_native_raw"
    assert result["source_native_input"] == describe_file(inp)
    # Different object namespace must fail even if integer IDs match.
    d = json.loads(inp.read_text())
    d["source_geometry"] = describe_file(ev)
    write(inp, d)
    with pytest.raises(ValueError, match="geometry namespace"):
        scope_evidence.main(
            [
                "--groups",
                str(geometry),
                "--native-input",
                str(inp),
                "--output",
                str(tmp_path / "wrong_scope"),
            ]
        )


def test_catalog_rejects_stale_appearance_after_mask_refinement(tmp_path):
    geometry = write(tmp_path / "geometry.json", {})
    before = write(tmp_path / "before.json", dict(mask="before"))
    after = write(tmp_path / "after.json", dict(mask="after"))
    evidence = write(
        tmp_path / "evidence.json",
        dict(
            source_groups=describe_file(geometry),
            source_native_input=describe_file(before),
        ),
    )
    appearance = write(
        tmp_path / "appearance.json",
        dict(
            schema="farm.local-object-appearance.v1",
            physical_scope_assessed=False,
            reserved_test_opened=False,
            proposals=describe_file(evidence),
            objects=[],
        ),
    )
    stage = write(
        tmp_path / "stage.json",
        dict(
            schema="farm.quality-appearance-stage.v1",
            source_geometry=describe_file(geometry),
            source_native_input=describe_file(before),
            appearance=describe_file(appearance),
            selected_group_ids=[],
        ),
    )
    assert (
        bound_appearance(stage, describe_file(geometry), describe_file(before))[0] == {}
    )
    with pytest.raises(ValueError, match="effective native input"):
        bound_appearance(stage, describe_file(geometry), describe_file(after))
