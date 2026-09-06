import copy
import json
from pathlib import Path

import numpy as np
from PIL import Image
import pytest

from farm_runtime.quality.complementary_discovery import run, source_plan
from farm_runtime.quality_baseline import describe_file


def cohort(tmp_path):
    image = tmp_path / "view.png"
    Image.new("RGB", (512, 256), "white").save(image)
    row = dict(
        name=image.name,
        timestamp="000001",
        shape_hw=[256, 512],
        source_image=describe_file(image),
        camera_from_world_rotation=np.eye(3).tolist(),
        grid_shape_hw=[320, 640],
        applied_quarter_turns=0,
        queries=[dict(prompt="person")],
    )
    primary = tmp_path / "primary.json"
    write(primary, dict(test_opened=False, observations=[row]))
    return primary, row


def write(path, value):
    path.write_text(json.dumps(value))
    return path


def test_same_pixels_order_and_small_source_grid_are_preserved(tmp_path):
    primary, row = cohort(tmp_path)
    plan = source_plan(primary, [0, -1, 0])
    assert list(plan["sources"]) == [row["name"]]
    source = plan["sources"][row["name"]]
    assert source["source_image"] == row["source_image"]
    assert source["shape_hw"] == [256, 512]
    assert plan["variants"][0]["allow_upscale"] is True
    assert plan["primary"] == describe_file(primary)


@pytest.mark.parametrize(
    "mutation", ["test", "grid", "rotation", "turns", "person", "duplicate"]
)
def test_invalid_primary_is_rejected_before_inference(tmp_path, mutation):
    primary, row = cohort(tmp_path)
    doc = dict(test_opened=False, observations=[row])
    if mutation == "test":
        doc["test_opened"] = True
    if mutation == "grid":
        row["grid_shape_hw"] = [256, 512]
    if mutation == "rotation":
        row["camera_from_world_rotation"][0][0] = float("nan")
    if mutation == "turns":
        row["applied_quarter_turns"] = 1
    if mutation == "person":
        row["queries"] = []
    if mutation == "duplicate":
        doc["observations"].append(copy.deepcopy(row))
    write(primary, doc)
    with pytest.raises(ValueError):
        source_plan(primary, [0, -1, 0])


def test_changed_rgb_and_unbounded_cohort_are_rejected(tmp_path):
    primary, row = cohort(tmp_path)
    image = Path(row["source_image"]["path"])
    original = image.read_bytes()
    image.write_bytes(b"changed")
    with pytest.raises(ValueError):
        source_plan(primary, [0, -1, 0])
    image.write_bytes(original)
    write(
        primary,
        dict(
            test_opened=False,
            observations=[dict(row, name=f"{i}.png") for i in range(49)],
        ),
    )
    with pytest.raises(ValueError, match="48"):
        source_plan(primary, [0, -1, 0])


def test_runtime_checkpoint_mismatch_does_not_create_outputs(tmp_path, monkeypatch):
    from scene_graph import runtime_paths

    primary, row = cohort(tmp_path)
    root = tmp_path / "models"
    (root / "yoloe").mkdir(parents=True)
    (root / "yoloe/yoloe-v8l-seg.pt").write_bytes(b"base")
    (root / "mobileclip").mkdir()
    (root / "mobileclip/mobileclip_blt.pt").write_bytes(b"text")
    (root / "yoloe/yoloe-v8l-seg-pf.pt").write_bytes(b"weights")
    vocab = tmp_path / "vocab.txt"
    vocab.write_text("person\nbox\n")
    monkeypatch.setattr(runtime_paths, "find_model_file", lambda *args: None)
    output = tmp_path / "output"
    with pytest.raises(ValueError, match="checkpoint"):
        run(primary, root, vocab, [0, -1, 0], output)
    assert not output.exists()


def test_empty_primary_skips_inference_and_nonempty_result_is_bound(
    tmp_path, monkeypatch
):
    from farm_runtime.quality import discovery
    from scene_graph import runtime_paths

    primary, row = cohort(tmp_path)
    root = tmp_path / "models"
    (root / "yoloe").mkdir(parents=True)
    (root / "yoloe/yoloe-v8l-seg.pt").write_bytes(b"base")
    (root / "mobileclip").mkdir()
    (root / "mobileclip/mobileclip_blt.pt").write_bytes(b"text")
    weights = root / "yoloe/yoloe-v8l-seg-pf.pt"
    weights.write_bytes(b"weights")
    vocab = tmp_path / "vocab.txt"
    vocab.write_text("person\nbox\n")
    calls = []

    def infer(plan, output, **kwargs):
        calls.append(plan)
        dest = output / "yoloe"
        dest.mkdir()
        write(dest / "predictions.json", dict(observations=[row]))
        write(output / "results.json", dict(model_components={}))

    monkeypatch.setattr(discovery, "infer", infer)
    monkeypatch.setattr(
        runtime_paths, "find_model_file", lambda name, folder: root / folder / name
    )
    manifest = run(primary, root, vocab, [0, -1, 0], tmp_path / "nonempty")
    result = json.loads(manifest.read_text())
    assert len(calls) == 1
    assert result["primary"] == describe_file(primary)
    assert result["model_weights"] == describe_file(weights)
    assert result["observations"] == [row]
    write(primary, dict(test_opened=False, observations=[]))
    manifest = run(primary, root, vocab, [0, -1, 0], tmp_path / "empty")
    result = json.loads(manifest.read_text())
    assert len(calls) == 1
    assert result["observations"] == []
    assert result["no_inference_reason"] == "no_primary_views"


@pytest.mark.parametrize(
    "variable",
    ["MOBILECLIP_BLT_CKPT", "MOBILECLIP_CHECKPOINT", "MOBILECLIP_WEIGHTS_DIR"],
)
def test_external_text_weights_cannot_escape_pipeline_fingerprint(
    tmp_path, monkeypatch, variable
):
    from scene_graph import runtime_paths

    primary, _ = cohort(tmp_path)
    root = tmp_path / "models"
    (root / "yoloe").mkdir(parents=True)
    for name in ("yoloe-v8l-seg.pt", "yoloe-v8l-seg-pf.pt"):
        (root / "yoloe" / name).write_bytes(b"weights")
    (root / "mobileclip").mkdir()
    (root / "mobileclip/mobileclip_blt.pt").write_bytes(b"text")
    external = tmp_path / "external"
    external.mkdir()
    checkpoint = external / "mobileclip_blt.pt"
    checkpoint.write_bytes(b"other")
    for name in (
        "MOBILECLIP_BLT_CKPT",
        "MOBILECLIP_CHECKPOINT",
        "MOBILECLIP_WEIGHTS_DIR",
    ):
        monkeypatch.delenv(name, raising=False)
    monkeypatch.setenv(
        variable, str(external if variable.endswith("_DIR") else checkpoint)
    )
    monkeypatch.setattr(
        runtime_paths, "find_model_file", lambda name, folder: root / folder / name
    )
    vocab = tmp_path / "vocab.txt"
    vocab.write_text("person")
    output = tmp_path / "output"
    with pytest.raises(ValueError, match="MobileCLIP checkpoint"):
        run(primary, root, vocab, [0, -1, 0], output)
    assert not output.exists()
