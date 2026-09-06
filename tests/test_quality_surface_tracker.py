import json
from pathlib import Path

import numpy as np
from PIL import Image
import pytest

from farm_runtime.angular_discovery import rotate_image
from farm_runtime.quality.surface_tracker import (
    projected_crop,
    restore_crop_logits,
    main,
)
from farm_runtime.quality_baseline import describe_file


@pytest.mark.parametrize("turns", [0, 1, 2, 3])
def test_crop_logits_restore_original_grid_and_preserve_foreground(turns):
    logits = np.full((8, 12), -5.0, np.float32)
    logits[2:6, 2:10] = 5
    upright = rotate_image(np.repeat(np.repeat(logits, 2, axis=0), 2, axis=1), turns)
    restored = restore_crop_logits(upright, turns, [17, 31, 29, 39])
    assert restored.shape == logits.shape
    assert np.array_equal(restored > 0, logits > 0)
    assert np.isfinite(restored).all()


def test_projected_prompt_excludes_occluded_and_transient_surface():
    h, w = 40, 60
    yy, xx = np.mgrid[10:20, 15:25]
    points = np.stack([xx.ravel() / 20, yy.ravel() / 20, np.full(xx.size, 2)], axis=1)
    frame = dict(
        depth=np.full((h, w), 2.0),
        K=np.diag([40.0, 40.0, 1.0]),
        T_world_cam=np.eye(4),
        excluded=np.zeros((h, w), bool),
    )
    image = Image.new("RGB", (120, 80), "white")
    packet = projected_crop(points, frame, image, 1)
    assert packet["visible_surface_points"] == 100
    assert packet["seed"].shape == (packet["crop"].height, packet["crop"].width)
    box = packet["crop_grid_xyxy"]
    assert box[0] <= 15 and box[1] <= 10 and box[2] > 24 and box[3] > 19
    frame["excluded"][10:20, 15:25] = True
    assert projected_crop(points, frame, image, 1) is None
    frame["excluded"][:] = False
    frame["depth"][:] = 1.0
    assert projected_crop(points, frame, image, 1) is None


def test_empty_schedule_does_not_load_tracker_or_require_weights(tmp_path, monkeypatch):
    import farm_runtime.quality.surface_tracker as tracker

    def write(name, obj):
        path = tmp_path / name
        path.write_text(json.dumps(obj))
        return path

    plan = write("plan.json", {"sources": {}})
    geometry = write("geometry.json", {})
    cloud = tmp_path / "points.npz"
    np.savez_compressed(cloud)
    audit = write(
        "audit.json",
        dict(
            closed_test_opened=False,
            adaptive_plan=describe_file(plan),
            source_geometry=describe_file(geometry),
            point_evidence=describe_file(cloud),
            groups=[],
        ),
    )
    proposals = write(
        "proposals.json",
        dict(test_opened=False, plan=describe_file(plan), observations=[]),
    )
    monkeypatch.setattr(tracker, "SurfaceInputs", lambda _: None)

    def forbidden(_):
        raise AssertionError("empty schedule must not initialize the tracker")

    monkeypatch.setattr(tracker, "CachedSAMRefiner", forbidden)
    out = tmp_path / "out"
    assert (
        main(
            [
                "--audit",
                str(audit),
                "--proposals",
                str(proposals),
                "--model",
                str(tmp_path / "absent"),
                "--output",
                str(out),
            ]
        )
        == 0
    )
    report = json.loads((out / "manifest.json").read_text())
    assert report["observations"] == [] and report["model_weights"] is None
    assert report["source_image_encoder_calls"] == 0


def test_nonfinite_tracker_logits_are_rejected():
    with pytest.raises(ValueError, match="finite"):
        restore_crop_logits(np.full((4, 4), np.nan), 0, [0, 0, 4, 4])
