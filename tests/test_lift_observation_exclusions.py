"""Unknown RGB foreground must never become static Gaussian background evidence."""

from dataclasses import replace
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

from farm_runtime.quality_baseline import describe_file
from tools.farm_shaper_bridge.common import (
    Frame,
    MaskObservation,
    RunData,
    load_observation_exclusion,
)
from tools.farm_shaper_bridge.gaussian_lift import _load_view_masks, load_config


def fixture(tmp_path):
    depth = np.full((40, 40), 2.0, np.float32)
    np.save(tmp_path / "depth.npy", depth)
    frame = Frame(
        0,
        "a",
        100,
        "cam",
        "cam",
        "center",
        "source.png",
        (40, 40),
        np.array([[40.0, 0, 20], [0, 40.0, 20], [0, 0, 1]]),
        np.eye(4),
        tmp_path / "rgb.png",
        tmp_path / "depth.npy",
    )
    raw = np.zeros((40, 40), bool)
    raw[10:30, 10:30] = True
    path = tmp_path / "mask.npz"
    np.savez_compressed(
        path,
        image_shape=np.array([40, 40]),
        raw_shape=np.array([40, 40]),
        raw_bbox_xyxy=np.array([0, 0, 40, 40]),
        raw_bits=np.packbits(raw.ravel(), bitorder="little"),
    )
    observation = MaskObservation(7, 0, "mask.npz", {})
    run = RunData(
        tmp_path,
        "scene",
        {},
        (frame,),
        (),
        1.0,
        dict(depth_min_m=0.1, depth_max_m=20.0),
        "a" * 64,
        None,
        {},
        {},
        True,
        mask_overrides={(7, 0): path},
    )
    config, _ = load_config(Path("configs/gaussian_lift.v1.yaml"))
    return run, frame, observation, config


def test_absent_or_empty_exclusions_preserve_native_evidence(tmp_path):
    run, frame, observation, config = fixture(tmp_path)
    baseline = _load_view_masks(run, frame, {7: [observation]}, config)
    path = tmp_path / "excluded.npy"
    np.save(path, np.zeros(frame.depth_size, bool))
    changed = replace(run, observation_exclusions={0: describe_file(path)})
    current = _load_view_masks(changed, frame, {7: [observation]}, config)
    for key in ("raw", "inlier", "positive_weight", "negative_weight"):
        np.testing.assert_array_equal(current[0][7][key], baseline[0][7][key])
    np.testing.assert_array_equal(current[1], baseline[1])
    assert load_observation_exclusion(run, frame) == (None, None)


def test_transient_cannot_supply_positive_negative_or_visible_weight(tmp_path):
    run, frame, observation, config = fixture(tmp_path)
    excluded = np.zeros(frame.depth_size, bool)
    excluded[:, 15:25] = True
    path = tmp_path / "excluded.npy"
    np.save(path, excluded)
    changed = replace(run, observation_exclusions={0: describe_file(path)})
    decoded, surface, _, audit = _load_view_masks(
        changed, frame, {7: [observation]}, config
    )
    for key in ("positive_weight", "negative_weight"):
        assert not decoded[7][key][excluded].any()
        assert decoded[7][key][~excluded].any()
    assert not surface[excluded].any() and surface[~excluded].any()
    # Keep the original observation; unknown is an explicit independent layer.
    assert decoded[7]["raw"][excluded].any()
    assert audit["observation_exclusion"]["excluded_pixels"] == 400


def test_exclusion_requires_immutable_boolean_native_grid(tmp_path):
    run, frame, _, _ = fixture(tmp_path)
    path = tmp_path / "excluded.npy"
    for array in (np.zeros((40, 40), np.uint8), np.zeros((20, 20), bool)):
        np.save(path, array)
        changed = replace(run, observation_exclusions={0: describe_file(path)})
        with pytest.raises(ValueError, match="boolean native-grid"):
            load_observation_exclusion(changed, frame)
    np.save(path, np.zeros((40, 40), bool))
    changed = replace(run, observation_exclusions={0: describe_file(path)})
    np.save(path, np.ones((40, 40), bool))
    with pytest.raises(ValueError, match="artifact changed"):
        load_observation_exclusion(changed, frame)


@pytest.mark.parametrize("fully_occluded", [False, True])
def test_reverse_qc_excludes_unknown_and_never_scores_fully_hidden_target(
    tmp_path, monkeypatch, fully_occluded
):
    import torch
    from tools.farm_shaper_bridge import gaussian_lift as lift
    from tools.farm_shaper_bridge.common import FarmObject, load_mask_pair

    run, frame, observation, config = fixture(tmp_path)
    target, _, _ = load_mask_pair(tmp_path / "mask.npz", frame.depth_size)
    excluded = np.zeros(frame.depth_size, bool)
    excluded[:, 15:25] = True
    if fully_occluded:
        excluded |= target
    path = tmp_path / "excluded.npy"
    np.save(path, excluded)
    obj = FarmObject(
        7, "cabinet", "", np.zeros(3), np.ones(3), np.eye(3), (observation,)
    )
    run = replace(run, objects=(obj,), observation_exclusions={0: describe_file(path)})
    # False positive projection behind an excluded foreground is unobservable.
    mass = (target | excluded).astype(np.float32)[..., None]
    monkeypatch.setattr(
        lift,
        "reverse_render",
        lambda *args: (mass, mass, np.full_like(mass, 2.0), None, np.ones_like(mass)),
    )
    monkeypatch.setattr(torch.cuda, "synchronize", lambda: None)
    monkeypatch.setattr(torch.cuda, "empty_cache", lambda: None)
    split = dict(
        heldout_timestamps=["100"],
        objects=[dict(object_id=7, strict_eligible=False, eligibility="development")],
    )
    rows, accepted, _, audit = lift.heldout_qc(
        None, run, split, np.array([7], np.int32), config
    )
    assert not accepted
    if fully_occluded:
        assert rows[0]["heldout_timestamps"] == 0
        assert rows[0]["summaries"]["iou"] is None
        assert audit[0]["unobservable_object_ids"] == [7]
    else:
        result = rows[0]["views"][0]
        assert result["iou"] == 1 and result["target_pixels"] == 200
        assert result["target_excluded_pixels"] == 200
        assert result["evaluation_excluded_pixels"] == 400


def test_native_adapter_mask_matches_existing_bitstream_decoder(tmp_path):
    from farm_runtime.quality.native_observations import write_native_mask
    from tools.farm_shaper_bridge.common import load_mask_pair

    source = np.zeros((10, 20), bool)
    source[2:8, 5:15] = True
    path = tmp_path / "adapted.npz"
    write_native_mask(path, source, (20, 40), 2)
    raw, inlier, digest = load_mask_pair(path, (20, 40))
    assert raw.sum() == 240 and raw[4:16, 10:30].all()
    assert inlier.sum() == 128 and not (inlier & ~raw).any()
    assert len(digest) == 64 and source.sum() == 60


def test_thin_depth_boundary_can_supply_foreground_without_becoming_background(
    tmp_path,
):
    run, frame, observation, config = fixture(tmp_path)
    depth = np.full(frame.depth_size, 2.0, np.float32)
    depth[:, 20:] = 4.0
    depth[18, 18] = 0
    np.save(frame.depth_path, depth)
    baseline = _load_view_masks(run, frame, {7: [observation]}, config)
    assert not baseline[0][7]["positive_weight"][20, 19:21].any()
    config["mask"]["positive_depth_policy"] = "valid"
    decoded, visible, _, _ = _load_view_masks(run, frame, {7: [observation]}, config)
    assert decoded[7]["positive_weight"][20, 19:21].all()
    assert visible[20, 19:21].all()
    assert not decoded[7]["negative_weight"][20, 19:21].any()
    assert decoded[7]["positive_weight"][18, 18] == 0 and visible[18, 18] == 0


def test_full_background_adds_only_observed_non_object_pixels():
    import cv2
    from tools.farm_shaper_bridge.gaussian_lift import negative_mask_ring

    raw = np.zeros((60, 60), bool)
    raw[25:35, 25:35] = True
    other = np.zeros_like(raw)
    other[:10, :10] = True
    surface = np.ones_like(raw)
    surface[50:] = False
    kernel = np.ones((5, 5), np.uint8)
    ring = negative_mask_ring(raw, other, surface, kernel, 1)
    full = negative_mask_ring(
        raw, other, surface, kernel, 1, domain="visible_background"
    )
    assert np.all(full[ring]) and full[10, 10] and not ring[10, 10]
    assert not full[raw | other | ~surface].any()
    assert not full[24, 30]
    assert not negative_mask_ring(
        np.zeros_like(raw), other, surface, kernel, domain="visible_background"
    ).any()


def test_fully_excluded_foreground_cannot_cast_background_votes(tmp_path):
    run, frame, observation, config = fixture(tmp_path)
    excluded = np.zeros(frame.depth_size, bool)
    excluded[10:30, 10:30] = True
    path = tmp_path / "excluded.npy"
    np.save(path, excluded)
    run = replace(run, observation_exclusions={0: describe_file(path)})
    for policy in ("stable", "valid"):
        config["mask"]["positive_depth_policy"] = policy
        for domain in ("ring", "visible_background"):
            config["mask"]["negative_domain"] = domain
            decoded, _, _, _ = _load_view_masks(run, frame, {7: [observation]}, config)
            assert not decoded[7]["positive_weight"].any()
            assert not decoded[7]["negative_weight"].any()


def test_rendered_background_observes_mixed_depth_but_not_missing_or_excluded(tmp_path):
    run, frame, observation, config = fixture(tmp_path)
    depth = np.full(frame.depth_size, 2.0, np.float32)
    depth[:, 5:] = 4.0
    depth[5, 5] = 0
    np.save(frame.depth_path, depth)
    excluded = np.zeros(frame.depth_size, bool)
    excluded[8, 4] = True
    path = tmp_path / "excluded.npy"
    np.save(path, excluded)
    run = replace(run, observation_exclusions={0: describe_file(path)})
    config["mask"].update(
        negative_domain="visible_background", negative_depth_policy="valid"
    )
    decoded, visible, _, _ = _load_view_masks(run, frame, {7: [observation]}, config)
    assert decoded[7]["negative_weight"][20, 4] > 0
    assert visible[20, 4] > 0
    assert decoded[7]["negative_weight"][5, 5] == 0
    assert decoded[7]["negative_weight"][8, 4] == 0
    assert visible[5, 5] == visible[8, 4] == 0


def test_one_sided_depth_keeps_front_contributor_but_rejects_hidden_and_missing(
    tmp_path,
):
    from tools.farm_shaper_bridge.gaussian_lift import _depth_gate

    run, frame, _, config = fixture(tmp_path)
    points = np.array([[0, 0, 1], [0, 0, 2], [0, 0, 3], [-1, 0, 2]], float)
    gs = SimpleNamespace(means_m=points)
    evidence = SimpleNamespace(indices=np.arange(4), radius_m=np.full(4, 0.001))
    depth = np.full(frame.depth_size, 2.0, np.float32)
    depth[20, 0] = 0
    np.testing.assert_array_equal(
        _depth_gate(gs, evidence, frame, depth, config), [False, True, False, False]
    )
    config["candidate"]["depth_gate_policy"] = "not_behind"
    np.testing.assert_array_equal(
        _depth_gate(gs, evidence, frame, depth, config), [True, True, False, False]
    )


def test_control_split_uses_source_timestamp_not_bundle_local_nominal_time():
    from farm_runtime.quality.native_evaluation import require_new_timestamps

    sources = {
        "camera_a": {"timestamp": "000100", "timestamp_ns": 100000000},
        "camera_b": {"timestamp": "000200", "timestamp_ns": 100000000},
    }
    require_new_timestamps(["camera_b"], sources, {"000100"})
    with pytest.raises(ValueError, match="overlaps"):
        require_new_timestamps(["camera_a"], sources, {"000100"})
    with pytest.raises(ValueError, match="duplicate"):
        require_new_timestamps(["camera_b", "camera_b"], sources, {"000100"})
