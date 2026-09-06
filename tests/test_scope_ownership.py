from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

from farm_runtime.quality.scope_ownership import (
    competing_scopes,
    resolve_scope_alternative,
)
from tools.farm_shaper_bridge.gaussian_lift import load_config, make_claims


def item(oid, ids):
    n = len(ids)
    return SimpleNamespace(
        indices=np.asarray(ids, np.int64),
        positive_weight=np.ones(n, np.float32),
        negative_weight=np.zeros(n, np.float32),
        visible_weight=np.ones(n, np.float32),
        positive_timestamps=np.full(n, 2, np.uint16),
        negative_timestamps=np.zeros(n, np.uint16),
        build_timestamp_count=2,
        obj=SimpleNamespace(object_id=oid, category="object"),
    )


def test_nested_hypothesis_recovers_surface_without_stealing_unrelated_conflict():
    config, _ = load_config(Path("configs/gaussian_lift.v1.yaml"))
    config["refinement"]["enabled"] = False
    config["build"]["minimum_object_gaussians"] = 1
    config["build"]["maximum_object_fraction"] = 0.9
    evidence = {1: item(1, [0, 1, 2, 3]), 2: item(2, [0, 1]), 3: item(3, [2, 4])}
    original = {oid: row.positive_weight.copy() for oid, row in evidence.items()}
    primary = make_claims(evidence, 10, config)[0]
    assert primary[:5].tolist() == [-1, -1, -1, 1, 3]
    bank, audit = resolve_scope_alternative(
        evidence, 1, {2}, np.zeros((10, 3)), np.ones(10), config
    )
    assert bank["object_ids"].tolist() == [1]
    assert bank["indices"].tolist() == [0, 1, 3]
    # Unrelated object 3 still contests Gaussian 2; a nested exclusion is not carte blanche.
    assert 2 not in bank["indices"]
    for oid, value in evidence.items():
        np.testing.assert_array_equal(value.positive_weight, original[oid])
    child, _ = resolve_scope_alternative(
        evidence, 2, {1}, np.zeros((10, 3)), np.ones(10), config
    )
    assert child["indices"].tolist() == [0, 1]
    assert audit["conflicts"]["ambiguous"] == 1


def test_scope_alternatives_require_repeated_same_frame_nesting():
    groups = [dict(id=1, members=[0, 2]), dict(id=2, members=[1, 3])]
    nodes = [dict(id=i, frame=str(i // 2), timestamp=str(i // 2)) for i in range(4)]
    pairs = [dict(a=i, b=i + 1, containments=[0.4, 1.0]) for i in [0, 2]]
    neighbors, rows = competing_scopes(groups, nodes, pairs, {1, 2})
    assert neighbors == {1: {2}, 2: {1}}
    assert all(row["physical_relation"] == "unresolved" for row in rows)
    assert not competing_scopes(groups, nodes, pairs[:1], {1, 2})[0]
    assert not competing_scopes(groups, nodes, pairs, {1})[0]
    pairs[1]["containments"] = [1.0, 0.4]
    assert not competing_scopes(groups, nodes, pairs, {1, 2})[0]


def test_target_cannot_exclude_itself():
    with pytest.raises(ValueError, match="distinct"):
        resolve_scope_alternative(
            {1: item(1, [0])}, 1, {1}, np.zeros((2, 3)), np.ones(2), {}
        )


def current_fixture(tmp_path, *, same_timestamp=False):
    from farm_runtime.quality.native_observations import write_native_mask
    from tools.farm_shaper_bridge.common import MaskObservation

    frames = [
        SimpleNamespace(
            image_id=i,
            source_image=f"frame_{i}.png",
            depth_size=(8, 8),
            physical_timestamp=str(0 if same_timestamp else i),
        )
        for i in range(2)
    ]
    large = np.ones((8, 8), bool)
    small = np.zeros((8, 8), bool)
    small[2:6, 2:6] = True
    overrides = {}
    objects = []
    for oid, mask in [(1, large), (2, small)]:
        observations = []
        for frame in frames:
            path = tmp_path / f"{oid}_{frame.image_id}.npz"
            write_native_mask(path, mask, mask.shape, 0)
            overrides[oid, frame.image_id] = path
            observations.append(MaskObservation(oid, frame.image_id, path.name, {}))
        objects.append(SimpleNamespace(object_id=oid, observations=tuple(observations)))
    run = SimpleNamespace(
        run_dir=tmp_path,
        objects=tuple(objects),
        frames=tuple(frames),
        frame=lambda image_id: frames[image_id],
        mask_overrides=overrides,
        observation_exclusions=None,
    )
    times = sorted({f.physical_timestamp for f in frames})
    split = dict(
        build_timestamps=times,
        heldout_timestamps=[],
        objects=[dict(object_id=oid, build_timestamps=times) for oid in [1, 2]],
    )
    return run, split, large, small


def test_current_masks_include_recovered_view_and_remove_quarantined_view(tmp_path):
    from farm_runtime.quality.scope_ownership import current_scope_containment

    run, split, _, _ = current_fixture(tmp_path)
    neighbors, rows, nodes = current_scope_containment(run, split, {1, 2})
    assert neighbors == {1: {2}, 2: {1}}
    assert rows[0]["independent_timestamps"] == 2
    assert all(n["mask"]["sha256"] for n in nodes)
    # A quarantined observation is absent from the effective native input.
    run.objects[1].observations = run.objects[1].observations[:1]
    neighbors, rows, _ = current_scope_containment(run, split, {1, 2})
    assert neighbors == {}
    assert rows[0]["status"] == "single_timestamp"


def test_current_masks_do_not_count_two_virtual_views_as_two_timestamps(tmp_path):
    from farm_runtime.quality.scope_ownership import current_scope_containment

    run, split, _, _ = current_fixture(tmp_path, same_timestamp=True)
    neighbors, rows, _ = current_scope_containment(run, split, {1, 2})
    assert neighbors == {}
    assert rows[0]["independent_timestamps"] == 1


def test_current_replacement_can_reverse_containment(tmp_path):
    from farm_runtime.quality.scope_ownership import current_scope_containment
    from farm_runtime.quality.native_observations import write_native_mask

    run, split, large, small = current_fixture(tmp_path)
    write_native_mask(run.mask_overrides[1, 1], small, small.shape, 0)
    write_native_mask(run.mask_overrides[2, 1], large, large.shape, 0)
    neighbors, rows, _ = current_scope_containment(run, split, {1, 2})
    assert neighbors == {}
    assert len(rows) == 2
    assert {row["status"] for row in rows} == {"conflicting_directions"}


def test_current_masks_never_open_heldout_or_per_object_nonbuild(tmp_path):
    from farm_runtime.quality.scope_ownership import current_scope_containment

    run, split, _, _ = current_fixture(tmp_path)
    # Missing heldout files must never be opened, even for overlap statistics.
    for oid in [1, 2]:
        run.mask_overrides[oid, 1].unlink()
    split["build_timestamps"] = ["0"]
    split["heldout_timestamps"] = ["1"]
    assert current_scope_containment(run, split, {1, 2})[0] == {}
    split["build_timestamps"] = ["0", "1"]
    split["heldout_timestamps"] = []
    for row in split["objects"]:
        row["build_timestamps"] = ["0"]
    assert current_scope_containment(run, split, {1, 2})[0] == {}


def test_current_masks_remove_transient_exclusions(tmp_path):
    from farm_runtime.quality.scope_ownership import current_scope_containment
    from farm_runtime.quality_baseline import describe_file

    run, split, _, small = current_fixture(tmp_path)
    path = tmp_path / "excluded.npy"
    np.save(path, small)
    run.observation_exclusions = {i: describe_file(path) for i in [0, 1]}
    neighbors, rows, nodes = current_scope_containment(run, split, {1, 2})
    assert not neighbors and not rows
    assert {node["object_id"] for node in nodes} == {1}


def test_current_masks_reject_duplicate_observation_and_inactive_input(tmp_path):
    from farm_runtime.quality.scope_ownership import current_scope_containment

    run, split, _, _ = current_fixture(tmp_path)
    assert current_scope_containment(run, split, {1}) == ({}, [], [])
    with pytest.raises(ValueError, match="known active"):
        current_scope_containment(run, split, {9})
    run.objects[0].observations += run.objects[0].observations[:1]
    with pytest.raises(ValueError, match="unique aligned"):
        current_scope_containment(run, split, {1, 2})


def test_current_masks_reject_mixed_build_heldout(tmp_path):
    from farm_runtime.quality.scope_ownership import current_scope_containment

    run, split, _, _ = current_fixture(tmp_path)
    split["heldout_timestamps"] = ["0"]
    with pytest.raises(ValueError, match="disjoint"):
        current_scope_containment(run, split, {1, 2})
