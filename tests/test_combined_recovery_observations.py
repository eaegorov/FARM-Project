import copy
from types import SimpleNamespace

import numpy as np
import pytest

from farm_runtime.quality.recovery_observations import combine_recovery_batches


def fixture(tmp_path):
    def frame(name, t):
        return dict(
            frame_id=name,
            timestamp_ns=t,
            camera="cam",
            K=np.eye(3).tolist(),
            T_world_cam=np.eye(4).tolist(),
            depth_size=[5, 5],
            rgb_path=f"{name}.jpg",
            depth_path=f"{name}.npy",
        )

    def row(name):
        return dict(
            timestamp=name,
            source_image={"path": name},
            grid_shape_hw=[5, 5],
            applied_quarter_turns=0,
            detections=[{"label": "person"}],
            queries=[{"prompt": "person"}],
        )

    old = SimpleNamespace(
        frames_path=tmp_path / "frames.json",
        frames={"a": frame("a", 100), "b": frame("b", 200)},
    )
    new = SimpleNamespace(
        frames_path=old.frames_path,
        frames={**copy.deepcopy(old.frames), "c": frame("c", 300)},
    )
    mask = np.ones((5, 5), bool)

    def batch(inputs, gid, extra):
        return (
            inputs,
            {"groups": [{"group_id": gid}]},
            {
                (gid, "a"): dict(mask=mask.copy(), source_kind="original_geometry"),
                (gid, extra): dict(
                    mask=mask.copy(), source_kind="validated_additional_view"
                ),
            },
            {"a": row("a"), extra: row(extra)},
            {"a": row("a"), extra: row(extra)},
            {"a": [mask.copy()], extra: [mask.copy()]},
        )

    return [batch(old, 1, "b"), batch(new, 2, "c")], [
        {"sha256": "one"},
        {"sha256": "two"},
    ]


def test_combined_recovery_keeps_prior_observations_and_person_exclusions(tmp_path):
    batches, sources = fixture(tmp_path)
    inputs, validation, observations, rows, extras, masks = combine_recovery_batches(
        batches, sources
    )
    assert inputs is batches[1][0]
    assert [g["group_id"] for g in validation["groups"]] == [1, 2]
    assert set(observations) == {(1, "a"), (1, "b"), (2, "a"), (2, "c")}
    assert observations[1, "b"]["recovery_sources"] == [sources[0]]
    assert len(extras["a"]["detections"]) == len(masks["a"]) == 2
    assert len(batches[0][4]["a"]["detections"]) == 1
    assert "recovery_sources" not in batches[0][2][1, "a"]


def test_unconfirmed_groups_do_not_displace_the_confirmed_reserve(tmp_path):
    batches, sources = fixture(tmp_path)
    batches[1][1]["groups"].append({"group_id": 99})
    batches[1][2][99, "a"] = dict(
        mask=np.ones((5, 5), bool), source_kind="original_geometry"
    )
    result = combine_recovery_batches(batches, sources)
    assert [g["group_id"] for g in result[1]["groups"]] == [1, 2]
    assert (99, "a") not in result[2]


@pytest.mark.parametrize(
    "fault", ["timestamp", "pose", "depth", "incomplete_registry", "metadata"]
)
def test_disagreeing_recovery_sources_are_rejected(tmp_path, fault):
    batches, sources = fixture(tmp_path)
    new = batches[1][0]
    if fault == "timestamp":
        new.frames["b"]["timestamp_ns"] += 1
    elif fault == "pose":
        new.frames["b"]["T_world_cam"][0][3] += 0.01
    elif fault == "depth":
        new.frames["b"]["depth_path"] = "different.npy"
    elif fault == "incomplete_registry":
        del new.frames["b"]
    elif fault == "metadata":
        batches[1][3]["a"]["applied_quarter_turns"] = 1
    with pytest.raises(ValueError):
        combine_recovery_batches(batches, sources)


def test_conflicting_masks_cannot_silently_replace_prior_recovery(tmp_path):
    batches, sources = fixture(tmp_path)
    second = list(copy.deepcopy(batches[0]))
    second[2][1, "b"]["mask"][2, 2] = False
    with pytest.raises(ValueError, match="joint validation"):
        combine_recovery_batches([batches[0], tuple(second)], sources)


def test_repeated_equal_observation_is_not_a_second_timestamp_vote(tmp_path):
    batches, sources = fixture(tmp_path)
    result = combine_recovery_batches([batches[0], copy.deepcopy(batches[0])], sources)
    assert len(result[2]) == 2
    assert result[2][1, "b"]["recovery_sources"] == sources


def test_combined_confirmed_object_budget_remains_bounded(tmp_path):
    batches, sources = fixture(tmp_path)
    expanded = []
    for batch, ids in zip(batches, [range(9), range(9, 17)]):
        inputs, _, obs, rows, extras, masks = batch
        observations = {
            (gid, name): copy.deepcopy(value)
            for gid in ids
            for (_, name), value in obs.items()
        }
        expanded.append(
            (
                inputs,
                {"groups": [{"group_id": g} for g in ids]},
                observations,
                rows,
                extras,
                masks,
            )
        )
    with pytest.raises(ValueError, match="combined confirmed"):
        combine_recovery_batches(expanded, sources)
