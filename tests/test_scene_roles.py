"""Physical capture and provenance contracts for scene-surface routing."""

import json
from types import SimpleNamespace

import pytest

from farm_runtime.quality.scene_roles import (
    POLICY,
    bind_context_selection,
    surface_context,
)
from farm_runtime.quality_baseline import describe_file, write_json


def geometry():
    nodes, groups = [], []
    for oid, labels in enumerate(
        [["ceiling"], ["ceiling lamp"], ["wall", "cabinet"], ["floor panel"]]
    ):
        members = []
        for t in ("10", "20"):
            members.append(len(nodes))
            nodes.append(
                dict(id=len(nodes), labels=labels, timestamp=t, frame=f"cam_{t}_{oid}")
            )
        groups.append(
            dict(
                id=oid,
                members=members,
                candidate_labels=labels,
                independent_timestamps=2,
            )
        )
    return dict(test_opened=False, nodes=nodes, groups=groups)


def test_only_continuous_surfaces_are_routed_fixtures_and_mixed_categories_remain():
    g = geometry()
    rows = surface_context(g)
    assert [r["object_id"] for r in rows] == [0]
    assert not rows[0]["physical_role_validated"]
    assert rows[0]["observation_node_ids"] == [0, 1]
    assert rows[0]["physical_timestamps"] == ["10", "20"]
    assert g == geometry()  # no source category/mask mutation


def test_same_capture_stereo_pair_does_not_confirm_surface():
    g = geometry()
    g["nodes"][1]["timestamp"] = "10"
    g["groups"][0]["independent_timestamps"] = 1
    assert surface_context(g) == []


@pytest.mark.parametrize("bad_labels", [[], None, [""]])
def test_missing_category_evidence_does_not_remove_candidate(bad_labels):
    g = geometry()
    g["nodes"][0]["labels"] = bad_labels
    assert surface_context(g) == []


@pytest.mark.parametrize("fault", ["category", "timestamp"])
def test_inconsistent_geometry_evidence_is_rejected(fault):
    g = geometry()
    if fault == "category":
        g["groups"][0]["candidate_labels"] = ["wall"]
    else:
        g["groups"][0]["independent_timestamps"] = 5
    with pytest.raises(ValueError):
        surface_context(g)


def selection_fixture(tmp_path):
    g = geometry()
    path = tmp_path / "geometry.json"
    write_json(path, g)
    ref = describe_file(path)
    selection = dict(
        source_geometry=ref,
        context_policy=POLICY,
        context_groups=surface_context(g),
        selected_group_ids=[1, 2, 3],
    )
    p = tmp_path / "selection.json"
    write_json(p, selection)
    return p, ref, selection


def test_catalog_preserves_context_evidence_separately(tmp_path):
    p, ref, s = selection_fixture(tmp_path)
    assert bind_context_selection(p, ref, [1, 2, 3]) == s["context_groups"]


@pytest.mark.parametrize(
    "fault", ["namespace", "overlap", "invented_role", "duplicate", "missing_object"]
)
def test_context_export_rejects_wrong_namespace_or_ownership(tmp_path, fault):
    p, ref, s = selection_fixture(tmp_path)
    ids = [1, 2, 3]
    if fault == "namespace":
        ref = dict(ref, sha256="0" * 64)
    elif fault == "overlap":
        ids.append(0)
    elif fault == "invented_role":
        s["context_groups"][0]["physical_role_validated"] = True
    elif fault == "duplicate":
        s["context_groups"] *= 2
    else:
        ids.append(99)
    write_json(p, s)
    with pytest.raises(ValueError):
        bind_context_selection(p, ref, ids)


@pytest.mark.parametrize(
    "enabled,recovered,expected",
    [
        (False, False, [0, 1, 2, 3]),
        (True, False, [1, 2, 3]),
        (True, True, [0, 1, 2, 3]),
    ],
)
def test_native_routing_preserves_legacy_and_recovered_candidates(
    tmp_path, monkeypatch, enabled, recovered, expected
):
    from farm_runtime.quality import scene_profile, native_observations

    p = tmp_path / "geometry.json"
    write_json(p, geometry())
    args = SimpleNamespace(
        geometry=p,
        groups=4,
        alternatives=0,
        ply=tmp_path / "scene.ply",
        config=tmp_path / "config.json",
        world_up=[0, -1, 0],
        output=tmp_path / "native",
        separate_scene_surfaces=enabled,
        recovery_validation=tmp_path / "recovery.json" if recovered else None,
    )
    calls = []

    def native(argv):
        calls.append(argv)
        args.output.mkdir()

    monkeypatch.setattr(native_observations, "main", native)
    monkeypatch.setattr(
        native_observations, "confirmed_recovery_groups", lambda *a, **k: [0]
    )
    # The recovery source needs a real descriptor but not inference in this unit.
    if recovered:
        write_json(args.recovery_validation, {})
    scene_profile.native(args)
    cmd = calls[0]
    assert [int(cmd[i + 1]) for i, v in enumerate(cmd) if v == "--group-id"] == expected
    result = json.loads((args.output / "selection.json").read_text())
    assert result["selected_group_ids"] == expected
    if enabled:
        assert [r["object_id"] for r in result["context_groups"]] == (
            [] if recovered else [0]
        )
    else:
        assert "context_groups" not in result
