import sys
from pathlib import Path

import torch


SCRIPTS = Path(__file__).resolve().parents[1] / "scripts"
if str(SCRIPTS) not in sys.path:
    sys.path.insert(0, str(SCRIPTS))

from unify_farm_compound_presentation import unify_compound_boxes  # noqa: E402


def _box(center, support=0.96):
    return {
        "center_m": center,
        "dimensions_lwh_m": [2.0, 0.4, 2.2],
        "wxyz": [1.0, 0.0, 0.0, 0.0],
        "gaussian_supported_rate": support,
    }


def test_unified_export_replaces_only_presentation_components():
    payload = {
        "state": {
            "object_id": torch.tensor([405, 7]),
            "object_geometry_status": ["assembly_compound_geometry_probable", "geometry_pass"],
            "object_compound_boxes": [[_box([0, 0, 0]), _box([1, 0, 0])], []],
            "object_compound_geometry_diagnostics": [
                {"compound": {"parent_box": _box([0.5, 0, 0])}},
                {},
            ],
            "active": torch.tensor([False, True]),
        },
        "meta": {},
    }
    active_before = payload["state"]["active"].clone()

    report = unify_compound_boxes(payload, [405])

    assert report["accepted_objects"] == 1
    assert len(payload["state"]["object_compound_boxes"][0]) == 1
    assert payload["state"]["object_compound_boxes"][0][0]["center_m"] == [0.5, 0.0, 0.0]
    assert torch.equal(payload["state"]["active"], active_before)
    assert report["objects"][0]["source_component_count"] == 2
    assert payload["state"]["object_compound_presentation_unified_ids"] == [405]


def test_unified_export_fails_closed_on_low_surface_support():
    payload = {
        "state": {
            "object_id": torch.tensor([405]),
            "object_geometry_status": ["assembly_compound_geometry_probable"],
            "object_compound_boxes": [[_box([0, 0, 0]), _box([1, 0, 0])]],
            "object_compound_geometry_diagnostics": [
                {"compound": {"parent_box": _box([0.5, 0, 0], support=0.40)}}
            ],
        },
        "meta": {},
    }

    report = unify_compound_boxes(payload, [405])

    assert report["accepted_objects"] == 0
    assert len(payload["state"]["object_compound_boxes"][0]) == 2


def test_all_eligible_unifies_only_audited_supported_compounds():
    payload = {
        "state": {
            "object_id": torch.tensor([401, 402, 7]),
            "object_geometry_status": [
                "assembly_compound_geometry_probable",
                "assembly_compound_geometry_probable",
                "geometry_pass",
            ],
            "object_compound_boxes": [
                [_box([0, 0, 0]), _box([1, 0, 0])],
                [_box([3, 0, 0]), _box([4, 0, 0])],
                [],
            ],
            "object_compound_geometry_diagnostics": [
                {"compound": {"parent_box": _box([0.5, 0, 0], support=0.95)}},
                {"compound": {"parent_box": _box([3.5, 0, 0], support=0.55)}},
                {},
            ],
        },
        "meta": {},
    }

    report = unify_compound_boxes(payload, None)

    assert report["selection_mode"] == "all_eligible"
    assert report["accepted_objects"] == 1
    assert len(payload["state"]["object_compound_boxes"][0]) == 1
    assert len(payload["state"]["object_compound_boxes"][1]) == 2
    assert payload["state"]["object_compound_presentation_unified_ids"] == [401]
