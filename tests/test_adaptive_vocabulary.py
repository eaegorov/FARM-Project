from __future__ import annotations

import importlib.util
from pathlib import Path

import numpy as np


SCRIPT = Path(__file__).parents[1] / "scripts/build_farm_adaptive_vocabulary.py"
SPEC = importlib.util.spec_from_file_location("build_farm_adaptive_vocabulary", SCRIPT)
assert SPEC is not None and SPEC.loader is not None
MODULE = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(MODULE)


def _frame(index: int) -> dict:
    pose = np.eye(4, dtype=np.float64)
    pose[:3, 3] = [float(index % 4), float(index // 4), 0.1 * index]
    return {
        "timestamp_ns": index * 100,
        "camera": "cam00_center",
        "rgb_path": f"rgb/{index:06d}.jpg",
        "T_world_cam": pose.tolist(),
    }


def test_select_disjoint_folds_is_deterministic_and_physical() -> None:
    frames = [_frame(index) for index in range(30)]
    first = MODULE.select_disjoint_folds(frames, 8, 8)
    second = MODULE.select_disjoint_folds(frames, 8, 8)
    assert first == second
    discovery, verification = first
    assert len(discovery) == len(verification) == 8
    assert {row["timestamp_ns"] for row in discovery}.isdisjoint(
        {row["timestamp_ns"] for row in verification}
    )


def test_discovery_ignores_parts_and_normalizes_aliases() -> None:
    candidates = MODULE.discovery_candidates(
        [
            {
                "objects": [
                    {
                        "canonical_name": " Overhead Crane ",
                        "aliases": ["Bridge Crane", "crane"],
                        "topology": "multi_part_assembly",
                        "diagnostic_evidence": "bridge, hoist and hook",
                        "confidence": 0.91,
                    },
                    {
                        "canonical_name": "wheel",
                        "aliases": [],
                        "topology": "attached_component",
                        "confidence": 0.99,
                    },
                ]
            }
        ]
    )
    assert list(candidates) == ["overhead crane"]
    assert candidates["overhead crane"]["aliases"] == ["bridge crane", "crane"]


def test_verification_requires_whole_object_two_views_and_confidence() -> None:
    candidates = MODULE.discovery_candidates(
        [
            {
                "objects": [
                    {
                        "canonical_name": "overhead crane",
                        "aliases": ["bridge crane"],
                        "topology": "multi_part_assembly",
                        "confidence": 0.9,
                    },
                    {
                        "canonical_name": "control panel",
                        "aliases": [],
                        "topology": "standalone_whole",
                        "confidence": 0.8,
                    },
                ]
            }
        ]
    )
    responses = [
        {
            "objects": [
                {
                    "canonical_name": "overhead crane",
                    "decision": "keep",
                    "visible_views": 1,
                    "topology": "multi_part_assembly",
                    "confidence": 0.86,
                },
                {
                    "canonical_name": "control panel",
                    "decision": "keep",
                    "visible_views": 3,
                    "topology": "attached_component",
                    "confidence": 0.92,
                },
            ]
        },
        {
            "objects": [
                {
                    "canonical_name": "overhead crane",
                    "decision": "keep",
                    "visible_views": 2,
                    "topology": "multi_part_assembly",
                    "confidence": 0.9,
                }
            ]
        },
    ]
    audited = MODULE.verify_candidates(
        candidates, responses, min_visible_views=2, min_confidence=0.72
    )
    by_name = {row["canonical_name"]: row for row in audited}
    assert by_name["overhead crane"]["accepted"] is True
    assert by_name["overhead crane"]["verification_visible_views"] == 3
    assert by_name["control panel"]["accepted"] is False
    assert "not_complete_bounded_object" in by_name["control panel"]["rejection_reasons"]


def test_merge_preserves_base_and_bounds_additions() -> None:
    assert MODULE.dynamic_term("production unit") == ""
    assert MODULE.dynamic_term("industrial crane") == "industrial crane"
    vocabulary, additions = MODULE.merged_vocabulary(
        ["box", "cart", "box"],
        [
            {
                "canonical_name": "overhead crane",
                "aliases": ["bridge crane", "gantry crane"],
                "accepted": True,
                "verification_confidence": 0.95,
                "verification_visible_views": 4,
            },
            {
                "canonical_name": "wall panel",
                "aliases": [],
                "accepted": False,
                "verification_confidence": 0.99,
                "verification_visible_views": 5,
            },
        ],
        2,
    )
    assert vocabulary[:2] == ["box", "cart"]
    assert additions == ["overhead crane", "bridge crane"]
    assert "wall panel" not in vocabulary
