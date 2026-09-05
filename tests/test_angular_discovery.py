from collections import Counter
from types import SimpleNamespace

import numpy as np
import pytest

from farm_runtime.angular_discovery import (
    balanced_view_selection,
    rotate_image,
    upright_quarter_turns,
)


def rig(count=10):
    timestamps = [str(i) for i in range(count)]
    families = ["forward", "left", "right", "up", "down"]
    rows = {
        t: [
            SimpleNamespace(
                name=f"{s}_{t}_{f}",
                sensor=s,
                timestamp=t,
                family=f,
                tracks=frozenset(range(j * 10, j * 10 + 10)),
            )
            for s in ("a", "b")
            for j, f in enumerate(families)
        ]
        for t in timestamps
    }
    return rows, timestamps, families


def test_balanced_coverage_keeps_physical_and_sensor_budget():
    rows, ts, families = rig()
    views, report = balanced_view_selection(rows, ts, ["a", "b"], families)
    assert len(views) == len(ts) * 2
    assert Counter(v.timestamp for v in views) == {t: 2 for t in ts}
    assert report["family_counts_by_sensor"] == {
        s: {f: 2 for f in families} for s in ("a", "b")
    }
    assert [v.name for v in views] == [
        v.name
        for v in balanced_view_selection(
            {t: list(reversed(v)) for t, v in rows.items()}, ts, ["a", "b"], families
        )[0]
    ]


def test_missing_families_are_reported_and_duplicates_rejected():
    rows, ts, families = rig(1)
    rows["0"] = [v for v in rows["0"] if v.family != "down"]
    views, report = balanced_view_selection(rows, ts, ["a", "b"], families)
    assert len(views) == 2 and len(report["missing_family_observations"]) == 2
    with pytest.raises(ValueError, match="insufficient"):
        balanced_view_selection(rows, ts, ["a", "b"], families, views_per_sensor=5)
    rows["0"].append(rows["0"][0])
    with pytest.raises(ValueError, match="duplicate"):
        balanced_view_selection(rows, ts, ["a", "b"], families)


@pytest.mark.parametrize("turns", [0, 1, 2, 3])
def test_upright_predictions_return_to_original_pixels_without_interpolation(turns):
    logits = np.arange(35, dtype=np.float32).reshape(5, 7) / 35
    logits[0, 6] = -0.234
    rotated = rotate_image(logits, turns)
    restored = rotate_image(rotated, -turns)
    assert np.array_equal(restored, logits)
    assert restored.dtype == logits.dtype and restored.flags.c_contiguous


def test_gravity_upright_uses_camera_axes_and_reports_degenerate_views():
    r = np.array([[0.0, -1.0, 0.0], [1.0, 0.0, 0.0], [0.0, 0.0, 1.0]])
    orientation = upright_quarter_turns(r, [0, -1, 0])
    assert orientation["applied_quarter_turns_ccw"] == 1
    assert orientation["status"] == "applied"
    assert upright_quarter_turns(np.eye(3), [0, 0, 1])["status"] == "unavailable"
    with pytest.raises(ValueError, match="orthonormal"):
        upright_quarter_turns(np.ones((3, 3)), [0, -1, 0])
