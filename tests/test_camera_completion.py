import numpy as np
import pytest
from farm_runtime.quality.camera_completion import camera_candidate, independent_views


def fixture():
    x, y = np.meshgrid(np.linspace(-0.1, 0.1, 10), np.linspace(-0.05, 0.05, 10))
    points = np.column_stack((x.ravel(), y.ravel(), np.ones(100)))
    return (
        points,
        np.ones(100, bool),
        np.array([[400.0, 0, 1000], [0, 400.0, 1000], [0, 0, 1]]),
        np.eye(4)[:3],
        (2000, 2000),
    )


def test_small_object_is_not_rejected_by_whole_image_area():
    # 80 x 40 pixel object on a 4 megapixel image: 0.08% image area.
    row = camera_candidate(*fixture())
    assert row is not None and row["projected_area_fraction"] < 0.001
    assert row["frustum_fraction"] == 1
    np.testing.assert_allclose(row["center_m"], 0)
    assert (
        camera_candidate(*fixture(), min_short_pixels=64, min_long_pixels=128) is None
    )


def test_camera_gate_rejects_behind_camera_and_outside_fov():
    for translation in ([0, 0, -2], [4, 0, 0]):
        args = list(fixture())
        args[3][:, 3] = translation
        assert camera_candidate(*args) is None
    args = list(fixture())
    args[0][0, 0] = np.nan
    with pytest.raises(ValueError, match="finite metric"):
        camera_candidate(*args)


def test_unconfirmed_surface_must_also_fit_camera():
    args = list(fixture())
    args[0][:5, 0] = 4
    args[1][:] = False
    args[1][:5] = True
    assert camera_candidate(*args) is None
    args[1][:] = False
    assert camera_candidate(*args) is not None  # no unresolved subset: all support


def test_camera_selection_uses_timestamp_and_physical_baseline():
    rows = [
        dict(name=name, timestamp=ts, center_m=[x, 0, 0], score=score)
        for name, ts, x, score in [
            ("a", "1", 0, 4),
            ("b", "1", 1, 3.9),
            ("c", "2", 0.03, 3.8),
            ("d", "3", 0.3, 3.7),
            ("e", "4", 0.7, 3.6),
        ]
    ]
    assert [r["name"] for r in independent_views(rows, 2)] == ["a", "d"]
    assert [r["name"] for r in independent_views(list(reversed(rows)), 2)] == ["a", "d"]
    with pytest.raises(ValueError, match="budget"):
        independent_views(rows, 9)
