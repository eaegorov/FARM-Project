import re
import sys
from argparse import Namespace
from pathlib import Path

import numpy as np
import pytest


SCRIPTS = Path(__file__).resolve().parents[1] / "scripts"
if str(SCRIPTS) not in sys.path:
    sys.path.insert(0, str(SCRIPTS))

from select_colmap_keyframes import (  # noqa: E402
    Observation,
    RegisteredView,
    main as selector_main,
    pairwise_baselines,
    parse_image_identity,
    baseline_pair_key,
    resolve_baseline_sensors,
    save_dashboard,
    validate_meters_per_scene_unit,
)


def _registered(sensor: str, center_m: list[float]) -> RegisteredView:
    return RegisteredView(
        image_id=1,
        name=f"{sensor}-0042-forward.png",
        camera_id=1,
        sensor=sensor,
        timestamp="0042",
        family="forward",
        center=np.asarray(center_m, dtype=np.float64),
        rotation=np.eye(3),
        tracks=frozenset(),
    )


def test_custom_group_names_and_metric_baseline_schema():
    pattern = re.compile(
        r"^(?P<member>[^-]+)-(?P<stamp>\d+)-(?P<view>[^.]+)\.png$"
    )
    assert parse_image_identity(
        "left-0042-forward.png",
        pattern,
        7,
        "default",
        "member",
        "stamp",
        "view",
    ) == ("left", "0042", "forward")
    with pytest.raises(ValueError, match="timestamp group"):
        parse_image_identity(
            "left-0042-forward.png", pattern, 7, "default", "member", "missing", "view"
        )

    left = _registered("left", [0.0, 0.0, 0.0])
    right = _registered("right", [0.05, 0.0, 0.0])
    observation = Observation(
        timestamp="0042",
        views=(left, right),
        center=np.asarray([0.025, 0.0, 0.0]),
        rotation=np.eye(3),
        tracks=frozenset(),
    )
    stats = pairwise_baselines([observation], ["left", "right"], 0.1)["left::right"]
    assert stats["metres"]["median"] == pytest.approx(0.05)
    assert stats["scene_units"]["median"] == pytest.approx(0.5)
    assert stats["median"] == pytest.approx(0.05)  # legacy flat field remains metric


@pytest.mark.parametrize("invalid", ["0", "-0.1", "nan", "inf"])
def test_selector_invalid_scale_has_no_output_side_effect(tmp_path, invalid):
    output = tmp_path / f"selector-{invalid}"
    with pytest.raises(ValueError, match="finite and strictly positive"):
        selector_main(
            [
                "--colmap", str(tmp_path / "missing-colmap"),
                "--output", str(output),
                "--meters-per-scene-unit", invalid,
            ]
        )
    assert not output.exists()


def test_scale_validation_accepts_legacy_default_and_rejects_nonfinite():
    assert validate_meters_per_scene_unit(1.0) == 1.0
    for value in (0.0, -1.0, np.nan, np.inf):
        with pytest.raises(ValueError):
            validate_meters_per_scene_unit(value)


def test_exact_baseline_pair_is_required_and_resolved_for_more_than_two_sensors():
    sensors = ["front", "left", "right"]
    pair = resolve_baseline_sensors(sensors, ["right", "front"], required=True)
    assert pair == ("right", "front")
    assert baseline_pair_key(pair, sensors) == "front::right"
    with pytest.raises(ValueError, match="requires --baseline-sensors"):
        resolve_baseline_sensors(sensors, None, required=True)
    with pytest.raises(ValueError, match="distinct"):
        resolve_baseline_sensors(sensors, ["front", "front"], required=True)
    with pytest.raises(ValueError, match="not selected/present"):
        resolve_baseline_sensors(sensors, ["front", "rear"], required=True)


def test_generic_selector_always_builds_six_panel_dashboard(tmp_path):
    observations = []
    for index in range(8):
        left = _registered("left", [float(index), 0.0, float(index % 3)])
        right = _registered("right", [float(index) + 0.05, 0.0, float(index % 3)])
        observations.append(
            Observation(
                timestamp=f"{index:04d}", views=(left, right),
                center=np.asarray([index + 0.025, 0.0, index % 3], dtype=float),
                rotation=np.eye(3), tracks=frozenset(range(index, index + 100)),
            )
        )
    selected = [0, 2, 4, 6, 7]
    cumulative = np.arange(8, dtype=float) * 0.7
    edges = [
        {
            "motion_gap": cumulative[right] - cumulative[left],
            "translation_m": float(right - left), "rotation_deg": 0.0,
            "overlap_ratio": 0.75, "shared_tracks": 75, "violations": [],
        }
        for left, right in zip(selected, selected[1:])
    ]
    output = tmp_path / "selection.jpg"
    save_dashboard(
        observations, selected, cumulative, edges, output, "synthetic",
        Namespace(
            max_motion_gap=2.0, max_rotation_gap_deg=50.0,
            min_overlap_ratio=0.015, min_shared_tracks=20,
        ),
        10,
        {"observed_median_m": 0.05},
    )
    from PIL import Image
    with Image.open(output) as image:
        assert image.size == (3840, 2160)
