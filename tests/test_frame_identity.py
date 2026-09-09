import copy
import json

import pytest

from farm_runtime.frame_identity import normalize_capture_timestamps


def test_registered_rgbd_loads_without_adapter_and_sibling_cameras_count_once(tmp_path):
    from scripts.geometry.refine_farm_object_geometry import _load_frames

    rows = [
        dict(frame_id="000260", timestamp_ns=17, rgb_path="left.jpg"),
        dict(frame_id="000260", timestamp_ns=17, rgb_path="right.jpg"),
        dict(frame_id="000580", timestamp_ns=34, rgb_path="later.jpg"),
    ]
    path = tmp_path / "frames.json"
    path.write_text(json.dumps(dict(schema_version="farm_frames_json_v1", frames=rows)))
    frozen = path.read_bytes()
    _, loaded = _load_frames(path)
    assert loaded["left"]["physical_timestamp"] == loaded["right"]["physical_timestamp"] == "17"
    assert loaded["later"]["physical_timestamp"] == "34"
    assert path.read_bytes() == frozen


def test_mixed_legacy_key_is_shared_with_missing_sibling_without_mutation():
    rows = [
        dict(frame_id="frame-a", timestamp_ns=100, physical_timestamp="00042"),
        dict(frame_id="frame-a", timestamp_ns=100),
        dict(frame_id="frame-b", timestamp_ns=200),
    ]
    frozen = copy.deepcopy(rows)
    normalized = normalize_capture_timestamps(rows)
    assert [r["physical_timestamp"] for r in normalized] == ["42", "42", "200"]
    assert rows == frozen


@pytest.mark.parametrize("fault", [
    "missing_clock", "float_clock", "bool_clock", "zero_clock",
    "different_clocks", "clock_collision", "key_collision",
    "contradictory_key", "invalid_key",
])
def test_inconsistent_capture_identity_fails_instead_of_counting_extra_views(fault):
    rows = [
        dict(frame_id="a", timestamp_ns=100),
        dict(frame_id="a", timestamp_ns=100),
        dict(frame_id="b", timestamp_ns=200),
    ]
    if fault == "missing_clock":
        del rows[0]["timestamp_ns"]
    elif fault == "float_clock":
        rows[0]["timestamp_ns"] = 100.0
    elif fault == "bool_clock":
        rows[0]["timestamp_ns"] = True
    elif fault == "zero_clock":
        rows[0]["timestamp_ns"] = 0
    elif fault == "different_clocks":
        rows[1]["timestamp_ns"] = 101
    elif fault == "clock_collision":
        rows[2]["timestamp_ns"] = 100
    elif fault == "key_collision":
        rows[0]["physical_timestamp"] = "200"
    elif fault == "contradictory_key":
        rows[0]["physical_timestamp"] = "42"
        rows[1]["physical_timestamp"] = "43"
    elif fault == "invalid_key":
        rows[0]["physical_timestamp"] = "bad"
    with pytest.raises(ValueError):
        normalize_capture_timestamps(rows)
