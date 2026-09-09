import hashlib
import json

import pytest

from farm_runtime.quality.detector_comparison import make_plan


def test_comparison_samples_capture_groups_without_dropping_sibling_cameras(tmp_path):
    frames = []
    for i in range(5):
        for camera in ("left", "right"):
            name = f"{i}_{camera}.jpg"
            (tmp_path / name).write_bytes(f"image {i} {camera}".encode())
            frames.append(
                dict(
                    frame_id=f"{i:06d}",
                    timestamp_ns=(i + 1) * 100,
                    camera=camera,
                    rgb_path=name,
                    T_world_cam=[],
                )
            )
    path = tmp_path / "frames.json"
    path.write_text(json.dumps(dict(frames=frames)))
    vocabulary = tmp_path / "vocabulary.txt"
    vocabulary.write_text("cabinet\ncart\n")
    frozen = path.read_bytes()
    scene = dict(name="sample", frames=str(path), world_up=[0, -1, 0])
    plan = make_plan([scene], vocabulary, captures=3)
    assert [f["capture"] for f in plan["frames"]] == [
        "100",
        "100",
        "300",
        "300",
        "500",
        "500",
    ]
    assert [f["camera"] for f in plan["frames"]] == ["left", "right"] * 3
    assert len({f["key"] for f in plan["frames"]}) == 6
    assert (
        plan["vocabulary_sha256"] == hashlib.sha256(vocabulary.read_bytes()).hexdigest()
    )
    assert path.read_bytes() == frozen
    assert plan["ground_truth"] is plan["independent_heldout"] is False
    with pytest.raises(ValueError, match="duplicate scene"):
        make_plan([scene, scene], vocabulary)
    with pytest.raises(ValueError, match="at least two"):
        make_plan([scene], vocabulary, captures=1)
