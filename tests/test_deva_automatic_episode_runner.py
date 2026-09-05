from __future__ import annotations

import importlib.util
import json
from pathlib import Path

import pytest


ROOT = Path(__file__).resolve().parents[1]
SCRIPT = ROOT / "scripts/tracking/run_farm_deva_automatic_episode.py"
SPEC = importlib.util.spec_from_file_location("farm_deva_episode_runner", SCRIPT)
assert SPEC is not None and SPEC.loader is not None
MODULE = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(MODULE)


def test_frame_list_and_directory_must_match_exactly(tmp_path: Path) -> None:
    image_dir = tmp_path / "images"
    image_dir.mkdir()
    (image_dir / "b.png").write_bytes(b"b")
    (image_dir / "a.jpg").write_bytes(b"a")
    frame_list = tmp_path / "frames.txt"
    frame_list.write_text("# ordered by the planner\nb.png\na.jpg\n", encoding="utf-8")

    expected = MODULE._read_expected_frames(frame_list)
    paths = MODULE._validate_image_directory(image_dir, expected_frames=expected)

    assert [path.name for path in paths] == ["a.jpg", "b.png"]


def test_non_image_or_unexpected_entry_is_rejected(tmp_path: Path) -> None:
    image_dir = tmp_path / "images"
    image_dir.mkdir()
    (image_dir / "frame.png").write_bytes(b"image")
    (image_dir / "notes.txt").write_text("not a frame", encoding="utf-8")

    with pytest.raises(ValueError, match="only image"):
        MODULE._validate_image_directory(image_dir, expected_frames=None)


def test_duplicate_or_path_frame_names_are_rejected(tmp_path: Path) -> None:
    duplicate = tmp_path / "duplicate.txt"
    duplicate.write_text("a.png\na.png\n", encoding="utf-8")
    with pytest.raises(ValueError, match="duplicate"):
        MODULE._read_expected_frames(duplicate)

    nested = tmp_path / "nested.txt"
    nested.write_text("stream/a.png\n", encoding="utf-8")
    with pytest.raises(ValueError, match="basenames"):
        MODULE._read_expected_frames(nested)


def test_prediction_summary_counts_local_identities(tmp_path: Path) -> None:
    prediction = tmp_path / "pred.json"
    prediction.write_text(
        json.dumps(
            {
                "annotations": [
                    {
                        "segments_info": [
                            {"id": 4, "area": 10},
                            {"id": 7, "area": 25},
                        ]
                    },
                    {"segments_info": [{"id": 4, "area": 13}]},
                ]
            }
        ),
        encoding="utf-8",
    )

    summary = MODULE._summarize_predictions(prediction)

    assert summary == {
        "annotated_frames": 2,
        "unique_local_identities": 2,
        "identity_observations": 3,
        "minimum_observation_area_px": 10,
        "maximum_observation_area_px": 25,
    }


def test_output_directory_must_be_empty(tmp_path: Path) -> None:
    output = tmp_path / "output"
    output.mkdir()
    (output / "old.png").write_bytes(b"old")

    with pytest.raises(ValueError, match="not empty"):
        MODULE._prepare_output(output)
