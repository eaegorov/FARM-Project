from __future__ import annotations

import json
import sys

import pytest

from scripts.evaluation.build_farm_experiment_run import (
    _load_unique_input_frames,
    _validate_presentation_catalog,
    main,
)


def test_presentation_catalog_requires_explicit_visible_rows(tmp_path) -> None:
    path = tmp_path / "catalog.json"
    path.write_text(json.dumps([{"id": 7}]), encoding="utf-8")

    with pytest.raises(ValueError, match="not explicitly visible"):
        _validate_presentation_catalog(path)


def test_presentation_catalog_accepts_unique_nonnegative_visible_ids(tmp_path) -> None:
    path = tmp_path / "catalog.json"
    path.write_text(
        json.dumps(
            [
                {"id": 7, "presentation_visible": True},
                {"id": 11, "presentation_visible": True},
            ]
        ),
        encoding="utf-8",
    )

    assert _validate_presentation_catalog(path) == [7, 11]


@pytest.mark.parametrize(
    "rows, message",
    (
        ([], "non-empty"),
        ([{"id": -1, "presentation_visible": True}], "non-negative"),
        (
            [
                {"id": 7, "presentation_visible": True},
                {"id": 7, "presentation_visible": True},
            ],
            "unique",
        ),
    ),
)
def test_presentation_catalog_rejects_invalid_identity_contract(
    tmp_path, rows, message
) -> None:
    path = tmp_path / "catalog.json"
    path.write_text(json.dumps(rows), encoding="utf-8")

    with pytest.raises(ValueError, match=message):
        _validate_presentation_catalog(path)


def test_input_frames_require_unique_source_images(tmp_path) -> None:
    path = tmp_path / "frames.json"
    path.write_text(
        json.dumps(
            {
                "frames": [
                    {"source_image": "cam0_100_center.png"},
                    {"source_image": "cam0_100_center.png"},
                ]
            }
        ),
        encoding="utf-8",
    )

    with pytest.raises(
        ValueError,
        match=r"duplicate source_image .* rows 0 and 1",
    ):
        _load_unique_input_frames(path)


def test_duplicate_input_frames_fail_before_output_creation(
    tmp_path, monkeypatch
) -> None:
    template = tmp_path / "template"
    template.mkdir()
    presentation = tmp_path / "presentation.json"
    presentation.write_text(
        json.dumps([{"id": 7, "presentation_visible": True}]),
        encoding="utf-8",
    )
    frames = tmp_path / "frames.json"
    frames.write_text(
        json.dumps(
            {
                "frames": [
                    {"source_image": "cam0_100_center.png"},
                    {"source_image": "cam0_100_center.png"},
                ]
            }
        ),
        encoding="utf-8",
    )
    output = tmp_path / "must_not_exist"
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "build_farm_experiment_run.py",
            "--template-run",
            str(template),
            "--scene-state",
            str(tmp_path / "unused-state.pt"),
            "--catalog",
            str(tmp_path / "unused-catalog.json"),
            "--presentation-catalog",
            str(presentation),
            "--cloud",
            str(tmp_path / "unused-cloud.npz"),
            "--frames-json",
            str(frames),
            "--mask-root",
            str(tmp_path / "unused-masks"),
            "--output-run",
            str(output),
        ],
    )

    with pytest.raises(ValueError, match="duplicate source_image"):
        main()

    assert not output.exists()
