from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest
from PIL import Image

from farm_runtime.gaussian_grouping_visualization import (
    EXPORT_SCHEMA,
    build_contact_sheet,
    compact_id_palette,
    export_heldout_frame,
    make_identity_overlay,
    sha256_file,
    validate_compact_mask,
    write_export_manifest,
)


def _fixture(
    height: int = 9, width: int = 13
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    y, x = np.indices((height, width))
    rgb = np.stack(
        (
            (x * 17) % 255,
            (y * 29) % 255,
            ((x + y) * 11) % 255,
        ),
        axis=-1,
    ).astype(np.uint8)
    target = np.zeros((height, width), dtype=np.uint8)
    target[2:7, 3:9] = 1
    prediction = target.copy()
    prediction[1:3, 8:11] = 2
    return rgb, target, prediction


def test_palette_is_deterministic_and_background_is_explicit() -> None:
    first = compact_id_palette()
    second = compact_id_palette()
    np.testing.assert_array_equal(first, second)
    assert first.shape == (256, 3)
    assert first.dtype == np.uint8
    assert first[0].tolist() == [18, 21, 27]
    assert len({tuple(row) for row in first[1:20]}) == 19


def test_compact_mask_validation_is_fail_closed() -> None:
    valid = np.asarray([[0, 1], [2, 0]], dtype=np.int64)
    result = validate_compact_mask(valid, expected_size_wh=(2, 2), maximum_id=2)
    assert result.dtype == np.uint8
    with pytest.raises(ValueError, match="shape"):
        validate_compact_mask(valid, expected_size_wh=(3, 2), maximum_id=2)
    with pytest.raises(ValueError, match="undeclared"):
        validate_compact_mask(valid, expected_size_wh=(2, 2), maximum_id=1)
    with pytest.raises(TypeError, match="integer"):
        validate_compact_mask(
            valid.astype(np.float32), expected_size_wh=(2, 2), maximum_id=2
        )
    with pytest.raises(ValueError, match=r"\[0, 254\]"):
        validate_compact_mask(valid, expected_size_wh=(2, 2), maximum_id=255)


def test_overlay_marks_disagreement_and_both_boundaries() -> None:
    rgb, target, prediction = _fixture()
    overlay = make_identity_overlay(rgb, target, prediction)
    assert overlay.shape == rgb.shape
    assert overlay.dtype == np.uint8
    assert np.any(np.all(overlay == [0, 255, 255], axis=-1))
    assert np.any(np.all(overlay == [255, 0, 255], axis=-1))
    assert not np.array_equal(overlay[target != prediction], rgb[target != prediction])


def test_frame_exports_are_exact_resolution_pngs_with_hashes(tmp_path: Path) -> None:
    rgb, target, prediction = _fixture()
    row = export_heldout_frame(
        tmp_path,
        source_name="heldout_001.png",
        source_sha256="a" * 64,
        rgb=rgb,
        target=target,
        prediction=prediction,
        maximum_id=2,
    )

    assert row["training_resolution_wh"] == [13, 9]
    assert row["target_ids"] == [0, 1]
    assert row["prediction_ids"] == [0, 1, 2]
    for name, artifact in row["artifacts"].items():
        path = tmp_path / artifact["relative_path"]
        assert path.is_file()
        assert artifact["sha256"] == sha256_file(path)
        assert artifact["size_wh"] == [13, 9]
        with Image.open(path) as image:
            assert image.format == "PNG"
            assert image.mode == ("L" if name == "compact_prediction" else "RGB")
    assert not list(tmp_path.rglob("*.html"))


def test_contact_sheet_and_manifest_cover_every_unique_frame(tmp_path: Path) -> None:
    rgb, target, prediction = _fixture()
    rows = [
        export_heldout_frame(
            tmp_path,
            source_name=f"heldout_{index:03d}.png",
            source_sha256=str(index) * 64,
            rgb=rgb,
            target=target,
            prediction=np.roll(prediction, index, axis=1),
            maximum_id=2,
        )
        for index in range(3)
    ]
    contact = build_contact_sheet(tmp_path, rows, preview_width=72, columns=2)
    summary = write_export_manifest(
        tmp_path,
        rows,
        contact,
        maximum_id=2,
        resolution_factor=4,
    )

    manifest_path = Path(summary["path"])
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    assert summary["sha256"] == sha256_file(manifest_path)
    assert manifest["schema"] == EXPORT_SCHEMA
    assert manifest["frame_count"] == 3
    assert manifest["training_resolutions_wh"] == [[13, 9]]
    assert manifest["invariants"]["all_heldout_frames_exported"] is True
    contact_path = tmp_path / contact["relative_path"]
    assert contact["sha256"] == sha256_file(contact_path)
    with Image.open(contact_path) as image:
        assert image.format == "PNG"
        assert image.mode == "RGB"
        assert image.width == 72 * 3 * 2
    assert not list(tmp_path.rglob("*.html"))


def test_manifest_rejects_duplicate_or_empty_source_names(tmp_path: Path) -> None:
    row = {"source_name": "same.png", "training_resolution_wh": [2, 2]}
    contact = {"relative_path": "contact.png", "sha256": "a" * 64}
    with pytest.raises(ValueError, match="unique"):
        write_export_manifest(
            tmp_path,
            [row, dict(row)],
            contact,
            maximum_id=1,
            resolution_factor=4,
        )


def test_manifest_rejects_output_stem_collisions(tmp_path: Path) -> None:
    rows = [
        {"source_name": "same.jpg", "training_resolution_wh": [2, 2]},
        {"source_name": "same.png", "training_resolution_wh": [2, 2]},
    ]
    contact = {"relative_path": "contact.png", "sha256": "a" * 64}
    with pytest.raises(ValueError, match="output stems"):
        write_export_manifest(
            tmp_path,
            rows,
            contact,
            maximum_id=1,
            resolution_factor=4,
        )
