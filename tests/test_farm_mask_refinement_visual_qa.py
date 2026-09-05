from __future__ import annotations

import hashlib
import json
from pathlib import Path

import cv2
import numpy as np
import pytest

from scripts.evaluation.build_farm_mask_refinement_visual_qa import (
    INDEX_SCHEMA,
    build_qa,
    compute_context_crop,
    validate_result,
)


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _write_fixture(tmp_path: Path) -> Path:
    root = tmp_path / "run"
    visuals = root / "visuals"
    masks = root / "masks" / "object_000007"
    visuals.mkdir(parents=True)
    masks.mkdir(parents=True)
    height = width = 64
    yy, xx = np.mgrid[:height, :width]
    source = np.stack(
        ((xx * 3 + 20) % 255, (yy * 4 + 30) % 255, ((xx + yy) * 2 + 40) % 255),
        axis=2,
    ).astype(np.uint8)
    seed_mask = np.zeros((height, width), dtype=bool)
    seed_mask[19:39, 20:37] = True
    sam_mask = np.zeros_like(seed_mask)
    sam_mask[17:42, 18:40] = True
    final_mask = np.zeros_like(seed_mask)
    final_mask[20:40, 21:38] = True

    def overlay(mask: np.ndarray, color: tuple[int, int, int]) -> np.ndarray:
        panel = source.copy()
        panel[mask] = (0.32 * panel[mask] + 0.68 * np.asarray(color)).astype(np.uint8)
        cv2.rectangle(panel, (20, 19), (38, 40), (255, 255, 255), 1)
        return panel

    # Fixture is BGR because cv2 writes/reads it without a colour-space conversion.
    sheet = np.concatenate(
        [source, overlay(seed_mask, (90, 70, 245)), overlay(sam_mask, (45, 190, 255)), overlay(final_mask, (155, 225, 40))],
        axis=1,
    )
    visual_relative = Path("visuals/object_000007__img_000011.jpg")
    assert cv2.imwrite(str(root / visual_relative), sheet, [int(cv2.IMWRITE_JPEG_QUALITY), 98])

    x0, y0, x1, y1 = 21, 20, 38, 40
    cropped_mask = final_mask[y0:y1, x0:x1]
    raw_bits = np.packbits(cropped_mask.reshape(-1), bitorder="little")
    mask_relative = Path("masks/object_000007/img_000011_det_0000.npz")
    np.savez_compressed(
        root / mask_relative,
        image_shape=np.asarray([height, width], dtype=np.int32),
        raw_bits=raw_bits,
        raw_shape=np.asarray(cropped_mask.shape, dtype=np.int32),
        raw_bbox_xyxy=np.asarray([x0, y0, x1, y1], dtype=np.int32),
    )
    mask_sha256 = _sha256(root / mask_relative)
    view = {
        "source_image": "cam00_000001_center.png",
        "global_image_id": 11,
        "physical_timestamp_ns": 1,
        "accepted": True,
        "seed_origin": "synthetic_seed",
        "confidence": 0.99,
        "raw_pixels": int(seed_mask.sum()),
        "sam_pixels": int(sam_mask.sum()),
        "final_pixels": int(final_mask.sum()),
        "stored_mask_pixels": int(final_mask.sum()),
        "raw_recall": 0.95,
        "area_expansion": 1.0,
        "positive_inclusion": 1.0,
        "negative_inclusion": 0.0,
        "other_mask_fraction": 0.0,
        "depth_inlier_fraction": 0.98,
        "obb_inside_fraction": 0.92,
        "quality_gates": {"confidence": True, "depth": True},
        "advisory_gates": {"multiview": False},
        "prompt_box_xyxy": [18.2, 16.1, 41.8, 43.2],
        "visual_relative": visual_relative.as_posix(),
        "mask_relative": mask_relative.as_posix(),
        "mask_sha256": mask_sha256,
    }
    result = {
        "schema": "farm.full-colmap-mask-refinement.v1",
        "status": "PASS",
        "accepted_objects": 1,
        "accepted_object_ids": [7],
        "objects": [
            {
                "object_id": 7,
                "status": "accepted",
                "accepted": True,
                "accepted_views": 1,
                "views": [view],
            }
        ],
    }
    result_path = root / "result.json"
    result_path.write_text(json.dumps(result), encoding="utf-8")
    return result_path


def test_context_crop_contains_union_and_clips_only_at_frame_boundary() -> None:
    crop = compute_context_crop([(0, 5, 9, 15), (20, 12, 31, 29)], 40, 32, context_fraction=0.25, minimum_context_pixels=4)
    assert crop == (0, 0, 39, 32)
    assert crop[0] <= 0 and crop[1] <= 5 and crop[2] >= 31 and crop[3] >= 29


def test_validate_result_rejects_non_pass_and_duplicate_views() -> None:
    payload = {
        "schema": "farm.full-colmap-mask-refinement.v1",
        "status": "FAIL",
        "accepted_objects": 0,
        "accepted_object_ids": [],
        "objects": [],
    }
    with pytest.raises(ValueError, match="not PASS"):
        validate_result(payload)


def test_build_qa_is_nonblank_unique_and_reproducible(tmp_path: Path) -> None:
    result_path = _write_fixture(tmp_path)
    output = tmp_path / "qa"
    first = build_qa(result_path, output, panel_width=180, panel_height=180)
    assert first["schema"] == INDEX_SCHEMA
    assert first["release_acceptance"] is False
    assert first["accepted_object_ids"] == [7]
    assert first["accepted_views"] == 1
    assert len(first["objects"]) == 1
    view = first["objects"][0]["views"][0]
    assert view["qa_checks"]["same_crop_all_four_panels"] is True
    assert view["qa_checks"]["all_bbox_inputs_contained"] is True
    assert view["qa_checks"]["panel_crops_nonblank"] is True

    expected = {"object_000007.jpg", "index.json", "REPORT_RU.md"}
    assert {path.name for path in output.iterdir()} == expected
    sheet = cv2.imread(str(output / "object_000007.jpg"), cv2.IMREAD_COLOR)
    assert sheet is not None and min(sheet.shape[:2]) > 180
    assert float(np.std(sheet)) > 1.0
    report = (output / "REPORT_RU.md").read_text(encoding="utf-8")
    assert "не release acceptance" in report
    assert "`7`" in report

    hashes = {name: _sha256(output / name) for name in expected}
    second = build_qa(result_path, output, panel_width=180, panel_height=180)
    assert second == first
    assert hashes == {name: _sha256(output / name) for name in expected}
    assert not list(output.glob("*.html"))
    assert not list(output.glob(".farm-mask-qa-*.tmp"))
