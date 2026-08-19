from __future__ import annotations

import importlib.util
from pathlib import Path

import cv2
import numpy as np


MODULE_PATH = Path(__file__).resolve().parents[1] / "scripts/build_farm_run_report.py"
SPEC = importlib.util.spec_from_file_location("build_farm_run_report", MODULE_PATH)
assert SPEC and SPEC.loader
MODULE = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(MODULE)


def test_compose_slide_is_full_hd_and_source_visible() -> None:
    source = np.full((300, 500, 3), (20, 120, 220), dtype=np.uint8)
    slide = MODULE.compose_slide(source, "TEST VISUAL", 2, 9)
    assert slide.shape == (1080, 1920, 3)
    assert float(slide.std()) > 10
    assert np.count_nonzero(slide[:, :, 2] > 150) > 10_000


def test_video_builder_uses_every_indexed_image(tmp_path: Path) -> None:
    visual = tmp_path / "visuals"
    visual.mkdir()
    rows = []
    for index, colour in enumerate(((40, 80, 200), (200, 120, 40)), 1):
        path = visual / f"0{index}_test.jpg"
        assert cv2.imwrite(str(path), np.full((240, 400, 3), colour, dtype=np.uint8))
        rows.append({"path": path.name, "bytes": path.stat().st_size})
    (visual / "index.json").write_text(
        __import__("json").dumps({"images": rows}), encoding="utf-8"
    )
    result = MODULE.build_video(tmp_path, visual / "scene_summary.mp4", fps=4, slide_seconds=0.5)
    assert result["slides"] == 2
    assert result["frame_count"] == 4
    assert result["duration_seconds"] == 1.0
    assert result["bytes"] > 10_000
    capture = cv2.VideoCapture(str(visual / "scene_summary.mp4"))
    assert capture.isOpened()
    assert int(capture.get(cv2.CAP_PROP_FRAME_WIDTH)) == 1920
    assert int(capture.get(cv2.CAP_PROP_FRAME_HEIGHT)) == 1080
    capture.release()


def test_warning_table_preserves_code_measure_and_threshold() -> None:
    rendered = MODULE.warning_table({
        "warnings": [{
            "code": "surface_diagnostic_high_rejection",
            "detail": "Surface support rejected candidates",
            "value": 18.7,
            "threshold": 25.0,
        }]
    })

    assert "surface_diagnostic_high_rejection" in rendered
    assert "Surface support rejected candidates" in rendered
    assert "18.7" in rendered
    assert "25" in rendered


def test_human_measure_is_compact_and_stable() -> None:
    assert MODULE.human_measure(29.5048828125) == "29.5"
    assert MODULE.human_measure(None) == "н/д"


def test_rendered_video_timing_uses_emitted_frame_count() -> None:
    timing = MODULE.rendered_video_timing(11, 12, 1.8)

    assert timing["frames_per_slide"] == 22
    assert timing["fade_frames"] == 3
    assert timing["frame_count"] == 272
    assert timing["duration_seconds"] == 272 / 12


def test_hidden_breakdown_is_disjoint_and_discloses_overlap() -> None:
    catalog = [
        {"id": 1, "semantic_tier": "geometry_only", "display_status": "duplicate_suppressed"},
        {"id": 2, "semantic_tier": "geometry_only", "display_status": "canonical"},
        {"id": 3, "semantic_tier": "probable", "display_status": "part_suppressed"},
        {"id": 4, "semantic_tier": "confirmed", "display_status": "canonical"},
    ]

    rendered = MODULE.describe_hidden_breakdown(catalog, [{"id": 4}])

    assert rendered == (
        "3 = 2 geometry-only + 1 resolved-suppressed "
        "(2 suppression records total; 1 overlap geometry-only)"
    )
