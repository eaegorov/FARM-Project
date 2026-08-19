from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pytest
import yaml

SRC = Path(__file__).resolve().parents[1] / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from farm_pipeline.preflight import fingerprint_file, run_preflight
from farm_pipeline.scene_config import SceneConfigError, load_scene_config


POINTS = np.asarray(
    [
        (-1.0, -1.0, -1.0), (1.0, -1.0, -1.0), (-1.0, 1.0, -1.0),
        (1.0, 1.0, -1.0), (-1.0, -1.0, 1.0), (1.0, -1.0, 1.0),
        (-1.0, 1.0, 1.0), (1.0, 1.0, 1.0), (0.0, 0.0, 0.0),
        (0.5, 0.0, 0.0), (0.0, 0.5, 0.0), (0.0, 0.0, 0.5),
    ],
    dtype=np.float64,
)


def _write_colmap(root: Path) -> tuple[Path, Path]:
    model = root / "sparse" / "0"
    images = root / "images"
    model.mkdir(parents=True)
    images.mkdir()
    (model / "cameras.txt").write_text(
        "# cameras\n1 PINHOLE 640 480 500 500 320 240\n", encoding="utf-8"
    )
    image_lines = []
    for image_id, name, tx in (
        (1, "cam0_000001_center.png", 0.0),
        (2, "cam0_000002_center.png", -0.2),
    ):
        image_lines.extend(
            [
                f"{image_id} 1 0 0 0 {tx} 0 0 1 {name}",
                "10 10 1 20 20 2",
            ]
        )
        (images / name).write_bytes(b"synthetic-image")
    (model / "images.txt").write_text("\n".join(image_lines) + "\n", encoding="utf-8")
    point_lines = [
        f"{index} {x} {y} {z} 128 128 128 0.1 1 0"
        for index, (x, y, z) in enumerate(POINTS, 1)
    ]
    (model / "points3D.txt").write_text("\n".join(point_lines) + "\n", encoding="utf-8")
    return model, images


def _write_ply(path: Path, offset: float = 0.0, gaussian: bool = True) -> None:
    properties = ["x", "y", "z"]
    if gaussian:
        properties += [
            "f_dc_0", "f_dc_1", "f_dc_2", "opacity",
            "scale_0", "scale_1", "scale_2", "rot_0", "rot_1", "rot_2", "rot_3",
        ]
    header = ["ply", "format ascii 1.0", f"element vertex {len(POINTS)}"]
    header.extend(f"property float {name}" for name in properties)
    header.append("end_header")
    rows = []
    for point in POINTS:
        values = [point[0] + offset, point[1] + offset, point[2] + offset]
        if gaussian:
            values += [0.1, 0.2, 0.3, 1.0, -2.0, -2.0, -2.0, 1.0, 0.0, 0.0, 0.0]
        rows.append(" ".join(str(value) for value in values))
    path.write_text("\n".join(header + rows) + "\n", encoding="ascii")


def _write_config(root: Path, **overrides: object) -> Path:
    raw: dict[str, object] = {
        "schema_version": "farm.scene.v1",
        "scene_id": "synthetic_scene",
        "inputs": {
            "colmap_model": "dataset/sparse/0",
            "image_root": "dataset/images",
            "gaussian_ply": "dataset/scene.ply",
        },
        "output_root": "results",
        "camera_grouping": {"mode": "single"},
        "metric_scale": {
            "required": True,
            "source": "declared_metric",
            "meters_per_colmap_unit": 1.0,
            "evidence": "synthetic fixture coordinates are metres",
        },
        "gravity": {"required": True, "mode": "axis", "axis": "-y"},
        "preflight": {
            "min_registered_images": 2,
            "min_sparse_points": 10,
            "min_gaussians": 10,
            "alignment": {"sample_points": 10, "nearest_neighbor_points": 10},
        },
    }
    raw.update(overrides)
    path = root / "scene.yaml"
    path.write_text(yaml.safe_dump(raw, sort_keys=False), encoding="utf-8")
    return path


def _fixture(
    tmp_path: Path,
    *,
    ply_offset: float = 0.0,
    gaussian: bool = True,
    **config_overrides: object,
) -> Path:
    dataset = tmp_path / "dataset"
    _model, _images = _write_colmap(dataset)
    _write_ply(dataset / "scene.ply", offset=ply_offset, gaussian=gaussian)
    return _write_config(tmp_path, **config_overrides)


def _check(report, name: str):
    return next(item for item in report.checks if item.name == name)


def test_relative_paths_and_defaults_are_canonical(tmp_path: Path) -> None:
    config_path = _fixture(tmp_path)
    config = load_scene_config(config_path)
    assert config.inputs.colmap_model == (tmp_path / "dataset/sparse/0").resolve()
    assert config.output_root == (tmp_path / "results").resolve()
    assert config.viewer.scene_point_size_m == pytest.approx(0.004)
    assert config.selection.target_timestamps == 160
    assert config.selection.max_translation_gap_m == pytest.approx(1.30)
    assert config.selection.max_rotation_gap_deg == pytest.approx(50.0)
    assert config.normalized_dict()["schema_version"] == "farm.scene.v1"


def test_selection_geometric_gap_thresholds_are_configurable(tmp_path: Path) -> None:
    config = load_scene_config(_fixture(tmp_path, selection={
        "target_timestamps": 10,
        "max_timestamps": 20,
        "max_motion_gap": 1.6,
        "max_translation_gap_m": 1.55,
        "max_rotation_gap_deg": 60.0,
    }))
    assert config.selection.max_motion_gap == pytest.approx(1.6)
    assert config.selection.max_translation_gap_m == pytest.approx(1.55)
    assert config.selection.max_rotation_gap_deg == pytest.approx(60.0)


@pytest.mark.parametrize("field", ["max_motion_gap", "max_translation_gap_m", "max_rotation_gap_deg"])
def test_selection_gap_thresholds_fail_closed(tmp_path: Path, field: str) -> None:
    with pytest.raises(SceneConfigError):
        load_scene_config(_fixture(tmp_path, selection={
            "target_timestamps": 10,
            "max_timestamps": 20,
            field: float("nan"),
        }))


def test_aligned_scene_passes_and_report_is_json_serializable(tmp_path: Path) -> None:
    report = run_preflight(load_scene_config(_fixture(tmp_path)))
    assert report.status == "pass", report.to_dict()
    assert report.strict_ready
    alignment = _check(report, "colmap_3dgs_alignment")
    assert alignment.metrics["robust_bbox_iou"] == pytest.approx(1.0)
    assert alignment.metrics["centroid_offset_ratio"] == pytest.approx(0.0)
    json.dumps(report.to_dict())


def test_missing_registered_image_fails(tmp_path: Path) -> None:
    config_path = _fixture(tmp_path)
    (tmp_path / "dataset/images/cam0_000002_center.png").unlink()
    report = run_preflight(load_scene_config(config_path))
    assert report.status == "fail"
    assert any(finding.code == "images.missing" for finding in _check(report, "images").findings)


def test_non_gaussian_ply_fails_schema_gate(tmp_path: Path) -> None:
    report = run_preflight(load_scene_config(_fixture(tmp_path, gaussian=False)))
    assert any(
        finding.code == "ply.schema.not_3dgs"
        for finding in _check(report, "gaussian_ply").findings
    )


def test_gross_colmap_ply_transform_fails_alignment(tmp_path: Path) -> None:
    report = run_preflight(load_scene_config(_fixture(tmp_path, ply_offset=100.0)))
    alignment = _check(report, "colmap_3dgs_alignment")
    assert alignment.status == "fail"
    assert any(finding.code == "alignment.spatial_mismatch" for finding in alignment.findings)


def test_required_unknown_metric_scale_fails(tmp_path: Path) -> None:
    config_path = _fixture(tmp_path)
    raw = yaml.safe_load(config_path.read_text())
    raw["metric_scale"] = {"required": True, "source": "unknown"}
    config_path.write_text(yaml.safe_dump(raw), encoding="utf-8")
    report = run_preflight(load_scene_config(config_path))
    assert _check(report, "metric_scale").status == "fail"


def test_invalid_regex_contract_is_rejected(tmp_path: Path) -> None:
    config_path = _fixture(tmp_path)
    raw = yaml.safe_load(config_path.read_text())
    raw["camera_grouping"] = {"mode": "regex", "pattern": "(?P<camera>cam[0-9]+)"}
    config_path.write_text(yaml.safe_dump(raw), encoding="utf-8")
    with pytest.raises(SceneConfigError, match="timestamp"):
        load_scene_config(config_path)


def test_large_file_fingerprint_is_explicitly_sampled(tmp_path: Path) -> None:
    path = tmp_path / "large.bin"
    path.write_bytes(bytes(range(64)))
    first = fingerprint_file(path, full_hash_max_bytes=8, chunk_bytes=4)
    second = fingerprint_file(path, full_hash_max_bytes=8, chunk_bytes=4)
    assert first["algorithm"] == "sha256-sampled-v1"
    assert first["digest"] == second["digest"]
    assert first["sampled_offsets"] == [0, 30, 60]


def test_release_critical_gaussian_ply_fingerprint_is_always_full(tmp_path: Path) -> None:
    config_path = _fixture(tmp_path)
    raw = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    raw["preflight"]["full_hash_max_bytes"] = 1
    raw["preflight"]["hash_chunk_bytes"] = 7
    config_path.write_text(yaml.safe_dump(raw, sort_keys=False), encoding="utf-8")
    config = load_scene_config(config_path)
    report = run_preflight(config)
    fingerprints = _check(report, "fingerprints").metrics["files"]
    by_path = {row["path"]: row for row in fingerprints}
    ply = by_path[str(config.inputs.gaussian_ply)]
    assert ply["algorithm"] == "sha256"
    assert ply["sampled_offsets"] == []
    generic = fingerprint_file(config.config_path, full_hash_max_bytes=1, chunk_bytes=7)
    assert generic["algorithm"] == "sha256-sampled-v1"


def test_json_schema_and_example_are_valid_json_yaml() -> None:
    root = Path(__file__).resolve().parents[1]
    schema = json.loads((root / "configs/schema/farm_scene.schema.json").read_text())
    example = yaml.safe_load((root / "configs/scenes/example_3dgs_colmap.yaml").read_text())
    assert schema["properties"]["schema_version"]["const"] == "farm.scene.v1"
    assert example["schema_version"] == "farm.scene.v1"
