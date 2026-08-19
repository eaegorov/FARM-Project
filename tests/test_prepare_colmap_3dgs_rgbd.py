import re
import sys
from dataclasses import replace
from pathlib import Path
from types import SimpleNamespace

import cv2
import numpy as np
import pytest


SCRIPTS = Path(__file__).resolve().parents[1] / "scripts"
if str(SCRIPTS) not in sys.path:
    sys.path.insert(0, str(SCRIPTS))

from prepare_colmap_3dgs_rgbd import (  # noqa: E402
    DEFAULT_IDENTITY_REGEX,
    View,
    appearance_alignment_metrics,
    atomic_write_jpeg,
    atomic_write_npy,
    baseline_report,
    camera_intrinsics,
    compute_baselines,
    distributed_indices,
    evaluate_qa,
    format_metric,
    metre_clip_to_scene_units,
    output_geometry,
    output_key,
    parse_args as parse_rgbd_args,
    parse_identity,
    pose_to_metres,
    read_selected_names,
    run as run_rgbd,
    scene_depth_to_metres,
    sparse_depth_alignment_metrics,
    validate_identities,
    validate_meters_per_scene_unit,
    write_dashboard,
    _model_files,
    _sparse_observations,
    _valid_pair,
)


class _Camera:
    def __init__(self, model_name, params):
        self.model_name = model_name
        self.params = np.asarray(params, dtype=np.float64)


def _view(name="cam00_000001_center.png", camera_id=1) -> View:
    pose = np.eye(4, dtype=np.float64)
    return View(
        image_id=1,
        camera_id=camera_id,
        name=name,
        width=80,
        height=40,
        fx=60.0,
        fy=62.0,
        cx=40.0,
        cy=20.0,
        world_to_camera=pose,
        camera_to_world=pose,
        sparse_xy=np.empty((0, 2), dtype=np.float64),
        sparse_depth_z=np.empty((0,), dtype=np.float64),
    )


def _qa_args():
    return SimpleNamespace(
        qa_min_views=3,
        qa_min_valid_depth_ratio=0.5,
        qa_min_sparse_samples=6,
        qa_max_sparse_median_relative_error=0.2,
        qa_min_luma_correlation=0.35,
        qa_min_edge_correlation=0.15,
        qa_max_affine_rgb_mae=0.2,
    )


def test_selected_names_are_exact_ordered_and_duplicates_fail(tmp_path):
    names = tmp_path / "selected_names.txt"
    names.write_text("nested/b.png\na.png\ncam 2.jpg\n", encoding="utf-8")
    assert read_selected_names(names) == ["nested/b.png", "a.png", "cam 2.jpg"]
    names.write_text("a.png\nb.png\na.png\n", encoding="utf-8")
    with pytest.raises(ValueError, match="duplicate exact names"):
        read_selected_names(names)


def test_pinhole_intrinsics_identity_and_output_geometry():
    assert camera_intrinsics(_Camera("PINHOLE", [10, 11, 4, 5])) == (10, 11, 4, 5)
    assert camera_intrinsics(_Camera("SIMPLE_PINHOLE", [10, 4, 5])) == (10, 10, 4, 5)
    with pytest.raises(ValueError, match="Convert fisheye"):
        camera_intrinsics(_Camera("OPENCV_FISHEYE", [10, 10, 4, 5, 0, 0, 0, 0]))

    width, height, K = output_geometry(_view(), 40)
    assert (width, height) == (40, 16)
    assert np.allclose(K, [[30, 0, 20], [0, 24.8, 8], [0, 0, 1]])
    pattern = re.compile(DEFAULT_IDENTITY_REGEX, re.IGNORECASE)
    identity = parse_identity("cam00_000001_center.png", 7, pattern)
    assert (identity.sensor, identity.timestamp, identity.family, identity.camera) == (
        "cam00", "000001", "center", "cam00_center"
    )
    fallback = parse_identity("nested/arbitrary name.jpg", 7, pattern)
    assert fallback.camera == "camera_7" and fallback.timestamp == "arbitrary name"
    with pytest.raises(ValueError, match="same \\(timestamp, camera\\)"):
        validate_identities([_view("a/x.jpg"), _view("b/x.jpg")], [fallback, fallback])

    custom = re.compile(
        r"^(?P<member>[^-]+)-(?P<stamp>\d+)-(?P<view>[^.]+)\.png$"
    )
    custom_identity = parse_identity(
        "left-0042-forward.png", 3, custom, "member", "stamp", "view"
    )
    assert custom_identity == type(custom_identity)("left", "0042", "forward")
    with pytest.raises(ValueError, match="timestamp group"):
        parse_identity("left-0042-forward.png", 3, custom, "member", "missing", "view")


def test_metric_scale_converts_depth_pose_clip_sparse_and_baseline():
    assert validate_meters_per_scene_unit(0.1) == pytest.approx(0.1)
    raw_depth = np.asarray([[0.0, 10.0, 20.0]], dtype=np.float32)
    assert np.allclose(scene_depth_to_metres(raw_depth, 0.1), [[0.0, 1.0, 2.0]])
    assert metre_clip_to_scene_units(0.05, 80.0, 0.1) == pytest.approx((0.5, 800.0))

    pose = np.eye(4, dtype=np.float64)
    pose[:3, 3] = [10.0, 20.0, 30.0]
    metric_pose = pose_to_metres(pose, 0.1)
    assert np.allclose(metric_pose[:3, :3], np.eye(3))
    assert np.allclose(metric_pose[:3, 3], [1.0, 2.0, 3.0])
    assert np.allclose(pose[:3, 3], [10.0, 20.0, 30.0])  # input is not mutated

    depth = np.zeros((20, 40), dtype=np.float32)
    depth[4:7, 9:12] = 1.0
    sparse = sparse_depth_alignment_metrics(
        depth,
        np.asarray([[20.0, 10.0]]),
        scene_depth_to_metres(np.asarray([10.0]), 0.1),
        source_width=80,
        source_height=40,
    )
    assert sparse["median_relative_error"] == pytest.approx(0.0)

    second_pose = np.eye(4, dtype=np.float64)
    second_pose[0, 3] = 0.5
    views = [_view("left_0001_center.png", 1), replace(_view("right_0001_center.png", 1), camera_to_world=second_pose)]
    pattern = re.compile(DEFAULT_IDENTITY_REGEX)
    identities = [parse_identity(view.name, view.camera_id, pattern) for view in views]
    raw_baselines = compute_baselines(views, identities, sensors=("left", "right"))
    assert raw_baselines == pytest.approx([0.5])
    report = baseline_report(raw_baselines, 0.1, expected_m=0.05, tolerance_m=1e-8)
    assert report["passed"] is True
    assert report["scene_units"]["median"] == pytest.approx(0.5)
    assert report["metres"]["median"] == pytest.approx(0.05)
    assert baseline_report(raw_baselines, 0.1, 0.06, 0.001)["passed"] is False


@pytest.mark.parametrize("invalid", ["0", "-0.1", "nan", "inf"])
def test_invalid_metric_scale_fails_before_output_mutation(tmp_path, invalid):
    output = tmp_path / f"rgbd-{invalid}"
    args = parse_rgbd_args(
        [
            "--colmap-model", str(tmp_path / "missing-colmap"),
            "--image-root", str(tmp_path / "missing-images"),
            "--ply", str(tmp_path / "missing.ply"),
            "--selected-names", str(tmp_path / "missing-names.txt"),
            "--output-dir", str(output),
            "--scene-id", "test",
            "--meters-per-scene-unit", invalid,
        ]
    )
    with pytest.raises(ValueError, match="finite and strictly positive"):
        run_rgbd(args)
    assert not output.exists()


def test_sparse_depth_alignment_uses_scaled_colmap_coordinates():
    depth = np.zeros((20, 40), dtype=np.float32)
    xy = np.asarray([[20, 10], [40, 20], [60, 30]], dtype=np.float64)
    z = np.asarray([2.0, 4.0, 8.0], dtype=np.float64)
    for point, value in zip(xy, z, strict=True):
        x, y = int(round(point[0] * 0.5)), int(round(point[1] * 0.5))
        depth[max(0, y - 1):y + 2, max(0, x - 1):x + 2] = value
    metrics = sparse_depth_alignment_metrics(depth, xy, z, source_width=80, source_height=40)
    assert metrics["sample_count"] == 3
    assert metrics["median_relative_error"] == pytest.approx(0.0)
    shifted = sparse_depth_alignment_metrics(depth * 1.5, xy, z, source_width=80, source_height=40)
    assert shifted["median_relative_error"] == pytest.approx(0.5)


def test_appearance_metrics_detect_identical_and_misaligned_images():
    yy, xx = np.mgrid[0:64, 0:96]
    source_rgb = np.stack((xx * 2, yy * 3, (xx + yy) * 1.2), axis=-1).clip(0, 255).astype(np.uint8)
    source_bgr = cv2.cvtColor(source_rgb, cv2.COLOR_RGB2BGR)
    rendered = source_rgb.astype(np.float32) / 255.0
    mask = np.ones((64, 96), dtype=bool)
    identical = appearance_alignment_metrics(source_bgr, rendered, mask)
    assert identical["luma_correlation"] == pytest.approx(1.0, abs=1e-5)
    assert identical["edge_correlation"] == pytest.approx(1.0, abs=1e-5)
    assert identical["affine_rgb_mae"] < 1e-5
    wrong = appearance_alignment_metrics(source_bgr, np.flip(rendered, axis=1).copy(), mask)
    assert wrong["edge_correlation"] < identical["edge_correlation"]
    assert wrong["affine_rgb_mae"] > identical["affine_rgb_mae"]


def test_qa_is_fail_closed_and_dashboard_handles_missing_metrics(tmp_path):
    missing = [
        {
            "selected_index": 0,
            "valid_depth_ratio": 0.0,
            "sparse_depth": {"sample_count": 0, "median_relative_error": None},
            "appearance": {"luma_correlation": None, "edge_correlation": None, "affine_rgb_mae": None},
        }
    ]
    result = evaluate_qa(missing, _qa_args())
    assert result["passed"] is False
    assert len(result["failure_reasons"]) >= 5
    assert format_metric(None) == "n/a"
    dashboard = tmp_path / "dashboard.jpg"
    write_dashboard([_view()], missing, missing, result, "synthetic", dashboard)
    assert dashboard.is_file() and dashboard.stat().st_size > 10_000
    image = cv2.imread(str(dashboard), cv2.IMREAD_COLOR)
    assert image is not None and image.shape[:2] == (2160, 3840)


def test_qa_passes_good_evidence_and_atomic_pair_validation(tmp_path):
    evidence = []
    for index in range(3):
        evidence.append(
            {
                "selected_index": index,
                "valid_depth_ratio": 0.95,
                "sparse_depth": {"sample_count": 4, "median_relative_error": 0.03},
                "appearance": {"luma_correlation": 0.9, "edge_correlation": 0.8, "affine_rgb_mae": 0.04},
            }
        )
    assert evaluate_qa(evidence, _qa_args())["passed"] is True
    assert distributed_indices(631, 12)[0] == 0 and distributed_indices(631, 12)[-1] == 630
    assert output_key(0, "a/x.png") != output_key(0, "b/x.png")

    rgb = tmp_path / "rgb.jpg"
    depth = tmp_path / "depth.npy"
    atomic_write_jpeg(rgb, np.zeros((8, 16, 3), dtype=np.uint8), 90)
    atomic_write_npy(depth, np.ones((8, 16), dtype=np.float32))
    assert _valid_pair(rgb, depth, 16, 8)
    atomic_write_npy(depth, np.ones((7, 16), dtype=np.float32))
    assert not _valid_pair(rgb, depth, 16, 8)


def test_dual_colmap_formats_and_pycolmap_indexed_point_map(tmp_path):
    for stem in ("cameras", "images", "points3D"):
        (tmp_path / f"{stem}.bin").write_bytes(b"binary")
        (tmp_path / f"{stem}.txt").write_text("text", encoding="utf-8")
    files = _model_files(tmp_path)
    assert len(files) == 6
    assert {path.suffix for path in files} == {".bin", ".txt"}

    class IndexedPoints:
        def __getitem__(self, point_id):
            if point_id != 9:
                raise KeyError(point_id)
            return SimpleNamespace(xyz=np.asarray([0.0, 0.0, 3.0]))

    point2d = SimpleNamespace(
        has_point3D=lambda: True,
        point3D_id=9,
        xy=np.asarray([12.0, 14.0]),
    )
    image = SimpleNamespace(points2D=[point2d])
    reconstruction = SimpleNamespace(points3D=IndexedPoints())
    xys, depths = _sparse_observations(image, reconstruction, np.eye(4))
    assert np.allclose(xys, [[12.0, 14.0]])
    assert np.allclose(depths, [3.0])
