import json
import sys
from pathlib import Path

import cv2
import numpy as np


SCRIPTS = Path(__file__).resolve().parents[1] / "scripts"
if str(SCRIPTS) not in sys.path:
    sys.path.insert(0, str(SCRIPTS))

from convert_colmap_kb4_to_pinhole import (  # noqa: E402
    ColmapModel,
    PointTable,
    RegisteredImage,
    ViewSpec,
    build_virtual_geometry,
    build_kb4_remap,
    camera_center,
    main,
    pinhole_intrinsics,
    qvec_to_rotmat,
    read_colmap_model,
    virtual_pose,
    virtual_rotation,
    Camera,
)


def _write_synthetic_source(root: Path) -> tuple[Path, Path]:
    sparse = root / "sparse" / "0"
    images = root / "images"
    sparse.mkdir(parents=True)
    (images / "00").mkdir(parents=True)
    (images / "10").mkdir(parents=True)
    cameras = (
        "# synthetic KB4 rig\n"
        "1 KANNALABRANDT4 64 64 18 18 32 32 0.01 0.001 0 0\n"
        "2 KANNALABRANDT4 64 64 18.2 18.1 32.2 31.8 0.012 0.001 0 0\n"
    )
    (sparse / "cameras.txt").write_text(cameras, encoding="utf-8")
    rows = []
    image_rows = [
        (1, 1, "00/000001.png", [0.0, 0.0, 0.0]),
        (2, 2, "10/000001.png", [-0.10, 0.0, 0.0]),
        (3, 1, "00/000002.png", [0.0, -0.03, 0.0]),
        (4, 2, "10/000002.png", [-0.10, -0.03, 0.0]),
    ]
    for image_id, camera_id, name, tvec in image_rows:
        rows.append(
            f"{image_id} 1 0 0 0 {tvec[0]} {tvec[1]} {tvec[2]} {camera_id} {name}\n"
            "30 32 10 32 32 11 34 32 12\n"
        )
        yy, xx = np.mgrid[0:64, 0:64]
        rgb = np.stack((xx * 4, yy * 4, np.full_like(xx, image_id * 40)), axis=-1).astype(np.uint8)
        assert cv2.imwrite(str(images / name), rgb)
    (sparse / "images.txt").write_text("# synthetic images\n" + "".join(rows), encoding="utf-8")
    points = [
        "10 -0.2 0 3 255 0 0 0.2 1 0 2 0 3 0 4 0",
        "11 0 0 3 0 255 0 0.2 1 1 2 1 3 1 4 1",
        "12 0.2 0 3 0 0 255 0.2 1 2 2 2 3 2 4 2",
    ]
    (sparse / "points3D.txt").write_text("# synthetic points\n" + "\n".join(points) + "\n", encoding="utf-8")
    return sparse, images


def test_virtual_axes_and_pose_match_factory_reference():
    qvec = np.asarray([0.8997061725064306, 0.2997058112691261,
                       -0.018112827772026253, 0.31682353971169])
    tvec = np.asarray([-0.22673542020595622, -2.365081046821031, 2.4962242597038076])
    expected = {
        "yaw_left": (
            np.asarray([0.8230655219757400, 0.4055211242048354, 0.36381646184700445, 0.16047849216507573]),
            np.asarray([1.7664760041394267, -2.365081046821031, 1.778231433843581]),
        ),
        "yaw_right": (
            np.asarray([0.8077558983992952, 0.13773029694185568, -0.3966480555570867, 0.4138007901493642]),
            np.asarray([-2.0579614417103462, -2.365081046821031, 1.4308526164295425]),
        ),
        "pitch_up": (
            np.asarray([0.9420718591796617, -0.10860654812870001, 0.11747961677644878, 0.2947944529454423]),
            np.asarray([-0.22673542020595622, 0.39197393012385767, 3.416299218579828]),
        ),
        "pitch_down": (
            np.asarray([0.6887495611953733, 0.6518579692753912, -0.15031121048653098, 0.27948482936899754]),
            np.asarray([-0.22673542020595622, -3.432463515725915, -0.20721516830670464]),
        ),
    }
    views = {
        "yaw_left": ViewSpec("yaw_left", -50, 0),
        "yaw_right": ViewSpec("yaw_right", 50, 0),
        "pitch_up": ViewSpec("pitch_up", 0, 50),
        "pitch_down": ViewSpec("pitch_down", 0, -50),
    }
    original_center = camera_center(qvec, tvec)
    for name, view in views.items():
        converted_q, converted_t = virtual_pose(qvec, tvec, virtual_rotation(view))
        expected_q, expected_t = expected[name]
        assert np.allclose(converted_q, expected_q, atol=1e-12)
        assert np.allclose(converted_t, expected_t, atol=1e-12)
        assert np.allclose(camera_center(converted_q, converted_t), original_center, atol=1e-12)
        rotation = qvec_to_rotmat(converted_q)
        assert np.allclose(rotation.T @ rotation, np.eye(3), atol=1e-12)
        assert np.isclose(np.linalg.det(rotation), 1.0, atol=1e-12)


def test_kb4_remap_has_expected_axis_and_coverage():
    camera = Camera(1, "KANNALABRANDT4", 128, 128,
                    np.asarray([36, 36, 64, 64, 0.01, 0.001, 0, 0], dtype=np.float64))
    intrinsics = pinhole_intrinsics(64, 64, 80, 80)
    map_x, map_y, stats = build_kb4_remap(camera, intrinsics, 64, 64, ViewSpec("right", 35, 0))
    assert map_x.shape == (64, 64) and map_y.shape == (64, 64)
    assert stats["valid_ratio"] > 0.99
    expected_axis = np.asarray([np.sin(np.deg2rad(35)), 0, np.cos(np.deg2rad(35))])
    assert np.allclose([stats["axis_x"], stats["axis_y"], stats["axis_z"]], expected_axis)


def test_kb4_remap_uses_colmap_half_pixel_centers():
    camera = Camera(
        1,
        "KANNALABRANDT4",
        128,
        128,
        np.asarray([36, 36, 64, 64, 0.01, 0.001, 0, 0], dtype=np.float64),
    )
    intrinsics = pinhole_intrinsics(64, 64, 80, 80)
    map_x, map_y, stats = build_kb4_remap(
        camera, intrinsics, 64, 64, ViewSpec("center", 0, 0)
    )

    # COLMAP's origin is the image corner, so the four middle pixel centers of
    # an even-sized image surround the optical axis symmetrically at +/-0.5 px.
    middle = np.s_[31:33, 31:33]
    assert np.isclose(float(map_x[middle].mean()), 63.5, atol=1e-6)
    assert np.isclose(float(map_y[middle].mean()), 63.5, atol=1e-6)
    assert np.isclose(float(map_x[31, 31] + map_x[32, 32]), 127.0, atol=1e-5)
    assert np.isclose(float(map_y[31, 31] + map_y[32, 32]), 127.0, atol=1e-5)

    fx, fy, cx, cy = (
        float(intrinsics[0, 0]),
        float(intrinsics[1, 1]),
        float(intrinsics[0, 2]),
        float(intrinsics[1, 2]),
    )
    corner_x = (0.5 - cx) / fx
    corner_y = (0.5 - cy) / fy
    expected_theta_max = np.degrees(np.arctan2(np.hypot(corner_x, corner_y), 1.0))
    assert np.isclose(stats["theta_max_deg"], expected_theta_max, atol=1e-10)


def test_kb4_remap_converts_colmap_coordinates_to_opencv_indices():
    width = height = 32
    source_width = source_height = 64
    focal = 10_000.0
    # tan(theta) series makes this KB4 camera numerically pinhole over the tiny
    # test FOV, while still exercising the real KB4 projection and cv2.remap.
    camera = Camera(
        1,
        "KANNALABRANDT4",
        source_width,
        source_height,
        np.asarray(
            [
                focal,
                focal,
                source_width / 2.0,
                source_height / 2.0,
                1.0 / 3.0,
                2.0 / 15.0,
                17.0 / 315.0,
                62.0 / 2835.0,
            ],
            dtype=np.float64,
        ),
    )
    intrinsics = np.asarray(
        [
            [focal, 0.0, width / 2.0],
            [0.0, focal, height / 2.0],
            [0.0, 0.0, 1.0],
        ],
        dtype=np.float64,
    )
    map_x, map_y, stats = build_kb4_remap(
        camera, intrinsics, width, height, ViewSpec("center", 0, 0)
    )
    source = np.arange(
        source_width * source_height, dtype=np.uint16
    ).reshape(source_height, source_width)
    remapped = cv2.remap(
        source,
        map_x,
        map_y,
        cv2.INTER_LINEAR,
        borderMode=cv2.BORDER_CONSTANT,
        borderValue=0,
    )

    assert stats["valid_ratio"] == 1.0
    offset_x = (source_width - width) // 2
    offset_y = (source_height - height) // 2
    assert np.allclose(map_x, np.arange(width)[None, :] + offset_x, atol=1e-6)
    assert np.allclose(map_y, np.arange(height)[:, None] + offset_y, atol=1e-6)
    assert np.array_equal(
        remapped,
        source[offset_y : offset_y + height, offset_x : offset_x + width],
    )


def test_virtual_geometry_keeps_full_colmap_continuous_image_domain():
    point_ids = np.asarray([10, 11, 12, 13], dtype=np.int64)
    points = PointTable(
        ids=point_ids,
        xyz=np.asarray(
            [
                [-2.0, 0.0, 1.0],   # u = 0: left image boundary, included
                [1.5, 0.0, 1.0],    # u = 3.5: rightmost pixel center, included
                [1.999, 0.0, 1.0],  # u = 3.999: inside the right half-pixel strip
                [2.0, 0.0, 1.0],    # u = 4: right image boundary, excluded
            ],
            dtype=np.float64,
        ),
        rgb=np.zeros((4, 3), dtype=np.uint8),
        error=np.zeros(4, dtype=np.float64),
    )
    source = RegisteredImage(
        image_id=1,
        qvec=np.asarray([1.0, 0.0, 0.0, 0.0]),
        tvec=np.zeros(3, dtype=np.float64),
        camera_id=1,
        name="00/000001.png",
        xys=np.zeros((4, 2), dtype=np.float64),
        point3d_ids=point_ids,
    )
    model = ColmapModel(
        cameras={
            1: Camera(
                1,
                "KANNALABRANDT4",
                4,
                4,
                np.asarray([1, 1, 2, 2, 0, 0, 0, 0], dtype=np.float64),
            )
        },
        images={1: source},
        points=points,
        source_format="text",
        source_files=(),
    )
    intrinsics = np.asarray(
        [[1.0, 0.0, 2.0], [0.0, 1.0, 2.0], [0.0, 0.0, 1.0]], dtype=np.float64
    )
    images, tracks, _ = build_virtual_geometry(
        model,
        [ViewSpec("center", 0, 0)],
        intrinsics,
        4,
        4,
        {(1, "center"): "cam00_000001_center.png"},
        min_source_observations=1,
    )

    assert images[0].point_indices.tolist() == [0, 1, 2]
    assert np.allclose(images[0].xys[:, 0], [0.0, 3.5, 3.999])
    assert tracks.kept_point_indices.tolist() == [0, 1, 2]
    assert tracks.image_ids.tolist() == [1, 1, 1]
    assert tracks.point2d_indices.tolist() == [0, 1, 2]


def test_end_to_end_writes_standard_text_and_binary_and_is_idempotent(tmp_path):
    sparse, images = _write_synthetic_source(tmp_path / "source")
    output = tmp_path / "converted"
    args = [
        "--input-model", str(sparse), "--input-images", str(images),
        "--output", str(output), "--width", "32", "--height", "32",
        "--fov-x-deg", "70", "--fov-y-deg", "70", "--side-angle-deg", "25",
        "--min-valid-map-ratio", "0.95", "--workers", "1",
    ]
    assert main(args) == 0
    manifest = json.loads((output / "conversion_manifest.json").read_text(encoding="utf-8"))
    assert manifest["status"] == "complete"
    assert manifest["geometry"]["virtual_images"] == 20
    assert manifest["geometry"]["retained_points3D"] == 3
    assert manifest["geometry_validation"]["reprojection_max_px"] < 1e-9
    assert (output / "conversion_qc_contact_sheet.jpg").is_file()
    for fmt in ("text", "binary"):
        reconstructed = read_colmap_model(output / "sparse" / "0", fmt)
        assert len(reconstructed.cameras) == 1
        assert next(iter(reconstructed.cameras.values())).model == "PINHOLE"
        assert len(reconstructed.images) == 20
        assert reconstructed.points.ids.tolist() == [10, 11, 12]
        assert sorted(image.name for image in reconstructed.images.values())[0].startswith("cam00_")
    image_mtime = (output / "images" / "cam00_000001_center.png").stat().st_mtime_ns
    assert main(args) == 0
    assert (output / "images" / "cam00_000001_center.png").stat().st_mtime_ns == image_mtime


def test_preflight_fails_closed_on_incomplete_two_lens_rig(tmp_path):
    sparse, images = _write_synthetic_source(tmp_path / "source")
    (images / "10" / "000002.png").unlink()
    output = tmp_path / "converted"
    assert main([
        "--input-model", str(sparse), "--input-images", str(images), "--output", str(output),
        "--require-complete-rig", "--preflight-only", "--width", "32", "--height", "32",
    ]) == 2
    assert not output.exists()
