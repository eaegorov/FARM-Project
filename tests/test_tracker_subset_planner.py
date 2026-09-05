from __future__ import annotations

import json
import math
from decimal import Decimal

import numpy as np
import pytest

from farm_runtime.colmap_pose_reader import read_colmap_cameras_and_poses
from farm_runtime.tracker_subset_planner import (
    ContinuityRegion,
    ImageIdentity,
    TrackerFrame,
    assign_global_timestamp_folds,
    build_continuity_regions,
    camera_motion,
    compile_identity_pattern,
    episodes_to_manifest_rows,
    load_anchor_names,
    metric_world_from_camera,
    parse_image_identity,
    select_balanced_episodes,
    validate_episode_rows,
)

PATTERN = r"(?P<camera>cam\d+)_(?P<timestamp>\d+)_(?P<family>[a-z_]+)\.png"


def _pose(x_m: float = 0.0, yaw_degrees: float = 0.0) -> np.ndarray:
    angle = math.radians(yaw_degrees)
    cosine, sine = math.cos(angle), math.sin(angle)
    pose = np.eye(4, dtype=np.float64)
    pose[:3, :3] = np.asarray(
        [[cosine, -sine, 0.0], [sine, cosine, 0.0], [0.0, 0.0, 1.0]]
    )
    pose[0, 3] = x_m
    return pose


def _frame(
    camera: str,
    family: str,
    timestamp: int,
    *,
    x_m: float = 0.0,
    yaw_degrees: float = 0.0,
) -> TrackerFrame:
    name = f"{camera}_{timestamp:06d}_{family}.png"
    return TrackerFrame(
        name=name,
        colmap_image_id=timestamp + 1,
        colmap_camera_id=1,
        identity=ImageIdentity(
            camera=camera,
            view_family=family,
            physical_timestamp=f"{timestamp:06d}",
            timestamp_value=Decimal(timestamp),
        ),
        world_from_camera_m=_pose(x_m=x_m, yaw_degrees=yaw_degrees),
    )


def _region(
    camera: str,
    family: str,
    split: str,
    start: int,
    count: int,
) -> ContinuityRegion:
    return ContinuityRegion(
        camera=camera,
        view_family=family,
        split=split,
        frames=tuple(
            _frame(camera, family, value) for value in range(start, start + count)
        ),
        lower_quartile_gap=1.0,
        maximum_gap=2.5,
    )


def test_identity_regex_is_named_numeric_and_full_match() -> None:
    pattern = compile_identity_pattern(PATTERN)
    identity = parse_image_identity("cam17_001234_yaw_left.png", pattern)
    assert identity.camera == "cam17"
    assert identity.physical_timestamp == "001234"
    assert identity.timestamp_value == Decimal(1234)
    assert identity.view_family == "yaw_left"
    with pytest.raises(ValueError, match="does not satisfy"):
        parse_image_identity("prefix/cam17_001234_yaw_left.png", pattern)
    with pytest.raises(ValueError, match="missing named group"):
        compile_identity_pattern(r"(?P<camera>cam\d+)_(?P<timestamp>\d+)")
    nonnumeric = compile_identity_pattern(
        r"(?P<camera>cam\d+)_(?P<timestamp>[a-z]+)_(?P<family>[a-z]+)\.png"
    )
    with pytest.raises(ValueError, match="must be numeric"):
        parse_image_identity("cam1_later_center.png", nonnumeric)


def test_metric_pose_inversion_and_motion_are_correct() -> None:
    camera_from_world = np.asarray(
        [[1.0, 0.0, 0.0, -2.0], [0.0, 1.0, 0.0, 0.0], [0.0, 0.0, 1.0, 0.0]]
    )
    world_from_camera = metric_world_from_camera(
        camera_from_world, meters_per_scene_unit=0.25
    )
    assert world_from_camera[:3, 3].tolist() == pytest.approx([0.5, 0.0, 0.0])
    translation, rotation = camera_motion(_pose(), _pose(x_m=0.6, yaw_degrees=22.0))
    assert translation == pytest.approx(0.6)
    assert rotation == pytest.approx(22.0)


def test_global_fold_blocks_are_contiguous_deterministic_and_disjoint() -> None:
    timestamps = [Decimal(value) for value in range(96)]
    first, blocks = assign_global_timestamp_folds(
        timestamps,
        heldout_fraction=0.25,
        block_size=16,
        split_seed="scene-a",
    )
    second, _ = assign_global_timestamp_folds(
        reversed(timestamps),
        heldout_fraction=0.25,
        block_size=16,
        split_seed="scene-a",
    )
    assert first == second
    assert {row["split"] for row in blocks} == {"train", "heldout"}
    for start in range(0, 96, 16):
        assert len({first[Decimal(value)] for value in range(start, start + 16)}) == 1


def test_continuity_regions_split_on_gap_translation_rotation_and_fold() -> None:
    frames = [
        _frame("cam0", "center", 0, x_m=0.0),
        _frame("cam0", "center", 1, x_m=0.1),
        _frame("cam0", "center", 2, x_m=0.2),
        _frame("cam0", "center", 3, x_m=0.3),
        _frame("cam0", "center", 7, x_m=0.4),
        _frame("cam0", "center", 8, x_m=1.0),
        _frame("cam0", "center", 9, x_m=1.1, yaw_degrees=22.0),
        _frame("cam1", "yaw_left", 0),
        _frame("cam1", "yaw_left", 1),
    ]
    folds = {Decimal(value): "train" for value in range(10)}
    folds[Decimal(1)] = "heldout"
    regions, diagnostics = build_continuity_regions(frames, folds)
    assert all(
        len({frame.stream_key for frame in region.frames}) == 1 for region in regions
    )
    assert all(
        len({folds[frame.identity.timestamp_value] for frame in region.frames}) == 1
        for region in regions
    )
    assert diagnostics["break_counts"]["global_fold_boundary"] >= 1
    assert diagnostics["break_counts"]["timestamp_gap"] == 1
    assert diagnostics["break_counts"]["camera_translation"] == 1
    assert diagnostics["break_counts"]["camera_rotation"] == 1


def test_duplicate_timestamp_in_exact_stream_is_rejected() -> None:
    duplicate = [_frame("cam0", "center", 1), _frame("cam0", "center", 1)]
    with pytest.raises(ValueError, match="duplicate physical timestamp"):
        build_continuity_regions(duplicate, {Decimal(1): "train"})


def test_balanced_selection_hits_budget_and_anchor_without_sparse_mixing() -> None:
    regions = [
        _region(camera, family, split, offset, 30)
        for camera, family in (("cam0", "center"), ("cam1", "yaw_left"))
        for split, offset in (("train", 0), ("heldout", 100))
    ]
    anchor = "cam1_000125_yaw_left.png"
    selected, diagnostics = select_balanced_episodes(
        regions,
        anchor_names={anchor},
        target_frames=40,
        minimum_total_frames=32,
        maximum_total_frames=48,
        minimum_episode_frames=8,
        maximum_episode_frames=16,
    )
    assert diagnostics["selected_frames"] == 40
    assert diagnostics["frames_by_split"] == {"heldout": 20, "train": 20}
    assert set(diagnostics["frames_by_exact_stream"].values()) == {20}
    assert anchor in {frame.name for region in selected for frame in region.frames}
    assert all(8 <= len(region.frames) <= 16 for region in selected)
    for region in selected:
        values = [int(frame.identity.timestamp_value) for frame in region.frames]
        assert values == list(range(values[0], values[0] + len(values)))


def test_v10_v11_anchor_shapes_are_loaded_as_soft_names(tmp_path) -> None:
    path = tmp_path / "plan.json"
    path.write_text(
        json.dumps(
            {
                "schema": "farm.full-colmap-rescue-plan.v2",
                "selected_names": ["cam0_1_center.png"],
                "objects": [
                    {
                        "selected_views": [
                            {"name": "cam1_2_yaw_left.png"},
                            {"source_image": "cam1_3_yaw_left.png"},
                        ]
                    }
                ],
            }
        )
    )
    names, provenance = load_anchor_names(path)
    assert names == {
        "cam0_1_center.png",
        "cam1_2_yaw_left.png",
        "cam1_3_yaw_left.png",
    }
    assert provenance["provided"] is True
    assert len(str(provenance["sha256"])) == 64


def test_serialized_episode_validation_rejects_global_split_leakage() -> None:
    episodes = episodes_to_manifest_rows(
        [
            _region("cam0", "center", "train", 0, 8),
            _region("cam1", "center", "heldout", 0, 8),
        ],
        set(),
    )
    with pytest.raises(ValueError, match="timestamp leakage"):
        validate_episode_rows(
            episodes,
            minimum_episode_frames=8,
            maximum_episode_frames=32,
            maximum_translation_m=0.5,
            maximum_rotation_degrees=21.0,
        )


def test_text_colmap_reader_streams_only_cameras_and_poses(tmp_path) -> None:
    (tmp_path / "cameras.txt").write_text(
        "# cameras\n1 PINHOLE 640 480 500 500 320 240\n"
    )
    (tmp_path / "images.txt").write_text(
        "# images\n"
        "7 1 0 0 0 0 0 0 1 cam0_000001_center.png\n"
        "10.0 20.0 -1\n"
        "8 1 0 0 0 -1 0 0 1 cam0_000002_center.png\n"
        "\n"
    )
    source_format, cameras, images, files = read_colmap_cameras_and_poses(tmp_path)
    assert source_format == "text"
    assert cameras[1].params == (500.0, 500.0, 320.0, 240.0)
    assert [image.name for image in images] == [
        "cam0_000001_center.png",
        "cam0_000002_center.png",
    ]
    assert images[1].camera_from_world[:, 3].tolist() == [-1.0, 0.0, 0.0]
    assert [path.name for path in files] == ["cameras.txt", "images.txt"]
