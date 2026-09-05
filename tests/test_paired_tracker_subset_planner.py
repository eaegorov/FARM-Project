from __future__ import annotations

import copy
import math
from decimal import Decimal

import numpy as np
import pytest

from farm_runtime.paired_tracker_subset_planner import (
    SupportEvidence,
    enumerate_object_visible_windows,
    pair_object_windows,
    select_paired_object_episodes,
    selected_pairs_to_episode_rows,
    validate_paired_episode_rows,
)
from farm_runtime.tracker_subset_planner import ImageIdentity, TrackerFrame


def _pose(x: float, yaw_degrees: float = 0.0) -> np.ndarray:
    angle = math.radians(yaw_degrees)
    pose = np.eye(4, dtype=np.float64)
    pose[:3, :3] = np.asarray(
        [
            [math.cos(angle), -math.sin(angle), 0.0],
            [math.sin(angle), math.cos(angle), 0.0],
            [0.0, 0.0, 1.0],
        ]
    )
    pose[0, 3] = x
    return pose


def _frame(camera: str, family: str, timestamp: int) -> TrackerFrame:
    return TrackerFrame(
        name=f"{camera}_{timestamp:06d}_{family}.png",
        colmap_image_id=1000 * (int(camera[-1]) + 1) + timestamp,
        colmap_camera_id=int(camera[-1]) + 1,
        identity=ImageIdentity(
            camera=camera,
            view_family=family,
            physical_timestamp=f"{timestamp:06d}",
            timestamp_value=Decimal(timestamp),
        ),
        world_from_camera_m=_pose(0.02 * timestamp, 0.25 * timestamp),
    )


def _evidence(object_id: int, frame: TrackerFrame, *, rescue: bool = False) -> SupportEvidence:
    return SupportEvidence(
        object_id=object_id,
        frame=frame,
        track_support_count=8,
        track_support_fraction=0.04,
        projected_support_points=64,
        in_frame_support_ratio=0.92,
        area_ratio=0.04,
        margin_ratio=0.08,
        geometric_score=0.24,
        view_direction=(0.01 * int(frame.identity.timestamp_value), 1.0, 0.1),
        rescue_anchor=rescue,
    )


def test_windows_never_bridge_registered_frame_without_evidence() -> None:
    registered = [_frame("cam0", "center", value) for value in range(20)]
    evidence = [
        _evidence(7, frame, rescue=frame is registered[4])
        for index, frame in enumerate(registered)
        if index != 8
    ]
    windows, diagnostics = enumerate_object_visible_windows(
        registered,
        evidence,
        minimum_frames=8,
        maximum_frames=10,
        preferred_frames=9,
    )
    assert windows
    assert diagnostics["rejection_break_counts"][
        "frame_lacks_object_support_track_projection"
    ] == 1
    for window in windows:
        indices = [int(row.frame.identity.timestamp_value) for row in window.frames]
        assert indices == list(range(indices[0], indices[0] + len(indices)))
        assert 8 not in indices
        assert window.registered_stream_end - window.registered_stream_start + 1 == len(
            window.frames
        )
        assert all(row.object_id == 7 for row in window.frames)


def test_pairing_rejects_same_physical_timestamps_across_cameras() -> None:
    first = [_frame("cam0", "center", value) for value in range(12)]
    second = [_frame("cam1", "center", value) for value in range(12)]
    registered = first + second
    evidence = [_evidence(3, frame) for frame in registered]
    windows, _ = enumerate_object_visible_windows(
        registered,
        evidence,
        minimum_frames=8,
        maximum_frames=8,
        preferred_frames=8,
    )
    pairs = pair_object_windows(windows)
    # Every possible eight-frame pair overlaps in physical time here. Different
    # RGB names/cameras do not weaken the global split-isolation contract.
    assert pairs == []


def test_selection_produces_one_train_and_heldout_episode_per_object() -> None:
    registered: list[TrackerFrame] = []
    pairs_by_object = {}
    for object_id, camera, offset in ((1, "cam0", 0), (2, "cam1", 100)):
        frames = [_frame(camera, "center", offset + value) for value in range(28)]
        registered.extend(frames)
        evidence = [_evidence(object_id, frame) for frame in frames]
        windows, _ = enumerate_object_visible_windows(
            registered,
            evidence,
            minimum_frames=8,
            maximum_frames=8,
            preferred_frames=8,
        )
        pairs_by_object[object_id] = pair_object_windows(windows)
    selected, diagnostics = select_paired_object_episodes(
        pairs_by_object,
        target_frames=32,
        minimum_total_frames=32,
        maximum_total_frames=40,
    )
    assert diagnostics["selected_frames"] == 32
    assert diagnostics["selected_objects"] == [1, 2]
    episodes = selected_pairs_to_episode_rows(selected)
    validation = validate_paired_episode_rows(
        episodes, minimum_frames=8, maximum_frames=8
    )
    assert validation == {
        "objects": 2,
        "episodes": 4,
        "frames": 32,
        "train_physical_timestamps": 16,
        "heldout_physical_timestamps": 16,
        "global_timestamp_overlap": 0,
        "duplicate_rgb": 0,
    }
    assert all(
        episode["object_evidence_frame_count"] == episode["frame_count"]
        for episode in episodes
    )


def test_selection_fails_closed_when_whole_pairs_cannot_meet_budget() -> None:
    frames = [_frame("cam0", "center", value) for value in range(20)]
    evidence = [_evidence(4, frame) for frame in frames]
    windows, _ = enumerate_object_visible_windows(
        frames,
        evidence,
        minimum_frames=8,
        maximum_frames=8,
        preferred_frames=8,
    )
    with pytest.raises(ValueError, match="coverage is insufficient"):
        select_paired_object_episodes(
            {4: pair_object_windows(windows)},
            target_frames=32,
            minimum_total_frames=32,
            maximum_total_frames=40,
        )


def test_serialized_validation_rejects_obb_evidence_and_split_leakage() -> None:
    frames = [_frame("cam0", "center", value) for value in range(24)]
    evidence = [_evidence(5, frame) for frame in frames]
    windows, _ = enumerate_object_visible_windows(
        frames,
        evidence,
        minimum_frames=8,
        maximum_frames=8,
        preferred_frames=8,
    )
    selected, _ = select_paired_object_episodes(
        {5: pair_object_windows(windows)},
        target_frames=16,
        minimum_total_frames=16,
        maximum_total_frames=20,
    )
    episodes = selected_pairs_to_episode_rows(selected)
    bad_source = copy.deepcopy(episodes)
    bad_source[0]["frames"][0]["object_evidence"]["bbox_source"] = "projected_current_obb"
    with pytest.raises(ValueError, match="OBB/detector projection"):
        validate_paired_episode_rows(bad_source, minimum_frames=8, maximum_frames=8)
    leaked = copy.deepcopy(episodes)
    leaked[1]["frames"][0]["physical_timestamp"] = leaked[0]["frames"][0][
        "physical_timestamp"
    ]
    with pytest.raises(ValueError, match="timestamp leakage"):
        validate_paired_episode_rows(leaked, minimum_frames=8, maximum_frames=8)
