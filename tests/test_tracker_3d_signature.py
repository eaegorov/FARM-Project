from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest
from PIL import Image

from farm_runtime.colmap_pose_reader import read_colmap_cameras_and_poses
from farm_runtime.tracker_3d_signature import (
    ColmapImageObservations,
    GatePolicy,
    SparsePoint3D,
    build_v2_shadow_evidence,
    evaluate_arrays,
    geometry_signature,
    point_label_evidence,
    read_selected_image_observations,
    read_selected_points3d,
    validate_episode_contract,
    validate_mask_directory,
)


def _write_text_model(root: Path) -> None:
    (root / "cameras.txt").write_text(
        "# cameras\n1 PINHOLE 4 3 10 10 2 1.5\n", encoding="utf-8"
    )
    (root / "images.txt").write_text(
        "# images\n"
        "1 1 0 0 0 0 0 0 1 cam0_000001_center.png\n"
        "0 0 10 1 1 11 3 2 -1\n"
        "2 1 0 0 0 0 0 0 1 cam0_000002_center.png\n"
        "0 0 10 1 1 11 2 2 12\n",
        encoding="utf-8",
    )
    (root / "points3D.txt").write_text(
        "# points\n"
        "10 0 0 1 255 0 0 0.1 1 0 2 0\n"
        "11 0.1 0 1 0 255 0 0.2 1 1 2 1\n"
        "12 0.2 0 1 0 0 255 0.3 2 2\n",
        encoding="utf-8",
    )


def _episode() -> dict:
    frames = []
    for index in range(2):
        source = f"cam0_00000{index + 1}_center.png"
        frames.append(
            {
                "episode_index": index,
                "materialized_name": f"{index:06d}__{source}",
                "source_name": source,
                "physical_timestamp": f"00000{index + 1}",
                "colmap_image_id": index + 1,
                "colmap_camera_id": 1,
                "T_world_camera_m": np.eye(4).tolist(),
            }
        )
    return {
        "schema": "farm.materialized-tracker-episode.v1",
        "episode_id": "cam0__center__train__000",
        "camera": "cam0",
        "view_family": "center",
        "split": "train",
        "frame_count": 2,
        "frames": frames,
    }


def test_text_colmap_sampling_keeps_background_and_competing_votes(
    tmp_path: Path,
) -> None:
    _write_text_model(tmp_path)
    source_format, cameras, poses, _ = read_colmap_cameras_and_poses(tmp_path)
    observations, _ = read_selected_image_observations(
        tmp_path, source_format=source_format, selected_image_ids={1, 2}
    )
    points, path = read_selected_points3d(
        tmp_path, source_format=source_format, selected_point_ids={10, 11, 12}
    )
    frames = validate_episode_contract(
        _episode(),
        cameras=cameras,
        poses_by_id={pose.image_id: pose for pose in poses},
        observations_by_id=observations,
        meters_per_scene_unit=1.0,
        maximum_translation_error_m=1e-8,
        maximum_rotation_error_degrees=1e-6,
    )
    first = np.zeros((3, 4), dtype=np.uint8)
    second = np.zeros((3, 4), dtype=np.uint8)
    first[0, 0] = 1
    first[1, 1] = 1
    second[0, 0] = 1
    second[1, 1] = 2
    second[2, 2] = 2
    episode_raw, identities = evaluate_arrays(
        episode_id=_episode()["episode_id"],
        frame_rows=frames,
        masks=[first, second],
        observations_by_id=observations,
        points3d=points,
        points3d_file_available=path is not None,
        meters_per_scene_unit=1.0,
        policy=GatePolicy(
            minimum_frame_support=0,
            minimum_physical_timestamps=0,
            minimum_persistence_ratio=0,
            minimum_median_area_px=0,
            maximum_tiny_frame_fraction=1,
            minimum_median_largest_2d_component_fraction=0,
            minimum_sparse_observations=0,
            minimum_unique_sparse_points=0,
            minimum_point_observations=1,
            minimum_point_label_purity=0,
            minimum_qualified_sparse_points=0,
            minimum_qualified_sparse_point_fraction=0,
            maximum_ambiguous_sparse_point_fraction=1,
            minimum_largest_3d_component_fraction=0,
        ),
    )
    by_id = {row["local_id"]: row for row in identities}
    evidence = {
        row["point3d_id"]: row
        for row in by_id[1]["raw"]["sparse_evidence"]["point_label_evidence"]
    }
    assert source_format == "text"
    assert episode_raw["foreground_sparse_observations"] == 5
    assert by_id[1]["identity_key"].endswith("::local:1")
    assert evidence[10]["local_label_purity"] == 1.0
    assert evidence[11]["local_label_purity"] == 0.5
    assert evidence[11]["dominant_label_tie"] is True
    assert evidence[11]["competing_foreground_label_counts"] == {"2": 1}
    provenance = evidence[11]["colmap_visibility_provenance"]
    assert provenance["available"] is True
    assert [row["sampled_local_id"] for row in provenance["observations"]] == [1, 2]
    assert all(row["within_identity_active_span"] for row in provenance["observations"])
    states = {
        row["point3d_id"]: row["state"]
        for row in by_id[1]["shadow_v2_evidence"]["point_states"]
    }
    assert states == {10: "supported_consistent", 11: "competing_id_conflict"}
    assert by_id[1]["decision_v2_shadow"]["global_association_authorized"] is False
    assert by_id[1]["raw"]["sparse_evidence"]["unique_sparse_point3d_ids"] == [10, 11]


def test_episode_contract_rejects_name_camera_and_pose_mismatch(tmp_path: Path) -> None:
    _write_text_model(tmp_path)
    source_format, cameras, poses, _ = read_colmap_cameras_and_poses(tmp_path)
    observations, _ = read_selected_image_observations(
        tmp_path, source_format=source_format, selected_image_ids={1, 2}
    )
    broken = _episode()
    broken["frames"][0]["source_name"] = "wrong.png"
    broken["frames"][0]["materialized_name"] = "000000__wrong.png"
    with pytest.raises(ValueError, match="name/image ID mismatch"):
        validate_episode_contract(
            broken,
            cameras=cameras,
            poses_by_id={pose.image_id: pose for pose in poses},
            observations_by_id=observations,
            meters_per_scene_unit=1.0,
            maximum_translation_error_m=1e-8,
            maximum_rotation_error_degrees=1e-6,
        )
    broken = _episode()
    broken["frames"][0]["colmap_camera_id"] = 99
    with pytest.raises(ValueError, match="metadata mismatch|camera mismatch"):
        validate_episode_contract(
            broken,
            cameras=cameras,
            poses_by_id={pose.image_id: pose for pose in poses},
            observations_by_id=observations,
            meters_per_scene_unit=1.0,
            maximum_translation_error_m=1e-8,
            maximum_rotation_error_degrees=1e-6,
        )
    broken = _episode()
    broken["frames"][1]["T_world_camera_m"][0][3] = 0.1
    with pytest.raises(ValueError, match="translation differs"):
        validate_episode_contract(
            broken,
            cameras=cameras,
            poses_by_id={pose.image_id: pose for pose in poses},
            observations_by_id=observations,
            meters_per_scene_unit=1.0,
            maximum_translation_error_m=1e-8,
            maximum_rotation_error_degrees=1e-6,
        )


def test_mask_directory_requires_exact_png_names_and_native_shape(
    tmp_path: Path,
) -> None:
    rows = [
        {
            "expected_mask_name": "000000__frame.png",
            "height": 3,
            "width": 4,
            "episode_index": 0,
            "physical_timestamp": "1",
            "materialized_name": "000000__frame.png",
            "source_name": "frame.png",
            "colmap_image_id": 1,
        }
    ]
    mask_dir = tmp_path / "masks"
    mask_dir.mkdir()
    Image.fromarray(np.zeros((3, 4), dtype=np.uint8)).save(
        mask_dir / "000000__frame.png"
    )
    assert [path.name for path in validate_mask_directory(mask_dir, rows)] == [
        "000000__frame.png"
    ]
    (mask_dir / "notes.txt").write_text("junk", encoding="utf-8")
    with pytest.raises(ValueError, match="only PNG"):
        validate_mask_directory(mask_dir, rows)
    observation = ColmapImageObservations(
        1, 1, "frame.png", np.zeros((0, 2)), np.zeros(0, dtype=np.int64)
    )
    with pytest.raises(ValueError, match="does not match COLMAP camera"):
        evaluate_arrays(
            episode_id="episode",
            frame_rows=rows,
            masks=[np.zeros((2, 4), dtype=np.uint8)],
            observations_by_id={1: observation},
            points3d={},
            points3d_file_available=False,
            meters_per_scene_unit=1,
            policy=GatePolicy(require_points3d=False),
        )


def test_geometry_signature_exposes_distant_components_and_compactness() -> None:
    points = np.asarray(
        [[0, 0, 0], [0.01, 0, 0], [0.02, 0, 0], [10, 0, 0], [10.01, 0, 0]],
        dtype=np.float64,
    )
    signature = geometry_signature([1, 2, 3, 4, 5], points)
    assert signature["connectivity"]["component_sizes"] == [3, 2]
    assert signature["connectivity"]["largest_component_fraction"] == pytest.approx(0.6)
    assert signature["aabb_diagonal_m"] == pytest.approx(10.01)
    assert len(signature["covariance_m2"]) == 3

    cluster_with_floater = np.asarray(
        [[0, 0, 0], [0.01, 0, 0], [0.02, 0, 0], [0.03, 0, 0], [0.04, 0, 0], [20, 0, 0]],
        dtype=np.float64,
    )
    floater_signature = geometry_signature(range(6), cluster_with_floater)
    assert floater_signature["connectivity"]["component_sizes"] == [5, 1]


def test_gate_policy_changes_decision_not_raw_measurements() -> None:
    frames = []
    masks = []
    observations = {}
    points = {}
    for index in range(4):
        frames.append(
            {
                "episode_index": index,
                "expected_mask_name": f"{index}.png",
                "materialized_name": f"{index}.png",
                "source_name": f"source-{index}.png",
                "physical_timestamp": str(index),
                "colmap_image_id": index + 1,
                "height": 4,
                "width": 4,
            }
        )
        mask = np.zeros((4, 4), dtype=np.uint8)
        mask[:2, :2] = 1
        masks.append(mask)
        xy = np.asarray([[0, 0], [1, 0], [0, 1], [1, 1]], dtype=np.float64)
        point_ids = np.asarray([10, 11, 12, 13], dtype=np.int64)
        observations[index + 1] = ColmapImageObservations(
            index + 1, 1, f"source-{index}.png", xy, point_ids
        )
    for offset, point_id in enumerate([10, 11, 12, 13]):
        points[point_id] = SparsePoint3D(point_id, (0.01 * offset, 0, 1), 0.1, 4)
    permissive = GatePolicy(
        minimum_frame_support=1,
        minimum_physical_timestamps=1,
        minimum_persistence_ratio=0,
        minimum_median_area_px=1,
        tiny_area_px=1,
        maximum_tiny_frame_fraction=1,
        minimum_median_largest_2d_component_fraction=0,
        minimum_sparse_observations=1,
        minimum_unique_sparse_points=1,
        minimum_point_observations=1,
        minimum_point_label_purity=0,
        minimum_qualified_sparse_points=1,
        minimum_qualified_sparse_point_fraction=0,
        maximum_ambiguous_sparse_point_fraction=1,
        minimum_largest_3d_component_fraction=0,
    )
    strict = GatePolicy(minimum_unique_sparse_points=100)
    _, accepted = evaluate_arrays(
        episode_id="episode-a",
        frame_rows=frames,
        masks=masks,
        observations_by_id=observations,
        points3d=points,
        points3d_file_available=True,
        meters_per_scene_unit=1,
        policy=permissive,
    )
    _, rejected = evaluate_arrays(
        episode_id="episode-a",
        frame_rows=frames,
        masks=masks,
        observations_by_id=observations,
        points3d=points,
        points3d_file_available=True,
        meters_per_scene_unit=1,
        policy=strict,
    )
    assert accepted[0]["decision"]["passed"] is True
    assert rejected[0]["decision"]["passed"] is False
    assert "unique_sparse_points" in rejected[0]["decision"]["failed_checks"]
    assert accepted[0]["raw"] == rejected[0]["raw"]


def test_point_purity_counts_background_as_disagreement() -> None:
    rows = point_label_evidence(
        {7: {0: 2, 3: 2}, 8: {3: 3}, 9: {3: 1, 4: 2}}, local_id=3
    )
    by_point = {row["point3d_id"]: row for row in rows}
    assert by_point[7]["local_label_purity"] == 0.5
    assert by_point[7]["has_background_disagreement"] is True
    assert by_point[8]["local_label_purity"] == 1.0
    assert by_point[9]["dominant_label_ids"] == [4]


def test_v2_shadow_separates_single_observation_support_without_changing_v1() -> None:
    frames = []
    masks = []
    observations = {}
    points = {}
    xy_all = np.asarray([[0, 0], [1, 0], [2, 0], [3, 0], [0, 1]], dtype=np.float64)
    all_ids = np.asarray([10, 11, 12, 13, 14], dtype=np.int64)
    for index in range(3):
        frames.append(
            {
                "episode_index": index,
                "expected_mask_name": f"{index}.png",
                "materialized_name": f"{index}.png",
                "source_name": f"source-{index}.png",
                "physical_timestamp": str(index),
                "colmap_image_id": index + 1,
                "height": 3,
                "width": 4,
            }
        )
        mask = np.zeros((3, 4), dtype=np.uint8)
        mask[0, 0] = 1
        mask[0, 1] = 1
        if index == 0:
            mask[0, 2] = mask[0, 3] = mask[1, 0] = 1
            xy, point_ids = xy_all, all_ids
        else:
            xy, point_ids = xy_all[:2], all_ids[:2]
        masks.append(mask)
        observations[index + 1] = ColmapImageObservations(
            index + 1, 1, f"source-{index}.png", xy, point_ids
        )
    for offset, point_id in enumerate(all_ids.tolist()):
        points[point_id] = SparsePoint3D(
            point_id, (0.01 * offset, 0.0, 1.0), 0.1, 3 if point_id < 12 else 1
        )
    policy = GatePolicy(
        minimum_frame_support=1,
        minimum_physical_timestamps=1,
        minimum_persistence_ratio=0,
        minimum_median_area_px=1,
        tiny_area_px=1,
        maximum_tiny_frame_fraction=1,
        minimum_median_largest_2d_component_fraction=0,
        minimum_sparse_observations=1,
        minimum_unique_sparse_points=1,
        minimum_point_observations=2,
        minimum_point_label_purity=0.67,
        minimum_qualified_sparse_points=2,
        minimum_qualified_sparse_point_fraction=0.25,
        maximum_ambiguous_sparse_point_fraction=0.35,
        minimum_largest_3d_component_fraction=0,
    )
    _, identities = evaluate_arrays(
        episode_id="episode-shadow",
        frame_rows=frames,
        masks=masks,
        observations_by_id=observations,
        points3d=points,
        points3d_file_available=True,
        meters_per_scene_unit=1,
        policy=policy,
    )
    identity = identities[0]
    assert identity["decision"]["passed"] is False
    assert identity["decision"]["failed_checks"] == ["ambiguous_sparse_point_fraction"]
    shadow = identity["shadow_v2_evidence"]
    assert shadow["counts"]["insufficient_support_points"] == 3
    assert shadow["counts"]["supported_points"] == 2
    assert shadow["counts"]["visible_conflict_points"] == 0
    assert shadow["fractions"]["visible_conflict_fraction_among_supported"] == 0
    assert identity["decision_v2_shadow"]["passed"] is True
    assert identity["decision_v2_shadow"]["publication_effect"].startswith("none")


def test_shadow_state_classifier_keeps_background_and_id_conflicts_separate() -> None:
    rows = point_label_evidence(
        {
            1: {3: 1},
            2: {3: 3},
            3: {0: 1, 3: 1},
            4: {3: 1, 4: 1},
            5: {0: 1, 3: 1, 4: 1},
        },
        local_id=3,
    )
    shadow = build_v2_shadow_evidence(rows, local_id=3, policy=GatePolicy())
    states = {row["point3d_id"]: row["state"] for row in shadow["point_states"]}
    assert states == {
        1: "insufficient_support",
        2: "supported_consistent",
        3: "background_dropout",
        4: "competing_id_conflict",
        5: "background_and_competing_id_conflict",
    }
    assert shadow["counts"]["supported_points"] == 4
    assert shadow["counts"]["visible_conflict_points"] == 3
    assert shadow["fractions"]["visible_conflict_fraction_among_supported"] == 0.75
