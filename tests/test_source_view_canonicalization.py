from __future__ import annotations

import copy
import shutil
from pathlib import Path

import numpy as np
import pytest
from PIL import Image

from farm_runtime.assembly_mask_reconstruction import reconstruct_assembly_mask
from farm_runtime.source_state_canonicalization import remap_source_state
from farm_runtime.source_view_canonicalization import (
    canonicalize_source_views,
    require_unique_rescue_sources,
)


def _artifacts(root: Path, stem: str, pixels: np.ndarray, depth: np.ndarray) -> tuple[str, str]:
    rgb = root / f"{stem}.png"
    depth_path = root / f"{stem}.npy"
    Image.fromarray(pixels, mode="RGB").save(rgb)
    np.save(depth_path, depth)
    return rgb.name, depth_path.name


def _frame(source: str, rgb: str, depth: str) -> dict:
    return {
        "source_image": source,
        "frame_id": "42",
        "physical_timestamp": "42",
        "timestamp_ns": 4_200_000_000,
        "camera": "cam01",
        "depth_size": [2, 2],
        "colmap_image_id": 17,
        "colmap_camera_id": 3,
        "K": np.eye(3).tolist(),
        "T_world_cam": np.eye(4).tolist(),
        "rgb_path": rgb,
        "depth_path": depth,
    }


def _duplicate_fixture(tmp_path: Path) -> tuple[Path, list[dict]]:
    pixels = np.arange(12, dtype=np.uint8).reshape(2, 2, 3)
    depth = np.arange(4, dtype=np.float32).reshape(2, 2)
    first_rgb, first_depth = _artifacts(tmp_path, "first", pixels, depth)
    second_rgb, second_depth = _artifacts(tmp_path, "second", pixels, depth)
    frames_path = tmp_path / "frames.json"
    frames_path.write_text("{}", encoding="utf-8")
    return frames_path, [
        _frame("cam01_000042_center.png", first_rgb, first_depth),
        _frame("cam01_000042_center.png", second_rgb, second_depth),
    ]


def test_canonicalizes_exact_duplicate_to_first_occurrence(tmp_path: Path) -> None:
    frames_path, frames = _duplicate_fixture(tmp_path)
    result = canonicalize_source_views(frames, frames_path, [[1, 2, 3], [1, 2, 3]])

    assert result.canonical_old_indices == (0,)
    assert result.old_to_new == (0, 0)
    assert result.audit["input_rows"] == 2
    assert result.audit["unique_source_images"] == 1
    assert result.audit["duplicate_rows"] == 1
    assert result.audit["duplicates"][0]["equivalence"]["rgb"]["method"] == "exact_sha256"


def test_accepts_only_exact_decoded_rgb_and_numeric_depth_fallback(tmp_path: Path) -> None:
    pixels = np.arange(12, dtype=np.uint8).reshape(2, 2, 3)
    first_rgb = tmp_path / "first.png"
    second_rgb = tmp_path / "second.png"
    Image.fromarray(pixels, mode="RGB").save(first_rgb, compress_level=0)
    Image.fromarray(pixels, mode="RGB").save(second_rgb, compress_level=9)
    first_depth = tmp_path / "first.npy"
    second_depth = tmp_path / "second.npy"
    np.save(first_depth, np.ones((2, 2), dtype=np.float32))
    np.save(second_depth, np.full((2, 2), 1.0000005, dtype=np.float32))
    frames_path = tmp_path / "frames.json"
    frames_path.write_text("{}", encoding="utf-8")
    frames = [
        _frame("same.png", first_rgb.name, first_depth.name),
        _frame("same.png", second_rgb.name, second_depth.name),
    ]

    result = canonicalize_source_views(frames, frames_path, [[0], [0]])
    equivalence = result.audit["duplicates"][0]["equivalence"]
    assert equivalence["rgb"]["method"] == "decoded_rgb_exact"
    assert equivalence["depth"]["method"] == "numeric_allclose"


@pytest.mark.parametrize("conflict", ["rgb", "depth", "K", "T_world_cam"])
def test_rejects_incompatible_duplicate(tmp_path: Path, conflict: str) -> None:
    frames_path, frames = _duplicate_fixture(tmp_path)
    if conflict == "rgb":
        pixels = np.full((2, 2, 3), 255, dtype=np.uint8)
        Image.fromarray(pixels, mode="RGB").save(tmp_path / frames[1]["rgb_path"])
    elif conflict == "depth":
        np.save(tmp_path / frames[1]["depth_path"], np.full((2, 2), 9.0, dtype=np.float32))
    else:
        frames[1][conflict][0][0] += 1.0e-6

    with pytest.raises(ValueError, match="RGB|depth|calibration"):
        canonicalize_source_views(frames, frames_path, [[0], [0]])


def _mask(root: Path, object_id: int, basename: str, payload: bytes) -> str:
    path = root / f"object_{object_id:06d}" / basename
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(payload)
    return f"/farm-run/mapping/masks/object_{object_id:06d}/{basename}"


def _packed_mask(
    path: Path,
    raw: np.ndarray,
    inlier: np.ndarray,
    crop_bytes: list[int],
) -> str:
    path.parent.mkdir(parents=True, exist_ok=True)
    height, width = raw.shape
    np.savez_compressed(
        path,
        image_shape=np.asarray([height, width], dtype=np.int32),
        raw_bits=np.packbits(raw.reshape(-1), bitorder="little"),
        raw_shape=np.asarray(raw.shape, dtype=np.int32),
        raw_bbox_xyxy=np.asarray([0, 0, width, height], dtype=np.int32),
        inlier_bits=np.packbits(inlier.reshape(-1), bitorder="little"),
        inlier_shape=np.asarray(inlier.shape, dtype=np.int32),
        inlier_bbox_xyxy=np.asarray([0, 0, width, height], dtype=np.int32),
        crop_jpeg_bytes=np.asarray(crop_bytes, dtype=np.uint8),
        crop_bbox_xyxy=np.asarray([0, 0, width, height], dtype=np.int32),
        crop_shape=np.asarray([height, width], dtype=np.int32),
    )
    return f"/farm-run/mapping/masks/{path.parent.name}/{path.name}"


def _assembly_state(mask_root: Path, *, existing_first: bool = False) -> dict:
    raw_a = np.asarray([[1, 0], [0, 1]], dtype=np.uint8)
    raw_b = np.asarray([[0, 1], [1, 0]], dtype=np.uint8)
    inlier_a = np.asarray([[1, 0], [0, 0]], dtype=np.uint8)
    inlier_b = np.asarray([[0, 0], [0, 1]], dtype=np.uint8)
    rows: list[list[dict]] = []
    for object_id, raw, inlier, crop in (
        (28, raw_a, inlier_a, [1, 2, 3]),
        (58, raw_b, inlier_b, [4, 5, 6, 7, 8]),
    ):
        first = mask_root / f"object_{object_id:06d}/img_000000_det_0000.npz"
        second = mask_root / f"object_{object_id:06d}/img_000001_det_0000.npz"
        first_ref = _packed_mask(first, raw, inlier, crop)
        second.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(first, second)
        second_ref = f"/farm-run/mapping/masks/{second.parent.name}/{second.name}"
        rows.append(
            [
                {"image_id": 0, "path": first_ref},
                {"image_id": 1, "path": second_ref},
            ]
        )
    assembly_rows = [
        {
            "image_id": image_id,
            "path": (
                f"/farm-run/mapping/masks/object_000360/"
                f"img_{image_id:06d}_assembly.npz"
            ),
            "assembly_member_ids_in_frame": [28, 58],
            "raw_pixels": 999,
            "inlier_pixels": 999,
            "crop_jpeg_bytes_len": 1,
        }
        for image_id in range(2)
    ]
    if existing_first:
        raw_union = np.logical_or(raw_a, raw_b).astype(np.uint8)
        inlier_union = np.logical_or(inlier_a, inlier_b).astype(np.uint8)
        _packed_mask(
            mask_root / "object_000360/img_000000_assembly.npz",
            raw_union,
            inlier_union,
            [4, 5, 6, 7, 8],
        )
    rows.append(assembly_rows)
    return {
        "object_id": np.asarray([28, 58, 360], dtype=np.int64),
        "images": [{"image_id": 0}, {"image_id": 1}],
        "image_positions": [[0], [0]],
        "object_image_ids": [[0, 1], [0, 1], [0, 1]],
        "viewpoint_image_ids": [[0, 1], [0, 1], [0, 1]],
        "object_assembly_member_ids": [[], [], [28, 58]],
        "object_mask_observations": rows,
    }


@pytest.mark.parametrize("existing_first", [False, True])
def test_missing_assembly_is_reconstructed_and_duplicate_view_collapsed(
    tmp_path: Path, existing_first: bool
) -> None:
    frames_path, frames = _duplicate_fixture(tmp_path)
    canonical = canonicalize_source_views(frames, frames_path, [[0], [0]])
    mask_root = tmp_path / "masks"
    state = _assembly_state(mask_root, existing_first=existing_first)
    staging = tmp_path / "staging"
    staging.mkdir()

    audit = remap_source_state(
        state, canonical, mask_root, staging_mask_root=staging
    )

    observation = state["object_mask_observations"][2][0]
    assert len(state["object_mask_observations"][2]) == 1
    assert observation["image_id"] == 0
    assert observation["path"].endswith("/img_000000_assembly.npz")
    assert observation["raw_pixels"] == 4
    assert observation["inlier_pixels"] == 2
    assert observation["crop_jpeg_bytes_len"] == 5
    assert audit["mask_observations_collapsed"] == 3
    assert audit["mask_observation_duplicate_groups_collapsed"] == 3
    if existing_first:
        assert audit["assembly_masks_reconstructed"] == 0
    else:
        assert audit["assembly_masks_reconstructed"] == 1
        emitted = staging / "object_000360/img_000000_assembly.npz"
        assert emitted.is_file()
        reconstruction = audit["assembly_mask_reconstructions"][0]
        assert len(reconstruction["member_provenance"]) == 2
        corrections = reconstruction["observation_metadata_corrections"]
        assert corrections["raw_pixels"]["previous"] == 999
        assert corrections["raw_pixels"]["reconstructed"] == 4


def test_missing_assembly_rejects_ambiguous_member_observation(tmp_path: Path) -> None:
    mask_root = tmp_path / "masks"
    state = _assembly_state(mask_root)
    state["object_mask_observations"][0].append(
        copy.deepcopy(state["object_mask_observations"][0][0])
    )

    with pytest.raises(ValueError, match="exactly one observation"):
        reconstruct_assembly_mask(
            state,
            assembly_object_id=360,
            old_image_id=0,
            assembly_observation=state["object_mask_observations"][2][0],
            source_mask_root=mask_root,
        )


def test_state_remap_preserves_different_objects_and_avoids_double_weighting(tmp_path: Path) -> None:
    frames_path, frames = _duplicate_fixture(tmp_path)
    canonical = canonicalize_source_views(frames, frames_path, [[0], [0]])
    mask_root = tmp_path / "masks"
    object7_first = _mask(mask_root, 7, "img_000000_det_0000.npz", b"same-mask")
    object7_second = _mask(mask_root, 7, "img_000001_det_0000.npz", b"same-mask")
    object8_second = _mask(mask_root, 8, "img_000001_det_0002.npz", b"object-eight")
    state = {
        "object_id": np.asarray([7, 8], dtype=np.int64),
        "images": [{"image_id": 0}, {"image_id": 1}],
        "image_positions": [[0], [0]],
        "object_image_ids": [[0, 1], [1]],
        "viewpoint_image_ids": [[1, 0], [1]],
        "object_mask_observations": [
            [
                {"image_id": 0, "path": object7_first},
                {"image_id": 1, "path": object7_second},
            ],
            [{"image_id": 1, "path": object8_second}],
        ],
    }

    audit = remap_source_state(state, canonical, mask_root)

    assert len(state["images"]) == 1
    assert state["object_image_ids"] == [[0], [0]]
    assert state["viewpoint_image_ids"] == [[0], [0]]
    assert len(state["object_mask_observations"][0]) == 1
    assert len(state["object_mask_observations"][1]) == 1
    assert state["object_mask_observations"][0][0]["path"].endswith(
        "/object_000007/img_000000_det_0000.npz"
    )
    assert state["object_mask_observations"][1][0]["path"].endswith(
        "/object_000008/img_000000_det_0002.npz"
    )
    assert audit["mask_observations_collapsed"] == 1
    assert {row["object_id"] for row in audit["mask_aliases"]} == {7, 8}


def test_state_remap_rejects_conflicting_same_object_masks(tmp_path: Path) -> None:
    frames_path, frames = _duplicate_fixture(tmp_path)
    canonical = canonicalize_source_views(frames, frames_path, [[0], [0]])
    mask_root = tmp_path / "masks"
    first = _mask(mask_root, 7, "img_000000_det_0000.npz", b"first")
    second = _mask(mask_root, 7, "img_000001_det_0000.npz", b"second")
    state = {
        "object_id": np.asarray([7], dtype=np.int64),
        "images": [{}, {}],
        "image_positions": [[0], [0]],
        "object_image_ids": [[0, 1]],
        "object_mask_observations": [[
            {"image_id": 0, "path": first},
            {"image_id": 1, "path": second},
        ]],
    }

    with pytest.raises(ValueError, match="not exactly equivalent"):
        remap_source_state(copy.deepcopy(state), canonical, mask_root)


def test_rejects_duplicate_or_overlapping_rescue_sources() -> None:
    with pytest.raises(ValueError, match="already exists"):
        require_unique_rescue_sources([{"source_image": "a.png"}], {"a.png"})
    with pytest.raises(ValueError, match="repeat"):
        require_unique_rescue_sources(
            [{"source_image": "b.png"}, {"source_image": "b.png"}], set()
        )
