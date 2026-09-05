from __future__ import annotations

import json
import shutil
import sys
from pathlib import Path

import numpy as np
import pytest
from PIL import Image

torch = pytest.importorskip("torch")

from scripts.geometry.merge_farm_full_colmap_rescue import main


def _frame(source: str, frame_id: str, rgb: str, depth: str) -> dict:
    return {
        "source_image": source,
        "frame_id": frame_id,
        "physical_timestamp": frame_id,
        "timestamp_ns": int(frame_id) * 100_000_000,
        "camera": "cam01",
        "depth_size": [2, 2],
        "colmap_image_id": int(frame_id),
        "colmap_camera_id": 1,
        "K": np.eye(3).tolist(),
        "T_world_cam": np.eye(4).tolist(),
        "rgb_path": rgb,
        "depth_path": depth,
    }


def _npz(path: Path, value: int) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        path,
        image_shape=np.asarray([2, 2], dtype=np.int32),
        raw_mask=np.full((2, 2), value, dtype=np.uint8),
    )


def _packed_npz(path: Path, mask: np.ndarray, crop: list[int]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    height, width = mask.shape
    np.savez_compressed(
        path,
        image_shape=np.asarray([height, width], dtype=np.int32),
        raw_bits=np.packbits(mask.reshape(-1), bitorder="little"),
        raw_shape=np.asarray(mask.shape, dtype=np.int32),
        raw_bbox_xyxy=np.asarray([0, 0, width, height], dtype=np.int32),
        inlier_bits=np.packbits(mask.reshape(-1), bitorder="little"),
        inlier_shape=np.asarray(mask.shape, dtype=np.int32),
        inlier_bbox_xyxy=np.asarray([0, 0, width, height], dtype=np.int32),
        crop_jpeg_bytes=np.asarray(crop, dtype=np.uint8),
        crop_bbox_xyxy=np.asarray([0, 0, width, height], dtype=np.int32),
        crop_shape=np.asarray([height, width], dtype=np.int32),
    )


def test_merge_publishes_canonical_source_snapshot_and_rescue_mask(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    pixels = np.arange(12, dtype=np.uint8).reshape(2, 2, 3)
    Image.fromarray(pixels, mode="RGB").save(tmp_path / "source.png")
    np.save(tmp_path / "source.npy", np.ones((2, 2), dtype=np.float32))
    Image.fromarray(pixels, mode="RGB").save(tmp_path / "rescue.png")
    np.save(tmp_path / "rescue.npy", np.ones((2, 2), dtype=np.float32))

    source_frames = tmp_path / "source_frames.json"
    source_frames.write_text(
        json.dumps(
            {
                "frames": [
                    _frame("cam01_000042_center.png", "42", "source.png", "source.npy"),
                    _frame("cam01_000042_center.png", "42", "source.png", "source.npy"),
                ]
            }
        ),
        encoding="utf-8",
    )
    rescue_frames = tmp_path / "rescue_frames.json"
    rescue_frames.write_text(
        json.dumps(
            {
                "frames": [
                    _frame("cam01_000043_center.png", "43", "rescue.png", "rescue.npy")
                ]
            }
        ),
        encoding="utf-8",
    )

    masks = tmp_path / "source_masks"
    source_mask7 = masks / "object_000007/img_000000_det_0000.npz"
    source_mask7_dup = masks / "object_000007/img_000001_det_0000.npz"
    source_mask8 = masks / "object_000008/img_000000_det_0002.npz"
    source_mask8_dup = masks / "object_000008/img_000001_det_0002.npz"
    _packed_npz(
        source_mask7,
        np.asarray([[1, 0], [0, 0]], dtype=np.uint8),
        [1, 2],
    )
    _packed_npz(
        source_mask8,
        np.asarray([[0, 1], [1, 0]], dtype=np.uint8),
        [3, 4, 5],
    )
    shutil.copyfile(source_mask7, source_mask7_dup)
    shutil.copyfile(source_mask8, source_mask8_dup)
    source_state = tmp_path / "source.pt"
    torch.save(
        {
            "state": {
                "object_id": torch.tensor([7, 8, 360]),
                "images": [{"image_id": 0}, {"image_id": 1}],
                "image_positions": [torch.zeros(3), torch.zeros(3)],
                "object_image_ids": [[0, 1], [0, 1], [0, 1]],
                "viewpoint_image_ids": [[0, 1], [0, 1], [0, 1]],
                "object_assembly_member_ids": [[], [], [7, 8]],
                "object_mask_observations": [
                    [
                        {"image_id": 0, "path": str(source_mask7)},
                        {"image_id": 1, "path": str(source_mask7_dup)},
                    ],
                    [
                        {"image_id": 0, "path": str(source_mask8)},
                        {"image_id": 1, "path": str(source_mask8_dup)},
                    ],
                    [
                        {
                            "image_id": image_id,
                            "path": (
                                "/farm-run/mapping/masks/object_000360/"
                                f"img_{image_id:06d}_assembly.npz"
                            ),
                            "assembly_member_ids_in_frame": [7, 8],
                            "raw_pixels": 999,
                            "inlier_pixels": 999,
                        }
                        for image_id in range(2)
                    ],
                ],
                "means": torch.zeros((3, 3)),
                "cov6": torch.zeros((3, 6)),
                "object_box_centers_m": torch.zeros((3, 3)),
                "object_box_dimensions_m": torch.ones((3, 3)),
            },
            "meta": {},
        },
        source_state,
    )
    rescue_state = tmp_path / "rescue.pt"
    torch.save(
        {"state": {"images": [{}], "image_positions": [torch.ones(3)]}},
        rescue_state,
    )

    refinement_root = tmp_path / "refinement"
    rescue_mask = refinement_root / "masks/object_000007/img_000002_det_0000.npz"
    _npz(rescue_mask, 3)
    refinement = refinement_root / "result.json"
    refinement_root.mkdir(parents=True, exist_ok=True)
    refinement.write_text(
        json.dumps(
            {
                "objects": [
                    {
                        "object_id": 7,
                        "accepted": True,
                        "views": [
                            {
                                "accepted": True,
                                "source_image": "cam01_000043_center.png",
                                "physical_timestamp_ns": 43,
                                "rescue_image_id": 0,
                                "global_image_id": 2,
                                "mask_relative": "masks/object_000007/img_000002_det_0000.npz",
                                "final_pixels": 4,
                                "confidence": 0.9,
                            }
                        ],
                    }
                ]
            }
        ),
        encoding="utf-8",
    )
    catalog = tmp_path / "catalog.json"
    catalog.write_text(json.dumps([{"id": 7}]), encoding="utf-8")
    output = tmp_path / "combined"
    monkeypatch.setattr(
        "scripts.geometry.merge_farm_full_colmap_rescue.require_train_fit_refinement",
        lambda *args, **kwargs: None,
    )
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "merge_farm_full_colmap_rescue.py",
            "--source-state", str(source_state),
            "--source-frames", str(source_frames),
            "--source-mask-root", str(masks),
            "--rescue-state", str(rescue_state),
            "--rescue-frames", str(rescue_frames),
            "--refinement", str(refinement),
            "--prior-catalog", str(catalog),
            "--output-dir", str(output),
        ],
    )

    assert main() == 0
    assert output.is_dir()
    assert not list(tmp_path.glob(".combined.*.tmp"))
    frames = json.loads((output / "frames.json").read_text(encoding="utf-8"))["frames"]
    assert len(frames) == 2
    assert frames[1]["frame_id"] == "43"
    canonical = torch.load(output / "scene_state_source_canonical.pt", weights_only=False)["state"]
    combined = torch.load(output / "scene_state_pre_geometry.pt", weights_only=False)["state"]
    assert len(canonical["images"]) == 1
    assert len(combined["images"]) == 2
    assert canonical["object_image_ids"] == [[0], [0], [0]]
    assert (output / "masks/object_000008/img_000000_det_0002.npz").is_file()
    assembly_observation = canonical["object_mask_observations"][2][0]
    assert assembly_observation["raw_pixels"] == 3
    assert assembly_observation["inlier_pixels"] == 3
    assert (
        output / "masks/object_000360/img_000000_assembly.npz"
    ).is_file()
    report = json.loads((output / "merge_report.json").read_text(encoding="utf-8"))
    assert report["assembly_masks_reconstructed"] == 1
    rescue_observation = combined["object_mask_observations"][0][-1]
    assert rescue_observation["image_id"] == 1
    assert rescue_observation["path"].endswith("/img_000001_det_0000.npz")
    assert (output / "masks/object_000007/img_000001_det_0000.npz").is_file()
