from pathlib import Path

import numpy as np
import torch

from scene_graph.map_update.mask_observations import (
    get_pairwise_mask_overlap_records,
    register_detection_mask_observations,
    resolve_object_mask_observations,
)


def _touch_npz(path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(path, marker=np.asarray([1], dtype=np.uint8))


def _register(state: dict, root: Path, image_id: int, cap: int = 1) -> None:
    mask = np.ones((8, 8), dtype=bool)
    register_detection_mask_observations(
        state,
        {"masks": [mask], "masks_inlier": [mask]},
        [image_id],
        [0],
        root,
        max_per_object=cap,
    )


def test_cap_unlinks_evicted_sidecar_in_exact_object_dir(tmp_path: Path) -> None:
    root = tmp_path / "masks"
    state = {"object_id": torch.tensor([7]), "object_mask_observations": [[]]}
    _register(state, root, 1)
    evicted = Path(state["object_mask_observations"][0][0]["path"])
    assert evicted.is_file()
    _register(state, root, 2)
    assert not evicted.exists()
    assert len(state["object_mask_observations"][0]) == 1
    assert Path(state["object_mask_observations"][0][0]["path"]).is_file()


def test_cap_never_unlinks_sidecar_outside_exact_object_dir(tmp_path: Path) -> None:
    root = tmp_path / "masks"
    external = tmp_path / "external.npz"
    _touch_npz(external)
    state = {
        "object_id": torch.tensor([7]),
        "object_mask_observations": [[{"path": str(external), "image_id": 0}]],
    }
    _register(state, root, 2)
    assert external.is_file()


def test_pairwise_overlap_survives_per_object_cap_and_sidecar_deletion(
    tmp_path: Path,
) -> None:
    root = tmp_path / "masks"
    state = {
        "object_id": torch.tensor([7, 8]),
        "object_mask_observations": [[], []],
        "id_redirect": {},
    }
    first = np.zeros((64, 64), dtype=bool)
    first[10:50, 10:50] = True
    near_duplicate = np.zeros((64, 64), dtype=bool)
    near_duplicate[10:50, 12:50] = True
    register_detection_mask_observations(
        state,
        {"masks": [first, near_duplicate]},
        [1, 1],
        [0, 1],
        root,
        max_per_object=1,
        max_pair_images_per_pair=1,
    )
    evicted_paths = [Path(row[0]["path"]) for row in state["object_mask_observations"]]
    assert all(path.is_file() for path in evicted_paths)

    left = np.zeros((64, 64), dtype=bool)
    left[5:25, 5:25] = True
    right = np.zeros((64, 64), dtype=bool)
    right[5:25, 35:55] = True
    register_detection_mask_observations(
        state,
        {"masks": [left, right]},
        [2, 2],
        [0, 1],
        root,
        max_per_object=1,
        max_pair_images_per_pair=1,
    )

    assert all(not path.exists() for path in evicted_paths)
    assert [row[0]["image_id"] for row in state["object_mask_observations"]] == [2, 2]
    records = get_pairwise_mask_overlap_records(state, 7, 8)
    assert len(records) == 1
    assert records[0]["image_id"] == 1
    assert records[0]["raw_iou"] >= 0.80
    assert records[0]["raw_containment"] >= 0.95
    assert len(records[0]["evidence_fingerprint"]) == 64
    assert not any("path" in key for key in records[0])


def test_archived_absolute_path_is_remapped_and_orphans_reported(tmp_path: Path) -> None:
    root = tmp_path / "masks"
    wanted = root / "object_000007" / "img_000001_det_0000.npz"
    orphan = root / "object_000007" / "img_000002_det_0000.npz"
    _touch_npz(wanted)
    _touch_npz(orphan)
    state = {
        "object_id": torch.tensor([7]),
        "object_mask_observations": [[{
            "image_id": 1,
            "path": "/archived/run/masks/object_000007/img_000001_det_0000.npz",
        }]],
    }
    index = resolve_object_mask_observations(state, root)
    assert index.for_object_id(7) == [wanted.resolve()]
    assert index.diagnostics["mode"] == "canonical_state"
    assert index.diagnostics["remapped_archived_absolute_record_count"] == 1
    assert index.diagnostics["orphan_sidecar_count"] == 1


def test_present_empty_canonical_list_never_falls_back_to_glob(tmp_path: Path) -> None:
    root = tmp_path / "masks"
    _touch_npz(root / "object_000007" / "stale.npz")
    state = {"object_id": torch.tensor([7]), "object_mask_observations": [[]]}
    index = resolve_object_mask_observations(state, root)
    assert index.for_object_id(7) == []
    assert index.diagnostics["legacy_fallback_used"] is False
    assert index.diagnostics["orphan_sidecar_count"] == 1


def test_absent_canonical_list_uses_reported_legacy_fallback(tmp_path: Path) -> None:
    root = tmp_path / "masks"
    wanted = root / "object_000007" / "legacy.npz"
    _touch_npz(wanted)
    index = resolve_object_mask_observations({"object_id": torch.tensor([7])}, root)
    assert index.for_object_id(7) == [wanted.resolve()]
    assert index.diagnostics["mode"] == "legacy_filesystem_fallback"
    assert index.diagnostics["legacy_fallback_used"] is True


def test_derivative_crop_bank_expands_only_canonical_image_ids(tmp_path: Path) -> None:
    root = tmp_path / "review_masks"
    first = root / "object_000008" / "img_000012_member_000001.npz"
    second = root / "object_000008" / "img_000012_member_000002.npz"
    stale = root / "object_000008" / "img_000013_member_000001.npz"
    for path in (first, second, stale):
        _touch_npz(path)
    state = {
        "object_id": torch.tensor([8]),
        "object_mask_observations": [[{
            "image_id": 12,
            "path": "object_000008/img_000012_assembly.npz",
        }]],
    }
    index = resolve_object_mask_observations(state, root, expand_image_matches=True)
    assert index.for_object_id(8) == [first.resolve(), second.resolve()]
    assert index.diagnostics["expanded_derivative_sidecar_count"] == 1
    assert index.diagnostics["orphan_sidecar_count"] == 1


def test_object_scope_excludes_unreviewed_rows_and_files(tmp_path: Path) -> None:
    root = tmp_path / "review_masks"
    wanted = root / "object_000008" / "img_000012_member_000001.npz"
    unrelated = root / "object_000009" / "img_000013_member_000001.npz"
    _touch_npz(wanted)
    _touch_npz(unrelated)
    state = {
        "object_id": torch.tensor([8, 9]),
        "object_mask_observations": [
            [{"image_id": 12, "path": "object_000008/img_000012_assembly.npz"}],
            [{"image_id": 13, "path": "object_000009/img_000013_assembly.npz"}],
        ],
    }
    index = resolve_object_mask_observations(
        state,
        root,
        expand_image_matches=True,
        include_object_ids=[8],
    )
    assert index.for_object_id(8) == [wanted.resolve()]
    assert index.for_object_id(9) == []
    assert index.diagnostics["object_id_scope"] == [8]
    assert index.diagnostics["missing_record_count"] == 0
    assert index.diagnostics["orphan_sidecar_count"] == 0
