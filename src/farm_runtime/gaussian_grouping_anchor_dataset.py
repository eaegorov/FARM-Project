"""Fail-closed Gaussian Grouping dataset adapter for an anchor allowlist.

This adapter deliberately does not turn a shadow-v2 tracker decision into an
authoritative v1 association.  It consumes the narrower, reconciler-produced
allowlist and materializes only its exact train/heldout object pairs.  Every
observed identity not named as foreground is explicitly mapped to background.
"""

from __future__ import annotations

import json
import os
import re
import shutil
import subprocess
import tempfile
from collections.abc import Mapping
from pathlib import Path
from typing import Any

import numpy as np
from PIL import Image

from .anchor_cross_fold_reconciliation import ALLOWLIST_SCHEMA
from .anchor_cross_fold_reconciliation import SCHEMA as RECON_SCHEMA
from .gaussian_grouping_dataset import (
    MAX_FOREGROUND_ID,
    _load_episode_rows,
    _load_json,
    _mask_array,
    _read_colmap_binary,
    _safe_symlink,
    _validate_declared_provenance,
    _validate_source_frame,
    file_provenance,
    render_colmap_text,
    sha256_file,
)

OUTPUT_SCHEMA = "farm.gaussian-grouping-anchor-dataset.v1"
MEASUREMENT_SCHEMA = "farm.sam3-concept-episode.v1"
SIGNATURE_SCHEMA = "farm.tracker-episode-3d-signatures.v1"
SHADOW_SCHEMA = "farm.tracker-episode-3d-signatures.v2-shadow"
_SCOPE = "bounded-diagnostic-gaussian-grouping-identity-dataset"
_REQUIRED_ADAPTER = (
    "consume this exact shadow allowlist without treating v2 shadow as v1 publication"
)
_IDENTITY_RE = re.compile(r"^(?P<episode>[^/]+)::local:(?P<local>[1-9][0-9]{0,2})$")
_COLMAP_CONTRACT_NAMES = {
    "cameras.bin",
    "cameras.txt",
    "images.bin",
    "images.txt",
    "points3D.bin",
    "points3D.txt",
}
_GG_SOURCE_FILES = (
    "train.py",
    "arguments/__init__.py",
    "scene/__init__.py",
    "scene/dataset_readers.py",
    "scene/gaussian_model.py",
    "gaussian_renderer/__init__.py",
    "utils/loss_utils.py",
    "LICENSE",
)


def _sequence(value: Any, label: str) -> list[Any]:
    if not isinstance(value, list):
        raise ValueError(f"{label} must be a list")
    return value


def _unique_strings(value: Any, label: str) -> list[str]:
    result = [str(item) for item in _sequence(value, label)]
    if any(not item for item in result) or len(result) != len(set(result)):
        raise ValueError(f"{label} contains an empty or duplicate value")
    return result


def _integer(
    value: Any, label: str, *, minimum: int = 0, maximum: int | None = None
) -> int:
    if isinstance(value, bool) or not isinstance(value, int):
        raise ValueError(f"{label} must be an integer")
    result = int(value)
    if result < minimum or (maximum is not None and result > maximum):
        raise ValueError(f"{label} is outside the supported range")
    return result


def _parse_identity_key(key: str) -> tuple[str, int]:
    match = _IDENTITY_RE.fullmatch(key)
    if match is None or Path(match.group("episode")).name != match.group("episode"):
        raise ValueError(f"unsafe allowlist identity key: {key!r}")
    local_id = int(match.group("local"))
    if local_id > 255:
        raise ValueError(f"allowlist local ID exceeds uint8: {key}")
    return match.group("episode"), local_id


def _validate_allowlist_contract(
    allowlist_path: Path,
) -> tuple[dict[str, Any], dict[str, int], list[dict[str, Any]], list[dict[str, Any]]]:
    allowlist = _load_json(allowlist_path, "Gaussian Grouping anchor allowlist")
    if (
        allowlist.get("schema") != ALLOWLIST_SCHEMA
        or allowlist.get("status") != "pass"
        or allowlist.get("scope") != _SCOPE
    ):
        raise ValueError("unsupported/failed Gaussian Grouping anchor allowlist")
    publication = allowlist.get("publication")
    expected_publication = {
        "production_farm_identity_authorized": False,
        "production_gaussian_grouping_training_authorized": False,
        "bounded_dataset_materialization_authorized": True,
        "gpu_training_executed": False,
        "native_v1_builder_compatible": False,
        "required_adapter": _REQUIRED_ADAPTER,
    }
    if publication != expected_publication:
        raise ValueError("unsupported allowlist publication contract")

    raw_mapping = allowlist.get("identity_key_to_compact_gg_id")
    if not isinstance(raw_mapping, Mapping) or not raw_mapping:
        raise ValueError("allowlist identity mapping is missing")
    mapping: dict[str, int] = {}
    for raw_key, raw_value in raw_mapping.items():
        key = str(raw_key)
        _parse_identity_key(key)
        if key in mapping:
            raise ValueError(f"duplicate allowlist identity key: {key}")
        mapping[key] = _integer(
            raw_value, f"allowlist compact ID for {key}", maximum=MAX_FOREGROUND_ID
        )
    selected = set(
        _unique_strings(allowlist.get("selected_identity_keys"), "selected identities")
    )
    dropped = set(
        _unique_strings(allowlist.get("dropped_identity_keys"), "dropped identities")
    )
    for key in selected | dropped:
        _parse_identity_key(key)
    if selected & dropped or selected | dropped != set(mapping):
        raise ValueError(
            "selected/drop identity sets do not exactly cover the allowlist mapping"
        )
    if {key for key, value in mapping.items() if value > 0} != selected:
        raise ValueError("foreground mapping disagrees with selected identities")
    if {key for key, value in mapping.items() if value == 0} != dropped:
        raise ValueError("background mapping disagrees with dropped identities")
    compact_ids = sorted(set(mapping.values()).difference({0}))
    if not compact_ids or compact_ids != list(range(1, max(compact_ids) + 1)):
        raise ValueError("allowlist foreground IDs must be compact from 1")

    episode_rows = _sequence(allowlist.get("episodes"), "allowlist episodes")
    object_rows = _sequence(allowlist.get("objects"), "allowlist objects")
    if any(not isinstance(row, Mapping) for row in episode_rows + object_rows):
        raise ValueError("allowlist episode/object row is not an object")
    episode_ids = [str(row.get("episode_id") or "") for row in episode_rows]
    if any(not value or Path(value).name != value for value in episode_ids):
        raise ValueError("allowlist contains an unsafe episode ID")
    if len(episode_ids) != len(set(episode_ids)):
        raise ValueError("allowlist contains duplicate episode rows")
    episode_by_id = {str(row["episode_id"]): dict(row) for row in episode_rows}

    object_ids: set[int] = set()
    declared_selected: set[str] = set()
    declared_episodes: set[str] = set()
    declared_compact_ids: set[int] = set()
    for row in object_rows:
        object_id = _integer(
            row.get("canonical_object_id"), "canonical object ID", minimum=1
        )
        compact_id = _integer(
            row.get("gg_identity_id"),
            "GG identity ID",
            minimum=1,
            maximum=MAX_FOREGROUND_ID,
        )
        if object_id in object_ids or compact_id in declared_compact_ids:
            raise ValueError("allowlist contains duplicate object/compact IDs")
        object_ids.add(object_id)
        declared_compact_ids.add(compact_id)
        keys = _unique_strings(
            row.get("identity_keys"), f"object {object_id} identities"
        )
        episodes = _unique_strings(
            row.get("episode_ids"), f"object {object_id} episodes"
        )
        if len(keys) != 2 or len(episodes) != 2:
            raise ValueError(
                "every allowlisted object must have exactly one train and one heldout identity"
            )
        if {mapping.get(key) for key in keys} != {compact_id}:
            raise ValueError(
                f"object {object_id} identities disagree with compact mapping"
            )
        if {episode_by_id.get(episode, {}).get("split") for episode in episodes} != {
            "train",
            "heldout",
        }:
            raise ValueError(
                f"object {object_id} must contain one train and one heldout episode"
            )
        if any(_parse_identity_key(key)[0] not in episodes for key in keys):
            raise ValueError(f"object {object_id} identity/episode mismatch")
        if any(
            _integer(
                episode_by_id[episode].get("canonical_object_id"),
                "episode object ID",
                minimum=1,
            )
            != object_id
            for episode in episodes
        ):
            raise ValueError(f"object {object_id} episode ownership mismatch")
        declared_selected.update(keys)
        declared_episodes.update(episodes)
    if declared_selected != selected or declared_episodes != set(episode_ids):
        raise ValueError(
            "allowlist objects do not exactly cover selected identities/episodes"
        )
    if declared_compact_ids != set(compact_ids):
        raise ValueError("allowlist objects do not exactly cover compact IDs")

    summary = allowlist.get("summary")
    expected_summary = {
        "canonical_object_count": len(object_rows),
        "episode_count": len(episode_rows),
        "foreground_identity_key_count": len(selected),
        "explicitly_dropped_identity_key_count": len(dropped),
        "mapping_key_count": len(mapping),
    }
    if summary != expected_summary:
        raise ValueError("allowlist summary disagrees with its exact contents")
    return (
        allowlist,
        mapping,
        [dict(row) for row in episode_rows],
        [dict(row) for row in object_rows],
    )


def _validate_reconciliation(
    allowlist: Mapping[str, Any],
    object_rows: list[dict[str, Any]],
    episodes_manifest: Path,
) -> dict[str, Any]:
    inputs = allowlist.get("inputs")
    if not isinstance(inputs, Mapping):
        raise ValueError("allowlist inputs are missing")
    declared = inputs.get("reconciliation")
    if not isinstance(declared, Mapping):
        raise ValueError("allowlist reconciliation provenance is missing")
    path = Path(str(declared.get("path") or "")).expanduser().resolve(strict=True)
    actual = file_provenance(path)
    _validate_declared_provenance(declared, actual, "allowlist reconciliation")
    reconciliation = _load_json(path, "anchor reconciliation")
    if (
        reconciliation.get("schema") != RECON_SCHEMA
        or reconciliation.get("status") != "pass"
        or reconciliation.get("mode") != "diagnostic-bounded-fail-closed"
    ):
        raise ValueError("unsupported/failed anchor reconciliation")
    publication = reconciliation.get("publication")
    if (
        not isinstance(publication, Mapping)
        or not publication
        or any(value is not False for value in publication.values())
    ):
        raise ValueError("reconciliation publication contract is not fail-closed")
    recon_inputs = reconciliation.get("inputs")
    if not isinstance(recon_inputs, Mapping):
        raise ValueError("reconciliation inputs are missing")
    _validate_declared_provenance(
        recon_inputs.get("episodes_manifest"),
        file_provenance(episodes_manifest),
        "reconciliation episodes manifest",
    )
    accepted = {
        _integer(row.get("object_id"), "reconciled object ID", minimum=1): row
        for row in _sequence(reconciliation.get("objects"), "reconciliation objects")
        if isinstance(row, Mapping) and row.get("status") == "accepted"
    }
    expected_ids = {int(row["canonical_object_id"]) for row in object_rows}
    if set(accepted) != expected_ids:
        raise ValueError(
            "allowlist object set differs from accepted reconciliation set"
        )
    for row in object_rows:
        object_id = int(row["canonical_object_id"])
        reconciled = accepted[object_id]
        mapping = reconciled.get("accepted_mapping")
        if not isinstance(mapping, Mapping):
            raise ValueError(
                f"accepted reconciliation mapping is missing for object {object_id}"
            )
        expected_keys = set(row["identity_keys"])
        actual_keys = {
            str(mapping.get("train_identity_key") or ""),
            str(mapping.get("heldout_identity_key") or ""),
        }
        if (
            expected_keys != actual_keys
            or _integer(mapping.get("raw_global_id"), "raw shadow global ID", minimum=1)
            != _integer(
                row.get("raw_shadow_global_id"), "allowlist raw shadow ID", minimum=1
            )
            or {
                str(reconciled.get("train_episode_id")),
                str(reconciled.get("heldout_episode_id")),
            }
            != set(row["episode_ids"])
            or (reconciled.get("timestamp_independence") or {}).get("verified")
            is not True
        ):
            raise ValueError(
                f"allowlist/reconciliation mismatch for object {object_id}"
            )
    return actual


def _validate_declared_named_files(
    declared_rows: Any, root: Path, expected_names: set[str], label: str
) -> list[dict[str, Any]]:
    rows = _sequence(declared_rows, label)
    by_name: dict[str, Mapping[str, Any]] = {}
    for row in rows:
        if not isinstance(row, Mapping):
            raise ValueError(f"{label} row is not an object")
        name = str(row.get("name") or Path(str(row.get("path") or "")).name)
        if name in by_name:
            raise ValueError(f"{label} contains duplicate file {name!r}")
        by_name[name] = row
    if set(by_name) != expected_names:
        raise ValueError(f"{label} file-name set mismatch")
    result: list[dict[str, Any]] = []
    for name in sorted(expected_names):
        actual = file_provenance(root / name)
        declared = dict(by_name[name])
        declared.setdefault("path", actual["path"])
        _validate_declared_provenance(declared, actual, f"{label} {name}")
        result.append({"name": name, **actual})
    return result


def _official_repo_provenance(repository: Path) -> dict[str, Any]:
    root = Path(repository).expanduser().resolve(strict=True)
    files = [
        file_provenance(root / relative) | {"relative_path": relative}
        for relative in _GG_SOURCE_FILES
    ]
    try:
        revision = subprocess.check_output(
            ["git", "-C", str(root), "rev-parse", "HEAD"],
            text=True,
            stderr=subprocess.DEVNULL,
        ).strip()
    except (OSError, subprocess.CalledProcessError) as exc:
        raise ValueError(
            "Gaussian Grouping repository has no readable git revision"
        ) from exc
    return {"path": str(root), "git_revision": revision, "source_files": files}


def build_anchor_dataset(
    *,
    episodes_root: Path,
    allowlist_path: Path,
    colmap_model: Path,
    source_gaussians: Path,
    gaussian_grouping_repo: Path,
    output_root: Path,
    sparse_points_ply: Path | None = None,
) -> dict[str, Any]:
    """Validate and atomically materialize one exact anchor allowlist."""

    allowlist_path = Path(allowlist_path).expanduser().resolve(strict=True)
    allowlist, mapping, allowlist_episodes, object_rows = _validate_allowlist_contract(
        allowlist_path
    )
    episode_ids = {str(row["episode_id"]) for row in allowlist_episodes}
    episodes_root = Path(episodes_root).expanduser().resolve(strict=True)
    episodes_manifest, frames = _load_episode_rows(episodes_root, episode_ids)
    manifest_path = episodes_root / "manifest.json"
    reconciliation_provenance = _validate_reconciliation(
        allowlist, object_rows, manifest_path
    )
    by_episode: dict[str, list[dict[str, Any]]] = {}
    for frame in frames:
        by_episode.setdefault(str(frame["episode_id"]), []).append(frame)
    if set(by_episode) != episode_ids:
        raise ValueError("materialized episode selection differs from allowlist")

    declared_episode_by_id = {str(row["episode_id"]): row for row in allowlist_episodes}
    train_timestamps: set[str] = set()
    heldout_timestamps: set[str] = set()
    for episode_id, episode_frames in by_episode.items():
        declared = declared_episode_by_id[episode_id]
        split = str(episode_frames[0]["split"])
        timestamps = [str(row["physical_timestamp"]) for row in episode_frames]
        if (
            declared.get("split") != split
            or _integer(
                declared.get("frame_count"), "allowlist episode frame_count", minimum=1
            )
            != len(episode_frames)
            or declared.get("physical_timestamps") != timestamps
            or _integer(
                declared.get("canonical_object_id"),
                "allowlist episode object ID",
                minimum=1,
            )
            not in {int(row["canonical_object_id"]) for row in object_rows}
        ):
            raise ValueError(f"allowlist episode/frame contract mismatch: {episode_id}")
        _validate_declared_provenance(
            declared.get("episode_json"),
            file_provenance(Path(str(episode_frames[0]["episode_json"]))),
            f"allowlist episode {episode_id}",
        )
        (train_timestamps if split == "train" else heldout_timestamps).update(
            timestamps
        )
    leaked = sorted(train_timestamps & heldout_timestamps)
    if leaked:
        raise ValueError(
            "physical timestamp leakage across train/heldout: " + ", ".join(leaked[:8])
        )

    inputs = allowlist["inputs"]
    colmap_root = Path(colmap_model).expanduser().resolve(strict=True)
    if str(colmap_root) != str(
        Path(str(inputs.get("colmap_model") or "")).expanduser().resolve(strict=True)
    ):
        raise ValueError("CLI COLMAP model differs from allowlist source")
    colmap_contract = _validate_declared_named_files(
        inputs.get("colmap_contract_files"),
        colmap_root,
        _COLMAP_CONTRACT_NAMES,
        "allowlist COLMAP",
    )
    source_gaussians = Path(source_gaussians).expanduser().resolve(strict=True)
    _validate_declared_provenance(
        inputs.get("source_gaussians"),
        file_provenance(source_gaussians),
        "allowlist source Gaussians",
    )
    points_source = (
        Path(sparse_points_ply or colmap_root / "points3D.ply")
        .expanduser()
        .resolve(strict=True)
    )
    _validate_declared_provenance(
        inputs.get("sparse_points_ply"),
        file_provenance(points_source),
        "allowlist sparse points",
    )
    selected_names = {str(row.get("source_name") or "") for row in frames}
    if "" in selected_names or len(selected_names) != len(frames):
        raise ValueError("selected frames contain empty or duplicate source names")
    cameras, images_by_name, _ = _read_colmap_binary(colmap_root, selected_names)

    observed_keys: set[str] = set()
    mask_inputs: dict[tuple[str, int], tuple[np.ndarray, dict[str, Any]]] = {}
    signature_inputs: list[dict[str, Any]] = []
    measurement_inputs: list[dict[str, Any]] = []
    for episode_id in sorted(episode_ids):
        declared_episode = declared_episode_by_id[episode_id]
        episode_frames = by_episode[episode_id]
        signature_path = (
            Path(str((declared_episode.get("signature") or {}).get("path") or ""))
            .expanduser()
            .resolve(strict=True)
        )
        signature_provenance = file_provenance(signature_path)
        _validate_declared_provenance(
            declared_episode.get("signature"),
            signature_provenance,
            f"allowlist signature {episode_id}",
        )
        signature = _load_json(signature_path, f"signature {episode_id}")
        signature_inputs_raw = signature.get("inputs")
        if (
            signature.get("schema") != SIGNATURE_SCHEMA
            or signature.get("status") != "pass"
            or (signature.get("input_contract") or {}).get("passed") is not True
            or not isinstance(signature_inputs_raw, Mapping)
        ):
            raise ValueError(f"unsupported/failed signature: {episode_id}")
        signature_inputs_map: Mapping[str, Any] = signature_inputs_raw
        if (
            signature_inputs_map.get("episode_id") != episode_id
            or signature_inputs_map.get("split") != episode_frames[0]["split"]
            or signature_inputs_map.get("camera") != episode_frames[0]["camera"]
            or signature_inputs_map.get("view_family")
            != episode_frames[0]["view_family"]
            or Path(str(signature_inputs_map.get("colmap_model") or ""))
            .expanduser()
            .resolve(strict=True)
            != colmap_root
        ):
            raise ValueError(f"signature episode/COLMAP mismatch: {episode_id}")
        _validate_declared_provenance(
            signature_inputs_map.get("episode"),
            file_provenance(Path(str(episode_frames[0]["episode_json"]))),
            f"signature episode {episode_id}",
        )

        measurement_path = (
            Path(
                str(
                    (declared_episode.get("tracker_measurement") or {}).get("path")
                    or ""
                )
            )
            .expanduser()
            .resolve(strict=True)
        )
        measurement_provenance = file_provenance(measurement_path)
        _validate_declared_provenance(
            declared_episode.get("tracker_measurement"),
            measurement_provenance,
            f"allowlist tracker measurement {episode_id}",
        )
        measurement = _load_json(measurement_path, f"tracker measurement {episode_id}")
        measurement_input = measurement.get("input")
        measurement_output = measurement.get("output")
        if (
            measurement.get("schema") != MEASUREMENT_SCHEMA
            or measurement.get("status") != "pass"
            or not isinstance(measurement_input, Mapping)
            or not isinstance(measurement_output, Mapping)
            or measurement_input.get("episode_id") != episode_id
            or measurement_input.get("split") != episode_frames[0]["split"]
            or measurement_input.get("camera") != episode_frames[0]["camera"]
            or measurement_input.get("view_family") != episode_frames[0]["view_family"]
            or _integer(
                measurement_input.get("frame_count"),
                "measurement frame_count",
                minimum=1,
            )
            != len(episode_frames)
            or measurement_input.get("physical_timestamps")
            != [str(row["physical_timestamp"]) for row in episode_frames]
            or str(measurement_input.get("episode_json_sha256") or "")
            != sha256_file(Path(str(episode_frames[0]["episode_json"])))
        ):
            raise ValueError(f"tracker measurement input mismatch: {episode_id}")
        annotation_root = (
            Path(str(signature_inputs_map.get("mask_directory") or ""))
            .expanduser()
            .resolve(strict=True)
        )
        tracker_output_root = (
            Path(str(measurement_output.get("directory") or ""))
            .expanduser()
            .resolve(strict=True)
        )
        if annotation_root != tracker_output_root / "Annotations":
            raise ValueError(
                f"signature/measurement mask directory mismatch: {episode_id}"
            )
        expected_names = {str(row["materialized_name"]) for row in episode_frames}
        actual_names = {
            path.name for path in annotation_root.iterdir() if path.is_file()
        }
        if actual_names != expected_names:
            raise ValueError(f"tracker mask name set mismatch: {episode_id}")
        declared_measurement_masks = measurement_output.get("masks")
        declared_signature_masks = signature_inputs_map.get("mask_files")
        if not isinstance(declared_measurement_masks, list) or not isinstance(
            declared_signature_masks, list
        ):
            raise ValueError(f"tracker mask provenance is malformed: {episode_id}")
        measurement_by_name = {
            str(row.get("name") or ""): row
            for row in declared_measurement_masks
            if isinstance(row, Mapping)
        }
        signature_by_name = {
            Path(str(row.get("path") or "")).name: row
            for row in declared_signature_masks
            if isinstance(row, Mapping)
        }
        if (
            len(measurement_by_name) != len(declared_measurement_masks)
            or len(signature_by_name) != len(declared_signature_masks)
            or set(measurement_by_name) != expected_names
            or set(signature_by_name) != expected_names
            or _integer(measurement_output.get("mask_count"), "measurement mask_count")
            != len(expected_names)
        ):
            raise ValueError(f"tracker mask provenance coverage mismatch: {episode_id}")
        for frame in episode_frames:
            image = images_by_name[str(frame["source_name"])]
            if (
                int(frame.get("colmap_image_id", -1)) != image.image_id
                or int(frame.get("colmap_camera_id", -1)) != image.camera_id
            ):
                raise ValueError(
                    f"episode/COLMAP frame mismatch: {frame['source_name']}"
                )
            camera = cameras[image.camera_id]
            mask_path = annotation_root / str(frame["materialized_name"])
            mask = _mask_array(mask_path, camera.width, camera.height)
            provenance = file_provenance(mask_path)
            provenance["shape_hw"] = [int(mask.shape[0]), int(mask.shape[1])]
            measurement_row = dict(measurement_by_name[mask_path.name])
            measurement_row["path"] = provenance["path"]
            _validate_declared_provenance(
                measurement_row,
                provenance,
                f"measurement mask {episode_id}/{mask_path.name}",
            )
            _validate_declared_provenance(
                signature_by_name[mask_path.name],
                provenance,
                f"signature mask {episode_id}/{mask_path.name}",
            )
            if (
                list(signature_by_name[mask_path.name].get("shape_hw") or [])
                != provenance["shape_hw"]
            ):
                raise ValueError(
                    f"signature mask shape mismatch: {episode_id}/{mask_path.name}"
                )
            for local_id in np.unique(mask):
                if int(local_id):
                    observed_keys.add(f"{episode_id}::local:{int(local_id)}")
            mask_inputs[(episode_id, int(frame["episode_index"]))] = (mask, provenance)
        identities = signature.get("local_identities")
        if not isinstance(identities, list):
            raise ValueError(f"signature identities are missing: {episode_id}")
        identity_by_key: dict[str, Mapping[str, Any]] = {}
        for identity in identities:
            if not isinstance(identity, Mapping):
                raise ValueError(f"signature identity is malformed: {episode_id}")
            key = str(identity.get("identity_key") or "")
            parsed_episode, local_id = _parse_identity_key(key)
            if (
                key in identity_by_key
                or parsed_episode != episode_id
                or identity.get("episode_id") != episode_id
                or identity.get("local_id") != local_id
            ):
                raise ValueError(
                    f"signature identity provenance mismatch: {episode_id}"
                )
            identity_by_key[key] = identity
        episode_mapping_keys = {
            key for key in mapping if _parse_identity_key(key)[0] == episode_id
        }
        if set(identity_by_key) != episode_mapping_keys:
            raise ValueError(
                f"allowlist/signature identity coverage mismatch: {episode_id}"
            )
        for key in episode_mapping_keys:
            if mapping[key] > 0:
                shadow = identity_by_key[key].get("decision_v2_shadow")
                if (
                    not isinstance(shadow, Mapping)
                    or shadow.get("schema") != SHADOW_SCHEMA
                    or shadow.get("mode") != "diagnostic-only-no-publish"
                    or shadow.get("status") != "would_pass_shadow_gate"
                    or shadow.get("passed") is not True
                    or shadow.get("global_association_authorized") is not False
                ):
                    raise ValueError(
                        f"selected identity lacks an exact passing shadow decision: {key}"
                    )
        signature_inputs.append(signature_provenance | {"episode_id": episode_id})
        measurement_inputs.append(measurement_provenance | {"episode_id": episode_id})
    if observed_keys != set(mapping):
        missing = sorted(observed_keys.difference(mapping))
        extra = sorted(set(mapping).difference(observed_keys))
        raise ValueError(
            f"allowlist identity coverage mismatch; missing={missing}, extra={extra}"
        )

    split_counts = {
        split: sum(str(row["split"]) == split for row in frames)
        for split in ("train", "heldout")
    }
    if min(split_counts.values()) <= 0:
        raise ValueError("both train and heldout frames are required")
    used_ids = sorted(set(mapping.values()).difference({0}))
    repository_provenance = _official_repo_provenance(gaussian_grouping_repo)
    destination = Path(output_root).expanduser().resolve()
    if destination.exists():
        raise FileExistsError(
            f"refusing to overwrite Gaussian Grouping anchor dataset: {destination}"
        )
    destination.parent.mkdir(parents=True, exist_ok=True)
    staging = Path(
        tempfile.mkdtemp(prefix=f".{destination.name}.tmp-", dir=destination.parent)
    )
    output_rows: list[dict[str, Any]] = []
    try:
        for directory in (
            "images",
            "images_train",
            "images_heldout",
            "object_mask",
            "sparse/0",
        ):
            (staging / directory).mkdir(parents=True, exist_ok=True)
        _safe_symlink(source_gaussians, staging / "source_gaussians.ply")
        _safe_symlink(points_source, staging / "sparse/0/points3D.ply")
        for frame in sorted(frames, key=lambda row: str(row["source_name"])):
            source, rgb_provenance = _validate_source_frame(frame)
            source_name = str(frame["source_name"])
            _safe_symlink(source, staging / "images" / source_name)
            _safe_symlink(source, staging / f"images_{frame['split']}" / source_name)
            episode_id = str(frame["episode_id"])
            mask, tracker_provenance = mask_inputs[
                (episode_id, int(frame["episode_index"]))
            ]
            lut = np.zeros(256, dtype=np.uint8)
            for local_id in np.unique(mask):
                if int(local_id):
                    lut[int(local_id)] = mapping[f"{episode_id}::local:{int(local_id)}"]
            remapped = lut[mask]
            mask_relative = f"object_mask/{Path(source_name).stem}.png"
            mask_destination = staging / mask_relative
            Image.fromarray(remapped, mode="L").save(
                mask_destination, format="PNG", optimize=False
            )
            output_rows.append(
                {
                    "episode_id": episode_id,
                    "split": frame["split"],
                    "physical_timestamp": str(frame["physical_timestamp"]),
                    "source_name": source_name,
                    "colmap_image_id": int(frame["colmap_image_id"]),
                    "colmap_camera_id": int(frame["colmap_camera_id"]),
                    "rgb": rgb_provenance,
                    "tracker_mask": tracker_provenance,
                    "object_mask_relative": mask_relative,
                    "object_mask_sha256": sha256_file(mask_destination),
                    "foreground_global_ids": [
                        int(value) for value in np.unique(remapped) if value
                    ],
                }
            )
        cameras_text, images_text = render_colmap_text(
            cameras, [images_by_name[name] for name in selected_names]
        )
        cameras_output = staging / "sparse/0/cameras.txt"
        images_output = staging / "sparse/0/images.txt"
        cameras_output.write_text(cameras_text, encoding="utf-8")
        images_output.write_text(images_text, encoding="utf-8")
        official_config = {
            "densify_until_iter": 0,
            "num_classes": max(used_ids) + 1,
            "reg3d_interval": 2,
            "reg3d_k": 5,
            "reg3d_lambda_val": 2,
            "reg3d_max_points": 300000,
            "reg3d_sample_size": 1000,
        }
        config_output = staging / "official_gaussian_grouping_config.json"
        config_output.write_text(
            json.dumps(official_config, indent=2, sort_keys=True, allow_nan=False)
            + "\n",
            encoding="utf-8",
        )
        manifest = {
            "schema": OUTPUT_SCHEMA,
            "status": "ready",
            "mode": "bounded-anchor-allowlist-identity-only",
            "publication": {
                "production_farm_identity_authorized": False,
                "production_gaussian_grouping_training_authorized": False,
                "bounded_identity_only_pilot_authorized": True,
                "raw_tracker_outputs_changed": False,
                "shadow_gate_promoted_to_v1": False,
            },
            "dataset_contract": {
                "consumer": "Gaussian Grouping ECCV 2024 COLMAP layout plus FARM frozen identity-only runner",
                "foreground": "only exact allowlist identities; compact IDs 1..N",
                "background": "all other observed episode-local IDs explicitly map to 0/drop",
                "train_membership": "images_train exact basenames; fitting source",
                "heldout_membership": "images_heldout exact basenames; evaluation only, never fitting",
                "source_gaussians": "relative symlink; geometry/appearance must remain frozen",
                "direct_official_train_py_authorized": False,
            },
            "licenses": {
                "gaussian_grouping_top_level": "Apache-2.0 repository license",
                "diff_gaussian_rasterization": "Gaussian-Splatting non-commercial research/evaluation license",
                "commercial_use": "not cleared",
            },
            "artifacts": {
                "cameras_txt": {
                    "relative_path": "sparse/0/cameras.txt",
                    "bytes": cameras_output.stat().st_size,
                    "sha256": sha256_file(cameras_output),
                },
                "images_txt": {
                    "relative_path": "sparse/0/images.txt",
                    "bytes": images_output.stat().st_size,
                    "sha256": sha256_file(images_output),
                },
                "official_config": {
                    "relative_path": config_output.name,
                    "bytes": config_output.stat().st_size,
                    "sha256": sha256_file(config_output),
                },
            },
            "inputs": {
                "allowlist": file_provenance(allowlist_path),
                "allowlist_schema": ALLOWLIST_SCHEMA,
                "reconciliation": reconciliation_provenance,
                "episodes_manifest": file_provenance(manifest_path),
                "source_plan": file_provenance(
                    Path(str(episodes_manifest["source_plan"]))
                ),
                "tracker_signatures": sorted(
                    signature_inputs, key=lambda row: row["episode_id"]
                ),
                "tracker_measurements": sorted(
                    measurement_inputs, key=lambda row: row["episode_id"]
                ),
                "colmap_model": str(colmap_root),
                "colmap_contract_files": colmap_contract,
                "sparse_points_ply": file_provenance(points_source),
                "source_gaussians": file_provenance(source_gaussians),
                "gaussian_grouping_repository": repository_provenance,
            },
            "identity_mapping": {
                "canonical_objects": object_rows,
                "selected_foreground_identity_keys": sorted(
                    key for key, value in mapping.items() if value
                ),
                "explicit_dropped_background_identity_keys": sorted(
                    key for key, value in mapping.items() if value == 0
                ),
                "identity_key_to_compact_gg_id": dict(sorted(mapping.items())),
            },
            "summary": {
                "frame_count": len(output_rows),
                "episode_count": len(episode_ids),
                "canonical_object_count": len(object_rows),
                "frames_by_split": split_counts,
                "physical_timestamps_by_split": {
                    "train": len(train_timestamps),
                    "heldout": len(heldout_timestamps),
                },
                "physical_timestamp_leakage_count": 0,
                "observed_episode_local_identity_count": len(mapping),
                "accepted_scene_global_identity_count": len(used_ids),
                "dropped_episode_local_identity_count": sum(
                    value == 0 for value in mapping.values()
                ),
                "num_classes_including_background": max(used_ids) + 1,
                "maximum_scene_global_id": max(used_ids),
            },
            "frames": output_rows,
        }
        (staging / "manifest.json").write_text(
            json.dumps(
                manifest, ensure_ascii=False, indent=2, sort_keys=True, allow_nan=False
            )
            + "\n",
            encoding="utf-8",
        )
        os.replace(staging, destination)
        return manifest
    except BaseException:
        shutil.rmtree(staging, ignore_errors=True)
        raise
