"""Scene-state reference remapping for canonicalized source views."""

from __future__ import annotations

import copy
import re
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np
from farm_runtime.assembly_mask_reconstruction import (
    AssemblyMaskReconstruction,
    load_assembly_mask_payload,
    reconstruction_observation_metadata,
    reconstruct_assembly_mask,
    write_reconstructed_assembly_mask,
)

from farm_runtime.source_view_canonicalization import (
    SourceCanonicalization,
    _sha256,
)


def _object_ids(state: Mapping[str, Any], expected: int) -> list[int]:
    value = state.get("object_id")
    if hasattr(value, "detach"):
        value = value.detach()
    if hasattr(value, "cpu"):
        value = value.cpu()
    if hasattr(value, "numpy"):
        value = value.numpy()
    identifiers = np.asarray(value).astype(np.int64).reshape(-1).tolist()
    if len(identifiers) != expected or len(identifiers) != len(set(identifiers)):
        raise ValueError("source object IDs are not unique/object-aligned")
    return [int(value) for value in identifiers]


def _stable_remap(values: object, old_to_new: Sequence[int], field: str) -> list[int]:
    if values is None:
        return []
    if not isinstance(values, (list, tuple, set)):
        raise ValueError(f"{field} row must be a sequence")
    result: list[int] = []
    seen: set[int] = set()
    for raw in values:
        try:
            old_id = int(raw)
        except (TypeError, ValueError) as exc:
            raise ValueError(f"{field} contains an invalid image ID") from exc
        if old_id < 0 or old_id >= len(old_to_new):
            raise ValueError(f"{field} references image outside source frames: {old_id}")
        new_id = int(old_to_new[old_id])
        if new_id not in seen:
            result.append(new_id)
            seen.add(new_id)
    return result


def _mask_path(
    mask_root: Path, object_id: int, observation: Mapping[str, Any]
) -> Path:
    basename = Path(str(observation.get("path") or "")).name
    if not basename or basename in {".", ".."}:
        raise ValueError("duplicate-view mask observation has no safe basename")
    path = (mask_root / f"object_{object_id:06d}" / basename).resolve(strict=True)
    if not path.is_file() or not path.is_relative_to(mask_root):
        raise ValueError("duplicate-view mask artifact escapes the source mask root")
    return path


def _mask_sha(mask_root: Path, object_id: int, observation: Mapping[str, Any]) -> str:
    return _sha256(_mask_path(mask_root, object_id, observation))


def _canonical_mask_basename(original: str, image_id: int) -> str:
    match = re.fullmatch(
        r"img_[0-9]+(?P<tail>_(?:det_[0-9]+|assembly)[.]npz)", original
    )
    if match is None:
        raise ValueError(
            f"mask basename does not encode authoritative image/detection IDs: {original}"
        )
    return f"img_{image_id:06d}{match.group('tail')}"


def _resolve_mask_artifact(
    state: Mapping[str, Any],
    *,
    object_id: int,
    old_image_id: int,
    observation: Mapping[str, Any],
    mask_root: Path,
    reconstruction_cache: dict[
        tuple[int, int, tuple[int, ...]], AssemblyMaskReconstruction
    ],
) -> dict[str, Any]:
    basename = Path(str(observation.get("path") or "")).name
    if not basename or basename in {".", ".."}:
        raise ValueError("mask observation has no safe basename")
    candidate = mask_root / f"object_{object_id:06d}" / basename
    resolved = candidate.resolve(strict=False)
    if not resolved.is_relative_to(mask_root):
        raise ValueError("mask observation escapes source mask root")
    if candidate.is_symlink():
        raise ValueError("mask source must not be a symlink")
    if resolved.exists():
        if not resolved.is_file():
            raise ValueError("mask source must be a regular file")
        assembly_payload = None
        logical_sha256 = _sha256(resolved)
        if re.fullmatch(r"img_[0-9]+_assembly[.]npz", basename) is not None:
            assembly_payload = load_assembly_mask_payload(resolved)
            logical_sha256 = assembly_payload.logical_sha256
        return {
            "kind": "direct",
            "logical_sha256": logical_sha256,
            "source_path": resolved,
            "assembly_payload": assembly_payload,
        }
    if re.fullmatch(r"img_[0-9]+_assembly[.]npz", basename) is None:
        raise FileNotFoundError(f"source mask artifact is missing: {resolved}")
    member_ids = observation.get("assembly_member_ids_in_frame")
    try:
        member_key = tuple(sorted(int(value) for value in member_ids))
    except (TypeError, ValueError) as exc:
        raise ValueError(
            "missing assembly mask has invalid assembly_member_ids_in_frame"
        ) from exc
    cache_key = (object_id, old_image_id, member_key)
    reconstruction = reconstruction_cache.get(cache_key)
    if reconstruction is None:
        reconstruction = reconstruct_assembly_mask(
            state,
            assembly_object_id=object_id,
            old_image_id=old_image_id,
            assembly_observation=observation,
            source_mask_root=mask_root,
        )
        reconstruction_cache[cache_key] = reconstruction
    return {
        "kind": "reconstructed_assembly",
        "logical_sha256": reconstruction.logical_sha256,
        "reconstruction": reconstruction,
        "source_relative": str(candidate.relative_to(mask_root)),
    }


def _remap_mask_row(
    rows: Sequence[Mapping[str, Any]],
    *,
    state: Mapping[str, Any],
    object_id: int,
    old_to_new: Sequence[int],
    mask_root: Path,
    staging_mask_root: Path | None,
    reconstruction_cache: dict[
        tuple[int, int, tuple[int, ...]], AssemblyMaskReconstruction
    ],
    staged_logical_sha: dict[str, str],
) -> tuple[
    list[dict[str, Any]],
    list[dict[str, Any]],
    list[dict[str, Any]],
    int,
]:
    groups: dict[int, dict[int, list[tuple[int, dict[str, Any]]]]] = {}
    artifacts: dict[int, dict[str, Any]] = {}
    remapped_count = 0
    for ordinal, raw in enumerate(rows):
        if not isinstance(raw, Mapping):
            raise ValueError("object_mask_observations contains a non-object")
        observation = copy.deepcopy(dict(raw))
        if "image_id" not in observation:
            raise ValueError(
                "object_mask_observations row has no authoritative image_id"
            )
        try:
            old_id = int(observation["image_id"])
        except (TypeError, ValueError) as exc:
            raise ValueError(
                "object_mask_observations contains an invalid image ID"
            ) from exc
        if old_id < 0 or old_id >= len(old_to_new):
            raise ValueError(
                "object_mask_observations references image outside source frames"
            )
        artifacts[ordinal] = _resolve_mask_artifact(
            state,
            object_id=object_id,
            old_image_id=old_id,
            observation=observation,
            mask_root=mask_root,
            reconstruction_cache=reconstruction_cache,
        )
        new_id = int(old_to_new[old_id])
        remapped_count += int(new_id != old_id)
        observation["image_id"] = new_id
        groups.setdefault(new_id, {}).setdefault(old_id, []).append(
            (ordinal, observation)
        )

    selected: list[tuple[int, dict[str, Any]]] = []
    collapsed: list[dict[str, Any]] = []
    for new_id, by_old_id in groups.items():
        ordered_old_ids = sorted(
            by_old_id, key=lambda old_id: min(item[0] for item in by_old_id[old_id])
        )
        kept_old_id = ordered_old_ids[0]
        kept_rows = by_old_id[kept_old_id]
        selected.extend(kept_rows)
        if len(ordered_old_ids) == 1:
            continue
        kept_hashes = sorted(
            str(artifacts[ordinal]["logical_sha256"])
            for ordinal, _ in kept_rows
        )
        for duplicate_old_id in ordered_old_ids[1:]:
            duplicate_hashes = sorted(
                str(artifacts[ordinal]["logical_sha256"])
                for ordinal, _ in by_old_id[duplicate_old_id]
            )
            if duplicate_hashes != kept_hashes:
                raise ValueError(
                    "same-object mask observations collapse onto one source view "
                    "but logical mask payloads are not exactly equivalent"
                )
        collapsed.append(
            {
                "object_id": object_id,
                "canonical_image_id": new_id,
                "kept_old_image_id": kept_old_id,
                "collapsed_old_image_ids": ordered_old_ids[1:],
                "mask_logical_sha256": kept_hashes,
                "collapsed_observations": sum(
                    len(by_old_id[value]) for value in ordered_old_ids[1:]
                ),
            }
        )

    selected.sort(key=lambda item: item[0])
    rewritten: list[dict[str, Any]] = []
    aliases: list[dict[str, Any]] = []
    for ordinal, observation in selected:
        artifact = artifacts[ordinal]
        original_basename = Path(str(observation["path"])).name
        canonical_image_id = int(observation["image_id"])
        canonical_basename = _canonical_mask_basename(
            original_basename, canonical_image_id
        )
        canonical_relative = f"object_{object_id:06d}/{canonical_basename}"
        observation["path"] = (
            "/farm-run/qa/full_colmap_rescue/combined/masks/"
            f"{canonical_relative}"
        )
        metadata_payload = artifact.get("assembly_payload")
        if metadata_payload is None:
            metadata_payload = artifact.get("reconstruction")
        corrections: dict[str, dict[str, Any]] = {}
        if isinstance(metadata_payload, AssemblyMaskReconstruction):
            exact_metadata = reconstruction_observation_metadata(metadata_payload)
            for field, exact_value in exact_metadata.items():
                prior_value = observation.get(field)
                if prior_value != exact_value:
                    corrections[field] = {
                        "previous": copy.deepcopy(prior_value),
                        "reconstructed": copy.deepcopy(exact_value),
                    }
                observation[field] = copy.deepcopy(exact_value)
        if artifact["kind"] == "direct":
            source_path = Path(artifact["source_path"])
            aliases.append(
                {
                    "kind": "direct",
                    "object_id": object_id,
                    "canonical_image_id": canonical_image_id,
                    "source_path": str(source_path),
                    "source_relative": str(source_path.relative_to(mask_root)),
                    "source_basename": source_path.name,
                    "canonical_relative": canonical_relative,
                    "logical_sha256": str(artifact["logical_sha256"]),
                    "sha256": _sha256(source_path),
                    "observation_metadata_corrections": corrections,
                }
            )
        else:
            if staging_mask_root is None:
                raise ValueError(
                    "missing assembly mask requires an explicit staging mask root"
                )
            reconstruction = artifact["reconstruction"]
            if not isinstance(reconstruction, AssemblyMaskReconstruction):
                raise TypeError("invalid cached assembly reconstruction")
            prior_digest = staged_logical_sha.get(canonical_relative)
            if (
                prior_digest is not None
                and prior_digest != reconstruction.logical_sha256
            ):
                raise ValueError(
                    "canonical reconstructed assembly mask has conflicting payloads"
                )
            destination = staging_mask_root / canonical_relative
            if prior_digest is None:
                write_reconstructed_assembly_mask(
                    reconstruction, destination, staging_mask_root
                )
                staged_logical_sha[canonical_relative] = (
                    reconstruction.logical_sha256
                )
            aliases.append(
                {
                    "kind": "reconstructed_assembly",
                    "object_id": object_id,
                    "canonical_image_id": canonical_image_id,
                    "source_relative": str(artifact["source_relative"]),
                    "source_basename": original_basename,
                    "canonical_relative": canonical_relative,
                    "logical_sha256": reconstruction.logical_sha256,
                    "sha256": _sha256(destination),
                    "member_provenance": [
                        dict(row) for row in reconstruction.member_provenance
                    ],
                    "observation_metadata_corrections": corrections,
                    "materialized_in_staging": True,
                }
            )
        rewritten.append(observation)
    return rewritten, collapsed, aliases, remapped_count


def remap_source_state(
    state: dict[str, Any],
    canonicalization: SourceCanonicalization,
    source_mask_root: Path,
    staging_mask_root: Path | None = None,
) -> dict[str, Any]:
    """Apply canonical IDs and materialize proven virtual assemblies."""

    images = list(state.get("images") or [])
    positions = list(state.get("image_positions") or [])
    old_count = len(canonicalization.old_to_new)
    if len(images) != old_count or len(positions) != old_count:
        raise ValueError("source state images/image_positions are not frame-aligned")
    state["images"] = [
        copy.deepcopy(images[index])
        for index in canonicalization.canonical_old_indices
    ]
    state["image_positions"] = [
        copy.deepcopy(positions[index])
        for index in canonicalization.canonical_old_indices
    ]
    audit: dict[str, Any] = {
        "object_image_ids_collapsed": 0,
        "viewpoint_image_ids_collapsed": 0,
        "mask_observations_remapped": 0,
        "mask_observation_duplicate_groups_collapsed": 0,
        "mask_observations_collapsed": 0,
        "mask_observation_equivalence": [],
        "mask_aliases": [],
        "assembly_masks_reconstructed": 0,
        "assembly_mask_reconstructions": [],
    }
    for field in ("object_image_ids", "viewpoint_image_ids"):
        if field not in state:
            continue
        rows = state[field]
        if not isinstance(rows, (list, tuple)):
            raise ValueError(f"source state {field} must be object-aligned")
        remapped: list[list[int]] = []
        for row in rows:
            before = len(row or []) if isinstance(row, (list, tuple, set)) else 0
            values = _stable_remap(row, canonicalization.old_to_new, field)
            audit[f"{field}_collapsed"] += before - len(values)
            remapped.append(values)
        state[field] = remapped

    if "object_mask_observations" in state:
        rows = state["object_mask_observations"]
        if not isinstance(rows, (list, tuple)):
            raise ValueError(
                "source state object_mask_observations must be object-aligned"
            )
        object_ids = _object_ids(state, len(rows))
        mask_root = source_mask_root.expanduser().resolve(strict=True)
        staging_root = (
            staging_mask_root.expanduser().resolve(strict=True)
            if staging_mask_root is not None
            else None
        )
        reconstruction_cache: dict[
            tuple[int, int, tuple[int, ...]], AssemblyMaskReconstruction
        ] = {}
        staged_logical_sha: dict[str, str] = {}
        remapped_rows: list[list[dict[str, Any]]] = []
        for object_index, object_rows in enumerate(rows):
            if not isinstance(object_rows, (list, tuple)):
                raise ValueError("object_mask_observations row must be a sequence")
            updated, collapsed, aliases, remapped_count = _remap_mask_row(
                object_rows,
                state=state,
                object_id=object_ids[object_index],
                old_to_new=canonicalization.old_to_new,
                mask_root=mask_root,
                staging_mask_root=staging_root,
                reconstruction_cache=reconstruction_cache,
                staged_logical_sha=staged_logical_sha,
            )
            audit["mask_observations_remapped"] += remapped_count
            audit["mask_observation_duplicate_groups_collapsed"] += len(collapsed)
            audit["mask_observations_collapsed"] += sum(
                int(row["collapsed_observations"]) for row in collapsed
            )
            audit["mask_observation_equivalence"].extend(collapsed)
            audit["mask_aliases"].extend(aliases)
            reconstructed = [
                row
                for row in aliases
                if row.get("kind") == "reconstructed_assembly"
            ]
            audit["assembly_masks_reconstructed"] += len(reconstructed)
            audit["assembly_mask_reconstructions"].extend(
                copy.deepcopy(reconstructed)
            )
            remapped_rows.append(updated)
        state["object_mask_observations"] = remapped_rows
    return audit
