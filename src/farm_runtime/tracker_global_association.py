"""Fail-closed global association for episode-local tracker identities.

The association is deliberately independent of FARM OBBs.  It uses the
metric COLMAP support encoded by ``tracker_3d_signature`` reports, optional
prompt/appearance evidence, and immutable train/heldout provenance.  Heldout
identities are queries against train-only clusters and never participate in
fitting those clusters.
"""

from __future__ import annotations

import hashlib
import json
import math
import re
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

import numpy as np
from PIL import Image

from farm_runtime.tracker_3d_signature import (
    SCHEMA as SIGNATURE_SCHEMA,
    read_selected_points3d,
)

SCHEMA = "farm.tracker-global-association.v1"
SHADOW_ABLATION_SCHEMA = "farm.tracker-global-association.v2-shadow-ablation.v1"
LOCAL_GATE_V1 = "v1_publish"
LOCAL_GATE_V2_SHADOW = "v2_shadow"
IDENTITY_EVIDENCE_SCHEMA = "farm.tracker-identity-evidence.v1"
MAX_GLOBAL_ID = 254
_IDENTITY_RE = re.compile(r"^(?P<episode>[^/]+)::local:(?P<local>[1-9][0-9]*)$")


@dataclass(frozen=True)
class AssociationPolicy:
    voxel_size_m: float = 0.10
    voxel_neighbor_radius: int = 0
    minimum_shared_sparse_tracks: int = 3
    minimum_sparse_track_containment: float = 0.30
    minimum_sparse_track_jaccard: float = 0.10
    minimum_shared_metric_voxels: int = 2
    minimum_metric_voxel_containment: float = 0.30
    minimum_metric_voxel_jaccard: float = 0.10
    maximum_centroid_distance_m: float = 0.75
    maximum_centroid_distance_scale: float = 1.50
    maximum_scale_ratio: float = 3.0
    minimum_appearance_similarity: float = 0.78
    minimum_pair_score: float = 0.58
    minimum_assignment_margin: float = 0.08
    maximum_global_ids: int = MAX_GLOBAL_ID

    def validate(self) -> None:
        positive = (
            "voxel_size_m",
            "maximum_centroid_distance_m",
            "maximum_centroid_distance_scale",
            "maximum_scale_ratio",
        )
        for field in positive:
            value = float(getattr(self, field))
            if not math.isfinite(value) or value <= 0:
                raise ValueError(f"{field} must be finite and positive")
        if float(self.maximum_scale_ratio) < 1.0:
            raise ValueError("maximum_scale_ratio must be at least 1")
        if int(self.voxel_neighbor_radius) < 0:
            raise ValueError("voxel_neighbor_radius must be non-negative")
        for field in (
            "minimum_shared_sparse_tracks",
            "minimum_shared_metric_voxels",
        ):
            if int(getattr(self, field)) < 0:
                raise ValueError(f"{field} must be non-negative")
        for field in (
            "minimum_sparse_track_containment",
            "minimum_sparse_track_jaccard",
            "minimum_metric_voxel_containment",
            "minimum_metric_voxel_jaccard",
            "minimum_appearance_similarity",
            "minimum_pair_score",
            "minimum_assignment_margin",
        ):
            value = float(getattr(self, field))
            if not math.isfinite(value) or not 0 <= value <= 1:
                raise ValueError(f"{field} must be in [0,1]")
        if not 1 <= int(self.maximum_global_ids) <= MAX_GLOBAL_ID:
            raise ValueError(f"maximum_global_ids must be in [1,{MAX_GLOBAL_ID}]")


@dataclass(frozen=True)
class IdentityEvidence:
    identity_key: str
    episode_id: str
    local_id: int
    split: str
    passed_local_gate: bool
    local_failed_checks: tuple[str, ...]
    core_point_ids: frozenset[int]
    metric_voxels: frozenset[tuple[int, int, int]]
    centroid_m: tuple[float, float, float] | None
    scale_diagonal_m: float | None
    visible_frame_indices: frozenset[int]
    physical_timestamps: frozenset[str]
    visible_source_names: frozenset[str]
    prompts: frozenset[str]
    appearance_embedding: tuple[float, ...] | None
    report_index: int


def sha256_file(path: Path, chunk_size: int = 8 * 1024 * 1024) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        while chunk := stream.read(int(chunk_size)):
            digest.update(chunk)
    return digest.hexdigest()


def normalize_prompt(value: object) -> str:
    return " ".join(str(value or "").strip().casefold().split())


def _safe_json_object(path: Path) -> dict[str, Any]:
    source = Path(path).expanduser().resolve(strict=True)
    value = json.loads(source.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"JSON root must be an object: {source}")
    return value


def _file_provenance(path: Path) -> dict[str, Any]:
    source = Path(path).expanduser().resolve(strict=True)
    stat = source.stat()
    return {
        "path": str(source),
        "bytes": int(stat.st_size),
        "sha256": sha256_file(source),
    }


def _identity_parts(identity_key: object) -> tuple[str, int]:
    key = str(identity_key or "")
    match = _IDENTITY_RE.fullmatch(key)
    if match is None:
        raise ValueError(f"invalid episode-local identity key: {key!r}")
    local_id = int(match.group("local"))
    if local_id > MAX_GLOBAL_ID:
        raise ValueError(f"local tracker ID exceeds 8-bit contract: {key!r}")
    return match.group("episode"), local_id


def _finite_vector(
    value: object, *, label: str, length: int | None = None
) -> np.ndarray:
    result = np.asarray(value, dtype=np.float64)
    if result.ndim != 1 or result.size == 0 or not np.isfinite(result).all():
        raise ValueError(f"{label} must be a finite non-empty vector")
    if length is not None and result.size != int(length):
        raise ValueError(f"{label} must have length {length}")
    return result


def load_identity_evidence_files(
    paths: Sequence[Path],
) -> tuple[dict[str, dict[str, Any]], list[dict[str, Any]]]:
    """Load optional prompts/embeddings without allowing silent overwrites."""

    combined: dict[str, dict[str, Any]] = {}
    provenance: list[dict[str, Any]] = []
    embedding_size: int | None = None
    for path in sorted((Path(value) for value in paths), key=lambda value: str(value)):
        source = path.expanduser().resolve(strict=True)
        payload = _safe_json_object(source)
        if payload.get("schema") != IDENTITY_EVIDENCE_SCHEMA:
            raise ValueError(f"unsupported identity evidence schema: {source}")
        rows = payload.get("identities")
        if not isinstance(rows, list):
            raise ValueError(f"identity evidence has no identities list: {source}")
        for row in rows:
            if not isinstance(row, Mapping):
                raise ValueError("identity evidence row must be an object")
            key = str(row.get("identity_key") or "")
            _identity_parts(key)
            prompts_raw = row.get("prompts", [])
            if not isinstance(prompts_raw, list):
                raise ValueError(f"prompts must be a list for {key}")
            prompts = sorted(
                {
                    normalize_prompt(value)
                    for value in prompts_raw
                    if normalize_prompt(value)
                }
            )
            embedding = row.get("appearance_embedding")
            normalized_embedding: list[float] | None = None
            if embedding is not None:
                vector = _finite_vector(
                    embedding, label=f"appearance embedding for {key}"
                )
                norm = float(np.linalg.norm(vector))
                if norm <= 1.0e-12:
                    raise ValueError(f"appearance embedding is zero for {key}")
                if embedding_size is None:
                    embedding_size = int(vector.size)
                elif vector.size != embedding_size:
                    raise ValueError(
                        "appearance embeddings have inconsistent dimensions"
                    )
                normalized_embedding = (vector / norm).tolist()
            incoming = {
                "prompts": prompts,
                "appearance_embedding": normalized_embedding,
            }
            if key in combined and combined[key] != incoming:
                raise ValueError(f"conflicting identity evidence for {key}")
            combined[key] = incoming
        item = _file_provenance(source)
        item["schema"] = payload.get("schema")
        item["identity_count"] = len(rows)
        provenance.append(item)
    return combined, provenance


def merge_identity_evidence(
    *sources: Mapping[str, Mapping[str, Any]],
) -> dict[str, dict[str, Any]]:
    """Merge independent evidence sources without silently replacing vectors."""

    result: dict[str, dict[str, Any]] = {}
    embedding_size: int | None = None
    for source in sources:
        for key in sorted(source):
            _identity_parts(key)
            raw = source[key]
            prompts = {
                normalize_prompt(value)
                for value in raw.get("prompts", [])
                if normalize_prompt(value)
            }
            current = result.setdefault(
                key, {"prompts": [], "appearance_embedding": None}
            )
            current["prompts"] = sorted(set(current["prompts"]).union(prompts))
            incoming = raw.get("appearance_embedding")
            if incoming is None:
                continue
            vector = _finite_vector(incoming, label=f"appearance embedding for {key}")
            norm = float(np.linalg.norm(vector))
            if norm <= 1.0e-12:
                raise ValueError(f"appearance embedding is zero for {key}")
            vector = vector / norm
            if embedding_size is None:
                embedding_size = int(vector.size)
            elif vector.size != embedding_size:
                raise ValueError("appearance embeddings have inconsistent dimensions")
            existing = current["appearance_embedding"]
            if existing is not None and not np.allclose(
                np.asarray(existing, dtype=np.float64),
                vector,
                rtol=1.0e-8,
                atol=1.0e-10,
            ):
                raise ValueError(f"conflicting appearance embeddings for {key}")
            current["appearance_embedding"] = vector.tolist()
    return result


def load_tracker_measurements(
    paths: Sequence[Path],
    *,
    allowed_identity_keys: set[str] | None = None,
) -> tuple[dict[str, dict[str, Any]], list[dict[str, Any]]]:
    """Extract proposal prompts from official FARM SAM3 episode measurements.

    DEVA measurements are accepted as provenance but intentionally provide no
    semantic evidence. Automatic-mask numeric IDs carry no category meaning.
    """

    evidence: dict[str, dict[str, Any]] = {}
    provenance: list[dict[str, Any]] = []
    seen_sam3_episodes: set[str] = set()
    for raw_path in sorted(
        (Path(value) for value in paths), key=lambda value: str(value)
    ):
        source = raw_path.expanduser().resolve(strict=True)
        payload = _safe_json_object(source)
        schema = str(payload.get("schema") or "")
        if schema not in {
            "farm.sam3-concept-episode.v1",
            "farm.deva-automatic-episode.v1",
        }:
            raise ValueError(f"unsupported tracker measurement schema: {source}")
        if payload.get("status") != "pass":
            raise ValueError(f"tracker measurement is not a passing run: {source}")
        item = _file_provenance(source)
        item["schema"] = schema
        prompt_rows = 0
        if schema == "farm.sam3-concept-episode.v1":
            input_row = payload.get("input")
            output_row = payload.get("output")
            if not isinstance(input_row, Mapping) or not isinstance(
                output_row, Mapping
            ):
                raise ValueError(f"malformed SAM3 tracker measurement: {source}")
            image_directory = Path(str(input_row.get("image_directory") or ""))
            episode_id = str(input_row.get("episode_id") or image_directory.parent.name)
            if not episode_id or Path(episode_id).name != episode_id:
                raise ValueError(
                    f"cannot resolve episode ID from tracker measurement: {source}"
                )
            if episode_id in seen_sam3_episodes:
                raise ValueError(f"duplicate SAM3 measurement episode: {episode_id}")
            seen_sam3_episodes.add(episode_id)
            identity_rows = output_row.get("identities")
            if isinstance(identity_rows, list):
                object_groups = [identity_rows]
            else:
                frames = output_row.get("frames")
                if not isinstance(frames, list):
                    raise ValueError(
                        f"SAM3 measurement has no identities/frames: {source}"
                    )
                object_groups = []
                for frame in frames:
                    if not isinstance(frame, Mapping):
                        raise ValueError("SAM3 output frame must be an object")
                    objects = frame.get("objects", [])
                    if not isinstance(objects, list):
                        raise ValueError("SAM3 output objects must be a list")
                    object_groups.append(objects)
            for objects in object_groups:
                for obj in objects:
                    if not isinstance(obj, Mapping):
                        raise ValueError("SAM3 output object must be an object")
                    if obj.get("local_id") is None:
                        continue
                    local_id = int(obj.get("local_id", 0))
                    prompt = normalize_prompt(obj.get("prompt"))
                    if not 1 <= local_id <= MAX_GLOBAL_ID:
                        raise ValueError("SAM3 output object lacks valid local_id")
                    if not prompt:
                        continue
                    key = f"{episode_id}::local:{local_id}"
                    row = evidence.setdefault(
                        key, {"prompts": [], "appearance_embedding": None}
                    )
                    row["prompts"] = sorted(set(row["prompts"]).union({prompt}))
                    prompt_rows += 1
            for key, row in evidence.items():
                if key.startswith(f"{episode_id}::local:") and len(row["prompts"]) > 1:
                    raise ValueError(
                        f"SAM3 local ID changed prompt within a run: {key}"
                    )
        dropped = sorted(
            key
            for key in evidence
            if allowed_identity_keys is not None and key not in allowed_identity_keys
        )
        for key in dropped:
            evidence.pop(key)
        item["prompt_observation_count"] = prompt_rows
        item["ignored_ids_absent_from_rasterized_3d_report"] = dropped
        provenance.append(item)
    return evidence, provenance


def signature_identity_keys(paths: Sequence[Path]) -> set[str]:
    keys: set[str] = set()
    for raw_path in sorted(
        {Path(value).expanduser().resolve(strict=True) for value in paths},
        key=lambda value: str(value),
    ):
        payload = _safe_json_object(raw_path)
        if payload.get("schema") != SIGNATURE_SCHEMA:
            raise ValueError(f"unsupported 3D signature report schema: {raw_path}")
        rows = payload.get("local_identities")
        if not isinstance(rows, list):
            raise ValueError(f"3D signature report has no local identities: {raw_path}")
        for row in rows:
            if not isinstance(row, Mapping):
                raise ValueError("local identity row must be an object")
            key = str(row.get("identity_key") or "")
            _identity_parts(key)
            if key in keys:
                raise ValueError(f"duplicate identity key across reports: {key}")
            keys.add(key)
    return keys


def _verify_declared_file(row: Mapping[str, Any], *, label: str) -> Path:
    path = Path(str(row.get("path") or "")).expanduser().resolve(strict=True)
    if int(row.get("bytes", -1)) != path.stat().st_size:
        raise ValueError(f"{label} size mismatch: {path}")
    if str(row.get("sha256") or "") != sha256_file(path):
        raise ValueError(f"{label} checksum mismatch: {path}")
    return path


def derive_color_appearance_from_reports(
    paths: Sequence[Path],
    *,
    verify_artifact_hashes: bool = True,
    local_gate_source: str = LOCAL_GATE_V1,
) -> tuple[dict[str, dict[str, Any]], list[dict[str, Any]]]:
    """Build deterministic masked HSV histograms as CPU-only appearance cues.

    Each visible frame contributes an L1-normalized 16x4x4 HSV histogram, so
    a large close-up cannot dominate a track. The square-root transform turns
    cosine similarity into the Bhattacharyya coefficient for those histograms.
    """

    evidence: dict[str, dict[str, Any]] = {}
    provenance: list[dict[str, Any]] = []
    ordered_paths = sorted(
        {Path(value).expanduser().resolve(strict=True) for value in paths},
        key=lambda value: str(value),
    )
    for raw_path in ordered_paths:
        report = _safe_json_object(raw_path)
        if report.get("schema") != SIGNATURE_SCHEMA:
            raise ValueError(f"unsupported 3D signature report schema: {raw_path}")
        inputs = report.get("inputs")
        identity_rows = report.get("local_identities")
        if not isinstance(inputs, Mapping) or not isinstance(identity_rows, list):
            raise ValueError(f"malformed 3D signature report: {raw_path}")
        episode_id = str(inputs.get("episode_id") or "")
        accepted_ids = {
            int(row["local_id"])
            for row in identity_rows
            if isinstance(row, Mapping)
            and _local_gate_passed(
                row,
                str(row.get("identity_key") or ""),
                local_gate_source=local_gate_source,
            )
        }
        if not accepted_ids:
            provenance.append(
                {
                    "signature_report": str(raw_path),
                    "method": "mean-per-frame-masked-HSV-16x4x4-sqrt-histogram",
                    "embedding_dimensions": 256,
                    "accepted_identity_count": 0,
                    "local_gate_source": local_gate_source,
                    "computed_embedding_count": 0,
                    "artifact_hashes_verified": ("not_required_no_accepted_identities"),
                }
            )
            continue
        episode_row = inputs.get("episode")
        mask_rows = inputs.get("mask_files")
        if not isinstance(episode_row, Mapping) or not isinstance(mask_rows, list):
            raise ValueError(f"report lacks episode/mask provenance: {raw_path}")
        episode_path = (
            _verify_declared_file(episode_row, label="episode JSON")
            if verify_artifact_hashes
            else Path(str(episode_row.get("path") or ""))
            .expanduser()
            .resolve(strict=True)
        )
        episode = _safe_json_object(episode_path)
        frames = episode.get("frames")
        if not isinstance(frames, list) or len(frames) != len(mask_rows):
            raise ValueError(f"episode/mask frame count mismatch: {raw_path}")
        accumulators = {
            local_id: np.zeros(256, dtype=np.float64) for local_id in accepted_ids
        }
        observation_counts = {local_id: 0 for local_id in accepted_ids}
        for frame, mask_row in zip(frames, mask_rows):
            if not isinstance(frame, Mapping) or not isinstance(mask_row, Mapping):
                raise ValueError("episode/mask provenance row must be an object")
            mask_path = (
                _verify_declared_file(mask_row, label="tracker mask")
                if verify_artifact_hashes
                else Path(str(mask_row.get("path") or ""))
                .expanduser()
                .resolve(strict=True)
            )
            source_path = (
                Path(str(frame.get("source_path") or ""))
                .expanduser()
                .resolve(strict=True)
            )
            if verify_artifact_hashes:
                expected_hash = str(frame.get("sha256") or "")
                expected_bytes = int(frame.get("bytes", -1))
                if (
                    expected_bytes != source_path.stat().st_size
                    or expected_hash != sha256_file(source_path)
                ):
                    raise ValueError(f"source frame provenance mismatch: {source_path}")
            with Image.open(mask_path) as image:
                mask = np.asarray(image.convert("L"), dtype=np.uint8)
            with Image.open(source_path) as image:
                hsv = np.asarray(image.convert("HSV"), dtype=np.uint8)
            if hsv.shape[:2] != mask.shape:
                raise ValueError(f"source/mask shape mismatch: {mask_path}")
            present = accepted_ids.intersection(
                int(value) for value in np.unique(mask) if value
            )
            if not present:
                continue
            color_bin = (
                (hsv[..., 0].astype(np.int32) // 16) * 16
                + (hsv[..., 1].astype(np.int32) // 64) * 4
                + (hsv[..., 2].astype(np.int32) // 64)
            )
            for local_id in sorted(present):
                counts = np.bincount(color_bin[mask == local_id], minlength=256).astype(
                    np.float64
                )
                total = float(counts.sum())
                if total > 0:
                    accumulators[local_id] += counts / total
                    observation_counts[local_id] += 1
        for local_id in sorted(accepted_ids):
            count = observation_counts[local_id]
            if count <= 0:
                continue
            histogram = accumulators[local_id] / float(count)
            histogram /= max(float(histogram.sum()), 1.0e-12)
            embedding = np.sqrt(histogram)
            embedding /= max(float(np.linalg.norm(embedding)), 1.0e-12)
            evidence[f"{episode_id}::local:{local_id}"] = {
                "prompts": [],
                "appearance_embedding": embedding.tolist(),
            }
        provenance.append(
            {
                "signature_report": str(raw_path),
                "episode_json": str(episode_path),
                "method": "mean-per-frame-masked-HSV-16x4x4-sqrt-histogram",
                "embedding_dimensions": 256,
                "accepted_identity_count": len(accepted_ids),
                "local_gate_source": local_gate_source,
                "computed_embedding_count": sum(
                    value > 0 for value in observation_counts.values()
                ),
                "artifact_hashes_verified": bool(verify_artifact_hashes),
            }
        )
    return evidence, provenance


def load_synonym_groups(
    path: Path | None,
) -> tuple[dict[str, str], dict[str, Any] | None]:
    """Load explicit synonym groups; no implicit category rewriting is allowed."""

    if path is None:
        return {}, None
    source = Path(path).expanduser().resolve(strict=True)
    payload = _safe_json_object(source)
    groups = payload.get("groups")
    if not isinstance(groups, list):
        raise ValueError("synonym JSON must contain a groups list")
    canonical: dict[str, str] = {}
    for index, raw_group in enumerate(groups):
        if not isinstance(raw_group, list) or len(raw_group) < 2:
            raise ValueError(f"synonym group {index} must contain at least two strings")
        values = sorted(
            {normalize_prompt(value) for value in raw_group if normalize_prompt(value)}
        )
        if len(values) < 2:
            raise ValueError(f"synonym group {index} collapses to fewer than two terms")
        representative = values[0]
        for value in values:
            if value in canonical and canonical[value] != representative:
                raise ValueError(f"synonym term appears in multiple groups: {value}")
            canonical[value] = representative
    provenance = _file_provenance(source)
    provenance["group_count"] = len(groups)
    return canonical, provenance


def _canonical_prompts(
    values: Iterable[str], synonyms: Mapping[str, str]
) -> frozenset[str]:
    return frozenset(synonyms.get(value, value) for value in values)


def _decision_for_gate_source(
    identity: Mapping[str, Any], identity_key: str, local_gate_source: str
) -> Mapping[str, Any]:
    if local_gate_source == LOCAL_GATE_V1:
        field = "decision"
        accepted_status = "accepted_for_cross_episode_association"
        rejected_status = "rejected"
    elif local_gate_source == LOCAL_GATE_V2_SHADOW:
        field = "decision_v2_shadow"
        accepted_status = "would_pass_shadow_gate"
        rejected_status = "shadow_rejected"
    else:
        raise ValueError(f"unsupported local gate source: {local_gate_source!r}")
    decision = identity.get(field)
    if not isinstance(decision, Mapping):
        raise ValueError(f"identity lacks {field} evidence: {identity_key}")
    status = str(decision.get("status") or "")
    if status not in {accepted_status, rejected_status}:
        raise ValueError(
            f"unsupported {local_gate_source} status for {identity_key}: {status!r}"
        )
    passed = status == accepted_status
    if (decision.get("passed") is True) != passed:
        raise ValueError(f"inconsistent passed/status fields for {identity_key}")
    if local_gate_source == LOCAL_GATE_V2_SHADOW:
        if decision.get("mode") != "diagnostic-only-no-publish":
            raise ValueError(f"shadow decision is not diagnostic-only: {identity_key}")
        if decision.get("global_association_authorized") is not False:
            raise ValueError(
                f"shadow decision unexpectedly authorizes association: {identity_key}"
            )
    return decision


def _core_point_ids(
    identity: Mapping[str, Any], *, local_gate_source: str = LOCAL_GATE_V1
) -> frozenset[int]:
    identity_key = str(identity.get("identity_key") or "")
    decision = _decision_for_gate_source(identity, identity_key, local_gate_source)
    geometry = decision.get("qualified_geometry")
    if not isinstance(geometry, Mapping) or geometry.get("available") is not True:
        return frozenset()
    connectivity = geometry.get("connectivity")
    values: object = None
    if isinstance(connectivity, Mapping):
        values = connectivity.get("largest_component_point3d_ids")
    if not isinstance(values, list) or not values:
        values = geometry.get("point3d_ids")
    if not isinstance(values, list):
        return frozenset()
    result = frozenset(int(value) for value in values)
    if any(value < 0 for value in result):
        raise ValueError("qualified point IDs must be non-negative")
    return result


def _local_gate_passed(
    identity: Mapping[str, Any],
    identity_key: str,
    *,
    local_gate_source: str = LOCAL_GATE_V1,
) -> bool:
    decision = _decision_for_gate_source(identity, identity_key, local_gate_source)
    return bool(decision["passed"])


def _quantize_points(
    xyz_m: np.ndarray, voxel_size_m: float
) -> frozenset[tuple[int, int, int]]:
    points = np.asarray(xyz_m, dtype=np.float64)
    if points.ndim != 2 or points.shape[1:] != (3,) or not np.isfinite(points).all():
        raise ValueError("metric points must have finite shape [N,3]")
    if points.shape[0] == 0:
        return frozenset()
    keys = np.floor(points / float(voxel_size_m)).astype(np.int64)
    return frozenset(tuple(int(value) for value in row) for row in keys)


def _metric_geometry(
    xyz_m: np.ndarray, voxel_size_m: float
) -> tuple[
    frozenset[tuple[int, int, int]], tuple[float, float, float] | None, float | None
]:
    points = np.asarray(xyz_m, dtype=np.float64)
    if points.shape[0] == 0:
        return frozenset(), None, None
    centroid = tuple(float(value) for value in np.median(points, axis=0))
    if points.shape[0] == 1:
        diagonal = float(voxel_size_m)
    else:
        low, high = np.quantile(points, [0.02, 0.98], axis=0)
        diagonal = max(float(np.linalg.norm(high - low)), float(voxel_size_m))
    return _quantize_points(points, voxel_size_m), centroid, diagonal


def _validate_colmap_provenance(
    report: Mapping[str, Any],
    *,
    digest_cache: dict[tuple[str, int, int], str] | None = None,
) -> None:
    inputs = report.get("inputs")
    if not isinstance(inputs, Mapping):
        raise ValueError("signature report inputs must be an object")
    rows = inputs.get("colmap_contract_files")
    if not isinstance(rows, list) or not rows:
        raise ValueError("signature report lacks COLMAP file provenance")
    for row in rows:
        if not isinstance(row, Mapping):
            raise ValueError("invalid COLMAP provenance row")
        path = Path(str(row.get("path") or "")).expanduser().resolve(strict=True)
        stat = path.stat()
        if int(row.get("bytes", -1)) != stat.st_size:
            raise ValueError(f"COLMAP provenance size mismatch: {path}")
        cache_key = (str(path), int(stat.st_size), int(stat.st_mtime_ns))
        actual_hash = (digest_cache or {}).get(cache_key)
        if actual_hash is None:
            actual_hash = sha256_file(path)
            if digest_cache is not None:
                digest_cache[cache_key] = actual_hash
        if str(row.get("sha256") or "") != actual_hash:
            raise ValueError(f"COLMAP provenance checksum mismatch: {path}")


def load_signature_reports(
    paths: Sequence[Path],
    *,
    policy: AssociationPolicy,
    identity_evidence: Mapping[str, Mapping[str, Any]] | None = None,
    verify_colmap_hashes: bool = True,
    local_gate_source: str = LOCAL_GATE_V1,
) -> tuple[list[IdentityEvidence], list[dict[str, Any]]]:
    """Validate reports and reconstruct true metric voxel support from COLMAP."""

    policy.validate()
    if local_gate_source not in {LOCAL_GATE_V1, LOCAL_GATE_V2_SHADOW}:
        raise ValueError(f"unsupported local gate source: {local_gate_source!r}")
    evidence_by_key = identity_evidence or {}
    identities: list[IdentityEvidence] = []
    provenance: list[dict[str, Any]] = []
    seen_keys: set[str] = set()
    unknown_evidence = set(evidence_by_key)
    ordered_paths = sorted(
        {Path(value).expanduser().resolve(strict=True) for value in paths},
        key=lambda value: str(value),
    )
    if not ordered_paths:
        raise ValueError("at least one 3D signature report is required")
    colmap_digest_cache: dict[tuple[str, int, int], str] = {}
    for report_index, source in enumerate(ordered_paths):
        report = _safe_json_object(source)
        if report.get("schema") != SIGNATURE_SCHEMA:
            raise ValueError(f"unsupported 3D signature report schema: {source}")
        if local_gate_source == LOCAL_GATE_V2_SHADOW:
            shadow_contract = report.get("v2_shadow")
            if (
                not isinstance(shadow_contract, Mapping)
                or shadow_contract.get("schema")
                != "farm.tracker-episode-3d-signatures.v2-shadow"
                or shadow_contract.get("mode") != "diagnostic-only-no-publish"
                or shadow_contract.get("automatic_promotion_allowed") is not False
                or shadow_contract.get("global_association_authorized") is not False
            ):
                raise ValueError(f"missing fail-closed v2 shadow contract: {source}")
        inputs = report.get("inputs")
        rows = report.get("local_identities")
        if not isinstance(inputs, Mapping) or not isinstance(rows, list):
            raise ValueError(f"malformed 3D signature report: {source}")
        episode_id = str(inputs.get("episode_id") or "")
        split = str(inputs.get("split") or "")
        if split not in {"train", "heldout"}:
            raise ValueError(f"invalid report split {split!r}: {source}")
        if verify_colmap_hashes:
            _validate_colmap_provenance(report, digest_cache=colmap_digest_cache)
        scale = float(inputs.get("meters_per_scene_unit", math.nan))
        if not math.isfinite(scale) or scale <= 0:
            raise ValueError(f"invalid metric scale: {source}")
        source_format = str(inputs.get("colmap_format") or "")
        model_dir = Path(str(inputs.get("colmap_model") or ""))

        point_ids_by_key: dict[str, frozenset[int]] = {}
        all_point_ids: set[int] = set()
        for row in rows:
            if not isinstance(row, Mapping):
                raise ValueError("local identity row must be an object")
            key = str(row.get("identity_key") or "")
            parsed_episode, parsed_local = _identity_parts(key)
            if (
                parsed_episode != episode_id
                or int(row.get("local_id", -1)) != parsed_local
            ):
                raise ValueError(f"identity key/report namespace mismatch: {key}")
            if str(row.get("episode_id") or "") != episode_id:
                raise ValueError(f"identity episode mismatch: {key}")
            if key in seen_keys:
                raise ValueError(f"duplicate identity key across reports: {key}")
            seen_keys.add(key)
            point_ids = (
                _core_point_ids(row, local_gate_source=local_gate_source)
                if _local_gate_passed(row, key, local_gate_source=local_gate_source)
                else frozenset()
            )
            point_ids_by_key[key] = point_ids
            all_point_ids.update(point_ids)
        points, points_path = read_selected_points3d(
            model_dir,
            source_format=source_format,
            selected_point_ids=all_point_ids,
        )
        if all_point_ids and points_path is None:
            raise ValueError(f"COLMAP points3D file unavailable for {source}")

        source_name_by_index: dict[int, str] = {}
        episode_provenance = inputs.get("episode")
        if isinstance(episode_provenance, Mapping) and episode_provenance.get("path"):
            episode_path = _verify_declared_file(
                episode_provenance, label="materialized episode JSON"
            )
            episode_payload = _safe_json_object(episode_path)
            episode_frames = episode_payload.get("frames")
            if not isinstance(episode_frames, list):
                raise ValueError(f"materialized episode has no frames: {episode_path}")
            for index, frame in enumerate(episode_frames):
                if not isinstance(frame, Mapping):
                    raise ValueError("materialized episode frame must be an object")
                episode_index = int(frame.get("episode_index", index))
                source_name = str(frame.get("source_name") or "")
                if not source_name or Path(source_name).name != source_name:
                    raise ValueError(
                        "materialized episode source_name must be a basename"
                    )
                source_name_by_index[episode_index] = source_name

        accepted_keys_this_report: list[str] = []
        for row in rows:
            key = str(row["identity_key"])
            _, local_id = _identity_parts(key)
            decision = _decision_for_gate_source(row, key, local_gate_source)
            raw = row.get("raw")
            if not isinstance(raw, Mapping):
                raise ValueError(f"identity lacks raw evidence: {key}")
            passed = _local_gate_passed(row, key, local_gate_source=local_gate_source)
            failed = decision.get("failed_checks", [])
            if not isinstance(failed, list):
                raise ValueError(f"failed_checks must be a list: {key}")
            if passed and failed:
                raise ValueError(f"accepted identity has failed checks: {key}")
            if passed:
                accepted_keys_this_report.append(key)
            core_ids = point_ids_by_key[key]
            missing = sorted(core_ids.difference(points))
            if passed and missing:
                raise ValueError(f"accepted identity has missing COLMAP points: {key}")
            xyz = np.asarray(
                [
                    points[value].xyz_scene
                    for value in sorted(core_ids)
                    if value in points
                ],
                dtype=np.float64,
            ).reshape(-1, 3)
            xyz *= scale
            voxels, centroid, diagonal = _metric_geometry(xyz, policy.voxel_size_m)
            frame_rows = raw.get("frame_observations", [])
            if not isinstance(frame_rows, list):
                raise ValueError(f"frame_observations must be a list: {key}")
            frame_indices = frozenset(
                int(value["episode_index"]) for value in frame_rows
            )
            timestamps = frozenset(
                str(value["physical_timestamp"]) for value in frame_rows
            )
            source_names = frozenset(
                source_name_by_index[index]
                for index in frame_indices
                if index in source_name_by_index
            )
            extra = evidence_by_key.get(key, {})
            prompts_raw = extra.get("prompts", [])
            prompts = frozenset(
                normalize_prompt(value)
                for value in prompts_raw
                if normalize_prompt(value)
            )
            embedding_raw = extra.get("appearance_embedding")
            embedding = None
            if embedding_raw is not None:
                vector = _finite_vector(
                    embedding_raw, label=f"appearance embedding for {key}"
                )
                norm = float(np.linalg.norm(vector))
                if norm <= 1.0e-12:
                    raise ValueError(f"appearance embedding is zero for {key}")
                embedding = tuple(float(value) for value in vector / norm)
            identities.append(
                IdentityEvidence(
                    identity_key=key,
                    episode_id=episode_id,
                    local_id=local_id,
                    split=split,
                    passed_local_gate=passed,
                    local_failed_checks=tuple(str(value) for value in failed),
                    core_point_ids=core_ids,
                    metric_voxels=voxels,
                    centroid_m=centroid,
                    scale_diagonal_m=diagonal,
                    visible_frame_indices=frame_indices,
                    physical_timestamps=timestamps,
                    visible_source_names=source_names,
                    prompts=prompts,
                    appearance_embedding=embedding,
                    report_index=report_index,
                )
            )
            unknown_evidence.discard(key)
        summary = report.get("summary")
        if not isinstance(summary, Mapping):
            raise ValueError(f"signature report has no summary: {source}")
        if int(summary.get("local_identity_count", -1)) != len(rows):
            raise ValueError(f"signature summary identity count mismatch: {source}")
        if local_gate_source == LOCAL_GATE_V1:
            expected_accepted_count = summary.get(
                "accepted_for_cross_episode_association_count", -1
            )
            expected_accepted_keys = summary.get("accepted_identity_keys", [])
        else:
            shadow_summary = report["v2_shadow"].get("summary")
            if not isinstance(shadow_summary, Mapping):
                raise ValueError(f"signature shadow summary missing: {source}")
            expected_accepted_count = shadow_summary.get("shadow_would_pass_count", -1)
            expected_accepted_keys = shadow_summary.get(
                "shadow_would_pass_identity_keys", []
            )
        if int(expected_accepted_count) != len(accepted_keys_this_report):
            raise ValueError(f"signature summary accepted count mismatch: {source}")
        if sorted(expected_accepted_keys) != sorted(accepted_keys_this_report):
            raise ValueError(f"signature summary accepted keys mismatch: {source}")
        item = _file_provenance(source)
        item.update(
            {
                "schema": report.get("schema"),
                "episode_id": episode_id,
                "split": split,
                "local_identity_count": len(rows),
                "accepted_local_identity_count": len(accepted_keys_this_report),
                "local_gate_source": local_gate_source,
            }
        )
        provenance.append(item)
    if unknown_evidence:
        raise ValueError(
            "identity evidence references identities absent from reports: "
            + ", ".join(sorted(unknown_evidence)[:8])
        )
    return sorted(identities, key=lambda value: value.identity_key), provenance


def _set_overlap(left: frozenset[Any], right: frozenset[Any]) -> dict[str, Any]:
    shared = len(left.intersection(right))
    union = len(left.union(right))
    minimum = min(len(left), len(right))
    return {
        "left_count": len(left),
        "right_count": len(right),
        "shared_count": shared,
        "jaccard": float(shared / union) if union else 0.0,
        "minimum_set_containment": float(shared / minimum) if minimum else 0.0,
    }


def _dilate_voxels(
    values: frozenset[tuple[int, int, int]], radius: int
) -> frozenset[tuple[int, int, int]]:
    if radius <= 0 or not values:
        return values
    offsets = range(-int(radius), int(radius) + 1)
    return frozenset(
        (x + dx, y + dy, z + dz)
        for x, y, z in values
        for dx in offsets
        for dy in offsets
        for dz in offsets
    )


def _voxel_overlap(
    left: frozenset[tuple[int, int, int]],
    right: frozenset[tuple[int, int, int]],
    radius: int,
) -> dict[str, Any]:
    if radius <= 0:
        result = _set_overlap(left, right)
        result["neighbor_radius_voxels"] = 0
        return result
    left_matches = sum(value in _dilate_voxels(right, radius) for value in left)
    right_matches = sum(value in _dilate_voxels(left, radius) for value in right)
    shared = min(left_matches, right_matches)
    union_estimate = max(0, len(left) + len(right) - shared)
    minimum = min(len(left), len(right))
    return {
        "left_count": len(left),
        "right_count": len(right),
        "shared_count": int(shared),
        "jaccard": float(shared / union_estimate) if union_estimate else 0.0,
        "minimum_set_containment": float(shared / minimum) if minimum else 0.0,
        "neighbor_radius_voxels": int(radius),
    }


def _prompt_comparison(
    left: IdentityEvidence,
    right: IdentityEvidence,
    synonyms: Mapping[str, str],
) -> dict[str, Any]:
    first = _canonical_prompts(left.prompts, synonyms)
    second = _canonical_prompts(right.prompts, synonyms)
    if not first or not second:
        return {"status": "unavailable", "compatible": None, "score": None}
    shared = sorted(first.intersection(second))
    return {
        "status": "compatible" if shared else "incompatible",
        "compatible": bool(shared),
        "score": 1.0 if shared else 0.0,
        "left": sorted(first),
        "right": sorted(second),
        "shared": shared,
    }


def _appearance_comparison(
    left: IdentityEvidence, right: IdentityEvidence
) -> dict[str, Any]:
    if left.appearance_embedding is None or right.appearance_embedding is None:
        return {"status": "unavailable", "compatible": None, "cosine_similarity": None}
    first = np.asarray(left.appearance_embedding, dtype=np.float64)
    second = np.asarray(right.appearance_embedding, dtype=np.float64)
    if first.shape != second.shape:
        return {
            "status": "incompatible_dimension",
            "compatible": False,
            "cosine_similarity": None,
        }
    cosine = float(np.clip(np.dot(first, second), -1.0, 1.0))
    return {"status": "available", "compatible": None, "cosine_similarity": cosine}


def compare_pair(
    left: IdentityEvidence,
    right: IdentityEvidence,
    *,
    policy: AssociationPolicy,
    synonyms: Mapping[str, str] | None = None,
) -> dict[str, Any]:
    """Return a complete deterministic pair decision with hard vetoes."""

    policy.validate()
    if left.identity_key == right.identity_key:
        raise ValueError("cannot compare an identity with itself")
    synonyms = synonyms or {}
    left, right = sorted((left, right), key=lambda value: value.identity_key)
    shared_source_names = left.visible_source_names.intersection(
        right.visible_source_names
    )
    simultaneous = bool(shared_source_names) or (
        left.episode_id == right.episode_id
        and bool(left.visible_frame_indices.intersection(right.visible_frame_indices))
    )
    sparse = _set_overlap(left.core_point_ids, right.core_point_ids)
    voxels = _voxel_overlap(
        left.metric_voxels,
        right.metric_voxels,
        int(policy.voxel_neighbor_radius),
    )
    sparse_gate = sparse["shared_count"] >= policy.minimum_shared_sparse_tracks and (
        sparse["minimum_set_containment"] >= policy.minimum_sparse_track_containment
        or sparse["jaccard"] >= policy.minimum_sparse_track_jaccard
    )
    voxel_gate = voxels["shared_count"] >= policy.minimum_shared_metric_voxels and (
        voxels["minimum_set_containment"] >= policy.minimum_metric_voxel_containment
        or voxels["jaccard"] >= policy.minimum_metric_voxel_jaccard
    )
    centroid_distance = None
    normalized_centroid_distance = None
    centroid_gate = False
    scale_ratio = None
    scale_gate = False
    if left.centroid_m is not None and right.centroid_m is not None:
        centroid_distance = float(
            np.linalg.norm(np.asarray(left.centroid_m) - np.asarray(right.centroid_m))
        )
    if left.scale_diagonal_m is not None and right.scale_diagonal_m is not None:
        smaller = min(left.scale_diagonal_m, right.scale_diagonal_m)
        larger = max(left.scale_diagonal_m, right.scale_diagonal_m)
        scale_ratio = float(larger / max(smaller, policy.voxel_size_m))
        scale_gate = scale_ratio <= policy.maximum_scale_ratio
        if centroid_distance is not None:
            normalized_centroid_distance = float(
                centroid_distance / max(larger, policy.voxel_size_m)
            )
            centroid_gate = (
                centroid_distance <= policy.maximum_centroid_distance_m
                and normalized_centroid_distance
                <= policy.maximum_centroid_distance_scale
            )
    prompt = _prompt_comparison(left, right, synonyms)
    appearance = _appearance_comparison(left, right)
    if appearance["cosine_similarity"] is not None:
        appearance["compatible"] = (
            appearance["cosine_similarity"] >= policy.minimum_appearance_similarity
        )
        appearance["status"] = (
            "compatible" if appearance["compatible"] else "incompatible"
        )
    prompt_incompatible = prompt["compatible"] is False
    compatibility_available = (
        prompt["compatible"] is True or appearance["compatible"] is True
    )
    overlap_score = max(
        0.7 * sparse["minimum_set_containment"] + 0.3 * sparse["jaccard"],
        0.7 * voxels["minimum_set_containment"] + 0.3 * voxels["jaccard"],
    )
    centroid_score = (
        max(0.0, 1.0 - centroid_distance / policy.maximum_centroid_distance_m)
        if centroid_distance is not None
        else 0.0
    )
    scale_score = (
        max(
            0.0,
            1.0
            - math.log(max(scale_ratio, 1.0)) / math.log(policy.maximum_scale_ratio),
        )
        if scale_ratio is not None and policy.maximum_scale_ratio > 1.0
        else 0.0
    )
    compatibility_score = max(
        float(prompt["score"] or 0.0),
        max(0.0, float(appearance["cosine_similarity"] or 0.0)),
    )
    score = float(
        0.55 * overlap_score
        + 0.20 * centroid_score
        + 0.10 * scale_score
        + 0.15 * compatibility_score
    )
    failed: list[str] = []
    if not left.passed_local_gate or not right.passed_local_gate:
        failed.append("local_3d_gate")
    if simultaneous:
        failed.append("simultaneous_visibility_conflict")
    if not (sparse_gate or voxel_gate):
        failed.append("metric_support_overlap")
    if not centroid_gate:
        failed.append("centroid_compatibility")
    if not scale_gate:
        failed.append("scale_compatibility")
    if prompt_incompatible:
        failed.append("prompt_conflict")
    if not compatibility_available:
        failed.append("appearance_or_prompt_compatibility")
    if score < policy.minimum_pair_score:
        failed.append("pair_score")
    eligible = not failed
    return {
        "left_identity_key": left.identity_key,
        "right_identity_key": right.identity_key,
        "eligible": eligible,
        "score": score,
        "failed_checks": failed,
        "hard_conflict": simultaneous or prompt_incompatible,
        "simultaneous_visibility": {
            "same_episode": left.episode_id == right.episode_id,
            "shared_frame_indices": sorted(
                left.visible_frame_indices.intersection(right.visible_frame_indices)
            ),
            "shared_source_names": sorted(shared_source_names),
            "conflict": simultaneous,
        },
        "sparse_track_overlap": sparse,
        "metric_voxel_overlap": voxels,
        "overlap_gate": {"sparse_tracks": sparse_gate, "metric_voxels": voxel_gate},
        "centroid": {
            "distance_m": centroid_distance,
            "distance_over_larger_scale": normalized_centroid_distance,
            "compatible": centroid_gate,
        },
        "scale": {"diagonal_ratio": scale_ratio, "compatible": scale_gate},
        "prompt": prompt,
        "appearance": appearance,
    }


def _pair_key(left: str, right: str) -> tuple[str, str]:
    return tuple(sorted((left, right)))  # type: ignore[return-value]


def _ambiguous_train_identities(
    identities: Sequence[IdentityEvidence],
    pair_by_key: Mapping[tuple[str, str], Mapping[str, Any]],
    margin: float,
) -> tuple[set[str], list[dict[str, Any]]]:
    ambiguous: set[str] = set()
    rows: list[dict[str, Any]] = []
    for identity in identities:
        candidates: list[tuple[float, str]] = []
        for other in identities:
            if other.identity_key == identity.identity_key:
                continue
            pair = pair_by_key[_pair_key(identity.identity_key, other.identity_key)]
            if pair["eligible"]:
                candidates.append((float(pair["score"]), other.identity_key))
        candidates.sort(key=lambda value: (-value[0], value[1]))
        if len(candidates) < 2:
            continue
        best_score, best_key = candidates[0]
        conflicting_alternatives: list[tuple[float, str]] = []
        for score, key in candidates[1:]:
            relation = pair_by_key[_pair_key(best_key, key)]
            if not relation["eligible"]:
                conflicting_alternatives.append((score, key))
        if not conflicting_alternatives:
            continue
        second_score, second_key = conflicting_alternatives[0]
        difference = float(best_score - second_score)
        if difference < margin:
            ambiguous.add(identity.identity_key)
            rows.append(
                {
                    "identity_key": identity.identity_key,
                    "best_candidate": best_key,
                    "best_score": best_score,
                    "conflicting_candidate": second_key,
                    "conflicting_score": second_score,
                    "margin": difference,
                    "minimum_margin": margin,
                }
            )
    return ambiguous, rows


def _complete_link_clusters(
    identities: Sequence[IdentityEvidence],
    pair_by_key: Mapping[tuple[str, str], Mapping[str, Any]],
) -> list[list[str]]:
    clusters: list[tuple[str, ...]] = [(value.identity_key,) for value in identities]
    while True:
        candidates: list[
            tuple[float, float, tuple[str, ...], tuple[str, ...], int, int]
        ] = []
        for left_index in range(len(clusters)):
            for right_index in range(left_index + 1, len(clusters)):
                left, right = clusters[left_index], clusters[right_index]
                pairs = [pair_by_key[_pair_key(a, b)] for a in left for b in right]
                if pairs and all(pair["eligible"] for pair in pairs):
                    scores = [float(pair["score"]) for pair in pairs]
                    candidates.append(
                        (
                            min(scores),
                            float(sum(scores) / len(scores)),
                            left,
                            right,
                            left_index,
                            right_index,
                        )
                    )
        if not candidates:
            break
        candidates.sort(key=lambda value: (-value[0], -value[1], value[2], value[3]))
        _, _, left, right, left_index, right_index = candidates[0]
        merged = tuple(sorted((*left, *right)))
        clusters = [
            value
            for index, value in enumerate(clusters)
            if index not in {left_index, right_index}
        ]
        clusters.append(merged)
        clusters.sort()
    return [list(value) for value in sorted(clusters)]


def _cluster_priority(
    cluster: Sequence[str], by_key: Mapping[str, IdentityEvidence]
) -> tuple[Any, ...]:
    point_support = len(set().union(*(by_key[key].core_point_ids for key in cluster)))
    timestamp_support = len(
        set().union(*(by_key[key].physical_timestamps for key in cluster))
    )
    return (-len(cluster), -point_support, -timestamp_support, tuple(cluster))


def associate_identities(
    identities: Sequence[IdentityEvidence],
    *,
    policy: AssociationPolicy,
    synonyms: Mapping[str, str] | None = None,
) -> dict[str, Any]:
    """Fit train-only complete-link clusters and apply heldout queries."""

    policy.validate()
    synonyms = synonyms or {}
    ordered = sorted(identities, key=lambda value: value.identity_key)
    by_key = {value.identity_key: value for value in ordered}
    if len(by_key) != len(ordered):
        raise ValueError("identity keys must be unique")
    train = [
        value for value in ordered if value.split == "train" and value.passed_local_gate
    ]
    heldout = [
        value
        for value in ordered
        if value.split == "heldout" and value.passed_local_gate
    ]

    pair_by_key: dict[tuple[str, str], dict[str, Any]] = {}
    for left_index, left in enumerate(train):
        for right in train[left_index + 1 :]:
            pair_by_key[_pair_key(left.identity_key, right.identity_key)] = (
                compare_pair(left, right, policy=policy, synonyms=synonyms)
            )
    ambiguous, ambiguity_rows = _ambiguous_train_identities(
        train, pair_by_key, policy.minimum_assignment_margin
    )
    eligible_train = [value for value in train if value.identity_key not in ambiguous]
    eligible_pair_map = {
        key: value
        for key, value in pair_by_key.items()
        if key[0] not in ambiguous and key[1] not in ambiguous
    }
    clusters = _complete_link_clusters(eligible_train, eligible_pair_map)

    ranked = sorted(clusters, key=lambda value: _cluster_priority(value, by_key))
    selected_cluster_keys = {
        tuple(value) for value in ranked[: policy.maximum_global_ids]
    }
    selected_clusters = sorted(
        (value for value in clusters if tuple(value) in selected_cluster_keys),
        key=lambda value: tuple(value),
    )
    cluster_to_global = {
        tuple(cluster): index + 1 for index, cluster in enumerate(selected_clusters)
    }
    train_global: dict[str, int] = {}
    for cluster in clusters:
        global_id = cluster_to_global.get(tuple(cluster), 0)
        for key in cluster:
            train_global[key] = global_id

    heldout_pair_rows: list[dict[str, Any]] = []
    heldout_assignment: dict[str, tuple[int, str, dict[str, Any] | None]] = {}
    for query in heldout:
        candidates: list[tuple[float, int, str, list[dict[str, Any]]]] = []
        for cluster in selected_clusters:
            global_id = cluster_to_global[tuple(cluster)]
            comparisons = [
                compare_pair(query, by_key[key], policy=policy, synonyms=synonyms)
                for key in cluster
            ]
            eligible = [value for value in comparisons if value["eligible"]]
            if eligible:
                best = max(
                    eligible,
                    key=lambda value: (
                        float(value["score"]),
                        value["left_identity_key"],
                        value["right_identity_key"],
                    ),
                )
                candidates.append(
                    (
                        float(best["score"]),
                        global_id,
                        str(
                            best["right_identity_key"]
                            if best["left_identity_key"] == query.identity_key
                            else best["left_identity_key"]
                        ),
                        comparisons,
                    )
                )
            heldout_pair_rows.extend(comparisons)
        candidates.sort(key=lambda value: (-value[0], value[1], value[2]))
        if not candidates:
            heldout_assignment[query.identity_key] = (
                0,
                "heldout_unknown_no_compatible_train_cluster",
                None,
            )
            continue
        best = candidates[0]
        second_score = candidates[1][0] if len(candidates) > 1 else None
        margin = float(best[0] - second_score) if second_score is not None else None
        detail = {
            "matched_train_identity_key": best[2],
            "score": best[0],
            "second_best_score": second_score,
            "margin": margin,
            "minimum_margin": policy.minimum_assignment_margin,
        }
        if margin is not None and margin < policy.minimum_assignment_margin:
            heldout_assignment[query.identity_key] = (
                0,
                "heldout_unknown_weak_margin",
                detail,
            )
        else:
            heldout_assignment[query.identity_key] = (
                best[1],
                "heldout_application_matched",
                detail,
            )

    assignments: list[dict[str, Any]] = []
    local_to_global: dict[str, int] = {}
    for identity in ordered:
        global_id = 0
        status = "rejected"
        reason = "local_3d_gate_rejected"
        details: dict[str, Any] | None = None
        if identity.passed_local_gate and identity.split == "train":
            if identity.identity_key in ambiguous:
                status, reason = "unknown", "train_unknown_weak_margin"
            else:
                global_id = train_global.get(identity.identity_key, 0)
                if global_id:
                    cluster = next(
                        value for value in clusters if identity.identity_key in value
                    )
                    status = "assigned"
                    reason = (
                        "train_complete_link_cluster"
                        if len(cluster) > 1
                        else "train_singleton_local_3d_support"
                    )
                else:
                    status, reason = "rejected", "global_id_capacity_exceeded"
        elif identity.passed_local_gate and identity.split == "heldout":
            global_id, reason, details = heldout_assignment[identity.identity_key]
            status = "applied" if global_id else "unknown"
        row = {
            "identity_key": identity.identity_key,
            "episode_id": identity.episode_id,
            "local_id": identity.local_id,
            "split": identity.split,
            "global_id": int(global_id),
            "status": status,
            "reason": reason,
            "local_failed_checks": list(identity.local_failed_checks),
        }
        if details is not None:
            row["match"] = details
        assignments.append(row)
        local_to_global[identity.identity_key] = int(global_id)

    global_objects: list[dict[str, Any]] = []
    for cluster in selected_clusters:
        global_id = cluster_to_global[tuple(cluster)]
        members = [by_key[key] for key in cluster]
        heldout_members = sorted(
            key
            for key, (value, _, _) in heldout_assignment.items()
            if value == global_id
        )
        point_ids = set().union(*(value.core_point_ids for value in members))
        voxels = set().union(*(value.metric_voxels for value in members))
        prompts = sorted(
            set().union(
                *(_canonical_prompts(value.prompts, synonyms) for value in members)
            )
        )
        global_objects.append(
            {
                "global_id": global_id,
                "fit_member_identity_keys": list(cluster),
                "heldout_application_identity_keys": heldout_members,
                "train_member_count": len(cluster),
                "core_sparse_track_count": len(point_ids),
                "metric_voxel_count": len(voxels),
                "canonical_prompts": prompts,
            }
        )

    pair_rows = sorted(
        [*pair_by_key.values(), *heldout_pair_rows],
        key=lambda value: (value["left_identity_key"], value["right_identity_key"]),
    )
    interesting_pairs = [
        value
        for value in pair_rows
        if value["eligible"]
        or value["simultaneous_visibility"]["conflict"]
        or value["sparse_track_overlap"]["shared_count"]
        or value["metric_voxel_overlap"]["shared_count"]
    ]
    return {
        "fit_splits": ["train"],
        "local_to_global": local_to_global,
        "assignments": assignments,
        "global_objects": global_objects,
        "pair_decisions": interesting_pairs,
        "ambiguities": ambiguity_rows,
        "summary": {
            "input_local_identity_count": len(ordered),
            "locally_accepted_train_count": len(train),
            "locally_accepted_heldout_count": len(heldout),
            "global_object_count": len(global_objects),
            "assigned_train_identity_count": sum(
                row["global_id"] > 0 and row["split"] == "train" for row in assignments
            ),
            "applied_heldout_identity_count": sum(
                row["global_id"] > 0 and row["split"] == "heldout"
                for row in assignments
            ),
            "unknown_count": sum(row["status"] == "unknown" for row in assignments),
            "rejected_count": sum(row["status"] == "rejected" for row in assignments),
            "pair_count_evaluated": len(pair_rows),
            "pair_count_reported": len(interesting_pairs),
            "eligible_pair_count": sum(value["eligible"] for value in pair_rows),
            "train_pair_count_evaluated": len(pair_by_key),
            "train_eligible_pair_count": sum(
                value["eligible"] for value in pair_by_key.values()
            ),
            "heldout_pair_count_evaluated": len(heldout_pair_rows),
            "heldout_eligible_pair_count": sum(
                value["eligible"] for value in heldout_pair_rows
            ),
            "hard_conflict_pair_count": sum(
                value["hard_conflict"] for value in pair_rows
            ),
            "capacity": int(policy.maximum_global_ids),
            "capacity_exceeded": len(clusters) > policy.maximum_global_ids,
        },
    }


def policy_dict(policy: AssociationPolicy) -> dict[str, Any]:
    policy.validate()
    return asdict(policy)
