"""Read-only multi-scene viewer for FARM, bridge, ShapeR and dense lift artifacts.

The module deliberately keeps artifact validation separate from the optional
Viser runtime.  ``validate_registry()`` therefore works in the host-side test
environment, while ``serve_registry()`` imports Viser only when a server is
actually requested.

Viser 1.0.30 provides an experimental anisotropic Gaussian-splat primitive.
The source and dense-instance layers keep every selected source row, but the
browser payload is a Viser float16/uint8-quantized DC preview.  Higher-order SH
is deliberately not claimed because Viser does not evaluate it.
"""

from __future__ import annotations

import colorsys
import functools
import hashlib
import ipaddress
import io
import json
import math
import re
import struct
import threading
import time
import warnings
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

import numpy as np
import yaml

from .integrity import DirectoryIntegrityError, verify_directory_tree_descriptor


REGISTRY_SCHEMA = "farm.unified-viewer-registry.v1"
LIFT_RESULT_SCHEMA = "farm.gaussian-lift.result.v1"
LIFT_SUCCESS_SCHEMA = "farm.gaussian-lift.success.v1"
SHAPER_SCENE_SCHEMA = "farm.shaper-scene.v2"
SHAPER_SUCCESS_SCHEMA = "farm.shaper-scene.success.v2"
SHAPER_VIEWER_SCHEMA = "farm.shaper-viewer-assets.v1"
LEGACY_SHAPER_SCENE_SCHEMA = "farm-shaper-bridge.combined-scene.v1"
LEGACY_SHAPER_VIEWER_SCHEMA = "farm-shaper-bridge.viewer-assets.v1"
REQUIRED_VISER_VERSION = "1.0.30"
UNKNOWN_INSTANCE_ID = -1
SH_DC_C0 = 0.28209479177387814
VISER_PACKED_BYTES_PER_GAUSSIAN = 32
_SCENE_ID_RE = re.compile(r"^[a-z0-9][a-z0-9_-]{0,63}$")
_SHA256_RE = re.compile(r"^[0-9a-f]{64}$")
_ABSOLUTE_UI_PATH_RE = re.compile(r"(?<![A-Za-z0-9_.-])/(?:[^\s`'\";,]+)")
_VERIFIED_BANK_FIELDS = {
    "object_ids", "indptr", "indices", "confidence", "timestamp_support",
}


class UnifiedViewerError(RuntimeError):
    """Raised when a registry or a required artifact violates its contract."""


def _load_json(path: Path, *, mapping: bool = True) -> Any:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise UnifiedViewerError(f"cannot read JSON artifact {path}: {exc}") from exc
    expected = Mapping if mapping else list
    if not isinstance(value, expected):
        kind = "object" if mapping else "array"
        raise UnifiedViewerError(f"expected JSON {kind}: {path}")
    return value


def sha256_file(path: Path, *, chunk_bytes: int = 8 << 20) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while True:
            payload = handle.read(chunk_bytes)
            if not payload:
                return digest.hexdigest()
            digest.update(payload)


def _canonical_sampled_sha256(path: Path, *, chunk_bytes: int) -> tuple[str, tuple[int, ...]]:
    size = int(path.stat().st_size)
    span = min(int(chunk_bytes), size)
    offsets = tuple(sorted({0, max(0, (size - span) // 2), max(0, size - span)}))
    digest = hashlib.sha256()
    digest.update(b"farm-sampled-sha256-v1\0")
    digest.update(struct.pack("<Q", size))
    with path.open("rb") as handle:
        for offset in offsets:
            handle.seek(offset)
            payload = handle.read(span)
            digest.update(struct.pack("<QQ", offset, len(payload)))
            digest.update(payload)
    return digest.hexdigest(), offsets


def _legacy_bridge_fingerprint(path: Path) -> dict[str, Any]:
    stat = path.stat()
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        digest.update(handle.read(1 << 20))
        if stat.st_size > (2 << 20):
            handle.seek(max(0, stat.st_size - (1 << 20)))
            digest.update(handle.read(1 << 20))
    return {
        "bytes": int(stat.st_size),
        "mtime_ns": int(stat.st_mtime_ns),
        "sample_sha256": digest.hexdigest(),
    }


@dataclass(frozen=True)
class SourceFingerprint:
    algorithm: str
    digest: str
    size_bytes: int
    chunk_bytes: int
    sampled_offsets: tuple[int, ...] = ()

    @classmethod
    def from_mapping(cls, value: Any, *, field_name: str) -> "SourceFingerprint":
        if not isinstance(value, Mapping):
            raise UnifiedViewerError(f"{field_name} must be a mapping")
        allowed = {"algorithm", "digest", "size_bytes", "chunk_bytes", "sampled_offsets"}
        unknown = sorted(set(value) - allowed)
        if unknown:
            raise UnifiedViewerError(f"unknown {field_name} fields: {', '.join(unknown)}")
        algorithm = str(value.get("algorithm") or "")
        digest = str(value.get("digest") or "")
        if algorithm not in {"sha256", "sha256-sampled-v1"}:
            raise UnifiedViewerError(f"unsupported {field_name}.algorithm: {algorithm!r}")
        if not _SHA256_RE.fullmatch(digest):
            raise UnifiedViewerError(f"{field_name}.digest must be lowercase SHA-256")
        try:
            size = int(value["size_bytes"])
            chunk = int(value.get("chunk_bytes", 1 << 20))
            offsets = tuple(int(item) for item in value.get("sampled_offsets", ()))
        except (KeyError, TypeError, ValueError) as exc:
            raise UnifiedViewerError(f"invalid numeric {field_name} fields") from exc
        if size <= 0 or chunk <= 0 or any(item < 0 for item in offsets):
            raise UnifiedViewerError(f"invalid {field_name} size/chunk/offset")
        if algorithm == "sha256" and offsets:
            raise UnifiedViewerError(f"{field_name}.sampled_offsets must be empty for sha256")
        return cls(algorithm, digest, size, chunk, offsets)

    def validate(self, path: Path, *, full: bool = False) -> None:
        if not path.is_file():
            raise UnifiedViewerError(f"source PLY is missing: {path}")
        actual_size = int(path.stat().st_size)
        if actual_size != self.size_bytes:
            raise UnifiedViewerError(
                f"source PLY size mismatch: expected {self.size_bytes}, got {actual_size}"
            )
        if self.algorithm == "sha256":
            actual = sha256_file(path)
        else:
            actual, offsets = _canonical_sampled_sha256(path, chunk_bytes=self.chunk_bytes)
            if self.sampled_offsets and offsets != self.sampled_offsets:
                raise UnifiedViewerError(
                    f"source PLY sampled offsets mismatch: expected {self.sampled_offsets}, got {offsets}"
                )
        if actual != self.digest:
            raise UnifiedViewerError(
                f"source PLY {self.algorithm} mismatch: expected {self.digest}, got {actual}"
            )
        if full and self.algorithm != "sha256":
            # ``full`` is only meaningful when another artifact supplies the
            # expected full digest (the dense-lift result does this).
            raise UnifiedViewerError("full source verification requires an expected full SHA-256")


@dataclass(frozen=True)
class ArtifactBinding:
    """Expected immutable file identity supplied by a completed producer."""

    path: Path
    size_bytes: int
    sha256: str

    def verify(self, *, field_name: str) -> None:
        if not self.path.is_file():
            raise UnifiedViewerError(f"{field_name} is missing: {self.path}")
        actual_size = int(self.path.stat().st_size)
        if actual_size != self.size_bytes:
            raise UnifiedViewerError(
                f"{field_name} size changed after producer completion: "
                f"expected {self.size_bytes}, got {actual_size}"
            )
        if sha256_file(self.path) != self.sha256:
            raise UnifiedViewerError(f"{field_name} SHA-256 changed after producer completion")


@dataclass(frozen=True)
class DirectoryArtifactBinding:
    """Expected immutable directory-tree identity supplied by a completed run."""

    path: Path
    descriptor: Mapping[str, Any]

    def verify(self, *, field_name: str) -> None:
        try:
            verify_directory_tree_descriptor(self.path, self.descriptor)
        except DirectoryIntegrityError as exc:
            raise UnifiedViewerError(
                f"{field_name} tree changed after producer completion: {exc}"
            ) from exc


@dataclass(frozen=True)
class FarmEvidenceBundle:
    """Lazy click-to-inspect authority for saved COLMAP object evidence."""

    scene_state: ArtifactBinding
    mapping_root: Path
    rgbd_root: Path
    mapping_binding: DirectoryArtifactBinding | None
    rgbd_binding: DirectoryArtifactBinding | None
    provenance_note: str
    provenance_warning: bool


@dataclass(frozen=True)
class SceneSpec:
    scene_id: str
    label: str
    farm_run: Path
    source_ply: Path
    source_fingerprint: SourceFingerprint
    bridge_roots: tuple[Path, ...]
    dense_lift_roots: tuple[Path, ...]
    shaper_roots: tuple[Path, ...]
    legacy_review_mode: bool


@dataclass(frozen=True)
class ViewerRegistry:
    path: Path
    scenes: tuple[SceneSpec, ...]


def _resolve_config_path(base: Path, value: Any, *, field_name: str) -> Path:
    if not isinstance(value, str) or not value.strip():
        raise UnifiedViewerError(f"{field_name} must be a non-empty path string")
    candidate = Path(value).expanduser()
    if not candidate.is_absolute():
        candidate = base / candidate
    return candidate.resolve(strict=False)


def _path_list(base: Path, value: Any, *, field_name: str) -> tuple[Path, ...]:
    if value is None:
        return ()
    if not isinstance(value, list) or not all(isinstance(item, str) and item.strip() for item in value):
        raise UnifiedViewerError(f"{field_name} must be a list of path strings")
    return tuple(_resolve_config_path(base, item, field_name=field_name) for item in value)


def load_registry(path: Path) -> ViewerRegistry:
    path = path.expanduser().resolve(strict=True)
    try:
        raw = yaml.safe_load(path.read_text(encoding="utf-8"))
    except (OSError, yaml.YAMLError) as exc:
        raise UnifiedViewerError(f"cannot read viewer registry {path}: {exc}") from exc
    if not isinstance(raw, Mapping):
        raise UnifiedViewerError("viewer registry must be a YAML mapping")
    unknown = sorted(set(raw) - {"schema_version", "scenes"})
    if unknown:
        raise UnifiedViewerError(f"unknown viewer registry fields: {', '.join(unknown)}")
    if raw.get("schema_version") != REGISTRY_SCHEMA:
        raise UnifiedViewerError(f"viewer registry schema must be {REGISTRY_SCHEMA!r}")
    rows = raw.get("scenes")
    if not isinstance(rows, list) or not rows:
        raise UnifiedViewerError("viewer registry scenes must be a non-empty list")
    base = path.parent
    scenes: list[SceneSpec] = []
    seen: set[str] = set()
    allowed = {
        "scene_id", "label", "farm_run", "source_ply", "source_fingerprint",
        "bridge_roots", "dense_lift_roots", "shaper_roots", "legacy_review_mode",
    }
    for index, row in enumerate(rows):
        name = f"scenes[{index}]"
        if not isinstance(row, Mapping):
            raise UnifiedViewerError(f"{name} must be a mapping")
        unknown = sorted(set(row) - allowed)
        if unknown:
            raise UnifiedViewerError(f"unknown {name} fields: {', '.join(unknown)}")
        scene_id = str(row.get("scene_id") or "")
        if not _SCENE_ID_RE.fullmatch(scene_id):
            raise UnifiedViewerError(f"invalid {name}.scene_id: {scene_id!r}")
        if scene_id in seen:
            raise UnifiedViewerError(f"duplicate scene_id: {scene_id}")
        seen.add(scene_id)
        label = str(row.get("label") or scene_id).strip()
        if not label:
            raise UnifiedViewerError(f"{name}.label must be non-empty")
        legacy_review_mode = row.get("legacy_review_mode", False)
        if not isinstance(legacy_review_mode, bool):
            raise UnifiedViewerError(f"{name}.legacy_review_mode must be boolean")
        scenes.append(SceneSpec(
            scene_id=scene_id,
            label=label,
            farm_run=_resolve_config_path(base, row.get("farm_run"), field_name=f"{name}.farm_run"),
            source_ply=_resolve_config_path(base, row.get("source_ply"), field_name=f"{name}.source_ply"),
            source_fingerprint=SourceFingerprint.from_mapping(
                row.get("source_fingerprint"), field_name=f"{name}.source_fingerprint"
            ),
            bridge_roots=_path_list(base, row.get("bridge_roots"), field_name=f"{name}.bridge_roots"),
            dense_lift_roots=_path_list(
                base, row.get("dense_lift_roots"), field_name=f"{name}.dense_lift_roots"
            ),
            shaper_roots=_path_list(
                base, row.get("shaper_roots"), field_name=f"{name}.shaper_roots"
            ),
            legacy_review_mode=legacy_review_mode,
        ))
    return ViewerRegistry(path=path, scenes=tuple(scenes))


_PLY_SCALARS = {
    "char": "i1", "uchar": "u1", "int8": "i1", "uint8": "u1",
    "short": "<i2", "ushort": "<u2", "int16": "<i2", "uint16": "<u2",
    "int": "<i4", "uint": "<u4", "int32": "<i4", "uint32": "<u4",
    "float": "<f4", "float32": "<f4", "double": "<f8", "float64": "<f8",
}


@dataclass(frozen=True)
class PlyTable:
    path: Path
    count: int
    header_bytes: int
    dtype: np.dtype

    def memmap(self) -> np.memmap:
        return np.memmap(
            self.path, dtype=self.dtype, mode="r", offset=self.header_bytes, shape=(self.count,)
        )


def open_binary_ply(path: Path) -> PlyTable:
    path = path.expanduser().resolve(strict=True)
    lines: list[str] = []
    with path.open("rb") as handle:
        while True:
            raw = handle.readline()
            if not raw:
                raise UnifiedViewerError(f"truncated PLY header: {path}")
            try:
                line = raw.decode("ascii").rstrip("\r\n")
            except UnicodeDecodeError as exc:
                raise UnifiedViewerError(f"non-ASCII PLY header: {path}") from exc
            lines.append(line)
            if line == "end_header":
                header_bytes = handle.tell()
                break
            if len(lines) > 4096:
                raise UnifiedViewerError(f"unreasonably large PLY header: {path}")
    if "format binary_little_endian 1.0" not in lines:
        raise UnifiedViewerError("only binary_little_endian PLY is supported")
    count: int | None = None
    properties: list[tuple[str, str]] = []
    in_vertices = False
    for line in lines:
        parts = line.split()
        if parts[:2] == ["element", "vertex"] and len(parts) == 3:
            count = int(parts[2])
            in_vertices = True
        elif parts[:1] == ["element"]:
            in_vertices = False
        elif in_vertices and parts[:1] == ["property"]:
            if len(parts) != 3 or parts[1] == "list" or parts[1] not in _PLY_SCALARS:
                raise UnifiedViewerError(f"unsupported PLY vertex property: {line}")
            properties.append((parts[2], _PLY_SCALARS[parts[1]]))
    if count is None or count <= 0:
        raise UnifiedViewerError("PLY has no positive vertex element")
    dtype = np.dtype(properties, align=False)
    required = {
        "x", "y", "z", "f_dc_0", "f_dc_1", "f_dc_2", "opacity",
        "scale_0", "scale_1", "scale_2", "rot_0", "rot_1", "rot_2", "rot_3",
    }
    missing = sorted(required - set(dtype.names or ()))
    if missing:
        raise UnifiedViewerError(f"source is not a complete Graphdeco 3DGS PLY; missing {missing}")
    expected = header_bytes + int(count) * dtype.itemsize
    if path.stat().st_size < expected:
        raise UnifiedViewerError(f"truncated PLY body: expected at least {expected} bytes")
    return PlyTable(path=path, count=int(count), header_bytes=header_bytes, dtype=dtype)


def _inside(path: Path, parent: Path) -> bool:
    try:
        path.relative_to(parent)
        return True
    except ValueError:
        return False


def _resolve_inside(
    parent: Path,
    relative: str,
    *,
    field_name: str,
    guard_root: Path | None = None,
) -> Path:
    candidate = (parent / relative).resolve(strict=False)
    guard = (guard_root or parent).resolve()
    if not _inside(candidate, guard):
        raise UnifiedViewerError(f"{field_name} escapes its artifact root")
    return candidate


def _preflight_source_fingerprint(
    run_dir: Path,
    source_path: Path,
    configured: SourceFingerprint,
) -> Mapping[str, Any]:
    """Bind a FARM run's recorded Gaussian input to the configured source file.

    The registry and the producer are allowed to use different supported hash
    algorithms. This is needed for old registries that pin the historical
    sampled digest while current FARM runs record a full SHA-256 for the same
    release-critical PLY. Algorithm conversion is never inferred from a path:
    both fingerprints are verified against the actual configured file.
    """

    report = _load_json(run_dir / "input" / "scene_preflight.json")
    checks = [
        check for check in report.get("checks", [])
        if isinstance(check, Mapping) and check.get("name") == "fingerprints"
    ]
    if len(checks) != 1:
        raise UnifiedViewerError(
            "FARM scene preflight must contain exactly one fingerprints check "
            f"(got {len(checks)})"
        )
    check = checks[0]
    if str(check.get("status", "")).lower() != "pass":
        raise UnifiedViewerError("FARM scene-preflight fingerprints check did not pass")
    metrics = check.get("metrics")
    files = metrics.get("files") if isinstance(metrics, Mapping) else None
    if not isinstance(files, list):
        raise UnifiedViewerError("FARM scene-preflight fingerprints files must be a list")

    canonical_source = source_path.expanduser().resolve(strict=True)
    path_matches: list[Mapping[str, Any]] = []
    for row in files:
        if not isinstance(row, Mapping):
            continue
        recorded_path = row.get("path")
        if not isinstance(recorded_path, str) or not recorded_path.strip():
            continue
        candidate = Path(recorded_path).expanduser()
        if not candidate.is_absolute():
            # Preflight records resolved absolute inputs. Treat a relative
            # value as malformed instead of guessing a base directory.
            continue
        if candidate.resolve(strict=False) == canonical_source:
            path_matches.append(row)
    if len(path_matches) != 1:
        raise UnifiedViewerError(
            "FARM run is not bound to the configured source PLY path "
            f"({len(path_matches)} exact records)"
        )

    row = path_matches[0]
    required = {"algorithm", "digest", "size_bytes", "chunk_bytes", "sampled_offsets"}
    missing = sorted(required - set(row))
    if missing:
        raise UnifiedViewerError(
            "FARM source PLY fingerprint record is incomplete: " + ", ".join(missing)
        )
    producer = SourceFingerprint.from_mapping(
        {name: row[name] for name in required},
        field_name="FARM source PLY fingerprint",
    )
    if producer != configured:
        producer.validate(canonical_source)
    return row


def _validate_source_snapshot(run_dir: Path, manifest: Mapping[str, Any]) -> tuple[str, bool]:
    snapshot = manifest.get("source_snapshot")
    if snapshot is None:
        return "legacy run: no immutable execution-source snapshot", True
    if not isinstance(snapshot, Mapping):
        raise UnifiedViewerError("run source_snapshot must be a mapping")
    try:
        from .source_snapshot import validated_source_snapshot_project_root

        root = validated_source_snapshot_project_root(run_dir, required=True)
    except Exception as exc:
        raise UnifiedViewerError(f"execution-source snapshot validation failed: {exc}") from exc
    if root is None:
        raise UnifiedViewerError("execution-source snapshot is absent")
    tree = str(snapshot.get("tree_sha256") or "")
    return f"immutable source tree {tree[:12]}", False


@dataclass(frozen=True)
class FarmBundle:
    run_dir: Path
    run_id: str
    config_sha256: str
    success_sha256: str
    catalog_path: Path
    catalog_sha256: str
    preview_path: Path
    preview_sha256: str
    resolved_context_path: Path
    resolved_context_sha256: str
    catalog_binding: ArtifactBinding
    preview_binding: ArtifactBinding
    context_binding: ArtifactBinding | None
    integrity_note: str
    integrity_warning: bool
    world_up: np.ndarray
    meters_per_scene_unit: float
    evidence: FarmEvidenceBundle | None
    evidence_unavailable_reason: str
    catalog: tuple[Mapping[str, Any], ...]
    source_note: str
    source_warning: bool


@dataclass(frozen=True)
class BridgeBundle:
    root: Path
    instance_cloud: Path
    segmentation_report: Path
    instance_rows: Mapping[int, Mapping[str, Any]]
    provenance_warning: str


@dataclass(frozen=True)
class DenseLiftBundle:
    root: Path
    result_path: Path
    result: Mapping[str, Any]
    result_sha256: str
    bank_path: Path
    bank_sha256: str
    object_id_path: Path
    confidence_path: Path
    timestamp_support_path: Path
    status_path: Path
    artifact_bindings: tuple[ArtifactBinding, ...]
    source_full_sha256: str
    verified_gaussians: int
    verified_objects: int
    release_eligible: bool


@dataclass(frozen=True)
class ShaperBundle:
    root: Path
    scene_manifest_path: Path
    scene_manifest_sha256: str
    viewer_assets_path: Path
    viewer_assets_sha256: str
    mesh_npz: ArtifactBinding
    combined_gltf: ArtifactBinding
    artifact_bindings: tuple[ArtifactBinding, ...]
    manifest: Mapping[str, Any]
    viewer_assets: Mapping[str, Any]
    object_rows: Mapping[int, Mapping[str, Any]]


@dataclass(frozen=True)
class LegacyShaperBundle:
    root: Path
    scene_manifest: ArtifactBinding
    viewer_assets: ArtifactBinding
    asset_bindings: Mapping[int, ArtifactBinding]
    manifest: Mapping[str, Any]
    object_rows: Mapping[int, Mapping[str, Any]]
    provenance_warning: str


@dataclass(frozen=True)
class ObjectEvidenceFrame:
    image: np.ndarray
    image_id: int
    camera_id: str
    score: float
    bbox_xyxy: tuple[int, int, int, int] | None
    source_filename: str
    source_kind: str


@dataclass(frozen=True)
class ObjectEvidenceGallery:
    object_id: int
    frames: tuple[ObjectEvidenceFrame, ...]
    provenance_note: str
    provenance_warning: bool

@dataclass(frozen=True)
class LayerAvailability:
    ready: bool
    reason: str
    warning: bool = False

    def as_dict(self) -> dict[str, Any]:
        return {"ready": self.ready, "reason": self.reason, "warning": self.warning}


@dataclass
class ValidatedScene:
    spec: SceneSpec
    farm: FarmBundle
    source_table: PlyTable
    bridge: BridgeBundle | None
    dense_lift: DenseLiftBundle | None
    review_lift: DenseLiftBundle | None
    shaper: ShaperBundle | None
    review_shaper: ShaperBundle | None
    legacy_shaper: LegacyShaperBundle | None
    layers: dict[str, LayerAvailability]

    def as_dict(self) -> dict[str, Any]:
        return {
            "scene_id": self.spec.scene_id,
            "label": self.spec.label,
            "status": "ready",
            "farm": {
                "run_id": self.farm.run_id,
                "config_sha256": self.farm.config_sha256,
                "success_sha256": self.farm.success_sha256,
                "catalog_sha256": self.farm.catalog_sha256,
                "preview_sha256": self.farm.preview_sha256,
                "source_snapshot": self.farm.source_note,
            },
            "source_3dgs": {
                "path": str(self.source_table.path),
                "vertices": self.source_table.count,
                "bytes": self.source_table.path.stat().st_size,
                "fingerprint_algorithm": self.spec.source_fingerprint.algorithm,
                "fingerprint": self.spec.source_fingerprint.digest,
                "rendering": "viser-all-N-float16-uint8-quantized-dc-preview",
                "higher_order_sh": False,
                "packed_bytes_per_client": self.source_table.count * VISER_PACKED_BYTES_PER_GAUSSIAN,
            },
            "layers": {name: state.as_dict() for name, state in self.layers.items()},
        }


@dataclass(frozen=True)
class SceneValidationFailure:
    scene_id: str
    label: str
    reason: str

    def as_dict(self) -> dict[str, Any]:
        return {"scene_id": self.scene_id, "label": self.label, "status": "invalid", "reason": self.reason}


@dataclass(frozen=True)
class RegistryValidation:
    registry: ViewerRegistry
    scenes: tuple[ValidatedScene | SceneValidationFailure, ...]

    @property
    def ready_scenes(self) -> tuple[ValidatedScene, ...]:
        return tuple(item for item in self.scenes if isinstance(item, ValidatedScene))

    @property
    def ok(self) -> bool:
        return len(self.ready_scenes) == len(self.scenes)

    def as_dict(self) -> dict[str, Any]:
        return {
            "schema_version": "farm.unified-viewer-validation.v1",
            "status": "PASS" if self.ok else "FAIL",
            "registry": str(self.registry.path),
            "scenes": [item.as_dict() for item in self.scenes],
        }


def _binding_from_descriptor(
    descriptor_root: Path,
    descriptor: Any,
    *,
    field_name: str,
    guard_root: Path,
    expected_path: Path | None = None,
) -> ArtifactBinding:
    if not isinstance(descriptor, Mapping) or descriptor.get("kind", "file") != "file":
        raise UnifiedViewerError(f"{field_name} immutable file descriptor is absent")
    path = _resolve_inside(
        descriptor_root,
        str(descriptor.get("path") or ""),
        field_name=field_name,
        guard_root=guard_root,
    )
    digest = str(descriptor.get("sha256") or "")
    try:
        size = int(descriptor["bytes"])
    except (KeyError, TypeError, ValueError) as exc:
        raise UnifiedViewerError(f"{field_name} descriptor size is invalid") from exc
    if not _SHA256_RE.fullmatch(digest) or size <= 0:
        raise UnifiedViewerError(f"{field_name} descriptor identity is invalid")
    if expected_path is not None and path != expected_path.resolve():
        raise UnifiedViewerError(f"{field_name} descriptor resolves to an unexpected path")
    binding = ArtifactBinding(path=path, size_bytes=size, sha256=digest)
    binding.verify(field_name=field_name)
    return binding


def _farm_artifact_bindings(
    run_dir: Path,
    success: Mapping[str, Any],
    *,
    catalog_path: Path,
    preview_path: Path,
    context_path: Path,
    allow_unbound_review: bool,
) -> tuple[ArtifactBinding, ArtifactBinding, ArtifactBinding | None, str, bool]:
    """Bind viewer inputs to producer-recorded expected hashes.

    Current runs use a success-marker hash of ``viewer/bundle.json`` and the
    bundle's ``artifact_integrity`` inventory.  Frozen v7 runs may use their
    signed-off self-containment inventory as an explicitly legacy fallback.
    """

    bundle_path = _resolve_inside(
        run_dir,
        str(success.get("viewer_bundle") or ""),
        field_name="FARM viewer bundle",
        guard_root=run_dir,
    )
    bundle_sha = str(success.get("viewer_bundle_sha256") or "")
    if bundle_sha:
        if not _SHA256_RE.fullmatch(bundle_sha) or sha256_file(bundle_path) != bundle_sha:
            raise UnifiedViewerError("FARM viewer bundle SHA-256 mismatch")
        bundle = _load_json(bundle_path)
        if (
            bundle.get("schema") != "farm.viewer-bundle.v1"
            or bundle.get("scene_id") != success.get("scene_id")
            or bundle.get("run_id") != success.get("run_id")
        ):
            raise UnifiedViewerError("FARM viewer bundle schema/scene/run mismatch")
        artifacts = bundle.get("artifacts")
        integrity = bundle.get("artifact_integrity")
        if not isinstance(artifacts, Mapping) or not isinstance(integrity, Mapping):
            raise UnifiedViewerError("FARM viewer bundle has no artifact_integrity inventory")

        def current(name: str, expected: Path) -> ArtifactBinding:
            descriptor = integrity.get(name)
            if not isinstance(descriptor, Mapping) or artifacts.get(name) != descriptor.get("path"):
                raise UnifiedViewerError(f"FARM viewer bundle {name} path/integrity mismatch")
            return _binding_from_descriptor(
                bundle_path.parent,
                descriptor,
                field_name=f"FARM {name}",
                guard_root=run_dir,
                expected_path=expected,
            )

        catalog = current("presentation_catalog", catalog_path)
        preview = current("cloud", preview_path)
        context = current("resolved_context", context_path)
        return catalog, preview, context, f"viewer bundle {bundle_sha[:12]} exact inventory", False

    report_relative = success.get("self_containment")
    if not isinstance(report_relative, str) or not report_relative:
        if allow_unbound_review:
            def observed(path: Path) -> ArtifactBinding:
                if not path.is_file():
                    raise UnifiedViewerError(f"legacy review artifact is missing: {path.name}")
                return ArtifactBinding(
                    path=path,
                    size_bytes=int(path.stat().st_size),
                    sha256=sha256_file(path),
                )

            return (
                observed(catalog_path), observed(preview_path), observed(context_path),
                "LEGACY REVIEW: viewer files current-hashed at startup; no producer inventory",
                True,
            )
        raise UnifiedViewerError(
            "FARM success has neither viewer_bundle_sha256 nor a legacy self-containment inventory"
        )
    report_path = _resolve_inside(
        run_dir,
        report_relative,
        field_name="FARM self-containment report",
        guard_root=run_dir,
    )
    report = _load_json(report_path)
    if (
        report.get("schema") != "farm.run-self-containment.v1"
        or report.get("status") != "PASS"
        or report.get("scene_id") != success.get("scene_id")
        or report.get("run_id") != success.get("run_id")
    ):
        raise UnifiedViewerError("legacy FARM self-containment inventory schema/status/run mismatch")
    rows = report.get("required_artifacts")
    if not isinstance(rows, list):
        raise UnifiedViewerError("legacy FARM self-containment artifact inventory is absent")
    by_path = {
        str(row.get("path")): row
        for row in rows
        if isinstance(row, Mapping) and row.get("exists") is True
    }

    def legacy(expected: Path, name: str, *, required: bool) -> ArtifactBinding | None:
        try:
            relative = expected.resolve().relative_to(run_dir.resolve()).as_posix()
        except ValueError as exc:
            raise UnifiedViewerError(f"FARM {name} is outside the run") from exc
        descriptor = by_path.get(relative)
        if descriptor is None:
            if required:
                raise UnifiedViewerError(f"legacy FARM inventory does not bind {relative}")
            return None
        normalized = dict(descriptor)
        normalized["kind"] = "file"
        return _binding_from_descriptor(
            run_dir,
            normalized,
            field_name=f"legacy FARM {name}",
            guard_root=run_dir,
            expected_path=expected,
        )

    catalog = legacy(catalog_path, "presentation_catalog", required=True)
    preview = legacy(preview_path, "cloud", required=True)
    context = legacy(context_path, "resolved_context", required=False)
    assert catalog is not None and preview is not None
    return (
        catalog,
        preview,
        context,
        f"legacy self-containment inventory {sha256_file(report_path)[:12]}",
        True,
    )


def _farm_evidence_bundle(
    run_dir: Path,
    success: Mapping[str, Any],
    *,
    allow_unbound_review: bool,
) -> tuple[FarmEvidenceBundle | None, str]:
    """Resolve saved scene-state/crop inputs without decoding any images."""

    try:
        bundle_path = _resolve_inside(
            run_dir,
            str(success.get("viewer_bundle") or ""),
            field_name="FARM viewer bundle",
            guard_root=run_dir,
        )
        bundle = _load_json(bundle_path)
        artifacts = bundle.get("artifacts")
        if not isinstance(artifacts, Mapping):
            raise UnifiedViewerError("FARM viewer bundle has no artifact mapping")
        required = ("scene_state", "mapping", "rgbd")
        if any(not isinstance(artifacts.get(name), str) for name in required):
            return None, "run bundle has no scene_state/mapping/rgbd evidence artifacts"
        scene_path = _resolve_inside(
            bundle_path.parent,
            str(artifacts["scene_state"]),
            field_name="FARM scene_state",
            guard_root=run_dir,
        )
        mapping_path = _resolve_inside(
            bundle_path.parent,
            str(artifacts["mapping"]),
            field_name="FARM mapping",
            guard_root=run_dir,
        )
        rgbd_path = _resolve_inside(
            bundle_path.parent,
            str(artifacts["rgbd"]),
            field_name="FARM rgbd",
            guard_root=run_dir,
        )
        if not mapping_path.is_dir() or not rgbd_path.is_dir():
            raise UnifiedViewerError("FARM mapping/rgbd evidence directories are missing")

        bundle_sha = str(success.get("viewer_bundle_sha256") or "")
        if bundle_sha:
            if not _SHA256_RE.fullmatch(bundle_sha) or sha256_file(bundle_path) != bundle_sha:
                raise UnifiedViewerError("FARM viewer bundle SHA-256 mismatch")
            integrity = bundle.get("artifact_integrity")
            if not isinstance(integrity, Mapping):
                raise UnifiedViewerError("FARM viewer bundle has no artifact_integrity inventory")
            scene_binding = _binding_from_descriptor(
                bundle_path.parent,
                integrity.get("scene_state"),
                field_name="FARM scene_state",
                guard_root=run_dir,
                expected_path=scene_path,
            )
            mapping_binding = _directory_binding_from_descriptor(
                bundle_path, run_dir, artifacts, integrity, "mapping"
            )
            rgbd_binding = _directory_binding_from_descriptor(
                bundle_path, run_dir, artifacts, integrity, "rgbd"
            )
            return FarmEvidenceBundle(
                scene_state=scene_binding,
                mapping_root=mapping_path,
                rgbd_root=rgbd_path,
                mapping_binding=mapping_binding,
                rgbd_binding=rgbd_binding,
                provenance_note=f"exact viewer bundle {bundle_sha[:12]} directory trees; verified lazily",
                provenance_warning=False,
            ), ""

        report_relative = success.get("self_containment")
        if not isinstance(report_relative, str) or not report_relative:
            if allow_unbound_review:
                scene_binding = ArtifactBinding(
                    path=scene_path,
                    size_bytes=int(scene_path.stat().st_size),
                    sha256=sha256_file(scene_path),
                )
                return FarmEvidenceBundle(
                    scene_state=scene_binding,
                    mapping_root=mapping_path,
                    rgbd_root=rgbd_path,
                    mapping_binding=None,
                    rgbd_binding=None,
                    provenance_note=(
                        "LEGACY REVIEW: scene_state current-hashed at startup; "
                        "crop/RGB directory trees were not producer-signed"
                    ),
                    provenance_warning=True,
                ), ""
            raise UnifiedViewerError("legacy run has no self-containment inventory")
        report_path = _resolve_inside(
            run_dir,
            report_relative,
            field_name="FARM self-containment report",
            guard_root=run_dir,
        )
        report = _load_json(report_path)
        rows = report.get("required_artifacts")
        if not isinstance(rows, list):
            raise UnifiedViewerError("legacy self-containment artifact inventory is absent")
        try:
            relative_scene = scene_path.relative_to(run_dir).as_posix()
        except ValueError as exc:
            raise UnifiedViewerError("legacy scene_state escapes the FARM run") from exc
        descriptor = next(
            (
                row for row in rows
                if isinstance(row, Mapping)
                and row.get("exists") is True
                and row.get("path") == relative_scene
            ),
            None,
        )
        if descriptor is None:
            raise UnifiedViewerError("legacy inventory does not bind scene_state")
        normalized = dict(descriptor)
        normalized["kind"] = "file"
        scene_binding = _binding_from_descriptor(
            run_dir,
            normalized,
            field_name="legacy FARM scene_state",
            guard_root=run_dir,
            expected_path=scene_path,
        )
        return FarmEvidenceBundle(
            scene_state=scene_binding,
            mapping_root=mapping_path,
            rgbd_root=rgbd_path,
            mapping_binding=None,
            rgbd_binding=None,
            provenance_note=(
                f"legacy scene_state inventory {scene_binding.sha256[:12]}; "
                "crop/RGB directory trees were not producer-signed"
            ),
            provenance_warning=True,
        ), ""
    except (UnifiedViewerError, OSError, KeyError, ValueError) as exc:
        return None, str(exc)


def _directory_binding_from_descriptor(
    bundle_path: Path,
    run_dir: Path,
    artifacts: Mapping[str, Any],
    integrity: Mapping[str, Any],
    name: str,
) -> DirectoryArtifactBinding:
    relative = artifacts.get(name)
    descriptor = integrity.get(name)
    if (
        not isinstance(relative, str)
        or not isinstance(descriptor, Mapping)
        or descriptor.get("kind") != "directory"
        or descriptor.get("path") != relative
        or descriptor.get("algorithm") != "farm.directory-tree.v1"
    ):
        raise UnifiedViewerError(f"FARM viewer bundle has no immutable {name} directory")
    path = _resolve_inside(
        bundle_path.parent,
        relative,
        field_name=f"FARM {name}",
        guard_root=run_dir,
    )
    try:
        files = int(descriptor["files"])
        size = int(descriptor["bytes"])
    except (KeyError, TypeError, ValueError) as exc:
        raise UnifiedViewerError(f"FARM {name} directory descriptor is invalid") from exc
    tree_sha = str(descriptor.get("tree_sha256") or "")
    if files < 1 or size < 1 or not _SHA256_RE.fullmatch(tree_sha):
        raise UnifiedViewerError(f"FARM {name} directory identity is invalid")
    if path.is_symlink() or not path.is_dir():
        raise UnifiedViewerError(f"FARM {name} directory is missing or a symlink")
    return DirectoryArtifactBinding(path=path, descriptor=dict(descriptor))


def _validate_farm(spec: SceneSpec) -> tuple[FarmBundle, PlyTable]:
    run_dir = spec.farm_run.resolve(strict=True)
    if not run_dir.is_dir():
        raise UnifiedViewerError(f"FARM run is not a directory: {run_dir}")
    success_path = run_dir / "_SUCCESS.json"
    manifest_path = run_dir / "manifest.json"
    success = _load_json(success_path)
    manifest = _load_json(manifest_path)
    if success.get("schema") != "farm.pipeline-success.v1" or success.get("status") != "success":
        raise UnifiedViewerError("FARM success marker has the wrong schema/status")
    if manifest.get("schema") != "farm.pipeline-run.v1" or manifest.get("status") != "success":
        raise UnifiedViewerError("FARM manifest is not a successful pipeline run")
    for field_name in ("scene_id", "run_id", "config_sha256"):
        if success.get(field_name) != manifest.get(field_name):
            raise UnifiedViewerError(f"FARM success/manifest mismatch: {field_name}")
    if manifest.get("scene_id") != spec.scene_id:
        raise UnifiedViewerError(
            f"registry scene {spec.scene_id!r} points to FARM scene {manifest.get('scene_id')!r}"
        )
    config_sha = str(manifest.get("config_sha256") or "")
    if not _SHA256_RE.fullmatch(config_sha):
        raise UnifiedViewerError("FARM config_sha256 is invalid")
    final = _load_json(run_dir / "final" / "result.json")
    viewer = _load_json(run_dir / "viewer" / "result.json")
    if final.get("status") != "PASS" or final.get("scene_id") != spec.scene_id:
        raise UnifiedViewerError("FARM final result is not PASS for this scene")
    if viewer.get("schema") != "farm.viewer-result.v1" or viewer.get("status") != "PASS":
        raise UnifiedViewerError("FARM viewer result is not PASS")
    artifacts = viewer.get("artifacts")
    if not isinstance(artifacts, Mapping):
        raise UnifiedViewerError("FARM viewer result has no artifact mapping")
    viewer_dir = run_dir / "viewer"
    catalog_path = _resolve_inside(
        viewer_dir,
        str(artifacts.get("catalog") or ""),
        field_name="catalog",
        guard_root=run_dir,
    )
    preview_path = _resolve_inside(
        viewer_dir,
        str(artifacts.get("cloud") or ""),
        field_name="cloud",
        guard_root=run_dir,
    )
    if not catalog_path.is_file() or not preview_path.is_file():
        raise UnifiedViewerError("FARM viewer catalog/cloud artifacts are missing")
    presentation_catalog = run_dir / "final" / "presentation_catalog.json"
    if presentation_catalog.is_file():
        catalog_path = presentation_catalog
    context_path = run_dir / "input" / "resolved_context.json"
    catalog_binding, preview_binding, context_binding, integrity_note, integrity_warning = (
        _farm_artifact_bindings(
            run_dir,
            success,
            catalog_path=catalog_path,
            preview_path=preview_path,
            context_path=context_path,
            allow_unbound_review=spec.legacy_review_mode,
        )
    )
    catalog_raw = _load_json(catalog_path, mapping=False)
    catalog: list[Mapping[str, Any]] = []
    ids: set[int] = set()
    for row in catalog_raw:
        if not isinstance(row, Mapping):
            raise UnifiedViewerError("FARM catalog rows must be objects")
        try:
            object_id = int(row["id"])
            center = np.asarray(row["center_m"], dtype=np.float64).reshape(3)
            dimensions = np.asarray(row["dimensions_m"], dtype=np.float64).reshape(3)
            wxyz = np.asarray(row["wxyz"], dtype=np.float64).reshape(4)
        except (KeyError, TypeError, ValueError) as exc:
            raise UnifiedViewerError("invalid FARM catalog geometry row") from exc
        if object_id < 0 or object_id in ids or not np.isfinite(center).all() or not np.isfinite(wxyz).all():
            raise UnifiedViewerError("invalid/duplicate FARM object geometry")
        if not np.isfinite(dimensions).all() or np.any(dimensions <= 0) or np.linalg.norm(wxyz) < 1e-8:
            raise UnifiedViewerError("invalid FARM object dimensions/quaternion")
        ids.add(object_id)
        catalog.append(row)
    if not catalog:
        raise UnifiedViewerError("FARM presentation catalog is empty")
    try:
        with np.load(preview_path, allow_pickle=False) as archive:
            xyz = archive["xyz"]
            rgb = archive["rgb"]
            if xyz.ndim != 2 or xyz.shape[1] != 3 or rgb.shape != xyz.shape:
                raise UnifiedViewerError("FARM preview cloud arrays have invalid shapes")
            if xyz.dtype != np.float32 or rgb.dtype != np.uint8 or len(xyz) == 0:
                raise UnifiedViewerError("FARM preview cloud arrays have invalid dtypes/length")
    except (OSError, KeyError, ValueError) as exc:
        if isinstance(exc, UnifiedViewerError):
            raise
        raise UnifiedViewerError(f"cannot validate FARM preview cloud: {exc}") from exc
    context = _load_json(context_path)
    try:
        world_up = np.asarray(context["resolved_up"], dtype=np.float64).reshape(3)
        scale = float(context["meters_per_scene_unit"])
    except (KeyError, TypeError, ValueError) as exc:
        raise UnifiedViewerError("invalid FARM resolved coordinate context") from exc
    if not np.isfinite(world_up).all() or np.linalg.norm(world_up) < 1e-8 or not math.isfinite(scale) or scale <= 0:
        raise UnifiedViewerError("invalid FARM world-up or metric scale")
    world_up = world_up / np.linalg.norm(world_up)
    spec.source_fingerprint.validate(spec.source_ply)
    _preflight_source_fingerprint(run_dir, spec.source_ply, spec.source_fingerprint)
    source_table = open_binary_ply(spec.source_ply)
    source_note, source_warning = _validate_source_snapshot(run_dir, manifest)
    evidence, evidence_reason = _farm_evidence_bundle(
        run_dir, success, allow_unbound_review=spec.legacy_review_mode
    )
    return FarmBundle(
        run_dir=run_dir,
        run_id=str(manifest["run_id"]),
        config_sha256=config_sha,
        success_sha256=sha256_file(success_path),
        catalog_path=catalog_path,
        catalog_sha256=catalog_binding.sha256,
        preview_path=preview_path,
        preview_sha256=preview_binding.sha256,
        resolved_context_path=context_path,
        resolved_context_sha256=(
            context_binding.sha256 if context_binding is not None else sha256_file(context_path)
        ),
        catalog_binding=catalog_binding,
        preview_binding=preview_binding,
        context_binding=context_binding,
        integrity_note=integrity_note,
        integrity_warning=integrity_warning,
        world_up=world_up.astype(np.float32),
        meters_per_scene_unit=scale,
        evidence=evidence,
        evidence_unavailable_reason=evidence_reason,
        catalog=tuple(catalog),
        source_note=source_note,
        source_warning=source_warning,
    ), source_table


def _candidate_dirs(roots: Sequence[Path]) -> Iterable[Path]:
    seen: set[Path] = set()
    for root in roots:
        candidates = [root, root / "latest"]
        if root.is_dir():
            try:
                candidates.extend(
                    sorted((item for item in root.iterdir() if item.is_dir()), key=lambda item: item.name, reverse=True)
                )
            except OSError:
                pass
        for candidate in candidates:
            resolved = candidate.resolve(strict=False)
            if resolved not in seen:
                seen.add(resolved)
                yield resolved


def _legacy_fingerprint_matches(path: Path, expected: Any) -> bool:
    if not isinstance(expected, Mapping):
        return False
    actual = _legacy_bridge_fingerprint(path)
    return all(actual.get(key) == expected.get(key) for key in ("bytes", "mtime_ns", "sample_sha256"))


def _validate_bridge_candidate(root: Path, spec: SceneSpec, farm: FarmBundle) -> BridgeBundle:
    result_path = root / "result.json"
    report_path = root / "instances" / "segmentation_report.json"
    cloud_path = root / "instances" / "instance_visual_cloud.npz"
    result = _load_json(result_path)
    report = _load_json(report_path)
    if result.get("schema") != "farm-shaper-bridge.result.v1":
        raise UnifiedViewerError("unsupported legacy bridge result schema")
    if result.get("status") not in {"PASS", "REVIEW"} or report.get("status") != "PASS":
        raise UnifiedViewerError("legacy bridge result/segmentation did not pass")
    source_run = Path(str(result.get("source_run") or "")).expanduser().resolve(strict=False)
    current_success = (farm.run_dir / "_SUCCESS.json").resolve()
    inputs = report.get("inputs")
    if not isinstance(inputs, Mapping):
        raise UnifiedViewerError("legacy bridge segmentation has no input provenance")
    farm_input = inputs.get("farm_run")
    source_input = inputs.get("source_ply")
    recorded_farm_path = (
        Path(str(farm_input.get("path") or "")).expanduser().resolve(strict=False)
        if isinstance(farm_input, Mapping)
        else None
    )
    if recorded_farm_path is None or source_run != recorded_farm_path:
        raise UnifiedViewerError("legacy bridge result/report FARM paths are internally inconsistent")
    if not _legacy_fingerprint_matches(current_success, farm_input):
        raise UnifiedViewerError("legacy bridge belongs to a different FARM run")
    if not _legacy_fingerprint_matches(spec.source_ply, source_input):
        raise UnifiedViewerError("legacy bridge belongs to a different source PLY")
    with np.load(cloud_path, allow_pickle=False) as archive:
        xyz = archive["xyz"]
        rgb = archive["rgb"]
        object_id = archive["object_id"]
        if xyz.ndim != 2 or xyz.shape[1] != 3 or rgb.shape != xyz.shape or object_id.shape != (len(xyz),):
            raise UnifiedViewerError("legacy bridge instance cloud arrays have invalid shapes")
        if xyz.dtype != np.float32 or rgb.dtype != np.uint8 or object_id.dtype != np.int32:
            raise UnifiedViewerError("legacy bridge instance cloud arrays have invalid dtypes")
    objects = report.get("objects")
    rows = objects.get("rows") if isinstance(objects, Mapping) else None
    if not isinstance(rows, list):
        raise UnifiedViewerError("legacy bridge segmentation object rows are absent")
    by_id = {int(row["id"]): row for row in rows if isinstance(row, Mapping) and "id" in row}
    return BridgeBundle(
        root=root,
        instance_cloud=cloud_path,
        segmentation_report=report_path,
        instance_rows=by_id,
        provenance_warning=(
            "legacy instance preview only; both FARM and source sampled fingerprints matched, "
            "but legacy ShapeR meshes are intentionally not treated as canonical"
        ),
    )


def _find_bridge(spec: SceneSpec, farm: FarmBundle) -> tuple[BridgeBundle | None, str]:
    if not spec.bridge_roots:
        return None, "not configured"
    reasons: list[str] = []
    for candidate in _candidate_dirs(spec.bridge_roots):
        if not (candidate / "result.json").is_file():
            reasons.append(f"{candidate}: missing result.json")
            continue
        try:
            return _validate_bridge_candidate(candidate, spec, farm), "ready (legacy review provenance)"
        except (UnifiedViewerError, OSError, KeyError, ValueError) as exc:
            reasons.append(f"{candidate}: {exc}")
    return None, "; ".join(reasons[-3:]) or "no compatible bridge output"


def _artifact_path(root: Path, descriptor: Any, *, field_name: str) -> tuple[Path, str, int]:
    if not isinstance(descriptor, Mapping):
        raise UnifiedViewerError(f"dense lift {field_name} descriptor is absent")
    path = _resolve_inside(root, str(descriptor.get("path") or ""), field_name=field_name)
    digest = str(descriptor.get("sha256") or "")
    try:
        size = int(descriptor["bytes"])
    except (KeyError, TypeError, ValueError) as exc:
        raise UnifiedViewerError(f"dense lift {field_name} size is invalid") from exc
    if size <= 0 or not _SHA256_RE.fullmatch(digest) or not path.is_file() or path.stat().st_size != size:
        raise UnifiedViewerError(f"dense lift {field_name} descriptor/file mismatch")
    return path, digest, size


def _validate_verified_bank(
    bank_path: Path,
    *,
    object_id_path: Path,
    confidence_path: Path,
    timestamp_support_path: Path,
    status_path: Path,
    source_gaussian_count: int,
    expected_gaussians: int,
    expected_objects: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Validate the canonical verified-only CSR and its dense source-order authority."""

    try:
        with np.load(bank_path, allow_pickle=False) as archive:
            if set(archive.files) != _VERIFIED_BANK_FIELDS:
                raise UnifiedViewerError("verified CSR fields differ from the v1 contract")
            object_ids = np.asarray(archive["object_ids"])
            indptr = np.asarray(archive["indptr"])
            indices = np.asarray(archive["indices"])
            confidence = np.asarray(archive["confidence"])
            support = np.asarray(archive["timestamp_support"])
    except (OSError, KeyError, ValueError) as exc:
        if isinstance(exc, UnifiedViewerError):
            raise
        raise UnifiedViewerError(f"cannot read verified instance bank: {exc}") from exc

    expected_dtypes = (
        np.dtype(np.int32), np.dtype(np.int64), np.dtype(np.int64),
        np.dtype(np.float32), np.dtype(np.uint16),
    )
    if tuple(value.dtype for value in (object_ids, indptr, indices, confidence, support)) != expected_dtypes:
        raise UnifiedViewerError("verified CSR dtypes differ from the v1 contract")
    if any(value.ndim != 1 for value in (object_ids, indptr, indices, confidence, support)):
        raise UnifiedViewerError("verified CSR arrays must all be one-dimensional")
    if (
        len(object_ids) != expected_objects
        or len(indices) != expected_gaussians
        or indptr.shape != (len(object_ids) + 1,)
        or len(indptr) == 0
        or int(indptr[0]) != 0
        or int(indptr[-1]) != len(indices)
        or np.any(np.diff(indptr) <= 0)
    ):
        raise UnifiedViewerError("verified CSR counts/pointers differ from the lift result")
    if confidence.shape != indices.shape or support.shape != indices.shape:
        raise UnifiedViewerError("verified CSR value arrays do not match indices")
    if len(object_ids) and (int(object_ids[0]) < 0 or np.any(np.diff(object_ids) <= 0)):
        raise UnifiedViewerError("verified CSR object IDs must be sorted, unique, and non-negative")
    if np.any(indices < 0) or np.any(indices >= int(source_gaussian_count)):
        raise UnifiedViewerError("verified CSR contains source indices outside the PLY")
    for start, end in zip(indptr[:-1], indptr[1:]):
        segment = indices[int(start):int(end)]
        if len(segment) > 1 and np.any(np.diff(segment) <= 0):
            raise UnifiedViewerError("each verified CSR slice must be strictly source-order sorted")
    if len(indices) != len(np.unique(indices)):
        raise UnifiedViewerError("one source Gaussian occurs in multiple verified objects")
    if not np.isfinite(confidence).all() or np.any((confidence <= 0) | (confidence > 1)):
        raise UnifiedViewerError("verified CSR confidence must be finite in (0,1]")
    if np.any(support == 0):
        raise UnifiedViewerError("verified CSR timestamp support must be positive")

    sidecars = {
        "object_id": (object_id_path, np.dtype(np.int32)),
        "confidence": (confidence_path, np.dtype(np.float32)),
        "timestamp_support": (timestamp_support_path, np.dtype(np.uint16)),
        "status": (status_path, np.dtype(np.uint8)),
    }
    dense: dict[str, np.ndarray] = {}
    for name, (path, dtype) in sidecars.items():
        try:
            value = np.load(path, mmap_mode="r", allow_pickle=False)
        except (OSError, ValueError) as exc:
            raise UnifiedViewerError(f"cannot read dense lift {name} sidecar: {exc}") from exc
        if value.shape != (source_gaussian_count,) or value.dtype != dtype:
            raise UnifiedViewerError(f"dense lift {name} sidecar shape/dtype mismatch")
        dense[name] = value

    # Compare in bounded chunks so a single large object does not create four
    # additional all-object-sized advanced-indexing arrays at validation time.
    chunk_rows = 1_000_000
    for bank_row, object_id in enumerate(object_ids):
        start = int(indptr[bank_row])
        end = int(indptr[bank_row + 1])
        for offset in range(start, end, chunk_rows):
            stop = min(offset + chunk_rows, end)
            selected = indices[offset:stop]
            if not np.all(dense["object_id"][selected] == object_id):
                raise UnifiedViewerError("verified CSR labels differ from the hashed dense sidecar")
            if not np.array_equal(dense["confidence"][selected], confidence[offset:stop]):
                raise UnifiedViewerError("verified CSR confidence differs from the hashed dense sidecar")
            if not np.array_equal(dense["timestamp_support"][selected], support[offset:stop]):
                raise UnifiedViewerError("verified CSR support differs from the hashed dense sidecar")
            if not np.all(dense["status"][selected] == 2):
                raise UnifiedViewerError("verified CSR rows are not verified in the hashed status sidecar")
    return object_ids, indptr, indices


def _validate_dense_candidate(
    root: Path,
    spec: SceneSpec,
    farm: FarmBundle,
    source_table: PlyTable,
    *,
    verify_full_source: bool,
    marker_name: str = "_SUCCESS.json",
    release_eligible: bool = True,
) -> DenseLiftBundle:
    marker_path = root / marker_name
    marker = _load_json(marker_path)
    if (
        marker.get("schema_version") != LIFT_SUCCESS_SCHEMA
        or marker.get("status") != "success"
        or marker.get("release_eligible") is not release_eligible
        or marker.get("scene_id") != spec.scene_id
    ):
        raise UnifiedViewerError("dense lift success marker release/scene contract mismatch")
    result_path = _resolve_inside(root, str(marker.get("result") or ""), field_name="lift result")
    expected_result_sha = str(marker.get("result_sha256") or "")
    if not _SHA256_RE.fullmatch(expected_result_sha) or sha256_file(result_path) != expected_result_sha:
        raise UnifiedViewerError("dense lift result SHA-256 mismatch")
    artifact_bindings: list[ArtifactBinding] = [
        ArtifactBinding(result_path, int(result_path.stat().st_size), expected_result_sha)
    ]
    result = _load_json(result_path)
    if (
        result.get("schema_version") != LIFT_RESULT_SCHEMA
        or result.get("status") != "PASS"
        or result.get("release_eligible") is not release_eligible
        or result.get("scene_id") != spec.scene_id
    ):
        raise UnifiedViewerError("dense lift result schema/status/scene mismatch")
    inputs = result.get("inputs")
    if not isinstance(inputs, Mapping):
        raise UnifiedViewerError("dense lift inputs are absent")
    if inputs.get("farm_success_sha256") != farm.success_sha256:
        raise UnifiedViewerError("dense lift belongs to a different FARM success marker")
    acceptance_path = farm.run_dir / "qa" / "acceptance" / "result.json"
    if release_eligible and (
        not acceptance_path.is_file()
        or inputs.get("farm_acceptance_sha256") != sha256_file(acceptance_path)
    ):
        raise UnifiedViewerError("dense lift belongs to a different/missing FARM acceptance result")
    source_full_sha = str(inputs.get("source_ply_sha256") or "")
    if not _SHA256_RE.fullmatch(source_full_sha) or marker.get("source_ply_sha256") != source_full_sha:
        raise UnifiedViewerError("dense lift source full SHA-256 contract is invalid")
    if int(inputs.get("source_gaussian_count", -1)) != source_table.count:
        raise UnifiedViewerError("dense lift source Gaussian count mismatch")
    contracts = result.get("contracts")
    required_contracts = {
        "exact_pinned_gsplat_contributor_vjp",
        "dense_arrays_in_source_ply_order",
        "verified_only_canonical_labels",
        "labeled_ply_full_graphdeco_schema",
    }
    if release_eligible:
        required_contracts.add("clean_versioned_source_snapshot")
    if not isinstance(contracts, Mapping) or any(contracts.get(name) is not True for name in required_contracts):
        raise UnifiedViewerError("dense lift required contributor/order/schema contracts did not pass")
    if (
        contracts.get("source_ply_mutated") is not False
        or int(contracts.get("gaussians_deleted", -1)) != 0
        or int(contracts.get("unknown_instance_id", -2)) != UNKNOWN_INSTANCE_ID
    ):
        raise UnifiedViewerError("dense lift source-preservation/unknown-ID contract is invalid")
    final_artifacts = result.get("artifacts", {}).get("final") if isinstance(result.get("artifacts"), Mapping) else None
    if not isinstance(final_artifacts, Mapping):
        raise UnifiedViewerError("dense lift final artifacts are absent")
    bank_path, bank_sha, _ = _artifact_path(
        root, final_artifacts.get("verified_instance_bank"), field_name="verified_instance_bank"
    )
    if sha256_file(bank_path) != bank_sha:
        raise UnifiedViewerError("verified instance bank SHA-256 mismatch")
    artifact_bindings.append(ArtifactBinding(bank_path, int(bank_path.stat().st_size), bank_sha))
    expected_dtypes = {
        "object_id": np.dtype(np.int32),
        "confidence": np.dtype(np.float32),
        "timestamp_support": np.dtype(np.uint16),
        "status": np.dtype(np.uint8),
    }
    sidecar_paths: dict[str, Path] = {}
    for name, dtype in expected_dtypes.items():
        path, digest, _ = _artifact_path(root, final_artifacts.get(name), field_name=name)
        if sha256_file(path) != digest:
            raise UnifiedViewerError(f"dense lift {name} SHA-256 mismatch")
        artifact_bindings.append(ArtifactBinding(path, int(path.stat().st_size), digest))
        array = np.load(path, mmap_mode="r", allow_pickle=False)
        if array.shape != (source_table.count,) or array.dtype != dtype:
            raise UnifiedViewerError(f"dense lift {name} array shape/dtype mismatch")
        sidecar_paths[name] = path
    labeled_path, labeled_sha, labeled_bytes = _artifact_path(
        root, final_artifacts.get("instance_labeled_full"), field_name="instance_labeled_full"
    )
    labeled_manifest_path, labeled_manifest_sha, _ = _artifact_path(
        root, final_artifacts.get("labeled_ply_manifest"), field_name="labeled_ply_manifest"
    )
    if sha256_file(labeled_manifest_path) != labeled_manifest_sha:
        raise UnifiedViewerError("labeled PLY manifest SHA-256 mismatch")
    if sha256_file(labeled_path) != labeled_sha:
        raise UnifiedViewerError("instance_labeled_full PLY SHA-256 mismatch")
    artifact_bindings.extend((
        ArtifactBinding(labeled_path, labeled_bytes, labeled_sha),
        ArtifactBinding(
            labeled_manifest_path,
            int(labeled_manifest_path.stat().st_size),
            labeled_manifest_sha,
        ),
    ))
    labeled_manifest = _load_json(labeled_manifest_path)
    appended = (
        "farm_instance_id", "farm_instance_confidence", "farm_timestamp_support",
        "red", "green", "blue",
    )
    preservation_ok = (
        labeled_manifest.get("schema_version") == "farm.gaussian-lift.labeled-ply.v1"
        and int(labeled_manifest.get("source_bytes", -1)) == spec.source_ply.stat().st_size
        and labeled_manifest.get("source_full_sha256") == source_full_sha
        and int(labeled_manifest.get("output_bytes", -1)) == labeled_bytes
        and labeled_manifest.get("output_sha256") == labeled_sha
        and int(labeled_manifest.get("vertex_count", -1)) == source_table.count
        and tuple(labeled_manifest.get("original_properties", ())) == tuple(source_table.dtype.names or ())
        and tuple(labeled_manifest.get("appended_properties", ())) == appended
        and int(labeled_manifest.get("original_row_bytes", -1)) == source_table.dtype.itemsize
        and int(labeled_manifest.get("output_row_bytes", -1)) == source_table.dtype.itemsize + 13
        and labeled_manifest.get("original_fields_bitwise_preserved") is True
        and labeled_manifest.get("source_order_preserved") is True
        and int(labeled_manifest.get("rows_deleted", -1)) == 0
    )
    if not preservation_ok:
        raise UnifiedViewerError("labeled PLY full-schema/source-order preservation contract is invalid")
    labeled_table = open_binary_ply(labeled_path)
    if (
        labeled_table.count != source_table.count
        or tuple(labeled_table.dtype.names or ()) != tuple(source_table.dtype.names or ()) + appended
        or labeled_table.dtype.itemsize != source_table.dtype.itemsize + 13
    ):
        raise UnifiedViewerError("labeled PLY readback schema/count mismatch")
    counts = result.get("counts")
    if not isinstance(counts, Mapping):
        raise UnifiedViewerError("dense lift counts are absent")
    verified_gaussians = int(counts.get("verified_gaussians", -1))
    verified_objects = int(counts.get("verified_objects", -1))
    if (
        result.get("quality_status") != ("PASS" if release_eligible else "WARN")
        or verified_gaussians <= 0
        or verified_objects <= 0
        or verified_gaussians > source_table.count
    ):
        raise UnifiedViewerError("dense lift verified counts are invalid")
    if not release_eligible:
        quality_gate = result.get("quality_gate")
        if (
            not isinstance(quality_gate, Mapping)
            or quality_gate.get("calibrated") is not False
            or quality_gate.get("passed") is not False
            or "configuration_not_calibrated" not in quality_gate.get("reasons", [])
        ):
            raise UnifiedViewerError("review lift is not an explicit uncalibrated quality-gate result")
    _validate_verified_bank(
        bank_path,
        object_id_path=sidecar_paths["object_id"],
        confidence_path=sidecar_paths["confidence"],
        timestamp_support_path=sidecar_paths["timestamp_support"],
        status_path=sidecar_paths["status"],
        source_gaussian_count=source_table.count,
        expected_gaussians=verified_gaussians,
        expected_objects=verified_objects,
    )
    if verify_full_source and sha256_file(spec.source_ply) != source_full_sha:
        raise UnifiedViewerError("dense lift full source PLY SHA-256 mismatch")
    return DenseLiftBundle(
        root=root,
        result_path=result_path,
        result=result,
        result_sha256=expected_result_sha,
        bank_path=bank_path,
        bank_sha256=bank_sha,
        object_id_path=sidecar_paths["object_id"],
        confidence_path=sidecar_paths["confidence"],
        timestamp_support_path=sidecar_paths["timestamp_support"],
        status_path=sidecar_paths["status"],
        artifact_bindings=tuple(artifact_bindings),
        source_full_sha256=source_full_sha,
        verified_gaussians=verified_gaussians,
        verified_objects=verified_objects,
        release_eligible=release_eligible,
    )


def _find_dense_lift(
    spec: SceneSpec,
    farm: FarmBundle,
    source_table: PlyTable,
    *,
    verify_full_source: bool,
) -> tuple[DenseLiftBundle | None, str]:
    if not spec.dense_lift_roots:
        return None, "not configured"
    reasons: list[str] = []
    for candidate in _candidate_dirs(spec.dense_lift_roots):
        if not (candidate / "_SUCCESS.json").is_file():
            reasons.append(f"{candidate}: missing canonical _SUCCESS.json")
            continue
        try:
            bundle = _validate_dense_candidate(
                candidate, spec, farm, source_table, verify_full_source=verify_full_source
            )
            suffix = "full source SHA-256 verified" if verify_full_source else "full source SHA-256 deferred until load"
            return bundle, f"ready; {suffix}"
        except (UnifiedViewerError, OSError, KeyError, ValueError) as exc:
            reasons.append(f"{candidate}: {exc}")
    return None, "; ".join(reasons[-3:]) or "no compatible canonical dense lift"



def _find_review_lift(
    spec: SceneSpec,
    farm: FarmBundle,
    source_table: PlyTable,
    *,
    verify_full_source: bool,
) -> tuple[DenseLiftBundle | None, str]:
    """Find an explicitly nonrelease exact-contributor calibration artifact."""

    if not spec.dense_lift_roots:
        return None, "not configured"
    reasons: list[str] = []
    for candidate in _candidate_dirs(spec.dense_lift_roots):
        marker_name = next(
            (
                name for name in ("_NONRELEASE_SUCCESS.json", "_LEGACY_SUCCESS.json")
                if (candidate / name).is_file()
            ),
            None,
        )
        if marker_name is None:
            continue
        try:
            bundle = _validate_dense_candidate(
                candidate,
                spec,
                farm,
                source_table,
                verify_full_source=verify_full_source,
                marker_name=marker_name,
                release_eligible=False,
            )
            return bundle, (
                f"REVIEW ONLY: exact contributor calibration; {bundle.verified_objects} verified objects, "
                f"{bundle.verified_gaussians:,} source Gaussians; uncalibrated/nonrelease"
            )
        except (UnifiedViewerError, OSError, KeyError, TypeError, ValueError) as exc:
            reasons.append(f"{candidate}: {exc}")
    return None, "; ".join(reasons[-3:]) or "no compatible nonrelease exact calibration"


def _validate_legacy_shaper_candidate(
    bridge: BridgeBundle,
) -> LegacyShaperBundle:
    """Validate frozen v1 ShapeR viewer assets as mutable-history REVIEW evidence."""

    root = bridge.root
    result = _load_json(root / "result.json")
    shaper_result = result.get("shaper_meshes")
    if not isinstance(shaper_result, Mapping) or shaper_result.get("status") != "REVIEW":
        raise UnifiedViewerError("legacy ShapeR result is not explicitly REVIEW")
    scene_path = root / "shaper_scene" / "scene_manifest.json"
    assets_path = root / "viewer" / "assets" / "viewer_assets.json"
    manifest = _load_json(scene_path)
    assets = _load_json(assets_path)
    if (
        manifest.get("schema_version") != LEGACY_SHAPER_SCENE_SCHEMA
        or manifest.get("status") != "REVIEW"
        or manifest.get("coordinate_units") != "metres"
        or manifest.get("coordinate_frame") != "FARM metric world"
    ):
        raise UnifiedViewerError("legacy ShapeR scene schema/status/metric frame mismatch")
    if assets.get("schema_version") != LEGACY_SHAPER_VIEWER_SCHEMA or assets.get("status") != "PASS":
        raise UnifiedViewerError("legacy ShapeR viewer-assets schema/status mismatch")
    rows = manifest.get("objects", {}).get("rows") if isinstance(manifest.get("objects"), Mapping) else None
    viewer_rows = assets.get("objects")
    if not isinstance(rows, list) or not isinstance(viewer_rows, list):
        raise UnifiedViewerError("legacy ShapeR object inventories are absent")
    included: dict[int, Mapping[str, Any]] = {}
    for row in rows:
        if not isinstance(row, Mapping) or row.get("included") is not True:
            continue
        object_id = int(row.get("object_id", -1))
        qa = row.get("geometry_qa")
        if object_id < 0 or object_id in included or row.get("status") != "PASS":
            raise UnifiedViewerError("legacy ShapeR included object IDs/status are invalid")
        if not isinstance(qa, Mapping) or qa.get("status") != "PASS":
            raise UnifiedViewerError("legacy ShapeR included object lacks PASS metric QA")
        bounds = np.asarray(row.get("bounds_m"), dtype=np.float64)
        if bounds.shape != (2, 3) or not np.isfinite(bounds).all() or np.any(bounds[1] < bounds[0]):
            raise UnifiedViewerError("legacy ShapeR object bounds are invalid")
        included[object_id] = row
    expected_count = int(shaper_result.get("included", -1))
    if not included or len(included) != expected_count:
        raise UnifiedViewerError("legacy ShapeR result/scene included counts differ")
    viewer_by_id = {
        int(row["object_id"]): row
        for row in viewer_rows
        if isinstance(row, Mapping) and "object_id" in row
    }
    if set(viewer_by_id) != set(included):
        raise UnifiedViewerError("legacy ShapeR scene/viewer object IDs differ")
    asset_root = assets_path.parent.resolve()
    bindings: dict[int, ArtifactBinding] = {}
    for object_id, row in viewer_by_id.items():
        asset_path = _resolve_inside(
            asset_root,
            str(row.get("path") or ""),
            field_name=f"legacy ShapeR asset #{object_id}",
            guard_root=asset_root,
        )
        with np.load(asset_path, allow_pickle=False) as archive:
            if set(archive.files) != {"vertices", "faces", "color"}:
                raise UnifiedViewerError("legacy ShapeR NPZ fields differ from v1")
            vertices = np.asarray(archive["vertices"])
            faces = np.asarray(archive["faces"])
            color = np.asarray(archive["color"])
        if (
            vertices.dtype != np.dtype(np.float32)
            or faces.dtype != np.dtype(np.uint32)
            or color.dtype != np.dtype(np.uint8)
            or vertices.ndim != 2 or vertices.shape[1] != 3
            or faces.ndim != 2 or faces.shape[1] != 3
            or color.shape != (3,)
            or not len(vertices) or not len(faces)
            or not np.isfinite(vertices).all()
            or np.any(faces >= len(vertices))
        ):
            raise UnifiedViewerError(f"legacy ShapeR asset #{object_id} geometry is invalid")
        bindings[object_id] = ArtifactBinding(
            path=asset_path,
            size_bytes=int(asset_path.stat().st_size),
            sha256=sha256_file(asset_path),
        )
    return LegacyShaperBundle(
        root=root,
        scene_manifest=ArtifactBinding(scene_path, int(scene_path.stat().st_size), sha256_file(scene_path)),
        viewer_assets=ArtifactBinding(assets_path, int(assets_path.stat().st_size), sha256_file(assets_path)),
        asset_bindings=bindings,
        manifest=manifest,
        object_rows=included,
        provenance_warning=(
            "REVIEW ONLY: frozen v1 ShapeR meshes match the legacy bridge/FARM/source fingerprints, "
            "but have no producer-recorded per-mesh hashes and are not canonical ShapeR v2"
        ),
    )


def _find_legacy_shaper(
    bridge: BridgeBundle | None,
) -> tuple[LegacyShaperBundle | None, str]:
    if bridge is None:
        return None, "requires a compatible legacy bridge output"
    try:
        bundle = _validate_legacy_shaper_candidate(bridge)
        return bundle, f"REVIEW ONLY: {len(bundle.object_rows)} frozen v1 meshes; not canonical ShapeR v2"
    except (UnifiedViewerError, OSError, KeyError, TypeError, ValueError) as exc:
        return None, str(exc)

def _validate_shaper_candidate(
    root: Path,
    spec: SceneSpec,
    farm: FarmBundle,
    dense: DenseLiftBundle,
    *,
    marker_name: str = "_SUCCESS.json",
    release_eligible: bool = True,
) -> ShaperBundle:
    success_path = root / "_SUCCESS.json"
    nonrelease_path = root / "_NONRELEASE_SUCCESS.json"
    if nonrelease_path.exists() and success_path.exists():
        raise UnifiedViewerError("ShapeR assembly has conflicting release/nonrelease markers")
    marker_path = root / marker_name
    if marker_name not in {"_SUCCESS.json", "_NONRELEASE_SUCCESS.json"}:
        raise UnifiedViewerError("unsupported ShapeR completion marker")
    if not marker_path.is_file():
        raise UnifiedViewerError(f"ShapeR assembly has no {marker_name}")
    marker = _load_json(marker_path)
    if (
        marker.get("schema_version") != SHAPER_SUCCESS_SCHEMA
        or marker.get("status") != "success"
        or marker.get("release_eligible") is not release_eligible
        or marker.get("scene_id") != spec.scene_id
    ):
        raise UnifiedViewerError("ShapeR completion marker schema/status/release/scene mismatch")
    scene_path = _resolve_inside(root, str(marker.get("result") or ""), field_name="ShapeR scene manifest")
    assets_path = _resolve_inside(root, str(marker.get("viewer_assets") or ""), field_name="ShapeR viewer assets")
    if scene_path != (root / "scene_manifest.json").resolve() or assets_path != (root / "viewer_assets.json").resolve():
        raise UnifiedViewerError("ShapeR marker paths differ from the v2 canonical basenames")
    scene_sha = str(marker.get("result_sha256") or "")
    assets_sha = str(marker.get("viewer_assets_sha256") or "")
    if (
        not _SHA256_RE.fullmatch(scene_sha)
        or not _SHA256_RE.fullmatch(assets_sha)
        or sha256_file(scene_path) != scene_sha
        or sha256_file(assets_path) != assets_sha
    ):
        raise UnifiedViewerError("ShapeR scene/viewer manifest SHA-256 mismatch")
    manifest = _load_json(scene_path)
    assets = _load_json(assets_path)
    if (
        manifest.get("schema_version") != SHAPER_SCENE_SCHEMA
        or manifest.get("status") != "PASS"
        or manifest.get("release_eligible") is not release_eligible
        or manifest.get("scene_id") != spec.scene_id
        or manifest.get("coordinate_units") != "metres"
        or manifest.get("coordinate_frame") != "FARM metric world"
    ):
        raise UnifiedViewerError("ShapeR scene manifest schema/status/metric-frame mismatch")
    if (
        assets.get("schema_version") != SHAPER_VIEWER_SCHEMA
        or assets.get("status") != "PASS"
        or assets.get("scene_id") != spec.scene_id
        or assets.get("source_scene_manifest_sha256") != scene_sha
    ):
        raise UnifiedViewerError("ShapeR viewer-assets schema/status/source-manifest mismatch")

    acceptance_sha: str | None = None
    if release_eligible:
        acceptance_path = farm.run_dir / "qa" / "acceptance" / "result.json"
        if not acceptance_path.is_file():
            raise UnifiedViewerError("ShapeR canonical chain requires FARM final acceptance")
        acceptance_sha = sha256_file(acceptance_path)
    inputs = manifest.get("inputs")
    if not isinstance(inputs, Mapping):
        raise UnifiedViewerError("ShapeR scene inputs are absent")
    farm_input_path = Path(str(inputs.get("farm_run") or "")).expanduser().resolve(strict=False)
    farm_path_matches = farm_input_path == farm.run_dir.resolve()
    if not release_eligible and not farm_path_matches:
        farm_path_matches = (
            farm_input_path.name == farm.run_dir.name
            and inputs.get("farm_run_id") == farm.run_id
            and marker.get("farm_run_id") == farm.run_id
            and assets.get("farm_run_id") == farm.run_id
        )
    expected_chain = {
        "farm_success_sha256": farm.success_sha256,
        "farm_acceptance_sha256": acceptance_sha,
        "resolved_context_sha256": farm.resolved_context_sha256,
        "gaussian_lift_result_sha256": dense.result_sha256,
        "verified_instance_bank_sha256": dense.bank_sha256,
        "source_ply_sha256": dense.source_full_sha256,
    }
    if not farm_path_matches or any(
        inputs.get(name) != value for name, value in expected_chain.items()
    ):
        raise UnifiedViewerError("ShapeR scene belongs to a different FARM/lift/source chain")
    for name in ("shaper_batch_result_sha256", "shaper_inputs_result_sha256"):
        if not _SHA256_RE.fullmatch(str(inputs.get(name) or "")):
            raise UnifiedViewerError(f"ShapeR scene {name} is invalid")
    source_authority = manifest.get("source_authority")
    if not isinstance(source_authority, Mapping):
        raise UnifiedViewerError("ShapeR source authority is absent")
    if release_eligible:
        farm_manifest = _load_json(farm.run_dir / "manifest.json")
        farm_snapshot = farm_manifest.get("source_snapshot")
        if (
            source_authority.get("signed") is not True
            or not isinstance(farm_snapshot, Mapping)
            or source_authority.get("tree_sha256") != farm_snapshot.get("tree_sha256")
            or not _SHA256_RE.fullmatch(str(source_authority.get("tree_sha256") or ""))
        ):
            raise UnifiedViewerError("ShapeR release is not tied to the FARM signed source snapshot")
    elif (
        source_authority.get("signed") is not False
        or source_authority.get("tree_sha256") is not None
        or manifest.get("quality_status") != "WARN"
    ):
        raise UnifiedViewerError("ShapeR review output lacks explicit unsigned/WARN provenance")
    contracts = manifest.get("contracts")
    if not isinstance(contracts, Mapping) or any(
        contracts.get(name) is not expected
        for name, expected in {
            "qa_pass_meshes_only": True,
            "metric_scale_preserved": True,
            "lift_labels_mutated": False,
            "source_ply_mutated": False,
        }.items()
    ):
        raise UnifiedViewerError("ShapeR scene preservation/QA contracts failed")

    direct_chain = {
        "farm_success_sha256": expected_chain["farm_success_sha256"],
        "farm_acceptance_sha256": expected_chain["farm_acceptance_sha256"],
        "gaussian_lift_result_sha256": expected_chain["gaussian_lift_result_sha256"],
        "verified_instance_bank_sha256": expected_chain["verified_instance_bank_sha256"],
        "source_ply_sha256": expected_chain["source_ply_sha256"],
        "source_shaper_batch_result_sha256": inputs["shaper_batch_result_sha256"],
        "shaper_inputs_result_sha256": inputs["shaper_inputs_result_sha256"],
    }
    if any(assets.get(name) != value for name, value in direct_chain.items()):
        raise UnifiedViewerError("ShapeR viewer assets differ from the canonical scene input chain")
    recorded_batch = Path(str(inputs.get("shaper_batch") or "")).expanduser()
    viewer_recorded_batch = Path(str(assets.get("source_shaper_batch") or "")).expanduser()
    if recorded_batch != viewer_recorded_batch or not recorded_batch.name:
        raise UnifiedViewerError("ShapeR scene/viewer batch paths are internally inconsistent")
    batch_result_sha = str(inputs.get("shaper_batch_result_sha256") or "")
    if not _SHA256_RE.fullmatch(batch_result_sha):
        raise UnifiedViewerError("ShapeR batch result digest is invalid")
    batch_candidates = (
        recorded_batch.resolve(strict=False),
        (root.parent / recorded_batch.name).resolve(strict=False),
        (root.parent.parent / recorded_batch.name).resolve(strict=False),
    )
    batch_root: Path | None = None
    for candidate in batch_candidates:
        result_candidate = candidate / "result.json"
        if candidate.is_dir() and result_candidate.is_file() and sha256_file(result_candidate) == batch_result_sha:
            batch_root = candidate
            break
    if batch_root is None:
        raise UnifiedViewerError("ShapeR batch result SHA-256 mismatch")
    batch_result_path = batch_root / "result.json"
    batch_result_binding = ArtifactBinding(
        batch_result_path,
        int(batch_result_path.stat().st_size),
        batch_result_sha,
    )

    layers = assets.get("layers")
    if not isinstance(layers, Mapping):
        raise UnifiedViewerError("ShapeR viewer layer descriptors are absent")
    combined_gltf = _binding_from_descriptor(
        root,
        layers.get("combined_gltf"),
        field_name="ShapeR combined GLB",
        guard_root=root,
        expected_path=root / "combined_metric_objects.glb",
    )
    mesh_npz = _binding_from_descriptor(
        root,
        layers.get("combined_farm_metric_npz"),
        field_name="ShapeR combined FARM-metric NPZ",
        guard_root=root,
        expected_path=root / "combined_metric_objects.npz",
    )
    artifact_bindings: list[ArtifactBinding] = [
        ArtifactBinding(scene_path, int(scene_path.stat().st_size), scene_sha),
        ArtifactBinding(assets_path, int(assets_path.stat().st_size), assets_sha),
        combined_gltf,
        mesh_npz,
        batch_result_binding,
    ]
    objects = manifest.get("objects")
    rows = objects.get("rows") if isinstance(objects, Mapping) else None
    if not isinstance(rows, list) or not rows:
        raise UnifiedViewerError("ShapeR scene has no QA-passing object rows")
    object_rows: dict[int, Mapping[str, Any]] = {}
    for row in rows:
        if (
            not isinstance(row, Mapping)
            or row.get("included") is not True
            or not isinstance(row.get("geometry_qa"), Mapping)
            or row["geometry_qa"].get("status") != "PASS"
        ):
            raise UnifiedViewerError("ShapeR scene contains a non-QA-pass object row")
        object_id = int(row["object_id"])
        if object_id < 0 or object_id in object_rows:
            raise UnifiedViewerError("ShapeR scene object IDs are invalid/duplicated")
        source_glb = _binding_from_descriptor(
            batch_root,
            row.get("source_glb"),
            field_name=f"ShapeR source GLB #{object_id}",
            guard_root=batch_root,
        )
        artifact_bindings.append(source_glb)
        object_rows[object_id] = row
    viewer_rows = assets.get("objects")
    if not isinstance(viewer_rows, list) or len(viewer_rows) != len(object_rows):
        raise UnifiedViewerError("ShapeR viewer object inventory count mismatch")
    viewer_by_id = {
        int(row["object_id"]): row for row in viewer_rows if isinstance(row, Mapping) and "object_id" in row
    }
    if set(viewer_by_id) != set(object_rows):
        raise UnifiedViewerError("ShapeR viewer/scene object IDs differ")
    for object_id, row in object_rows.items():
        viewer_row = viewer_by_id[object_id]
        if viewer_row.get("geometry_qa_status") != "PASS" or viewer_row.get("source_glb") != row.get("source_glb"):
            raise UnifiedViewerError(f"ShapeR viewer object #{object_id} provenance mismatch")

    with np.load(mesh_npz.path, allow_pickle=False) as archive:
        if set(archive.files) != {"vertices", "faces", "object_id", "colors"}:
            raise UnifiedViewerError("ShapeR combined NPZ fields differ from v1")
        vertices = np.asarray(archive["vertices"])
        faces = np.asarray(archive["faces"])
        object_ids = np.asarray(archive["object_id"])
        colours = np.asarray(archive["colors"])
    if (
        vertices.dtype != np.dtype(np.float32)
        or faces.dtype != np.dtype(np.int32)
        or object_ids.dtype != np.dtype(np.int32)
        or colours.dtype != np.dtype(np.uint8)
        or vertices.ndim != 2 or vertices.shape[1] != 3
        or faces.ndim != 2 or faces.shape[1] != 3
        or object_ids.shape != (len(vertices),)
        or colours.shape != vertices.shape
        or not len(vertices) or not len(faces)
        or not np.isfinite(vertices).all()
        or np.any(faces < 0) or int(faces.max()) >= len(vertices)
    ):
        raise UnifiedViewerError("ShapeR combined NPZ geometry/dtypes are invalid")
    face_ids = object_ids[faces]
    if np.any(face_ids != face_ids[:, :1]) or set(int(value) for value in np.unique(object_ids)) != set(object_rows):
        raise UnifiedViewerError("ShapeR combined NPZ object membership is inconsistent")
    for object_id in object_rows:
        object_colours = colours[object_ids == object_id]
        if not len(object_colours) or np.any(object_colours != object_colours[0]):
            raise UnifiedViewerError(
                f"ShapeR object #{object_id} must have one uniform RGB color for Viser"
            )
    combined = manifest.get("combined")
    if (
        not isinstance(combined, Mapping)
        or int(combined.get("vertices", -1)) != len(vertices)
        or int(combined.get("faces", -1)) != len(faces)
        or combined.get("npz") != layers.get("combined_farm_metric_npz")
        or combined.get("glb") != layers.get("combined_gltf")
    ):
        raise UnifiedViewerError("ShapeR scene combined inventory mismatch")
    return ShaperBundle(
        root=root,
        scene_manifest_path=scene_path,
        scene_manifest_sha256=scene_sha,
        viewer_assets_path=assets_path,
        viewer_assets_sha256=assets_sha,
        mesh_npz=mesh_npz,
        combined_gltf=combined_gltf,
        artifact_bindings=tuple(artifact_bindings),
        manifest=manifest,
        viewer_assets=assets,
        object_rows=object_rows,
    )


def _find_shaper(
    spec: SceneSpec,
    farm: FarmBundle,
    dense: DenseLiftBundle | None,
) -> tuple[ShaperBundle | None, str]:
    if dense is None:
        return None, "canonical ShapeR v2 requires a compatible canonical Gaussian lift"
    if not spec.shaper_roots:
        return None, "not configured"
    reasons: list[str] = []
    seen: set[Path] = set()
    for parent in _candidate_dirs(spec.shaper_roots):
        for candidate in (parent, parent / "shaper_scene", parent / "assembly"):
            candidate = candidate.resolve(strict=False)
            if candidate in seen:
                continue
            seen.add(candidate)
            if not (candidate / "_SUCCESS.json").is_file():
                if (candidate / "_NONRELEASE_SUCCESS.json").is_file():
                    reasons.append(f"{candidate}: explicitly nonrelease")
                continue
            try:
                return _validate_shaper_candidate(candidate, spec, farm, dense), "ready; canonical ShapeR v2 exact chain"
            except (UnifiedViewerError, OSError, KeyError, TypeError, ValueError) as exc:
                reasons.append(f"{candidate}: {exc}")
    return None, "; ".join(reasons[-3:]) or "no compatible canonical ShapeR v2 assembly"



def _find_review_shaper(
    spec: SceneSpec,
    farm: FarmBundle,
    review_lift: DenseLiftBundle | None,
) -> tuple[ShaperBundle | None, str]:
    if review_lift is None:
        return None, "review ShapeR v2 requires a compatible exact-calibration lift"
    if not spec.shaper_roots:
        return None, "not configured"
    reasons: list[str] = []
    seen: set[Path] = set()
    for parent in _candidate_dirs(spec.shaper_roots):
        for candidate in (parent, parent / "scene-v2", parent / "scene-v1", parent / "shaper_scene", parent / "assembly"):
            candidate = candidate.resolve(strict=False)
            if candidate in seen:
                continue
            seen.add(candidate)
            if not (candidate / "_NONRELEASE_SUCCESS.json").is_file():
                continue
            try:
                bundle = _validate_shaper_candidate(
                    candidate,
                    spec,
                    farm,
                    review_lift,
                    marker_name="_NONRELEASE_SUCCESS.json",
                    release_eligible=False,
                )
                return bundle, (
                    f"REVIEW ONLY: exact-calibration ShapeR v2; {len(bundle.object_rows)} QA-PASS meshes; "
                    "uncalibrated/nonrelease"
                )
            except (UnifiedViewerError, OSError, KeyError, TypeError, ValueError) as exc:
                reasons.append(f"{candidate}: {exc}")
    return None, "; ".join(reasons[-3:]) or "no compatible nonrelease ShapeR v2 assembly"

def validate_scene(spec: SceneSpec, *, verify_full_source: bool = False) -> ValidatedScene:
    farm, source_table = _validate_farm(spec)
    bridge, bridge_reason = _find_bridge(spec, farm)
    dense, dense_reason = _find_dense_lift(
        spec, farm, source_table, verify_full_source=verify_full_source
    )
    review_lift, review_lift_reason = _find_review_lift(
        spec, farm, source_table, verify_full_source=verify_full_source
    )
    review_shaper, review_shaper_reason = _find_review_shaper(spec, farm, review_lift)
    legacy_shaper, legacy_shaper_reason = _find_legacy_shaper(bridge)
    shaper, shaper_reason = _find_shaper(spec, farm, dense)
    packed_mb = source_table.count * VISER_PACKED_BYTES_PER_GAUSSIAN / 1_000_000.0
    layers = {
        "farm_preview": LayerAvailability(
            True,
            f"{len(farm.catalog)} objects; {farm.integrity_note}",
            warning=farm.integrity_warning,
        ),
        "farm_obbs": LayerAvailability(
            True,
            f"{len(farm.catalog)} final presentation OBBs; catalog bound by producer inventory",
            warning=farm.integrity_warning,
        ),
        "source_gaussians": LayerAvailability(
            True,
            f"all {source_table.count:,} source rows; Viser float16/uint8 quantized DC preview; "
            f"~{packed_mb:.0f} MB/client",
            warning=True,
        ),
        "object_evidence": LayerAvailability(
            farm.evidence is not None,
            farm.evidence.provenance_note if farm.evidence is not None else farm.evidence_unavailable_reason,
            warning=farm.evidence.provenance_warning if farm.evidence is not None else False,
        ),
        "bridge_instances": LayerAvailability(
            bridge is not None,
            bridge_reason,
            warning=bridge is not None,
        ),
        "legacy_shaper_review": LayerAvailability(
            review_shaper is not None or legacy_shaper is not None,
            review_shaper_reason if review_shaper is not None else legacy_shaper_reason,
            warning=review_shaper is not None or legacy_shaper is not None,
        ),
        "shaper_meshes": LayerAvailability(
            shaper is not None,
            shaper_reason,
            warning=shaper is not None,
        ),
        "exact_lift_review": LayerAvailability(
            review_lift is not None,
            review_lift_reason,
            warning=review_lift is not None,
        ),
        "dense_lift": LayerAvailability(dense is not None, dense_reason),
        "dense_lift_points": LayerAvailability(
            dense is not None,
            f"explicit sampled-centre fallback; {dense_reason}",
            warning=True,
        ),
    }
    return ValidatedScene(
        spec=spec,
        farm=farm,
        source_table=source_table,
        bridge=bridge,
        dense_lift=dense,
        review_lift=review_lift,
        shaper=shaper,
        review_shaper=review_shaper,
        legacy_shaper=legacy_shaper,
        layers=layers,
    )


def validate_registry(path: Path, *, verify_full_source: bool = False) -> RegistryValidation:
    registry = load_registry(path)
    scenes: list[ValidatedScene | SceneValidationFailure] = []
    for spec in registry.scenes:
        try:
            scenes.append(validate_scene(spec, verify_full_source=verify_full_source))
        except (UnifiedViewerError, OSError, KeyError, ValueError) as exc:
            scenes.append(SceneValidationFailure(spec.scene_id, spec.label, str(exc)))
    return RegistryValidation(registry, tuple(scenes))


def _stable_sample(indices: np.ndarray, limit: int, *, seed: int) -> np.ndarray:
    indices = np.asarray(indices, dtype=np.int64).reshape(-1)
    if limit < 1:
        raise UnifiedViewerError("point limits must be positive")
    if len(indices) <= limit:
        return indices
    rng = np.random.default_rng(seed)
    return np.sort(rng.choice(indices, size=limit, replace=False))


def load_farm_preview(scene: ValidatedScene, *, max_points: int) -> tuple[np.ndarray, np.ndarray]:
    scene.farm.preview_binding.verify(field_name="FARM preview cloud")
    with np.load(scene.farm.preview_path, allow_pickle=False) as archive:
        xyz = np.asarray(archive["xyz"], dtype=np.float32)
        rgb = np.asarray(archive["rgb"], dtype=np.uint8)
    use = _stable_sample(np.arange(len(xyz), dtype=np.int64), max_points, seed=20260819)
    points = np.ascontiguousarray(xyz[use])
    colors = np.ascontiguousarray(rgb[use])
    if not np.isfinite(points).all():
        raise UnifiedViewerError("FARM preview contains non-finite points")
    return points, colors


def load_bridge_instances(scene: ValidatedScene, *, max_points: int) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    if scene.bridge is None:
        raise UnifiedViewerError(scene.layers["bridge_instances"].reason)
    _validate_bridge_candidate(scene.bridge.root, scene.spec, scene.farm)
    with np.load(scene.bridge.instance_cloud, allow_pickle=False) as archive:
        xyz = np.asarray(archive["xyz"], dtype=np.float32)
        rgb = np.asarray(archive["rgb"], dtype=np.uint8)
        labels = np.asarray(archive["object_id"], dtype=np.int32)
    eligible = np.flatnonzero(labels >= 0)
    use = _stable_sample(eligible, max_points, seed=20260820)
    points = np.ascontiguousarray(xyz[use])
    if not np.isfinite(points).all():
        raise UnifiedViewerError("bridge instance overlay contains non-finite points")
    return points, np.ascontiguousarray(rgb[use]), np.ascontiguousarray(labels[use])


def _instance_color(object_id: int) -> tuple[int, int, int]:
    # Match the canonical instance_labeled_full.ply writer palette.
    hue = (0.11 + (int(object_id) * 0.6180339887498949)) % 1.0
    red, green, blue = colorsys.hsv_to_rgb(hue, 0.72, 0.96)
    return tuple(int(round(255.0 * value)) for value in (red, green, blue))


def load_gaussian_splat_arrays(
    table: PlyTable,
    *,
    meters_per_scene_unit: float,
    source_indices: np.ndarray | None = None,
    rgb_override: np.ndarray | None = None,
    chunk_rows: int = 262_144,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Decode Graphdeco PLY rows into Viser's anisotropic splat contract.

    The result contains every requested row.  Chunking limits conversion
    scratch memory; it never changes membership or geometry.
    """

    metric = float(meters_per_scene_unit)
    if not math.isfinite(metric) or metric <= 0 or int(chunk_rows) < 1:
        raise UnifiedViewerError("invalid Gaussian metric scale or conversion chunk size")
    if source_indices is None:
        count = table.count
        indices: np.ndarray | None = None
    else:
        indices = np.asarray(source_indices)
        if indices.ndim != 1 or indices.dtype.kind not in "iu":
            raise UnifiedViewerError("Gaussian source indices must be one integer vector")
        indices = np.ascontiguousarray(indices.astype(np.int64, copy=False))
        if np.any(indices < 0) or np.any(indices >= table.count):
            raise UnifiedViewerError("Gaussian source indices are outside the source PLY")
        count = len(indices)
    if count <= 0:
        raise UnifiedViewerError("Gaussian layer contains no source rows")
    override: np.ndarray | None = None
    if rgb_override is not None:
        override = np.asarray(rgb_override)
        if override.shape != (count, 3) or override.dtype not in (np.dtype(np.uint8), np.dtype(np.float32)):
            raise UnifiedViewerError("Gaussian RGB override must be uint8 or float32 N x 3")

    centers = np.empty((count, 3), dtype=np.float32)
    covariances = np.empty((count, 3, 3), dtype=np.float32)
    rgbs = np.empty((count, 3), dtype=np.float32)
    opacities = np.empty((count, 1), dtype=np.float32)
    source = table.memmap()
    for start in range(0, count, int(chunk_rows)):
        stop = min(count, start + int(chunk_rows))
        rows = source[start:stop] if indices is None else source[indices[start:stop]]
        centers[start:stop, 0] = np.asarray(rows["x"], dtype=np.float32) * metric
        centers[start:stop, 1] = np.asarray(rows["y"], dtype=np.float32) * metric
        centers[start:stop, 2] = np.asarray(rows["z"], dtype=np.float32) * metric

        log_scales = np.column_stack((rows["scale_0"], rows["scale_1"], rows["scale_2"]))
        scales = np.exp(np.asarray(log_scales, dtype=np.float64)) * metric
        variances = np.square(scales)
        quaternions = np.column_stack((rows["rot_0"], rows["rot_1"], rows["rot_2"], rows["rot_3"]))
        quaternions = np.asarray(quaternions, dtype=np.float64)
        norms = np.linalg.norm(quaternions, axis=1)
        if np.any(~np.isfinite(norms)) or np.any(norms <= 1e-12):
            raise UnifiedViewerError("source PLY contains an invalid Gaussian quaternion")
        quaternions /= norms[:, None]
        w, x, y, z = quaternions.T
        rotation = np.empty((stop - start, 3, 3), dtype=np.float64)
        rotation[:, 0, 0] = 1.0 - 2.0 * (y * y + z * z)
        rotation[:, 0, 1] = 2.0 * (x * y - w * z)
        rotation[:, 0, 2] = 2.0 * (x * z + w * y)
        rotation[:, 1, 0] = 2.0 * (x * y + w * z)
        rotation[:, 1, 1] = 1.0 - 2.0 * (x * x + z * z)
        rotation[:, 1, 2] = 2.0 * (y * z - w * x)
        rotation[:, 2, 0] = 2.0 * (x * z - w * y)
        rotation[:, 2, 1] = 2.0 * (y * z + w * x)
        rotation[:, 2, 2] = 1.0 - 2.0 * (x * x + y * y)
        covariances[start:stop] = np.einsum(
            "nik,nk,njk->nij", rotation, variances, rotation, optimize=False
        ).astype(np.float32)

        logits = np.asarray(rows["opacity"], dtype=np.float64)
        exp_negative_abs = np.exp(-np.abs(logits))
        alpha = np.where(
            logits >= 0.0,
            1.0 / (1.0 + exp_negative_abs),
            exp_negative_abs / (1.0 + exp_negative_abs),
        )
        opacities[start:stop, 0] = alpha.astype(np.float32)
        if override is None:
            dc = np.column_stack((rows["f_dc_0"], rows["f_dc_1"], rows["f_dc_2"]))
            rgbs[start:stop] = np.clip(
                0.5 + SH_DC_C0 * np.asarray(dc, dtype=np.float64), 0.0, 1.0
            ).astype(np.float32)
        elif override.dtype == np.uint8:
            rgbs[start:stop] = override[start:stop].astype(np.float32) / 255.0
        else:
            rgbs[start:stop] = np.clip(override[start:stop], 0.0, 1.0)

    if not all(np.isfinite(value).all() for value in (centers, covariances, rgbs, opacities)):
        raise UnifiedViewerError("source PLY decodes to non-finite Gaussian splat values")
    return centers, covariances, rgbs, opacities


def _verify_dense_bundle_for_load(scene: ValidatedScene) -> DenseLiftBundle:
    bundle = scene.dense_lift
    if bundle is None:
        raise UnifiedViewerError(scene.layers["dense_lift"].reason)
    if sha256_file(scene.spec.source_ply) != bundle.source_full_sha256:
        raise UnifiedViewerError("dense lift no longer matches the full source PLY SHA-256")
    for index, binding in enumerate(bundle.artifact_bindings):
        binding.verify(field_name=f"dense lift artifact[{index}]")
    return bundle


def _load_verified_bank(scene: ValidatedScene) -> tuple[np.ndarray, np.ndarray]:
    bundle = _verify_dense_bundle_for_load(scene)
    object_ids, indptr, indices = _validate_verified_bank(
        bundle.bank_path,
        object_id_path=bundle.object_id_path,
        confidence_path=bundle.confidence_path,
        timestamp_support_path=bundle.timestamp_support_path,
        status_path=bundle.status_path,
        source_gaussian_count=scene.source_table.count,
        expected_gaussians=bundle.verified_gaussians,
        expected_objects=bundle.verified_objects,
    )
    labels = np.repeat(object_ids, np.diff(indptr))
    return np.ascontiguousarray(indices), np.ascontiguousarray(labels.astype(np.int32))


def load_source_gaussian_splats(
    scene: ValidatedScene,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    scene.spec.source_fingerprint.validate(scene.spec.source_ply)
    if scene.dense_lift is not None and sha256_file(scene.spec.source_ply) != scene.dense_lift.source_full_sha256:
        raise UnifiedViewerError("source PLY differs from the canonical dense-lift full SHA-256")
    return load_gaussian_splat_arrays(
        scene.source_table,
        meters_per_scene_unit=scene.farm.meters_per_scene_unit,
    )


def load_dense_gaussian_splats(
    scene: ValidatedScene,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    indices, labels = _load_verified_bank(scene)
    unique_labels, inverse = np.unique(labels, return_inverse=True)
    palette = np.asarray([_instance_color(int(value)) for value in unique_labels], dtype=np.uint8)
    colours = np.ascontiguousarray(palette[inverse])
    centers, covariances, rgbs, opacities = load_gaussian_splat_arrays(
        scene.source_table,
        meters_per_scene_unit=scene.farm.meters_per_scene_unit,
        source_indices=indices,
        rgb_override=colours,
    )
    return centers, covariances, rgbs, opacities, labels



def load_review_gaussian_splats(
    scene: ValidatedScene,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    bundle = scene.review_lift
    if bundle is None:
        raise UnifiedViewerError(scene.layers["exact_lift_review"].reason)
    if bundle.release_eligible:
        raise UnifiedViewerError("review layer cannot load a release-eligible bundle")
    if sha256_file(scene.spec.source_ply) != bundle.source_full_sha256:
        raise UnifiedViewerError("review lift no longer matches the full source PLY SHA-256")
    for index, binding in enumerate(bundle.artifact_bindings):
        binding.verify(field_name=f"review lift artifact[{index}]")
    object_ids, indptr, indices = _validate_verified_bank(
        bundle.bank_path,
        object_id_path=bundle.object_id_path,
        confidence_path=bundle.confidence_path,
        timestamp_support_path=bundle.timestamp_support_path,
        status_path=bundle.status_path,
        source_gaussian_count=scene.source_table.count,
        expected_gaussians=bundle.verified_gaussians,
        expected_objects=bundle.verified_objects,
    )
    labels = np.repeat(object_ids, np.diff(indptr)).astype(np.int32, copy=False)
    unique_labels, inverse = np.unique(labels, return_inverse=True)
    palette = np.asarray([_instance_color(int(value)) for value in unique_labels], dtype=np.uint8)
    colours = np.ascontiguousarray(palette[inverse])
    centers, covariances, rgbs, opacities = load_gaussian_splat_arrays(
        scene.source_table,
        meters_per_scene_unit=scene.farm.meters_per_scene_unit,
        source_indices=np.ascontiguousarray(indices),
        rgb_override=colours,
    )
    return centers, covariances, rgbs, opacities, np.ascontiguousarray(labels)

def load_dense_lift(
    scene: ValidatedScene,
    *,
    max_points: int,
    verify_full_source: bool = True,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    if not verify_full_source:
        raise UnifiedViewerError("dense point fallback cannot disable full source verification")
    indices, labels = _load_verified_bank(scene)
    use = _stable_sample(np.arange(len(indices), dtype=np.int64), max_points, seed=20260821)
    source_indices = indices[use]
    table = scene.source_table.memmap()
    points = np.column_stack((table["x"][source_indices], table["y"][source_indices], table["z"][source_indices]))
    points = np.ascontiguousarray(points.astype(np.float32) * scene.farm.meters_per_scene_unit)
    selected_labels = np.ascontiguousarray(labels[use].astype(np.int32))
    colors = np.asarray([_instance_color(int(value)) for value in selected_labels], dtype=np.uint8)
    if not np.isfinite(points).all():
        raise UnifiedViewerError("dense lift source indices resolve to non-finite points")
    return points, np.ascontiguousarray(colors), selected_labels



def load_farm_evidence_state(scene: ValidatedScene) -> Mapping[str, Any]:
    """Verify evidence authority and load only saved metadata (no image decode)."""

    evidence = scene.farm.evidence
    if evidence is None:
        raise UnifiedViewerError(scene.layers["object_evidence"].reason)
    evidence.scene_state.verify(field_name="FARM click-inspection scene_state")
    if evidence.mapping_binding is not None:
        evidence.mapping_binding.verify(field_name="FARM mapping evidence")
    if evidence.rgbd_binding is not None:
        evidence.rgbd_binding.verify(field_name="FARM RGBD evidence")
    try:
        import torch
        payload = torch.load(
            evidence.scene_state.path,
            map_location="cpu",
            weights_only=False,
        )
    except Exception as exc:
        raise UnifiedViewerError(f"cannot load trusted FARM scene_state evidence: {exc}") from exc
    state = payload.get("state", payload) if isinstance(payload, Mapping) else None
    if not isinstance(state, Mapping):
        raise UnifiedViewerError("FARM scene_state evidence payload is not a mapping")
    if not isinstance(state.get("object_mask_observations"), (list, tuple)):
        raise UnifiedViewerError("FARM scene_state has no object mask observations")
    if not isinstance(state.get("images"), (list, tuple)):
        raise UnifiedViewerError("FARM scene_state has no COLMAP image records")
    return state


def _evidence_record_field(record: Any, key: str) -> Any:
    return record.get(key) if isinstance(record, Mapping) else getattr(record, key, None)


def _resize_evidence_image(image: np.ndarray, longest_side: int = 320) -> np.ndarray:
    from PIL import Image

    rgb = np.asarray(image, dtype=np.uint8)
    if rgb.ndim != 3 or rgb.shape[2] != 3 or not rgb.size:
        raise UnifiedViewerError("decoded object evidence is not RGB")
    current = max(rgb.shape[:2])
    if current <= longest_side:
        return np.ascontiguousarray(rgb)
    scale = float(longest_side) / float(current)
    size = (max(1, int(round(rgb.shape[1] * scale))), max(1, int(round(rgb.shape[0] * scale))))
    resampling = getattr(Image, "Resampling", Image).LANCZOS
    return np.asarray(Image.fromarray(rgb).resize(size, resampling), dtype=np.uint8).copy()


def _resolve_evidence_archive(evidence: FarmEvidenceBundle, row: Mapping[str, Any]) -> Path | None:
    raw = Path(str(row.get("path") or ""))
    names = [
        evidence.mapping_root / "masks" / raw.parent.name / raw.name,
        evidence.mapping_root / raw.parent.name / raw.name,
        evidence.mapping_root / "masks" / "assemblies" / raw.parent.name / raw.name,
        evidence.mapping_root / raw.name,
    ]
    root = evidence.mapping_root.resolve()
    for candidate in names:
        resolved = candidate.resolve(strict=False)
        if _inside(resolved, root) and resolved.is_file():
            return resolved
    return None


def extract_object_evidence(
    scene: ValidatedScene,
    state: Mapping[str, Any],
    object_id: int,
    *,
    max_images: int = 6,
) -> ObjectEvidenceGallery:
    """Decode the strongest saved COLMAP crops for one selected object only."""

    if max_images < 1:
        raise UnifiedViewerError("max object evidence images must be positive")
    evidence = scene.farm.evidence
    if evidence is None:
        raise UnifiedViewerError(scene.layers["object_evidence"].reason)
    raw_ids = state.get("object_id")
    if hasattr(raw_ids, "detach"):
        raw_ids = raw_ids.detach().cpu().numpy()
    ids = np.asarray(raw_ids).reshape(-1)
    matches = np.flatnonzero(ids == int(object_id))
    observations = state.get("object_mask_observations")
    if len(matches) != 1 or not isinstance(observations, (list, tuple)):
        raise UnifiedViewerError(f"object #{object_id} has no unique scene_state evidence row")
    object_index = int(matches[0])
    if object_index >= len(observations) or not isinstance(observations[object_index], (list, tuple)):
        raise UnifiedViewerError(f"object #{object_id} has no saved mask observations")
    image_records = state.get("images")
    by_image: dict[int, Any] = {}
    if isinstance(image_records, (list, tuple)):
        for record in image_records:
            try:
                by_image[int(_evidence_record_field(record, "image_id"))] = record
            except (TypeError, ValueError):
                continue
    ranked = sorted(
        (row for row in observations[object_index] if isinstance(row, Mapping)),
        key=lambda row: (
            float(row.get("score") or 0.0),
            int(row.get("crop_jpeg_bytes_len") or 0),
        ),
        reverse=True,
    )
    frames: list[ObjectEvidenceFrame] = []
    seen: set[int] = set()
    from PIL import Image

    for row in ranked:
        try:
            image_id = int(row.get("image_id", -1))
        except (TypeError, ValueError):
            continue
        if image_id < 0 or image_id in seen:
            continue
        record = by_image.get(image_id)
        camera_id = str(_evidence_record_field(record, "camera_id") or "unknown")
        source_ref = str(
            _evidence_record_field(record, "source_ref")
            or _evidence_record_field(record, "storage_path")
            or ""
        )
        source_filename = Path(source_ref).name
        bbox_value = row.get("crop_bbox_xyxy")
        bbox: tuple[int, int, int, int] | None = None
        try:
            values = tuple(int(value) for value in bbox_value)
            if len(values) == 4:
                bbox = values
        except (TypeError, ValueError):
            bbox = None
        image: np.ndarray | None = None
        kind = "saved masked crop"
        archive_path = _resolve_evidence_archive(evidence, row)
        if archive_path is not None:
            try:
                with np.load(archive_path, allow_pickle=False) as archive:
                    jpeg = np.asarray(archive["crop_jpeg_bytes"], dtype=np.uint8).tobytes()
                expected = int(row.get("crop_jpeg_bytes_len") or len(jpeg))
                if len(jpeg) != expected:
                    raise UnifiedViewerError("saved crop byte length differs from scene_state")
                with Image.open(io.BytesIO(jpeg)) as decoded:
                    image = np.asarray(decoded.convert("RGB"), dtype=np.uint8).copy()
            except (OSError, KeyError, ValueError) as exc:
                raise UnifiedViewerError(f"cannot decode saved crop for image {image_id}: {exc}") from exc
        elif source_filename:
            frame_path = (evidence.rgbd_root / "rgb" / source_filename).resolve(strict=False)
            rgb_root = evidence.rgbd_root.resolve()
            if _inside(frame_path, rgb_root) and frame_path.is_file():
                try:
                    with Image.open(frame_path) as decoded:
                        decoded = decoded.convert("RGB")
                        width, height = decoded.size
                        if bbox is not None:
                            x0, y0, x1, y1 = bbox
                            x0 = max(0, min(width - 1, x0)); x1 = max(x0 + 1, min(width, x1))
                            y0 = max(0, min(height - 1, y0)); y1 = max(y0 + 1, min(height, y1))
                            decoded = decoded.crop((x0, y0, x1, y1))
                        image = np.asarray(decoded, dtype=np.uint8).copy()
                    kind = "RGB frame bbox fallback"
                except OSError as exc:
                    raise UnifiedViewerError(f"cannot decode RGB frame {image_id}: {exc}") from exc
        if image is None:
            continue
        frames.append(ObjectEvidenceFrame(
            image=_resize_evidence_image(image),
            image_id=image_id,
            camera_id=camera_id,
            score=float(row.get("score") or 0.0),
            bbox_xyxy=bbox,
            source_filename=source_filename,
            source_kind=kind,
        ))
        seen.add(image_id)
        if len(frames) >= max_images:
            break
    if not frames:
        raise UnifiedViewerError(f"object #{object_id} has no decodable saved COLMAP evidence")
    return ObjectEvidenceGallery(
        object_id=int(object_id),
        frames=tuple(frames),
        provenance_note=evidence.provenance_note,
        provenance_warning=evidence.provenance_warning,
    )


def compose_evidence_gallery(gallery: ObjectEvidenceGallery) -> np.ndarray:
    frames = [frame.image for frame in gallery.frames]
    columns = min(3, len(frames))
    rows = int(math.ceil(len(frames) / columns))
    tile_h = max(image.shape[0] for image in frames)
    tile_w = max(image.shape[1] for image in frames)
    canvas = np.zeros((rows * tile_h, columns * tile_w, 3), dtype=np.uint8)
    for index, image in enumerate(frames):
        row, column = divmod(index, columns)
        y = row * tile_h + (tile_h - image.shape[0]) // 2
        x = column * tile_w + (tile_w - image.shape[1]) // 2
        canvas[y:y + image.shape[0], x:x + image.shape[1]] = image
    return canvas

def _normalise(value: Sequence[float], *, field_name: str) -> np.ndarray:
    array = np.asarray(value, dtype=np.float64).reshape(3)
    norm = float(np.linalg.norm(array))
    if not np.isfinite(array).all() or norm <= 1e-9:
        raise UnifiedViewerError(f"invalid {field_name} vector")
    return array / norm


def _horizontal_basis(world_up: Sequence[float]) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    up = _normalise(world_up, field_name="world-up")
    seed = np.asarray([1.0, 0.0, 0.0], dtype=np.float64)
    if abs(float(np.dot(seed, up))) > 0.9:
        seed = np.asarray([0.0, 0.0, 1.0], dtype=np.float64)
    axis_a = seed - float(np.dot(seed, up)) * up
    axis_a /= np.linalg.norm(axis_a)
    axis_b = np.cross(up, axis_a)
    axis_b /= np.linalg.norm(axis_b)
    return axis_a, axis_b, up


def camera_presets(
    points: np.ndarray,
    world_up: Sequence[float],
) -> dict[str, tuple[np.ndarray, np.ndarray, np.ndarray]]:
    """Build deterministic metric overview/front/side/top camera poses."""

    pts = np.asarray(points, dtype=np.float64).reshape(-1, 3)
    pts = pts[np.isfinite(pts).all(axis=1)]
    if len(pts) < 3:
        raise UnifiedViewerError("at least three finite points are required for camera presets")
    axis_a, axis_b, up = _horizontal_basis(world_up)
    origin = np.median(pts, axis=0)
    centered = pts - origin
    coordinates = np.column_stack((centered @ axis_a, centered @ axis_b, centered @ up))
    minimum, maximum = np.percentile(coordinates, (2.0, 98.0), axis=0)
    inside = np.all((coordinates >= minimum) & (coordinates <= maximum), axis=1)
    ground = coordinates[inside, :2] if int(np.count_nonzero(inside)) >= 3 else coordinates[:, :2]
    ground = ground - np.median(ground, axis=0, keepdims=True)
    covariance = ground.T @ ground / max(1, ground.shape[0] - 1)
    try:
        _, vectors = np.linalg.eigh(covariance)
        long_axis = vectors[:, -1]
    except np.linalg.LinAlgError:
        long_axis = np.asarray([1.0, 0.0])
    long_axis /= max(float(np.linalg.norm(long_axis)), 1e-8)
    if long_axis[int(np.argmax(np.abs(long_axis)))] < 0:
        long_axis *= -1
    diagonal = np.asarray([long_axis[1], -long_axis[0]]) + long_axis
    diagonal /= max(float(np.linalg.norm(diagonal)), 1e-8)
    center_coordinates = 0.5 * (minimum + maximum)
    center = origin + center_coordinates[0] * axis_a + center_coordinates[1] * axis_b + center_coordinates[2] * up
    extent = maximum - minimum
    radius = max(0.62 * float(np.linalg.norm(extent)), 2.0)
    standoff = max(0.84 * float(np.linalg.norm(extent[:2])), 2.0)
    view = diagonal[0] * axis_a + diagonal[1] * axis_b
    target = center - 0.15 * float(extent[2]) * up
    return {
        "overview": (
            (center + standoff * view + 0.18 * standoff * up).astype(np.float32),
            target.astype(np.float32),
            up.astype(np.float32),
        ),
        "front": (
            (center - radius * axis_b + 0.18 * radius * up).astype(np.float32),
            center.astype(np.float32),
            up.astype(np.float32),
        ),
        "side": (
            (center + radius * axis_a + 0.18 * radius * up).astype(np.float32),
            center.astype(np.float32),
            up.astype(np.float32),
        ),
        "top": (
            (center + radius * up).astype(np.float32),
            center.astype(np.float32),
            (-axis_b).astype(np.float32),
        ),
    }


def focus_pose(
    bounds: np.ndarray,
    camera_position: Sequence[float],
    camera_look_at: Sequence[float],
    world_up: Sequence[float],
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Frame one object while retaining the current horizontal bearing."""

    box = np.asarray(bounds, dtype=np.float64).reshape(2, 3)
    if not np.isfinite(box).all() or np.any(box[1] < box[0]):
        raise UnifiedViewerError("object bounds must be finite and ordered")
    up = _normalise(world_up, field_name="world-up")
    center = box.mean(axis=0)
    extent = box[1] - box[0]
    bearing = np.asarray(camera_position, dtype=np.float64).reshape(3) - np.asarray(
        camera_look_at, dtype=np.float64
    ).reshape(3)
    ground = bearing - float(np.dot(bearing, up)) * up
    if not np.isfinite(ground).all() or float(np.linalg.norm(ground)) <= 1e-8:
        axis_a, axis_b, _ = _horizontal_basis(up)
        ground = axis_a + axis_b
    ground /= np.linalg.norm(ground)
    view = ground + 0.28 * up
    view /= np.linalg.norm(view)
    distance = max(1.55 * float(np.linalg.norm(extent)), 0.65)
    return (
        (center + distance * view).astype(np.float32),
        center.astype(np.float32),
        up.astype(np.float32),
    )


def _set_markdown(handle: Any, content: str) -> None:
    if hasattr(handle, "content"):
        handle.content = content
    elif hasattr(handle, "value"):
        handle.value = content


def _markdown_text(value: Any) -> str:
    text = " ".join(str(value).replace("\x00", "").splitlines())
    text = _ABSOLUTE_UI_PATH_RE.sub("[local path]", text)
    replacements = {
        "&": "&amp;", "<": "&lt;", ">": "&gt;", "`": "&#96;",
        "[": "&#91;", "]": "&#93;", "(": "&#40;", ")": "&#41;",
        "!": "&#33;", "\\": "&#92;", "*": "&#42;", "_": "&#95;",
    }
    return "".join(replacements.get(character, character) for character in text)


def _short_reason(value: str, limit: int = 420) -> str:
    clean = " ".join(str(value).split())
    return clean if len(clean) <= limit else clean[: limit - 1] + "…"


def is_loopback_host(host: str) -> bool:
    """Return whether a bind target is unambiguously local-only."""

    normalized = str(host).strip().lower()
    if normalized == "localhost":
        return True
    if normalized.startswith("[") and normalized.endswith("]"):
        normalized = normalized[1:-1]
    try:
        return ipaddress.ip_address(normalized).is_loopback
    except ValueError:
        return False


def viewer_exposure_warning(host: str) -> str | None:
    if is_loopback_host(host):
        return None
    return (
        f"viewer bind {host!r} is not loopback; this single-user viewer has no authentication. "
        "Use an authenticated reverse proxy/firewall before exposing it to a network"
    )


def _serialized_mutation(method: Any) -> Any:
    """Serialize scene/layer/UI mutations across Viser callback threads."""

    @functools.wraps(method)
    def wrapped(self: Any, *args: Any, **kwargs: Any) -> Any:
        with self._mutation_lock:
            return method(self, *args, **kwargs)

    return wrapped


@dataclass
class _LayerRuntime:
    root_handle: Any | None = None
    handles: list[Any] = field(default_factory=list)
    loaded: bool = False
    error: str | None = None


@dataclass
class _SceneRuntime:
    scene: ValidatedScene
    root_handle: Any
    layers: dict[str, _LayerRuntime] = field(default_factory=dict)
    visibility: dict[str, bool] = field(default_factory=lambda: {
        "farm_preview": True,
        "farm_obbs": True,
        "source_gaussians": False,
        "bridge_instances": False,
        "exact_lift_review": False,
        "legacy_shaper_review": False,
        "dense_lift": False,
        "dense_lift_points": False,
        "shaper_meshes": False,
        "labels": False,
    })
    camera_points: np.ndarray | None = None
    presets: dict[str, tuple[np.ndarray, np.ndarray, np.ndarray]] | None = None
    selected_object_id: int | None = None
    bounds_by_id: dict[int, np.ndarray] = field(default_factory=dict)
    mesh_rows_by_id: dict[int, Mapping[str, Any]] = field(default_factory=dict)
    review_mesh_ids: set[int] = field(default_factory=set)
    evidence_state: Mapping[str, Any] | None = None
    evidence_galleries: dict[int, ObjectEvidenceGallery] = field(default_factory=dict)


class UnifiedViserViewer:
    """One Viser server and one focal scene canvas for every configured scene."""

    def __init__(
        self,
        scenes: Sequence[ValidatedScene],
        *,
        host: str,
        port: int,
        initial_scene: str | None,
        unavailable_scenes: Sequence[SceneValidationFailure],
        point_size: float,
        max_preview_points: int,
        max_instance_points: int,
        max_dense_points: int,
    ) -> None:
        try:
            import viser
        except Exception as exc:  # pragma: no cover - depends on Docker UI runtime
            raise UnifiedViewerError("Viser is required to serve the unified viewer") from exc
        if getattr(viser, "__version__", None) != REQUIRED_VISER_VERSION:
            raise UnifiedViewerError(
                f"unified viewer requires viser=={REQUIRED_VISER_VERSION}; "
                f"found {getattr(viser, '__version__', 'unknown')}"
            )
        if not scenes:
            raise UnifiedViewerError("unified viewer has no validated scenes")
        if not 1 <= int(port) <= 65535 or point_size <= 0:
            raise UnifiedViewerError("invalid viewer port or point size")
        self.viser = viser
        self.point_size = float(point_size)
        self.max_preview_points = int(max_preview_points)
        self.max_instance_points = int(max_instance_points)
        self.max_dense_points = int(max_dense_points)
        if min(self.max_preview_points, self.max_instance_points, self.max_dense_points) < 1:
            raise UnifiedViewerError("viewer point limits must be positive")
        self._mutation_lock = threading.RLock()
        self.exposure_warning = viewer_exposure_warning(host)
        if self.exposure_warning:
            warnings.warn(self.exposure_warning, RuntimeWarning, stacklevel=2)
        self.server = viser.ViserServer(host=host, port=int(port), label="FARM · unified scenes")
        self.server.gui.configure_theme(
            control_layout="fixed",
            control_width="large",
            dark_mode=True,
            show_logo=False,
            show_share_button=False,
            brand_color=(45, 212, 191),
        )
        background = np.zeros((2, 2, 3), dtype=np.uint8)
        background[...] = [8, 14, 25]
        self.server.scene.set_background_image(background, format="png")
        self.runtime: dict[str, _SceneRuntime] = {}
        self.label_to_id: dict[str, str] = {}
        for scene in scenes:
            option = f"{scene.spec.label} · {scene.spec.scene_id}"
            self.label_to_id[option] = scene.spec.scene_id
            root = self.server.scene.add_frame(
                f"/scenes/{scene.spec.scene_id}", show_axes=False, visible=False
            )
            runtime = _SceneRuntime(scene=scene, root_handle=root)
            runtime.visibility["bridge_instances"] = scene.bridge is not None
            runtime.visibility["legacy_shaper_review"] = scene.review_shaper is not None or scene.legacy_shaper is not None
            self.runtime[scene.spec.scene_id] = runtime
        self.current_scene_id = initial_scene if initial_scene in self.runtime else scenes[0].spec.scene_id
        self.previous_camera: dict[Any, tuple[np.ndarray, np.ndarray, np.ndarray, float]] = {}
        self._updating_controls = False

        initial_option = next(label for label, scene_id in self.label_to_id.items() if scene_id == self.current_scene_id)
        with self.server.gui.add_folder("Scene", expand_by_default=True):
            self.scene_select = self.server.gui.add_dropdown(
                "Scene", options=tuple(self.label_to_id), initial_value=initial_option
            )
            self.scene_status = self.server.gui.add_markdown("")
            security_note = (
                "⚠️ **Security:** " + _markdown_text(self.exposure_warning)
                if self.exposure_warning
                else "🔒 **Security:** single-user inspection service; loopback-only bind; no authentication."
            )
            self.server.gui.add_markdown(security_note)
            if unavailable_scenes:
                unavailable = "\n".join(
                    f"- **{_markdown_text(item.label)}:** {_markdown_text(_short_reason(item.reason))}"
                    for item in unavailable_scenes
                )
                self.server.gui.add_markdown(
                    "**Unavailable registry scenes** (not rendered):\n" + unavailable
                )
        with self.server.gui.add_folder("Independent layers", expand_by_default=True):
            self.show_farm = self.server.gui.add_checkbox(
                "FARM context preview", initial_value=True,
                hint="Deterministic final/cloud.npz point sample; not a 3DGS splat preview.",
            )
            self.show_obbs = self.server.gui.add_checkbox("Final FARM OBBs", initial_value=True)
            self.show_source = self.server.gui.add_checkbox(
                "All source rows (quantized DC)", initial_value=False,
                hint=(
                    "Explicit heavy opt-in: all source PLY rows with source-derived anisotropic "
                    "covariance and opacity, shown as a Viser float16/uint8 quantized DC preview; "
                    "no view-dependent higher-order SH; ~191/283 MB per client."
                ),
            )
            self.show_bridge = self.server.gui.add_checkbox(
                "Bridge instance preview", initial_value=False,
                hint="Legacy 700k-point review layer when compatible with this exact FARM run.",
            )
            self.show_review_lift = self.server.gui.add_checkbox(
                "Exact lift calibration (REVIEW only)", initial_value=False,
                hint="Exact contributor transfer, but uncalibrated/nonrelease; canonical dense lift stays separate.",
            )
            self.show_legacy_shaper = self.server.gui.add_checkbox(
                "ShapeR meshes (REVIEW only)", initial_value=False,
                hint="Exact-calibration ShapeR v2 when available; validated frozen v1 fallback; never canonical.",
            )
            self.show_dense = self.server.gui.add_checkbox(
                "Verified dense-lift splats (all)", initial_value=False,
                hint=(
                    "All verified source rows with source-derived covariance/opacity and instance RGB; "
                    "Viser float16/uint8 quantized preview."
                ),
            )
            self.show_dense_points = self.server.gui.add_checkbox(
                "Dense centres fallback (sampled)", initial_value=False,
                hint="Explicit low-memory point fallback; geometry is sampled and is not a splat render.",
            )
            self.show_shaper = self.server.gui.add_checkbox(
                "Canonical ShapeR v2 hypotheses", initial_value=False,
                hint="Generated surfaces, not measured Gaussian geometry.",
            )
            self.show_labels = self.server.gui.add_checkbox("Object labels", initial_value=False)
            self.point_slider = self.server.gui.add_slider(
                "Point size (m)", min=0.002, max=0.018, step=0.001, initial_value=self.point_size
            )
        with self.server.gui.add_folder("Camera controls", expand_by_default=True):
            self.server.gui.add_markdown(
                "Left-drag orbit · right-drag pan · wheel/pinch zoom. "
                "Click an OBB or mesh to select it and make it the orbit centre; use Focus for a close view."
            )
            self.reset_button = self.server.gui.add_button("Reset overview")
            self.focus_button = self.server.gui.add_button("Focus selected object")
            self.previous_button = self.server.gui.add_button("Previous view")
            self.front_button = self.server.gui.add_button("Front view")
            self.side_button = self.server.gui.add_button("Side view")
            self.top_button = self.server.gui.add_button("Top view")
        with self.server.gui.add_folder("Selected object", expand_by_default=True):
            self.selected_markdown = self.server.gui.add_markdown(
                "Click a final FARM box or ShapeR surface to inspect it."
            )
            evidence_placeholder = np.zeros((64, 64, 3), dtype=np.uint8)
            self.evidence_image = self.server.gui.add_image(
                evidence_placeholder,
                label="Saved COLMAP object views",
                format="jpeg",
            )
            self.evidence_markdown = self.server.gui.add_markdown(
                "Saved object crops are decoded lazily after selection."
            )
        with self.server.gui.add_folder("Source provenance", expand_by_default=False):
            self.source_markdown = self.server.gui.add_markdown("")

        @self.scene_select.on_update
        def _scene_changed(_event: Any = None) -> None:
            with self._mutation_lock:
                if self._updating_controls:
                    return
                self._activate_scene(self.label_to_id[str(self.scene_select.value)])

        layer_controls = {
            "farm_preview": self.show_farm,
            "farm_obbs": self.show_obbs,
            "source_gaussians": self.show_source,
            "bridge_instances": self.show_bridge,
            "exact_lift_review": self.show_review_lift,
            "legacy_shaper_review": self.show_legacy_shaper,
            "dense_lift": self.show_dense,
            "dense_lift_points": self.show_dense_points,
            "shaper_meshes": self.show_shaper,
            "labels": self.show_labels,
        }
        for layer_name, control in layer_controls.items():
            @control.on_update
            def _layer_changed(_event: Any = None, *, name: str = layer_name, handle: Any = control) -> None:
                with self._mutation_lock:
                    if self._updating_controls:
                        return
                    runtime = self.runtime[self.current_scene_id]
                    requested = bool(handle.value)
                    runtime.visibility[name] = requested
                    if requested:
                        try:
                            self._ensure_layer(runtime, name)
                        except UnifiedViewerError as exc:
                            runtime.visibility[name] = False
                            self._updating_controls = True
                            handle.value = False
                            self._updating_controls = False
                            _set_markdown(self.scene_status, self._status_markdown(runtime, error=str(exc)))
                    elif name in {"source_gaussians", "exact_lift_review", "dense_lift"}:
                        self._unload_layer(runtime, name)
                    self._apply_layer_visibility(runtime, name)

        @self.point_slider.on_update
        def _point_size_changed(_event: Any = None) -> None:
            with self._mutation_lock:
                self.point_size = float(self.point_slider.value)
                for runtime in self.runtime.values():
                    for layer_name, multiplier in (
                        ("farm_preview", 1.0), ("bridge_instances", 1.35), ("dense_lift_points", 1.5)
                    ):
                        layer = runtime.layers.get(layer_name)
                        if layer and layer.loaded:
                            for handle in layer.handles:
                                if hasattr(handle, "point_size"):
                                    handle.point_size = self.point_size * multiplier

        @self.reset_button.on_click
        def _reset(_event: Any = None) -> None:
            self._apply_preset("overview")

        @self.front_button.on_click
        def _front(_event: Any = None) -> None:
            self._apply_preset("front")

        @self.side_button.on_click
        def _side(_event: Any = None) -> None:
            self._apply_preset("side")

        @self.top_button.on_click
        def _top(_event: Any = None) -> None:
            self._apply_preset("top")

        @self.previous_button.on_click
        def _previous(_event: Any = None) -> None:
            with self._mutation_lock:
                for client in self.server.get_clients().values():
                    key = self._client_key(client)
                    previous = self.previous_camera.get(key)
                    if previous is None:
                        continue
                    current = self._camera_state(client)
                    self._set_camera(client, previous[:3], remember=False)
                    client.camera.fov = previous[3]
                    self.previous_camera[key] = current

        @self.focus_button.on_click
        def _focus(_event: Any = None) -> None:
            with self._mutation_lock:
                runtime = self.runtime[self.current_scene_id]
                object_id = runtime.selected_object_id
                if object_id is None or object_id not in runtime.bounds_by_id:
                    _set_markdown(self.selected_markdown, "Select a final FARM box or ShapeR surface first.")
                    return
                for client in self.server.get_clients().values():
                    pose = focus_pose(
                        runtime.bounds_by_id[object_id],
                        client.camera.position,
                        client.camera.look_at,
                        runtime.scene.farm.world_up,
                    )
                    self._set_camera(client, pose)

        @self.server.on_client_connect
        def _client_connected(client: Any) -> None:
            def apply() -> None:
                time.sleep(0.2)
                with self._mutation_lock:
                    runtime = self.runtime[self.current_scene_id]
                    if runtime.presets:
                        self._set_camera(client, runtime.presets["overview"], remember=False)
            threading.Thread(target=apply, daemon=True).start()

        self._activate_scene(self.current_scene_id, first=True)

    @staticmethod
    def _client_key(client: Any) -> Any:
        return getattr(client, "client_id", id(client))

    @staticmethod
    def _camera_state(client: Any) -> tuple[np.ndarray, np.ndarray, np.ndarray, float]:
        return (
            np.asarray(client.camera.position, dtype=np.float32).copy(),
            np.asarray(client.camera.look_at, dtype=np.float32).copy(),
            np.asarray(client.camera.up_direction, dtype=np.float32).copy(),
            float(client.camera.fov),
        )

    @_serialized_mutation
    def _set_camera(
        self,
        client: Any,
        pose: tuple[np.ndarray, np.ndarray, np.ndarray],
        *,
        remember: bool = True,
    ) -> None:
        if remember:
            self.previous_camera[self._client_key(client)] = self._camera_state(client)
        position, look_at, up = pose
        client.camera.position = np.asarray(position, dtype=np.float32).copy()
        client.camera.look_at = np.asarray(look_at, dtype=np.float32).copy()
        client.camera.up_direction = np.asarray(up, dtype=np.float32).copy()
        client.camera.fov = np.deg2rad(52.0)

    @_serialized_mutation
    def _apply_preset(self, name: str) -> None:
        runtime = self.runtime[self.current_scene_id]
        if not runtime.presets:
            return
        for client in self.server.get_clients().values():
            self._set_camera(client, runtime.presets[name])

    @_serialized_mutation
    def _activate_scene(self, scene_id: str, *, first: bool = False) -> None:
        for candidate_id, runtime in self.runtime.items():
            if candidate_id != scene_id:
                for heavy in ("source_gaussians", "exact_lift_review", "dense_lift"):
                    runtime.visibility[heavy] = False
                    self._unload_layer(runtime, heavy)
            runtime.root_handle.visible = candidate_id == scene_id
        self.current_scene_id = scene_id
        runtime = self.runtime[scene_id]
        self.server.scene.set_up_direction(tuple(float(value) for value in runtime.scene.farm.world_up))
        self._updating_controls = True
        controls = {
            "farm_preview": self.show_farm,
            "farm_obbs": self.show_obbs,
            "source_gaussians": self.show_source,
            "bridge_instances": self.show_bridge,
            "exact_lift_review": self.show_review_lift,
            "legacy_shaper_review": self.show_legacy_shaper,
            "dense_lift": self.show_dense,
            "dense_lift_points": self.show_dense_points,
            "shaper_meshes": self.show_shaper,
            "labels": self.show_labels,
        }
        for name, handle in controls.items():
            available = True if name == "labels" else runtime.scene.layers[name].ready
            handle.disabled = not available
            if not available:
                runtime.visibility[name] = False
            handle.value = bool(runtime.visibility[name])
        self._updating_controls = False
        for name, visible in runtime.visibility.items():
            if visible:
                self._ensure_layer(runtime, name)
            self._apply_layer_visibility(runtime, name)
        runtime.selected_object_id = None
        _set_markdown(self.selected_markdown, "Click a final FARM box or ShapeR surface to inspect it.")
        self.evidence_image.image = np.zeros((64, 64, 3), dtype=np.uint8)
        evidence_state = runtime.scene.layers["object_evidence"]
        evidence_text = (
            "Saved object crops are decoded lazily after selection."
            if evidence_state.ready
            else "Saved COLMAP views unavailable: " + _short_reason(evidence_state.reason)
        )
        _set_markdown(self.evidence_markdown, evidence_text)
        _set_markdown(self.scene_status, self._status_markdown(runtime))
        _set_markdown(self.source_markdown, self._source_markdown(runtime))
        if not first:
            self._apply_preset("overview")

    @_serialized_mutation
    def _layer_runtime(self, runtime: _SceneRuntime, name: str) -> _LayerRuntime:
        if name not in runtime.layers:
            root = self.server.scene.add_frame(
                f"/scenes/{runtime.scene.spec.scene_id}/{name}", show_axes=False, visible=False
            )
            runtime.layers[name] = _LayerRuntime(root_handle=root)
        return runtime.layers[name]

    @_serialized_mutation
    def _apply_layer_visibility(self, runtime: _SceneRuntime, name: str) -> None:
        layer = runtime.layers.get(name)
        if layer and layer.root_handle is not None:
            layer.root_handle.visible = bool(runtime.visibility.get(name, False))

    @_serialized_mutation
    def _unload_layer(self, runtime: _SceneRuntime, name: str) -> None:
        layer = runtime.layers.pop(name, None)
        if layer is None:
            return
        try:
            if layer.root_handle is not None and hasattr(layer.root_handle, "remove"):
                layer.root_handle.remove()
                return
        except Exception:
            pass
        for handle in reversed(layer.handles):
            try:
                if hasattr(handle, "remove"):
                    handle.remove()
            except Exception:
                pass

    @_serialized_mutation
    def _ensure_layer(self, runtime: _SceneRuntime, name: str) -> None:
        layer = self._layer_runtime(runtime, name)
        if layer.loaded:
            return
        if layer.error:
            raise UnifiedViewerError(layer.error)
        if name != "labels" and not runtime.scene.layers[name].ready:
            raise UnifiedViewerError(runtime.scene.layers[name].reason)
        _set_markdown(self.scene_status, self._status_markdown(runtime, loading=name))
        try:
            if name == "farm_preview":
                self._load_farm_layer(runtime, layer)
            elif name == "farm_obbs":
                self._load_obb_layer(runtime, layer)
            elif name == "source_gaussians":
                self._load_source_layer(runtime, layer)
            elif name == "bridge_instances":
                self._load_bridge_layer(runtime, layer)
            elif name == "exact_lift_review":
                self._load_review_lift_layer(runtime, layer)
            elif name == "legacy_shaper_review":
                self._load_legacy_shaper_layer(runtime, layer)
            elif name == "dense_lift":
                self._load_dense_layer(runtime, layer)
            elif name == "dense_lift_points":
                self._load_dense_points_layer(runtime, layer)
            elif name == "shaper_meshes":
                self._load_shaper_layer(runtime, layer)
            elif name == "labels":
                self._load_label_layer(runtime, layer)
            else:
                raise UnifiedViewerError(f"unknown viewer layer: {name}")
            layer.loaded = True
        except Exception as exc:
            layer.error = str(exc)
            if isinstance(exc, UnifiedViewerError):
                raise
            raise UnifiedViewerError(f"failed to load {name}: {exc}") from exc
        finally:
            _set_markdown(self.scene_status, self._status_markdown(runtime))

    def _load_farm_layer(self, runtime: _SceneRuntime, layer: _LayerRuntime) -> None:
        points, colors = load_farm_preview(runtime.scene, max_points=self.max_preview_points)
        handle = self.server.scene.add_point_cloud(
            f"/scenes/{runtime.scene.spec.scene_id}/farm_preview/points",
            points=points,
            colors=colors,
            point_size=self.point_size,
            point_shape="circle",
            point_shading="flat",
            precision="float16",
        )
        layer.handles.append(handle)
        runtime.camera_points = points
        runtime.presets = camera_presets(points, runtime.scene.farm.world_up)

    def _load_obb_layer(self, runtime: _SceneRuntime, layer: _LayerRuntime) -> None:
        runtime.scene.farm.catalog_binding.verify(field_name="FARM presentation catalog")
        for row in runtime.scene.farm.catalog:
            object_id = int(row["id"])
            center = np.asarray(row["center_m"], dtype=np.float32)
            dimensions = np.asarray(row["dimensions_m"], dtype=np.float32)
            wxyz = np.asarray(row["wxyz"], dtype=np.float32)
            wxyz /= np.linalg.norm(wxyz)
            color = _instance_color(object_id)
            handle = self.server.scene.add_box(
                f"/scenes/{runtime.scene.spec.scene_id}/farm_obbs/{object_id:06d}",
                color=color,
                dimensions=dimensions,
                wireframe=True,
                opacity=0.85,
                side="double",
                cast_shadow=False,
                receive_shadow=False,
                wxyz=wxyz,
                position=center,
            )

            @handle.on_click
            def _clicked(_event: Any = None, captured: int = object_id) -> None:
                self._select_object(runtime, captured)

            layer.handles.append(handle)
            half = 0.5 * float(np.linalg.norm(dimensions))
            runtime.bounds_by_id[object_id] = np.stack((center - half, center + half)).astype(np.float32)

    def _load_source_layer(self, runtime: _SceneRuntime, layer: _LayerRuntime) -> None:
        centers, covariances, rgbs, opacities = load_source_gaussian_splats(runtime.scene)
        handle = self.server.scene.add_gaussian_splats(
            f"/scenes/{runtime.scene.spec.scene_id}/source_gaussians/all_original_dc_only",
            centers=centers,
            covariances=covariances,
            rgbs=rgbs,
            opacities=opacities,
        )
        layer.handles.append(handle)

    def _load_bridge_layer(self, runtime: _SceneRuntime, layer: _LayerRuntime) -> None:
        points, colors, _ = load_bridge_instances(
            runtime.scene, max_points=self.max_instance_points
        )
        handle = self.server.scene.add_point_cloud(
            f"/scenes/{runtime.scene.spec.scene_id}/bridge_instances/points",
            points=points,
            colors=colors,
            point_size=self.point_size * 1.35,
            point_shape="circle",
            point_shading="flat",
            precision="float16",
        )
        layer.handles.append(handle)

    def _load_review_lift_layer(self, runtime: _SceneRuntime, layer: _LayerRuntime) -> None:
        centers, covariances, rgbs, opacities, _ = load_review_gaussian_splats(runtime.scene)
        handle = self.server.scene.add_gaussian_splats(
            f"/scenes/{runtime.scene.spec.scene_id}/exact_lift_review/all_verified_instance_splats",
            centers=centers,
            covariances=covariances,
            rgbs=rgbs,
            opacities=opacities,
        )
        layer.handles.append(handle)

    def _load_legacy_shaper_layer(self, runtime: _SceneRuntime, layer: _LayerRuntime) -> None:
        review = runtime.scene.review_shaper
        if review is not None:
            for index, binding in enumerate(review.artifact_bindings):
                binding.verify(field_name=f"review ShapeR v2 artifact[{index}]")
            with np.load(review.mesh_npz.path, allow_pickle=False) as archive:
                vertices = np.asarray(archive["vertices"], dtype=np.float32)
                faces = np.asarray(archive["faces"], dtype=np.int32)
                object_ids = np.asarray(archive["object_id"], dtype=np.int32)
                colours = np.asarray(archive["colors"], dtype=np.uint8)
            face_object_ids = object_ids[faces[:, 0]]
            for object_id, row in review.object_rows.items():
                object_faces = faces[face_object_ids == object_id]
                if not len(object_faces):
                    raise UnifiedViewerError(f"review ShapeR object #{object_id} has no faces")
                used = np.unique(object_faces.reshape(-1))
                local_faces = np.searchsorted(used, object_faces).astype(np.uint32)
                local_vertices = np.ascontiguousarray(vertices[used])
                local_colours = np.ascontiguousarray(colours[used])
                if not len(local_colours) or np.any(local_colours != local_colours[0]):
                    raise UnifiedViewerError(f"review ShapeR object #{object_id} RGB changed")
                handle = self.server.scene.add_mesh_simple(
                    f"/scenes/{runtime.scene.spec.scene_id}/legacy_shaper_review/{object_id:06d}",
                    vertices=local_vertices,
                    faces=local_faces,
                    color=tuple(int(channel) for channel in local_colours[0]),
                    material="standard",
                    flat_shading=False,
                    side="double",
                    cast_shadow=False,
                    receive_shadow=False,
                )

                @handle.on_click
                def _clicked(_event: Any = None, captured: int = object_id) -> None:
                    self._select_object(runtime, captured)

                layer.handles.append(handle)
                runtime.bounds_by_id[object_id] = np.asarray(row["bounds_m"], dtype=np.float32).reshape(2, 3)
                runtime.mesh_rows_by_id[object_id] = row
                runtime.review_mesh_ids.add(object_id)
            return

        bundle = runtime.scene.legacy_shaper
        if bundle is None:
            raise UnifiedViewerError(runtime.scene.layers["legacy_shaper_review"].reason)
        bundle.scene_manifest.verify(field_name="legacy ShapeR scene manifest")
        bundle.viewer_assets.verify(field_name="legacy ShapeR viewer assets")
        for object_id, binding in bundle.asset_bindings.items():
            binding.verify(field_name=f"legacy ShapeR asset #{object_id}")
            with np.load(binding.path, allow_pickle=False) as archive:
                vertices = np.asarray(archive["vertices"], dtype=np.float32)
                faces = np.asarray(archive["faces"], dtype=np.uint32)
                color = tuple(int(channel) for channel in np.asarray(archive["color"], dtype=np.uint8))
            handle = self.server.scene.add_mesh_simple(
                f"/scenes/{runtime.scene.spec.scene_id}/legacy_shaper_review/{object_id:06d}",
                vertices=vertices,
                faces=faces,
                color=color,
                material="standard",
                flat_shading=False,
                side="double",
                cast_shadow=False,
                receive_shadow=False,
            )

            @handle.on_click
            def _clicked(_event: Any = None, captured: int = object_id) -> None:
                self._select_object(runtime, captured)

            layer.handles.append(handle)
            row = bundle.object_rows[object_id]
            runtime.bounds_by_id[object_id] = np.asarray(row["bounds_m"], dtype=np.float32).reshape(2, 3)
            runtime.mesh_rows_by_id[object_id] = row
            runtime.review_mesh_ids.add(object_id)

    def _load_dense_layer(self, runtime: _SceneRuntime, layer: _LayerRuntime) -> None:
        centers, covariances, rgbs, opacities, _ = load_dense_gaussian_splats(runtime.scene)
        handle = self.server.scene.add_gaussian_splats(
            f"/scenes/{runtime.scene.spec.scene_id}/dense_lift/all_verified_instance_splats",
            centers=centers,
            covariances=covariances,
            rgbs=rgbs,
            opacities=opacities,
        )
        layer.handles.append(handle)

    def _load_dense_points_layer(self, runtime: _SceneRuntime, layer: _LayerRuntime) -> None:
        points, colors, _ = load_dense_lift(
            runtime.scene, max_points=self.max_dense_points, verify_full_source=True
        )
        handle = self.server.scene.add_point_cloud(
            f"/scenes/{runtime.scene.spec.scene_id}/dense_lift_points/verified_centres_sampled",
            points=points,
            colors=colors,
            point_size=self.point_size * 1.5,
            point_shape="circle",
            point_shading="flat",
            precision="float16",
        )
        layer.handles.append(handle)

    def _load_label_layer(self, runtime: _SceneRuntime, layer: _LayerRuntime) -> None:
        runtime.scene.farm.catalog_binding.verify(field_name="FARM presentation catalog")
        for row in runtime.scene.farm.catalog:
            object_id = int(row["id"])
            category = _markdown_text(row.get("category") or "object")
            handle = self.server.scene.add_label(
                f"/scenes/{runtime.scene.spec.scene_id}/labels/{object_id:06d}",
                f"#{object_id}  {category}",
                position=np.asarray(row["center_m"], dtype=np.float32),
                font_size_mode="screen",
                font_screen_scale=0.72,
                depth_test=True,
                anchor="bottom-center",
            )
            layer.handles.append(handle)

    def _load_shaper_layer(self, runtime: _SceneRuntime, layer: _LayerRuntime) -> None:
        shaper = runtime.scene.shaper
        if shaper is None:
            raise UnifiedViewerError(runtime.scene.layers["shaper_meshes"].reason)
        for index, binding in enumerate(shaper.artifact_bindings):
            binding.verify(field_name=f"ShapeR artifact[{index}]")
        with np.load(shaper.mesh_npz.path, allow_pickle=False) as archive:
            vertices = np.asarray(archive["vertices"], dtype=np.float32)
            faces = np.asarray(archive["faces"], dtype=np.int32)
            object_ids = np.asarray(archive["object_id"], dtype=np.int32)
            colours = np.asarray(archive["colors"], dtype=np.uint8)
        face_object_ids = object_ids[faces[:, 0]]
        for object_id, row in shaper.object_rows.items():
            object_faces = faces[face_object_ids == object_id]
            if not len(object_faces):
                raise UnifiedViewerError(f"ShapeR object #{object_id} has no faces in combined NPZ")
            used = np.unique(object_faces.reshape(-1))
            local_faces = np.searchsorted(used, object_faces).astype(np.uint32)
            local_vertices = np.ascontiguousarray(vertices[used])
            local_colours = np.ascontiguousarray(colours[used])
            if not len(local_colours) or np.any(local_colours != local_colours[0]):
                raise UnifiedViewerError(
                    f"ShapeR object #{object_id} no longer has one uniform RGB color"
                )
            mesh_color = tuple(int(channel) for channel in local_colours[0])
            handle = self.server.scene.add_mesh_simple(
                f"/scenes/{runtime.scene.spec.scene_id}/shaper_meshes/{object_id:06d}",
                vertices=local_vertices,
                faces=local_faces,
                color=mesh_color,
                material="standard",
                flat_shading=False,
                side="double",
                cast_shadow=False,
                receive_shadow=False,
            )

            @handle.on_click
            def _clicked(_event: Any = None, captured: int = object_id) -> None:
                self._select_object(runtime, captured)

            layer.handles.append(handle)
            bounds = np.asarray(row.get("bounds_m"), dtype=np.float32).reshape(2, 3)
            runtime.bounds_by_id[object_id] = bounds
            runtime.mesh_rows_by_id[object_id] = row

    @_serialized_mutation
    def _select_object(self, runtime: _SceneRuntime, object_id: int) -> None:
        if runtime.scene.spec.scene_id != self.current_scene_id:
            return
        runtime.scene.farm.catalog_binding.verify(field_name="FARM presentation catalog")
        object_id = int(object_id)
        runtime.selected_object_id = object_id
        catalog = {int(row["id"]): row for row in runtime.scene.farm.catalog}
        row = catalog.get(object_id, {})
        category = _markdown_text(row.get("category") or "object")
        description = _markdown_text(row.get("description") or "No description.")
        lines = [
            f"## #{object_id} · {category}",
            description,
            f"**FARM evidence:** {_markdown_text(row.get('evidence_tier', 'unknown'))} · "
            f"semantic {_markdown_text(row.get('semantic_tier', 'unknown'))}",
        ]
        if runtime.scene.bridge and object_id in runtime.scene.bridge.instance_rows:
            source = runtime.scene.bridge.instance_rows[object_id]
            lines.append(
                "**Legacy instance preview:** "
                f"{int(source.get('final_gaussians', 0)):,} assigned · "
                f"{int(source.get('observations', 0))} observations · "
                f"median confidence {float(source.get('median_final_confidence', 0.0)):.3f}"
            )
        mesh = runtime.mesh_rows_by_id.get(object_id)
        if mesh is not None and isinstance(mesh.get("geometry_qa"), Mapping):
            qa = mesh["geometry_qa"]
            label = "ShapeR REVIEW QA" if object_id in runtime.review_mesh_ids else "Canonical ShapeR v2 QA"
            lines.append(
                f"**{label}:** "
                f"{int(qa.get('vertices', qa.get('vertex_count', 0))):,} vertices · "
                f"{int(qa.get('faces', qa.get('face_count', 0))):,} faces · "
                f"centre error {float(qa.get('center_error_m', 0.0)):.3f} m"
            )
        if runtime.scene.review_lift is not None:
            lines.append(
                "⚠️ **Exact contributor lift (REVIEW only):** uncalibrated/nonrelease; "
                "unknown and ambiguous source Gaussians stay unassigned."
            )
        if runtime.scene.dense_lift is not None:
            lines.append(
                "**Canonical dense contributor lift:** labels contain verified objects only; "
                "unobserved/ambiguous Gaussians remain unknown."
            )
        _set_markdown(self.selected_markdown, "\n\n".join(lines))

        blank = np.zeros((64, 64, 3), dtype=np.uint8)
        self.evidence_image.image = blank
        evidence_layer = runtime.scene.layers["object_evidence"]
        if not evidence_layer.ready:
            _set_markdown(
                self.evidence_markdown,
                "— **Saved COLMAP views unavailable:** " + _markdown_text(_short_reason(evidence_layer.reason)),
            )
        else:
            try:
                if runtime.evidence_state is None:
                    runtime.evidence_state = load_farm_evidence_state(runtime.scene)
                gallery = runtime.evidence_galleries.get(object_id)
                if gallery is None:
                    gallery = extract_object_evidence(
                        runtime.scene, runtime.evidence_state, object_id, max_images=6
                    )
                    runtime.evidence_galleries[object_id] = gallery
                self.evidence_image.image = compose_evidence_gallery(gallery)
                icon = "⚠️" if gallery.provenance_warning else "✅"
                gallery_lines = [f"{icon} {_markdown_text(gallery.provenance_note)}"]
                for index, frame in enumerate(gallery.frames, start=1):
                    bbox = "—" if frame.bbox_xyxy is None else ",".join(str(value) for value in frame.bbox_xyxy)
                    gallery_lines.append(
                        f"{index}. image `{frame.image_id}` · camera `{_markdown_text(frame.camera_id)}` · "
                        f"score {frame.score:.3f} · bbox `{bbox}` · "
                        f"{_markdown_text(frame.source_kind)} · `{_markdown_text(frame.source_filename or 'embedded')}`"
                    )
                _set_markdown(self.evidence_markdown, "\n\n".join(gallery_lines))
            except Exception as exc:
                self.evidence_image.image = blank
                _set_markdown(
                    self.evidence_markdown,
                    "⚠️ **Saved COLMAP views could not be decoded:** "
                    + _markdown_text(_short_reason(str(exc))),
                )

        bounds = runtime.bounds_by_id.get(object_id)
        if bounds is not None:
            centre = np.asarray(bounds, dtype=np.float32).reshape(2, 3).mean(axis=0)
            up = np.asarray(runtime.scene.farm.world_up, dtype=np.float32)
            for client in self.server.get_clients().values():
                self.previous_camera[self._client_key(client)] = self._camera_state(client)
                client.camera.look_at = centre.copy()
                client.camera.up_direction = up.copy()

    def _status_markdown(
        self,
        runtime: _SceneRuntime,
        *,
        loading: str | None = None,
        error: str | None = None,
    ) -> str:
        scene = runtime.scene
        rows = [
            f"### {_markdown_text(scene.spec.label)}",
            f"FARM `{scene.farm.run_id}` · config `{scene.farm.config_sha256[:12]}`",
        ]
        if loading:
            rows.append(f"⏳ Loading `{loading}` read-only…")
        if error:
            rows.append(f"❌ {_markdown_text(_short_reason(error))}")
        for name in (
            "object_evidence",
            "bridge_instances",
            "exact_lift_review",
            "legacy_shaper_review",
            "source_gaussians",
            "dense_lift",
            "shaper_meshes",
        ):
            state = scene.layers[name]
            icon = "✅" if state.ready and not state.warning else "⚠️" if state.ready else "—"
            rows.append(f"{icon} **{name.replace('_', ' ')}:** {_markdown_text(_short_reason(state.reason))}")
        rows.append(
            "ℹ️ **Splat preview contract:** all selected source rows; Viser float16/uint8 "
            "quantized DC preview, without view-dependent higher-order SH."
        )
        return "\n\n".join(rows)

    def _source_markdown(self, runtime: _SceneRuntime) -> str:
        scene = runtime.scene
        size_gib = scene.source_table.path.stat().st_size / float(1 << 30)
        packed_mb = scene.source_table.count * VISER_PACKED_BYTES_PER_GAUSSIAN / 1_000_000.0
        warning = "⚠️" if scene.farm.source_warning else "✅"
        integrity = "⚠️" if scene.farm.integrity_warning else "✅"
        return (
            f"**Validated source asset**  \n`{_markdown_text(scene.source_table.path.name)}`  \n"
            f"{scene.source_table.count:,} Gaussian rows · {size_gib:.2f} GiB  \n"
            f"{scene.spec.source_fingerprint.algorithm}: `{scene.spec.source_fingerprint.digest}`  \n"
            f"{warning} {_markdown_text(scene.farm.source_note)}  \n"
            f"{integrity} {_markdown_text(scene.farm.integrity_note)}\n\n"
            f"**All source rows** is an explicit heavy opt-in: ~{packed_mb:.0f} MB packed per client. "
            "It is a Viser float16/uint8 quantized DC preview of every source row, with source-derived "
            "anisotropic covariance and sigmoid opacity in metric coordinates. Viser does not evaluate "
            "higher-order SH, so this is not a bit-exact or view-dependent reproduction."
        )

    def run_forever(self) -> None:
        try:
            while True:
                time.sleep(1.0)
        except KeyboardInterrupt:
            return


def serve_registry(
    registry_path: Path,
    *,
    validation: RegistryValidation | None = None,
    host: str = "127.0.0.1",
    port: int = 8080,
    initial_scene: str | None = None,
    point_size: float = 0.004,
    max_preview_points: int = 1_500_000,
    max_instance_points: int = 700_000,
    max_dense_points: int = 700_000,
) -> UnifiedViserViewer:
    resolved_registry = registry_path.expanduser().resolve(strict=True)
    if validation is None:
        validation = validate_registry(resolved_registry, verify_full_source=False)
    elif validation.registry.path != resolved_registry:
        raise UnifiedViewerError("prevalidated registry path differs from the requested registry")
    failures = tuple(
        item for item in validation.scenes if isinstance(item, SceneValidationFailure)
    )
    if not validation.ready_scenes:
        raise UnifiedViewerError(
            "registry has no validated scene: "
            + "; ".join(f"{item.scene_id}: {item.reason}" for item in failures)
        )
    viewer = UnifiedViserViewer(
        validation.ready_scenes,
        host=host,
        port=port,
        initial_scene=initial_scene,
        unavailable_scenes=failures,
        point_size=point_size,
        max_preview_points=max_preview_points,
        max_instance_points=max_instance_points,
        max_dense_points=max_dense_points,
    )
    return viewer
