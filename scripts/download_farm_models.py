#!/usr/bin/env python3
"""Download and verify exactly the models pinned by a FARM model manifest."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
import sys
import tempfile
import urllib.request
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping, Sequence
from urllib.parse import urlparse


PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = PROJECT_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from farm_pipeline.resources import (  # noqa: E402
    ModelManifest,
    ModelSpec,
    ResourceConfigError,
    load_model_manifest,
    sha256_file,
)


DEFAULT_MANIFEST = PROJECT_ROOT / "configs" / "models" / "farm_models.v1.json"
SHA256_RE = re.compile(r"^[0-9a-f]{64}$")
REVISION_RE = re.compile(r"^[0-9a-f]{40}$")
SOURCE_KINDS = {"url", "huggingface_snapshot"}


class ModelDownloadError(ValueError):
    """Raised when download provenance or a resulting artifact is invalid."""


@dataclass(frozen=True)
class LocalSource:
    model_name: str
    target: Path
    expected_sha256: str
    kind: str | None
    url: str | None = None
    repo_id: str | None = None
    revision: str | None = None
    required_files: tuple[str, ...] = ()


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, default=DEFAULT_MANIFEST)
    parser.add_argument(
        "--secrets-file",
        type=Path,
        default=Path(os.environ.get("FARM_SECRETS_FILE", PROJECT_ROOT.parent / "secrets.json")),
        help="Optional JSON containing HF_TOKEN; the value is never printed",
    )
    parser.add_argument(
        "--local-files-only",
        action="store_true",
        help="Forbid network access and only validate/download from local HF cache",
    )
    parser.add_argument(
        "--verify-full-hashes",
        action="store_true",
        help="Read complete HF blobs instead of validating content-addressed blob names",
    )
    return parser.parse_args(argv)


def _mapping(value: Any, label: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        raise ModelDownloadError(f"{label} must be a mapping")
    return value


def _relative_files(value: Any, label: str) -> tuple[str, ...]:
    if not isinstance(value, list) or not value:
        raise ModelDownloadError(f"{label} must be a non-empty list")
    output: list[str] = []
    for index, item in enumerate(value):
        if not isinstance(item, str) or not item.strip():
            raise ModelDownloadError(f"{label}[{index}] must be a path string")
        path = Path(item)
        if path.is_absolute() or not path.parts or any(part in {"", ".", ".."} for part in path.parts):
            raise ModelDownloadError(f"{label}[{index}] is not a safe relative path")
        output.append(path.as_posix())
    if len(set(output)) != len(output):
        raise ModelDownloadError(f"{label} contains duplicate paths")
    return tuple(output)


def load_raw_manifest(path: Path) -> Mapping[str, Any]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        raise ModelDownloadError(f"invalid JSON manifest {path}: {exc}") from exc
    if not isinstance(value, Mapping) or value.get("schema_version") != "farm.models.v1":
        raise ModelDownloadError("manifest schema_version must be 'farm.models.v1'")
    return value


def parse_local_sources(
    raw: Mapping[str, Any], manifest: ModelManifest
) -> tuple[LocalSource, ...]:
    pipeline_raw = _mapping(raw.get("pipeline_models"), "pipeline_models")
    if set(pipeline_raw) != set(manifest.pipeline_models):
        raise ModelDownloadError("raw/validated pipeline model key sets differ")
    sources: list[LocalSource] = []
    for name in sorted(manifest.pipeline_models):
        spec = manifest.pipeline_models[name]
        if spec.kind != "local_file" or not spec.sha256 or not SHA256_RE.fullmatch(spec.sha256):
            raise ModelDownloadError(f"pipeline model {name!r} must be a SHA256-pinned local_file")
        row = _mapping(pipeline_raw[name], f"pipeline_models.{name}")
        source_value = row.get("source")
        if source_value is None:
            sources.append(LocalSource(name, spec.local_path, spec.sha256, None))
            continue
        source = _mapping(source_value, f"pipeline_models.{name}.source")
        unknown = sorted(set(source) - {"kind", "url", "repo_id", "revision", "required_files"})
        if unknown:
            raise ModelDownloadError(
                f"pipeline_models.{name}.source has unknown fields: {', '.join(unknown)}"
            )
        kind = str(source.get("kind") or "")
        if kind not in SOURCE_KINDS:
            raise ModelDownloadError(f"pipeline model {name!r} has unsupported source kind {kind!r}")
        if kind == "url":
            url = str(source.get("url") or "")
            parsed = urlparse(url)
            if parsed.scheme != "https" or not parsed.netloc or parsed.username or parsed.password:
                raise ModelDownloadError(f"pipeline model {name!r} requires one credential-free HTTPS URL")
            if any(source.get(field) is not None for field in ("repo_id", "revision", "required_files")):
                raise ModelDownloadError(f"pipeline model {name!r} URL source has incompatible fields")
            sources.append(LocalSource(name, spec.local_path, spec.sha256, kind, url=url))
            continue
        repo_id = str(source.get("repo_id") or "")
        revision = str(source.get("revision") or "").lower()
        if not repo_id or "/" not in repo_id or not REVISION_RE.fullmatch(revision):
            raise ModelDownloadError(
                f"pipeline model {name!r} requires repo_id and immutable 40-hex revision"
            )
        required = _relative_files(
            source.get("required_files"), f"pipeline_models.{name}.source.required_files"
        )
        target_relative = spec.local_path.relative_to(spec.local_path.parent).as_posix()
        if target_relative not in required:
            raise ModelDownloadError(
                f"pipeline model {name!r} source.required_files omits target {target_relative!r}"
            )
        sources.append(
            LocalSource(
                name,
                spec.local_path,
                spec.sha256,
                kind,
                repo_id=repo_id,
                revision=revision,
                required_files=required,
            )
        )
    return tuple(sources)


def load_token(path: Path) -> str | None:
    environment = os.environ.get("HF_TOKEN", "").strip()
    if not path.is_file():
        return environment or None
    try:
        raw = json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        raise ModelDownloadError(f"invalid secrets JSON {path}: {exc}") from exc
    if not isinstance(raw, Mapping):
        raise ModelDownloadError("secrets JSON must contain an object")
    value = raw.get("HF_TOKEN")
    if value is not None and not isinstance(value, str):
        raise ModelDownloadError("secrets HF_TOKEN must be a string")
    return str(value).strip() if value and str(value).strip() else (environment or None)


def redact_error_message(message: str, secrets_file: Path) -> str:
    """Remove known token values even when a dependency includes one in an exception."""

    secrets: set[str] = set()
    environment = os.environ.get("HF_TOKEN", "").strip()
    if environment:
        secrets.add(environment)
    try:
        raw = json.loads(secrets_file.expanduser().read_text(encoding="utf-8"))
    except (OSError, UnicodeError, json.JSONDecodeError):
        raw = None
    if isinstance(raw, Mapping):
        value = raw.get("HF_TOKEN")
        if isinstance(value, str) and value.strip():
            secrets.add(value.strip())
    redacted = message
    for secret in sorted(secrets, key=len, reverse=True):
        redacted = redacted.replace(secret, "<redacted>")
    return redacted


def _snapshot_download(**kwargs: Any) -> str:
    try:
        from huggingface_hub import snapshot_download
    except ImportError as exc:
        raise ModelDownloadError("huggingface_hub is required for pinned model download") from exc
    try:
        return str(snapshot_download(**kwargs))
    except Exception as exc:
        # Hub clients raise several requests/httpx-specific exception classes.
        # Normalise all of them so main() can emit one redacted, structured
        # failure instead of an unhandled traceback that might echo a token.
        raise ModelDownloadError(f"Hugging Face snapshot download failed: {exc}") from exc


def _download_url(source: LocalSource) -> None:
    assert source.url is not None
    source.target.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary_name = tempfile.mkstemp(
        prefix=f".{source.target.name}.", suffix=".download", dir=source.target.parent
    )
    temporary = Path(temporary_name)
    try:
        digest = hashlib.sha256()
        with os.fdopen(descriptor, "wb") as destination:
            with urllib.request.urlopen(source.url, timeout=60) as response:  # noqa: S310
                final_url = urlparse(str(response.geturl()))
                if final_url.scheme != "https":
                    raise ModelDownloadError(
                        f"pipeline model {source.model_name!r} redirected outside HTTPS"
                    )
                while chunk := response.read(8 * 1024 * 1024):
                    destination.write(chunk)
                    digest.update(chunk)
            destination.flush()
            os.fsync(destination.fileno())
        if digest.hexdigest() != source.expected_sha256:
            raise ModelDownloadError(
                f"pipeline model {source.model_name!r} downloaded SHA256 mismatch"
            )
        os.replace(temporary, source.target)
    finally:
        temporary.unlink(missing_ok=True)


def download_services(
    manifest: ModelManifest,
    *,
    token: str | None,
    local_files_only: bool,
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    seen: set[tuple[str, str]] = set()
    for service_name in sorted(manifest.services):
        model = manifest.services[service_name].model
        assert model.repo_id is not None and model.revision is not None
        identity = (model.repo_id, model.revision)
        if identity in seen:
            continue
        seen.add(identity)
        required = [item.path for item in model.required_files]
        returned = Path(
            _snapshot_download(
                repo_id=model.repo_id,
                revision=model.revision,
                cache_dir=str(manifest.cache_root / "hub"),
                allow_patterns=required,
                token=token,
                local_files_only=local_files_only,
            )
        ).expanduser().resolve(strict=True)
        expected = model.local_path.expanduser().resolve(strict=True)
        if returned != expected:
            raise ModelDownloadError(
                f"pinned snapshot path mismatch for {model.repo_id}: {returned} != {expected}"
            )
        rows.append(
            {
                "kind": "huggingface_snapshot",
                "name": model.repo_id,
                "revision": model.revision,
                "target": str(expected),
            }
        )
    return rows


def download_local_sources(
    sources: Sequence[LocalSource],
    *,
    token: str | None,
    local_files_only: bool,
    cache_root: Path | None = None,
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for source in sources:
        if source.target.is_file() and sha256_file(source.target) == source.expected_sha256:
            action = "already_verified"
        elif source.kind is None:
            raise ModelDownloadError(
                f"pipeline model {source.model_name!r} is missing/mismatched and has no download source"
            )
        elif local_files_only and source.kind == "url":
            raise ModelDownloadError(
                f"pipeline model {source.model_name!r} is not locally verified and URL access is disabled"
            )
        elif source.kind == "url":
            _download_url(source)
            action = "downloaded"
        else:
            assert source.repo_id is not None and source.revision is not None
            source.target.parent.mkdir(parents=True, exist_ok=True)
            returned = Path(
                _snapshot_download(
                    repo_id=source.repo_id,
                    revision=source.revision,
                    cache_dir=(str(cache_root / "hub") if cache_root is not None else None),
                    local_dir=str(source.target.parent),
                    allow_patterns=list(source.required_files),
                    token=token,
                    local_files_only=local_files_only,
                )
            ).expanduser().resolve(strict=True)
            if returned != source.target.parent.resolve(strict=True):
                raise ModelDownloadError(
                    f"local snapshot path mismatch for {source.model_name!r}: {returned}"
                )
            action = "downloaded"
        actual = sha256_file(source.target) if source.target.is_file() else ""
        if actual != source.expected_sha256:
            raise ModelDownloadError(f"pipeline model {source.model_name!r} SHA256 mismatch")
        rows.append(
            {
                "kind": source.kind or "preprovisioned",
                "name": source.model_name,
                "revision": source.revision,
                "target": str(source.target),
                "sha256": actual,
                "action": action,
            }
        )
    return rows


def verify_snapshot(model: ModelSpec, *, full_hashes: bool) -> dict[str, Any]:
    if not model.local_path.is_dir():
        raise ModelDownloadError(f"pinned snapshot is missing: {model.local_path}")
    checked = 0
    for requirement in model.required_files:
        candidate = model.local_path / requirement.path
        if not candidate.is_file() or candidate.stat().st_size <= 0:
            raise ModelDownloadError(f"required snapshot file is missing/empty: {candidate}")
        if requirement.sha256:
            resolved = candidate.resolve(strict=True)
            if requirement.checksum_mode == "content_addressed" and not full_hashes:
                actual = resolved.name.lower()
                if not SHA256_RE.fullmatch(actual):
                    raise ModelDownloadError(f"snapshot file is not SHA256-addressed: {candidate}")
            else:
                actual = sha256_file(candidate)
            if actual != requirement.sha256:
                raise ModelDownloadError(f"snapshot checksum mismatch: {candidate}")
        checked += 1
    return {
        "repo_id": model.repo_id,
        "revision": model.revision,
        "target": str(model.local_path),
        "required_files_verified": checked,
    }


def execute(args: argparse.Namespace) -> dict[str, Any]:
    manifest_path = args.manifest.expanduser().resolve(strict=True)
    raw = load_raw_manifest(manifest_path)
    manifest = load_model_manifest(manifest_path)
    sources = parse_local_sources(raw, manifest)
    token = load_token(args.secrets_file.expanduser())
    downloads = download_services(
        manifest, token=token, local_files_only=bool(args.local_files_only)
    )
    downloads.extend(
        download_local_sources(
            sources,
            token=token,
            local_files_only=bool(args.local_files_only),
            cache_root=manifest.cache_root,
        )
    )
    snapshots = []
    seen: set[tuple[str | None, str | None]] = set()
    for name in sorted(manifest.services):
        model = manifest.services[name].model
        identity = (model.repo_id, model.revision)
        if identity in seen:
            continue
        seen.add(identity)
        snapshots.append(verify_snapshot(model, full_hashes=bool(args.verify_full_hashes)))
    return {
        "schema": "farm.model-download.v2",
        "status": "PASS",
        "manifest": str(manifest_path),
        "manifest_sha256": sha256_file(manifest_path),
        "local_files_only": bool(args.local_files_only),
        "full_hash_verification": bool(args.verify_full_hashes),
        "token_source": "configured" if token else "absent",
        "downloads": downloads,
        "service_snapshots": snapshots,
    }


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    try:
        report = execute(args)
        print(json.dumps(report, ensure_ascii=False, indent=2, sort_keys=True))
        return 0
    except (ModelDownloadError, ResourceConfigError, OSError, UnicodeError) as exc:
        error = redact_error_message(f"{type(exc).__name__}: {exc}", args.secrets_file)
        print(
            json.dumps(
                {
                    "schema": "farm.model-download.v2",
                    "status": "ERROR",
                    "manifest": str(args.manifest.expanduser().absolute()),
                    "error": error,
                },
                ensure_ascii=False,
                indent=2,
                sort_keys=True,
            ),
            file=sys.stderr,
        )
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
