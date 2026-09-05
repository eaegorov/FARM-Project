"""Fail-closed contracts shared by multi-episode tracker benchmarks.

The GPU runners intentionally live in ``scripts/`` because their optional
dependencies are large.  This module contains the CPU-only provenance,
atomic-output and 8-bit mask checks so they can be tested without CUDA.
"""

from __future__ import annotations

import hashlib
import json
import os
import resource
import shutil
import tempfile
import threading
from contextlib import contextmanager
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable, Iterator

import numpy as np

MATERIALIZED_SUITE_SCHEMA = "farm.materialized-tracker-episodes.v1"
MATERIALIZED_EPISODE_SCHEMA = "farm.materialized-tracker-episode.v1"


def sha256_file(path: Path, chunk_size: int = 8 * 1024 * 1024) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while chunk := handle.read(chunk_size):
            digest.update(chunk)
    return digest.hexdigest()


def sha256_lines(values: Iterable[str]) -> str:
    payload = "".join(f"{value}\n" for value in values).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


@dataclass(frozen=True)
class EpisodeInput:
    episode_id: str
    camera: str
    view_family: str
    split: str
    root: Path
    frame_root: Path
    frame_list: Path
    episode_json: Path
    episode_json_sha256: str
    frame_names: tuple[str, ...]
    source_names: tuple[str, ...]
    physical_timestamps: tuple[str, ...]
    frame_sha256: tuple[str, ...]
    ordered_frame_names_sha256: str
    ordered_frame_content_sha256: str

    @property
    def frame_count(self) -> int:
        return len(self.frame_names)


@dataclass(frozen=True)
class SuiteInput:
    root: Path
    manifest_path: Path
    manifest_sha256: str
    source_plan: Path
    source_plan_sha256: str
    episodes: tuple[EpisodeInput, ...]


def _safe_leaf(value: object, *, label: str) -> str:
    text = str(value or "")
    if not text or Path(text).name != text or text in {".", ".."}:
        raise ValueError(f"unsafe {label}: {text!r}")
    return text


def _read_json(path: Path) -> dict[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError(f"JSON root must be an object: {path}")
    return payload


def load_materialized_suite(
    root: Path,
    *,
    episode_ids: set[str] | None = None,
    splits: set[str] | None = None,
    verify_frame_hashes: bool = True,
) -> SuiteInput:
    """Load and content-verify materialized episodes before any model is built."""

    root = root.expanduser().resolve(strict=True)
    manifest_path = root / "manifest.json"
    manifest = _read_json(manifest_path)
    if manifest.get("schema") != MATERIALIZED_SUITE_SCHEMA:
        raise ValueError("unsupported materialized tracker suite schema")
    manifest_hash = sha256_file(manifest_path)

    source_plan = (
        Path(str(manifest.get("source_plan") or "")).expanduser().resolve(strict=True)
    )
    expected_plan_hash = str(manifest.get("source_plan_sha256") or "")
    actual_plan_hash = sha256_file(source_plan)
    if not expected_plan_hash or actual_plan_hash != expected_plan_hash:
        raise ValueError("source plan checksum mismatch")
    if manifest.get("hashes_verified") is not True:
        raise ValueError(
            "materialized input was not created with verified frame hashes"
        )

    requested_splits = set(splits or ())
    if requested_splits.difference({"train", "heldout"}):
        raise ValueError("splits must contain only train and/or heldout")
    requested_ids = set(episode_ids or ())
    rows = manifest.get("episodes")
    if not isinstance(rows, list) or not rows:
        raise ValueError("materialized suite has no episode rows")
    if int(manifest.get("episode_count", -1)) != len(rows):
        raise ValueError("materialized manifest episode_count mismatch")
    declared_frame_count = 0
    available_ids: set[str] = set()
    selected_rows: list[dict[str, Any]] = []
    for raw in rows:
        if not isinstance(raw, dict):
            raise ValueError("episode manifest row must be an object")
        episode_id = _safe_leaf(raw.get("episode_id"), label="episode ID")
        if episode_id in available_ids:
            raise ValueError(f"duplicate episode ID: {episode_id}")
        available_ids.add(episode_id)
        if str(raw.get("relative_path") or "") != episode_id:
            raise ValueError(f"episode relative_path mismatch: {episode_id}")
        declared_frame_count += int(raw.get("frame_count", -1))
        split = str(raw.get("split") or "")
        if split not in {"train", "heldout"}:
            raise ValueError(f"invalid episode split: {split!r}")
        if (not requested_ids or episode_id in requested_ids) and (
            not requested_splits or split in requested_splits
        ):
            selected_rows.append(raw)
    if int(manifest.get("frame_count", -1)) != declared_frame_count:
        raise ValueError("materialized manifest frame_count mismatch")
    missing = sorted(requested_ids.difference(available_ids))
    if missing:
        raise ValueError("unknown episode IDs: " + ", ".join(missing))
    if not selected_rows:
        raise ValueError("episode selection is empty")

    episodes: list[EpisodeInput] = []
    seen_sources: set[str] = set()
    timestamp_split: dict[str, str] = {}
    for row in selected_rows:
        episode_id = str(row["episode_id"])
        episode_root = root / episode_id
        if episode_root.is_symlink() or not episode_root.is_dir():
            raise ValueError(f"episode root must be a real directory: {episode_root}")
        payload_path = episode_root / "episode.json"
        payload = _read_json(payload_path)
        if payload.get("schema") != MATERIALIZED_EPISODE_SCHEMA:
            raise ValueError(f"unsupported episode schema: {episode_id}")
        for key in ("episode_id", "camera", "view_family", "split", "frame_count"):
            if payload.get(key) != row.get(key):
                raise ValueError(f"episode/manifest {key} mismatch: {episode_id}")
        if str(payload.get("source_plan_sha256") or "") != actual_plan_hash:
            raise ValueError(f"episode source plan checksum mismatch: {episode_id}")
        episode_source_plan = Path(str(payload.get("source_plan") or "")).resolve(
            strict=True
        )
        if episode_source_plan != source_plan:
            raise ValueError(f"episode source plan path mismatch: {episode_id}")

        frames = payload.get("frames")
        if not isinstance(frames, list) or not frames:
            raise ValueError(f"episode has no frames: {episode_id}")
        if len(frames) != int(row["frame_count"]):
            raise ValueError(f"episode frame count mismatch: {episode_id}")
        frame_root = episode_root / "frames"
        frame_list = episode_root / "frames.txt"
        listed_names = [
            line.strip()
            for line in frame_list.read_text(encoding="utf-8").splitlines()
            if line.strip() and not line.lstrip().startswith("#")
        ]
        expected_names: list[str] = []
        source_names: list[str] = []
        timestamps: list[str] = []
        frame_hashes: list[str] = []
        for index, frame in enumerate(frames):
            if (
                not isinstance(frame, dict)
                or int(frame.get("episode_index", -1)) != index
            ):
                raise ValueError(f"non-contiguous episode_index: {episode_id}")
            name = _safe_leaf(frame.get("materialized_name"), label="frame name")
            source_name = _safe_leaf(
                frame.get("source_name"), label="source frame name"
            )
            expected_hash = str(frame.get("sha256") or "")
            if len(expected_hash) != 64:
                raise ValueError(f"missing exact frame hash: {episode_id}/{name}")
            link = frame_root / name
            if not link.is_symlink():
                raise ValueError(f"materialized frame must remain a symlink: {link}")
            target = link.resolve(strict=True)
            declared_source = Path(str(frame.get("source_path") or "")).resolve(
                strict=True
            )
            if target != declared_source:
                raise ValueError(f"materialized frame target changed: {link}")
            if int(frame.get("bytes", -1)) != target.stat().st_size:
                raise ValueError(f"materialized frame size changed: {link}")
            if verify_frame_hashes and sha256_file(target) != expected_hash:
                raise ValueError(f"materialized frame checksum changed: {link}")
            timestamp = str(frame.get("physical_timestamp") or "")
            if not timestamp:
                raise ValueError(f"frame lacks physical timestamp: {link}")
            split = str(payload["split"])
            previous_split = timestamp_split.setdefault(timestamp, split)
            if previous_split != split:
                raise ValueError(
                    f"physical timestamp leaks across train/heldout: {timestamp}"
                )
            if source_name in seen_sources:
                raise ValueError(
                    f"source frame is repeated across episodes: {source_name}"
                )
            seen_sources.add(source_name)
            expected_names.append(name)
            source_names.append(source_name)
            timestamps.append(timestamp)
            frame_hashes.append(expected_hash)
        if listed_names != expected_names:
            raise ValueError(f"frames.txt order/content mismatch: {episode_id}")
        entries = sorted(path.name for path in frame_root.iterdir())
        if entries != sorted(expected_names):
            raise ValueError(f"frames directory content mismatch: {episode_id}")
        episodes.append(
            EpisodeInput(
                episode_id=episode_id,
                camera=str(payload["camera"]),
                view_family=str(payload["view_family"]),
                split=str(payload["split"]),
                root=episode_root,
                frame_root=frame_root,
                frame_list=frame_list,
                episode_json=payload_path,
                episode_json_sha256=sha256_file(payload_path),
                frame_names=tuple(expected_names),
                source_names=tuple(source_names),
                physical_timestamps=tuple(timestamps),
                frame_sha256=tuple(frame_hashes),
                ordered_frame_names_sha256=sha256_lines(expected_names),
                ordered_frame_content_sha256=sha256_lines(
                    f"{name}\0{digest}"
                    for name, digest in zip(expected_names, frame_hashes)
                ),
            )
        )
    return SuiteInput(
        root=root,
        manifest_path=manifest_path,
        manifest_sha256=manifest_hash,
        source_plan=source_plan,
        source_plan_sha256=actual_plan_hash,
        episodes=tuple(episodes),
    )


@contextmanager
def atomic_output_directory(destination: Path) -> Iterator[Path]:
    """Publish a directory only after its complete successful construction."""

    destination = destination.expanduser().resolve()
    if destination.exists():
        raise FileExistsError(f"refusing to overwrite suite output: {destination}")
    destination.parent.mkdir(parents=True, exist_ok=True)
    staging = Path(
        tempfile.mkdtemp(prefix=f".{destination.name}.tmp-", dir=destination.parent)
    )
    try:
        yield staging
        os.replace(staging, destination)
    except BaseException:
        shutil.rmtree(staging, ignore_errors=True)
        raise


def _current_rss_mib() -> float:
    try:
        for line in Path("/proc/self/status").read_text(encoding="utf-8").splitlines():
            if line.startswith("VmRSS:"):
                return float(line.split()[1]) / 1024.0
    except (FileNotFoundError, OSError, ValueError, IndexError):
        pass
    value = float(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss)
    return value / 1024.0


class PeakRssSampler:
    """Small polling sampler giving episode-local RSS rather than process max RSS."""

    def __init__(self, interval_seconds: float = 0.05) -> None:
        if interval_seconds <= 0:
            raise ValueError("RSS sampling interval must be positive")
        self.interval_seconds = float(interval_seconds)
        self.start_rss_mib = 0.0
        self.peak_rss_mib = 0.0
        self.end_rss_mib = 0.0
        self._stop = threading.Event()
        self._thread: threading.Thread | None = None

    def __enter__(self) -> "PeakRssSampler":
        self.start_rss_mib = _current_rss_mib()
        self.peak_rss_mib = self.start_rss_mib

        def sample() -> None:
            while not self._stop.wait(self.interval_seconds):
                self.peak_rss_mib = max(self.peak_rss_mib, _current_rss_mib())

        self._thread = threading.Thread(target=sample, daemon=True)
        self._thread.start()
        return self

    def __exit__(self, exc_type: object, exc: object, traceback: object) -> None:
        self._stop.set()
        if self._thread is not None:
            self._thread.join(timeout=max(1.0, self.interval_seconds * 4.0))
        self.end_rss_mib = _current_rss_mib()
        self.peak_rss_mib = max(self.peak_rss_mib, self.end_rss_mib)

    def report(self) -> dict[str, float]:
        current = _current_rss_mib()
        end = self.end_rss_mib if self.end_rss_mib > 0 else current
        return {
            "start_rss_mib": self.start_rss_mib,
            "peak_rss_mib": max(self.peak_rss_mib, current),
            "end_rss_mib": end,
        }


def audit_8bit_masks(
    mask_root: Path,
    expected_frame_names: Iterable[str],
    *,
    reference_frame_root: Path | None = None,
) -> dict[str, Any]:
    """Require exactly one uint8 PNG per frame and forbid reserved ID 255."""

    from PIL import Image

    source_frame_names = list(expected_frame_names)
    expected_names = [f"{Path(name).stem}.png" for name in source_frame_names]
    source_by_mask = dict(zip(expected_names, source_frame_names))
    entries = sorted(mask_root.iterdir(), key=lambda path: path.name)
    actual_names = [path.name for path in entries if path.is_file()]
    if actual_names != sorted(expected_names) or len(entries) != len(expected_names):
        raise ValueError("tracker emitted an incomplete or unexpected mask set")
    identities: set[int] = set()
    rows: list[dict[str, Any]] = []
    for path in entries:
        with Image.open(path) as image:
            array = np.asarray(image)
        if array.ndim != 2 or array.dtype != np.uint8:
            raise ValueError(
                f"identity mask must be a single-channel uint8 PNG: {path}"
            )
        unique = np.unique(array)
        maximum = int(unique[-1]) if unique.size else 0
        if maximum > 254 or 255 in unique:
            raise ValueError(f"identity mask exceeds the 8-bit 1..254 budget: {path}")
        if reference_frame_root is not None:
            with Image.open(
                reference_frame_root / source_by_mask[path.name]
            ) as reference:
                expected_shape = (reference.height, reference.width)
            if array.shape != expected_shape:
                raise ValueError(
                    f"identity mask/RGB shape mismatch: {path}; "
                    f"mask={array.shape}, rgb={expected_shape}"
                )
        identities.update(int(value) for value in unique if value != 0)
        rows.append(
            {
                "name": path.name,
                "bytes": path.stat().st_size,
                "sha256": sha256_file(path),
                "maximum_local_id": maximum,
                "foreground_pixels": int(np.count_nonzero(array)),
            }
        )
    if len(identities) > 254:
        raise ValueError("tracker suite episode exceeds 254 local identities")
    return {
        "mask_count": len(rows),
        "local_identity_count": len(identities),
        "maximum_local_id": max(identities, default=0),
        "masks": rows,
    }
