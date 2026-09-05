#!/usr/bin/env python3
"""Build a self-contained, explicitly legacy/non-release FARM experiment run."""

from __future__ import annotations

# Allow direct execution from the source checkout.
import sys as _entry_sys
from pathlib import Path as _EntryPath
_entry_root = _EntryPath(__file__).resolve().parents[2]
_entry_sys.path[:0] = [str(_entry_root), str(_entry_root / "src")]

import argparse
import copy
import json
import os
import re
import shutil
import time
from pathlib import Path

IDENTITY = re.compile(
    r"(?:^|/)cam[0-9]+_(?P<timestamp>[0-9]+)_"
    r"(?:center|yaw_left|yaw_right|pitch_up|pitch_down)[.](?:png|jpg|jpeg)$",
    re.IGNORECASE,
)


def _atomic_json(path: Path, payload: object) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(payload, indent=2, ensure_ascii=False, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    temporary.replace(path)


def _link(source: Path, destination: Path) -> None:
    source = source.expanduser().resolve(strict=True)
    if source.is_symlink() or not source.is_file():
        raise ValueError(f"source must be a regular non-symlink file: {source}")
    destination.parent.mkdir(parents=True, exist_ok=True)
    if destination.exists():
        raise FileExistsError(destination)
    try:
        os.link(source, destination)
    except OSError:
        shutil.copy2(source, destination)


def _link_tree(source_root: Path, destination_root: Path, *, suffix: str = "") -> int:
    source_root = source_root.expanduser().resolve(strict=True)
    count = 0
    for source in sorted(source_root.rglob("*")):
        if source.is_symlink():
            raise ValueError(f"source tree contains a symlink: {source}")
        if source.is_dir():
            continue
        if suffix and source.suffix.lower() != suffix:
            raise ValueError(f"unexpected file in source tree: {source}")
        _link(source, destination_root / source.relative_to(source_root))
        count += 1
    return count


def _parse_source_aliases(values: list[str]) -> dict[str, Path]:
    aliases: dict[str, Path] = {}
    for value in values:
        name, separator, raw_root = value.partition("=")
        if not separator or not name or "/" in name or name in {".", ".."}:
            raise ValueError(f"invalid --source-alias, expected NAME=ROOT: {value}")
        if name in aliases:
            raise ValueError(f"duplicate --source-alias: {name}")
        aliases[name] = Path(raw_root).expanduser().resolve(strict=True)
    return aliases


def _validate_presentation_catalog(path: Path) -> list[int]:
    """Fail before writing when a full catalog is passed as presentation data."""

    resolved = path.expanduser().resolve(strict=True)
    payload = json.loads(resolved.read_text(encoding="utf-8"))
    if not isinstance(payload, list) or not payload:
        raise ValueError("presentation catalog must be a non-empty JSON array")
    identifiers: list[int] = []
    for index, row in enumerate(payload):
        if not isinstance(row, dict):
            raise ValueError(f"presentation catalog row {index} must be a JSON object")
        if row.get("presentation_visible") is not True:
            raise ValueError(
                f"presentation catalog row {index} is not explicitly visible"
            )
        try:
            object_id = int(row["id"])
        except (KeyError, TypeError, ValueError) as exc:
            raise ValueError(
                f"presentation catalog row {index} has no valid object ID"
            ) from exc
        if object_id < 0:
            raise ValueError("presentation catalog object IDs must be non-negative")
        identifiers.append(object_id)
    if len(identifiers) != len(set(identifiers)):
        raise ValueError("presentation catalog object IDs must be unique")
    return identifiers


def _load_unique_input_frames(path: Path) -> tuple[Path, dict, list[dict]]:
    """Load the frame manifest and fail closed on duplicate physical sources."""

    resolved = path.expanduser().resolve(strict=True)
    payload = json.loads(resolved.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError("frames manifest must be a JSON object")
    raw_frames = payload.get("frames")
    if not isinstance(raw_frames, list) or not raw_frames:
        raise ValueError("frames manifest is empty")

    frames: list[dict] = []
    source_indices: dict[str, int] = {}
    for index, row in enumerate(raw_frames):
        if not isinstance(row, dict):
            raise ValueError(f"frames manifest row {index} must be a JSON object")
        source_image = row.get("source_image")
        if not isinstance(source_image, str) or not source_image.strip():
            raise ValueError(
                f"frames manifest row {index} has no valid source_image"
            )
        if source_image in source_indices:
            raise ValueError(
                "input frames contain duplicate source_image "
                f"{source_image!r} at rows {source_indices[source_image]} and {index}"
            )
        source_indices[source_image] = index
        frames.append(row)
    return resolved, payload, frames


def _resolve_source_path(
    frames_path: Path, raw_path: str, aliases: dict[str, Path]
) -> Path:
    candidate = frames_path.parent / raw_path
    try:
        return candidate.resolve(strict=True)
    except FileNotFoundError as original_error:
        parts = Path(raw_path).parts
        matches = [
            (index, aliases[part])
            for index, part in enumerate(parts)
            if part in aliases
        ]
        if len(matches) != 1:
            raise original_error
        index, root = matches[0]
        resolved = root.joinpath(*parts[index + 1 :]).resolve(strict=True)
        if not resolved.is_relative_to(root):
            raise ValueError(f"source alias escapes its root: {raw_path}")
        return resolved


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--template-run", type=Path, required=True)
    parser.add_argument("--scene-state", type=Path, required=True)
    parser.add_argument("--catalog", type=Path, required=True)
    parser.add_argument("--presentation-catalog", type=Path, required=True)
    parser.add_argument("--cloud", type=Path, required=True)
    parser.add_argument("--frames-json", type=Path, required=True)
    parser.add_argument("--mask-root", type=Path, required=True)
    parser.add_argument(
        "--source-alias",
        action="append",
        default=[],
        metavar="NAME=ROOT",
        help="Resolve a named path component in portable frame paths against ROOT.",
    )
    parser.add_argument("--output-run", type=Path, required=True)
    args = parser.parse_args()
    started = time.perf_counter()

    template = args.template_run.expanduser().resolve(strict=True)
    presentation_object_ids = _validate_presentation_catalog(args.presentation_catalog)
    source_aliases = _parse_source_aliases(args.source_alias)
    frames_path, frames_doc, frames = _load_unique_input_frames(args.frames_json)
    output = args.output_run.expanduser().resolve()
    if output.exists():
        raise FileExistsError(f"refusing to overwrite experiment run: {output}")
    output.mkdir(parents=True)
    for relative in (
        "final",
        "viewer",
        "input",
        "rgbd/rgb",
        "rgbd/depth",
        "mapping/masks",
    ):
        (output / relative).mkdir(parents=True, exist_ok=True)

    for source, relative in (
        (args.scene_state, "final/scene_state.pt"),
        (args.catalog, "final/catalog.json"),
        (args.presentation_catalog, "final/presentation_catalog.json"),
        (args.cloud, "final/cloud.npz"),
        (template / "input/resource_preflight.json", "input/resource_preflight.json"),
        (template / "input/scene_preflight.json", "input/scene_preflight.json"),
        (template / "input/resolved_context.json", "input/resolved_context.json"),
        (template / "rgbd/run_manifest.json", "rgbd/run_manifest.json"),
        (template / "rgbd/prep_summary.json", "rgbd/prep_summary.json"),
        (template / "final/result.json", "final/result.json"),
        (template / "viewer/result.json", "viewer/result.json"),
        (template / "viewer/bundle.json", "viewer/bundle.json"),
    ):
        _link(Path(source), output / relative)

    normalized_frames = []
    linked_assets = 0
    for image_id, source_row in enumerate(frames):
        row = copy.deepcopy(source_row)
        match = IDENTITY.search(str(row.get("source_image") or ""))
        if match is None:
            raise ValueError(
                f"cannot recover physical timestamp: {row.get('source_image')}"
            )
        timestamp = match.group("timestamp")
        row["frame_id"] = timestamp
        row["timestamp_ns"] = int(timestamp) * 100_000_000
        for key, subdirectory in (("rgb_path", "rgb"), ("depth_path", "depth")):
            source = _resolve_source_path(frames_path, str(row[key]), source_aliases)
            name = f"{image_id:06d}_{source.name}"
            _link(source, output / "rgbd" / subdirectory / name)
            row[key] = f"{subdirectory}/{name}"
            linked_assets += 1
        normalized_frames.append(row)
    frames_doc["frames"] = normalized_frames
    frames_doc["experiment_identity_normalization"] = {
        "source": "source_image timestamp token",
        "timestamp_ns_scale": 100_000_000,
        "physical_timestamps": len({row["frame_id"] for row in normalized_frames}),
    }
    _atomic_json(output / "rgbd/frames.json", frames_doc)
    linked_masks = _link_tree(
        args.mask_root.expanduser().resolve(strict=True),
        output / "mapping/masks",
        suffix=".npz",
    )

    template_success = json.loads(
        (template / "_SUCCESS.json").read_text(encoding="utf-8")
    )
    success = {
        "schema": "farm.pipeline-success.v1",
        "status": "success",
        "scene_id": str(template_success["scene_id"]),
        "run_id": output.name,
        "config_sha256": str(template_success["config_sha256"]),
        "viewer_bundle": "viewer/bundle.json",
        "viewer_bundle_sha256": "",
        "review_mode": "explicit_legacy_nonrelease_experiment",
    }
    manifest = {
        "schema": "farm.pipeline-run.v1",
        "status": "success",
        "scene_id": success["scene_id"],
        "run_id": output.name,
        "config_sha256": success["config_sha256"],
        "source_snapshot": None,
        "review_mode": "explicit_legacy_nonrelease_experiment",
    }
    _atomic_json(output / "_SUCCESS.json", success)
    _atomic_json(output / "manifest.json", manifest)
    report = {
        "schema": "farm.experiment-run-build.v1",
        "status": "PASS",
        "run_id": output.name,
        "frames": len(normalized_frames),
        "physical_timestamps": frames_doc["experiment_identity_normalization"][
            "physical_timestamps"
        ],
        "presentation_objects": len(presentation_object_ids),
        "linked_rgbd_assets": linked_assets,
        "linked_masks": linked_masks,
        "hardlink_preferred": True,
        "release_eligible": False,
        "timing_seconds": time.perf_counter() - started,
    }
    _atomic_json(output / "experiment_build.json", report)
    print(json.dumps(report, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
