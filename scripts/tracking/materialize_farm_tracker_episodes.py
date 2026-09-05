#!/usr/bin/env python3
"""Materialize an audited tracker plan as tiny, ordered symlink episodes."""

from __future__ import annotations

# Allow direct execution from the source checkout.
import sys as _entry_sys
from pathlib import Path as _EntryPath
_entry_root = _EntryPath(__file__).resolve().parents[2]
_entry_sys.path[:0] = [str(_entry_root), str(_entry_root / "src")]

import argparse
import hashlib
import json
import os
import shutil
import tempfile
from pathlib import Path
from typing import Any


PLAN_SCHEMA = "farm.tracker-subset-plan.v1"
PAIRED_PLAN_SCHEMA = "farm.tracker-subset-plan.v2"
PLAN_SCHEMAS = {PLAN_SCHEMA, PAIRED_PLAN_SCHEMA}
OUTPUT_SCHEMA = "farm.materialized-tracker-episodes.v1"


def _sha256(path: Path, chunk_size: int = 8 * 1024 * 1024) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while True:
            chunk = handle.read(chunk_size)
            if not chunk:
                break
            digest.update(chunk)
    return digest.hexdigest()


def _safe_source(root: Path, name: str) -> Path:
    source = (root / name).resolve(strict=True)
    try:
        source.relative_to(root)
    except ValueError as exc:
        raise ValueError(f"frame escapes image root: {name!r}") from exc
    if not source.is_file():
        raise ValueError(f"frame is not a file: {source}")
    return source


def _materialized_name(index: int, source_name: str) -> str:
    name = Path(source_name).name
    if name != source_name:
        raise ValueError(f"COLMAP frame name must be a basename: {source_name!r}")
    return f"{int(index):06d}__{name}"


def _validate_episode(episode: dict[str, Any]) -> None:
    frames = episode.get("frames")
    if not isinstance(frames, list) or not frames:
        raise ValueError("episode has no frames")
    names = [str(frame.get("name") or "") for frame in frames]
    if not all(names) or len(names) != len(set(names)):
        raise ValueError("episode contains empty or duplicate source names")
    for index, frame in enumerate(frames):
        if int(frame.get("episode_index", -1)) != index:
            raise ValueError("episode_index is not contiguous")
    if int(episode.get("frame_count", -1)) != len(frames):
        raise ValueError("episode frame_count disagrees with frames")


def _validate_paired_plan(episodes: list[dict[str, Any]]) -> None:
    """Preserve the v2 object/evidence/split contract during materialization."""

    by_object: dict[int, set[str]] = {}
    names: set[str] = set()
    split_timestamps = {"train": set(), "heldout": set()}
    for episode in episodes:
        _validate_episode(episode)
        object_id = int(episode.get("anchor_object_id", -1))
        split = str(episode.get("split") or "")
        if object_id < 0 or split not in split_timestamps:
            raise ValueError("v2 episode lacks canonical object/split")
        frames = episode["frames"]
        if not bool(episode.get("all_frames_object_visible")) or int(
            episode.get("object_evidence_frame_count", -1)
        ) != len(frames):
            raise ValueError("v2 episode is not evidence-complete")
        for frame in frames:
            name = str(frame.get("name") or "")
            if name in names:
                raise ValueError(f"v2 plan reuses RGB across episodes: {name}")
            names.add(name)
            timestamp = str(frame.get("physical_timestamp") or "")
            split_timestamps[split].add(timestamp)
            evidence = frame.get("object_evidence")
            if not isinstance(evidence, dict):
                raise ValueError("v2 frame lacks support-track evidence")
            if int(evidence.get("canonical_object_id", -1)) != object_id:
                raise ValueError("v2 frame evidence object mismatch")
            if (
                evidence.get("retrieval_source") != "object_support_colmap_tracks"
                or evidence.get("bbox_source") != "projected_object_support"
            ):
                raise ValueError("v2 frame is not support-track/projected-support sourced")
        by_object.setdefault(object_id, set()).add(split)
    unpaired = sorted(
        object_id
        for object_id, splits in by_object.items()
        if splits != {"train", "heldout"}
    )
    if unpaired:
        raise ValueError(f"v2 materialization would break object pairs: {unpaired}")
    overlap = sorted(split_timestamps["train"] & split_timestamps["heldout"])
    if overlap:
        raise ValueError(f"v2 physical timestamp leakage: {overlap[:5]}")


def materialize(
    *,
    plan_path: Path,
    output_root: Path,
    verify_hashes: bool,
    selected_episode_ids: set[str] | None,
) -> dict[str, Any]:
    plan_path = plan_path.expanduser().resolve(strict=True)
    plan = json.loads(plan_path.read_text(encoding="utf-8"))
    plan_schema = str(plan.get("schema") or "")
    if plan_schema not in PLAN_SCHEMAS:
        raise ValueError("unsupported tracker subset plan schema")
    image_root = Path(plan["inputs"]["image_root"]).expanduser().resolve(strict=True)
    if not image_root.is_dir():
        raise ValueError("plan image_root is not a directory")
    validation_rows = {
        str(row["name"]): row
        for row in plan.get("selected_image_validation", {}).get("entries", [])
    }
    episodes = list(plan.get("episodes") or [])
    if selected_episode_ids is not None:
        available = {str(episode.get("episode_id")) for episode in episodes}
        missing = sorted(selected_episode_ids.difference(available))
        if missing:
            raise ValueError("unknown episode IDs: " + ", ".join(missing))
        episodes = [
            episode
            for episode in episodes
            if str(episode.get("episode_id")) in selected_episode_ids
        ]
    if not episodes:
        raise ValueError("no episodes selected")
    if plan_schema == PAIRED_PLAN_SCHEMA:
        _validate_paired_plan(episodes)
    output_root = output_root.expanduser().resolve()
    if output_root.exists():
        raise FileExistsError(f"refusing to overwrite materialized episodes: {output_root}")
    output_root.parent.mkdir(parents=True, exist_ok=True)
    staging = Path(
        tempfile.mkdtemp(prefix=f".{output_root.name}.tmp-", dir=output_root.parent)
    )
    episode_rows: list[dict[str, Any]] = []
    try:
        for episode in episodes:
            _validate_episode(episode)
            episode_id = str(episode["episode_id"])
            if Path(episode_id).name != episode_id:
                raise ValueError(f"unsafe episode ID: {episode_id!r}")
            episode_root = staging / episode_id
            frame_root = episode_root / "frames"
            frame_root.mkdir(parents=True)
            mappings: list[dict[str, Any]] = []
            for index, frame in enumerate(episode["frames"]):
                source_name = str(frame["name"])
                source = _safe_source(image_root, source_name)
                provenance = validation_rows.get(source_name)
                if provenance is None:
                    raise ValueError(f"frame lacks selected-image provenance: {source_name}")
                stat = source.stat()
                if int(provenance["bytes"]) != int(stat.st_size):
                    raise ValueError(f"frame size changed since planning: {source_name}")
                expected_hash = provenance.get("sha256")
                if verify_hashes:
                    if not expected_hash:
                        raise ValueError("plan has no selected image hashes to verify")
                    if _sha256(source) != str(expected_hash):
                        raise ValueError(f"frame hash changed since planning: {source_name}")
                materialized_name = _materialized_name(index, source_name)
                os.symlink(str(source), frame_root / materialized_name)
                mappings.append(
                    {
                        "episode_index": index,
                        "materialized_name": materialized_name,
                        "source_name": source_name,
                        "source_path": str(source),
                        "bytes": int(stat.st_size),
                        "sha256": expected_hash,
                        "physical_timestamp": frame["physical_timestamp"],
                        "colmap_image_id": frame["colmap_image_id"],
                        "colmap_camera_id": frame["colmap_camera_id"],
                        "T_world_camera_m": frame["T_world_camera_m"],
                        **(
                            {
                                "object_evidence": frame["object_evidence"],
                                "is_anchor_preference": bool(
                                    frame.get("is_anchor_preference")
                                ),
                                "is_rescue_plan_anchor": bool(
                                    frame.get("is_rescue_plan_anchor")
                                ),
                            }
                            if plan_schema == PAIRED_PLAN_SCHEMA
                            else {}
                        ),
                    }
                )
            frame_names = [row["materialized_name"] for row in mappings]
            (episode_root / "frames.txt").write_text(
                "\n".join(frame_names) + "\n", encoding="utf-8"
            )
            episode_payload = {
                "schema": "farm.materialized-tracker-episode.v1",
                "episode_id": episode_id,
                "camera": episode["camera"],
                "view_family": episode["view_family"],
                "split": episode["split"],
                "frame_count": len(mappings),
                "source_plan": str(plan_path),
                "source_plan_sha256": _sha256(plan_path),
                "source_plan_schema": plan_schema,
                "storage": "absolute_symlink-no-rgb-copy",
                "frames": mappings,
                **(
                    {
                        "pair_id": episode["pair_id"],
                        "anchor_object_id": int(episode["anchor_object_id"]),
                        "all_frames_object_visible": True,
                        "object_evidence_frame_count": len(mappings),
                    }
                    if plan_schema == PAIRED_PLAN_SCHEMA
                    else {}
                ),
            }
            (episode_root / "episode.json").write_text(
                json.dumps(episode_payload, ensure_ascii=False, indent=2) + "\n",
                encoding="utf-8",
            )
            episode_rows.append(
                {
                    "episode_id": episode_id,
                    "camera": episode["camera"],
                    "view_family": episode["view_family"],
                    "split": episode["split"],
                    "frame_count": len(mappings),
                    "relative_path": episode_id,
                    **(
                        {
                            "pair_id": episode["pair_id"],
                            "anchor_object_id": int(episode["anchor_object_id"]),
                            "all_frames_object_visible": True,
                        }
                        if plan_schema == PAIRED_PLAN_SCHEMA
                        else {}
                    ),
                }
            )
        manifest = {
            "schema": OUTPUT_SCHEMA,
            "source_plan": str(plan_path),
            "source_plan_sha256": _sha256(plan_path),
            "source_plan_schema": plan_schema,
            "image_root": str(image_root),
            "hashes_verified": bool(verify_hashes),
            "storage": "symlinks-only-no-rgb-copy",
            "episode_count": len(episode_rows),
            "frame_count": sum(int(row["frame_count"]) for row in episode_rows),
            "episodes": episode_rows,
        }
        (staging / "manifest.json").write_text(
            json.dumps(manifest, ensure_ascii=False, indent=2) + "\n",
            encoding="utf-8",
        )
        os.replace(staging, output_root)
        return manifest
    except BaseException:
        shutil.rmtree(staging, ignore_errors=True)
        raise


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--plan", required=True, type=Path)
    parser.add_argument("--output-root", required=True, type=Path)
    parser.add_argument("--episode-id", action="append")
    parser.add_argument(
        "--no-verify-hashes",
        action="store_true",
        help="Skip image content verification (size is always checked).",
    )
    return parser.parse_args()


def main() -> int:
    args = _parse_args()
    report = materialize(
        plan_path=args.plan,
        output_root=args.output_root,
        verify_hashes=not bool(args.no_verify_hashes),
        selected_episode_ids=set(args.episode_id) if args.episode_id else None,
    )
    print(json.dumps(report, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
