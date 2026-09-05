#!/usr/bin/env python3
"""Render a readable, source-backed tracker A/B contact sheet (PNG only)."""

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
import time
from pathlib import Path
from typing import Any

import numpy as np
from PIL import Image, ImageDraw, ImageFont


EPISODE_SCHEMA = "farm.materialized-tracker-episode.v1"
REPORT_SCHEMA = "farm.tracker-ab-visualization.v1"


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _sample_indices(frame_count: int, sample_count: int) -> list[int]:
    if frame_count < 1 or sample_count < 1:
        raise ValueError("frame and sample counts must be positive")
    count = min(int(frame_count), int(sample_count))
    return sorted(set(int(round(value)) for value in np.linspace(0, frame_count - 1, count)))


def _font(size: int, *, bold: bool = False):
    name = "DejaVuSans-Bold.ttf" if bold else "DejaVuSans.ttf"
    candidates = [
        Path("/usr/share/fonts/truetype/dejavu") / name,
        Path("/usr/share/fonts/dejavu") / name,
    ]
    for candidate in candidates:
        if candidate.is_file():
            return ImageFont.truetype(str(candidate), size=size)
    return ImageFont.load_default()


def _identity_color(method_name: str, identity: int) -> np.ndarray:
    digest = hashlib.sha256(f"{method_name}\0{identity}".encode("utf-8")).digest()
    return np.asarray([72 + digest[0] % 176, 72 + digest[1] % 176, 72 + digest[2] % 176], dtype=np.uint8)


def _metadata_prompts(path: Path | None) -> dict[int, str]:
    if path is None:
        return {}
    payload = json.loads(path.read_text(encoding="utf-8"))
    identities = payload.get("output", {}).get("identities", [])
    return {
        int(row["local_id"]): str(row.get("prompt") or "")
        for row in identities
        if row.get("local_id") is not None
    }


def _overlay(
    image: Image.Image,
    mask: Image.Image,
    *,
    method_name: str,
    prompts: dict[int, str],
) -> Image.Image:
    rgb = np.asarray(image.convert("RGB"), dtype=np.float32).copy()
    ids = np.asarray(mask, dtype=np.uint16)
    if ids.shape != rgb.shape[:2]:
        raise ValueError("mask and RGB shapes differ")
    output = rgb.copy()
    draw_labels: list[tuple[int, int, str, tuple[int, int, int]]] = []
    for identity in np.unique(ids):
        identity = int(identity)
        if identity == 0:
            continue
        selected = ids == identity
        area = int(selected.sum())
        if area == 0:
            continue
        color = _identity_color(method_name, identity)
        output[selected] = output[selected] * 0.34 + color.astype(np.float32) * 0.66
        if area >= max(256, int(ids.size * 0.0025)):
            ys, xs = np.nonzero(selected)
            label = f"#{identity}"
            prompt = prompts.get(identity)
            if prompt:
                label += f" {prompt}"
            draw_labels.append((int(xs.min()), int(ys.min()), label, tuple(int(v) for v in color)))
    rendered = Image.fromarray(np.clip(output, 0, 255).astype(np.uint8))
    draw = ImageDraw.Draw(rendered)
    font = _font(20, bold=True)
    for x, y, label, color in sorted(draw_labels, key=lambda value: (value[1], value[0]))[:18]:
        box = draw.textbbox((x, y), label, font=font, stroke_width=2)
        draw.rectangle(box, fill=(8, 12, 18, 220))
        draw.text((x, y), label, font=font, fill=color, stroke_width=1, stroke_fill=(0, 0, 0))
    return rendered


def _fit_panel(image: Image.Image, width: int, height: int) -> Image.Image:
    panel = Image.new("RGB", (width, height), (5, 9, 15))
    resized = image.copy()
    resized.thumbnail((width, height), Image.Resampling.LANCZOS)
    panel.paste(resized, ((width - resized.width) // 2, (height - resized.height) // 2))
    return panel


def render(
    *,
    episode_json: Path,
    methods: list[tuple[str, Path, Path | None]],
    output_png: Path,
    output_json: Path,
    sample_count: int,
    title: str,
) -> dict[str, Any]:
    episode_json = episode_json.expanduser().resolve(strict=True)
    episode = json.loads(episode_json.read_text(encoding="utf-8"))
    if episode.get("schema") != EPISODE_SCHEMA:
        raise ValueError("unsupported materialized episode")
    frames = list(episode.get("frames") or [])
    indices = _sample_indices(len(frames), sample_count)
    episode_root = episode_json.parent
    resolved_methods = []
    for name, mask_dir, metadata in methods:
        mask_dir = mask_dir.expanduser().resolve(strict=True)
        metadata = metadata.expanduser().resolve(strict=True) if metadata else None
        resolved_methods.append((name, mask_dir, metadata, _metadata_prompts(metadata)))

    panel_width, panel_height = 480, 320
    title_height, row_header, column_header, outer = 58, 56, 46, 20
    rows = 1 + len(resolved_methods)
    width = outer * 2 + panel_width * len(indices)
    height = outer * 2 + title_height + column_header + rows * (row_header + panel_height)
    sheet = Image.new("RGB", (width, height), (7, 11, 18))
    draw = ImageDraw.Draw(sheet)
    title_font = _font(28, bold=True)
    label_font = _font(21, bold=True)
    small_font = _font(16)
    draw.text((outer, outer), title, font=title_font, fill=(245, 248, 252))

    selected_rows = []
    for column, index in enumerate(indices):
        frame = frames[index]
        materialized_name = str(frame["materialized_name"])
        rgb_path = (episode_root / "frames" / materialized_name).resolve(strict=True)
        with Image.open(rgb_path) as handle:
            rgb = handle.convert("RGB")
        x = outer + column * panel_width
        draw.text(
            (x + 8, outer + title_height + 10),
            f"{index:02d} · {frame['source_name']}",
            font=small_font,
            fill=(170, 185, 205),
        )
        y = outer + title_height + column_header + row_header
        sheet.paste(_fit_panel(rgb, panel_width, panel_height), (x, y))
        method_rows = []
        for method_index, (name, mask_dir, metadata, prompts) in enumerate(resolved_methods, 1):
            mask_path = mask_dir / f"{Path(materialized_name).stem}.png"
            if not mask_path.is_file():
                raise FileNotFoundError(mask_path)
            with Image.open(mask_path) as handle:
                mask = handle.copy()
            visualization = _overlay(rgb, mask, method_name=name, prompts=prompts)
            y = outer + title_height + column_header + method_index * (row_header + panel_height) + row_header
            sheet.paste(_fit_panel(visualization, panel_width, panel_height), (x, y))
            method_rows.append(
                {
                    "name": name,
                    "mask_path": str(mask_path.resolve()),
                    "mask_sha256": _sha256(mask_path),
                    "metadata": str(metadata) if metadata else None,
                }
            )
        selected_rows.append(
            {
                "episode_index": index,
                "materialized_name": materialized_name,
                "source_name": frame["source_name"],
                "rgb_path": str(rgb_path),
                "methods": method_rows,
            }
        )

    row_names = ["INPUT RGB"] + [name for name, *_ in resolved_methods]
    for row_index, name in enumerate(row_names):
        y = outer + title_height + column_header + row_index * (row_header + panel_height)
        draw.rectangle((outer, y, width - outer, y + row_header), fill=(15, 23, 35))
        draw.text((outer + 10, y + 13), name, font=label_font, fill=(244, 196, 84))

    output_png = output_png.expanduser().resolve()
    output_json = output_json.expanduser().resolve()
    output_png.parent.mkdir(parents=True, exist_ok=True)
    temporary_png = output_png.with_name(f".{output_png.name}.tmp-{os.getpid()}.png")
    sheet.save(temporary_png, format="PNG", optimize=True)
    os.replace(temporary_png, output_png)
    report = {
        "schema": REPORT_SCHEMA,
        "created_unix_s": time.time(),
        "episode_json": str(episode_json),
        "episode_json_sha256": _sha256(episode_json),
        "episode_id": episode["episode_id"],
        "sample_indices": indices,
        "methods": [name for name, *_ in resolved_methods],
        "output_png": str(output_png),
        "output_png_sha256": _sha256(output_png),
        "selected_frames": selected_rows,
    }
    temporary_json = output_json.with_name(f".{output_json.name}.tmp-{os.getpid()}")
    temporary_json.write_text(
        json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    os.replace(temporary_json, output_json)
    return report


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--episode-json", required=True, type=Path)
    parser.add_argument(
        "--method",
        action="append",
        nargs=3,
        metavar=("LABEL", "MASK_DIR", "METADATA_JSON_OR_DASH"),
        required=True,
    )
    parser.add_argument("--output-png", required=True, type=Path)
    parser.add_argument("--output-json", required=True, type=Path)
    parser.add_argument("--sample-count", type=int, default=5)
    parser.add_argument("--title", default="FARM tracker A/B")
    args = parser.parse_args()
    methods = [
        (label, Path(mask_dir), None if metadata == "-" else Path(metadata))
        for label, mask_dir, metadata in args.method
    ]
    report = render(
        episode_json=args.episode_json,
        methods=methods,
        output_png=args.output_png,
        output_json=args.output_json,
        sample_count=args.sample_count,
        title=args.title,
    )
    print(json.dumps({"output": report["output_png"], "samples": report["sample_indices"]}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
