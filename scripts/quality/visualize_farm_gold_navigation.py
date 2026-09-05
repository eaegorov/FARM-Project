#!/usr/bin/env python3
"""Render train/dev original RGB navigation sheets; test pixels remain sealed."""
from __future__ import annotations

# Allow direct execution from the source checkout.
import sys as _entry_sys
from pathlib import Path as _EntryPath
_entry_root = _EntryPath(__file__).resolve().parents[2]
_entry_sys.path[:0] = [str(_entry_root), str(_entry_root / "src")]

import argparse
import json
from pathlib import Path
import resource
import time

from PIL import Image, ImageDraw, ImageFont

from farm_runtime.quality_baseline import describe_file, json_digest, sha256_file, write_json
from farm_runtime.quality_benchmark import DIRECTIONS, verify_packet


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--packet", required=True, type=Path)
    p.add_argument("--navigation", required=True, type=Path)
    p.add_argument("--output", required=True, type=Path)
    args = p.parse_args()
    if args.output.exists():
        raise ValueError("visual output must be new")
    packet = json.loads(args.packet.read_text())
    errors = verify_packet(packet)
    if errors:
        raise ValueError(errors)
    navigation = json.loads(args.navigation.read_text())["views"]
    allowed = {r["image_name"]: r for r in packet["observations"] if r["split"] in ("train", "dev")}
    by_key = {(r["object_id"], r["physical_timestamp"], r["camera"], r["direction"]): r
              for r in packet["observations"] if r["split"] in ("train", "dev")}
    started = time.monotonic()
    args.output.mkdir(parents=True)
    font = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf", 17)
    title_font = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf", 26)
    decoded = {}
    source_hashes = {}
    artifacts = []

    def rgb(name):
        if name not in allowed:
            raise ValueError("visualization attempted to decode non-development image")
        row = allowed[name]
        path = Path(row["source_image"]["path"])
        if name not in source_hashes:
            source_hashes[name] = sha256_file(path)
            if source_hashes[name] != row["source_image"]["sha256"]:
                raise ValueError("source RGB changed")
        with Image.open(path) as im:
            if [im.height, im.width] != row["shape_hw"]:
                raise ValueError("source image dimensions changed")
            decoded[name] = im.width * im.height
            return im.convert("RGB")

    def paste_fit(canvas, im, box):
        x, y, width, height = box
        im = im.copy()
        im.thumbnail((width, height), Image.Resampling.LANCZOS)
        canvas.paste(im, (x + (width-im.width)//2, y + (height-im.height)//2))

    for obj in packet["objects"]:
        oid = obj["object_id"]
        views = [n for n in navigation if n["object_id"] == oid and n["split"] in ("train", "dev")]
        canvas = Image.new("RGB", (1260, 100 + 380 * len(views)), "#111827")
        draw = ImageDraw.Draw(canvas)
        draw.text((20, 15), f"FARM V12 / object #{oid} / original RGB", font=title_font, fill="white")
        draw.text((20, 55), "Navigation anchor only. Human must verify identity, full extent, attachments and visibility.", font=font, fill="#d1d5db")
        for i, n in enumerate(views):
            im = rgb(n["name"])
            y = 100 + i*380
            draw.text((20, y), f"{n['split'].upper()} / {n['name']}", font=font, fill="#67e8f9")
            paste_fit(canvas, im, (10, y+30, 560, 330))
            u, v = n["center_px"]
            side = max(256, min(1800, n["diameter_px"] * 1.3))
            left = max(0, min(im.width-side, u-side/2))
            top = max(0, min(im.height-side, v-side/2))
            crop = im.crop((int(left), int(top), int(left+side), int(top+side)))
            paste_fit(canvas, crop, (590, y+30, 650, 330))
            draw.text((850, y), "Context crop", font=font, fill="#d1d5db")
            cameras = sorted({r["camera"] for r in packet["observations"]
                              if r["physical_timestamp"] == n["physical_timestamp"] and r["object_id"] == oid})
            sheet = Image.new("RGB", (1600, 65 + 345*len(cameras)), "#111827")
            sd = ImageDraw.Draw(sheet)
            sd.text((12, 10), f"#{oid} / {n['split']} / timestamp {n['physical_timestamp']} / all directions", font=title_font, fill="white")
            for j, cam in enumerate(cameras):
                for k, direction in enumerate(DIRECTIONS):
                    row = by_key.get((oid, n["physical_timestamp"], cam, direction))
                    sd.text((10+320*k, 65+345*j), f"{cam} / {direction}", font=font, fill="#67e8f9")
                    if row:
                        paste_fit(sheet, rgb(row["image_name"]), (320*k+5, 95+345*j, 310, 310))
            dest = args.output / f"object_{oid:06d}_{n['split']}_{n['physical_timestamp']}_directions.jpg"
            sheet.save(dest, quality=88)
            artifacts.append(describe_file(dest))
        dest = args.output / f"object_{oid:06d}_navigation.jpg"
        canvas.save(dest, quality=90)
        artifacts.append(describe_file(dest))
    report = {"schema": "farm.gold-navigation-visuals.v1", "packet_sha256": json_digest(packet),
              "is_ground_truth": False, "test_pixels_decoded": 0,
              "source_hashes": source_hashes, "unique_decoded_pixels": sum(decoded.values()),
              "unique_decoded_images": len(decoded), "artifacts": artifacts,
              "wall_seconds": time.monotonic()-started,
              "peak_rss_kib": resource.getrusage(resource.RUSAGE_SELF).ru_maxrss, "gpu_used": False}
    write_json(args.output / "manifest.json", report)
    print(json.dumps({k: v for k, v in report.items() if k not in ("source_hashes", "artifacts")}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
