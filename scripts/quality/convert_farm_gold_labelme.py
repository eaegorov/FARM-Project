#!/usr/bin/env python3
"""Convert F1 templates to/from ordinary LabelMe polygon files, without a GUI dependency."""
from __future__ import annotations

# Allow direct execution from the source checkout.
import sys as _entry_sys
from pathlib import Path as _EntryPath
_entry_root = _EntryPath(__file__).resolve().parents[2]
_entry_sys.path[:0] = [str(_entry_root), str(_entry_root / "src")]

import argparse
import json
from pathlib import Path

from farm_runtime.quality_baseline import json_digest, sha256_file, write_json
from farm_runtime.quality_benchmark import LAYERS, annotation_template, verify_packet


def export_templates(packet: dict, split: str, output: Path) -> None:
    if output.exists():
        raise ValueError("output must be new")
    if verify_packet(packet) or split not in ("train", "dev", "test"):
        raise ValueError("invalid packet or split")
    output.mkdir(parents=True)
    for row in packet["observations"]:
        if row["split"] != split:
            continue
        stem = json_digest(row["observation_id"])[:24]
        write_json(output / (stem + ".json"), {
            "version": "5.0.0", "flags": {"reviewed": False, "identity_confirmed": False, "visible": False,
                                            "partly_occluded": False, "fully_occluded": False,
                                            "out_of_view": False, "absent": False, "ambiguous": False},
            "shapes": [], "imagePath": row["source_image"]["path"], "imageData": None,
            "imageHeight": row["shape_hw"][0], "imageWidth": row["shape_hw"][1],
        })
    write_json(output / "_farm_binding.json", {
        "schema": "farm.labelme-binding.v1", "packet_sha256": json_digest(packet), "split": split,
        "labels": list(LAYERS), "gold": False,
        "files": {json_digest(r["observation_id"])[:24] + ".json": r["observation_id"]
                  for r in packet["observations"] if r["split"] == split}})


def import_annotations(packet: dict, annotation: dict, folder: Path) -> dict:
    if annotation.get("packet_sha256") != json_digest(packet):
        raise ValueError("annotation packet hash mismatch")
    binding = json.loads((folder / "_farm_binding.json").read_text())
    if binding.get("packet_sha256") != json_digest(packet) or binding.get("split") != annotation["split"]:
        raise ValueError("LabelMe binding packet/split mismatch")
    expected = {r["observation_id"]: r for r in packet["observations"] if r["split"] == annotation["split"]}
    files = binding["files"]
    if set(files.values()) != set(expected) or len(files) != len(expected):
        raise ValueError("LabelMe binding must cover exactly the fold")
    by_id = {}
    for name, key in files.items():
        if Path(name).name != name:
            raise ValueError("invalid LabelMe file name")
        row = expected[key]
        label = json.loads((folder / name).read_text())
        if (Path(label["imagePath"]).resolve() != Path(row["source_image"]["path"]).resolve()
                or [label["imageHeight"], label["imageWidth"]] != row["shape_hw"]
                or sha256_file(Path(label["imagePath"])) != row["source_image"]["sha256"]):
            raise ValueError("LabelMe image identity/dimensions/hash changed")
        flags = label.get("flags", {})
        states = [s for s in ("visible", "partly_occluded", "fully_occluded", "out_of_view", "absent", "ambiguous") if flags.get(s) is True]
        if flags.get("reviewed") is not True or flags.get("identity_confirmed") is not True or len(states) != 1:
            raise ValueError("each image needs reviewed, identity_confirmed, and exactly one visibility state")
        regions = []
        for shape in label.get("shapes", []):
            if shape.get("shape_type") != "polygon" or shape.get("label") not in LAYERS:
                raise ValueError("use closed polygons with one of the declared FARM layers")
            regions.append({"layer": shape["label"], "points": shape["points"]})
        by_id[key] = {"observation_id": key, "object_identity": str(row["object_id"]),
                      "state": states[0], "reviewed": True, "regions": regions,
                      "notes": str(label.get("description") or "")}
    result = dict(annotation)
    result["observations"] = [by_id[r["observation_id"]] for r in annotation["observations"]]
    # Reviewer identity, independence and scope remain explicit user-entered
    # metadata in the base annotation; importing polygons cannot approve them.
    result["labelme_binding_sha256"] = sha256_file(folder / "_farm_binding.json")
    return result


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--packet", required=True, type=Path)
    p.add_argument("--output", required=True, type=Path)
    sub = p.add_subparsers(dest="command", required=True)
    export = sub.add_parser("export")
    export.add_argument("--split", choices=("train", "dev", "test"), required=True)
    ingest = sub.add_parser("import")
    ingest.add_argument("--folder", type=Path, required=True)
    ingest.add_argument("--annotation", type=Path, required=True)
    args = p.parse_args()
    if args.output.exists():
        raise ValueError("output must be new")
    packet = json.loads(args.packet.read_text())
    if args.command == "export":
        export_templates(packet, args.split, args.output)
    else:
        result = import_annotations(packet, json.loads(args.annotation.read_text()), args.folder)
        write_json(args.output, result)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
