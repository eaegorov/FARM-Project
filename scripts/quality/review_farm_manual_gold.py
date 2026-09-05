#!/usr/bin/env python3
"""Validate independent human annotations, preserve disagreements, freeze gold.

Validation/freezing is an annotation activity. Development evaluation is
separately restricted to train/dev and checks every supplied prediction ID.
"""
from __future__ import annotations

# Allow direct execution from the source checkout.
import sys as _entry_sys
from pathlib import Path as _EntryPath
_entry_root = _EntryPath(__file__).resolve().parents[2]
_entry_sys.path[:0] = [str(_entry_root), str(_entry_root / "src")]

import argparse
from collections import defaultdict
import json
from pathlib import Path

import numpy as np

from farm_runtime.quality_baseline import describe_file, json_digest, sha256_file, write_json
from farm_runtime.quality_benchmark import (
    check_annotation, compare_reviews, mask_metrics, rasterize_regions, timestamp_macro_metrics, verify_packet,
)


def freeze_gold(packet: dict, pairs: list[tuple[dict, dict]], output: Path,
                adjudications: list[dict] | None = None) -> dict:
    if output.exists():
        raise ValueError("gold output must be new")
    errors = verify_packet(packet)
    comparisons = []
    seen_splits = set()
    annotations = []
    visible = defaultdict(set)
    for left, right in pairs:
        split = left.get("split")
        if split in seen_splits:
            errors.append("duplicate_review_split")
        seen_splits.add(split)
        comparison = compare_reviews(packet, left, right)
        comparisons.append({"split": split, **comparison})
        chosen = left
        if comparison["status"] == "REQUIRES_ADJUDICATION":
            candidates = [a for a in (adjudications or []) if a.get("split") == split]
            if len(candidates) != 1:
                errors.append(f"missing_unique_adjudication:{split}")
            else:
                decision = candidates[0]
                required = {json_digest(d) for d in comparison["disagreements"]}
                resolved = decision.get("resolutions", [])
                covered = {r.get("disagreement_sha256") for r in resolved if str(r.get("reason") or "").strip()}
                if (decision.get("schema") != "farm.manual-gold-adjudication.v1"
                        or decision.get("comparison_sha256") != json_digest(comparison)
                        or decision.get("reviewer_kind") != "human"
                        or not str(decision.get("adjudicator_id") or "").strip()
                        or covered != required or len(resolved) != len(required)):
                    errors.append(f"invalid_adjudication_log:{split}")
                final = decision.get("annotation", {})
                errors.extend(check_annotation(packet, final, require_independence=False))
                if final.get("split") != split or final.get("reviewer_id") != decision.get("adjudicator_id"):
                    errors.append(f"adjudicator_identity_or_split_mismatch:{split}")
                chosen = final
        elif comparison["status"] != "AGREED":
            errors.append(f"independent_reviews_not_valid:{split}")
        annotations.append(chosen)
        by_id = {r["observation_id"]: r for r in packet["observations"]}
        for row in chosen.get("observations", []):
            if row.get("state") in ("visible", "partly_occluded") and row.get("observation_id") in by_id:
                obs = by_id[row["observation_id"]]
                visible[(obs["object_id"], split)].add(obs["physical_timestamp"])
    if seen_splits != {"train", "dev", "test"}:
        errors.append("independent_reviews_required_for_all_three_splits")
    for obj in packet["objects"]:
        counts = [len(visible[obj["object_id"], split]) for split in ("train", "dev", "test")]
        if min(counts) < 1 or sum(counts) < 5:
            errors.append(f"insufficient_visible_physical_timestamps:{obj['object_id']}:{counts}")
    if packet.get("selection", {}).get("deficits"):
        errors.append("unresolved_packet_coverage_deficits")
    # Hash raw image bytes, not predictions or rescaled previews.
    if not errors:
        checked = set()
        for obs in packet["observations"]:
            source = obs["source_image"]
            if source["path"] not in checked:
                checked.add(source["path"])
                if sha256_file(Path(source["path"])) != source["sha256"]:
                    errors.append("source_image_changed:" + source["path"])
        if sha256_file(Path(packet["source_ply"]["path"])) != packet["source_ply"]["sha256"]:
            errors.append("source_gaussian_order_or_scene_changed")
    report = {"schema": "farm.manual-gold-freeze.v1", "packet_sha256": json_digest(packet),
              "status": "BLOCKED" if errors else "PILOT_FROZEN", "errors": errors,
              "comparisons": comparisons,
              "f1_full_gate": "BLOCKED", "algorithm_changes_allowed": False,
              "full_gate_remaining": ["20-30-object stratified cohort including previously missed objects",
                                      "independent 3D inspection labels", "historical test-exposure limitations reviewed"],
              "artifacts": []}
    output.mkdir(parents=True)
    # Always retain disagreement reports; never overwrite either human's input.
    for i, (left, right) in enumerate(pairs):
        write_json(output / "reviews" / f"{i}_review_a.json", left)
        write_json(output / "reviews" / f"{i}_review_b.json", right)
    for i, decision in enumerate(adjudications or []):
        write_json(output / "reviews" / f"adjudication_{i}.json", decision)
    if not errors:
        observations = {r["observation_id"]: r for r in packet["observations"]}
        for annotation in annotations:
            split = annotation["split"]
            for row in annotation["observations"]:
                obs = observations[row["observation_id"]]
                masks = rasterize_regions(row["regions"], tuple(obs["shape_hw"]))
                name = json_digest(row["observation_id"])[:24] + ".npz"
                path = output / split / name
                path.parent.mkdir(exist_ok=True)
                np.savez_compressed(path, **masks)
                report["artifacts"].append({"observation_id": row["observation_id"], "split": split,
                                             "state": row["state"], "path": str(path.relative_to(output)),
                                             "sha256": sha256_file(path)})
    report["content_sha256"] = json_digest(report)
    write_json(output / "freeze_manifest.json", report)
    return report


def evaluate_development(packet: dict, frozen_root: Path, predictions: dict, split: str) -> dict:
    if split not in ("train", "dev"):
        raise ValueError("test is sealed: this command only evaluates train/dev")
    errors = verify_packet(packet)
    if errors:
        raise ValueError(errors)
    manifest = json.loads((frozen_root / "freeze_manifest.json").read_text())
    if manifest.get("status") != "PILOT_FROZEN" or manifest.get("packet_sha256") != json_digest(packet):
        raise ValueError("frozen reviewed gold matching this packet is required")
    unsigned = {k: v for k, v in manifest.items() if k != "content_sha256"}
    if json_digest(unsigned) != manifest.get("content_sha256"):
        raise ValueError("gold manifest changed")
    if predictions.get("packet_sha256") != json_digest(packet):
        raise ValueError("prediction packet hash mismatch")
    obs = {r["observation_id"]: r for r in packet["observations"] if r["split"] == split}
    supplied = predictions.get("observations", [])
    pred_rows = {r["observation_id"]: r for r in supplied}
    if len(pred_rows) != len(supplied) or set(pred_rows) != set(obs):
        raise ValueError("predictions must cover exactly the requested fold; no test IDs or dropped difficult views")
    gold = {r["observation_id"]: r for r in manifest["artifacts"] if r["split"] == split}
    if set(gold) != set(obs):
        raise ValueError("gold fold incomplete")
    metrics = []
    for key, row in obs.items():
        artifact = gold[key]
        path = frozen_root / artifact["path"]
        if not path.resolve().is_relative_to(frozen_root.resolve()) or sha256_file(path) != artifact["sha256"]:
            raise ValueError("gold mask path/hash mismatch")
        pred = pred_rows[key]
        if sha256_file(Path(pred["path"])) != pred["sha256"]:
            raise ValueError("prediction hash mismatch")
        with np.load(path, allow_pickle=False) as f:
            layers = {name: f[name] for name in f.files}
        prediction = np.load(pred["path"], allow_pickle=False)
        values = mask_metrics(prediction, layers)
        metrics.append({"object_id": row["object_id"], "image_name": row["image_name"],
                        "physical_timestamp": row["physical_timestamp"], "direction": row["direction"],
                        "state": artifact["state"], **values})
    return {"schema": "farm.manual-gold-dev-evaluation.v1", "split": split,
            "packet_sha256": json_digest(packet), "gold_sha256": manifest["content_sha256"],
            "prediction_manifest_sha256": json_digest(predictions), "observations": metrics,
            **timestamp_macro_metrics(metrics), "test_opened": False,
            "limitation": "Conditional precision on annotated negatives; unknown pixels and uncertain boundary are reported separately."}


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--packet", type=Path, required=True)
    p.add_argument("--output", type=Path, required=True)
    sub = p.add_subparsers(dest="command", required=True)
    validate = sub.add_parser("validate")
    validate.add_argument("--annotation", type=Path, required=True)
    compare = sub.add_parser("compare")
    compare.add_argument("--review-a", type=Path, required=True)
    compare.add_argument("--review-b", type=Path, required=True)
    freeze = sub.add_parser("freeze")
    freeze.add_argument("--review-a", nargs=3, type=Path, required=True)
    freeze.add_argument("--review-b", nargs=3, type=Path, required=True)
    freeze.add_argument("--adjudication", nargs="+", type=Path)
    evaluate = sub.add_parser("evaluate-dev")
    evaluate.add_argument("--gold", type=Path, required=True)
    evaluate.add_argument("--predictions", type=Path, required=True)
    evaluate.add_argument("--split", choices=("train", "dev"), default="dev")
    args = p.parse_args()
    if args.output.exists():
        raise ValueError("output must be new; preserve prior evidence")
    packet = json.loads(args.packet.read_text())
    read = lambda path: json.loads(path.read_text())
    if args.command == "validate":
        errors = check_annotation(packet, read(args.annotation))
        result = {"status": "BLOCKED" if errors else "VALID", "errors": errors}
    elif args.command == "compare":
        result = compare_reviews(packet, read(args.review_a), read(args.review_b))
    elif args.command == "freeze":
        result = freeze_gold(packet, [(read(a), read(b)) for a, b in zip(args.review_a, args.review_b)], args.output,
                             [read(p) for p in args.adjudication] if args.adjudication else None)
    else:
        result = evaluate_development(packet, args.gold, read(args.predictions), args.split)
    if args.command != "freeze":
        write_json(args.output, result)
    print(json.dumps({k: v for k, v in result.items() if k in ("status", "errors", "split")}, ensure_ascii=False))
    return 1 if result.get("status") in ("BLOCKED", "REQUIRES_ADJUDICATION") else 0


if __name__ == "__main__":
    raise SystemExit(main())
