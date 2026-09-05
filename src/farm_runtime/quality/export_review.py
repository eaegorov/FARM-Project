"""Export reviewed semantics, native masks and metric surface-envelope OBBs."""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

import numpy as np

from farm_runtime.instance_bank import filter_instance_bank
from farm_runtime.native_object_geometry import fit_native_obb
from farm_runtime.quality_baseline import describe_file, write_json
from tools.farm_shaper_bridge.common import load_run, open_graphdeco_ply


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("run", "ply", "ablation", "scope", "output"):
        parser.add_argument("--" + name, type=Path, required=True)
    args = parser.parse_args(argv)
    if args.output.exists():
        raise ValueError("output must be new")
    run = load_run(args.run, allow_legacy=True)
    scope = json.loads(args.scope.read_text())
    review = {int(r["object_id"]): r for r in scope["objects"]}
    source = open_graphdeco_ply(args.ply)
    report = json.loads((args.ablation / "obb_mask_surface/report.json").read_text())
    qc = {int(r["object_id"]): r for r in report["heldout_qc"]}
    allowed = sorted(i for i in review if qc.get(i, {}).get("status") == "verified")
    with np.load(
        args.ablation / "obb_mask_surface/proposal_bank.npz", allow_pickle=False
    ) as raw:
        bank = filter_instance_bank(raw, allowed, source_count=source.count)
    args.output.mkdir(parents=True)
    bank_path = args.output / "object_masks.npz"
    np.savez_compressed(bank_path, **bank)
    old = {o.object_id: o for o in run.objects}
    catalog = []
    for pos, oid in enumerate(bank["object_ids"]):
        oid = int(oid)
        start, end = int(bank["indptr"][pos]), int(bank["indptr"][pos + 1])
        indices = bank["indices"][start:end]
        table = source.data[indices]
        points = (
            np.column_stack([table[k] for k in ("x", "y", "z")])
            * run.meters_per_scene_unit
        )
        opacity = 1 / (1 + np.exp(-np.clip(table["opacity"].astype(float), -30, 30)))
        weights = opacity * bank["confidence"][start:end]
        obb = fit_native_obb(points, weights, old[oid].rotation)
        catalog.append(
            {
                **review[oid],
                "native_gaussians": len(indices),
                "mask_bank_row": pos,
                "obb": obb,
                "mask_quality_reference": "legacy automatic masks, corrected cameras",
                "mask_qc": {
                    k: qc[oid]["summaries"][k] for k in ("iou", "precision", "recall")
                },
                "semantics_source": "explicit user scope and assistant visual review",
                "automatic_semantic_accuracy_validated": False,
                "whole_hidden_geometry_claimed": False,
            }
        )
    write_json(
        args.output / "catalog.json",
        {
            "schema": "farm.reviewed-quality-catalog.v1",
            "scene_id": run.scene_id,
            "objects": catalog,
            "source_ply": describe_file(args.ply),
            "source_gaussian_count": source.count,
            "meters_per_scene_unit": run.meters_per_scene_unit,
            "native_mask_bank": {**describe_file(bank_path), "path": bank_path.name},
            "scope_review": {
                **describe_file(args.scope),
                "path": os.path.relpath(args.scope.resolve(), args.output.resolve()),
            },
            "ablation_comparison": {
                **describe_file(args.ablation / "comparison.json"),
                "path": os.path.relpath(
                    (args.ablation / "comparison.json").resolve(), args.output.resolve()
                ),
            },
            "candidate_policy": "obb_mask_surface; engineering proposal, not a universal default",
            "missing_mask_object_ids": sorted(set(review) - set(allowed)),
            "release_eligible": False,
            "status": "ENGINEERING_REVIEW",
            "source_order_preserved": True,
            "physical_size_note": "Metric estimates of observed reconstruction support; hidden surfaces and material thickness remain uncertain.",
        },
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
