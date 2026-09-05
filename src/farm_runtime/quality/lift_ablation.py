"""Replay one frozen FARM mask/split input with OBB and surface candidates."""

from __future__ import annotations

import argparse
import copy
from dataclasses import replace
import json
from pathlib import Path
import time

import numpy as np

from farm_runtime.quality_baseline import describe_file, write_json
from tools.farm_shaper_bridge import gaussian_lift as lift
from tools.farm_shaper_bridge.common import (
    build_verified_csr,
    load_run,
    open_graphdeco_ply,
)
from tools.farm_shaper_bridge.lift_refinement import (
    apply_provisional_geometry_gate,
    refine_connected_claims,
)


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__)
    for key in ("run", "ply", "split", "config", "packet", "output"):
        p.add_argument("--" + key, type=Path, required=True)
    p.add_argument("--camera-refinement", type=Path)
    args = p.parse_args(argv)
    if args.output.exists():
        raise ValueError("output must be new")
    config, _ = lift.load_config(args.config)
    split = json.loads(args.split.read_text())
    packet = json.loads(args.packet.read_text())
    run = load_run(args.run, allow_legacy=True)
    ids = {int(r["object_id"]) for r in split["objects"]}
    run = replace(run, objects=tuple(o for o in run.objects if o.object_id in ids))
    if ids != {o.object_id for o in run.objects}:
        raise ValueError("split objects absent from run")
    consumed = set(split["build_timestamps"]) | set(split["heldout_timestamps"])
    if set(split["build_timestamps"]) & set(split["heldout_timestamps"]):
        raise ValueError("overlapping frozen folds")
    for obj in run.objects:
        for observation in obj.observations:
            frame = run.frame(observation.image_id)
            ts = Path(frame.source_image).stem.split("_")[1]
            if frame.physical_timestamp in consumed and packet[
                "split_by_timestamp"
            ].get(ts) not in ("train", "dev"):
                raise ValueError("reserved test or unknown timestamp in input")
    if args.camera_refinement is not None:
        from farm_runtime.quality.camera_refinement import apply_camera_refinement

        run = apply_camera_refinement(run, args.camera_refinement)
    args.output.mkdir(parents=True)
    table = open_graphdeco_ply(args.ply)
    if describe_file(args.ply)["sha256"] != packet["source_ply"]["sha256"]:
        raise ValueError("source PLY does not match packet")
    gaussians = lift.load_gaussians(table, run.meters_per_scene_unit)
    alignment = lift.alignment_guard(gaussians, run, config)
    import torch

    reports, banks = [], {}
    for domain in ("obb", "obb_mask_surface"):
        policy = copy.deepcopy(config)
        policy["candidate"].update(
            domain=domain, surface_pixel_stride=4, surface_voxel_m=0.01
        )
        lift._require_sections(policy)
        dest = args.output / domain
        dest.mkdir()
        write_json(dest / "config.json", policy)
        started = time.monotonic()
        evidence, candidates = lift.build_candidates(run, gaussians, split, policy)
        candidate_seconds = time.monotonic() - started
        torch.cuda.reset_peak_memory_stats()
        build_rows, timing = lift.accumulate_build_evidence(
            gaussians, run, split, evidence, policy
        )
        owner, confidence, support, object_rows, conflicts, blocked = lift.make_claims(
            evidence, table.count, policy
        )
        owner, confidence, support, refinement = refine_connected_claims(
            gaussians.means_m,
            gaussians.radius_m,
            evidence,
            owner,
            confidence,
            support,
            blocked,
            policy,
        )
        for row in object_rows:
            row["refinement"] = next(
                r for r in refinement["objects"] if r["object_id"] == row["object_id"]
            )
        owner, confidence, support, geometry = apply_provisional_geometry_gate(
            owner, confidence, support, object_rows, table.count, policy
        )
        # Save immutable proposals before opening old heldout references. These
        # are engineering comparisons; the reserved V12 test remains closed.
        bank = build_verified_csr(owner, confidence, support)
        bank_path = dest / "proposal_bank.npz"
        np.savez_compressed(bank_path, **bank)
        qc, accepted, qc_timing, qc_audit = lift.heldout_qc(
            gaussians, run, split, owner, policy
        )
        report = {
            "schema": "farm.lift-candidate-ablation.v1",
            "domain": domain,
            "release_eligible": False,
            "candidate_seconds": candidate_seconds,
            "total_seconds": time.monotonic() - started,
            "timing": {**timing, **qc_timing},
            "peak_allocated_mib": torch.cuda.max_memory_allocated() / 2**20,
            "candidates": candidates,
            "objects": object_rows,
            "conflicts": conflicts,
            "refinement": refinement,
            "geometry_gate": geometry,
            "heldout_qc": qc,
            "legacy_reference_pass_ids": sorted(accepted),
            "proposal_bank": describe_file(bank_path),
            "source_gaussian_count": table.count,
            "source_order_preserved": True,
            "manual_gold_used": False,
            "reserved_test_opened": False,
        }
        write_json(dest / "report.json", report)
        write_json(
            dest / "evidence_sources.json", {"build": build_rows, "heldout": qc_audit}
        )
        reports.append(report)
        banks[domain] = {
            int(oid): bank["indices"][bank["indptr"][i] : bank["indptr"][i + 1]]
            for i, oid in enumerate(bank["object_ids"])
        }
        print(
            json.dumps(
                {
                    "variant": domain,
                    "seconds": report["total_seconds"],
                    "counts": {oid: len(v) for oid, v in banks[domain].items()},
                }
            ),
            flush=True,
        )
        del evidence, owner, confidence, support
    changes = []
    for oid in sorted(ids):
        before, after = (
            banks[d].get(oid, np.empty(0, dtype=np.int64))
            for d in ("obb", "obb_mask_surface")
        )
        changes.append(
            {
                "object_id": oid,
                "before": len(before),
                "after": len(after),
                "added": len(np.setdiff1d(after, before)),
                "removed": len(np.setdiff1d(before, after)),
            }
        )
    write_json(
        args.output / "comparison.json",
        {
            "schema": "farm.lift-ablation-comparison.v1",
            "changes": changes,
            "alignment": alignment,
            "split": describe_file(args.split),
            "config": describe_file(args.config),
            "packet": describe_file(args.packet),
            "source_ply": describe_file(args.ply),
            "camera_refinement": (
                describe_file(args.camera_refinement)
                if args.camera_refinement
                else None
            ),
            "legacy_mask_reference_only": True,
            "reserved_test_opened": False,
            "release_eligible": False,
        },
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
