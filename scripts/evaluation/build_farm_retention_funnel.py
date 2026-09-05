#!/usr/bin/env python3
"""Build a source-backed FARM retention and result-quality audit.

Structural QA answers whether artifacts are internally valid.  This audit is
deliberately separate: it exposes where detections/tracks are lost, whether a
diagnostic surface check would have removed geometry-valid objects, semantic
evidence strength, stale mask evidence, and unresolved strong duplicate pairs.
Low recall produces ``WARN`` rather than pretending that it is a structural
failure or a successful exhaustive inventory.
"""

from __future__ import annotations

# Allow direct execution from the source checkout.
import sys as _entry_sys
from pathlib import Path as _EntryPath
_entry_root = _EntryPath(__file__).resolve().parents[2]
_entry_sys.path[:0] = [str(_entry_root), str(_entry_root / "src")]

import argparse
import json
import math
import os
import re
import textwrap
from collections import Counter
from pathlib import Path
from typing import Any, Mapping

import matplotlib.pyplot as plt
import numpy as np
import torch


def _json(path: Path, *, required: bool = True) -> dict:
    if not path.is_file():
        if required:
            raise FileNotFoundError(path)
        return {}
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise TypeError(f"Expected JSON object: {path}")
    return value


def _state(path: Path) -> dict:
    payload = torch.load(path, map_location="cpu", weights_only=False)
    state = payload.get("state", payload) if isinstance(payload, dict) else None
    if not isinstance(state, dict):
        raise TypeError(f"Invalid FARM state: {path}")
    return state


def _trace_counts(path: Path) -> tuple[int, int, int]:
    raw = filtered = frames = 0
    if not path.is_file():
        return raw, filtered, frames
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            try:
                row = json.loads(line)
            except json.JSONDecodeError:
                continue
            if row.get("event") != "frame":
                continue
            frames += int(row.get("batch_size") or 0)
            raw += int((row.get("segmentation") or {}).get("n_raw") or 0)
            filtered += int((row.get("segmentation_post_filter") or {}).get("n_raw") or 0)
    return raw, filtered, frames


def _tensor_count(state: Mapping[str, Any], key: str) -> int:
    value = state.get(key)
    return int(value.numel()) if isinstance(value, torch.Tensor) else len(value or [])


def _evidence_counts(state: Mapping[str, Any]) -> dict[str, int]:
    counts = state.get("count")
    if not isinstance(counts, torch.Tensor):
        return {"at_least_3": 0, "at_least_5": 0}
    values = counts.detach().cpu().numpy().reshape(-1)
    return {
        "at_least_3": int(np.sum(values >= 3)),
        "at_least_5": int(np.sum(values >= 5)),
    }


def _canonical_mask_integrity(state: Mapping[str, Any], mask_root: Path) -> dict:
    ids_value = state.get("object_id")
    ids = (
        [int(value) for value in ids_value.detach().cpu().tolist()]
        if isinstance(ids_value, torch.Tensor) else []
    )
    rows = state.get("object_mask_observations")
    if not isinstance(rows, list):
        return {
            "canonical_contract_available": False,
            "referenced_files": 0,
            "stored_files": len(list(mask_root.glob("object_*/img_*.npz"))),
            "orphan_files": None,
            "missing_references": None,
        }
    referenced_names: set[tuple[int, str]] = set()
    missing = 0
    for index, object_id in enumerate(ids):
        for record in rows[index] if index < len(rows) and isinstance(rows[index], list) else []:
            raw = str(record.get("path") or record.get("mask_path") or "")
            if not raw:
                missing += 1
                continue
            name = Path(raw).name
            referenced_names.add((object_id, name))
            if not (mask_root / f"object_{object_id:06d}" / name).is_file():
                missing += 1
    stored = {
        (int(match.group(1)), path.name)
        for path in mask_root.glob("object_*/img_*.npz")
        if (match := re.fullmatch(r"object_(\d+)", path.parent.name))
    }
    return {
        "canonical_contract_available": True,
        "referenced_files": len(referenced_names),
        "stored_files": len(stored),
        "orphan_files": len(stored - referenced_names),
        "missing_references": missing,
    }


def classify_residual_candidates(
    rows: list[dict], presentation_ids: set[int],
) -> dict[str, list[dict]]:
    visible: list[dict] = []
    hidden: list[dict] = []
    for row in rows:
        if not isinstance(row, dict):
            continue
        try:
            first_id, second_id = int(row["first_id"]), int(row["second_id"])
        except (KeyError, TypeError, ValueError):
            continue
        target = visible if first_id in presentation_ids and second_id in presentation_ids else hidden
        target.append(row)
    duplicate_like = [
        row for row in visible
        if str(row.get("raw_relation") or "").lower().startswith("duplicate")
    ]
    return {
        "visible_duplicate_like": duplicate_like,
        "visible_similar_distinct": [row for row in visible if row not in duplicate_like],
        "hidden_endpoint": hidden,
    }


def confirmed_semantic_contract_violation(row: Mapping[str, Any]) -> bool:
    """Fail closed when a confirmed label lacks quantified independent support."""
    if str(row.get("semantic_tier") or "").lower() != "confirmed":
        return False
    try:
        group_count = float(row["semantic_independent_group_count"])
        unique_views = float(row["semantic_unique_view_count"])
        max_overlap = float(row["semantic_max_support_overlap"])
    except (KeyError, TypeError, ValueError):
        return True
    return not (
        row.get("semantic_confirmation_eligible") is True
        and math.isfinite(group_count)
        and group_count.is_integer()
        and group_count >= 2
        and math.isfinite(unique_views)
        and unique_views.is_integer()
        and unique_views >= 6
        and math.isfinite(max_overlap)
        and 0.0 <= max_overlap <= 0.25
    )


def dashboard_warning_layout(
    warnings: list[Mapping[str, Any]], *, max_rows: int = 4,
) -> list[tuple[str, float]]:
    """Bound warning text above the reserved dashboard footer."""
    if max_rows < 1:
        raise ValueError("max_rows must be positive")
    lines = [
        textwrap.shorten(
            f"• {row.get('code', 'warning')}: {row.get('detail', 'no detail')}",
            width=108,
            placeholder="…",
        )
        for row in warnings
        if isinstance(row, Mapping)
    ]
    if len(lines) > max_rows:
        hidden = len(lines) - (max_rows - 1)
        lines = lines[: max_rows - 1] + [
            f"• +{hidden} additional warning{'s' if hidden != 1 else ''}; see retention_funnel.json"
        ]
    if not lines:
        return []
    positions = np.linspace(0.29, 0.13, num=len(lines)).tolist()
    return list(zip(lines, positions, strict=True))


def presentation_parent_violations(
    catalog: list[dict], presentation_ids: set[int],
) -> list[int]:
    """Find resolved hidden rows whose canonical chain has no visible parent."""
    by_id = {
        int(row["id"]): row
        for row in catalog
        if isinstance(row, dict) and row.get("id") is not None
    }
    violations: list[int] = []
    for object_id, row in sorted(by_id.items()):
        tier = str(row.get("semantic_tier") or "").strip().lower()
        status = str(row.get("display_status") or "").strip().lower()
        if (
            row.get("metric_active") is False
            or tier not in {"confirmed", "probable"}
            or not status.endswith("_suppressed")
            or object_id in presentation_ids
        ):
            continue
        current = row
        seen = {object_id}
        valid_parent = False
        while True:
            try:
                parent_id = int(current["canonical_object_id"])
            except (KeyError, TypeError, ValueError):
                break
            if parent_id < 0 or parent_id in seen:
                break
            seen.add(parent_id)
            parent = by_id.get(parent_id)
            if parent is None:
                break
            if parent_id in presentation_ids and parent.get("presentation_visible") is True:
                valid_parent = True
                break
            current = parent
        if not valid_parent:
            violations.append(object_id)
    return violations


def build_funnel(run_dir: Path) -> dict:
    raw_state = _state(run_dir / "mapping/scene_state_raw.pt")
    raw_detections, filtered_detections, traced_views = _trace_counts(
        run_dir / "mapping/debug_trace.jsonl"
    )
    evidence = _evidence_counts(raw_state)
    geometry = _json(run_dir / "qa/geometry/audit.json")
    visual = _json(run_dir / "qa/visual_consistency/audit.json")
    surface = _json(run_dir / "qa/surface_support/audit.json")
    semantic_report = _json(run_dir / "qa/semantics/consensus/semantic_consensus_report.json")
    raw_catalog = json.loads(
        (run_dir / "qa/semantics/consensus/semantic_consensus_catalog.json").read_text(encoding="utf-8")
    )
    semantic_rows = (
        raw_catalog.get("objects") or []
        if isinstance(raw_catalog, dict)
        else raw_catalog if isinstance(raw_catalog, list) else []
    )
    tiers = Counter(str(row.get("semantic_tier") or "unclassified") for row in semantic_rows)
    dedup = _json(run_dir / "qa/dedup/audit.json")
    acceptance = _json(run_dir / "qa/acceptance/result.json", required=False)
    final_catalog = json.loads((run_dir / "final/catalog.json").read_text(encoding="utf-8"))
    presentation = json.loads((run_dir / "final/presentation_catalog.json").read_text(encoding="utf-8"))
    if not isinstance(final_catalog, list) or not isinstance(presentation, list):
        raise TypeError("Final catalogs must be JSON arrays")
    tracks = _tensor_count(raw_state, "object_id")
    geometry_evaluated = int(geometry.get("evaluated_objects") or 0)
    geometry_pass = int(geometry.get("passed_objects") or 0)
    visual_pass = int(visual.get("retained_objects") or 0)
    surface_pass = int(surface.get("passed_objects") or 0)
    diagnostic_nonpass = surface.get("diagnostic_nonpass_active_objects")
    surface_flagged = (
        len(diagnostic_nonpass)
        if isinstance(diagnostic_nonpass, list)
        else int(surface.get("borderline_objects") or 0)
        + int(surface.get("rejected_objects") or 0)
    )
    surface_active_before = int(surface.get("active_before") or 0)
    surface_active_after = int(surface.get("active_after") or 0)
    mask_integrity = _canonical_mask_integrity(raw_state, run_dir / "mapping/masks")
    residual = dedup.get("residual_strong_candidates") or []
    presentation_ids = {
        int(row["id"])
        for row in presentation
        if isinstance(row, dict) and row.get("id") is not None
    }
    legacy_residual_groups = classify_residual_candidates(residual, presentation_ids)
    acceptance_counts = acceptance.get("counts") or {}
    acceptance_after = acceptance.get("classification_after") or {}
    acceptance_available = acceptance.get("schema") == "farm.final-acceptance.v1"
    if acceptance_available:
        residual_duplicate_like_count = int(
            acceptance_after.get("auto_holdout") or 0
        ) + int(acceptance_after.get("release_blocking_ambiguous") or 0)
        residual_similar_distinct_count = int(
            acceptance_after.get("similar_distinct") or 0
        )
        residual_hidden_count = int(acceptance_after.get("hidden_context") or 0)
        residual_duplicate_like: list[dict] = []
        residual_similar_distinct: list[dict] = []
        residual_hidden: list[dict] = []
    else:
        residual_duplicate_like = legacy_residual_groups["visible_duplicate_like"]
        residual_similar_distinct = legacy_residual_groups["visible_similar_distinct"]
        residual_hidden = legacy_residual_groups["hidden_endpoint"]
        residual_duplicate_like_count = len(residual_duplicate_like)
        residual_similar_distinct_count = len(residual_similar_distinct)
        residual_hidden_count = len(residual_hidden)
    stages = [
        ("detections_raw", raw_detections),
        ("detections_filtered", filtered_detections),
        ("tracks_created", tracks),
        ("tracks_ge_3", evidence["at_least_3"]),
        ("tracks_ge_5", evidence["at_least_5"]),
        ("geometry_pass", geometry_pass),
        ("visual_pass", visual_pass),
        ("final_metric", len(final_catalog)),
        ("presentation_visible", len(presentation)),
    ]
    structural_errors: list[str] = []
    if any(value < 0 or not math.isfinite(float(value)) for _, value in stages):
        structural_errors.append("nonfinite_or_negative_funnel_count")
    if mask_integrity.get("missing_references") not in (None, 0):
        structural_errors.append("missing_canonical_mask_references")
    presentation_parent_violation_ids = presentation_parent_violations(
        final_catalog, presentation_ids
    )
    if presentation_parent_violation_ids:
        structural_errors.append(
            "presentation_eligible_object_suppressed_to_hidden_parent"
        )
    semantic_invariant_violations = [
        int(row.get("id", -1))
        for row in semantic_rows
        if confirmed_semantic_contract_violation(row)
    ]
    if semantic_invariant_violations:
        structural_errors.append("confirmed_semantics_violate_independence_contract")
    if acceptance_available:
        if str(acceptance.get("status") or "").upper() == "FAIL" or any(
            int(acceptance_counts.get(key) or 0) != 0
            for key in (
                "remaining_auto_clusters",
                "remaining_blocking_clusters",
                "label_hard_errors",
            )
        ):
            structural_errors.append("final_acceptance_gate_failed")
        if int(acceptance_counts.get("presentation_after") or -1) != len(presentation):
            structural_errors.append("acceptance_presentation_count_mismatch")
    warnings: list[dict[str, Any]] = []
    def warn(code: str, detail: str, value: Any, threshold: Any) -> None:
        warnings.append({"code": code, "detail": detail, "value": value, "threshold": threshold})
    if geometry_evaluated and geometry_pass / geometry_evaluated < 0.70:
        warn("low_geometry_retention", "geometry pass/evaluated", geometry_pass / geometry_evaluated, 0.70)
    if tracks and evidence["at_least_5"] / tracks < 0.25:
        warn(
            "low_track_maturation",
            "tracks with at least five observations / all tracks",
            evidence["at_least_5"] / tracks,
            0.25,
        )
    if visual_pass and surface_flagged / max(visual_pass, 1) > 0.25:
        warn(
            "surface_diagnostic_high_rejection",
            "diagnostic sampled-center support flags many visual/geometry-valid objects",
            surface_flagged / max(visual_pass, 1),
            0.25,
        )
    confirmed = int(tiers.get("confirmed", 0))
    if visual_pass and confirmed / visual_pass < 0.50:
        warn("low_confirmed_semantic_coverage", "confirmed/visual pass", confirmed / visual_pass, 0.50)
    if int(mask_integrity.get("orphan_files") or 0) > 0:
        warn("orphan_mask_sidecars", "stored sidecars are not referenced by canonical state", mask_integrity["orphan_files"], 0)
    if residual_duplicate_like_count:
        warn(
            "residual_duplicate_review_candidates",
            "acceptance-classified duplicate/blocking evidence remains presentation-visible",
            residual_duplicate_like_count,
            0,
        )
    if acceptance_available:
        for row in acceptance.get("warnings") or []:
            if not isinstance(row, Mapping):
                continue
            code = str(row.get("code") or "acceptance_warning")
            if any(existing.get("code") == code for existing in warnings):
                continue
            warn(
                code,
                "final acceptance label/runtime quality signal",
                row.get("count", row.get("rate", row.get("elapsed_seconds"))),
                row.get("threshold", row.get("budget_seconds")),
            )
    if surface_active_after < surface_active_before:
        warn("surface_gate_demoted_active", "diagnostic surface evidence changed canonical active state", surface_active_before - surface_active_after, 0)
    quality_status = "FAIL" if structural_errors else ("WARN" if warnings else "PASS")
    return {
        "schema": "farm.retention-quality.v1",
        "scene_id": str(_json(run_dir / "manifest.json").get("scene_id") or run_dir.parent.parent.name),
        "structural_status": "FAIL" if structural_errors else "PASS",
        "quality_status": quality_status,
        "structural_errors": structural_errors,
        "warnings": warnings,
        "funnel": {name: value for name, value in stages},
        "details": {
            "traced_views": traced_views,
            "geometry_evaluated": geometry_evaluated,
            "surface_diagnostic_pass": surface_pass,
            "surface_diagnostic_flagged": surface_flagged,
            "surface_enforced": bool((surface.get("policy") or {}).get("enforce")),
            "semantic_tiers": dict(tiers),
            "semantic_resolved": int(semantic_report.get("resolved") or 0),
            "semantic_invariant_violation_ids": semantic_invariant_violations,
            "presentation_parent_violation_ids": presentation_parent_violation_ids,
            "duplicate_groups": len(dedup.get("duplicate_groups") or []),
            "suppressed_objects": int(dedup.get("suppressed_objects") or 0),
            "acceptance": acceptance if acceptance_available else None,
            "acceptance_held_objects": int(acceptance_counts.get("held_objects") or 0),
            "acceptance_initial_auto_holdout_clusters": int(
                acceptance_counts.get("auto_holdout_clusters_initial") or 0
            ),
            "acceptance_initial_blocking_clusters": int(
                acceptance_counts.get("blocking_clusters_initial") or 0
            ),
            "residual_strong_duplicate_candidates": residual,
            "residual_strong_duplicate_candidates_total": len(residual),
            "residual_duplicate_like_visible": residual_duplicate_like,
            "residual_duplicate_like_visible_count": residual_duplicate_like_count,
            "residual_similar_distinct_visible": residual_similar_distinct,
            "residual_similar_distinct_visible_count": residual_similar_distinct_count,
            "residual_hidden_endpoint": residual_hidden,
            "residual_hidden_endpoint_count": residual_hidden_count,
            "legacy_residual_raw_relation_groups": legacy_residual_groups,
            "mask_evidence": mask_integrity,
        },
        "sources": {
            "mapping_trace": "mapping/debug_trace.jsonl",
            "mapping_state": "mapping/scene_state_raw.pt",
            "geometry": "qa/geometry/audit.json",
            "visual": "qa/visual_consistency/audit.json",
            "surface": "qa/surface_support/audit.json",
            "semantics": "qa/semantics/consensus/semantic_consensus_catalog.json",
            "dedup": "qa/dedup/audit.json",
            "final_acceptance": (
                "qa/acceptance/result.json" if acceptance_available else None
            ),
            "final_catalog": "final/catalog.json",
        },
    }


def render_dashboard(report: Mapping[str, Any], output: Path) -> None:
    funnel = report["funnel"]
    labels = list(funnel)
    values = np.asarray([funnel[label] for label in labels], dtype=np.float64)
    details = report["details"]
    plt.style.use("dark_background")
    fig = plt.figure(figsize=(24, 13.5), dpi=160, facecolor="#09111b")
    grid = fig.add_gridspec(2, 2, height_ratios=(1.05, 0.95), hspace=0.28, wspace=0.20)
    ax = fig.add_subplot(grid[0, :])
    colours = ["#55d6be", "#48bfe3", "#5e60ce", "#f4b942", "#f19c79", "#ee6c4d", "#e85d75", "#cf6aee", "#9be564"]
    bars = ax.bar(np.arange(len(values)), values, color=colours[: len(values)], alpha=0.93)
    ax.set_yscale("symlog", linthresh=10)
    ax.set_xticks(np.arange(len(values)), [label.replace("_", "\n") for label in labels], fontsize=11)
    ax.set_ylabel("objects / detections (symlog)")
    ax.set_title("Evidence retention funnel — absolute counts, no hidden normalization", fontsize=18, pad=14)
    ax.grid(axis="y", alpha=0.18)
    for bar, value in zip(bars, values):
        ax.text(bar.get_x() + bar.get_width() / 2, value * 1.06 + 0.3, f"{int(value):,}", ha="center", va="bottom", fontsize=11, fontweight="bold")

    ax2 = fig.add_subplot(grid[1, 0])
    semantic = details.get("semantic_tiers") or {}
    names = [name for name in ("confirmed", "probable", "geometry_only", "unclassified") if semantic.get(name, 0)]
    sem_values = [semantic[name] for name in names]
    if sem_values:
        ax2.bar(names, sem_values, color=["#55d6be", "#f4b942", "#7b8794", "#b0bec5"][: len(names)])
        for index, value in enumerate(sem_values):
            ax2.text(index, value + max(sem_values) * 0.025, str(value), ha="center", fontweight="bold")
    ax2.set_title("Semantic evidence tiers", fontsize=16)
    ax2.set_ylabel("objects")
    ax2.grid(axis="y", alpha=0.18)

    ax3 = fig.add_subplot(grid[1, 1])
    ax3.axis("off")
    status_colour = {"PASS": "#55d6be", "WARN": "#f4b942", "FAIL": "#ff6b6b"}.get(str(report["quality_status"]), "white")
    lines = [
        (f"RESULT QUALITY: {report['quality_status']}", status_colour, 18, "bold"),
        (f"Structural integrity: {report['structural_status']}", "#dbe8f4", 13, "normal"),
        (f"Surface diagnostic: {details.get('surface_diagnostic_pass', 0)} pass / {details.get('surface_diagnostic_flagged', 0)} flagged; enforce={details.get('surface_enforced')}", "#dbe8f4", 12, "normal"),
        (f"Mask evidence: {details.get('mask_evidence', {}).get('referenced_files', 0)} canonical / {details.get('mask_evidence', {}).get('orphan_files', 'n/a')} orphan", "#dbe8f4", 12, "normal"),
        (
            f"Acceptance: {details.get('acceptance_held_objects', 0)} held / "
            f"{details.get('residual_duplicate_like_visible_count', 0)} open overlap / "
            f"{details.get('acceptance_initial_blocking_clusters', 0)} initial blockers",
            "#dbe8f4", 12, "normal",
        ),
        (
            f"Similarity context: {details.get('residual_similar_distinct_visible_count', 0)} "
            f"classified distinct / {details.get('residual_hidden_endpoint_count', 0)} hidden endpoint",
            "#9fb0c3", 11, "normal",
        ),
    ]
    y = 0.95
    for line, colour, size, weight in lines:
        ax3.text(0.02, y, line, transform=ax3.transAxes, color=colour, fontsize=size, fontweight=weight, va="top")
        y -= 0.11
    del y
    for warning, warning_y in dashboard_warning_layout(report.get("warnings") or []):
        ax3.text(
            0.04, warning_y, warning, transform=ax3.transAxes,
            color="#f5c46b", fontsize=10, va="top",
        )
    ax3.text(
        0.02, 0.025,
        "Structural PASS does not imply exhaustive recall, correct labels, or duplicate-free output.",
        transform=ax3.transAxes, color="#8fa8bc", fontsize=10, va="bottom",
    )
    fig.suptitle(
        f"{str(report.get('scene_id') or 'scene').upper()} | RESULT-QUALITY AUDIT",
        fontsize=24,
        fontweight="bold",
        x=0.03,
        ha="left",
        y=0.985,
    )
    output.parent.mkdir(parents=True, exist_ok=True)
    fig.subplots_adjust(left=0.055, right=0.975, top=0.91, bottom=0.07)
    fig.savefig(output, dpi=160, facecolor=fig.get_facecolor())
    plt.close(fig)


def write_json_atomic(path: Path, payload: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp")
    try:
        with temporary.open("w", encoding="utf-8") as handle:
            json.dump(payload, handle, indent=2, ensure_ascii=False, allow_nan=False)
            handle.write("\n")
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--visual", type=Path)
    args = parser.parse_args()
    run_dir = args.run_dir.expanduser().resolve(strict=True)
    output = (args.output or run_dir / "qa/retention_funnel.json").expanduser().resolve()
    visual = (args.visual or run_dir / "visuals/07_retention_quality_dashboard_4k.jpg").expanduser().resolve()
    report = build_funnel(run_dir)
    write_json_atomic(output, report)
    render_dashboard(report, visual)
    print(json.dumps({
        "structural_status": report["structural_status"],
        "quality_status": report["quality_status"],
        "warnings": [row["code"] for row in report["warnings"]],
        "output": str(output),
        "visual": str(visual),
    }, indent=2))
    return 2 if report["structural_status"] != "PASS" else 0


if __name__ == "__main__":
    raise SystemExit(main())
