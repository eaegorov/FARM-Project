#!/usr/bin/env python3
"""Export resolved and unresolved catalogs from the multi-crop VLM review."""

from __future__ import annotations

# Allow direct execution from the source checkout.
import sys as _entry_sys
from pathlib import Path as _EntryPath
_entry_root = _EntryPath(__file__).resolve().parents[2]
_entry_sys.path[:0] = [str(_entry_root), str(_entry_root / "src")]

import argparse
import json
from collections import Counter
from pathlib import Path

import matplotlib
import numpy as np

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--review", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    output_dir = args.output_dir.expanduser().resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    payload = json.loads(args.review.read_text(encoding="utf-8"))
    rows = list(payload.get("objects") or [])
    crops_per_object = int(payload.get("crops_per_object") or 3)
    resolved, unresolved = [], []
    for row in rows:
        decision = str(row.get("review_decision") or "unknown").lower()
        category = str(row.get("review_category") or "unknown").strip().lower()
        base = {
            "id": int(row["id"]),
            "observations": int(row["observations"]),
            "position_world_m": row["position_world_m"],
            "cov6": row["cov6"],
            "raw_category": row["category"],
            "raw_caption": row["caption"],
            "review_decision": decision,
            "review_confidence": float(row.get("review_confidence", 0.0)),
            "crop_observation_count": crops_per_object,
            "qwen3_vl_embedding_dim": row.get("qwen3_vl_embedding_dim", 0),
            "siglip2_embedding_dim": row.get("siglip2_embedding_dim", 0),
        }
        if decision in {"keep", "relabel"} and category != "unknown":
            base.update(
                {
                    "category": category,
                    "description": str(row.get("review_description") or "").strip(),
                    "semantic_status": "vlm_multi_crop_resolved",
                }
            )
            resolved.append(base)
        else:
            base.update(
                {
                    "proposed_description": str(row.get("review_description") or "").strip(),
                    "semantic_status": "needs_manual_review",
                }
            )
            unresolved.append(base)

    resolved.sort(key=lambda row: (-row["observations"], row["category"], row["id"]))
    unresolved.sort(key=lambda row: (-row["observations"], row["id"]))
    decisions = Counter(str(row.get("review_decision") or "unknown") for row in rows)
    categories = Counter(row["category"] for row in resolved)
    summary = {
        "schema": "farm.factory-reviewed-catalog.v1",
        "robust_objects_reviewed": len(rows),
        "resolved_objects": len(resolved),
        "unresolved_objects": len(unresolved),
        "decisions": dict(decisions),
        "resolved_categories": dict(categories.most_common()),
        "policy": str(
            payload.get("policy")
            or "generic multi-view Qwen3-VL review without prior labels or scene-specific class hints"
        ),
        "source_review": str(args.review.resolve()),
    }
    (output_dir / "reviewed_robust_catalog.json").write_text(
        json.dumps(resolved, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )
    (output_dir / "semantic_review_queue.json").write_text(
        json.dumps(unresolved, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )
    (output_dir / "semantic_review_summary.json").write_text(
        json.dumps(summary, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )

    plt.style.use("dark_background")
    fig, axes = plt.subplots(2, 2, figsize=(24, 13.5), dpi=160, facecolor="#090b10")
    for axis in axes.flat:
        axis.set_facecolor("#0e121b")
    labels = ["keep", "relabel", "unknown"]
    values = [decisions[label] for label in labels]
    bars = axes[0, 0].bar(labels, values, color=["#4ade80", "#22d3ee", "#f4b942"])
    axes[0, 0].bar_label(bars, padding=5, fontsize=14)
    axes[0, 0].set_title("Multi-crop semantic decisions", fontsize=16)
    axes[0, 0].set_ylabel("robust objects")
    axes[0, 0].grid(axis="y", color="#344054", alpha=0.3)

    top_categories = categories.most_common(15)[::-1]
    axes[0, 1].barh(
        [item[0] for item in top_categories],
        [item[1] for item in top_categories],
        color="#7dd3fc",
    )
    axes[0, 1].set_title("Resolved open-vocabulary categories", fontsize=16)
    axes[0, 1].set_xlabel("objects")
    axes[0, 1].grid(axis="x", color="#344054", alpha=0.3)

    confidences = np.asarray([row["review_confidence"] for row in resolved], dtype=float)
    axes[1, 0].hist(confidences, bins=np.linspace(0, 1, 11), color="#4ade80", edgecolor="#101828")
    axes[1, 0].set_title("Resolved-label VLM confidence", fontsize=16)
    axes[1, 0].set_xlabel("reported confidence")
    axes[1, 0].set_ylabel("objects")
    axes[1, 0].grid(axis="y", color="#344054", alpha=0.3)

    axes[1, 1].axis("off")
    axes[1, 1].text(
        0.02, 0.97, "Top resolved catalog by multi-view evidence", transform=axes[1, 1].transAxes,
        fontsize=16, fontweight="bold", va="top",
    )
    y = 0.88
    for row in resolved[:10]:
        axes[1, 1].text(
            0.02, y,
            f"#{row['id']:03d}  {row['category'].upper()}  "
            f"(obs={row['observations']}, conf={row['review_confidence']:.2f})",
            transform=axes[1, 1].transAxes, fontsize=9.4, color="white", va="top",
        )
        y -= 0.057
    axes[1, 1].text(
        0.02, 0.06,
        f"Resolved: {len(resolved)}/{len(rows)} | Manual review queue: {len(unresolved)}\n"
        f"Policy: 5+ mapping observations, two-pass Qwen3-VL review over {crops_per_object} crops.\n"
        "The raw FARM scene_state remains unchanged and auditable.",
        transform=axes[1, 1].transAxes, fontsize=10.5, color="#b9c2cf", va="bottom",
    )

    fig.suptitle(
        "FARM FACTORY | ROBUST SEMANTIC REVIEW",
        fontsize=23, fontweight="bold", x=0.02, ha="left", y=0.985,
    )
    fig.text(
        0.02, 0.947,
        f"{len(rows)} robust objects reviewed | {len(resolved)} resolved | {len(unresolved)} intentionally left unknown",
        color="#b9c2cf", fontsize=11,
    )
    fig.tight_layout(rect=(0, 0, 1, 0.925))
    fig.savefig(
        output_dir / "semantic_review_dashboard_4k.jpg",
        facecolor=fig.get_facecolor(), dpi=160,
    )
    plt.close(fig)
    print(json.dumps(summary, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
