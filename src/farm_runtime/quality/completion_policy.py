"""Bind visible-scope proposals to accepted tracker observations.

This supplies a proposal-pool preference, never a replacement for geometric
eligibility. A failed or ambiguous completion falls back to the prior eligible
tracker observation.
"""

from __future__ import annotations

import json
from pathlib import Path

from farm_runtime.quality.mask_refinement import checked_file
from farm_runtime.quality.proposal_geometry import read_observations


def load_completion_preferences(path, audit, proposals, supplements, partial_view):
    completion = json.loads(Path(path).read_text())
    if (
        completion.get("schema") != "farm.visible-scope-completion.v1"
        or completion.get("closed_test_opened") is not False
        or completion.get("test_opened") is not False
        or completion.get("release_eligible") is not False
    ):
        raise ValueError("development visible-scope completion required")
    parent = json.loads(checked_file(completion["source_validation"]).read_text())
    if (
        parent.get("closed_test_opened") is not False
        or parent.get("release_eligible") is not False
        or parent.get("partial_view_association", False) != partial_view
    ):
        raise ValueError("completion requires the same development validation policy")
    for record, current in [
        (parent["source_audit"], audit),
        (parent["source_proposals"], proposals),
    ]:
        if checked_file(record).resolve() != Path(current).resolve():
            raise ValueError("completion validation source mismatch")
    bound = [checked_file(d).resolve() for d in parent.get("supplements", [])]
    if bound != [Path(p).resolve() for p in supplements]:
        raise ValueError("completion supplement order mismatch")
    if checked_file(completion["source_tracker"]).resolve() not in bound:
        raise ValueError("completion tracker absent from validated supplements")

    # One shared observation may contain proposals for several different objects.
    offsets = {}
    for source in [proposals, *supplements]:
        _, rows = read_observations(source)
        for row in rows:
            offsets[row["name"]] = offsets.get(row["name"], 0) + len(row["detections"])
    _, rows = read_observations(path)
    proposed = {r["name"]: r for r in rows}
    matches = {
        (g["group_id"], m["name"]): m
        for g in parent["groups"]
        for m in g["extra_matches"]
    }
    result, seen = {}, set()
    for row in completion["outputs"]:
        key = row["group_id"], row["name"]
        if key in seen or key not in matches:
            raise ValueError("unique bound completion object/view required")
        seen.add(key)
        match = matches[key]
        if (
            match["decision"] != "matched_static_surface"
            or row["selected_detection"] != match["selected_detection"]
            or not any(
                c["detection_index"] == row["selected_detection"]
                and c["eligible"] is True
                for c in match["candidates"]
            )
        ):
            raise ValueError("completion must start from an accepted tracker mask")
        response = row["response"]
        if response["validation_error"] is not None:
            continue
        parsed = json.loads(response["raw"])
        if parsed != response["parsed"]:
            raise ValueError("completion raw and parsed response differ")
        if parsed["scope_complete"] is not False or parsed["confidence"] != "high":
            continue
        name, gid = row["name"], row["group_id"]
        if name not in proposed or name not in offsets:
            raise ValueError("completion frame absent from validated sources")
        candidates = [
            offsets[name] + i
            for i, det in enumerate(proposed[name]["detections"])
            if det.get("source_group_id") == gid and det["label"] != "person"
        ]
        if len(candidates) != row["candidates"]:
            raise ValueError("completion object candidate binding mismatch")
        for field in ("source_crop", "source_image", "source_mask"):
            checked_file(row[field])
        if candidates:
            result[key] = dict(
                candidate_indices=candidates,
                fallback_detection=row["selected_detection"],
            )
    return result
