"""Retain existing annotations only for unchanged, namespace-bound evidence.

This is explicit annotation preservation, not a model inference cache or a
comparison of model versions. Changed views are reviewed through the normal VLM.
"""

from __future__ import annotations

from collections import defaultdict
import json
from pathlib import Path
import time

from farm_runtime.quality.mask_refinement import checked_file
from farm_runtime.quality.scene_catalog import bound_appearance
from farm_runtime.quality_baseline import describe_file, write_json


def evidence_rows(path, geometry, native, checked):
    document = json.loads(path.read_text())
    if (
        document.get("schema") != "farm.object-scope-evidence.v1"
        or document.get("reserved_test_opened") is not False
        or document.get("release_eligible") is not False
        or document.get("detector_labels_in_images") is not False
    ):
        raise ValueError("development appearance evidence required")
    for key, expected in (("source_groups", geometry), ("source_native_input", native)):
        record = document.get(key)
        if not record or record["sha256"] != expected["sha256"]:
            raise ValueError("appearance evidence binding mismatch")
        checked_file(record)
    groups, seen, seen_sources = defaultdict(list), set(), set()
    for row in document["observations"]:
        key = (row["object_id"], row["image_id"])
        if key in seen:
            raise ValueError("duplicate appearance observation")
        seen.add(key)
        name = row.get("source_name")
        if not isinstance(name, str) or not name:
            raise ValueError("stable appearance source name required")
        source_key = row["object_id"], name
        if source_key in seen_sources:
            raise ValueError("duplicate appearance source observation")
        seen_sources.add(source_key)
        normalized = dict(row)
        # The frame number is local to a prepared view list. The source name,
        # timestamp, RGB, mask, crop and their order still bind the annotation.
        normalized.pop("image_id")
        for field in ("source_image", "crop", "masks", "native_mask"):
            record = row.get(field)
            if record is None and field == "native_mask":
                continue
            cache_key = (record["path"], record["sha256"])
            if cache_key not in checked:
                checked_file(record)
                checked.add(cache_key)
            normalized[field] = record["sha256"]
        groups[row["object_id"]].append(normalized)
    ids = [g["id"] for g in document["groups"]]
    if len(ids) != len(set(ids)) or set(ids) != set(groups):
        raise ValueError("appearance evidence group selection mismatch")
    for rows in groups.values():
        if len(rows) != 2 or len({r["timestamp"] for r in rows}) != 2:
            raise ValueError("two independent appearance views required")
    return document, dict(groups)


def refresh_appearance(previous_stage, evidence, model, output):
    """Keep exact old object records; infer only changed/new object evidence."""
    from farm_runtime.quality import refinement
    from farm_runtime.semantic_refinement import APPEARANCE_PROMPT

    if output.exists():
        raise ValueError("new appearance output required")
    started = time.monotonic()
    current = json.loads(evidence.read_text())
    geometry, native = current["source_groups"], current["source_native_input"]
    if not native:
        raise ValueError("native-bound appearance evidence required")
    stage = json.loads(previous_stage.read_text())
    prior_native = stage.get("source_native_input")
    if not prior_native:
        raise ValueError("previous appearance must be bound to native input")
    objects, stage = bound_appearance(previous_stage, geometry, prior_native)
    prior_path = checked_file(stage["appearance"])
    prior = json.loads(prior_path.read_text())
    # Retention deliberately supports the existing two-view compact profile.
    expected = dict(
        scene_context_included=True,
        context_scale=None,
        masked_target_rgb=False,
        human_labels_in_prompt=False,
        detector_labels_in_prompt=False,
        reserved_test_opened=False,
        release_eligible=False,
    )
    if any(key not in prior or prior[key] != value for key, value in expected.items()):
        raise ValueError("standard compact appearance profile required")
    old_config = checked_file(prior["model_config"])
    if (
        old_config.resolve() != (model / "config.json").resolve()
        or prior["model_config"]["sha256"]
        != describe_file(model / "config.json")["sha256"]
        or checked_file(prior["prompt"]).read_text() != APPEARANCE_PROMPT
    ):
        raise ValueError(
            "retention requires the existing model configuration and prompt"
        )
    checked = set()
    old, old_rows = evidence_rows(
        checked_file(prior["proposals"]), geometry, prior_native, checked
    )
    fresh, fresh_rows = evidence_rows(evidence, geometry, native, checked)
    if set(old_rows) != set(objects):
        raise ValueError("previous annotations differ from evidence selection")
    retained = sorted(
        oid for oid, rows in fresh_rows.items() if rows == old_rows.get(oid)
    )
    changed = sorted(set(fresh_rows) - set(retained))
    for oid in retained:
        for sheet in objects[oid]["sheets"]:
            checked_file(sheet)
    rebindings = []
    preserved = []
    for oid in retained:
        old_ids = [r["image_id"] for r in old["observations"] if r["object_id"] == oid]
        new_views = [r for r in fresh["observations"] if r["object_id"] == oid]
        new_ids = [r["image_id"] for r in new_views]
        item = objects[oid]
        if old_ids != new_ids:
            rebindings.append(
                dict(
                    group_id=oid,
                    previous_image_ids=old_ids,
                    current_image_ids=new_ids,
                    source_names=[r["source_name"] for r in new_views],
                )
            )
            parsed = item.get("parsed")
            if isinstance(parsed, dict) and "observed_image_ids" in parsed:
                if parsed["observed_image_ids"] != old_ids:
                    raise ValueError(
                        "retained annotation image IDs differ from its evidence"
                    )
                # Preserve raw model output and sheets with their old provenance;
                # only translate parsed references into the current local IDs.
                item = dict(item, parsed=dict(parsed, observed_image_ids=new_ids))
        preserved.append(item)
    output.mkdir(parents=True)
    new = None
    if changed:
        subset_path = output / "changed_evidence.json"
        write_json(
            subset_path,
            dict(
                fresh,
                observations=[
                    r for r in fresh["observations"] if r["object_id"] in changed
                ],
                groups=[g for g in fresh["groups"] if g["id"] in changed],
                parent_evidence=describe_file(evidence),
            ),
        )
        refinement.main(
            [
                "vlm",
                "--proposals",
                str(subset_path),
                "--model",
                str(model),
                "--compact-semantics",
                "--views",
                "2",
                "--output",
                str(output / "updated"),
            ]
        )
        new = json.loads((output / "updated/manifest.json").read_text())
        if sorted(r["object_id"] for r in new["objects"]) != changed:
            raise ValueError("updated appearance selection mismatch")
    combined = dict(
        new if new else prior,
        proposals=describe_file(evidence),
        objects=sorted(
            preserved + (new["objects"] if new else []),
            key=lambda r: r["object_id"],
        ),
        model_load_seconds=new["model_load_seconds"] if new else 0.0,
        total_seconds=time.monotonic() - started,
        annotation_retention=dict(
            previous_stage=describe_file(previous_stage),
            previous_appearance=describe_file(prior_path),
            retained_group_ids=retained,
            image_id_rebindings=rebindings,
            refreshed_group_ids=changed,
            new_review_requests=len(changed),
            mode="preserve_existing_annotations_on_identical_evidence",
            model_cache_compatibility_claimed=False,
            timing_semantics=(
                "Top-level time is this refresh only. Retained object timings, "
                "tokens and sheets belong to the previous appearance provenance."
            ),
        ),
    )
    write_json(output / "manifest.json", combined)
    print(
        json.dumps(
            dict(
                retained=len(retained),
                refreshed=len(changed),
                seconds=combined["total_seconds"],
            )
        ),
        flush=True,
    )
