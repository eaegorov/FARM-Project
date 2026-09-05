"""Independent F1 annotation protocol and evaluation; never constructs FARM masks.

All lenses/directions at one physical timestamp share one split. Unlabelled
pixels are unknown, not background. Proposals can help navigation but cannot
become human gold through a rename or an empty review record.
"""
from __future__ import annotations

from collections import defaultdict
import hashlib
import json
from pathlib import Path
import re
from typing import Any, Mapping, Sequence

import numpy as np

from farm_runtime.quality_baseline import describe_file, json_digest, sha256_file, write_json

NAME = re.compile(r"(?P<camera>cam\d+)_(?P<timestamp>\d+)_(?P<direction>center|yaw_left|yaw_right|pitch_up|pitch_down)(?=[_.])")
DIRECTIONS = ("center", "yaw_left", "yaw_right", "pitch_up", "pitch_down")
LAYERS = ("whole_object", "core", "boundary", "thin_parts", "attachment", "protected", "carrier", "content", "background")
STATES = ("visible", "partly_occluded", "fully_occluded", "out_of_view", "absent", "ambiguous")


def image_identity(name: str) -> dict[str, str]:
    matches = list(NAME.finditer(Path(name).name))
    if len(matches) != 1:
        raise ValueError(f"cannot determine physical identity: {name}")
    return matches[0].groupdict()


def exposure_ledger(roots: Sequence[Path], excluded_root: Path | None = None) -> dict[str, Any]:
    """Conservative observed-frame ledger. No image pixels are read.

    Full COLMAP metadata listing every registered camera is intentionally not
    evidence that every image was used. Materialized frames and output images
    are evidence. Missing/deleted historical runs remain a stated limitation.
    """
    by_timestamp: dict[str, set[str]] = defaultdict(set)
    artifacts = []
    for root in roots:
        for p in sorted(root.rglob("*")):
            if not p.is_file() or "source_snapshot" in p.parts:
                continue
            if excluded_root and (p == excluded_root or excluded_root in p.parents):
                continue
            if "factory" not in str(p).lower():
                continue
            if p.name == "frames.json":
                payload = json.loads(p.read_text())
                frames = payload.get("frames", []) if isinstance(payload, dict) else payload
                matched = set()
                for frame in frames:
                    # Prefer original physical names over synthetic RGBD frame indices.
                    for key in ("source_image", "source_image_name", "image_name", "rgb_path", "file_path"):
                        value = frame.get(key)
                        if isinstance(value, str):
                            for match in NAME.finditer(Path(value).name):
                                matched.add(match["timestamp"])
                for ts in matched:
                    by_timestamp[ts].add(str(p))
                artifacts.append({**describe_file(p), "physical_timestamps": len(matched)})
            elif p.name == "manifest.json" and p.parent.name == "visuals":
                payload = json.loads(p.read_text())
                if payload.get("schema") == "farm.gold-navigation-visuals.v1":
                    matched = set()
                    for name in payload.get("source_hashes", {}):
                        matched.add(image_identity(name)["timestamp"])
                    for ts in matched:
                        by_timestamp[ts].add(str(p))
                    artifacts.append({**describe_file(p), "physical_timestamps": len(matched)})
            elif p.suffix.lower() in (".jpg", ".jpeg", ".png"):
                for match in NAME.finditer(p.name):
                    by_timestamp[match["timestamp"]].add(str(p))
    return {
        "schema": "farm.gold-exposure-ledger.v1", "roots": list(map(str, roots)),
        "observed_timestamps": sorted(by_timestamp),
        "evidence": {k: sorted(v) for k, v in sorted(by_timestamp.items())},
        "frame_manifests": artifacts,
        "limitation": "Covers extant FARM materialized frames/output images, not deleted runs or unlogged external viewing.",
    }


def assign_splits(timestamps: Sequence[str], observed: set[str], *, seed: str,
                  test_fraction: float = 0.20, dev_fraction: float = 0.25) -> dict[str, str]:
    if not 0 < test_fraction <= 1 or not 0 < dev_fraction < 1:
        raise ValueError("test_fraction must be in (0,1]; dev_fraction in (0,1)")
    if len(set(timestamps)) != len(timestamps):
        raise ValueError("duplicate physical timestamps")
    result = {}
    for ts in sorted(timestamps):
        def uniform(salt: str) -> float:
            return int(hashlib.sha256(f"{seed}:{salt}:{ts}".encode()).hexdigest()[:16], 16) / 2**64
        if ts not in observed and uniform("test") < test_fraction:
            result[ts] = "test"
        else:
            result[ts] = "dev" if uniform("development") < dev_fraction else "train"
    return result


def verify_packet(packet: Mapping[str, Any]) -> list[str]:
    errors = []
    if packet.get("schema") != "farm.manual-gold-packet.v1":
        errors.append("unsupported_packet_schema")
    split_by_ts = packet.get("split_by_timestamp", {})
    if set(split_by_ts.values()) - {"train", "dev", "test"}:
        errors.append("invalid_split")
    observed = set(packet.get("historically_observed_timestamps", []))
    if any(split_by_ts.get(ts) == "test" for ts in observed):
        errors.append("test_contains_historically_observed_timestamp")
    seen = set()
    for row in packet.get("observations", []):
        key = row["observation_id"]
        if key in seen:
            errors.append("duplicate_observation:" + key)
        seen.add(key)
        identity = image_identity(row["image_name"])
        if identity["timestamp"] != row["physical_timestamp"]:
            errors.append("timestamp_name_mismatch:" + key)
        if row["split"] != split_by_ts.get(identity["timestamp"]):
            errors.append("timestamp_split_leak:" + key)
        if row["direction"] != identity["direction"] or row["camera"] != identity["camera"]:
            errors.append("camera_name_mismatch:" + key)
    if not seen:
        errors.append("empty_packet")
    return errors


def annotation_template(packet: Mapping[str, Any], split: str) -> dict[str, Any]:
    if split not in {"train", "dev", "test"}:
        raise ValueError("invalid split")
    return {
        "schema": "farm.manual-gold-annotations.v1", "packet_sha256": json_digest(packet),
        "reviewer_id": None, "reviewer_kind": "human", "independent": None,
        "annotation_method": "manual_from_original_rgb", "split": split,
        "observations": [{"observation_id": r["observation_id"], "state": None,
                          "object_identity": None, "regions": [], "notes": "", "reviewed": False}
                         for r in packet["observations"] if r["split"] == split],
        "objects": [{"object_id": x["object_id"], "canonical_label": None, "caption": None,
                     "attachment_policy": None, "carrier_content_policy": None,
                     "physical_dimensions_m": None, "dimension_evidence": None,
                     "scope_reviewed": False, "notes": ""}
                    for x in packet["objects"]],
    }


def rasterize_regions(regions: Sequence[Mapping[str, Any]], shape: tuple[int, int]) -> dict[str, np.ndarray]:
    """Explicit human polygons in original image pixels, including holes.

    Polygon fill samples image pixels consistently through Pillow. Thin parts
    require closed polygons; open strokes with guessed widths are rejected.
    Unknown is the complement of all explicitly labelled layers.
    """
    from PIL import Image, ImageDraw
    height, width = shape
    if height <= 0 or width <= 0:
        raise ValueError("invalid image dimensions")
    masks = {name: np.zeros(shape, dtype=bool) for name in LAYERS}
    for region in regions:
        layer = region.get("layer")
        if layer not in LAYERS:
            raise ValueError(f"unknown layer: {layer}")
        canvas = Image.new("1", (width, height), 0)
        draw = ImageDraw.Draw(canvas)
        for polygon, fill in [(region.get("points"), 1)] + [(p, 0) for p in region.get("holes", [])]:
            points = np.asarray(polygon, dtype=np.float64)
            if points.ndim != 2 or points.shape[0] < 3 or points.shape[1] != 2 or not np.isfinite(points).all():
                raise ValueError("polygon requires >=3 finite xy vertices")
            if (points < 0).any() or (points[:, 0] > width - 1).any() or (points[:, 1] > height - 1).any():
                raise ValueError("polygon coordinates must use original image pixels")
            draw.polygon([tuple(p) for p in points], fill=fill)
        masks[layer] |= np.asarray(canvas, dtype=bool)
    labelled = np.logical_or.reduce(list(masks.values()))
    masks["unknown"] = ~labelled
    validate_layers(masks)
    return masks


def validate_layers(masks: Mapping[str, np.ndarray]) -> None:
    if set(masks) != set(LAYERS) | {"unknown"}:
        raise ValueError("missing or unexpected annotation layers")
    shapes = {np.asarray(x).shape for x in masks.values()}
    if len(shapes) != 1 or len(next(iter(shapes))) != 2:
        raise ValueError("all layers must have the same HxW shape")
    if any(np.asarray(x).dtype != np.bool_ for x in masks.values()):
        raise ValueError("annotation layers must be boolean")
    whole = masks["whole_object"]
    for layer in ("core", "boundary", "thin_parts"):
        if (masks[layer] & ~whole).any():
            raise ValueError(layer + " must be a subset of whole_object")
    if (masks["core"] & masks["boundary"]).any():
        raise ValueError("core and uncertain boundary must be disjoint")
    positive = np.logical_or.reduce([masks[x] for x in LAYERS if x not in ("protected", "background")])
    negative = masks["protected"] | masks["background"]
    if (positive & negative).any():
        raise ValueError("positive and protected/background overlap")
    labelled = positive | negative
    if not np.array_equal(masks["unknown"], ~labelled):
        raise ValueError("unlabelled pixels must be unknown")


def check_annotation(packet: Mapping[str, Any], annotation: Mapping[str, Any], *,
                     load_pixels: bool = True, require_independence: bool = True) -> list[str]:
    errors = verify_packet(packet)
    if annotation.get("schema") != "farm.manual-gold-annotations.v1":
        errors.append("unsupported_annotation_schema")
    if annotation.get("packet_sha256") != json_digest(packet):
        errors.append("packet_hash_mismatch")
    if not str(annotation.get("reviewer_id") or "").strip():
        errors.append("missing_human_reviewer")
    if annotation.get("reviewer_kind") != "human" or (require_independence and annotation.get("independent") is not True):
        errors.append("independent_human_review_required")
    if annotation.get("annotation_method") != "manual_from_original_rgb":
        errors.append("pseudo_gt_not_allowed")
    split = annotation.get("split")
    if split not in ("train", "dev", "test"):
        errors.append("invalid_annotation_split")
    expected = {r["observation_id"]: r for r in packet["observations"] if r["split"] == split}
    seen = set()
    for row in annotation.get("observations", []):
        key = row.get("observation_id")
        if key not in expected or key in seen:
            errors.append(f"unknown_or_duplicate_observation:{key}")
            continue
        seen.add(key)
        if row.get("reviewed") is not True or row.get("state") not in STATES:
            errors.append(f"unreviewed_observation:{key}")
            continue
        if str(row.get("object_identity")) != str(expected[key]["object_id"]):
            errors.append(f"unconfirmed_identity:{key}")
        if load_pixels:
            try:
                masks = rasterize_regions(row.get("regions", []), tuple(expected[key]["shape_hw"]))
                visible = row["state"] in ("visible", "partly_occluded")
                if visible and not masks["whole_object"].any():
                    errors.append(f"missing_visible_mask:{key}")
                if not visible and masks["whole_object"].any():
                    errors.append(f"invisible_object_has_positive_mask:{key}")
                if visible and not (masks["background"] | masks["protected"]).any():
                    errors.append(f"missing_independent_negative_region:{key}")
            except ValueError as exc:
                errors.append(f"invalid_regions:{key}:{exc}")
    errors.extend("missing_observation:" + key for key in sorted(expected.keys() - seen))
    object_rows = {x.get("object_id"): x for x in annotation.get("objects", [])}
    if len(object_rows) != len(annotation.get("objects", [])):
        errors.append("duplicate_scope_object")
    if set(object_rows) - {x["object_id"] for x in packet["objects"]}:
        errors.append("unknown_scope_object")
    for obj in packet["objects"]:
        row = object_rows.get(obj["object_id"], {})
        if row.get("scope_reviewed") is not True:
            errors.append(f"scope_not_reviewed:{obj['object_id']}")
        if row.get("attachment_policy") not in ("include", "exclude", "separate", "unknown"):
            errors.append(f"missing_attachment_policy:{obj['object_id']}")
        if row.get("carrier_content_policy") not in ("carrier_only", "content_only", "whole_assembly", "not_applicable", "unknown"):
            errors.append(f"missing_carrier_content_policy:{obj['object_id']}")
        dims = row.get("physical_dimensions_m")
        if dims is not None:
            evidence = row.get("dimension_evidence")
            try:
                a = np.asarray(dims, dtype=float)
                valid = a.shape == (3,) and np.isfinite(a).all() and (a > 0).all()
                if not isinstance(evidence, Mapping):
                    valid = False
                else:
                    valid = valid and all(isinstance(evidence.get(k), str) and evidence[k].strip()
                                          for k in ("measurement_method", "scale_source", "object_scope", "axis_definition"))
                    uncertainty = np.asarray(evidence.get("uncertainty_m"), dtype=float)
                    valid = valid and uncertainty.shape == (3,) and np.isfinite(uncertainty).all() and (uncertainty >= 0).all()
            except (TypeError, ValueError):
                valid = False
            if not valid:
                errors.append(f"unsubstantiated_metric_dimensions:{obj['object_id']}")
    return errors


def compare_reviews(packet: Mapping[str, Any], left: Mapping[str, Any], right: Mapping[str, Any]) -> dict[str, Any]:
    errors = check_annotation(packet, left) + check_annotation(packet, right)
    if left.get("reviewer_id") == right.get("reviewer_id"):
        errors.append("two_distinct_humans_required")
    if left.get("split") != right.get("split"):
        errors.append("review_split_mismatch")
    report: dict[str, Any] = {"schema": "farm.manual-review-comparison.v1", "errors": errors,
                              "left_sha256": json_digest(left), "right_sha256": json_digest(right),
                              "disagreements": [], "status": "BLOCKED"}
    if errors:
        return report
    observations = {r["observation_id"]: r for r in packet["observations"]}
    right_rows = {r["observation_id"]: r for r in right["observations"]}
    for a in left["observations"]:
        key = a["observation_id"]
        b = right_rows[key]
        ma = rasterize_regions(a["regions"], tuple(observations[key]["shape_hw"]))
        mb = rasterize_regions(b["regions"], tuple(observations[key]["shape_hw"]))
        differences = {k: int(np.count_nonzero(ma[k] ^ mb[k])) for k in ma}
        if a["state"] != b["state"] or any(differences.values()):
            report["disagreements"].append({"observation_id": key, "left_state": a["state"],
                                            "right_state": b["state"], "different_pixels": differences})
    right_objects = {o["object_id"]: o for o in right["objects"]}
    for a in left["objects"]:
        b = right_objects[a["object_id"]]
        differences = [k for k in ("canonical_label", "caption", "attachment_policy", "carrier_content_policy",
                                  "physical_dimensions_m", "dimension_evidence")
                       if a.get(k) != b.get(k)]
        if differences:
            report["disagreements"].append({"object_id": a["object_id"], "scope_fields": differences})
    report["status"] = "REQUIRES_ADJUDICATION" if report["disagreements"] else "AGREED"
    return report


def mask_metrics(prediction: np.ndarray, masks: Mapping[str, np.ndarray]) -> dict[str, Any]:
    validate_layers(masks)
    pred = np.asarray(prediction)
    if pred.dtype != np.bool_ or pred.shape != masks["whole_object"].shape:
        raise ValueError("prediction must be boolean in original image dimensions")
    # Scope components such as content/attachments are NOT automatic negatives.
    # Uncertain boundary is reported separately, outside strict core metrics.
    negative = masks["background"] | masks["protected"]
    positive = masks["whole_object"] & ~masks["boundary"]
    valid = positive | negative
    tp = int((pred & positive).sum())
    fp = int((pred & negative).sum())
    fn = int((~pred & positive).sum())
    ratio = lambda n, d: n / d if d else None
    return {"tp": tp, "fp": fp, "fn": fn, "evaluated_pixels": int(valid.sum()),
            "precision": ratio(tp, tp + fp), "recall": ratio(tp, tp + fn),
            "iou": ratio(tp, tp + fp + fn),
            "thin_recall": ratio(int((pred & masks["thin_parts"]).sum()), int(masks["thin_parts"].sum())),
            "protected_leakage": ratio(int((pred & masks["protected"]).sum()), int(masks["protected"].sum())),
            "boundary_coverage": ratio(int((pred & masks["boundary"]).sum()), int(masks["boundary"].sum())),
            "predicted_unknown_pixels": int((pred & masks["unknown"]).sum()),
            "unknown_pixels": int(masks["unknown"].sum())}


def timestamp_macro_metrics(rows: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    """First average directions at a timestamp; then timestamps at an object."""
    grouped: dict[tuple[Any, Any], list[Mapping[str, Any]]] = defaultdict(list)
    seen = set()
    for row in rows:
        key = (row["object_id"], row["image_name"])
        if key in seen:
            raise ValueError("duplicate metric observation")
        seen.add(key)
        if image_identity(row["image_name"])["timestamp"] != row["physical_timestamp"]:
            raise ValueError("metric timestamp mismatch")
        grouped[(row["object_id"], row["physical_timestamp"])].append(row)
    metrics = ("precision", "recall", "iou", "thin_recall", "protected_leakage", "boundary_coverage")
    ts_rows = []
    for (obj, ts), views in sorted(grouped.items()):
        values = {m: [r[m] for r in views if r.get(m) is not None] for m in metrics}
        ts_rows.append({"object_id": obj, "physical_timestamp": ts, "views": len(views),
                        **{m: float(np.mean(v)) if v else None for m, v in values.items()}})
    objects = []
    for obj in sorted({k[0] for k in grouped}):
        values = {m: [r[m] for r in ts_rows if r["object_id"] == obj and r[m] is not None] for m in metrics}
        objects.append({"object_id": obj, "timestamps": sum(r["object_id"] == obj for r in ts_rows),
                        **{m: float(np.mean(v)) if v else None for m, v in values.items()}})
    return {"per_timestamp": ts_rows, "per_object": objects}
