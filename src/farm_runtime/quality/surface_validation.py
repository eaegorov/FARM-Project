"""Verify an existing surface with bounded additional concept observations."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import time

import numpy as np
from PIL import Image, ImageDraw
from scipy.ndimage import binary_dilation
from scipy.spatial import cKDTree

from farm_runtime.angular_discovery import rotate_image
from farm_runtime.obb_proposals import fit_surface_envelope, project_world
from farm_runtime.proposal_geometry import (
    equivalent_masks,
    mask_overlap,
    project_evidence,
    surface_points,
)
from farm_runtime.quality.mask_refinement import checked_file
from farm_runtime.quality.proposal_geometry import read_masks, read_observations
from farm_runtime.quality.surface_evidence import SurfaceInputs
from farm_runtime.quality_baseline import describe_file, write_json
from farm_runtime.surface_evidence import aggregate_timestamps, point_observation
from scripts.geometry.refine_farm_object_geometry import _fit_robust_obb


def select_observation(points, core, masks, detections, frame, labels, radius_m):
    """Require visible core support and reciprocal static surface overlap.

    An optional label allowlist limits inspection. Identity always requires
    geometry; ambiguous nested scopes and no-instance stay unknown.
    """
    rows = []
    tree = cKDTree(points)
    candidates = [
        i
        for i, d in enumerate(detections)
        if d["label"] != "person" and (labels is None or d["label"] in labels)
    ]
    groups = equivalent_masks(
        [masks[i] for i in candidates], [detections[i]["score"] for i in candidates]
    )
    for equivalent in groups:
        i = candidates[equivalent[0]]
        evidence = point_observation(points, masks[i], **frame)
        visible = int(evidence["visible"][core].sum())
        exact_agreement = float(evidence["positive"][core].sum() / max(1, visible))
        # Reuse FARM proposal association's one-pixel tolerance for identity.
        # Point votes remain exact foreground and >3 px visible background.
        agreement = project_evidence(points[core], masks[i], **frame)["mask_agreement"]
        candidate, metrics = surface_points(masks[i], **frame)
        reciprocal = (
            float(np.mean(tree.query(candidate, k=1)[0] <= radius_m))
            if len(candidate)
            else 0.0
        )
        fraction = visible / max(1, int(core.sum()))
        accepted = (
            visible >= 20
            and fraction >= 0.25
            and agreement >= 0.8
            and reciprocal >= 0.3
        )
        score = 2 * agreement * reciprocal / max(1e-9, agreement + reciprocal)
        rows.append(
            dict(
                detection_index=i,
                label=detections[i]["label"],
                equivalent_detection_indices=[candidates[j] for j in equivalent],
                core_visible_points=visible,
                core_visible_fraction=fraction,
                core_mask_agreement=agreement,
                core_exact_mask_agreement=exact_agreement,
                reciprocal_surface_agreement=reciprocal,
                score=score,
                eligible=accepted,
                **metrics,
            )
        )
    eligible = sorted(
        [r for r in rows if r["eligible"]],
        key=lambda r: (-r["score"], r["detection_index"]),
    )
    if not eligible:
        return None, rows, "no_unambiguous_surface_match"
    best = eligible[0]
    if any(
        best["score"] - r["score"] < 0.08
        and mask_overlap(masks[best["detection_index"]], masks[r["detection_index"]])[0]
        < 0.8
        for r in eligible[1:]
    ):
        return None, rows, "ambiguous_competing_scope"
    return best["detection_index"], rows, "matched_static_surface"


def review_panel(points, counts, mask, obs, frame, path, title):
    H, W = frame["depth"].shape
    scale = 960 / max(H, W)
    width, height = round(W * scale), round(H * scale)
    with Image.open(checked_file(obs["source_image"])) as im:
        original = im.convert("RGB").resize((width, height))
    overlay = np.asarray(original).copy()
    foreground = np.asarray(
        Image.fromarray(mask).resize((width, height), Image.Resampling.NEAREST)
    )
    overlay[foreground] = (
        overlay[foreground] * 0.75 + np.array([30, 220, 130]) * 0.25
    ).astype(np.uint8)
    overlay = Image.fromarray(overlay)
    draw = ImageDraw.Draw(overlay)
    K = frame["K"].copy()
    K[0] *= width / frame["depth"].shape[1]
    K[1] *= height / frame["depth"].shape[0]
    uv, z = project_world(points, K, frame["T_world_cam"])
    # All projected source samples are shown, including behind observed depth;
    # state colours are aggregate evidence, not visibility in this one image.
    for state, color in [
        ("unknown", "#ffbc32"),
        ("corroborated", "#27dfff"),
        ("contradicted", "#ff304a"),
    ]:
        selected = (
            ~(counts["corroborated"] | counts["contradicted"])
            if state == "unknown"
            else counts[state].copy()
        )
        selected &= (
            (z > 0)
            & (uv[:, 0] >= 0)
            & (uv[:, 0] < width)
            & (uv[:, 1] >= 0)
            & (uv[:, 1] < height)
        )
        for x, y in uv[selected]:
            draw.ellipse((x - 2, y - 2, x + 2, y + 2), fill=color)
    turns = obs["applied_quarter_turns"]
    left = Image.fromarray(rotate_image(np.asarray(original), turns))
    right = Image.fromarray(rotate_image(np.asarray(overlay), turns))
    sheet = Image.new("RGB", (left.width * 2, left.height + 60), "#111827")
    sheet.paste(left, (0, 60))
    sheet.paste(right, (left.width, 60))
    draw = ImageDraw.Draw(sheet)
    draw.text((8, 8), title, fill="white")
    draw.text(
        (8, 28),
        "RGB | green: candidate mask (see decision); cyan: corroborated; red: >=2 negative timestamps; amber: unresolved. Points may be occluded in this view.",
        fill="white",
    )
    sheet.save(path, quality=94)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("audit", "proposals", "output"):
        parser.add_argument("--" + name, type=Path, required=True)
    parser.add_argument("--supplement", type=Path, action="append", default=[])
    parser.add_argument("--world-up", type=float, nargs=3, required=True)
    args = parser.parse_args(argv)
    if args.output.exists():
        raise ValueError("new output required")
    audit = json.loads(args.audit.read_text())
    if audit.get("closed_test_opened") is not False:
        raise ValueError("development-only point audit required")
    geometry_path = checked_file(audit["source_geometry"])
    inputs = SurfaceInputs(geometry_path)
    _, extra = read_observations(args.proposals)
    extra_plan = json.loads(checked_file(audit["adaptive_plan"]).read_text())
    source_manifest = json.loads(args.proposals.read_text())
    if source_manifest["plan"]["sha256"] != audit["adaptive_plan"]["sha256"]:
        raise ValueError("extra proposals must use the selected development plan")
    started = time.monotonic()
    with np.load(checked_file(audit["point_evidence"]), allow_pickle=False) as archive:
        old = {k: archive[k] for k in archive.files}
    prepared = {}
    for obs in extra:
        name = obs["name"]
        if (
            name not in inputs.trusted
            or obs["source_image"] != extra_plan["sources"][name]["source_image"]
            or obs["timestamp"] != str(inputs.frames[name]["frame_id"])
        ):
            raise ValueError("extra observation registration/identity mismatch")
        if not any(q["prompt"] == "person" for q in obs["queries"]):
            raise ValueError("explicit person exclusion query required")
        masks = read_masks(obs, args.proposals.parent)
        if tuple(obs["grid_shape_hw"]) != inputs.frame(name)["depth"].shape:
            raise ValueError("extra observation grid mismatch")
        frame = dict(inputs.frame(name))
        people = [m for m, d in zip(masks, obs["detections"]) if d["label"] == "person"]
        if people:
            frame["excluded"] = frame["excluded"] | binary_dilation(
                np.logical_or.reduce(people), iterations=2
            )
        prepared[name] = (obs, masks, frame)
    supplements = []
    for supplement_path in args.supplement:
        _, additions = read_observations(supplement_path)
        for obs in additions:
            name = obs["name"]
            if name not in prepared:
                raise ValueError(
                    "supplement must be limited to already selected development views"
                )
            base, masks, frame = prepared[name]
            if any(
                obs[key] != base[key]
                for key in (
                    "timestamp",
                    "source_image",
                    "grid_shape_hw",
                    "applied_quarter_turns",
                )
            ):
                raise ValueError("supplement source identity/grid mismatch")
            added = read_masks(obs, supplement_path.parent)
            base = dict(base, detections=base["detections"] + obs["detections"])
            people = [
                m for m, d in zip(added, obs["detections"]) if d["label"] == "person"
            ]
            if people:
                frame = dict(
                    frame,
                    excluded=frame["excluded"]
                    | binary_dilation(np.logical_or.reduce(people), iterations=2),
                )
            prepared[name] = base, masks + added, frame
        supplements.append(describe_file(supplement_path))
    args.output.mkdir(parents=True)
    (args.output / "visuals").mkdir()
    output_rows, arrays = [], {}
    by_id = {g["id"]: g for g in inputs.geometry["groups"]}
    for row in audit["groups"]:
        group_id = row["group_id"]
        prefix = f"group_{group_id:04d}"
        group = by_id[group_id]
        members, points, timestamps, radii = inputs.support(group)
        if not np.array_equal(points, old[prefix + "_points"]):
            raise ValueError("source point order changed")
        core = old[prefix + "_core"]
        observations = []
        visuals = []
        for node in members:
            observations.append(
                (
                    node["timestamp"],
                    point_observation(
                        points, inputs.mask(node), **inputs.frame(node["frame"])
                    ),
                )
            )
        before = aggregate_timestamps(observations, len(points), timestamps)
        matches = []
        for selection in row["selected_views"]:
            name = selection["name"]
            if name not in prepared:
                raise ValueError("selected extra view missing")
            obs, masks, frame = prepared[name]
            selected, candidates, decision = select_observation(
                points,
                core,
                masks,
                obs["detections"],
                frame,
                None,
                max(0.04, float(np.median(radii))),
            )
            matches.append(
                dict(
                    name=name,
                    timestamp=obs["timestamp"],
                    decision=decision,
                    selected_detection=selected,
                    candidates=candidates,
                )
            )
            if selected is not None:
                observations.append(
                    (
                        obs["timestamp"],
                        point_observation(points, masks[selected], **frame),
                    )
                )
            display_index = selected
            if display_index is None and candidates:
                display_index = max(candidates, key=lambda c: c["score"])[
                    "detection_index"
                ]
            display_mask = (
                masks[display_index]
                if display_index is not None
                else np.zeros(frame["depth"].shape, bool)
            )
            visuals.append((name, obs, display_mask, frame, decision))
        after = aggregate_timestamps(observations, len(points), timestamps)
        rotation = _fit_robust_obb(
            points, 0.005, orientation_mode="gravity_yaw", up_vector=args.world_up
        )["rotation_matrix"]
        weights = np.zeros(len(points))
        for timestamp in np.unique(timestamps):
            selected = timestamps == timestamp
            weights[selected] = 1 / selected.sum()
        boxes = {}
        for label, keep in [
            ("all_observed", np.ones(len(points), bool)),
            ("existing_evidence", ~before["contradicted"]),
            ("extra_evidence", ~after["contradicted"]),
        ]:
            # Preserve original per-timestamp weights after filtering; do not
            # amplify the surviving portion of a mostly rejected observation.
            boxes[label] = fit_surface_envelope(
                points[keep],
                weights[keep],
                rotation,
                "fixed original FARM gravity-PCA; observed support only",
            )
        metrics = dict(
            group_id=group_id,
            source_points=len(points),
            extra_matches=matches,
            point_counts_before={
                k: int(v.sum()) for k, v in before.items() if v.dtype == bool
            },
            point_counts_after={
                k: int(v.sum()) for k, v in after.items() if v.dtype == bool
            },
            boxes=boxes,
            physical_extent_validated=False,
            native_gaussian_ownership_changed=False,
        )
        for name, obs, mask, frame, decision in visuals:
            path = args.output / "visuals" / f"{prefix}_{Path(name).stem}.jpg"
            review_panel(
                points,
                after,
                mask,
                obs,
                frame,
                path,
                f"group {group_id} | {name} | {decision}; candidate only",
            )
            metrics.setdefault("visuals", []).append(describe_file(path))
        arrays.update(
            {prefix + "_points": points, prefix + "_source_timestamps": timestamps}
        )
        arrays.update({prefix + "_" + k: v for k, v in after.items()})
        output_rows.append(metrics)
        print(
            json.dumps(
                {
                    k: metrics[k]
                    for k in ("group_id", "point_counts_before", "point_counts_after")
                }
            ),
            flush=True,
        )
    np.savez_compressed(args.output / "point_evidence.npz", **arrays)
    write_json(
        args.output / "manifest.json",
        dict(
            schema="farm.surface-validation.v1",
            source_audit=describe_file(args.audit),
            source_proposals=describe_file(args.proposals),
            supplements=supplements,
            matching_labels_are_not_identity=True,
            groups=output_rows,
            point_evidence=describe_file(args.output / "point_evidence.npz"),
            total_seconds=time.monotonic() - started,
            interpretation="Additional development-mask consistency, not independent physical extent or closed-test accuracy. No disconnected component is removed merely for lacking support. Native Gaussian bank is unchanged.",
            closed_test_opened=False,
            release_eligible=False,
        ),
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
