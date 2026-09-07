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
    GeometryPolicy,
    compare_surfaces,
    equivalent_masks,
    mask_overlap,
    project_evidence,
    surface_points,
)
from farm_runtime.quality.mask_refinement import checked_file
from farm_runtime.quality.proposal_geometry import read_masks, read_observations
from farm_runtime.quality.surface_evidence import SurfaceInputs
from farm_runtime.quality.registered_frames import extension_kwargs
from farm_runtime.quality_baseline import describe_file, write_json
from farm_runtime.surface_evidence import aggregate_timestamps, point_observation
from scripts.geometry.refine_farm_object_geometry import _fit_robust_obb


def prepared_surface(name, points, mask, footprint, containers=()):
    return dict(
        frame=name,
        points=points,
        mask=mask,
        pixel_footprint_m=footprint,
        scope_containers=list(containers),
        tree=cKDTree(points),
        bounds=(points.min(0), points.max(0)),
        projection_mask=binary_dilation(mask, iterations=1),
    )


def reference_surfaces(inputs, members):
    references = []
    for node in members:
        mask = inputs.mask(node)
        containers = []
        for other in inputs.nodes.values():
            if (
                other["frame"] != node["frame"]
                or other["id"] == node["id"]
                or other.get("source_priority", 0) > node.get("source_priority", 0)
            ):
                continue
            _, a, b = mask_overlap(mask, inputs.mask(other))
            if a >= 0.85 > b:
                containers.append(other["id"])
        references.append(
            (
                prepared_surface(
                    node["frame"],
                    inputs.clouds[f"node_{node['id']:04d}"],
                    mask,
                    node["pixel_footprint_m"],
                    containers,
                ),
                inputs.frame(node["frame"]),
            )
        )
    return references


def source_support_conflicts(evidence, source_counts):
    """Reject new masks that erase enough visible original foreground.

    Check each source separately so a densely sampled view cannot hide loss
    from a smaller source. Occluded, excluded and boundary-uncertain points
    never count as background. Expansion beyond an incomplete source is allowed.
    """
    policy = GeometryPolicy()
    if any(type(count) is not int or count < 0 for _, count in source_counts):
        raise ValueError("nonnegative source point counts required")
    if sum(count for _, count in source_counts) != len(evidence["visible"]):
        raise ValueError("source point partition changed")
    start = 0
    rows = []
    for name, count in source_counts:
        end = start + count
        visible = int(evidence["visible"][start:end].sum())
        negative = int(evidence["negative"][start:end].sum())
        enough = (
            visible >= policy.min_points
            and visible / max(1, count) >= policy.min_visible_fraction
        )
        rows.append(
            dict(
                source_name=name,
                source_points=count,
                visible_points=visible,
                definite_background_points=negative,
                source_foreground_contradicted=bool(
                    enough
                    and negative / max(1, visible) > policy.negative_mask_fraction
                ),
            )
        )
        start = end
    return rows


def select_observation(
    points,
    core,
    masks,
    detections,
    frame,
    labels,
    radius_m,
    *,
    partial_context=None,
    scope_review=None,
    completion_preference=None,
    group_id=None,
    source_counts=None,
):
    """Require visible core support and reciprocal static surface overlap.

    An optional label allowlist limits inspection. Identity always requires
    geometry; no-instance stays unknown. Bound scope review may resolve an
    ambiguity only by supporting the existing geometric best.
    """
    rows = []
    tree = cKDTree(points)
    candidates = [
        i
        for i, d in enumerate(detections)
        if d["label"] != "person"
        and (labels is None or d["label"] in labels)
        and (group_id is None or d.get("source_group_id", group_id) == group_id)
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
        # Dense source samples can collapse onto a few target pixels.
        # Require the same minimum static surface support at both ends.
        accepted = (
            len(candidate) >= 20
            and visible >= 20
            and fraction >= 0.25
            and agreement >= 0.8
            and reciprocal >= 0.3
        )
        partial_evidence = []
        if not accepted and partial_context is not None and len(candidate) >= 20:
            name = partial_context["target_name"]
            containers = []
            for j in candidates:
                if j == i:
                    continue
                _, a, b = mask_overlap(masks[i], masks[j])
                if a >= 0.85 > b:
                    containers.append(j)
            target = prepared_surface(
                name, candidate, masks[i], metrics["pixel_footprint_m"], containers
            )
            for reference, reference_frame in partial_context["references"]:
                result = compare_surfaces(
                    reference,
                    target,
                    {reference["frame"]: reference_frame, name: frame},
                    GeometryPolicy(partial_view_association=True),
                )
                partial_evidence.append(dict(source_name=reference["frame"], **result))
            # Reuse the same FOV/visibility/negative/scope constraints as proposal
            # association. A missing surface outside the source image is unknown.
            accepted = any(
                r["decision"] == "match"
                and r["reason"] == "partial_view_surface_support"
                for r in partial_evidence
            ) and not any(r["decision"] == "separate" for r in partial_evidence)
        preservation = (
            source_support_conflicts(evidence, source_counts)
            if source_counts is not None
            else None
        )
        if preservation and any(
            row["source_foreground_contradicted"] for row in preservation
        ):
            accepted = False
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
                **(
                    {"source_support_preservation": preservation}
                    if preservation is not None
                    else {}
                ),
                **(
                    {"partial_view_evidence": partial_evidence}
                    if partial_evidence
                    else {}
                ),
                **metrics,
            )
        )
    eligible = sorted(
        [r for r in rows if r["eligible"]],
        key=lambda r: (-r["score"], r["detection_index"]),
    )
    if scope_review is not None and rows != scope_review["candidates"]:
        raise ValueError("scope review candidate evidence changed")
    if not eligible:
        return None, rows, "no_unambiguous_surface_match"
    if completion_preference is not None:
        if scope_review is not None:
            raise ValueError(
                "completion preference and scope review are separate policies"
            )
        fallback = completion_preference["fallback_detection"]
        seed = completion_preference.get("seed_detection", fallback)
        preferred = [
            r
            for r in eligible
            if r["detection_index"] in completion_preference["candidate_indices"]
            and mask_overlap(masks[seed], masks[r["detection_index"]])[1] >= 0.90
        ]
        if preferred:
            best = preferred[0]
            ambiguous = any(
                best["score"] - r["score"] < 0.08
                and mask_overlap(
                    masks[best["detection_index"]], masks[r["detection_index"]]
                )[0]
                < 0.8
                for r in preferred[1:]
            )
            if not ambiguous:
                # Among the same scope, prefer a mask reproduced by both SAM
                # prompt variants. Reciprocal overlap with a clipped source can
                # otherwise reward holes. This never resolves scope ambiguity
                # and never adds candidates that failed geometry/core preservation.
                for candidate in preferred:
                    variants = {
                        detections[i].get("variant")
                        for i in candidate["equivalent_detection_indices"]
                        if i in completion_preference["candidate_indices"]
                    }
                    if {"vlm_box", "vlm_missing_parts"} <= variants and mask_overlap(
                        masks[best["detection_index"]],
                        masks[candidate["detection_index"]],
                    )[0] >= 0.8:
                        best = candidate
                        break
                return best["detection_index"], rows, "matched_static_surface"
        if any(r["detection_index"] == fallback for r in eligible):
            return fallback, rows, "matched_static_surface"
        return None, rows, "ambiguous_competing_scope"
    best = eligible[0]
    if any(
        best["score"] - r["score"] < 0.08
        and mask_overlap(masks[best["detection_index"]], masks[r["detection_index"]])[0]
        < 0.8
        for r in eligible[1:]
    ):
        if scope_review is None or scope_review["choice"] != best["detection_index"]:
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
    parser.add_argument(
        "--partial-view-association",
        action="store_true",
        help="Reuse guarded partial-FOV matching for additional views",
    )
    parser.add_argument(
        "--scope-review",
        type=Path,
        help="Bounded VLM consensus supporting the best geometric candidate",
    )
    parser.add_argument(
        "--scope-completion",
        type=Path,
        help="Try geometrically eligible completion proposals before the prior tracker mask",
    )
    args = parser.parse_args(argv)
    if args.scope_completion and args.scope_review:
        raise ValueError(
            "scope completion preference and scope review cannot be combined"
        )
    if args.output.exists():
        raise ValueError("new output required")
    completion_preferences = {}
    if args.scope_completion:
        from farm_runtime.quality.completion_policy import load_completion_preferences

        completion_preferences = load_completion_preferences(
            args.scope_completion,
            args.audit,
            args.proposals,
            args.supplement,
            args.partial_view_association,
        )
        args.supplement.append(args.scope_completion)
    audit = json.loads(args.audit.read_text())
    if audit.get("closed_test_opened") is not False:
        raise ValueError("development-only point audit required")
    geometry_path = checked_file(audit["source_geometry"])
    inputs = SurfaceInputs(geometry_path, **extension_kwargs(audit))
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
    scope_choices = {}
    if args.scope_review:
        from farm_runtime.quality.scope_review import load_scope_choices

        scope_choices = load_scope_choices(
            args.scope_review,
            args.audit,
            args.proposals,
            args.supplement,
            args.partial_view_association,
        )
    args.output.mkdir(parents=True)
    (args.output / "visuals").mkdir()
    output_rows, arrays = [], {}
    by_id = {g["id"]: g for g in inputs.geometry["groups"]}
    for row in audit["groups"]:
        group_id = row["group_id"]
        prefix = f"group_{group_id:04d}"
        group = by_id[group_id]
        members, points, timestamps, radii = inputs.support(group)
        references = (
            reference_surfaces(inputs, members)
            if args.partial_view_association
            else None
        )
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
                scope_review=scope_choices.get((group_id, name)),
                completion_preference=completion_preferences.get((group_id, name)),
                group_id=group_id,
                source_counts=[
                    (node["frame"], len(inputs.clouds[f"node_{node['id']:04d}"]))
                    for node in members
                ],
                partial_context=(
                    {"target_name": name, "references": references}
                    if references is not None
                    else None
                ),
            )
            matches.append(
                dict(
                    name=name,
                    timestamp=obs["timestamp"],
                    decision=decision,
                    selected_detection=selected,
                    candidates=candidates,
                    **(
                        {
                            "completion_selection": (
                                "unresolved"
                                if selected is None
                                else (
                                    "accepted_completion"
                                    if selected
                                    in completion_preferences[group_id, name][
                                        "candidate_indices"
                                    ]
                                    else "retained_prior"
                                )
                            )
                        }
                        if (group_id, name) in completion_preferences
                        else {}
                    ),
                    **(
                        {"scope_resolution": "vlm_consensus_supports_geometry_best"}
                        if selected is not None
                        and (group_id, name) in scope_choices
                        and scope_choices[group_id, name]["choice"] == selected
                        else {}
                    ),
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
            source_support_policy="reject_visible_original_foreground_contradictions",
            partial_view_association=args.partial_view_association,
            **(
                {
                    "scope_completion": describe_file(args.scope_completion),
                    "completion_policy": "eligible_completion_then_eligible_prior_fallback",
                }
                if args.scope_completion
                else {}
            ),
            **(
                {"scope_review": describe_file(args.scope_review)}
                if args.scope_review
                else {}
            ),
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
