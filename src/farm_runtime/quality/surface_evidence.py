"""Audit per-point support and select bounded extra registered development views."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import time

import numpy as np
from PIL import Image
from scipy.ndimage import binary_dilation

from farm_runtime.proposal_geometry import project_evidence
from farm_runtime.quality.mask_refinement import checked_file
from farm_runtime.quality.proposal_geometry import read_masks, read_observations
from farm_runtime.quality_baseline import describe_file, write_json
from farm_runtime.surface_evidence import (
    aggregate_timestamps,
    point_observation,
    surface_components,
)


class SurfaceInputs:
    def __init__(self, geometry_path):
        self.geometry_path = geometry_path
        self.geometry = json.loads(geometry_path.read_text())
        if self.geometry.get("test_opened") is not False:
            raise ValueError("development-only geometry required")
        self.frames_path = checked_file(self.geometry["inputs"]["frames"])
        index = json.loads(self.frames_path.read_text())
        prep = json.loads(
            checked_file(self.geometry["inputs"]["prep_summary"]).read_text()
        )
        if (
            index.get("depth_units") != "metres"
            or index.get("pose_translation_units") != "metres"
            or prep.get("status") != "complete"
        ):
            raise ValueError("completed metric RGBD preparation required")
        self.trusted = {
            name
            for group in index["camera_registration"]["groups"]
            if group["trusted"]
            for name in group["source_images"]
        }
        self.frames = {
            r["source_image"]: r
            for r in index["frames"]
            if r["source_image"] in self.trusted
        }
        self.proposals_path = checked_file(self.geometry["inputs"]["proposals"])
        _, observations = read_observations(self.proposals_path)
        self.observations = {r["name"]: r for r in observations}
        self.transients_path = checked_file(self.geometry["inputs"]["transients"])
        _, transients = read_observations(self.transients_path)
        self.transients = {r["name"]: r for r in transients}
        self.nodes = {r["id"]: r for r in self.geometry["nodes"]}
        with np.load(
            checked_file(self.geometry["surface_artifact"]), allow_pickle=False
        ) as archive:
            self.clouds = {k: archive[k] for k in archive.files}
        self.recorded_depths = {
            r["source"]: r["depth_artifact"] for r in self.geometry["frames"]
        }
        self._frames, self._masks = {}, {}
        self.depth_artifacts = {}

    def frame(self, name):
        if name not in self._frames:
            row = self.frames[name]
            path = self.frames_path.parent / row["depth_path"]
            if name in self.recorded_depths:
                checked = checked_file(self.recorded_depths[name])
                if checked.resolve() != path.resolve():
                    raise ValueError("registered depth path changed")
            depth = np.load(path, allow_pickle=False)
            h, w = depth.shape
            scale = 640 / max(h, w)
            H, W = round(h * scale), round(w * scale)
            depth = np.asarray(
                Image.fromarray(depth).resize((W, H), Image.Resampling.NEAREST)
            )
            K = np.asarray(row["K"], float).copy()
            K[0] *= W / w
            K[1] *= H / h
            excluded = np.zeros((H, W), bool)
            if name in self.transients:
                masks = read_masks(self.transients[name], self.transients_path.parent)
                if masks:
                    excluded = binary_dilation(
                        np.logical_or.reduce(masks), iterations=2
                    )
            self._frames[name] = dict(
                depth=depth,
                K=K,
                T_world_cam=np.asarray(row["T_world_cam"], float),
                excluded=excluded,
            )
            self.depth_artifacts[name] = describe_file(path)
        return self._frames[name]

    def mask(self, node):
        name = node["frame"]
        if name not in self._masks:
            self._masks[name] = read_masks(
                self.observations[name], self.proposals_path.parent
            )
        return self._masks[name][node["representative_detection"]]

    def support(self, group):
        members = [self.nodes[i] for i in group["members"]]
        clouds = [self.clouds[f"node_{n['id']:04d}"] for n in members]
        points = np.concatenate(clouds)
        timestamps = np.concatenate(
            [np.full(len(c), n["timestamp"]) for n, c in zip(members, clouds)]
        )
        radii = np.concatenate(
            [
                np.full(len(c), max(0.025, 3 * n["pixel_footprint_m"]))
                for n, c in zip(members, clouds)
            ]
        )
        return members, points, timestamps, radii


def rank_views(points, core, existing_timestamps, inputs, limit, allowed_names=None):
    """Prioritize visible disconnected support while retaining object context.

    This geometric ranking precedes transient segmentation. It never casts
    foreground/background votes and does not claim a new mask is trustworthy.
    """
    center = np.median(points[core], axis=0)
    distance = np.linalg.norm(points - center, axis=1)
    priority = np.where(~core, distance, 0.0)
    if not priority.any():
        priority = np.ones(len(points))
    candidates = []
    for name, row in inputs.frames.items():
        if allowed_names is not None and name not in allowed_names:
            continue  # Restrict before reading depth, not after ranking.
        if str(row["frame_id"]) in existing_timestamps:
            continue
        frame = inputs.frame(name)
        evidence = project_evidence(
            points,
            np.ones(frame["depth"].shape, bool),
            **frame,
            mask_is_dilated=True,
            include_point_indices=True,
        )
        visible = np.zeros(len(points), bool)
        visible[evidence["surface_indices"]] = True
        fraction = float(visible[core].mean())
        if visible[core].sum() < 20 or fraction < 0.25:
            continue
        candidates.append(
            dict(
                name=name,
                timestamp=str(row["frame_id"]),
                core_visible_fraction=fraction,
                disconnected_visible_points=int((visible & ~core).sum()),
                weighted_disconnected_visibility=float(
                    priority[visible].sum() / priority.sum()
                ),
                transient_checked=name in inputs.transients,
                visible=visible,
            )
        )
    selected = []
    covered = np.zeros(len(points), bool)
    timestamps = set(existing_timestamps)
    for _ in range(limit):
        available = [r for r in candidates if r["timestamp"] not in timestamps]
        if not available:
            break

        def score(row):
            marginal = float(priority[row["visible"] & ~covered].sum() / priority.sum())
            return (
                marginal
                + 0.15 * row["weighted_disconnected_visibility"]
                + 0.05 * row["core_visible_fraction"]
            )

        best = max(available, key=lambda row: (score(row), row["name"]))
        selected.append(
            dict(
                name=best["name"],
                timestamp=best["timestamp"],
                selection_score=score(best),
            )
        )
        timestamps.add(best["timestamp"])
        covered |= best["visible"]
    return [
        {k: v for k, v in r.items() if k != "visible"} for r in candidates
    ], selected


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--geometry", type=Path, required=True)
    selection = parser.add_mutually_exclusive_group(required=True)
    selection.add_argument("--group-id", type=int, action="append")
    selection.add_argument(
        "--auto-groups",
        type=int,
        help="Select up to 16 unconfirmed single-timestamp groups",
    )
    parser.add_argument("--candidate-budget", type=int, default=64)
    parser.add_argument("--view-budget", type=int, default=12)
    parser.add_argument("--plan", type=Path, required=True)
    parser.add_argument("--extra-views", type=int, default=2)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args(argv)
    if args.output.exists() or not 1 <= args.extra_views <= 4:
        raise ValueError(
            "new output and one to four extra timestamps per group required"
        )
    if args.auto_groups is not None and (
        not 1 <= args.auto_groups <= 16
        or not args.auto_groups <= args.candidate_budget <= 128
        or not 1 <= args.view_budget <= 24
    ):
        raise ValueError(
            "recovery budgets: groups1..16, candidates>=groups up to128, views1..24"
        )
    plan = json.loads(args.plan.read_text())
    if plan.get("test_opened") is not False:
        raise ValueError("development-only sampling plan required")
    started = time.monotonic()
    inputs = SurfaceInputs(args.geometry)
    by_id = {g["id"]: g for g in inputs.geometry["groups"]}
    from farm_runtime.quality.recovery_schedule import (
        recovery_candidates,
        budgeted_views,
    )

    automatic = args.auto_groups is not None
    candidate_rows, available = (
        recovery_candidates(
            inputs.geometry, args.candidate_budget, set(plan["sources"])
        )
        if automatic
        else ([], None)
    )
    group_ids = [r["group_id"] for r in candidate_rows] if automatic else args.group_id
    if not set(group_ids) <= by_id.keys():
        raise ValueError("unknown group ID")
    for gid in group_ids:
        if any(
            inputs.nodes[i]["frame"] not in plan["sources"]
            for i in by_id[gid]["members"]
        ):
            raise ValueError("source group outside development plan")
    args.output.mkdir(parents=True)
    rows, arrays, names, prompts, screened = [], {}, [], [], []
    for group_id in dict.fromkeys(group_ids):
        if automatic and len(rows) >= args.auto_groups:
            break
        group = by_id[group_id]
        members, points, timestamps, radii = inputs.support(group)
        labels, components = surface_components(points, timestamps, radii)
        observations = []
        for node in members:
            observations.append(
                (
                    node["timestamp"],
                    point_observation(
                        points, inputs.mask(node), **inputs.frame(node["frame"])
                    ),
                )
            )
        counts = aggregate_timestamps(observations, len(points), timestamps)
        core = counts["corroborated"] | (labels == components[0]["id"])
        candidates, selected = rank_views(
            points,
            core,
            set(timestamps),
            inputs,
            args.extra_views,
            allowed_names=set(plan["sources"]),
        )
        if automatic:
            kept = budgeted_views(selected, names, args.view_budget)
            screened.append(
                dict(
                    group_id=group_id,
                    selected_views=kept,
                    reason=(
                        "scheduled"
                        if kept
                        else (
                            "view_budget"
                            if selected
                            else "no_depth_visible_new_timestamp"
                        )
                    ),
                )
            )
            selected = kept
            if not selected:
                continue
        for name in [r["name"] for r in selected]:
            if name not in plan["sources"]:
                raise ValueError("selected registered view not in development plan")
            names.append(name)
        prompts.extend(group["candidate_labels"])
        prefix = f"group_{group_id:04d}"
        arrays.update(
            {
                prefix + "_points": points,
                prefix + "_source_timestamps": timestamps,
                prefix + "_components": labels,
                prefix + "_core": core,
            }
        )
        arrays.update({prefix + "_" + k: v for k, v in counts.items()})
        row = dict(
            group_id=group_id,
            candidate_labels=group["candidate_labels"],
            source_points=len(points),
            source_timestamps=sorted(set(timestamps)),
            components=components,
            point_counts={
                k: int(v.sum()) for k, v in counts.items() if v.dtype == bool
            },
            positive_histogram=np.bincount(counts["positive"]).tolist(),
            negative_histogram=np.bincount(counts["negative"]).tolist(),
            candidates=candidates,
            selected_views=selected,
        )
        rows.append(row)
        print(
            json.dumps(
                {
                    k: row[k]
                    for k in (
                        "group_id",
                        "source_points",
                        "point_counts",
                        "selected_views",
                    )
                }
            ),
            flush=True,
        )
    names = list(dict.fromkeys(names))
    rescue = dict(plan)
    rescue.update(
        schema="farm.adaptive-surface-view-plan.v1",
        sources={n: plan["sources"][n] for n in names},
        timestamps=sorted({plan["sources"][n]["timestamp"] for n in names}),
        variants=[dict(name="balanced_upright", views=names)],
        parent_plan=describe_file(args.plan),
        source_geometry=describe_file(args.geometry),
        purpose="Additional registered development evidence for disconnected surface components; no mask or native ownership inferred by camera ranking.",
        selected_groups=rows,
    )
    write_json(args.output / "plan.json", rescue)
    # Person is a mandatory exclusion query, never an object-identity proposal.
    (args.output / "concepts.txt").write_text(
        "\n".join(dict.fromkeys(prompts + ["person"])) + "\n"
    )
    np.savez_compressed(args.output / "point_evidence.npz", **arrays)
    write_json(
        args.output / "manifest.json",
        dict(
            schema="farm.surface-evidence.v1",
            source_geometry=describe_file(args.geometry),
            source_plan=describe_file(args.plan),
            groups=rows,
            depth_artifacts=inputs.depth_artifacts,
            point_evidence=describe_file(args.output / "point_evidence.npz"),
            adaptive_plan=describe_file(args.output / "plan.json"),
            concepts=describe_file(args.output / "concepts.txt"),
            recovery_schedule=(
                dict(
                    policy="Single-timestamp; source/relative-size thirds round robin, static depth support descending; then registered visibility",
                    group_budget=args.auto_groups,
                    candidate_budget=args.candidate_budget,
                    view_budget=args.view_budget,
                    eligible_single_timestamp_groups=available,
                    candidates=candidate_rows,
                    screened=screened,
                    selected_group_ids=[r["group_id"] for r in rows],
                    selected_unique_rgb=len(names),
                    source_semantics_used=False,
                    segmentation_calls=0,
                )
                if automatic
                else None
            ),
            total_seconds=time.monotonic() - started,
            closed_test_opened=False,
            native_gaussian_ownership_changed=False,
            physical_extents_validated=False,
            release_eligible=False,
        ),
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
