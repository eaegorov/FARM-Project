"""Select a bounded additional view batch using observed geometric coverage."""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
import time

import numpy as np

from farm_runtime.quality.mask_refinement import checked_file
from farm_runtime.quality.surface_evidence import SurfaceInputs, rank_views
from farm_runtime.quality_baseline import describe_file, write_json


def choose_views(visibility, group_timestamps, frame_timestamps, budget):
    """Maximize marginal best visible fraction, favoring weakly observed groups.

    This schedules segmentation, not foreground votes. One frame per physical
    timestamp may be selected, and a group's existing timestamp adds no evidence.
    """
    if type(budget) is not int or not 1 <= budget <= 24:
        raise ValueError("view budget must be an integer from 1 to 24")
    if any(not timestamps for timestamps in group_timestamps.values()):
        raise ValueError("every group requires an observed timestamp")
    weights = {g: 1 / len(ts) for g, ts in group_timestamps.items()}
    candidates = {}
    for name, groups in visibility.items():
        timestamp = frame_timestamps[name]
        candidates[name] = {}
        for gid, fraction in groups.items():
            if (
                gid not in weights
                or not math.isfinite(fraction)
                or not 0 <= fraction <= 1
            ):
                raise ValueError("known groups and finite visibility in [0,1] required")
            if timestamp not in group_timestamps[gid]:
                candidates[name][gid] = fraction
    covered = dict.fromkeys(weights, 0.0)
    selected, used_timestamps = [], set()
    for _ in range(budget):
        gains = {
            name: sum(weights[g] * max(0, v - covered[g]) for g, v in groups.items())
            for name, groups in candidates.items()
            if frame_timestamps[name] not in used_timestamps
        }
        if not gains or max(gains.values()) <= 0:
            break
        name = max(gains, key=lambda n: (gains[n], n))
        groups = candidates[name]
        timestamp = frame_timestamps[name]
        selected.append(
            dict(
                name=name,
                timestamp=timestamp,
                marginal_score=gains[name],
                visible_groups=len(groups),
                improved_group_ids=sorted(
                    g for g, v in groups.items() if v > covered[g]
                ),
                visibility=groups,
            )
        )
        used_timestamps.add(timestamp)
        for gid, value in groups.items():
            covered[gid] = max(covered[gid], value)
    return selected, covered


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("geometry", "plan", "output"):
        parser.add_argument("--" + name, type=Path, required=True)
    parser.add_argument("--views", type=int, default=8)
    args = parser.parse_args(argv)
    if args.output.exists():
        raise ValueError("new output required")
    choose_views({}, {}, {}, args.views)  # Validate budget before input work.
    plan = json.loads(args.plan.read_text())
    if plan.get("test_opened") is not False:
        raise ValueError("development-only source plan required")
    inputs = SurfaceInputs(args.geometry)
    old_names = {f["source"] for f in inputs.geometry["frames"]}
    # Preserve the original preparation as the camera/timestamp authority.
    inputs.frames = {
        name: frame
        for name, frame in inputs.frames.items()
        if name not in old_names and name in plan["sources"]
    }
    for name, frame in inputs.frames.items():
        source = plan["sources"][name]
        if source["timestamp"] != str(frame["frame_id"]) or not np.allclose(
            source["camera_from_world_rotation"],
            np.asarray(frame["T_world_cam"])[:3, :3].T,
            rtol=0,
            atol=1e-6,
        ):
            raise ValueError("plan differs from registered timestamp/camera")
        checked_file(source["source_image"])
    started = time.monotonic()
    rows, by_view, group_timestamps = [], {}, {}
    for group in inputs.geometry["groups"]:
        members, points, timestamps, _ = inputs.support(group)
        source_timestamps = set(timestamps)
        if len(source_timestamps) != group["independent_timestamps"]:
            raise ValueError("geometry group timestamp count mismatch")
        group_timestamps[group["id"]] = source_timestamps
        candidates, _ = rank_views(
            points, np.ones(len(points), bool), source_timestamps, inputs, 1
        )
        rows.append(
            dict(
                group_id=group["id"],
                existing_timestamps=len(source_timestamps),
                candidate_labels=group["candidate_labels"],
                candidates=candidates,
            )
        )
        for candidate in candidates:
            by_view.setdefault(candidate["name"], {})[group["id"]] = candidate[
                "core_visible_fraction"
            ]
    selected, covered = choose_views(
        by_view,
        group_timestamps,
        {name: str(frame["frame_id"]) for name, frame in inputs.frames.items()},
        args.views,
    )
    names = [row["name"] for row in selected]
    adaptive = dict(plan)
    adaptive.update(
        schema="farm.adaptive-discovery-plan.v1",
        sources={n: plan["sources"][n] for n in names},
        variants=[dict(name="balanced_upright", views=names)],
        timestamps=sorted({r["timestamp"] for r in selected}),
        parent_plan=describe_file(args.plan),
        source_geometry=describe_file(args.geometry),
        selection=selected,
        purpose="Geometric visibility schedules new observations; segmentation and association still required.",
        test_opened=False,
    )
    args.output.mkdir(parents=True)
    write_json(args.output / "plan.json", adaptive)
    write_json(
        args.output / "manifest.json",
        dict(
            schema="farm.discovery-coverage.v1",
            source_geometry=describe_file(args.geometry),
            source_plan=describe_file(args.plan),
            adaptive_plan=describe_file(args.output / "plan.json"),
            group_candidates=rows,
            selected=selected,
            existing_groups=len(covered),
            groups_with_predicted_additional_coverage=sum(
                v > 0 for v in covered.values()
            ),
            total_seconds=time.monotonic() - started,
            policy=dict(
                maximum_new_views=args.views,
                group_weight="1 / original independent timestamps",
                coverage_objective="sum of marginal best visible fraction per group",
                candidate_gates="Existing rank_views: >=20 points and >=0.25 visible fraction",
                transient_handling="New images require person segmentation before supplying evidence.",
            ),
            status=(
                "additional_views_selected"
                if selected
                else "no_remaining_covisible_views"
            ),
            closed_test_opened=False,
            release_eligible=False,
        ),
    )
    print(
        json.dumps(
            dict(
                selected_views=len(selected),
                predicted_groups=sum(v > 0 for v in covered.values()),
            )
        ),
        flush=True,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
