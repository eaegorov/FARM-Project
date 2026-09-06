"""Bounded recovery from exact contributors when mean-depth search has no claims."""

from __future__ import annotations

from collections import defaultdict
from dataclasses import replace
import time

import numpy as np

from tools.farm_shaper_bridge import gaussian_lift as lift
from tools.farm_shaper_bridge.common import project_metric

MAXIMUM_VIEWS = 8
MAXIMUM_TIMESTAMPS = 4
MAXIMUM_CANDIDATES = 65536
DEPTH_SIGMA_MULTIPLIER = 2.0


def depth_spread_gate(gaussians, item, frame, depth, sigma, config):
    """Use measured depth-mixture spread; outside/invalid pixels stay unknown."""
    if sigma.shape != frame.depth_size or depth.shape != frame.depth_size:
        raise ValueError("depth and spread must use the native frame grid")
    u, v, z = project_metric(
        gaussians.means_m[item.indices], frame.T_world_cam, frame.K
    )
    ui, vi = np.rint(u).astype(np.int64), np.rint(v).astype(np.int64)
    h, w = frame.depth_size
    policy = config["candidate"]
    inside = (
        (z > float(policy["depth_absolute_tolerance_m"]))
        & (ui >= 0)
        & (ui < w)
        & (vi >= 0)
        & (vi < h)
    )
    selected = np.flatnonzero(inside)
    result = np.zeros(len(item.indices), bool)
    reference = depth[vi[selected], ui[selected]]
    spread = sigma[vi[selected], ui[selected]]
    tolerance = np.maximum.reduce(
        [
            float(policy["depth_absolute_tolerance_m"])
            + float(policy["depth_relative_tolerance"]) * reference,
            float(policy["radius_tolerance_multiplier"]) * item.radius_m[selected],
            DEPTH_SIGMA_MULTIPLIER * spread,
        ]
    )
    residual = z[selected] - reference
    compatible = (
        residual <= tolerance
        if policy.get("depth_gate_policy", "surface") == "not_behind"
        else np.abs(residual) <= tolerance
    )
    valid = (
        np.isfinite(reference)
        & np.isfinite(spread)
        & (spread >= 0)
        & (reference >= float(policy["depth_absolute_tolerance_m"]))
        & compatible
    )
    result[selected[valid]] = True
    return result


def _measure_view(gaussians, run, frame, object_id, config):
    """Keep the original candidate-specific foreground/background pixel policy."""
    import torch
    from gsplat import rasterization

    by_object = defaultdict(list)
    for obj in run.objects:
        for observation in obj.observations:
            if observation.image_id == frame.image_id:
                by_object[obj.object_id].append(observation)
    decoded, surface, masks, depth_source = lift._load_view_masks(
        run, frame, by_object, config
    )
    target = decoded[object_id]
    weights = np.stack(
        [target["positive_weight"], target["negative_weight"], surface], axis=-1
    )
    gradient, alpha, invariant = lift.contributor_vjp(
        gaussians, run, frame, weights, float(config["render"]["invariant_tolerance"])
    )
    contributions = gradient.float().cpu().numpy()
    del gradient, alpha, weights
    view, intrinsic, height, width = lift.camera_tensors(run, frame)
    z = (gaussians.means @ view[0, 2, :3] + view[0, 2, 3]) * run.meters_per_scene_unit
    with torch.inference_mode():
        moments, alpha, _ = rasterization(
            **lift.raster_kwargs(gaussians, run, view, intrinsic, height, width),
            colors=z.square()[:, None],
            sh_degree=None,
            render_mode="RGB+ED",
        )
    values = moments[0].float().cpu().numpy()
    opacity = alpha[0, ..., 0].float().cpu().numpy()
    expected = values[..., -1] * run.meters_per_scene_unit
    sigma = np.sqrt(
        np.maximum(values[..., 0] / np.maximum(opacity, 1e-8) - expected**2, 0)
    )
    valid = (
        np.isfinite(expected)
        & (expected >= float(run.rgbd_config["depth_min_m"]))
        & (expected <= float(run.rgbd_config["depth_max_m"]))
        & (opacity >= float(run.rgbd_config["alpha_min"]))
    )
    depth = np.load(frame.depth_path, mmap_mode="r")
    delta = float(np.max(np.abs(np.where(valid, expected, 0) - depth)))
    if delta > float(config["render"]["alignment_guard_max_absolute_depth_error_m"]):
        raise ValueError("depth moments do not reproduce the registered native depth")
    audit = dict(
        image_id=frame.image_id,
        physical_timestamp=frame.physical_timestamp,
        depth=depth_source,
        masks=masks,
        feature_alpha_max_error=invariant,
        depth_max_absolute_error_m=delta,
    )
    return contributions, sigma, depth, audit


def _aggregate(item, measurements, gaussians, config):
    """Max within a physical timestamp, then the same purity/voting as the lift."""
    grouped = defaultdict(list)
    for frame, values, sigma, depth, audit in measurements:
        selected = values[item.indices].copy()
        gate = depth_spread_gate(gaussians, item, frame, depth, sigma, config)
        selected[~gate] = 0
        audit["depth_spread_compatible_candidates"] = int(gate.sum())
        grouped[frame.physical_timestamp].append(selected)
    policy = config["build"]
    for views in grouped.values():
        positive, negative, visible = np.maximum.reduce(views).T
        denominator = np.maximum(positive + negative, positive + 1e-8)
        item.positive_timestamps += (
            positive >= float(policy["minimum_positive_weight"])
        ) & (positive / denominator >= float(policy["minimum_per_timestamp_purity"]))
        item.negative_timestamps += (
            negative >= float(policy["minimum_negative_weight"])
        ) & (negative / denominator >= float(policy["minimum_per_timestamp_purity"]))
        item.positive_weight += positive
        item.negative_weight += negative
        item.visible_weight += visible


def recover_contributor_candidates(run, gaussians, split, evidence, config, rows):
    """Replace only failed spatial evidence, before full-scene claim resolution."""
    budget = config["candidate"].get("rendered_recovery_budget", 0)
    if type(budget) is not int or not 0 <= budget <= 8:
        raise ValueError("rendered recovery budget must be in 0..8")
    started = time.monotonic()
    report = dict(
        enabled=bool(budget),
        maximum_objects=budget,
        maximum_observations_per_object=MAXIMUM_VIEWS,
        maximum_timestamps_per_object=MAXIMUM_TIMESTAMPS,
        maximum_candidates_per_object=MAXIMUM_CANDIDATES,
        depth_sigma_multiplier=DEPTH_SIGMA_MULTIPLIER,
        interpretation="Measured compositing depth spread, not a calibrated confidence interval",
        objects=[],
        updated_object_ids=[],
        model_calls=0,
        additional_vjp_calls=0,
        closed_test_opened=False,
    )
    if not budget:
        report["total_seconds"] = time.monotonic() - started
        return report
    if (
        config["mask"].get("negative_domain") != "visible_background"
        or config["mask"].get("other_mask_negative_policy") != "background"
    ):
        raise ValueError(
            "contributor recovery requires candidate-specific visible background"
        )
    allowed = set(split["build_timestamps"])
    if allowed & set(split["heldout_timestamps"]):
        raise ValueError("build and heldout timestamps must be disjoint")
    object_build = {
        row["object_id"]: set(row["build_timestamps"]) for row in split["objects"]
    }
    eligible = sorted(
        (r for r in rows if r["supported_claims_before_conflicts"] == 0),
        key=lambda r: (-r["build_timestamp_count"], r["object_id"]),
    )
    report["eligible_object_ids"] = [r["object_id"] for r in eligible]
    attempts = 0
    for row in eligible:
        oid = row["object_id"]
        old = evidence[oid]
        frames = [
            run.frame(obs.image_id)
            for obs in old.obj.observations
            if run.frame(obs.image_id).physical_timestamp in allowed & object_build[oid]
        ]
        if len({f.image_id for f in frames}) != len(frames):
            raise ValueError("unique object/frame observations required")
        times = {f.physical_timestamp for f in frames}
        audit = dict(
            object_id=oid,
            original_candidates=len(old.indices),
            observations=len(frames),
            independent_timestamps=len(times),
        )
        report["objects"].append(audit)
        if not 2 <= len(times) <= MAXIMUM_TIMESTAMPS or len(frames) > MAXIMUM_VIEWS:
            audit["status"] = "deferred_view_budget_or_insufficient_timestamps"
            continue
        if attempts >= budget:
            audit["status"] = "deferred_object_budget"
            continue
        attempts += 1
        measurements, ids = [], old.indices.copy()
        for frame in sorted(frames, key=lambda f: f.image_id):
            values, sigma, depth, view_audit = _measure_view(
                gaussians, run, frame, oid, config
            )
            report["additional_vjp_calls"] += 1
            positive_ids = np.flatnonzero(
                values[:, 0] >= float(config["build"]["minimum_positive_weight"])
            )
            ids = np.union1d(ids, positive_ids)
            ids = ids[
                gaussians.opacity_cpu[ids]
                >= float(config["candidate"]["minimum_opacity"])
            ]
            if len(ids) > MAXIMUM_CANDIDATES:
                audit["status"] = "deferred_candidate_budget"
                break
            measurements.append((frame, values, sigma, depth, view_audit))
        if "status" in audit:
            continue
        count = len(ids)
        item = replace(
            old,
            indices=ids,
            build_timestamp_count=len(times),
            radius_m=np.minimum(
                gaussians.radius_m[ids], float(config["candidate"]["maximum_radius_m"])
            ),
            positive_weight=np.zeros(count, np.float32),
            negative_weight=np.zeros(count, np.float32),
            visible_weight=np.zeros(count, np.float32),
            positive_timestamps=np.zeros(count, np.uint16),
            negative_timestamps=np.zeros(count, np.uint16),
        )
        _aggregate(item, measurements, gaussians, config)
        local_rows = lift.make_claims({oid: item}, gaussians.count, config)[3]
        claims = local_rows[0]["supported_claims_before_conflicts"]
        audit.update(
            candidate_gaussians=count,
            supported_claims=claims,
            views=[r[-1] for r in measurements],
            status=(
                "updated_for_global_competition" if claims else "no_supported_claims"
            ),
        )
        if claims:
            evidence[oid] = item
            report["updated_object_ids"].append(oid)
    report["total_seconds"] = time.monotonic() - started
    return report
