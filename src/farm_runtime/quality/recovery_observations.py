"""Combine independently validated recovery observations before shared native lifting."""

from __future__ import annotations

import numpy as np


def combine_recovery_batches(batches, sources):
    if not 2 <= len(batches) <= 4 or len(batches) != len(sources):
        raise ValueError("two to four bound recovery batches required")
    inputs = max((b[0] for b in batches), key=lambda x: len(x.frames))
    for other, *_ in batches:
        if not other.frames.keys() <= inputs.frames.keys():
            raise ValueError(
                "one registered frame extension must cover every recovery batch"
            )
        for name, row in other.frames.items():
            current = inputs.frames[name]
            if any(
                row[k] != current[k]
                for k in (
                    "frame_id",
                    "timestamp_ns",
                    "camera",
                    "K",
                    "T_world_cam",
                    "depth_size",
                )
            ):
                raise ValueError("recovery batches disagree on a registered camera")
            for key in ("rgb_path", "depth_path"):
                if (other.frames_path.parent / row[key]).resolve() != (
                    inputs.frames_path.parent / current[key]
                ).resolve():
                    raise ValueError(
                        "recovery batches disagree on registered image/depth"
                    )
    observations, source_rows, extras, masks, group_ids = {}, {}, {}, {}, set()
    metadata = ("timestamp", "source_image", "grid_shape_hw", "applied_quarter_turns")
    for (other, validation, observed, rows, extra, extra_masks), source in zip(
        batches, sources
    ):
        if len(validation["groups"]) > 16:
            raise ValueError("individual recovery cohort exceeds 16 groups")
        confirmed = set()
        for group in validation["groups"]:
            gid = group["group_id"]
            values = [(name, o) for (oid, name), o in observed.items() if oid == gid]
            timestamps = {other.frames[name]["timestamp_ns"] for name, _ in values}
            if len(timestamps) >= 2 and any(
                o["source_kind"] == "validated_additional_view" for _, o in values
            ):
                confirmed.add(gid)
        group_ids |= confirmed
        for key, observation in observed.items():
            if key[0] not in confirmed:
                continue
            if key in observations:
                if not np.array_equal(observations[key]["mask"], observation["mask"]):
                    raise ValueError("conflicting recovery masks need joint validation")
                observations[key]["recovery_sources"].append(source)
            else:
                observations[key] = dict(observation, recovery_sources=[source])
            name = key[1]
            if name in source_rows and any(
                source_rows[name][k] != rows[name][k] for k in metadata
            ):
                raise ValueError("recovery source observation metadata changed")
            source_rows[name] = rows[name]
        # Preserve all independently observed person exclusions, including views
        # where a proposed object did not acquire an accepted match.
        for name, row in extra.items():
            if name in extras:
                if any(extras[name][k] != row[k] for k in metadata):
                    raise ValueError("recovery supplement frame/grid mismatch")
                extras[name] = dict(
                    row,
                    detections=extras[name]["detections"] + row["detections"],
                    queries=extras[name]["queries"] + row["queries"],
                )
                masks[name] = masks[name] + list(extra_masks[name])
            else:
                extras[name] = dict(row)
                masks[name] = list(extra_masks[name])
    if len(group_ids) > 16:
        raise ValueError("combined confirmed recovery cohort exceeds 16 groups")
    validation = dict(groups=[dict(group_id=g) for g in sorted(group_ids)])
    return inputs, validation, observations, source_rows, extras, masks
