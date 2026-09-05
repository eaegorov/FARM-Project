"""Compare discovery proposals in source pixels and show changed observations.

Proposal agreement measures preservation, not accuracy or scene recall.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw
from scipy.optimize import linear_sum_assignment

from farm_runtime.angular_discovery import rotate_image
from farm_runtime.quality_baseline import describe_file, write_json


def masks_for(observation, directory):
    h, w = observation["grid_shape_hw"]
    path = (
        directory
        / "balanced_upright/masks"
        / Path(observation["mask_artifact"]["path"]).name
    )
    if describe_file(path)["sha256"] != observation["mask_artifact"]["sha256"]:
        raise ValueError("proposal mask artifact changed")
    masks = []
    with np.load(path, allow_pickle=False) as archive:
        for detection in observation["detections"]:
            x0, y0, x1, y1 = detection["grid_window_xyxy"]
            mask = np.zeros((h, w), bool)
            mask[y0:y1, x0:x1] = archive[detection["logit_key"]] > 0
            masks.append(mask)
    return masks


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    for field in ("baseline", "candidate", "output"):
        parser.add_argument("--" + field, type=Path, required=True)
    args = parser.parse_args(argv)
    if args.output.exists():
        raise ValueError("review output must be new")
    plans = [
        json.loads((d / "plan.json").read_text())
        for d in (args.baseline, args.candidate)
    ]
    if plans[0] != plans[1] or plans[0].get("test_opened") is not False:
        raise ValueError("identical development input plans required")
    left, right = [
        json.loads((d / "balanced_upright/predictions.json").read_text())[
            "observations"
        ]
        for d in (args.baseline, args.candidate)
    ]
    if [r["name"] for r in left] != [r["name"] for r in right]:
        raise ValueError("observation order differs")
    args.output.mkdir(parents=True)
    events, rows, cards = [], [], []
    for a, b in zip(left, right):
        if (
            a["grid_shape_hw"] != b["grid_shape_hw"]
            or a["source_image"] != b["source_image"]
        ):
            raise ValueError("proposal grids or source pixels differ")
        ma, mb = masks_for(a, args.baseline), masks_for(b, args.candidate)
        iou = np.array(
            [
                [np.count_nonzero(x & y) / max(1, np.count_nonzero(x | y)) for y in mb]
                for x in ma
            ]
        ).reshape(len(ma), len(mb))
        ii, jj = linear_sum_assignment(-iou)
        pairs = {int(j): int(i) for i, j in zip(ii, jj)}
        rows.append(
            dict(
                source=a["name"],
                baseline_count=len(ma),
                candidate_count=len(mb),
                matches=[
                    dict(baseline=int(i), candidate=int(j), iou=float(iou[i, j]))
                    for i, j in zip(ii, jj)
                ],
            )
        )
        selected = []
        for j, detection in enumerate(b["detections"]):
            i = pairs.get(j)
            same = i is not None and iou[i, j] >= 0.95
            label_changed = same and a["detections"][i]["label"] != detection["label"]
            if not same or label_changed:
                selected.append(
                    (i, j, "label_changed" if label_changed else "new_or_changed_mask")
                )
        for i in range(len(ma)):
            if i not in pairs.values():
                selected.append((i, None, "removed_mask"))
        if not selected:
            continue
        path = Path(a["source_image"]["path"])
        if describe_file(path)["sha256"] != a["source_image"]["sha256"]:
            raise ValueError("RGB source changed")
        h, w = a["grid_shape_hw"]
        with Image.open(path) as im:
            rgb = np.asarray(im.convert("RGB"))
        turns = a["applied_quarter_turns"]
        for i, j, kind in selected:
            chosen = mb[j] if j is not None else ma[i]
            yy, xx = np.where(rotate_image(chosen, turns))
            upright = rotate_image(rgb, turns)
            uh, uw = rotate_image(chosen, turns).shape
            full_h, full_w = upright.shape[:2]
            pad = max(15, round(0.2 * max(np.ptp(xx), np.ptp(yy))))
            x0, x1 = max(0, int(xx.min()) - pad), min(uw, int(xx.max()) + pad + 1)
            y0, y1 = max(0, int(yy.min()) - pad), min(uh, int(yy.max()) + pad + 1)
            bounds = (
                round(x0 * full_w / uw),
                round(y0 * full_h / uh),
                round(x1 * full_w / uw),
                round(y1 * full_h / uh),
            )
            fx0, fy0, fx1, fy1 = bounds
            la = a["detections"][i]["label"] if i is not None else "none"
            lb = b["detections"][j]["label"] if j is not None else "none"
            card = Image.new("RGB", (960, 350), "#111827")
            draw = ImageDraw.Draw(card)
            draw.text((8, 6), f"{a['name']} | {kind} | {la} -> {lb}", fill="white")
            for col, mask in enumerate(
                (
                    None,
                    ma[i] if i is not None else None,
                    mb[j] if j is not None else None,
                )
            ):
                panel = upright[fy0:fy1, fx0:fx1].copy()
                if mask is not None:
                    crop_mask = rotate_image(mask, turns)[y0:y1, x0:x1]
                    m = np.asarray(
                        Image.fromarray(crop_mask).resize(
                            (panel.shape[1], panel.shape[0]), Image.Resampling.NEAREST
                        )
                    )
                    panel[m] = (panel[m] * 0.6 + np.array([35, 205, 255]) * 0.4).astype(
                        np.uint8
                    )
                tile = Image.fromarray(panel)
                factor = min(312 / tile.width, 296 / tile.height)
                tile = tile.resize(
                    (
                        max(1, round(tile.width * factor)),
                        max(1, round(tile.height * factor)),
                    ),
                    Image.Resampling.LANCZOS,
                )
                card.paste(
                    tile,
                    (
                        col * 320 + (312 - tile.width) // 2,
                        44 + (296 - tile.height) // 2,
                    ),
                )
                draw.text(
                    (col * 320 + 8, 26),
                    ("RGB", "baseline", "candidate")[col],
                    fill="#fbbf24",
                )
            events.append(
                dict(
                    source=a["name"],
                    kind=kind,
                    baseline=i,
                    candidate=j,
                    baseline_label=la,
                    candidate_label=lb,
                    paired_iou=None if i is None or j is None else float(iou[i, j]),
                    card_index=len(cards),
                )
            )
            cards.append(card)
    sheets = []
    for offset in range(0, len(cards), 4):
        chunk = cards[offset : offset + 4]
        canvas = Image.new("RGB", (960, 350 * len(chunk)), "#111827")
        for index, card in enumerate(chunk):
            canvas.paste(card, (0, index * 350))
        path = args.output / f"changes_{offset//4:02d}.jpg"
        canvas.save(path, quality=94)
        sheets.append(describe_file(path))
    write_json(
        args.output / "manifest.json",
        dict(
            schema="farm.discovery-proposal-review.v1",
            rows=rows,
            events=events,
            sheets=sheets,
            baseline=describe_file(args.baseline / "results.json"),
            candidate=describe_file(args.candidate / "results.json"),
            gold=False,
            manual_recall_measured=False,
            test_opened=False,
        ),
    )
    print(json.dumps(dict(changes=len(events), sheets=len(sheets))))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
