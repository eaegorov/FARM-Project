"""Visualize native Gaussian additions and their actual reverse-rendered masks."""

from __future__ import annotations

import argparse
from dataclasses import replace
import json
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw

from farm_runtime.quality_baseline import write_json
from tools.farm_shaper_bridge import gaussian_lift as lift
from tools.farm_shaper_bridge.common import load_run, open_graphdeco_ply


def read_bank(path):
    with np.load(path) as bank:
        return {
            int(oid): bank["indices"][bank["indptr"][i] : bank["indptr"][i + 1]].copy()
            for i, oid in enumerate(bank["object_ids"])
        }


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__)
    for key in ("run", "ply", "ablation", "config", "output"):
        p.add_argument("--" + key, type=Path, required=True)
    p.add_argument(
        "--review-catalog",
        type=Path,
        help="Reviewed labels and native-support OBBs for display",
    )
    p.add_argument(
        "--baseline-ablation",
        type=Path,
        help="Compare the same domain before/after a development change",
    )
    p.add_argument(
        "--domain", choices=("obb", "obb_mask_surface"), default="obb_mask_surface"
    )
    args = p.parse_args(argv)
    if args.output.exists():
        raise ValueError("output must be new")
    args.output.mkdir(parents=True)
    config, _ = lift.load_config(args.config)
    run = load_run(args.run, allow_legacy=True)
    comparison = json.loads((args.ablation / "comparison.json").read_text())
    camera_refinement = comparison.get("camera_refinement")
    if camera_refinement:
        from farm_runtime.quality.camera_refinement import apply_camera_refinement

        run = apply_camera_refinement(run, Path(camera_refinement["path"]))
    if args.baseline_ablation:
        baseline_comparison = json.loads(
            (args.baseline_ablation / "comparison.json").read_text()
        )
        for field in ("split", "source_ply", "camera_refinement"):
            if baseline_comparison[field] != comparison[field]:
                raise ValueError(
                    "before/after comparison must use identical split, source and cameras"
                )
        before = read_bank(args.baseline_ablation / args.domain / "proposal_bank.npz")
        after = read_bank(args.ablation / args.domain / "proposal_bank.npz")
        names = ("Before mask refinement", "After mask refinement")
    else:
        before, after = [
            read_bank(args.ablation / d / "proposal_bank.npz")
            for d in ("obb", "obb_mask_surface")
        ]
        names = ("OBB candidates", "Surface union")
    run = replace(
        run,
        objects=tuple(
            o for o in run.objects if o.object_id in before or o.object_id in after
        ),
    )
    reviewed = {}
    if args.review_catalog:
        reviewed = {
            int(o["object_id"]): o
            for o in json.loads(args.review_catalog.read_text())["objects"]
        }
    source = open_graphdeco_ply(args.ply)
    gs = lift.load_gaussians(source, run.meters_per_scene_unit)
    qc = json.loads(
        (
            args.ablation
            / (args.domain if args.baseline_ablation else "obb_mask_surface")
            / "report.json"
        ).read_text()
    )["heldout_qc"]
    audit = []
    for obj in run.objects:
        ids_a, ids_b = before.get(obj.object_id, np.empty(0, np.int64)), after.get(
            obj.object_id, np.empty(0, np.int64)
        )
        added = np.setdiff1d(ids_b, ids_a)
        removed = np.setdiff1d(ids_a, ids_b)
        rows = next(r for r in qc if r["object_id"] == obj.object_id)["views"]
        # Show two distinct legacy heldout timestamps, not two virtual views of one pose.
        chosen, timestamps = [], set()
        for row in sorted(rows, key=lambda r: r["recall"]):
            if row["physical_timestamp_ns"] not in timestamps:
                chosen.append(row)
                timestamps.add(row["physical_timestamp_ns"])
            if len(chosen) == 2:
                break
        for rank, row in enumerate(chosen):
            frame = run.frame(row["image_id"])
            obs = [o for o in obj.observations if o.image_id == frame.image_id]
            decoded, _, _, _ = lift._load_view_masks(
                run, frame, {obj.object_id: obs}, config
            )
            target = decoded[obj.object_id]["raw"]
            a = lift.reverse_render(
                gs, run, frame, [obj.object_id], {obj.object_id: ids_a}
            )[0][..., 0]
            b = lift.reverse_render(
                gs, run, frame, [obj.object_id], {obj.object_id: ids_b}
            )[0][..., 0]
            h, w = target.shape
            rgb = np.array(Image.open(frame.rgb_path).convert("RGB").resize((w, h)))
            foreground = target | (a >= 0.1) | (b >= 0.1)
            ys, xs = np.where(foreground)
            if not len(xs):
                continue
            x0, x1 = max(0, int(xs.min()) - 25), min(w, int(xs.max()) + 26)
            y0, y1 = max(0, int(ys.min()) - 25), min(h, int(ys.max()) + 26)
            canvas = Image.new("RGB", (1500, 590), "#111827")
            draw = ImageDraw.Draw(canvas)
            draw.text(
                (15, 12),
                f"Object {obj.object_id} / {reviewed.get(obj.object_id, {}).get('label', obj.category)} / {frame.source_image}",
                fill="white",
            )
            for col, (mask, label) in enumerate(
                (
                    (target, "Legacy 2D reference (not human gold)"),
                    (a >= 0.1, f"{names[0]}: {len(ids_a)} Gaussians"),
                    (
                        b >= 0.1,
                        f"{names[1]}: {len(ids_b)}; +{len(added)} / -{len(removed)}",
                    ),
                )
            ):
                image = rgb.copy()
                image[mask] = (
                    image[mask] * 0.5 + np.array([70, 210, 255]) * 0.5
                ).astype(np.uint8)
                if col == 2:
                    growth = (b >= 0.1) & ~(a >= 0.1)
                    image[growth] = (
                        rgb[growth] * 0.35 + np.array([50, 255, 90]) * 0.65
                    ).astype(np.uint8)
                tile = Image.fromarray(image[y0:y1, x0:x1])
                scale = min(490 / tile.width, 500 / tile.height)
                tile = tile.resize(
                    (round(tile.width * scale), round(tile.height * scale)),
                    Image.Resampling.BILINEAR,
                )
                canvas.paste(
                    tile,
                    (
                        500 * col + (490 - tile.width) // 2,
                        55 + (500 - tile.height) // 2,
                    ),
                )
                draw.text((500 * col + 10, 565), label, fill="white")
            file = args.output / f"object_{obj.object_id:06d}_view_{rank}.jpg"
            canvas.save(file, quality=93)
            audit.append(
                {
                    "object_id": obj.object_id,
                    "image": file.name,
                    "frame_id": frame.frame_id,
                    "source_image": frame.source_image,
                    "crop_xyxy": [x0, y0, x1, y1],
                    "reference": "legacy heldout automatic mask",
                }
            )
        # Object-frame orthographic projections of actual source centers.
        local_a = (gs.means_m[ids_a] - obj.center_m) @ obj.rotation
        local_added = (gs.means_m[added] - obj.center_m) @ obj.rotation
        local_removed = (gs.means_m[removed] - obj.center_m) @ obj.rotation
        canvas = Image.new("RGB", (1500, 550), "#111827")
        draw = ImageDraw.Draw(canvas)
        draw.text(
            (12, 10),
            f"Object {obj.object_id}: native Gaussian centers; gray=baseline, green=added, red=removed; orange=initial OBB; cyan=reviewed OBB",
            fill="white",
        )
        for col, axes in enumerate(((0, 1), (0, 2), (1, 2))):
            points = np.concatenate((local_a, local_added))[:, axes]
            lo = np.minimum(points.min(axis=0), -obj.dimensions_m[list(axes)] / 2)
            hi = np.maximum(points.max(axis=0), obj.dimensions_m[list(axes)] / 2)
            scale = 420 / max(float((hi - lo).max()), 1e-6)

            def project(p):
                q = (p[:, axes] - (lo + hi) / 2) * scale
                return np.column_stack((col * 500 + 250 + q[:, 0], 285 - q[:, 1]))

            for cloud, color in (
                (local_a, "#777777"),
                (local_added, "#32ff5a"),
                (local_removed, "#ff625f"),
            ):
                for x, y in project(cloud):
                    draw.point((int(x), int(y)), fill=color)
            dims = obj.dimensions_m[list(axes)] * 0.5
            pts = np.zeros((2, 3))
            pts[:, list(axes)] = np.array([-dims, dims])
            box = project(pts)
            draw.rectangle(
                (box[:, 0].min(), box[:, 1].min(), box[:, 0].max(), box[:, 1].max()),
                outline="#fbbf24",
                width=2,
            )
            if obj.object_id in reviewed:
                obb = reviewed[obj.object_id]["obb"]
                center = (np.asarray(obb["center_m"]) - obj.center_m) @ obj.rotation
                dims_new = np.asarray(obb["dimensions_m"])[list(axes)] / 2
                # Exported OBB preserves the prior orientation; keep this explicit.
                if not np.allclose(obb["rotation_world_from_box"], obj.rotation):
                    raise ValueError("review OBB orientation differs from plotted axes")
                corners = np.tile(center, (2, 1))
                corners[:, list(axes)] += np.array([-dims_new, dims_new])
                q = project(corners)
                draw.rectangle(
                    (q[:, 0].min(), q[:, 1].min(), q[:, 0].max(), q[:, 1].max()),
                    outline="#46d2ff",
                    width=2,
                )
            draw.text(
                (col * 500 + 10, 520),
                f"axes {axes}; initial extents {2*dims[0]:.3f} x {2*dims[1]:.3f} m",
                fill="white",
            )
        canvas.save(
            args.output / f"object_{obj.object_id:06d}_native_3d.jpg", quality=93
        )
    write_json(
        args.output / "manifest.json",
        {
            "views": audit,
            "review_catalog": str(args.review_catalog) if args.review_catalog else None,
            "camera_refinement": camera_refinement,
            "baseline_ablation": (
                str(args.baseline_ablation) if args.baseline_ablation else None
            ),
            "domain": args.domain if args.baseline_ablation else "obb_vs_surface",
            "reserved_test_opened": False,
            "release_eligible": False,
        },
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
