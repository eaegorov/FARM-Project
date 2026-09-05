"""Compare source photographs with the source 3DGS RGB at identical cameras.

Saved-depth parity alone checks renderer reproducibility, not RGB alignment.
This diagnostic keeps those two questions separate.
"""

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


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__)
    for key in ("run", "ply", "view-manifest", "output"):
        p.add_argument("--" + key, type=Path, required=True)
    p.add_argument("--propose-camera", action="store_true")
    args = p.parse_args(argv)
    if args.output.exists():
        raise ValueError("output must be new")
    manifest = json.loads(args.view_manifest.read_text())
    rows = manifest["views"]
    run = load_run(args.run, allow_legacy=True)
    if manifest.get("camera_refinement"):
        from farm_runtime.quality.camera_refinement import apply_camera_refinement

        run = apply_camera_refinement(run, Path(manifest["camera_refinement"]["path"]))
    table = open_graphdeco_ply(args.ply)
    gs = lift.load_gaussians(table, run.meters_per_scene_unit)
    import torch
    from gsplat import rasterization

    dc = np.column_stack([table.data[f"f_dc_{i}"] for i in range(3)]).astype(np.float32)
    rest = np.column_stack([table.data[f"f_rest_{i}"] for i in range(45)]).astype(
        np.float32
    )
    rest = rest.reshape(table.count, 3, 15).transpose(0, 2, 1)
    colors = torch.from_numpy(np.concatenate([dc[:, None, :], rest], axis=1)).cuda()
    del dc, rest
    args.output.mkdir(parents=True)
    by_name = {f.source_image: f for f in run.frames}
    audit = []
    for i, row in enumerate(rows):
        frame = by_name[row["source_image"]]
        view, K, h, w = lift.camera_tensors(run, frame)
        with torch.inference_mode():
            rgb, alpha, _ = rasterization(
                **lift.raster_kwargs(gs, run, view, K, h, w),
                colors=colors,
                sh_degree=3,
                render_mode="RGB+ED",
            )
        rendered = np.clip(rgb[0, ..., :3].cpu().numpy() * 255, 0, 255).astype(np.uint8)
        source = np.array(Image.open(frame.rgb_path).convert("RGB").resize((w, h)))
        proposal = None
        corrected = rendered
        if args.propose_camera:
            from farm_runtime.camera_alignment import (
                match_render_to_source,
                propose_camera_pose,
            )

            depth = rgb[0, ..., 3].cpu().numpy() * run.meters_per_scene_unit
            points, pixels, matches = match_render_to_source(
                rendered,
                source,
                depth,
                alpha[0, ..., 0].cpu().numpy(),
                frame.K,
                frame.T_world_cam,
            )
            pose, proposal = propose_camera_pose(
                points, pixels, frame.K, frame.T_world_cam, frame.depth_size
            )
            proposal.update(matches)
            np.savez_compressed(
                args.output / f"correspondences_{i:02d}.npz",
                points_world_m=points,
                pixels=pixels,
                K=frame.K,
                initial_pose=frame.T_world_cam,
            )
            if proposal["accepted"]:
                adjusted = replace(frame, T_world_cam=pose)
                view2, K2, h2, w2 = lift.camera_tensors(run, adjusted)
                with torch.inference_mode():
                    refined, _, _ = rasterization(
                        **lift.raster_kwargs(gs, run, view2, K2, h2, w2),
                        colors=colors,
                        sh_degree=3,
                        render_mode="RGB",
                    )
                corrected = np.clip(refined[0].cpu().numpy() * 255, 0, 255).astype(
                    np.uint8
                )
            print(json.dumps({"frame": frame.source_image, **proposal}), flush=True)
        x0, y0, x1, y1 = row["crop_xyxy"]
        left = source[y0:y1, x0:x1]
        right = rendered[y0:y1, x0:x1]
        fixed = corrected[y0:y1, x0:x1]
        canvas = Image.new("RGB", (1500, 560), "#111827")
        draw = ImageDraw.Draw(canvas)
        draw.text(
            (10, 10), f"Object {row['object_id']} / {frame.source_image}", fill="white"
        )
        panels = (
            (
                (left, "Source RGB"),
                (right, "3DGS RGB, original camera"),
                (fixed, "3DGS RGB, bounded PnP proposal"),
            )
            if args.propose_camera
            else (
                (left, "Source RGB"),
                (right, "3DGS RGB, identical camera, SH degree 3"),
                (
                    (left.astype(float) * 0.5 + right.astype(float) * 0.5).astype(
                        np.uint8
                    ),
                    "50/50 blend: double edges expose disagreement",
                ),
            )
        )
        for col, (arr, label) in enumerate(panels):
            scale = min(490 / arr.shape[1], 480 / arr.shape[0])
            tile = Image.fromarray(arr).resize(
                (round(arr.shape[1] * scale), round(arr.shape[0] * scale)),
                Image.Resampling.BILINEAR,
            )
            canvas.paste(
                tile,
                (col * 500 + (490 - tile.width) // 2, 45 + (480 - tile.height) // 2),
            )
            draw.text((col * 500 + 8, 536), label, fill="white")
        file = args.output / f"object_{row['object_id']:06d}_{i:02d}.jpg"
        canvas.save(file, quality=93)
        audit.append(
            {
                **row,
                "image": file.name,
                "camera_proposal": proposal,
                "mean_absolute_rgb_error_0_255": float(
                    np.mean(np.abs(left.astype(float) - right.astype(float)))
                ),
                "corrected_mean_absolute_rgb_error_0_255": float(
                    np.mean(np.abs(left.astype(float) - fixed.astype(float)))
                ),
            }
        )
    write_json(
        args.output / "report.json",
        {
            "views": audit,
            "reserved_test_opened": False,
            "automatic_alignment_pass": None,
            "interpretation": "photometric diagnostic; neither renderer depth parity nor high mask IoU establishes source-image alignment",
        },
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
