"""Review saved native memberships against source masks without rebuilding them."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import time
from types import SimpleNamespace

import numpy as np
from PIL import Image, ImageDraw

from farm_runtime.angular_discovery import rotate_image
from farm_runtime.quality.mask_refinement import checked_file
from farm_runtime.quality.native_observations import load_prepared
from farm_runtime.quality_baseline import describe_file, write_json
from tools.farm_shaper_bridge import gaussian_lift as lift
from tools.farm_shaper_bridge.common import open_graphdeco_ply, metric_c2w_to_scene_w2c


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("input", "report", "config", "ply", "output"):
        parser.add_argument("--" + name, type=Path, required=True)
    parser.add_argument("--with-scene-render", action="store_true")
    args = parser.parse_args(argv)
    if args.output.exists():
        raise ValueError("new output required")
    started = time.monotonic()
    run, split, manifest = load_prepared(args.input)
    role = manifest.get("observation_role", "build")
    report = json.loads(args.report.read_text())
    bank_path = checked_file(report["bank"])
    with np.load(bank_path, allow_pickle=False) as bank:
        members = {
            int(oid): bank["indices"][bank["indptr"][i] : bank["indptr"][i + 1]]
            for i, oid in enumerate(bank["object_ids"])
        }
        supported = {
            int(oid): bank["indices"][bank["indptr"][i] : bank["indptr"][i + 1]][
                bank["timestamp_support"][bank["indptr"][i] : bank["indptr"][i + 1]]
                >= 2
            ]
            for i, oid in enumerate(bank["object_ids"])
        }
    config, _ = lift.load_config(args.config)
    if describe_file(args.ply)["sha256"] != manifest["source_ply"]["sha256"]:
        raise ValueError("source PLY changed")
    gs = lift.load_gaussians(open_graphdeco_ply(args.ply), run.meters_per_scene_unit)
    scene = None
    scene_cache, scene_parity = {}, []
    if args.with_scene_render:
        from scripts.prepare_colmap_3dgs_rgbd import (
            load_gaussians as load_scene,
            render_gaussians,
        )

        scene = load_scene(args.ply, "cuda", int(run.rgbd_config["sh_degree"]))
    args.output.mkdir(parents=True)
    frames = {r["image_id"]: r for r in manifest["frames"]}
    visuals = []
    counts = []
    for obj in run.objects:
        if obj.object_id not in members:
            continue
        counts.append(
            dict(
                object_id=obj.object_id,
                full=len(members[obj.object_id]),
                at_least_two_positive_timestamps=len(supported[obj.object_id]),
            )
        )
        for obs in obj.observations:
            frame = run.frame(obs.image_id)
            decoded, _, _, _ = lift._load_view_masks(
                run, frame, {obj.object_id: [obs]}, config
            )
            raw = decoded[obj.object_id]["raw"]
            full = lift.reverse_render(gs, run, frame, [obj.object_id], members)[0][
                ..., 0
            ]
            core = lift.reverse_render(gs, run, frame, [obj.object_id], supported)[0][
                ..., 0
            ]
            masks = [
                raw,
                core >= config["heldout"]["alpha_threshold"],
                full >= config["heldout"]["alpha_threshold"],
            ]
            region = np.logical_or.reduce(masks)
            yy, xx = np.where(region)
            if not len(xx):
                continue
            pad = max(12, round(0.08 * max(np.ptp(xx), np.ptp(yy))))
            h, w = raw.shape
            x0, x1 = max(0, int(xx.min()) - pad), min(w, int(xx.max()) + pad + 1)
            y0, y1 = max(0, int(yy.min()) - pad), min(h, int(yy.max()) + pad + 1)
            with Image.open(checked_file(frames[frame.image_id]["source"])) as im:
                full_rgb = im.convert("RGB")
                W, H = full_rgb.size
                crop = full_rgb.crop(
                    (
                        round(x0 * W / w),
                        round(y0 * H / h),
                        round(x1 * W / w),
                        round(y1 * H / h),
                    )
                )
            rgb = np.asarray(crop)
            layers = []
            for mask, color in [
                (None, None),
                (masks[0], [20, 200, 255]),
                (masks[1], [210, 100, 255]),
                (masks[2], [245, 170, 30]),
            ]:
                panel = rgb.copy()
                if mask is not None:
                    local = np.asarray(
                        Image.fromarray(mask[y0:y1, x0:x1]).resize(
                            crop.size, Image.Resampling.NEAREST
                        )
                    )
                    panel[local] = (
                        panel[local] * 0.5 + np.asarray(color) * 0.5
                    ).astype(np.uint8)
                tile = Image.fromarray(
                    rotate_image(panel, frames[frame.image_id]["applied_quarter_turns"])
                )
                tile.thumbnail((500, 550))
                layers.append(tile)
            labels = [
                "Original RGB crop",
                "2D proposal",
                "Native: >=2 positive timestamps",
                "Native: full connected growth",
            ]
            if scene is not None:
                if frame.image_id not in scene_cache:
                    view = SimpleNamespace(
                        world_to_camera=metric_c2w_to_scene_w2c(
                            frame.T_world_cam, run.meters_per_scene_unit
                        )
                    )
                    rendered, scene_depth, _ = render_gaussians(
                        scene,
                        view,
                        frame.K.astype(np.float32),
                        w,
                        h,
                        SimpleNamespace(**run.rgbd_config),
                        "cuda",
                    )
                    residual = float(
                        np.max(
                            np.abs(
                                scene_depth
                                - np.load(frame.depth_path, allow_pickle=False)
                            )
                        )
                    )
                    if (
                        residual
                        > config["render"]["alignment_guard_max_absolute_depth_error_m"]
                    ):
                        raise ValueError("true-colour scene render/depth parity failed")
                    scene_parity.append(
                        dict(
                            image_id=frame.image_id,
                            maximum_absolute_depth_error_m=residual,
                        )
                    )
                    scene_cache[frame.image_id] = np.rint(
                        np.clip(rendered, 0, 1) * 255
                    ).astype(np.uint8)
                tile = Image.fromarray(
                    scene_cache[frame.image_id][y0:y1, x0:x1]
                ).resize(crop.size, Image.Resampling.BILINEAR)
                tile = Image.fromarray(
                    rotate_image(
                        np.asarray(tile),
                        frames[frame.image_id]["applied_quarter_turns"],
                    )
                )
                tile.thumbnail((500, 550))
                layers.insert(1, tile)
                labels.insert(1, "Original 3DGS scene RGB (SH)")
            sheet = Image.new("RGB", (512 * len(layers), 610), "#111827")
            draw = ImageDraw.Draw(sheet)
            draw.text(
                (8, 8),
                f"group {obj.object_id} | {frame.source_image} | {role} observations; not independent accuracy",
                fill="white",
            )
            for i, (tile, label) in enumerate(zip(layers, labels)):
                draw.text((i * 512 + 8, 29), label, fill="white")
                sheet.paste(
                    tile,
                    (i * 512 + (500 - tile.width) // 2, 55 + (550 - tile.height) // 2),
                )
            path = (
                args.output / f"group_{obj.object_id:04d}_view_{frame.image_id:04d}.jpg"
            )
            sheet.save(path, quality=95)
            visuals.append(describe_file(path))
    write_json(
        args.output / "manifest.json",
        dict(
            schema="farm.native-membership-review.v1",
            input=describe_file(args.input),
            source_report=describe_file(args.report),
            source_bank=describe_file(bank_path),
            counts=counts,
            scene_rgb_rendered=scene is not None,
            scene_depth_parity=scene_parity,
            visuals=visuals,
            total_seconds=time.monotonic() - started,
            interpretation=f"{role} qualitative consistency. The >=2-timestamp subset is a diagnostic; do not promote it by mask size or sparsity alone.",
            closed_test_opened=False,
            release_eligible=False,
        ),
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
