# Generic COLMAP + 3DGS RGB-D preparation

`scripts/prepare_colmap_3dgs_rgbd.py` creates the metric RGB-D input consumed by
FARM from an aligned COLMAP reconstruction and Gaussian-splat PLY. It replaces
the removed scene-specific factory selection/render utilities; historical
experiments retain their commands and timing evidence in their run bundles.

## Input contract

- a registered COLMAP model using only `PINHOLE` or `SIMPLE_PINHOLE` cameras;
- its source image tree, with paths exactly matching COLMAP image names;
- the aligned 3DGS PLY (`x/y/z`, opacity, scale, quaternion and SH fields);
- an exact newline-delimited `selected_names.txt`, normally produced by
  `select_colmap_keyframes.py`;
- a caller-defined `--scene-id`.

The selected-name order is preserved. Unknown names, duplicates, path traversal,
source/COLMAP dimension mismatch and fisheye camera models fail before rendering.
Use `convert_colmap_kb4_to_pinhole.py` before this stage for KB4 source models.

## Example

```bash
python scripts/prepare_colmap_3dgs_rgbd.py \
  --colmap-model /data/scene_pinhole/sparse/0 \
  --image-root /data/scene_pinhole/images \
  --ply /data/scene/point_cloud.ply \
  --selected-names /output/selection/selected_names.txt \
  --output-dir /output/scene/rgbd \
  --scene-id scene_001 \
  --resolution 896
```

The default identity parser recognises names such as
`cam00_000123_yaw_left.png`. For other naming schemes pass a Python regular
expression with named `sensor`, `timestamp` and (optionally) `family` groups:

```bash
  --identity-regex '^(?P<timestamp>[^/]+)/(?P<sensor>[^/]+)/(?P<family>[^.]+)\\.jpg$'
```

Names which do not match still work generically (`camera_<COLMAP id>` and the
file stem), provided that they do not create duplicate `(timestamp, camera)`
pairs.

## Fail-closed alignment QA

Before the full batch, distributed views (first/last and evenly spaced
intermediate views) are rendered and checked in two independent ways:

1. registered sparse COLMAP track depth (`camera Z`) is compared with gsplat
   expected depth at the correctly scaled 2D observations;
2. source RGB is compared with the 3DGS RGB render using luminance correlation,
   Sobel-edge correlation and an exposure/white-balance-corrected RGB MAE.

Dense depth coverage is checked as well. If evidence is missing or any aggregate
threshold fails, the run stops before rendering the remaining views. Thresholds
are CLI-configurable (`--qa-*`) because outdoor, reflective and sparse scenes
have different practical envelopes. Do not disable the gate; calibrate it on a
small validated scene set.

Metric scale is not assumed. `--expected-baseline-m` is disabled by default. If
set, provide `--baseline-camera-ids A B` unless the selected model contains
exactly two camera IDs; the check then fails on missing pairs or scale mismatch.

## Output and resume semantics

```text
rgbd/
├── frames.json
├── prep_summary.json
├── run_manifest.json
├── rgb/                         # resized source RGB
├── depth/                       # float32 metres, invalid=0
├── qa/
│   ├── 01_alignment_dashboard.jpg
│   ├── 02_alignment_contact_sheet.jpg
│   └── alignment_metrics.json
└── .run_state/                  # explicit atomic resume checkpoints
```

Every RGB, depth, JSON and QA image is written to a same-directory temporary
file and atomically replaced. The manifest fingerprint covers render/QA config,
the exact selected list and input files. A matching interrupted run resumes only
validated RGB/depth pairs; a mismatching fingerprint refuses to modify the
directory. `--fingerprint-content` adds full SHA-256 input hashing when metadata
fingerprints are insufficient. A completed matching run is an idempotent no-op;
`--rerender` atomically regenerates it.

`prep_summary.json` records each view's source/raster/write time, stage totals,
Gaussian-load and pipeline peak VRAM, depth statistics, baseline evidence and QA
metrics. The dashboard/contact sheet are intended for both operator review and
the experiment deliverable.

## Tests

Focused helpers require no CUDA or gsplat:

```bash
pytest -q tests/test_prepare_colmap_3dgs_rgbd.py
```

The tests cover exact selection order, PINHOLE intrinsics, fisheye rejection,
identity collisions, scaled sparse-depth reprojection, RGB alignment metrics,
fail-closed/missing-evidence dashboard rendering, deterministic output names and
atomic resume-pair validation. A real aligned COLMAP/3DGS smoke run is still
required inside the FARM CUDA image before production use.
