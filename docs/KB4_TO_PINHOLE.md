# KB4 fisheye COLMAP → virtual PINHOLE COLMAP

`scripts/convert_colmap_kb4_to_pinhole.py` converts registered
`KANNALABRANDT4` or `OPENCV_FISHEYE` images into standard COLMAP `PINHOLE`
views. It does not modify its source. The default layout is five 2048×2048,
90° views per physical lens:

- `center`;
- `yaw_left` / `yaw_right` at ±50°;
- `pitch_up` / `pitch_down` at ±50°.

For a two-lens rig this produces ten virtual images per timestamp. The code is
not limited to two lenses: each COLMAP camera used by a registered image is
handled independently and receives a deterministic `cam00`, `cam01`, … label.

## Recommended conversion

```bash
python scripts/convert_colmap_kb4_to_pinhole.py \
  --input-model /data/scene/sparse/0 \
  --input-images /data/scene/images \
  --output /data/scene_pinhole \
  --require-complete-rig \
  --width 2048 --height 2048 \
  --fov-x-deg 90 --fov-y-deg 90 \
  --side-angle-deg 50 \
  --workers 4
```

The result contains:

```text
scene_pinhole/
├── images/
│   └── cam00_000001_center.png
├── sparse/0/
│   ├── cameras.{bin,txt}
│   ├── images.{bin,txt}
│   └── points3D.{bin,txt}
├── conversion_manifest.json
└── conversion_qc_contact_sheet.jpg
```

The output directory is published by an atomic rename only after geometry,
model round-trip, RGB and contact-sheet validation pass. Interrupted runs leave
a hidden signature-specific sibling staging directory and resume per source
fisheye frame. Re-running the same completed conversion is an idempotent no-op;
an existing output with a different signature is never overwritten.

## Preflight and custom layouts

Use `--preflight-only` to validate paths, camera models, image sizes, rig
completeness, FOV coverage and naming without creating an output. Use explicit
labels when downstream code requires stable physical-lens names:

```bash
--camera-label 17=front --camera-label 42=rear
```

Custom views replace the five defaults; repeat the argument:

```bash
--view center,0,0 --view left,-45,0 --view upper,0,40
```

Each row is `NAME,YAW_DEG,PITCH_DEG[,ROLL_DEG]`. Camera coordinates are x-right,
y-down, z-forward. Positive yaw looks right; positive pitch looks up.

If source pixels were uniformly resized after COLMAP calibration, conversion
fails by default. `--scale-intrinsics-to-images` explicitly scales `fx, fy, cx,
cy`; do not use it for cropped or otherwise transformed images.

## Geometry guarantees and limitations

- The virtual world-to-camera transform is
  `R_virtual_world = R_fisheye_virtual.T @ R_fisheye_world` and
  `t_virtual_world = R_fisheye_virtual.T @ t_fisheye_world`.
- Every virtual view preserves the physical source camera centre.
- Source XYZ/RGB/error values are unchanged. 2D observations and tracks are
  rebuilt by reprojection and points must be visible from at least two distinct
  source images by default.
- Text and binary outputs contain only the standard `PINHOLE` model. If
  `pycolmap` is installed it is also used for an external round-trip check.
- The input binary reader recognizes local camera model id `12` as
  `KANNALABRANDT4`; text input is preferable when exchanging a custom KB4 model
  between unrelated COLMAP builds.
- Five views share each physical optical centre by design; they add angular
  coverage, not stereo baseline.

Run the focused tests with:

```bash
pytest -q tests/test_convert_colmap_kb4_to_pinhole.py
```
