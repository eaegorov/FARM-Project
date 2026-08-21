# Scene quality, dense lift and object reconstruction

This document explains how to judge a FARM result, what each viewer layer
actually proves, and how selected objects become metric dense labels or
reconstructed surfaces. A clean-looking layer can be incomplete, and a dense
layer can still contain the wrong object; visual appearance alone is not an
acceptance gate.

## What each layer proves

| Layer | Evidence | What it does not prove |
| --- | --- | --- |
| FARM preview | sampled scene points plus final presentation objects | precise surfaces or exhaustive recall |
| FARM OBBs | metric state after geometry, surface and duplicate gates | that every box tightly follows the silhouette |
| Dense Gaussian lift | source 3DGS rows passing held-out mask QC | that unknown rows are background or every object is complete |
| ShapeR | a generated mesh hypothesis aligned to FARM geometry | measured surface truth, exact topology or texture fidelity |
| MV-SAM3D | real PINHOLE images, 2D masks and metric pose/scale refinement | perfect unseen surfaces or semantic correctness |

The unified viewer keeps these as independent controls. It never substitutes
ShapeR for dense lift or presents a calibration artifact as a release result.

## Diagnosis of the reviewed Factory and Knaack views

The Factory captures contain three materially different products:

1. Isolated coloured splats are high-precision verified Gaussian subsets. The
   holes and thin tails are recall/completeness limitations, not a full scene
   segmentation.
2. Coloured splats over the context cloud show that most retained objects are
   spatially aligned, while also exposing partial masks and fragments.
3. Isolated ShapeR meshes have coherent silhouettes for many cabinets, carts
   and frames, but they are generated hypotheses. Smooth surfaces must not be
   interpreted as improved recognition.

The old Knaack OBB capture exposed a real pipeline defect: 323
geometry/visual-valid tracks remained active even though many represented
walls, transient fragments, duplicate structures or unsupported oversized
boxes. Its old lift looked cleaner because only a small high-precision subset
survived held-out QC; that cleanliness concealed low recall. Its isolated
ShapeR view repeated easy classes and lacked context, so it could not
demonstrate pose or scale correctness.

## Corrective gates and measured results

The standard contract now enforces Gaussian surface evidence before final
publication. New snapshots also run that gate before semantic captioning, so
unsupported candidates do not consume VLM time or enter semantic review.

Measured cold runs on immutable source PLYs:

| Scene/run | Visual-valid before surface gate | Final metric | Presentation | Release lift |
| --- | ---: | ---: | ---: | --- |
| Factory `factory-production-v4-quality` | 90 | 85 | 36 | 15/21 strict; 188,051 rows; PASS |
| Knaack `knaack-production-v8-quality` | 323 | 96 | 52 | 31/36 strict; 163,271 rows; PASS |

Knaack demoted 227 unsupported active tracks. The new final viewer therefore
has 52 presentation OBBs instead of the old 169/323-object clutter. The new
Knaack lift verifies 86.1% of strict candidates; v7 verified 36.2% and
correctly remained nonrelease.

These numbers are not a claim of exhaustive inventory. Open-vocabulary
detection can miss small, occluded or unseen objects. FARM reports retention,
semantic uncertainty and surface rejection separately so missing recall is not
hidden behind structural PASS.

## Dense Gaussian lift completeness

The lift uses the exact contributor VJP of the pinned `gsplat` renderer at
native frame resolution. It builds labels from a global physical-timestamp
split, freezes the build result, and only then opens held-out masks for QC.

The connected refinement pass addresses partial 3D masks without weakening
that gate:

- growth starts only from a strong multi-timestamp core;
- only unknown rows with positive build evidence are eligible;
- new rows must be spatially connected at an adaptive metric radius;
- strong cross-object conflicts stay unknown;
- held-out masks are never opened during growth and never overwrite labels;
- rejected objects return to unknown rather than promoting a runner-up.

Release output includes dense source-order arrays, a verified CSR bank and a
full-schema labelled PLY. Exact sizes and SHA-256 values are recorded. The
independent audit checks dense/CSR equality and preservation of every original
PLY row and field.

## MV-SAM3D metric reconstruction contract

The Knaack reconstruction under `splatica_demo_app/outputs/farm_recon` was
built from 21 reviewed objects and 72 real images from
`data/knaack_colmap_pinhole/images`:

- 2–5 views per object, preferring complete and unoccluded views;
- lossless native crops with no non-uniform resize, stretch or blur;
- nearest-neighbour mask resampling only;
- checked connected masks clear of crop boundaries;
- metric pose initialised from FARM OBB/camera evidence and refined against
  all selected masks;
- reverse projection against every source mask.

The final result contains all 21 requested objects: 21 PASS and 0 REVIEW. The
median reverse-projection bbox IoU improved from 0.624 to 0.814. REVIEW remains
visible for manual inspection and is never silently promoted.

Canonical artifacts are one raw object directory per object plus:

- `combined_metric_objects.glb` in glTF right-handed +Y-up metres;
- `combined_world_metric_objects.npz` in original FARM world metres;
- `combined_world_metric_gaussians.ply` in original FARM world metres;
- `combined_viewer_y_up_gaussians.ply` in glTF/Viser +Y-up metres;
- `manifest.json`, `_SUCCESS.json` and `alignment_audit.json` binding bytes,
  object IDs, coordinate transforms and QA.

Do not apply another 180-degree rotation to the FARM-world PLY. Knaack's
resolved up is close to negative Y. The GLB/Y-up PLY already contain the one
recorded rigid FARM-world to glTF conversion.

## Inspection services

Normal multi-scene viewer:

```bash
python scripts/serve_farm_unified_viewer.py \
  --registry configs/viewer/scenes.local.v1.yaml --validate --verify-full-source
python scripts/serve_farm_unified_viewer.py \
  --registry configs/viewer/scenes.local.v1.yaml --host 127.0.0.1 --port 8080
```

Dedicated Knaack MV-SAM3D alignment viewer:

```bash
python scripts/serve_farm_recon_overlay.py \
  --cloud /absolute/output/farm_pipeline/knaack/runs/RUN/final/cloud.npz \
  --recon /absolute/splatica_demo_app/outputs/farm_recon \
  --host 127.0.0.1 --port 8082 \
  --max-context-points 1500000 --max-splats 1800000
```

It shows neutral context and a deterministic stratified sample retaining all
21 reconstructed object IDs. Natural colour and an instance palette are
switchable. Object selection changes the orbit pivot; overview, front, side,
top, focus and previous controls provide camera recovery. The full 10.32M-row
PLY remains authoritative; the browser sample is disclosed explicitly.

Both services are unauthenticated single-user inspection tools. Keep them on
loopback and use an SSH tunnel. A non-loopback bind requires a firewall or an
authenticated reverse proxy.

## Acceptance checklist for a new scene

1. Validate the 17-stage plan and strict model/resource preflight with full
   hashes.
2. Start a cold run from a clean commit; never resume across source, config,
   runtime-image or input changes.
3. Require structural PASS, zero hard label errors and zero remaining blocking
   overlap clusters.
4. Inspect selection, RGB-D alignment, OBB reprojection, surface support,
   retention and uncertainty dashboards.
5. Inspect saved real-image evidence for large, thin, repeated and occluded
   objects; do not accept an OBB-only overview.
6. Run lift without threshold changes. Release only if the frozen strict gate
   passes; ambiguous and failed rows remain unknown.
7. Rehash dense arrays, CSR and full PLY; verify source-row preservation.
8. For reconstruction, require 2–5 clean PINHOLE views, masks clear of crop
   boundaries and reverse-projection QA in every view.
9. Validate combined GLB/NPZ/PLY IDs, counts, coordinate frame, finite geometry
   and hashes before serving or handoff.
10. Record the commit, snapshot tree, image/model/config hashes and limitations
    with the result.

## Remaining limitations

- The selector is trajectory/connectivity-aware, not a learned coverage
  optimiser.
- Lift is conditioned on upstream all-view FARM geometry even though held-out
  masks are isolated from contributor assignment.
- Connected refinement is not a full semantic 3D island validator.
- Viser quantises covariance to float16 and RGBA to uint8 and displays DC
  colour only; it is not native 3DGS rendering.
- MV-SAM3D and ShapeR hallucinate unseen surfaces. Reverse-projection and
  metric alignment reduce risk but do not create ground truth.

A deployment may tighten gates for its vocabulary and recall target, but must
recalibrate on one scene and validate blindly on another before committing a
new configuration.
