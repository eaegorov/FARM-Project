# Production COLMAP + 3DGS pipeline

This is the production path for turning a new COLMAP reconstruction and its
aligned Graphdeco 3DGS PLY into four independently auditable products:

1. a final FARM object memory and evidence bundle;
2. an exact dense `gaussian_index -> instance_id` sidecar and a full-schema
   labeled PLY;
3. optional ShapeR mesh hypotheses for verified instances;
4. one read-only viewer on port 8080 with scene and layer selection.

The pipeline never deletes source Gaussians and never mutates the input PLY.
FARM labels are open-vocabulary evidence, not semantic ground truth. ShapeR
surfaces are generated hypotheses, not measured Gaussian geometry.

## Current acceptance boundary

The checked-in Gaussian-lift thresholds are a measured Factory freeze:
`configs/gaussian_lift.v1.yaml` has `release.calibrated: true` after 28 of
40 strict Factory objects passed without per-object threshold retuning. The
unchanged release floor is one verified object and a 0.50 verified-strict
fraction. Knaack is a blind/no-retune validation scene; failures remain
unknown. This freeze must be committed and captured by a new cold FARM run
before a lift can publish release `_SUCCESS.json`.

The canonical viewer admits a dense lift or ShapeR layer only through a
release `_SUCCESS.json` chain. It still shows the final FARM result, the full
source 3DGS, and a compatible legacy bridge preview until a release lift has
been produced.

There is no fixed runtime guarantee. Scene size, selected views, object count,
GPU, model cache state, and semantic workload all matter. Each run records the
actual wall time, RSS and VRAM evidence per stage.

## 1. Clone and host control plane

Required host components are Git, Python 3.10+, Docker with Compose, an NVIDIA
driver/runtime compatible with the selected images, and enough disk for model
snapshots and immutable run outputs.

Always clone recursively. GitHub source archives do not contain the YOLOE
submodule.

```bash
git clone --recursive <FARM_REPOSITORY_URL> FARM-Project
cd FARM-Project

python3 -m venv .venv-control
.venv-control/bin/python -m pip install --requirement requirements/control-plane.lock.txt
export FARM_PY="$PWD/.venv-control/bin/python"
```

The host environment is intentionally small: PyYAML, Hugging Face Hub,
Pillow for pre-Docker canonical-image validation, plan orchestration and
Docker lifecycle only. ML/CUDA stages run in pinned images.

If the repository was cloned without submodules:

```bash
git submodule update --init --recursive
```

## 2. Provision exactly pinned models

Accept the license for the gated DINOv3-S+/16 repository and make an HF token
available either in `HF_TOKEN` or in a permission-restricted JSON file:

```json
{"HF_TOKEN": "hf_..."}
```

Then run the single manifest-driven bootstrap. It downloads the exact Qwen,
DINOv3 and SigLIP2 revisions declared in
`configs/models/farm_models.v1.json`; YOLOE and MobileCLIP downloads are
accepted only after their complete SHA-256 matches the manifest.

```bash
FARM_MODEL_PYTHON="$FARM_PY" ./bootstrap_models.sh \
  --secrets-file /absolute/private/path/secrets.json
```

For an offline audit of already provisioned files and HF cache:

```bash
FARM_MODEL_PYTHON="$FARM_PY" ./bootstrap_models.sh \
  --secrets-file /absolute/private/path/secrets.json \
  --local-files-only
```

`--verify-full-hashes` reads complete large HF blobs instead of relying on
their content-addressed cache names. The downloader fails closed on a floating
HF revision, missing file, path mismatch or checksum mismatch. It does not
print the token, including when a dependency puts the token in an exception.

## 3. Build, audit and pin FARM images

The current production manifest names `scene_graph:latest` for the main
runtime and `rest3d:factory-baseline` for preparation. Build both tags from
the reviewed checkout:

```bash
FARM_UID=1000 FARM_GID=1000 ./run.sh build

docker build --check --file docker/Dockerfile.prep .
docker build --file docker/Dockerfile.prep \
  --build-arg FARM_UID=1000 \
  --build-arg FARM_GID=1000 \
  --tag rest3d:factory-baseline .
```

Tags are lookup names, not release identities. Audit every configured tag
against its local immutable Docker image ID:

```bash
"$FARM_PY" scripts/pin_farm_runtime_images.py
```

Exit 0 means every pin is already in sync. Exit 1 means drift and performs no
write. After reviewing the images, atomically update only
`runtimes.*.image_id`:

```bash
"$FARM_PY" scripts/pin_farm_runtime_images.py --write
git diff -- configs/models/farm_models.v1.json
git add configs/models/farm_models.v1.json
git commit -m "build: pin reviewed FARM runtime images"
"$FARM_PY" scripts/pin_farm_runtime_images.py
```

The prep Dockerfile is functionally reproducible, not bit-reproducible: base
tags, apt repositories and wheels are external mutable inputs. On 2026-08-19,
the portability candidate built without Docker build-check warnings and passed
an H100 GPU smoke with Python 3.11.15, Torch 2.5.1+cu121, CUDA 12.1,
gsplat 1.5.3, OpenCV 4.11.0, pycolmap 3.11.1 and SciPy 1.16.3. That candidate
was not silently repinned into an already running Factory experiment; a fresh
build still requires the audit, explicit pin and commit sequence above.

## 4. Input contract

A production scene needs:

- a registered COLMAP sparse model (`cameras`, `images`, `points3D` in binary
  or text form);
- the RGB files referenced by COLMAP image names;
- an aligned, binary little-endian, vertex-only Graphdeco 3DGS PLY;
- a trustworthy metric-scale source and gravity policy for metric outputs;
- explicit camera/timestamp/view grouping for a multi-camera or virtual-view
  rig.

The source PLY may contain all original higher-order SH properties. At minimum
the lift requires `x/y/z`, `f_dc_0..2`, `opacity`, `scale_0..2` and
`rot_0..3`. COLMAP, RGB and PLY must describe the same coordinate frame; the
preflight checks alignment but cannot repair a wrong reconstruction pairing.

Start a rig scene from `configs/scenes/example_3dgs_colmap.yaml`. Do not copy
Factory/Knaack baselines or gravity vectors into another rig. Measure them.
All relative paths resolve from the scene YAML, not the shell working
directory.

For a simple ordered monocular PINHOLE scene, the minimal contract is:

```yaml
schema_version: farm.scene.v1
scene_id: my_scene
inputs:
  colmap_model: /data/my_scene/sparse/0
  image_root: /data/my_scene/images
  gaussian_ply: /data/my_scene/scene.ply
output_root: /output/farm
```

For a rig, additionally declare `camera_grouping`, `selection`,
`metric_scale`, `gravity`, `resources.model_manifest` and
`resources.secrets_file`, as in the full example.

If the registered source cameras are KB4/fisheye rather than an accepted
PINHOLE model, first create and validate the virtual PINHOLE reconstruction
described in [KB4_TO_PINHOLE.md](KB4_TO_PINHOLE.md). Do not relabel raw
fisheye intrinsics as PINHOLE.

### Keyframe-selection limitation

The selector currently balances pose motion, COLMAP track support and graph
connectivity and can add bridge timestamps. It does not optimize RGB content,
semantic/object coverage, masks or set-cover over reliable surface voxels.

The current Factory config validates five virtual-view families but maps only
`output_views: [center]`. Its observed selection is 174 physical timestamps
from two lenses, or 348 center views. A coverage-balanced 480–640-view selector
has not been implemented and is not claimed by this repository.

## 5. Validate and run FARM

The standard scene contract compiles to 17 fail-closed stages:

```text
preflight -> selection -> rgbd -> mapping -> mapping_qa -> presentation
  -> geometry -> visual_consistency -> semantics -> part_whole -> assemblies
  -> surface_support -> compound_geometry -> geometry_qa -> dedup
  -> finalize -> qa_bundle
```

Before a release-capable run, commit the reviewed source, manifests, scene
configuration and optional ShapeR pins. Model binaries, data, output and
secrets remain outside Git. A dirty source can be captured for development,
but downstream release lift requires the FARM run to record a clean commit.

```bash
git status --short --untracked-files=all

"$FARM_PY" scripts/farm_pipeline.py validate-plan \
  --config configs/scenes/factory_3dgs_colmap.yaml

"$FARM_PY" scripts/farm_pipeline.py run \
  --config configs/scenes/factory_3dgs_colmap.yaml \
  --run-id factory-production-v3
```

Knaack uses the same runtime and a different declarative scene contract:

```bash
"$FARM_PY" scripts/farm_pipeline.py validate-plan \
  --config configs/scenes/knaack_3dgs_colmap.yaml

"$FARM_PY" scripts/farm_pipeline.py run \
  --config configs/scenes/knaack_3dgs_colmap.yaml \
  --run-id knaack-production-v1
```

For a new scene, substitute its reviewed YAML and a new unique run ID.
Successful run directories are immutable. To inspect the latest attempt,
including a failed one:

```bash
"$FARM_PY" scripts/farm_pipeline.py status \
  --config configs/scenes/factory_3dgs_colmap.yaml \
  --attempt --json
```

Resume only an interrupted/failed attempt whose source snapshot and artifacts
still validate:

```bash
"$FARM_PY" scripts/farm_pipeline.py run \
  --config configs/scenes/factory_3dgs_colmap.yaml \
  --run-id factory-production-v3 --resume
```

Use a new run ID, not resume, after changing source, model/image pins or scene
configuration when the objective is a cold acceptance run.

### Execution-source integrity

At run creation, FARM copies an allowlisted execution tree into
`config/source_snapshot/FARM-Project` and records every file's path, size,
mode, executable bit and SHA-256 plus a canonical tree SHA-256. Secrets,
caches, model binaries, outputs, symlinks and special files are excluded or
rejected. Runtime package import targets from the YOLOE/MobileCLIP submodules
are included.

Every standard stage, finalizer and report executes from that snapshot. The
tree is revalidated before and after execution; file-set or hash drift fails
closed. This is a hash-manifested snapshot, not a GPG/PKI signature.

### FARM outputs and measurements

The durable result is under:

```text
<output_root>/<scene_id>/runs/<run_id>/
  _SUCCESS.json                 # only after final acceptance and QA pass
  manifest.json
  config/source_snapshot/       # immutable execution source + hash manifest
  input/                        # scene/resource preflight and resolved context
  selection/                    # selected timestamps/views
  rgbd/                         # metric frames and rendered depth
  mapping/                      # raw and progressively refined scene states
  final/
    scene_state.pt
    catalog.json
    presentation_catalog.json
    cloud.npz
  qa/
    acceptance/result.json
    summary.json
    retention_funnel.json
  visuals/
  viewer/
  stages/<stage>/attempts/<n>/  # stdout, stderr, raw telemetry
  timing/summary.json           # per-stage wall/RSS/VRAM summary
```

`scene_state.pt` is PyTorch pickle data; open only locally produced or trusted
artifacts. GPU fields distinguish PID-attributed memory from whole-device
baseline/peak/delta. Whole-device numbers include other processes on that GPU.

The semantic stage uses mask-grounded multi-view evidence, adaptive review for
unresolved/low-confidence candidates, candidate-conditioned physical vetoes,
and final independent-evidence consensus. A physical guard cannot silently
invent a replacement label. Unsupported identities remain geometry-only.

## 6. Exact Gaussian lift

The lift is not nearest-neighbor transfer from a 700k preview cloud. It runs
inside the prep image pinned by the source FARM run and uses exact contributor
weights from the pinned gsplat renderer at native frame resolution.

In particular, a legacy bridge `instance_scene.ply` is a sampled
approximately-700k-point visualization without the complete source
SH/opacity/scale/rotation schema. It is never dense membership authority and
is not the PLY to use for the full labeled result.

It creates one global physical-timestamp build/held-out split, accumulates
mask-interior positive and boundary/negative evidence with depth/visibility
gates, resolves competing instance claims, freezes provisional membership,
then reverse-renders it on held-out timestamps. Failed objects return to
unknown; no runner-up is promoted. Held-out metrics weight distinct physical
timestamps equally instead of letting extra lenses or virtual-view families
dominate the acceptance score.

```bash
RUN=/absolute/output/farm_pipeline/factory/runs/factory-production-v3
PLY=/absolute/data/factory_scene_last.ply
LIFT=/absolute/output/farm_gaussian_lift/factory/factory-production-v3

"$FARM_PY" tools/farm_shaper_bridge/run_gaussian_lift.py \
  --run "$RUN" --ply "$PLY" --output "$LIFT" \
  --config configs/gaussian_lift.v1.yaml --gpu 0 --plan-only

"$FARM_PY" tools/farm_shaper_bridge/run_gaussian_lift.py \
  --run "$RUN" --ply "$PLY" --output "$LIFT" \
  --config configs/gaussian_lift.v1.yaml --gpu 0
```

`--output` must be new/empty and outside the immutable FARM run and source
repository. The launcher checkout must be clean for a canonical full run. Use
`--smoke-object-id ID` only for non-release development smoke tests.

A current canonical lift also requires the FARM root success marker to bind a
complete run-artifact inventory. The selected
`configs/gaussian_lift.v1.yaml` must match the copy inside that run's
hash-manifested source snapshot byte-for-byte. Changing a threshold therefore
requires a commit and a new cold FARM run; it cannot be injected into an older
release chain. `--allow-legacy-run` is an explicit non-release compatibility
path for pre-contract artifacts only.

The lift records a complete, streamed source-PLY SHA-256 and Gaussian count,
not the viewer's sampled fingerprint, and rechecks immutable inputs around
processing.

The dense output contains exactly one row per original source Gaussian, in
source PLY order:

```text
per_gaussian_object_id.npy          int32, -1 = unknown
per_gaussian_confidence.npy         float32
per_gaussian_timestamp_support.npy  uint16
per_gaussian_status.npy             uint8
verified_instance_bank.npz          CSR membership for verified instances
split_manifest.json
build_manifest.json
heldout_qc.json
result.json
instance_labeled_full.ply
labeled_ply_manifest.json
```

`instance_labeled_full.ply` preserves every original row and original field
bit-for-bit, including SH, opacity, scale and rotation, then appends
`farm_instance_id`, `farm_instance_confidence`,
`farm_timestamp_support`, and review RGB fields. It does not delete rows or
rewrite the original 3DGS attributes. The `.npy`/CSR sidecars are the canonical
membership data; RGB is visualization metadata.

“Exact contributor” means exact weights for the pinned renderer and inputs,
not guaranteed semantic truth. Unseen, ambiguous, boundary-leaking and
held-out-failing Gaussians deliberately remain `-1`.

## 7. Optional ShapeR hypotheses

ShapeR is downstream of verified CSR membership and never changes lift labels.
ShapeR inference runs in its pinned Docker image. The current host-side runtime
manager and inference launcher still import NumPy, OpenCV and SciPy for their
metadata validators, so do not run them from the minimal `$FARM_PY`
environment. SciPy 1.16.3 requires Python 3.11 or newer, so create this
optional exact direct-package launcher environment with an explicit 3.11+
interpreter:

```bash
python3.11 -m venv .venv-bridge-control
.venv-bridge-control/bin/python -m pip install \
  --requirement requirements/bridge-control.lock.txt
export FARM_BRIDGE_PY="$PWD/.venv-bridge-control/bin/python"
```

`$FARM_BRIDGE_PY` validates assets and launches Docker; it does not execute
ShapeR inference on the host. Prepare and assembly instead use the lightweight
`$FARM_PY` Docker wrapper. This split is intentional and should not be
collapsed into an unpinned system Python.

The ShapeR Docker recipe is functionally, not bit, reproducible. After the
exact external repo, checkpoints and HF cache named by
`configs/shaper_bridge.v1.yaml` are provisioned, build, inspect, repin and
commit:

```bash
"$FARM_BRIDGE_PY" tools/farm_shaper_bridge/manage_runtime.py build \
  --config configs/shaper_bridge.v1.yaml \
  --torch-cuda-arch-list 9.0

"$FARM_BRIDGE_PY" tools/farm_shaper_bridge/manage_runtime.py build \
  --config configs/shaper_bridge.v1.yaml \
  --torch-cuda-arch-list 9.0 --execute

export SHAPER_REPO=/absolute/ShapeR
export HF_CACHE=/absolute/hf-cache

"$FARM_BRIDGE_PY" tools/farm_shaper_bridge/manage_runtime.py repin \
  --config configs/shaper_bridge.v1.yaml \
  --shaper-repo "$SHAPER_REPO" --hf-cache "$HF_CACHE"
```

`repin` prints a review proposal; it does not modify source. Transfer the
reviewed image ID and asset hashes to `configs/shaper_bridge.v1.yaml`, commit
them, then validate:

```bash
"$FARM_BRIDGE_PY" tools/farm_shaper_bridge/manage_runtime.py validate \
  --config configs/shaper_bridge.v1.yaml \
  --shaper-repo "$SHAPER_REPO" --hf-cache "$HF_CACHE"
```

Both `repin` and `validate` perform a CUDA-visible runtime smoke by default
using `--gpus all`; the smoke must report CUDA available as well as the exact
package inventory. `--skip-smoke` performs assets-only inventory and is not
runtime validation or release evidence. ShapeR cache validation reads every
required blob: 64-hex blob names are checked as raw SHA-256 and 40-hex names as
Git blob SHA-1, rather than trusting a filename alone.

Create a new cold FARM run after that commit so its execution-source snapshot
contains the accepted ShapeR pins.

For a release lift, the default chain has no override flags:

```bash
SHAPER_INPUTS=/absolute/output/shaper/factory/inputs
SHAPER_BATCH=/absolute/output/shaper/factory/batch
SHAPER_SCENE=/absolute/output/shaper/factory/scene

"$FARM_PY" tools/farm_shaper_bridge/run_bridge_stage.py prepare \
  --run "$RUN" --ply "$PLY" --lift "$LIFT" \
  --output "$SHAPER_INPUTS" --print-command

"$FARM_BRIDGE_PY" tools/farm_shaper_bridge/run_shaper.py \
  --run "$RUN" --inputs "$SHAPER_INPUTS" --output "$SHAPER_BATCH" \
  --shaper-repo "$SHAPER_REPO" --hf-cache "$HF_CACHE" \
  --config configs/shaper_bridge.v1.yaml \
  --profile balance --gpu 0 --print-command

"$FARM_PY" tools/farm_shaper_bridge/run_bridge_stage.py assemble \
  --run "$RUN" --shaper-outputs "$SHAPER_BATCH" \
  --output "$SHAPER_SCENE" --print-command
```

Use `--plan-only` instead of `--print-command` to validate and print a
prepare/assemble plan without creating output or running Docker. For an
intentionally non-release lift, add `--allow-nonrelease-lift` only to
`run_bridge_stage.py prepare`. `--allow-legacy-run` is also a development
flag for prepare, but the canonical wrapper still requires a valid signed
source snapshot and byte-identical live wrapper/config. These flags cannot
promote output to release.

The ShapeR products are scene-prefixed input PKLs; per-object PASS GLBs and QA
JSON; `scene_manifest.json`; `viewer_assets.json`; and
`combined_metric_objects.glb/.npz`. Only QA-passing objects enter the combined
metric scene. Prepare and batch publish `result.json` plus exactly one of
`_SUCCESS.json` or `_NONRELEASE_SUCCESS.json`; assembly publishes the same
marker choice plus `scene_manifest.json` and `viewer_assets.json`.

All output directories must be new/empty. Prepare uses the exact prep image ID
and `/opt/conda/envs/rest3d/bin/python` bound by the source FARM run.
Assembly uses the exact ShapeR image/interpreter from its signed config.
Wrapper containers have no network, a read-only root filesystem, dropped
capabilities, `no-new-privileges`, and the current numeric UID:GID. Only the
exact output directory is writable; the FARM run, PLY, lift, batch, clean
archived ShapeR source, checkpoints and caches are read-only as applicable.

## 8. Unified viewer on port 8080

`configs/viewer/scenes.local.v1.yaml` registers Factory and Knaack. Each scene
binds one successful FARM run and source PLY fingerprint, plus optional search
roots for compatible bridge, release lift and ShapeR products. Paths resolve
relative to the registry.

For a new scene, copy the source PLY fingerprint record from the successful
run's `input/scene_preflight.json` into a new registry row:

```yaml
schema_version: farm.unified-viewer-registry.v1
scenes:
  - scene_id: my_scene
    label: My Scene
    farm_run: /absolute/output/farm_pipeline/my_scene/latest
    source_ply: /absolute/data/my_scene/scene.ply
    source_fingerprint:
      algorithm: sha256-sampled-v1
      digest: <digest-from-scene-preflight>
      size_bytes: <exact-size>
      chunk_bytes: 1048576
      sampled_offsets: [<offset-0>, <offset-1>, <offset-2>]
    bridge_roots: [/absolute/output/bridge/my_scene]
    dense_lift_roots: [/absolute/output/lift/my_scene]
    shaper_roots: [/absolute/output/shaper/my_scene]
```

Run the CLI inside the built main image (or another environment with the exact
viewer dependencies). Validate first:

```bash
python scripts/serve_farm_unified_viewer.py \
  --registry configs/viewer/scenes.local.v1.yaml --validate

python scripts/serve_farm_unified_viewer.py \
  --registry configs/viewer/scenes.local.v1.yaml \
  --host 127.0.0.1 --port 8080
```

`--validate` exits 0 only when every registry scene passes its required
FARM/source contracts; otherwise it exits 2. Serving may still expose a ready
scene while reporting another as unavailable. Dense-lift and ShapeR controls
appear only for the strict canonical release-marker chain; calibration outputs
cannot promote themselves into those viewer layers.

The current `scene_graph` image must run as UID/GID `1000:1000` so its venv is
readable, with the host artifact group added. Mount the repository, run
outputs and source scene inputs read-only. On Linux, an explicit launcher has
this shape:

```bash
docker run --rm --network host --read-only \
  --cap-drop ALL --security-opt no-new-privileges \
  --user 1000:1000 --group-add "$(id -g)" \
  --tmpfs /tmp:rw,nosuid,nodev,size=2g \
  --mount type=bind,src=/absolute/FARM-Project,dst=/workspace/FARM-Project,readonly \
  --mount type=bind,src=/absolute/output,dst=/workspace/output,readonly \
  --mount type=bind,src=/absolute/data,dst=/workspace/data,readonly \
  --workdir /workspace/FARM-Project \
  --entrypoint /home/scene_graph/.venv/bin/python \
  sha256:<PINNED_MAIN_IMAGE_ID> \
  scripts/serve_farm_unified_viewer.py \
  --registry configs/viewer/scenes.local.v1.yaml --host 127.0.0.1 --port 8080
```

Mount any additional path referenced by the registry, such as Knaack's source
root, at the corresponding read-only container path. Do not mount the whole
workspace merely to make path resolution convenient if it would expose
secrets.

The viewer is a single-user inspection tool and provides no authentication or
multi-tenant isolation. Keep the default `127.0.0.1` bind and use an SSH
tunnel for remote access. If `--host 0.0.0.0` is deliberately selected, put
the port behind a host firewall or an authenticated reverse proxy; never
assume the viewer itself protects the endpoint.

The independent viewer layers are:

- deterministic final FARM context preview and final object OBBs;
- every original source splat in a lazy Viser float16/uint8-quantized DC
  preview of anisotropic covariance, opacity and colour;
- compatible legacy 700k-point bridge instance preview;
- every verified original dense-lift splat and an explicit sampled-centre
  fallback;
- canonical ShapeR v2 mesh hypotheses;
- object labels and per-object inspection.

The full-source layer includes all N rows, but Viser transport/storage
quantizes floating values to float16 and display colours to uint8. It evaluates
DC spherical-harmonic colour only and ignores higher-order view-dependent SH.
It is therefore a geometry/instance inspection preview, not native-precision
data and not a photorealistic or exact-appearance 3DGS renderer. Full source
and dense splat layers are heavy, lazy opt-ins. `--verify-full-source` adds a
complete source-PLY SHA-256 read during validation.

## 9. Factory cold-run record

Do not use the historical `factory-generic-v7` output as proof for the current
checkout: it predates the current execution-source and final-acceptance
contracts.

Two current-source cold attempts were correctly rejected rather than promoted:

- `factory-production-v1` ran from commit `9ddcba7` and failed final acceptance
  after 1184.23 s;
- `factory-production-v2` ran from commit `6749025` and failed mapping after
  134.81 s because the source snapshot omitted editable runtime package
  targets needed by the main image.

Commit `757f654` extended the snapshot to include those YOLOE/MobileCLIP import
targets. It still requires a distinct from-zero run and `_SUCCESS.json` before
it can be called accepted. A failed run is diagnostic evidence, not a result to
publish through `latest`.

Apply the same rule to Knaack: an older Knaack artifact is useful for
comparison, but only a cold run of the current committed source ending in root
`_SUCCESS.json` proves the current pipeline for that scene.

## 10. Release checklist

Before treating a scene as production-ready, require all of the following:

1. recursive clone and reviewed clean source commit;
2. manifest-pinned model revisions/files verified;
3. both FARM Docker tags inspected, immutable IDs pinned and committed;
4. scene-specific scale, gravity, grouping and selection reviewed;
5. new cold 17-stage FARM run with root `_SUCCESS.json` and final acceptance;
6. timing/RSS/VRAM reviewed from the run, without cross-run guarantees;
7. exact lift source SHA/count match and held-out QA reviewed;
8. measured lift thresholds committed and a new run created before release;
9. optional ShapeR image/source/checkpoint/cache pins validated, with generated
   meshes treated as hypotheses;
10. viewer registry validated against the exact FARM/lift/ShapeR chain before
    serving it on port 8080.
