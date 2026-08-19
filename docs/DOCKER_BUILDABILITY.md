# FARM Docker buildability and runtime gates

This page is the focused Docker acceptance checklist. The end-to-end clone,
model, scene, lift, ShapeR and viewer procedure is
[PRODUCTION_PIPELINE.md](PRODUCTION_PIPELINE.md).

FARM separates a small host control plane from two ML runtimes:

- `scene_graph:latest`: main mapping, semantics and viewer runtime;
- `rest3d:factory-baseline`: preparation, RGB-D and exact-lift runtime.

Both names are local lookup tags. The production identity is the exact
`sha256:<64-hex>` Docker image ID stored in
`configs/models/farm_models.v1.json`. Do not write a local image ID as
`scene_graph@sha256:...`: that syntax is for registry repository digests and
is not the local image-ID contract used here.

## 1. Host control plane

The host needs Docker/Compose, the NVIDIA container runtime, Git and Python
3.11+. Its pinned CPU control environment includes Torch and scientific
packages used by host validation/orchestration; it is not a CUDA inference
runtime:

```bash
python3.11 -m venv .venv-control
.venv-control/bin/python -m pip install \
  --requirement requirements/control-plane.lock.txt
export FARM_PY="$PWD/.venv-control/bin/python"
```

The lock contains the exact CPU-side execution closure for plan/manifest
parsing, preflight, selection, catalog/report generation, trusted scene-state
inspection, pinned model provisioning and canonical-image validation. GPU
inference remains in the pinned images.

## 2. Build both configured tags

Build the main image through the repository launcher. Build the prep image
directly so the tag matches the production manifest:

```bash
FARM_UID=1000 FARM_GID=1000 ./run.sh build

docker build --check --file docker/Dockerfile.prep .
docker build --file docker/Dockerfile.prep \
  --build-arg FARM_UID=1000 \
  --build-arg FARM_GID=1000 \
  --tag rest3d:factory-baseline .
```

The prep image copies the reviewed preparation source into `/opt/farm`; model
files, scenes and experiment outputs are runtime mounts. Its build validates
the exact direct-package inventory and imports with
`/opt/conda/envs/rest3d/bin/python`. GPU execution is a runtime smoke because
Docker builds must not depend on a GPU.

The Dockerfiles are functionally reproducible, not bit-reproducible: base
image tags, apt indexes and Python wheels remain external mutable inputs.

## 3. Audit, pin and commit immutable local IDs

The helper inspects every configured `runtimes.*.image` tag before it writes
anything. Its default mode is read-only:

```bash
"$FARM_PY" scripts/pin_farm_runtime_images.py
```

Exit 0 means every current tag matches its committed ID. Exit 1 reports drift
and writes nothing. Exit 2 means inspection or manifest validation failed.
After independently reviewing both builds:

```bash
"$FARM_PY" scripts/pin_farm_runtime_images.py --write
git diff -- configs/models/farm_models.v1.json
git add configs/models/farm_models.v1.json
git commit -m "build: pin reviewed FARM runtime images"
"$FARM_PY" scripts/pin_farm_runtime_images.py
```

`--write` atomically changes only `runtimes.*.image_id`; there is no
caller-supplied digest or bypass. A cold FARM run must start after the reviewed
manifest commit so its source snapshot contains the accepted pins.

## 4. Prep and scene smokes

`docker/compose.pipeline.yml` uses three bounded mounts:

- `FARM_SCENE_ROOT` -> `/scene` read-only;
- `FARM_OUTPUT_ROOT` -> `/runs` read-write;
- repository source is baked into prep, or mounted read-only for the main
  runtime inventory service.

```bash
export FARM_SCENE_ROOT=/absolute/path/to/scene-bundle
export FARM_OUTPUT_ROOT=/absolute/path/to/smoke-output
export FARM_SCENE_CONFIG=scene.yaml
export FARM_PREP_IMAGE=rest3d:factory-baseline
export FARM_MAIN_IMAGE=scene_graph:latest

docker compose -f docker/compose.pipeline.yml --profile smoke \
  run --rm prep-runtime-validate

docker compose -f docker/compose.pipeline.yml --profile prep \
  run --rm prep \
  /opt/conda/envs/rest3d/bin/python -c \
  'import torch; assert torch.cuda.is_available(); print(torch.cuda.get_device_name(0))'

docker compose -f docker/compose.pipeline.yml --profile preflight \
  run --rm scene-preflight

docker compose -f docker/compose.pipeline.yml --profile main-smoke \
  run --rm main-runtime-validate
```

Set `FARM_PREP_GPU` to select a prep GPU. The services run read-only with
dropped capabilities; only `/runs` and bounded tmpfs locations are writable.
Do not mount `/var/run/docker.sock` into a stage container. Docker lifecycle
belongs to the host orchestrator.

For an actual prep stage, the generic runner supplies the explicit arguments.
The equivalent service shape is:

```bash
docker compose -f docker/compose.pipeline.yml --profile prep \
  run --rm prep \
  /opt/conda/envs/rest3d/bin/python \
  scripts/prepare_colmap_3dgs_rgbd.py <explicit-stage-arguments>
```

## 5. Main runtime inventory

`requirements/main-runtime.observed.txt` is the reviewed import inventory of
the known main image. It is evidence of installed versions, not an installable
cross-platform lock. To capture a candidate without overwriting the reviewed
file:

```bash
mkdir -p build/runtime-inventory
docker run --rm \
  --entrypoint /home/scene_graph/.venv/bin/python \
  --mount type=bind,src="$PWD",dst=/workspace/farm,readonly \
  --mount type=bind,src="$PWD/build/runtime-inventory",dst=/inventory \
  scene_graph:latest \
  /workspace/farm/scripts/farm_runtime_inventory.py capture \
  --packages-from /workspace/farm/requirements/main-runtime.observed.txt \
  --output /inventory/main-runtime.candidate.txt \
  --report /inventory/main-runtime.capture.json
```

Review that candidate together with the source commit, recursive submodule
SHAs, model manifest, Python/CUDA versions and exact image ID. Capturing never
installs packages.

## 6. Observed portability closure and acceptance

On 2026-08-19, the separate candidate `farm-prep:portability-v1` built with
no `docker build --check` warnings and passed an H100 smoke with Python
3.11.15, Torch 2.5.1+cu121, CUDA 12.1, gsplat 1.5.3, OpenCV 4.11.0,
pycolmap 3.11.1 and SciPy 1.16.3. This establishes buildability of that
candidate; it was not used to repin the active production manifest.

A new host is accepted only after:

1. recursive clone and manifest-driven model verification succeed;
2. main and prep images build locally;
3. image audit, explicit atomic repin and commit succeed;
4. prep inventory and assigned-GPU smoke pass;
5. read-only scene preflight passes without changing `/scene`;
6. a tiny RGB-D job writes only under `/runs` and emits finite metric data;
7. main runtime inventory and full resource preflight pass against committed
   IDs and model hashes;
8. a new cold 17-stage FARM run ends in root `_SUCCESS.json`.

No build smoke by itself proves scene-level quality or guarantees a runtime.
