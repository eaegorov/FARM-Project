# Generic FARM Docker build and runtime gates

This layer builds a self-contained preparation image without changing the
existing FARM main image, running viewer, or port 8080.  It uses three explicit
host mounts:

- `FARM_SCENE_ROOT` → `/scene` read-only. The scene config and every relative
  COLMAP/image/PLY input must live below this root.
- `FARM_OUTPUT_ROOT` → `/runs` writable.
- the repository is copied into the prep image. No host Python environment or
  source bind mount is required for preparation.

The preparation image contains version-pinned direct dependencies. This is a
repeatable build contract, but it is **not bit-reproducible**: the lock does not
yet contain wheel hashes and the CUDA base is a tag rather than an OCI digest.

## Build and validate the preparation image

```bash
export FARM_SCENE_ROOT=/absolute/path/to/scene-bundle
export FARM_OUTPUT_ROOT=/absolute/path/to/experiment-output
export FARM_MAIN_IMAGE=scene_graph@sha256:<manifest-pinned-image-id>

docker compose -f docker/compose.pipeline.yml --profile smoke build prep-runtime-validate
docker compose -f docker/compose.pipeline.yml --profile smoke run --rm prep-runtime-validate
```

The Dockerfile fails its build if any exact entry in
`requirements/prep-runtime.lock.txt` resolves to another version or if the
COLMAP/3DGS preparation imports fail. CUDA execution is deliberately tested at
runtime, because an image build must not require access to a GPU:

```bash
docker compose -f docker/compose.pipeline.yml --profile prep run --rm prep \
  python3 -c 'import torch; assert torch.cuda.is_available(); print(torch.cuda.get_device_name(0))'
```

Run the read-only scene contract before a heavy stage:

```bash
export FARM_SCENE_CONFIG=scene.yaml
docker compose -f docker/compose.pipeline.yml --profile preflight run --rm scene-preflight
```

For a stage, keep its complete command in the experiment run manifest and
override the safe `--help` default. Example shape (stage arguments are supplied
by the generic runner):

```bash
docker compose -f docker/compose.pipeline.yml --profile prep run --rm prep \
  python3 scripts/prepare_colmap_3dgs_rgbd.py <explicit stage arguments>
```

The prep root filesystem and scene mount are read-only; only `/runs` and the
bounded `/tmp` tmpfs are writable. The service drops Linux capabilities and
runs with the configured non-root UID/GID.

## Validate and capture the main runtime inventory

`requirements/main-runtime.observed.txt` is the required import inventory for
the known-good main image, not an installable cross-platform lock. Validate the
exact installed versions without rebuilding or modifying that image:

```bash
docker compose -f docker/compose.pipeline.yml --profile main-smoke run --rm \
  main-runtime-validate
```

To bootstrap an inventory from another reviewed main image, capture only the
same package-name profile to a separate writable directory. Do not overwrite a
reviewed inventory directly from a running container:

```bash
mkdir -p build/runtime-inventory
docker run --rm \
  --entrypoint /home/scene_graph/.venv/bin/python \
  -v "$PWD:/workspace/farm:ro" \
  -v "$PWD/build/runtime-inventory:/inventory" \
  "$FARM_MAIN_IMAGE" \
  /workspace/farm/scripts/farm_runtime_inventory.py capture \
  --packages-from /workspace/farm/requirements/main-runtime.observed.txt \
  --output /inventory/main-runtime.candidate.txt \
  --report /inventory/main-runtime.capture.json
```

Review the candidate together with the source commit, submodule SHAs, model
manifest, CUDA/Python versions and exact image ID. A capture records installed
versions only; it never installs packages and never claims wheel availability.
The host-side `farm_resource_preflight.py` remains responsible for matching the
main/vLLM tags to manifest-pinned image IDs and checking model revisions.

## Acceptance smoke before using a new host

1. `docker compose ... config` succeeds with the three required environment
   variables and contains no host-user-specific absolute paths.
2. The prep build and `prep-runtime-validate` pass from a clean build cache.
3. GPU runtime smoke reports the explicitly assigned `FARM_PREP_GPU`.
4. `scene-preflight` passes for a small synthetic COLMAP + 3DGS bundle while
   `/scene` remains unchanged.
5. A tiny RGB-D preparation writes only below `/runs`; its report contains
   finite depth/alignment/metric-scale evidence.
6. `main-runtime-validate` passes for the manifest-pinned main image. Missing
   or mismatched packages return a non-zero exit and machine-readable report.
7. `farm_resource_preflight.py --all-models` passes on the host; no token value
   appears in Docker metadata, logs, or reports.
8. The sequential vLLM lifecycle smoke verifies the exact served model and
   stops its scoped worker. The existing viewer and port 8080 are untouched.

Do not mount `/var/run/docker.sock` into pipeline containers. The host runner
owns Docker/Compose lifecycle; stage containers receive only declared data,
models, configuration and output mounts.
