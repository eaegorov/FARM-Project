# FARM resource and model reproducibility

FARM separates scene validation from GPU/runtime validation. Run both before a
new scene is processed:

```bash
python3 scripts/farm_preflight.py --config configs/scenes/my_scene.yaml --strict
python3 scripts/farm_resource_preflight.py \
  --gpu 0 \
  --secrets-file ../secrets.json \
  --all-models \
  --report /tmp/farm-resource-report.json
```

Both commands are read-only unless an explicit report path is supplied. The
resource report contains GPU inventory, free/total VRAM, immutable model
revisions, checksum results, exact Docker image IDs, and the calculated vLLM
memory fraction. Secret values and token lengths are never returned or logged.

## Model manifest

`configs/models/farm_models.v1.json` is the deployable model contract. Relative
paths are resolved against the manifest, not the caller's working directory.

- Hugging Face models use an immutable 40-hex revision and a local snapshot.
- Large cache blobs are checked through their SHA256-addressed symlink names by
  default. `--verify-full-hashes` rereads the complete blobs when a deep audit
  is required.
- Local YOLOE, DINOv3, and SigLIP weights have full SHA256 checksums.
- Docker tags are accepted only when they resolve to the pinned image ID.
- `--all-models` includes pipeline weights and both runtime images. A stage
  launcher checks only the selected vLLM model and its required runtime.

After intentionally replacing a model or rebuilding an image, update the
manifest revision/checksum/image ID in the same reviewed change. Do not make a
floating branch, `latest` tag alone, or a cache `refs/main` file the production
identity.

## Sequential vLLM lifecycle

The generic launcher keeps at most one vLLM endpoint resident in a lifecycle
scope. This avoids paying for three simultaneous model allocations and makes
free-VRAM checks meaningful:

```bash
# Caption stage
scripts/start_farm_vllm.sh switch caption --gpu 0 --scope scene-run
# use http://127.0.0.1:8000/v1, model qwen3-vl-8b

# Text embedding stage; caption worker is removed first
scripts/start_farm_vllm.sh switch text-embed --gpu 0 --scope scene-run
# use http://127.0.0.1:8002/v1, model qwen3-emb-0.6b

# Visual-language embedding stage
scripts/start_farm_vllm.sh switch vl-embed --gpu 0 --scope scene-run
# use http://127.0.0.1:8006/v1, model qwen3-vl-emb-2b

scripts/start_farm_vllm.sh status --scope scene-run
scripts/start_farm_vllm.sh stop all --scope scene-run
```

The launcher always uses an explicit GPU index or full UUID. Containers receive
three labels: FARM ownership, scope, and service. Stop/status operations filter
on those exact labels. They do not use broad name matching and cannot select an
unlabelled viewer or a worker owned by another scope.

Startup succeeds only after `/v1/models` reports the exact manifest
`served_model_name`. A merely open port is not readiness. On a crash or timeout,
the launcher prints a bounded log tail and removes only the failed scoped
worker.

The host model cache and optional JSON secret are mounted read-only. The vLLM
entrypoint reads `HF_TOKEN` after container creation and prints only a fixed
"loaded" message. Fully cached pinned snapshots work without a token. A missing
token is therefore a warning; a malformed secret JSON is an error. Prefer:

```bash
chmod 600 ../secrets.json
```

## Adaptive VRAM policy

Memory utilization is calculated from physical VRAM, then clamped by each
service's manifest bounds. Startup fails if the GPU is too small, the target is
unreachable within the maximum fraction, or current free VRAM cannot preserve
the configured reserve.

For the caption model the current policy targets 24 GiB, preserves 6 GiB free,
and bounds utilization to 0.30–0.65. Approximate clean-GPU plans are:

| GPU class | Calculated caption fraction | Result |
| --- | ---: | --- |
| 80 GiB | 0.30 | comfortable |
| 48 GiB | 0.50 | recommended general minimum |
| 40 GiB | about 0.60 | supported, less headroom |
| 32 GiB | target cannot meet the 39 GiB-class gate | fail |

The policy is deliberately quality-preserving: it changes reserved KV/cache
capacity, not model weights, precision, or caption prompts. Other heavy FARM
stages must be stopped before switching to vLLM; unrelated GPU processes remain
visible through the free-VRAM gate and are never terminated automatically.

## Runtime inventory

- `requirements/prep-runtime.lock.txt` is the reproducible preparation stack
  verified in the pinned prep image (Torch 2.5.1+cu121, pycolmap 3.11.1,
  gsplat 1.5.3, NumPy 1.26.0).
- `requirements/main-runtime.observed.txt` records imports verified in the
  pinned main image. It is an audit inventory, not a promise that its CUDA
  wheels can be recreated on every host with a plain `pip install`.

Record the JSON scene preflight report, resource preflight report, image IDs,
and model-manifest SHA256 beside each experiment's timing/results. Together
they identify both the input geometry and the executable/model environment.
