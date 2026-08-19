# Factory scene: generic FARM runbook

The factory scene is a validated example of the canonical `farm.scene.v1`
pipeline. Its configuration is `configs/scenes/factory_3dgs_colmap.yaml`.
The removed legacy launcher is preserved only in historical source snapshots;
new runs use the generic `farm` CLI.

The complete clone, image/model pinning, dense-lift, ShapeR and unified-viewer
procedure is [PRODUCTION_PIPELINE.md](PRODUCTION_PIPELINE.md). This page records
only Factory-specific inputs, commands and current acceptance status.

## Inputs

- virtual PINHOLE COLMAP: `../data/factory_pinhole_multi/sparse/0`;
- virtual RGB views: `../data/factory_pinhole_multi/images`;
- aligned 3DGS: `../data/factory_scene_last.ply`;
- Hugging Face secret: `../secrets.json`, mounted read-only and never persisted;
- output root: `../output/farm_pipeline`.

The full five-view conversion is validated at preflight. The current mapping
configuration uses only the geometry-selected cam00+cam01 centre-view
backbone. The observed selection is 174 physical timestamps / 348 individual
views. The selector uses motion, COLMAP tracks and connectivity bridges; a
coverage-balanced 480–640-view selector has not been implemented.

## Commands

```bash
export FARM_PY="$PWD/.venv-control/bin/python"

"$FARM_PY" scripts/farm_pipeline.py validate-plan \
  --config configs/scenes/factory_3dgs_colmap.yaml
"$FARM_PY" scripts/farm_pipeline.py run \
  --config configs/scenes/factory_3dgs_colmap.yaml \
  --run-id factory-production-v3
"$FARM_PY" scripts/farm_pipeline.py status \
  --config configs/scenes/factory_3dgs_colmap.yaml --attempt --json
```

Resume the latest interrupted attempt:

```bash
"$FARM_PY" scripts/farm_pipeline.py run \
  --config configs/scenes/factory_3dgs_colmap.yaml \
  --run-id factory-production-v3 --resume
```

Resume only when the recorded snapshot and existing artifacts still validate.
Successful runs are immutable. Use a new run ID after any source, pin or scene
configuration change and for every cold acceptance attempt.

## Current cold-validation record

The historical `factory-generic-v7` bundle predates the current
hash-manifested execution-snapshot and final-acceptance contracts. It is not
proof that the current checkout passes and its numbers must not be reported as
current-run measurements.

Two current-source cold attempts were rejected correctly:

- `factory-production-v1`, commit `9ddcba7`, reached final acceptance and
  failed after 1184.23 s;
- `factory-production-v2`, commit `6749025`, failed mapping after 134.81 s
  because the execution snapshot omitted editable runtime package targets.

Commit `757f654` adds those YOLOE/MobileCLIP import targets to the snapshot.
It still needs a separate from-zero 17-stage run ending in root
`_SUCCESS.json`. Until that exists, there is no accepted current Factory
result. Failed attempts remain diagnostic evidence and must not update
`latest`.

## Quality boundary

The detector is a static open-vocabulary YOLOE profile. The result is an
evidence-backed catalog, not a guaranteed exhaustive inventory: very small,
occluded or out-of-vocabulary objects can still be missed. Rollout requires a
cold validation on several scenes with different rigs, scales and content.

There is no fixed runtime or VRAM guarantee. Use only the per-stage wall/RSS/
VRAM records from the exact accepted run; whole-device VRAM includes unrelated
processes on the selected GPU.

The current standard semantic implementation also includes tight mask-grounded
object crops, conflict-only blind reconciliation and a non-regressive consensus
guard. These are generic pipeline stages; the factory configuration supplies no
object names or per-ID corrections.
