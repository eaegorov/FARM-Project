# Generic FARM pipeline runtime

The supported production entry point is `scripts/farm_pipeline.py` (installed
aliases: `farm` and `farm-pipeline`). It turns a COLMAP + aligned 3DGS scene
contract into an immutable, resumable FARM run. The complete build, lift,
ShapeR and unified-viewer runbook is
[PRODUCTION_PIPELINE.md](PRODUCTION_PIPELINE.md).

## Scene contract

For a monocular sequence, start with `configs/farm_scene.example.yaml`:

```yaml
schema_version: farm.scene.v1
scene_id: my_scene
inputs:
  colmap_model: /data/my_scene/sparse/0
  image_root: /data/my_scene/images
  gaussian_ply: /data/my_scene/scene.ply
output_root: /output/farm
```

Multi-camera rigs, virtual fisheye views, metric baselines and explicit
gravity should start from `configs/scenes/example_3dgs_colmap.yaml`. Relative
paths resolve from the YAML. Scene-specific object IDs, label tables and
category prompts are not part of the contract.

The fixed fail-closed DAG is:

```text
preflight -> selection -> rgbd -> mapping -> mapping_qa -> presentation
  -> geometry -> visual_consistency -> semantics -> part_whole -> assemblies
  -> surface_support -> compound_geometry -> geometry_qa -> dedup
  -> finalize -> qa_bundle
```

Every stage executes an argv list without a shell and must write a declared
PASS JSON. Root `_SUCCESS.json` is written only after final acceptance and QA
pass.

## Commands

With the control-plane venv from the production runbook:

```bash
export FARM_PY="$PWD/.venv-control/bin/python"

"$FARM_PY" scripts/farm_pipeline.py validate-plan --config scene.yaml
"$FARM_PY" scripts/farm_pipeline.py run --config scene.yaml --run-id my-run
"$FARM_PY" scripts/farm_pipeline.py status --config scene.yaml --attempt --json
```

Resume a failed/interrupted attempt only when its execution snapshot and
published stage outputs still validate:

```bash
"$FARM_PY" scripts/farm_pipeline.py run \
  --config scene.yaml --run-id my-run --resume
```

`--force-stage geometry` invalidates geometry and every descendant. Use a new
run ID for a cold run after source/config/model/runtime changes.

The per-run FARM viewer remains available independently of the unified viewer:

```bash
"$FARM_PY" scripts/farm_pipeline.py serve \
  --config scene.yaml --host 127.0.0.1 --port 8081 --runtime docker
"$FARM_PY" scripts/farm_pipeline.py stop --config scene.yaml
```

It refuses an occupied port and stops only the process identity it owns. The
multi-scene/layer viewer uses port 8080 and
`scripts/serve_farm_unified_viewer.py`; see the production runbook.

## Execution-source snapshot

A new standard run atomically captures an allowlisted first-party source tree
under `config/source_snapshot/FARM-Project`. The manifest records file modes,
sizes, executable bits, per-file SHA-256 and a canonical tree SHA-256. It also
contains the actual YOLOE/MobileCLIP editable-package import targets used by
the main image. Secrets, models, caches and outputs are excluded; symlinks and
special files are rejected.

Stages, finalizers and reports execute from that snapshot, not the later live
checkout. Full tree validation occurs around execution. The snapshot is
hash-manifested; it is not a cryptographic author signature.

## Output contract

```text
<output>/<scene_id>/
  latest -> runs/<last successful run>
  latest-attempt -> runs/<last attempted run>
  .runtime/<run_id>/viewer/
  runs/<run_id>/
    config/ input/ selection/ rgbd/ mapping/ final/ qa/
    visuals/ viewer/ logs/ stages/ timing/
    manifest.json
    _SUCCESS.json | _FAILED.json
```

Every stage state records its fingerprint, attempt, declared input/output
fingerprints, stdout/stderr, raw telemetry, wall time, process-tree peak RSS,
PID-attributed GPU memory where available, and whole-device baseline/peak/
delta. Whole-device values include unrelated GPU processes and are labeled as
such.

`qa_bundle` validates the final state, semantic/embedding coverage,
final-acceptance artifacts and viewer bundle. It writes `qa/summary.json`,
`qa/retention_funnel.json`, `visuals/index.json`, and the per-run viewer
launcher. A static open-vocabulary detector produces an evidence-backed
catalog, not guaranteed exhaustive semantic recall.

## Semantic contract

The standard semantic stage is class- and scene-agnostic:

1. every candidate gets mask-grounded multi-view review;
2. unresolved or low-confidence candidates receive additional independent
   adjudication;
3. the standard reconciliation path keeps blind replacement disabled;
4. a candidate-conditioned physical guard may veto an incompatible noun but
   cannot propose a replacement;
5. final consensus requires independent positive evidence and records
   confirmed, probable or geometry-only status.

A transport/model failure fails the stage. Unsupported semantic identity is
retained as geometry-only instead of receiving a fabricated specific label.

## Advanced declarative plans

Schema version 1 remains available for explicit developer DAGs. A stage can
declare `command` (argv), `needs`, `inputs`, `outputs`, `pass_json`,
`fingerprint_inputs`, `env`, `cwd` and timeout.

Secrets must use a dedicated environment mapping:

```yaml
env:
  HF_TOKEN: ${env:HF_TOKEN}
```

Environment placeholders are forbidden in command arguments and paths so
tokens cannot leak through persisted plans or process listings.
