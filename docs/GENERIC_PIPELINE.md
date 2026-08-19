# Generic FARM pipeline runtime

**farm** (backward-compatible alias: **farm-pipeline**) turns a COLMAP + 3DGS
scene into an immutable, resumable run.
The runtime contains no scene names, object IDs, category prompts, camera
baselines or up-axis assumptions.

## Canonical scene contract

For a simple monocular sequence, copy the minimal
**configs/farm_scene.example.yaml** and set the four scene fields:

    schema_version: farm.scene.v1
    scene_id: my_scene
    inputs:
      colmap_model: /data/my_scene/sparse/0
      image_root: /data/my_scene/images
      gaussian_ply: /data/my_scene/scene.ply
    output_root: /output/farm

Relative paths are resolved relative to the scene YAML. Multi-camera rigs,
virtual fisheye views, metric baselines or explicit gravity should instead
start from the fully annotated **configs/scenes/example_3dgs_colmap.yaml**.
Ordinary PINHOLE/SIMPLE_PINHOLE monocular scenes need no overrides.

The canonical contract compiles to this fixed, fail-closed DAG:

    preflight -> selection -> rgbd -> mapping -> mapping_qa -> presentation
      -> geometry -> visual_consistency -> semantics -> part_whole -> assemblies
      -> surface_support -> compound_geometry -> geometry_qa -> dedup
      -> finalize -> qa_bundle

Every stage is executed as an argv list without a shell. The standard stage
adapter is **scripts/farm_standard_stage.py**. A missing adapter is a declared
input failure in preflight, never a partial successful result. Every stage must
write a JSON result with status **PASS**. Root **_SUCCESS.json** is written only
after **qa/result.json** reports PASS.

## Commands

    farm validate-plan --config scene.yaml
    farm run --config scene.yaml
    farm status --config scene.yaml --json
    farm serve --config scene.yaml --host 127.0.0.1 --port 8081
    farm stop --config scene.yaml

**serve** is the only command that starts a web process. It refuses an occupied
port and never kills an unrelated process. PID plus Linux process-start
identity are kept outside the immutable run. An existing demo on port 8080 is
untouched unless that exact managed run is explicitly stopped.

Resume an interrupted attempt:

    farm run --config scene.yaml --resume
    farm run --config scene.yaml --resume --force-stage geometry

Forcing a stage also invalidates every descendant. Successful runs are
immutable and cannot be resumed or overwritten.

## Output contract

    <output>/<scene_id>/
      latest -> runs/<last successful run>
      latest-attempt -> runs/<last attempted run>
      .runtime/<run_id>/viewer/       # mutable PID and server logs
      runs/<run_id>/
        config/                        # redacted source + resolved plan
        input/  selection/  rgbd/
        mapping/  final/  qa/
        visuals/  viewer/
        logs/  stages/  timing/
        manifest.json
        _SUCCESS.json | _FAILED.json

Each stage stores its fingerprint, attempts, stdout, stderr, wall time,
process-tree peak RSS, PID-attributed GPU memory, whole-device GPU memory and
raw telemetry samples. Whole-device counters are explicitly tagged and the
selected preflight GPU gets its own baseline/peak/delta record. Resume is
accepted only when fingerprints and output metadata still match. Final viewer
artifact paths are relative and must remain inside the run.

`qa_bundle` reruns the generic quality analyzer on the final state, requires
semantic/embedding coverage, builds a visual-only index under `visuals/`, and
writes `qa/summary.json`, timing/resource snapshots, plus the explicit offline
`viewer/launch.sh`. It records that a static open-vocabulary detector produces
an evidence-backed catalog rather than a guaranteed exhaustive inventory.

Standard Docker workers receive only the configured GPU, immutable image IDs,
resolved read-only model/cache mounts, and the immutable run. Mapping writes to
a stage-owned attempt directory and promotes its state/masks only after the
worker exits successfully, so retry never consumes partial mask output.

## Autonomous semantic contract

The standard `semantics` stage is prompt-free with respect to scene contents:
there is no object-ID table, scene vocabulary or hand-authored class override.
It now performs, in order:

1. FARM multi-view review on the saved object crop, with only non-target pixels
   inside that crop dimmed by the stored segmentation mask;
2. independent verification and open-vocabulary ensemble adjudication;
3. a conflict-only blind pass over paired raw and mask-grounded crops;
4. a class-agnostic consensus guard that accepts a replacement only when an
   independent channel supports it, and prevents a last-resort generic form
   from silently replacing a stronger FARM identity.

The semantic evidence is persisted under `qa/semantics/{review,ensemble,
reconciliation,consensus}/`. Canonical processing consumes
`consensus/semantic_consensus_catalog.json`; resume fingerprints include every
semantic implementation module. A transport/model failure fails the stage and
is never converted into an `unknown` label. A genuinely unsupported object may
remain geometry-only in the audit layer rather than receiving a fabricated
specific identity.

## Advanced declarative plan

Schema version 1 remains available for developers. It accepts an explicit
pipeline stage DAG with command as an argv list, needs, inputs, outputs,
pass_json, fingerprint_inputs, env, cwd and timeout.

Secrets must use a dedicated environment mapping:

    env:
      HF_TOKEN: ${env:HF_TOKEN}

Environment placeholders are forbidden in command arguments and paths so
tokens cannot leak through process listings or persisted plans.
