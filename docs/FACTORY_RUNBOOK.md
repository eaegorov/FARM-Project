# Factory scene: generic FARM runbook

The factory scene is a validated example of the canonical `farm.scene.v1`
pipeline. Its configuration is `configs/scenes/factory_3dgs_colmap.yaml`.
The removed legacy launcher is preserved only in historical source snapshots;
new runs use the generic `farm` CLI.

## Inputs

- virtual PINHOLE COLMAP: `../data/factory_pinhole_multi/sparse/0`;
- virtual RGB views: `../data/factory_pinhole_multi/images`;
- aligned 3DGS: `../data/factory_scene_last.ply`;
- Hugging Face secret: `../secrets.json`, mounted read-only and never persisted;
- output root: `../output/farm_pipeline`.

The full five-view conversion is validated at preflight. Mapping uses the
geometry-selected cam00+cam01 centre-view backbone, avoiding all 6310 virtual
images while preserving COLMAP connectivity.

## Commands

```bash
farm validate-plan --config configs/scenes/factory_3dgs_colmap.yaml
farm run --config configs/scenes/factory_3dgs_colmap.yaml
farm status --config configs/scenes/factory_3dgs_colmap.yaml --json
farm serve --config configs/scenes/factory_3dgs_colmap.yaml --port 8081 --runtime docker
farm stop --config configs/scenes/factory_3dgs_colmap.yaml
```

Resume the latest interrupted attempt:

```bash
farm run --config configs/scenes/factory_3dgs_colmap.yaml --resume
```

Successful runs are immutable. Use a new run ID for a fresh experiment.

## Verified clean result

`factory-generic-v7` is the single retained autonomous factory bundle. Its
canonical stages completed in 776.04 seconds. Selection converged to 174
timestamps / 348 views with zero residual graph edges. Final QA retained 49
metric multi-view objects; all 49 have captions and both text/VL embeddings.
Peak selected-device usage was 27.51 GiB, with 25.36 GiB incremental delta.
These are whole-device measurements, not per-process attribution.

Use `final/scene_state.pt`, `final/catalog.json`,
`final/presentation_catalog.json` and `final/cloud.npz` downstream. The full
evidence trail remains under `qa/`, and `viewer/launch.sh` opens the offline
click-to-inspect presentation with a default point size of 0.004 m.
`reports/self_containment.json` records required artifact hashes and
`reports/FARM-Project-source-without-models.tar.gz` freezes the exact source
tree used for the consolidated result. Semantic lineage inputs are materialized
under `qa/semantics/consensus/evidence/`; no predecessor run is required.

## Quality boundary

The detector is a static open-vocabulary YOLOE profile. The result is an
evidence-backed catalog, not a guaranteed exhaustive inventory: very small,
occluded or out-of-vocabulary objects can still be missed. Rollout requires a
cold validation on several scenes with different rigs, scales and content.

The current standard semantic implementation also includes tight mask-grounded
object crops, conflict-only blind reconciliation and a non-regressive consensus
guard. These are generic pipeline stages; the factory configuration supplies no
object names or per-ID corrections.
