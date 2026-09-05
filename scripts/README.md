# FARM tools

The supported scene pipeline entry point is `farm` (`scripts/farm_pipeline.py`
in a source checkout). `farm quality --help` lists measured development runs.
Run Python commands inside the pinned FARM Docker environment.

| Directory | Purpose |
|---|---|
| `geometry/` | 2D mask refinement, 3D association, OBB fitting, assemblies and geometric audits |
| `semantics/` | Vocabulary proposals, crop reviews, labels, captions and semantic reconciliation |
| `evaluation/` | Frozen folds, heldout comparison, eligibility, release gates and reports |
| `tracking/` | Tracker episodes and cross-view identity experiments |
| `grouping/` | Native Gaussian identity grouping experiments |
| `quality/` | Baseline preservation, inspection packets and annotation interchange |

Root-level tools handle scene ingress, rendering, selection, environment setup,
pipeline execution and viewers. Reusable algorithms live in `src/`; the exact
source-Gaussian renderer bridge lives in `tools/farm_shaper_bridge/`.

Examples:

```bash
farm quality discovery --help
farm quality lift-ablation --help
python scripts/geometry/refine_farm_full_colmap_masks.py --help
python scripts/quality/prepare_farm_manual_gold.py --help
```

Historical experiment manifests retain their original commands and source
snapshots. For current development use the paths above; no duplicate wrapper
files are kept under the old flat layout. Experiment images, arrays, model
caches and reports belong under the configured scene output root, not here.
