# Script inventory

The supported production entry point is `farm` / `scripts/farm_pipeline.py`.
Canonical `farm.scene.v1` stages are declared in
`src/farm_runtime/standard.py`; their implementation programs remain in
`scripts/` and are fingerprinted per stage for safe resume invalidation.
See [PRODUCTION_PIPELINE.md](PRODUCTION_PIPELINE.md) for the end-to-end
production commands and release boundary.

## Canonical pipeline and preparation

- Runtime: `farm_pipeline.py`, `farm_standard_stage.py`, `farm_preflight.py`,
  `farm_resource_preflight.py`, `farm_runtime_inventory.py`.
- Inputs: `convert_colmap_kb4_to_pinhole.py`,
  `select_colmap_keyframes.py`, `prepare_colmap_3dgs_rgbd.py`.
- Mapping QA, geometry and semantics: every `farm_*`, `analyze_farm_*`,
  `audit_farm_gaussian_support.py`, `build_farm_*`, `review_farm_*`,
  `reconcile_farm_semantics.py`, `finalize_farm_semantic_consensus.py`,
  `refine_farm_*`, `resolve_farm_track_duplicates.py`,
  `unify_farm_compound_presentation.py`, `validate_farm_geometry.py`, and
  the generic `plot_farm_*` / `visualize_farm_scene_state.py` programs named
  by the standard stage map.
- Viewer/report: `view_scene_state.py`, `build_farm_run_report.py`.
- Models/services: `download_farm_models.py`,
  `download_dinov3_vits16plus.py`, `hf_download_with_secret.py`,
  `start_farm_vllm.sh`.
- Portability: `pin_farm_runtime_images.py` audits or atomically updates only
  manifest runtime image IDs after inspecting every configured Docker tag.
- Unified viewer: `serve_farm_unified_viewer.py` serves the validated scene
  registry on loopback port 8080 by default. It is an unauthenticated
  single-user inspection service; its all-N source layer is a float16/uint8
  Viser DC preview, not a native-precision 3DGS renderer.
- Reconstruction overlay: `serve_farm_recon_overlay.py` validates a completed
  `splatica.farm-mv-sam3d.v1` bundle, preserves every object ID in a stable
  browser sample, and overlays its FARM-world Gaussian PLY on a metric scene
  cloud with explicit orbit/focus controls.

## Dense lift and ShapeR bridge

`tools/farm_shaper_bridge/` contains the exact contributor lift launcher,
ShapeR runtime manager, verified-instance input builder, pinned inference
runner, Docker-wrapped `run_bridge_stage.py prepare/assemble` host control,
and metric-scene assembler. These are downstream products with their own
release markers; they do not mutate the FARM run or source PLY. The
prepare/assemble wrapper runs from the pinned control venv; runtime management
uses the separate exact `bridge-control.lock.txt` launcher because its direct
NumPy/SciPy/Torch pins conflict with the control-plane closure. Inference
computation remains in pinned images.

## Upstream research/evaluation utilities

`eval_*`, `run_scene_graph_*`, `render_hm3d_trajectory.py`,
`convert_ours_to_canonical.py`, `inspect_pipeline_trace.py`,
`run_pipeline.py`, `query_scene_graph.py`, `score_largescale_predictions.py`
and sensor conversion helpers are retained because they belong to upstream
FARM evaluation/debug workflows, not generated experiment clutter.

## Removed superseded factory experiment tools

The old factory selector/RGB-D launcher, H100 multi-service launcher, V6/V7/V8
management-board renderers and one-off benchmark/regression plots were removed.
They duplicated the generic stages, contained factory paths or fixed object
policies, and were not referenced by the canonical DAG or tests. Historical
commands remain in immutable historical run artifacts only. Those artifacts
do not prove that the current checkout passes its acceptance contract.
