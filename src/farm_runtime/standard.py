from __future__ import annotations

from pathlib import Path
from typing import Any

from farm_pipeline.resources import load_model_manifest
from farm_pipeline.scene_config import load_scene_config


STANDARD_STAGE_IDS = (
    "preflight",
    "selection",
    "rgbd",
    "mapping",
    "mapping_qa",
    "presentation",
    "geometry",
    "visual_consistency",
    "semantics",
    "part_whole",
    "assemblies",
    "surface_support",
    "compound_geometry",
    "geometry_qa",
    "dedup",
    "finalize",
    "qa_bundle",
)


# A standard stage is executed through ``farm_standard_stage.py``, but the
# wrapper delegates almost all material work to one or more stage-specific
# programs. Keep those code paths in the stage fingerprint so that resuming a
# run after a code update cannot silently reuse artifacts produced by an older
# implementation. The linear dependency chain then invalidates every
# downstream stage without unnecessarily invalidating earlier work.
STANDARD_SHARED_CODE_INPUTS = (
    "src/farm_pipeline/resources.py",
    "src/farm_pipeline/scene_config.py",
    "src/farm_runtime/process.py",
    "src/farm_runtime/integrity.py",
    "src/farm_runtime/source_snapshot.py",
    "src/farm_runtime/standard.py",
)


STANDARD_STAGE_CODE_INPUTS: dict[str, tuple[str, ...]] = {
    "preflight": (
        "scripts/farm_preflight.py",
        "scripts/farm_resource_preflight.py",
        "src/farm_pipeline/preflight.py",
    ),
    "selection": ("scripts/select_colmap_keyframes.py", "src/farm_runtime/angular_discovery.py", "src/scene_graph/captioning/evidence.py"),
    "rgbd": ("scripts/prepare_colmap_3dgs_rgbd.py",),
    "mapping": (
        "docker/entrypoint.sh",
        "configs/yoloe_vocabulary.txt",
        "scripts/semantics/build_farm_adaptive_vocabulary.py",
        "src/scene_graph/offline/run.py",
        "src/scene_graph/config.py",
        "src/scene_graph/map_update/filtering.py",
        "src/scene_graph/map_update/mask_observations.py",
        "src/scene_graph/pipeline/orchestrator.py",
        "src/scene_graph/segmentation/overlap.py",
        "src/scene_graph/segmentation/yoloe.py",
        "ros/mapping/mapping/lib/parameters.py",
        "ros/mapping/mapping/nodes/streaming_mapper.py",
    ),
    "mapping_qa": (
        "scripts/evaluation/analyze_farm_scene_quality.py",
        "scripts/geometry/farm_geometry_axes.py",
    ),
    "presentation": ("scripts/geometry/prepare_farm_presentation.py",),
    "geometry": (
        "scripts/geometry/refine_farm_object_geometry.py",
        "scripts/geometry/farm_geometry_axes.py",
        "scripts/geometry/plot_farm_geometry_audit.py",
        "scripts/geometry/plot_farm_obb_reprojections.py",
        "src/scene_graph/map_update/mask_observations.py",
    ),
    "visual_consistency": (
        "scripts/geometry/refine_farm_visual_consistency.py",
        "scripts/geometry/plot_farm_visual_consistency.py",
    ),
    "semantics": (
        "scripts/geometry/audit_farm_gaussian_support.py",
        "scripts/semantics/review_farm_object_crops.py",
        "scripts/semantics/export_reviewed_farm_catalog.py",
        "scripts/semantics/adjudicate_farm_semantics.py",
        "scripts/semantics/reconcile_farm_semantics.py",
        "scripts/semantics/finalize_farm_semantic_consensus.py",
        "scripts/geometry/prepare_farm_presentation.py",
        "scripts/start_farm_vllm.sh",
        "src/scene_graph/captioning/evidence.py",
        "src/scene_graph/captioning/label_contract.py",
        "src/scene_graph/captioning/physical_guard.py",
        "src/scene_graph/map_update/mask_observations.py",
    ),
    "part_whole": (
        "scripts/geometry/analyze_farm_part_whole.py",
        "scripts/geometry/apply_farm_full_colmap_rescue.py",
        "scripts/evaluation/build_farm_full_colmap_folds.py",
        "scripts/geometry/merge_farm_full_colmap_rescue.py",
        "scripts/geometry/plan_farm_full_colmap_rescue.py",
        "scripts/prepare_colmap_3dgs_rgbd.py",
        "src/farm_runtime/input_registration.py",
        "src/farm_runtime/camera_alignment.py",
        "scripts/geometry/refine_farm_full_colmap_masks.py",
        "scripts/geometry/refine_farm_object_geometry.py",
        "scripts/semantics/review_farm_object_crops.py",
        "scripts/start_farm_vllm.sh",
        "src/farm_runtime/assembly_mask_reconstruction.py",
        "src/farm_runtime/full_colmap_folds.py",
        "src/farm_runtime/full_colmap_refinement_contract.py",
        "src/farm_runtime/full_colmap_rescue.py",
        "src/farm_runtime/full_colmap_view_planner.py",
        "src/farm_runtime/source_state_canonicalization.py",
        "src/farm_runtime/source_view_canonicalization.py",
        "src/scene_graph/captioning/evidence.py",
        "src/scene_graph/captioning/label_contract.py",
        "src/scene_graph/map_update/mask_observations.py",
        "src/scene_graph/offline/run.py",
    ),
    "assemblies": (
        "scripts/geometry/build_farm_object_assemblies.py",
        "scripts/semantics/review_farm_object_crops.py",
        "scripts/geometry/refine_farm_object_geometry.py",
        "scripts/semantics/enrich_farm_embeddings.py",
        "scripts/geometry/farm_geometry_axes.py",
        "scripts/start_farm_vllm.sh",
        "src/scene_graph/captioning/evidence.py",
        "src/scene_graph/captioning/label_contract.py",
        "src/scene_graph/map_update/mask_observations.py",
    ),
    "surface_support": (
        "scripts/geometry/audit_farm_gaussian_support.py",
        "scripts/geometry/refine_farm_object_geometry.py",
        "src/scene_graph/map_update/mask_observations.py",
    ),
    "compound_geometry": (
        "scripts/geometry/refine_farm_compound_geometry.py",
        "scripts/geometry/refine_farm_compound_presentation.py",
        "scripts/geometry/refine_farm_object_geometry.py",
        "scripts/geometry/farm_geometry_axes.py",
    ),
    "geometry_qa": ("scripts/geometry/validate_farm_geometry.py",),
    "dedup": (
        "scripts/geometry/unify_farm_compound_presentation.py",
        "scripts/geometry/resolve_farm_track_duplicates.py",
    ),
    "finalize": (
        "scripts/geometry/build_farm_tiered_state.py",
        "scripts/evaluation/build_farm_final_acceptance.py",
        "src/farm_pipeline/final_acceptance.py",
        "src/scene_graph/captioning/evidence.py",
    ),
    "qa_bundle": (
        "scripts/evaluation/build_farm_run_report.py",
        "scripts/evaluation/build_farm_retention_funnel.py",
        "scripts/evaluation/analyze_farm_scene_quality.py",
        "scripts/visualize_farm_scene_state.py",
        "scripts/evaluation/build_farm_final_acceptance.py",
        "src/farm_pipeline/final_acceptance.py",
        "src/scene_graph/captioning/evidence.py",
        "scripts/view_scene_state.py",
        "scripts/geometry/farm_geometry_axes.py",
        "src/scene_graph/visualization/viser_visualizer.py",
    ),
}


def _project_root(config_path: Path, override: str | Path | None) -> Path:
    if override is not None:
        return Path(override).expanduser().resolve(strict=False)
    for candidate in config_path.parents:
        if (candidate / "pyproject.toml").is_file() and (candidate / "src").is_dir():
            return candidate
    return config_path.parent


def _stage(stage_id: str, needs: list[str], outputs: list[str], result: str) -> dict[str, Any]:
    wrapper = "${execution_project_root}/scripts/farm_standard_stage.py"
    code_inputs = [
        f"${{execution_project_root}}/{relative_path}"
        for relative_path in (
            *STANDARD_SHARED_CODE_INPUTS,
            *STANDARD_STAGE_CODE_INPUTS[stage_id],
        )
    ]
    return {
        "id": stage_id,
        "description": f"Canonical FARM stage: {stage_id}",
        "needs": needs,
        "command": [
            "${python_executable}",
            wrapper,
            "--config",
            "${config_path}",
            "--run-dir",
            "${run_dir}",
            "--stage",
            stage_id,
        ],
        "cwd": "${execution_project_root}",
        "env": {"PYTHONDONTWRITEBYTECODE": "1"},
        "inputs": [wrapper],
        "outputs": outputs,
        "pass_json": [result],
        "fingerprint_inputs": [wrapper, *code_inputs, "${config_path}"],
        "fingerprint_mode": "metadata",
    }


def compile_standard_scene(
    config_path: Path,
    *,
    project_root: str | Path | None = None,
) -> dict[str, Any]:
    """Compile the minimal farm.scene.v1 contract into the canonical full DAG."""

    scene = load_scene_config(config_path)
    root = _project_root(config_path, project_root)
    run = "${run_dir}"
    results = {
        "preflight": f"{run}/input/preflight.json",
        "selection": f"{run}/selection/result.json",
        "rgbd": f"{run}/rgbd/result.json",
        "mapping": f"{run}/mapping/result.json",
        "mapping_qa": f"{run}/qa/mapping/result.json",
        "presentation": f"{run}/mapping/presentation_result.json",
        "geometry": f"{run}/qa/geometry/result.json",
        "visual_consistency": f"{run}/qa/visual_consistency/result.json",
        "semantics": f"{run}/qa/semantics/result.json",
        "part_whole": f"{run}/qa/part_whole/result.json",
        "assemblies": f"{run}/qa/assemblies/result.json",
        "surface_support": f"{run}/qa/surface_support/result.json",
        "compound_geometry": f"{run}/qa/compound_geometry/result.json",
        "geometry_qa": f"{run}/qa/geometry_qa/result.json",
        "dedup": f"{run}/qa/dedup/result.json",
        "finalize": f"{run}/final/result.json",
        "qa_bundle": f"{run}/qa/result.json",
    }
    extra_outputs = {
        "preflight": [f"{run}/input/resolved_context.json"],
        "selection": [f"{run}/selection/selected_names.txt"],
        "rgbd": [f"{run}/rgbd/frames.json"],
        "mapping": [f"{run}/mapping/scene_state_raw.pt"],
        "geometry": [f"{run}/mapping/scene_state_geometry.pt"],
        "visual_consistency": [f"{run}/mapping/scene_state_visual.pt"],
        "semantics": [
            f"{run}/mapping/scene_state_surface_prefilter.pt",
            f"{run}/mapping/scene_state_semantic.pt",
            f"{run}/qa/surface_prefilter/audit.json",
        ],
        "assemblies": [f"{run}/mapping/scene_state_assemblies.pt"],
        "surface_support": [f"{run}/mapping/scene_state_surface.pt"],
        "compound_geometry": [f"{run}/mapping/scene_state_compound.pt"],
        "dedup": [f"{run}/mapping/scene_state_dedup.pt"],
        "finalize": [
            f"{run}/final/scene_state.pt",
            f"{run}/final/catalog.json",
            f"{run}/final/cloud.npz",
            f"{run}/qa/acceptance/result.json",
            f"{run}/qa/acceptance/duplicate_clusters.json",
            f"{run}/qa/acceptance/label_uncertainty.json",
        ],
        "qa_bundle": [
            f"{run}/visuals/index.json",
            f"{run}/viewer/result.json",
            f"{run}/viewer/launch.sh",
            f"{run}/qa/summary.json",
            f"{run}/qa/retention_funnel.json",
        ],
    }
    stages: list[dict[str, Any]] = []
    previous: str | None = None
    for stage_id in STANDARD_STAGE_IDS:
        outputs = [results[stage_id], *extra_outputs.get(stage_id, [])]
        stages.append(_stage(stage_id, [previous] if previous else [], outputs, results[stage_id]))
        previous = stage_id

    normalized = scene.normalized_dict()
    stages[0]["fingerprint_inputs"].extend(
        [
            normalized["inputs"]["colmap_model"],
            normalized["inputs"]["image_root"],
            normalized["inputs"]["gaussian_ply"],
        ]
    )
    model_manifest_path = scene.resources.model_manifest
    if model_manifest_path is not None:
        models = load_model_manifest(model_manifest_path)
        model_inputs = [models.manifest_path]
        model_inputs.extend(spec.local_path for spec in models.pipeline_models.values())
        model_inputs.extend(spec.model.local_path for spec in models.services.values())
        seen = set(stages[0]["fingerprint_inputs"])
        for model_input in model_inputs:
            value = str(model_input)
            if value not in seen:
                stages[0]["fingerprint_inputs"].append(value)
                seen.add(value)
    return {
        "schema_version": 1,
        "project_root": str(root),
        "scene": {
            "id": scene.scene_id,
            "inputs": normalized["inputs"],
            "standard_contract": "farm.scene.v1",
        },
        "output": {"root": str(scene.output_root)},
        "pipeline": {
            "telemetry_interval_seconds": 1.0,
            "stages": stages,
            "finalizers": [
                {
                    "command": [
                        "${python_executable}",
                        "${execution_project_root}/scripts/farm_standard_stage.py",
                        "--config",
                        "${config_path}",
                        "--run-dir",
                        "${run_dir}",
                        "--cleanup",
                    ],
                    "cwd": "${execution_project_root}",
                    "env": {"PYTHONDONTWRITEBYTECODE": "1"},
                    "timeout_seconds": 90.0,
                }
            ],
        },
        "artifacts": {
            "scene_state": f"{run}/final/scene_state.pt",
            "catalog": f"{run}/final/catalog.json",
            "presentation_catalog": f"{run}/final/presentation_catalog.json",
            "cloud": f"{run}/final/cloud.npz",
            "rgbd": f"{run}/rgbd",
            "mapping": f"{run}/mapping",
            "qa": f"{run}/qa/result.json",
            "qa_summary": f"{run}/qa/summary.json",
            "resolved_context": f"{run}/input/resolved_context.json",
            "scene_preflight": f"{run}/input/scene_preflight.json",
            "resource_preflight": f"{run}/input/resource_preflight.json",
            "final_acceptance": f"{run}/qa/acceptance/result.json",
        },
        "viewer": {
            "runtime": "docker",
            "command": [
                "${python_executable}",
                "${project_root}/scripts/view_scene_state.py",
                "--pt", "${artifact:scene_state}",
                "--cloud", "${artifact:cloud}",
                "--frames-dir", "${artifact:rgbd}",
                "--mapping-dir", "${artifact:mapping}",
                "--host", "${host}",
                "--port", "${port}",
                "--point-size", str(scene.viewer.scene_point_size_m),
                "--max-cloud-points", str(scene.viewer.max_context_points),
                "--resolved-context", "${artifact:resolved_context}",
                "--presentation",
                "--no-query",
                "--title", f"{scene.scene_id.replace('_', ' ').title()} Memory",
            ],
            "cwd": "${project_root}",
            "default_host": scene.viewer.host,
            "default_port": scene.viewer.port,
            "startup_timeout_seconds": 20.0,
            "health_url": "http://127.0.0.1:${port}/",
        },
        "source_scene": normalized,
    }
