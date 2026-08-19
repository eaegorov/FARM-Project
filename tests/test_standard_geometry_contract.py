from __future__ import annotations

import inspect
import json
import os
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
import torch
from PIL import Image

from scripts.farm_standard_stage import (
    Context,
    resolve_model_contract,
    validate_canonical_visuals,
)
from farm_pipeline.resources import load_model_manifest
from scripts.validate_farm_geometry import validate_state


def _metric_state(object_ids: list[int]) -> dict:
    count = len(object_ids)
    return {
        "means": np.zeros((count, 3), dtype=np.float32),
        "cov6": np.zeros((count, 6), dtype=np.float32),
        "object_id": np.asarray(object_ids, dtype=np.int64),
        "active": np.ones(count, dtype=bool),
        "object_box_dimensions_m": np.ones((count, 3), dtype=np.float32),
        "object_box_centers_m": np.zeros((count, 3), dtype=np.float32),
        "object_box_wxyz": np.tile([[1.0, 0.0, 0.0, 0.0]], (count, 1)),
        "object_geometry_status": ["geometry_pass"] * count,
        "object_assembly_member_ids": [[] for _ in object_ids],
        "object_geometry_up_vector": [0.0, 1.0, 0.0],
    }


def test_structural_gate_allows_only_documented_surface_demotions() -> None:
    final = _metric_state([10])
    direct = _metric_state([10, 20])
    failed = validate_state(final, resolved_up=[0, 1, 0], direct_state=direct)
    assert failed["status"] == "FAIL"
    passed = validate_state(
        final, resolved_up=[0, 1, 0], direct_state=direct,
        allowed_demoted_direct_ids=[20],
    )
    assert passed["status"] == "PASS"
    assert passed["documented_surface_demotions"] == 1


def test_structural_gate_separates_assembly_surface_demotions() -> None:
    final = _metric_state([10, 30])
    final["object_assembly_member_ids"] = [[], [10]]
    final["object_geometry_status"][1] = "assembly_geometry_pass"
    direct = _metric_state([10])
    report = validate_state(
        final, resolved_up=[0, 1, 0], direct_state=direct,
        allowed_demoted_direct_ids=[30],
    )
    assert report["status"] == "PASS"
    assert report["documented_direct_demotions"] == 0
    assert report["documented_assembly_demotions"] == 1


def test_structural_gate_rejects_corrupt_refined_or_compound_obb() -> None:
    final = _metric_state([10, 30])
    final["active"][1] = False
    final["object_geometry_status"][1] = "assembly_compound_geometry_probable"
    final["object_assembly_member_ids"] = [[], [10]]
    final["object_compound_boxes"] = [[], [
        {"center_m": [0, 0, 0], "dimensions_lwh_m": [1, 1, 1], "wxyz": [1, 0, 0, 0]},
        {"center_m": [1, 0, 0], "dimensions_lwh_m": [1, 1, 0], "wxyz": [1, 0, 0, 0]},
    ]]
    report = validate_state(final, resolved_up=[0, 0, 1])
    assert report["status"] == "FAIL"
    assert any("child 1" in error for error in report["errors"])
    final["object_box_centers_m"][0, 0] = np.nan
    report = validate_state(final, resolved_up=[0, 1, 0])
    assert any("refined OBB centres" in error for error in report["errors"])


def test_model_contract_rejects_unpinned_configured_ids() -> None:
    services = {
        "caption": SimpleNamespace(model=SimpleNamespace(repo_id="caption/repo")),
        "text-embed": SimpleNamespace(model=SimpleNamespace(repo_id="text/repo")),
        "vl-embed": SimpleNamespace(model=SimpleNamespace(repo_id="vl/repo")),
    }
    models = SimpleNamespace(
        pipeline_models={"yoloe-v8l-seg": object(), "dinov3-vits16plus": object()},
        services=services,
    )
    policy = SimpleNamespace(
        segmentation="yoloe-v8l", visual_features="dinov3-vits16plus",
        caption="caption/repo", text_embeddings="text/repo",
    )
    resolved = resolve_model_contract(SimpleNamespace(models=policy), models)
    assert resolved["segmentation_manifest_key"] == "yoloe-v8l-seg"
    policy.caption = "silent/fallback"
    try:
        resolve_model_contract(SimpleNamespace(models=policy), models)
    except ValueError as error:
        assert "caption model" in str(error)
    else:
        raise AssertionError("unpinned model ID must fail closed")


def test_rgbd_adapter_uses_generic_preparer_identity_regex_contract() -> None:
    source = inspect.getsource(Context.rgbd)
    assert '"--identity-regex"' in source
    assert '"--name-regex"' not in source
    assert '".stage_work/rgbd"' in source
    assert "replace_directory(work_output" in source
    assert ':/farm-run"' in source
    assert ':/run"' not in source
    assert 'cleanup_stage_work("rgbd")' in source


def test_main_runtime_uses_declared_non_root_uid_and_host_gid() -> None:
    context = object.__new__(Context)
    context.prefix = "farm-test"
    context.scope = "rtest"
    context.counter = 0
    context.models = SimpleNamespace(
        runtimes={"main": SimpleNamespace(user_uid=1000)}
    )
    command, _ = context.docker_base("mapping", runtime="main")
    assert command[command.index("--user") + 1] == f"1000:{os.getgid()}"
    assert "FARM_RUN_DIR=/farm-run" in command
    wrapper = Path(__file__).resolve().parents[1] / "docker/python-entrypoint.sh"
    wrapper_source = wrapper.read_text(encoding="utf-8")
    assert "umask 0002" in wrapper_source
    assert 'FARM_RUN_DIR:-/run' in wrapper_source
    assert 'find "$run_dir" -xdev' in wrapper_source
    assert '-path "$source_snapshot_dir"' in wrapper_source
    assert '-uid "$uid"' in wrapper_source
    mount_source = inspect.getsource(Context.prepare_main_mount)
    assert "info.st_uid != os.getuid()" in mount_source
    assert "path.is_symlink()" in mount_source
    assert '"config/source_snapshot"' in mount_source
    assert "snapshot_root in path.parents" in mount_source


def test_standard_wrapper_validates_snapshot_before_loading_scene_config() -> None:
    init_source = inspect.getsource(Context.__init__)
    assert init_source.index("_validate_execution_contract") < init_source.index(
        "load_scene_config"
    )
    for method in (Context.post, Context.prep_post, Context.rgbd, Context.mapping):
        assert 'f"{ROOT}:' in inspect.getsource(method)


def test_factory_manifest_pins_mobileclip_required_by_yoloe() -> None:
    manifest = load_model_manifest(
        Path(__file__).resolve().parents[1] / "configs/models/farm_models.v1.json"
    )
    mobileclip = manifest.pipeline_models["mobileclip"]
    assert mobileclip.local_path.name == "mobileclip_blt.pt"
    assert mobileclip.sha256 == "670844f7a886dd6eff7a9285adfc53f3d3c889c03bfc8354010cb5c6bf27441a"


def test_presentation_uses_pinned_prep_dependency_closure() -> None:
    init_source = inspect.getsource(Context.__init__)
    assert 'self.prep_python = self.models.runtimes["prep"].python' in init_source
    assert '"--entrypoint", self.prep_python, self.prep_image' in inspect.getsource(
        Context.prep_post
    )
    assert '"--entrypoint", self.prep_python, self.prep_image' in inspect.getsource(
        Context.rgbd
    )
    assert "/opt/conda/envs/rest3d/bin/python" not in inspect.getsource(Context.prep_post)
    assert "/opt/conda/envs/rest3d/bin/python" not in inspect.getsource(Context.rgbd)
    assert 'self.prep_post("prepare_farm_presentation.py"' in inspect.getsource(
        Context.presentation
    )
    assert 'self.prep_post("prepare_farm_presentation.py"' in inspect.getsource(
        Context.semantics
    )


def test_qa_bundle_indexes_all_canonical_visuals_and_cleans_workspace() -> None:
    source = inspect.getsource(Context.qa_bundle)
    for index in range(1, 9):
        assert f'visuals/0{index}_' in source
    assert "all eight canonical QA dashboards" in source
    assert "all six named final-scene visualizations" in source
    assert '"06_label_uncertainty_sample_4k.jpg"' in source
    assert '"--presentation-catalog", "/farm-run/final/presentation_catalog.json"' in source
    assert '"--dedup-audit", "/farm-run/qa/dedup/audit.json"' in source
    assert '"--mode", "verify"' in source
    assert '"--acceptance-report", "/farm-run/qa/acceptance/result.json"' in source
    assert '"release_status": release_status' in source
    assert "validate_canonical_visuals(images)" in source
    assert 'viewer="../viewer/result.json"' in source
    assert 'self.post("build_farm_retention_funnel.py"' in source
    assert 'cleanup_stage_work("qa_bundle")' in source


def test_canonical_visual_contract_decodes_and_enforces_4k(tmp_path: Path) -> None:
    exact = tmp_path / "01_dashboard_4k.jpg"
    ordinary = tmp_path / "02_contact_sheet.png"
    Image.new("RGB", (3840, 2160), "black").save(exact, quality=60)
    Image.new("RGB", (320, 180), "white").save(ordinary)
    validate_canonical_visuals([exact, ordinary])

    wrong = tmp_path / "03_wrong_4k.png"
    Image.new("RGB", (3072, 1728), "black").save(wrong)
    with pytest.raises(RuntimeError, match="expected 3840x2160"):
        validate_canonical_visuals([wrong])

    corrupt = tmp_path / "04_corrupt.jpg"
    corrupt.write_bytes(b"not an image")
    with pytest.raises(RuntimeError, match="cannot decode"):
        validate_canonical_visuals([corrupt])


def test_qa_result_viewer_path_resolves_from_qa_directory(tmp_path: Path) -> None:
    context = object.__new__(Context)
    context.run_dir = tmp_path
    context.stage = "qa_bundle"
    context.context_path = tmp_path / "missing-context.json"
    context.config = SimpleNamespace(scene_id="synthetic")

    context.result("PASS", viewer="../viewer/result.json")
    payload = json.loads(context.result_path.read_text(encoding="utf-8"))

    assert (context.result_path.parent / payload["viewer"]).resolve() == (
        tmp_path / "viewer/result.json"
    ).resolve()


def test_scoped_cleanup_prunes_empty_in_run_runtime_directory() -> None:
    assert "runtime_dir.rmdir()" in inspect.getsource(Context.stop_services)


def test_semantic_processing_state_preserves_geometry_valid_evidence_tiers() -> None:
    source = inspect.getsource(Context.semantics)
    assert '"/farm-run/qa/semantics/consensus/semantic_consensus_catalog.json"' in source
    assert '"/farm-run/qa/semantics/ensemble/semantic_ensemble_review.json"' in source
    assert '"/farm-run/qa/semantics/reconciliation/semantic_reconciled_catalog.json"' in source
    assert '"--no-resolved-only"' in source
    assert '"--min-observations", "3"' in source
    assert '"/farm-run/qa/mapping/analysis/reliable_objects_min3.json"' in source
    assert '"--max-camera-distance-m", "0"' in source
    semantic_presentation = source[source.index('self.prep_post("prepare_farm_presentation.py"'):]
    assert '"/farm-run/qa/semantics/catalog/reviewed_robust_catalog.json"' not in semantic_presentation


def test_surface_support_is_diagnostic_not_destructive() -> None:
    source = inspect.getsource(Context.surface_support)
    assert '"--no-enforce"' in source
    assert '"--enforce"' not in source


def test_finalization_uses_last_cross_pass_semantic_consensus() -> None:
    source = inspect.getsource(Context.finalize)
    assert (
        '"/farm-run/qa/semantics/consensus/semantic_consensus_catalog.json"'
        in source
    )
    assert '"/farm-run/qa/semantics/ensemble/semantic_tiered_catalog.json"' not in source
    assert 'self.post("build_farm_final_acceptance.py"' in source
    assert (
        '"--assembly-review", "/farm-run/qa/assemblies/review/reviewed_robust_objects.json"'
        in source
    )
    assert '"--mode", "apply"' in source
    assert "atomic_copy(accepted" in source
    assert "replace_directory(acceptance_work" in source
    assert source.index('self.post("build_farm_final_acceptance.py"') < source.index(
        "atomic_copy(accepted"
    )


def test_pre_qa_wall_seconds_uses_only_completed_prefinal_stage_states(tmp_path: Path) -> None:
    context = object.__new__(Context)
    context.run_dir = tmp_path
    for stage, value in (("selection", 4.5), ("mapping", 10), ("finalize", 99)):
        path = tmp_path / "stages" / stage / "state.json"
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps({"duration_seconds": value}), encoding="utf-8")

    assert context.pre_qa_wall_seconds() == 14.5


def test_generic_visual_stages_receive_real_scene_id() -> None:
    geometry = inspect.getsource(Context.geometry)
    visual = inspect.getsource(Context.visual_consistency)
    assert geometry.count('"--scene-id", self.config.scene_id') == 2
    assert visual.count('"--scene-id", self.config.scene_id') == 1


def test_standard_presentation_metadata_is_run_relative() -> None:
    presentation = inspect.getsource(Context.presentation)
    semantics = inspect.getsource(Context.semantics)
    assert '"--metadata-source-scene-state", "../scene_state_raw.pt"' in presentation
    assert '"--metadata-source-scene-state", "../scene_state_visual.pt"' in semantics
    assert '"--metadata-reviewed-catalog", "../../qa/semantics/consensus/semantic_consensus_catalog.json"' in semantics


def test_final_catalog_exports_refined_metric_obb(tmp_path: Path) -> None:
    context = object.__new__(Context)
    context.run_dir = tmp_path
    (tmp_path / "final").mkdir()
    state = {
        "object_id": torch.tensor([7]),
        "active": torch.tensor([True]),
        "means": torch.tensor([[1.0, 2.0, 3.0]]),
        "object_box_centers_m": torch.tensor([[4.0, 5.0, 6.0]]),
        "object_box_dimensions_m": torch.tensor([[1.0, 2.0, 3.0]]),
        "object_box_wxyz": torch.tensor([[1.0, 0.0, 0.0, 0.0]]),
        "object_category": ["machine"],
        "object_caption": ["compact machine"],
        "object_key_attributes": [["metal"]],
        "object_semantic_tier": ["confirmed"],
        "object_semantic_status": ["vlm_ensemble_confirmed"],
        "object_evidence_tier": ["stable"],
        "object_compound_boxes": [[]],
    }
    torch.save({"state": state}, tmp_path / "final/scene_state.pt")
    context.export_catalog()
    row = json.loads((tmp_path / "final/catalog.json").read_text())[0]
    presentation_row = json.loads(
        (tmp_path / "final/presentation_catalog.json").read_text()
    )[0]
    assert row["center_m"] == [4.0, 5.0, 6.0]
    assert row["wxyz"] == [1.0, 0.0, 0.0, 0.0]
    assert row["semantic_tier"] == "confirmed"
    assert row["metric_active"] is True
    assert row["presentation_visible"] is True
    assert presentation_row["id"] == 7


def test_standard_qa_profile_reports_whole_device_peak_and_delta(tmp_path: Path) -> None:
    context = object.__new__(Context)
    context.run_dir = tmp_path
    context.config = SimpleNamespace(scene_id="telemetry_scene")
    context.context_path = tmp_path / "input/resolved_context.json"
    context.context_path.parent.mkdir(parents=True)
    context.context_path.write_text(
        json.dumps({"selected_gpu": {"uuid": "GPU-TEST"}}),
        encoding="utf-8",
    )
    state_path = tmp_path / "stages/selection/state.json"
    state_path.parent.mkdir(parents=True)
    state_path.write_text(
        json.dumps(
            {
                "status": "succeeded",
                "duration_seconds": 4.5,
                "telemetry_summary": {
                    "gpu_device_baseline_used_by_uuid_mb": {"GPU-TEST": 2048},
                    "gpu_device_peak_used_by_uuid_mb": {"GPU-TEST": 6144},
                    "gpu_device_peak_delta_by_uuid_mb": {"GPU-TEST": 4096},
                },
            }
        ),
        encoding="utf-8",
    )

    context.qa_profile()

    timing = json.loads((tmp_path / "timing/qa_snapshot.json").read_text())
    resources = json.loads((tmp_path / "qa/resource_summary.json").read_text())
    assert timing["schema"] == "farm.qa-timing-snapshot.v2"
    assert timing["stages"][0]["device_peak_used_gib"] == 6.0
    assert timing["stages"][0]["incremental_peak_delta_gib"] == 4.0
    assert resources["schema"] == "farm.qa-resource-snapshot.v2"
    assert resources["device_peak_used_gib"] == 6.0
    assert resources["incremental_peak_delta_gib"] == 4.0
    assert (
        resources["measurement_scope"]
        == "whole_selected_device_not_pid_attributed"
    )
    assert "peak_vram_gib" not in resources
