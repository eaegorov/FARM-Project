import json
import sys
from pathlib import Path

import numpy as np
import pytest
import torch
from plyfile import PlyData, PlyElement


SCRIPTS = Path(__file__).resolve().parents[1] / "scripts"
if str(SCRIPTS) not in sys.path:
    sys.path.insert(0, str(SCRIPTS))

from scripts.geometry.prepare_farm_presentation import (  # noqa: E402
    build_demo_state,
    build_presentation_catalog,
    cloud_export_contract,
    export_cloud,
    main as presentation_main,
    validate_reusable_cloud,
)


def _state(path: Path) -> None:
    state = {
        "object_id": torch.tensor([7]),
        "means": torch.tensor([[1.0, 2.0, 3.0]]),
        "active": torch.tensor([True]),
        "object_image_ids": [[1, 2, 3]],
        "object_caption": [""],
        "object_category": [""],
        "object_supercategory": [""],
        "object_caption_decision": [""],
        "object_key_attributes": [[]],
        "object_box_centers_m": torch.tensor([[1.0, 2.0, 3.0]]),
        "object_box_dimensions_m": torch.tensor([[0.4, 0.5, 0.6]]),
        "object_box_wxyz": torch.tensor([[1.0, 0.0, 0.0, 0.0]]),
        "image_positions": [torch.tensor([1.0, 2.0, 3.2])],
    }
    torch.save({"state": state, "meta": {}}, path)


def test_geometry_only_catalog_semantics_survive_processing_state(tmp_path):
    source = tmp_path / "source.pt"
    output = tmp_path / "output.pt"
    catalog = tmp_path / "catalog.json"
    _state(source)
    catalog.write_text(json.dumps([{
        "id": 7,
        "category": "unresolved object",
        "description": "metric multi-view object with unresolved semantic identity",
        "attributes": ["rectangular"],
        "semantic_tier": "geometry_only",
        "semantic_status": "geometry_valid_semantics_unresolved",
        "review_decision": "unknown",
    }]), encoding="utf-8")
    build_demo_state(
        source, output, reviewed_catalog=catalog, min_observations=3,
        resolved_only=False, max_camera_distance_m=0.0,
    )
    state = torch.load(output, map_location="cpu", weights_only=False)["state"]
    assert bool(state["active"][0])
    assert state["object_category"][0] == "unresolved object"
    assert state["object_caption"][0]
    assert state["object_key_attributes"][0] == ["rectangular"]
    assert state["object_semantic_tier"][0] == "geometry_only"
    assert state["object_caption_decision"][0] == "unknown"


def test_presentation_provenance_can_be_stored_as_portable_paths(tmp_path):
    source = tmp_path / "source.pt"
    output = tmp_path / "output.pt"
    catalog = tmp_path / "catalog.json"
    _state(source)
    catalog.write_text("[]\n", encoding="utf-8")

    build_demo_state(
        source,
        output,
        reviewed_catalog=catalog,
        min_observations=3,
        resolved_only=False,
        max_camera_distance_m=0.0,
        metadata_source_scene_state="../scene_state.pt",
        metadata_reviewed_catalog="../../qa/semantics/consensus/catalog.json",
    )

    metadata = torch.load(output, map_location="cpu", weights_only=False)["meta"]["presentation"]
    assert metadata["source_scene_state"] == "../scene_state.pt"
    assert metadata["reviewed_catalog"] == "../../qa/semantics/consensus/catalog.json"
    assert str(tmp_path.resolve()) not in json.dumps(metadata)


def test_presentation_catalog_is_bound_to_filtered_state_geometry(tmp_path):
    source = tmp_path / "source.pt"
    filtered = tmp_path / "filtered.pt"
    catalog = tmp_path / "catalog.json"
    output = tmp_path / "presentation_catalog.json"
    _state(source)
    catalog.write_text(json.dumps([{
        "id": 7,
        "category": "stool",
        "description": "Low rectangular stool.",
        "semantic_tier": "probable",
        "semantic_status": "contextual_dual_panel_reviewed",
        "review_decision": "keep",
    }]), encoding="utf-8")
    build_demo_state(
        source, filtered, reviewed_catalog=catalog, min_observations=3,
        resolved_only=True, max_camera_distance_m=0.0,
    )

    summary = build_presentation_catalog(filtered, catalog, output)
    rows = json.loads(output.read_text(encoding="utf-8"))

    assert summary["visible_objects"] == 1
    assert rows == [{
        "id": 7,
        "category": "stool",
        "description": "Low rectangular stool.",
        "semantic_tier": "probable",
        "semantic_status": "contextual_dual_panel_reviewed",
        "review_decision": "keep",
        "center_m": [1.0, 2.0, 3.0],
        "dimensions_m": pytest.approx([0.4, 0.5, 0.6]),
        "wxyz": [1.0, 0.0, 0.0, 0.0],
        "observation_count": 3,
        "presentation_visible": True,
    }]



def test_presentation_catalog_excludes_suppressed_active_object(tmp_path):
    source = tmp_path / "source.pt"
    filtered = tmp_path / "filtered.pt"
    catalog = tmp_path / "catalog.json"
    output = tmp_path / "presentation_catalog.json"
    _state(source)
    catalog.write_text(json.dumps([{
        "id": 7,
        "category": "stool",
        "description": "Low rectangular stool.",
        "semantic_tier": "probable",
        "semantic_status": "contextual_dual_panel_reviewed",
        "review_decision": "keep",
    }]), encoding="utf-8")
    build_demo_state(
        source, filtered, reviewed_catalog=catalog, min_observations=3,
        resolved_only=True, max_camera_distance_m=0.0,
    )
    payload = torch.load(filtered, map_location="cpu", weights_only=False)
    payload["state"]["object_display_status"] = ["duplicate_suppressed"]
    torch.save(payload, filtered)

    with pytest.raises(ValueError, match="no active objects"):
        build_presentation_catalog(filtered, catalog, output)




def _ply(path: Path) -> None:
    vertices = np.asarray(
        [(1.0, 2.0, 3.0, 10, 20, 30), (-2.0, 1.0, 4.0, 40, 50, 60)],
        dtype=[
            ("x", "f4"), ("y", "f4"), ("z", "f4"),
            ("red", "u1"), ("green", "u1"), ("blue", "u1"),
        ],
    )
    PlyData([PlyElement.describe(vertices, "vertex")], text=False).write(path)


def test_cloud_export_scales_only_ply_coordinates_to_metres(tmp_path):
    source = tmp_path / "scene.ply"
    output = tmp_path / "cloud.npz"
    _ply(source)
    summary = export_cloud(
        source, output, max_points=0, min_opacity=0.0, seed=3,
        meters_per_scene_unit=0.1,
    )
    with np.load(output) as data:
        assert np.allclose(data["xyz"], [[0.1, 0.2, 0.3], [-0.2, 0.1, 0.4]])
    assert summary["coordinate_unit"] == "metre"
    assert summary["meters_per_scene_unit"] == pytest.approx(0.1)
    assert len(summary["fingerprint_sha256"]) == 64


def test_invalid_scale_fails_before_output_mutation(tmp_path, monkeypatch):
    output = tmp_path / "must-not-exist"
    monkeypatch.setattr(sys, "argv", [
        "prepare_farm_presentation.py",
        "--scene-state", str(tmp_path / "missing.pt"),
        "--scene-ply", str(tmp_path / "missing.ply"),
        "--output-dir", str(output),
        "--meters-per-scene-unit", "0",
    ])
    with pytest.raises(ValueError, match="finite and strictly positive"):
        presentation_main()
    assert not output.exists()


def test_legacy_cloud_reuse_rejects_metric_scale_mismatch(tmp_path):
    source = tmp_path / "scene.ply"
    _ply(source)
    cloud = tmp_path / "cloud.npz"
    np.savez_compressed(cloud, xyz=np.zeros((1, 3), dtype=np.float32))
    manifest = tmp_path / "manifest.json"
    manifest.write_text(json.dumps({"cloud": {
        "source_ply": str(source.resolve()), "meters_per_scene_unit": 1.0,
    }}), encoding="utf-8")
    requested = cloud_export_contract(
        source, max_points=10, min_opacity=0.0, seed=3,
        meters_per_scene_unit=0.1,
    )
    with pytest.raises(ValueError, match="metric scale"):
        validate_reusable_cloud(manifest, cloud, requested)
