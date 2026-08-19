import io
import sys
from pathlib import Path

import numpy as np
import pytest
import torch
from PIL import Image


ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
SCRIPTS = ROOT / "scripts"
for path in (SRC, SCRIPTS):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

from scene_graph.visualization.viser_visualizer import (  # noqa: E402
    PipelineViserVisualizer,
    _finite_consensus_points,
    _geometry_layer_render_mask,
    _normalise_world_up,
    _normalise_compound_boxes,
    _scene_relative_home_camera,
)
from view_scene_state import (  # noqa: E402
    attach_object_crops,
    choose_initial_object,
    has_geometry_layers,
    resolve_world_up,
    standard_run_crop_roots,
)


def _child_box(*, dimensions_key="dimensions_m"):
    return {
        "center_m": [1.0, 2.0, 3.0],
        dimensions_key: [2.0, 0.5, 0.25],
        "wxyz": [2.0, 0.0, 0.0, 0.0],
        "consensus_cells": 42,
    }


class _FakeHandle:
    def __init__(self, **kwargs):
        self.__dict__.update(kwargs)
        self.removed = False
        self.click_callback = None

    def remove(self):
        self.removed = True

    def on_click(self, callback):
        self.click_callback = callback
        return callback


class _FakeScene:
    def __init__(self):
        self.boxes = []
        self.labels = []
        self.point_clouds = []

    def add_box(self, name, **kwargs):
        handle = _FakeHandle(name=name, **kwargs)
        self.boxes.append(handle)
        return handle

    def add_label(self, name, **kwargs):
        handle = _FakeHandle(name=name, **kwargs)
        self.labels.append(handle)
        return handle

    def add_point_cloud(self, name, **kwargs):
        handle = _FakeHandle(name=name, **kwargs)
        self.point_clouds.append(handle)
        return handle


class _FakeServer:
    def __init__(self, clients=None):
        self.scene = _FakeScene()
        self._clients = dict(clients or {})

    def get_clients(self):
        return self._clients


class _FakeCamera:
    def __init__(self):
        self.position = np.zeros((3,), dtype=np.float32)
        self.look_at = np.zeros((3,), dtype=np.float32)
        self.up_direction = np.asarray([0.0, -1.0, 0.0], dtype=np.float32)
        self.fov = 0.0


class _FakeClient:
    def __init__(self):
        self.camera = _FakeCamera()


class _LookAtResetsUpCamera:
    """Model the viser client behaviour that previously exposed camera roll."""

    def __init__(self):
        self.position = np.zeros((3,), dtype=np.float32)
        self._look_at = np.zeros((3,), dtype=np.float32)
        self.up_direction = np.asarray([0.0, 0.0, 1.0], dtype=np.float32)
        self.fov = 0.0

    @property
    def look_at(self):
        return self._look_at

    @look_at.setter
    def look_at(self, value):
        self._look_at = np.asarray(value, dtype=np.float32)
        self.up_direction = np.asarray([0.0, 0.0, 1.0], dtype=np.float32)


def test_compound_box_schema_accepts_both_dimension_names_and_rejects_invalid_rows():
    rows = [
        _child_box(dimensions_key="dimensions_m"),
        _child_box(dimensions_key="dimensions_lwh_m"),
        {**_child_box(), "center_m": [np.nan, 0.0, 0.0]},
        {**_child_box(), "dimensions_m": [1.0, 0.0, 1.0]},
    ]

    boxes = _normalise_compound_boxes(rows)

    assert len(boxes) == 2
    assert np.allclose(boxes[0]["dimensions_m"], [2.0, 0.5, 0.25])
    assert np.allclose(boxes[1]["dimensions_m"], [2.0, 0.5, 0.25])
    assert np.isclose(np.linalg.norm(boxes[0]["wxyz"]), 1.0)


def test_consensus_points_drop_non_finite_rows():
    points = torch.tensor(
        [[1.0, 2.0, 3.0], [float("nan"), 0.0, 0.0], [0.0, float("inf"), 0.0]]
    )

    result = _finite_consensus_points(points)

    assert result.dtype == np.float32
    assert result.shape == (1, 3)
    assert np.allclose(result[0], [1.0, 2.0, 3.0])


def test_v9_presentation_suppresses_duplicates_without_reducing_audit_layers():
    active = np.asarray([True, True, True, False, False, False, True, False, True])
    semantic_tiers = [
        "confirmed", "probable", "geometry_only", "confirmed",
        "confirmed", "confirmed", "confirmed", "confirmed", "confirmed",
    ]
    statuses = [
        "geometry_pass",
        "geometry_pass",
        "assembly_geometry_pass",
        "assembly_compound_geometry_probable",
        "assembly_compound_geometry_rejected",
        "not_evaluated",
        "geometry_pass",
        "assembly_compound_geometry_probable",
        "geometry_pass",
    ]
    compounds = [[], [], [], [_child_box()], [], [], [], [_child_box()], []]
    consensus = [
        None, None, None, np.ones((4, 3)), np.ones((3, 3)), None, None,
        np.ones((4, 3)), None,
    ]
    display_statuses = [
        "", "", "", "", "", "", "duplicate_suppressed", "duplicate_suppressed",
        "part_suppressed",
    ]

    presentation = _geometry_layer_render_mask(
        active,
        semantic_tiers,
        statuses,
        compounds,
        consensus,
        "Presentation",
        display_statuses=display_statuses,
    )

    strict = _geometry_layer_render_mask(
        active, semantic_tiers, statuses, compounds, consensus, "Strict metric"
    )
    probable = _geometry_layer_render_mask(
        active, semantic_tiers, statuses, compounds, consensus, "Metric + compound probable"
    )
    diagnostics = _geometry_layer_render_mask(
        active, semantic_tiers, statuses, compounds, consensus, "Diagnostics"
    )

    assert presentation.tolist() == [True, True, False, True, False, False, False, False, False]
    assert strict.tolist() == [True, True, True, False, False, False, True, False, True]
    assert probable.tolist() == [True, True, True, True, False, False, True, True, True]
    assert diagnostics.tolist() == [True, True, True, True, True, False, True, True, True]
    assert int(strict.sum()) == int(active.sum())


def test_geometry_layer_detection_is_extension_driven():
    assert has_geometry_layers({"object_geometry_status": ["not_evaluated"]}) is False
    assert has_geometry_layers(
        {"object_geometry_status": ["assembly_compound_geometry_probable"]}
    ) is True
    assert has_geometry_layers({"object_compound_boxes": [[_child_box()]]}) is True


def test_standard_viewer_loads_direct_and_assembly_crops(tmp_path: Path):
    run_root = tmp_path / "run"
    state_path = run_root / "final" / "scene_state.pt"
    direct_crop = run_root / "mapping/masks/object_000001/direct.npz"
    assembly_crop = run_root / "qa/assemblies/masks/object_000002/assembly.npz"

    for path, color in (
        (direct_crop, (220, 30, 40)),
        (assembly_crop, (20, 180, 70)),
    ):
        path.parent.mkdir(parents=True, exist_ok=True)
        encoded = io.BytesIO()
        Image.new("RGB", (8, 6), color).save(encoded, format="JPEG")
        np.savez_compressed(
            path,
            crop_jpeg_bytes=np.frombuffer(encoded.getvalue(), dtype=np.uint8),
        )

    state = {
        "means": torch.zeros((2, 3), dtype=torch.float32),
        "active": torch.tensor([True, False]),
        "object_geometry_status": [
            "geometry_pass",
            "assembly_compound_geometry_probable",
        ],
        "object_mask_observations": [
            [{"path": "object_000001/direct.npz", "image_id": 10, "score": 0.9}],
            [{"path": "object_000002/assembly.npz", "image_id": 20, "score": 0.8}],
        ],
    }

    loaded = attach_object_crops(
        state,
        run_root / "mapping",
        3,
        additional_roots=standard_run_crop_roots(state_path),
    )

    assert loaded == 2
    assert [len(gallery) for gallery in state["rgb_observations"]] == [1, 1]
    assert state["rgb_observations"][0][0].shape == (6, 8, 3)
    assert state["rgb_observations"][1][0].shape == (6, 8, 3)


def test_standard_viewer_finds_assembly_crops_with_nested_final_data(tmp_path: Path):
    run_root = tmp_path / "run"
    state_path = run_root / "final" / "data" / "scene_state.pt"
    assembly_root = run_root / "qa" / "assemblies"

    assert standard_run_crop_roots(state_path) == (run_root, assembly_root)


def test_presentation_card_describes_children_not_rejected_parent_box():
    visualizer = PipelineViserVisualizer.__new__(PipelineViserVisualizer)
    visualizer._presentation_mode = True

    card = visualizer._format_caption_json_markdown(
        description="yellow industrial crane",
        category="crane",
        supercategory="equipment",
        attributes=["yellow", "metal"],
        decision="keep",
        geometry_status="assembly_compound_geometry_probable",
        box_center=[0.0, 0.0, 0.0],
        box_dimensions=[9.0, 9.0, 9.0],
        box_wxyz=[1.0, 0.0, 0.0, 0.0],
        compound_boxes=[_child_box(), {**_child_box(), "center_m": [3.0, 2.0, 3.0]}],
    )

    assert "Metric compound OBB" in card
    assert "2 parts" in card
    assert "part 1" in card and "part 2" in card
    assert "Metric assembly OBB" not in card
    assert "Rotation WXYZ" not in card


def test_presentation_card_describes_single_compound_as_unified_obb():
    visualizer = PipelineViserVisualizer.__new__(PipelineViserVisualizer)
    visualizer._presentation_mode = True

    card = visualizer._format_caption_json_markdown(
        description="yellow industrial crane",
        category="crane",
        supercategory="equipment",
        attributes=["yellow", "metal"],
        decision="keep",
        geometry_status="assembly_compound_geometry_probable",
        box_center=[0.0, 0.0, 0.0],
        box_dimensions=[9.0, 9.0, 9.0],
        box_wxyz=[1.0, 0.0, 0.0, 0.0],
        compound_boxes=[_child_box()],
    )

    assert "Metric unified OBB L×W×H" in card
    assert "1 parts" not in card
    assert "audited unified multi-view fit" in card


def test_compound_renderer_removes_parent_and_keeps_two_clickable_children():
    visualizer = PipelineViserVisualizer.__new__(PipelineViserVisualizer)
    visualizer._server = _FakeServer()
    parent = _FakeHandle()
    visualizer._object_cube_handles = {405: parent}
    visualizer._object_compound_cube_handles = {}
    visualizer._object_compound_label_handles = {}
    visualizer._focus_object_ids = None
    visualizer._hide_unclear_object_boxes = False
    visualizer._object_dims_hidden_by_size = lambda _dims: False
    visualizer._box_hidden_by_distance = lambda _center, _dims: False
    visualizer._box_hidden_by_view_depth = lambda _center, _dims: False
    visualizer._handle_object_click = lambda _object_id: None
    children = [_child_box(), {**_child_box(), "center_m": [3.0, 2.0, 3.0]}]

    visualizer._update_compound_cubes(
        ids=np.asarray([405]),
        colors=np.asarray([[255, 190, 74]], dtype=np.uint8),
        boxes_by_idx=[_normalise_compound_boxes(children)],
        labels=["crane"],
        visible_by_idx=[True],
        is_clear_by_idx=[True],
    )

    assert parent.removed is True
    assert len(visualizer._object_compound_cube_handles) == 2
    assert len(visualizer._server.scene.boxes) == 2
    assert all(handle.click_callback is not None for handle in visualizer._server.scene.boxes)
    assert visualizer._object_cube_handles[405] is visualizer._server.scene.boxes[0]
    assert visualizer._server.scene.labels[0].text == "#405  crane"


def test_consensus_points_render_without_falling_back_to_coarse_voxels():
    visualizer = PipelineViserVisualizer.__new__(PipelineViserVisualizer)
    visualizer._server = _FakeServer()
    visualizer._object_voxel_cloud_enabled = True
    visualizer._object_voxel_cloud_handle = None
    visualizer._object_voxel_cloud_dim_handle = None
    visualizer._object_voxel_max_points_per_object = 0
    visualizer._object_voxel_point_size = 0.025
    visualizer._focus_object_ids = None
    visualizer._presentation_mode = False
    visualizer._hide_unclear_object_boxes = False
    visualizer._object_dims_hidden_by_size = lambda _dims: False
    visualizer._view_depth_keep_mask = lambda points: np.ones((points.shape[0],), dtype=bool)
    visualizer._point_distance_keep_mask = lambda points: np.ones((points.shape[0],), dtype=bool)
    consensus = torch.tensor([[0.0, 0.0, 0.0], [0.1, 0.2, 0.3]], dtype=torch.float32)

    visualizer._update_object_voxel_cloud(
        {"object_geometry_consensus_points": [consensus]},
        active_indices_all=np.asarray([0]),
        ids=np.asarray([405]),
        colors=np.asarray([[255, 190, 74]], dtype=np.uint8),
        centers=np.zeros((1, 3), dtype=np.float32),
        dimensions=np.ones((1, 3), dtype=np.float32),
        orientations=np.asarray([[1.0, 0.0, 0.0, 0.0]], dtype=np.float32),
        is_clear_by_idx=[True],
        visible_by_idx=[True],
        compound_boxes_by_idx=[[]],
    )

    assert len(visualizer._server.scene.point_clouds) == 1
    assert np.allclose(visualizer._server.scene.point_clouds[0].points, consensus.numpy())


def _elongated_scene_points():
    x = np.linspace(-3.0, 3.0, 9)
    y = np.linspace(-1.0, 1.0, 5)
    z = np.linspace(-12.0, 12.0, 31)
    return np.stack(np.meshgrid(x, y, z, indexing="ij"), axis=-1).reshape(-1, 3)


def test_presentation_home_camera_is_low_oblique_and_scene_relative():
    points = _elongated_scene_points()
    home = _scene_relative_home_camera(points, presentation_mode=True)
    assert home is not None
    position, look_at = home
    offset = position - look_at

    # The scene is longest along Z, so viewing predominantly across X keeps
    # that long dimension horizontal rather than producing a floor-plan view.
    assert abs(float(offset[0])) > 3.0 * abs(float(offset[2]))
    horizontal_distance = float(np.linalg.norm(offset[[0, 2]]))
    elevation_ratio = abs(float(offset[1])) / horizontal_distance
    assert 0.12 < elevation_ratio < 0.24

    translation = np.asarray([100.0, -7.0, 23.0], dtype=np.float32)
    translated = _scene_relative_home_camera(points + translation, presentation_mode=True)
    assert translated is not None
    translated_position, translated_look_at = translated
    assert np.allclose(translated_position - position, translation, atol=1.0e-4)
    assert np.allclose(translated_look_at - look_at, translation, atol=1.0e-4)


def test_reset_overview_reapplies_the_exact_initial_home_camera():
    client = _FakeClient()
    visualizer = PipelineViserVisualizer.__new__(PipelineViserVisualizer)
    visualizer._server = _FakeServer({"test": client})
    visualizer._presentation_mode = True
    visualizer._home_camera = None
    visualizer._latest_scene_state = None
    visualizer._focus_object_ids = None
    visualizer._query_roles = None
    visualizer._query_results_display = None
    for attr in (
        "_search_highlight_box_handle",
        "_search_highlight_edges_handle",
        "_search_relations_handle",
        "_search_path_handle",
    ):
        setattr(visualizer, attr, None)

    visualizer.set_home_view(_elongated_scene_points())
    initial_position = client.camera.position.copy()
    initial_look_at = client.camera.look_at.copy()
    client.camera.position[:] = 0.0
    client.camera.look_at[:] = 0.0

    visualizer._reset_view()

    assert np.allclose(client.camera.position, initial_position)
    assert np.allclose(client.camera.look_at, initial_look_at)


def test_camera_world_up_is_applied_after_look_at_to_prevent_roll():
    visualizer = PipelineViserVisualizer.__new__(PipelineViserVisualizer)
    visualizer._presentation_mode = True
    client = _FakeClient()
    client.camera = _LookAtResetsUpCamera()

    visualizer._try_set_camera(
        client,
        position=np.asarray([4.0, -2.0, 1.0], dtype=np.float32),
        look_at=np.asarray([0.0, 0.0, 0.0], dtype=np.float32),
    )

    assert np.allclose(client.camera.up_direction, [0.0, -1.0, 0.0])


def test_exact_non_axis_world_up_drives_home_camera_and_is_translation_invariant():
    up = _normalise_world_up([1.0, 2.0, 3.0])
    seed = np.asarray([1.0, 0.0, 0.0], dtype=np.float32)
    first = seed - float(np.dot(seed, up)) * up
    first /= np.linalg.norm(first)
    second = np.cross(up, first)
    points = np.asarray([
        long * second + wide * first + height * up
        for long in np.linspace(-12.0, 12.0, 31)
        for wide in np.linspace(-3.0, 3.0, 9)
        for height in np.linspace(-1.0, 1.0, 5)
    ])
    home = _scene_relative_home_camera(points, presentation_mode=True, world_up=up)
    assert home is not None
    position, look_at = home
    offset = position - look_at
    elevation = float(np.dot(offset, up))
    horizontal = offset - elevation * up
    assert elevation > 0.0
    assert 0.12 < elevation / float(np.linalg.norm(horizontal)) < 0.24

    translation = np.asarray([23.0, -7.0, 101.0], dtype=np.float32)
    translated = _scene_relative_home_camera(
        points + translation, presentation_mode=True, world_up=up
    )
    assert translated is not None
    assert np.allclose(translated[0] - position, translation, atol=1.0e-4)
    assert np.allclose(translated[1] - look_at, translation, atol=1.0e-4)


def test_custom_world_up_is_set_after_camera_look_at_and_used_by_focus():
    up = _normalise_world_up([1.0, 2.0, 3.0])
    client = _FakeClient()
    client.camera = _LookAtResetsUpCamera()
    client.camera.position = np.asarray([4.0, 5.0, 6.0], dtype=np.float32)
    visualizer = PipelineViserVisualizer.__new__(PipelineViserVisualizer)
    visualizer._presentation_mode = True
    visualizer._world_up_explicit = True
    visualizer._world_up = up
    visualizer._server = _FakeServer({"test": client})
    visualizer._selected_object_id = 7
    visualizer._latest_ids = np.asarray([7])
    visualizer._latest_box_centers = np.asarray([[0.5, -0.5, 1.0]], dtype=np.float32)
    visualizer._latest_box_dimensions = np.asarray([[1.0, 2.0, 0.5]], dtype=np.float32)
    visualizer._latest_compound_boxes = [[]]
    visualizer._apply_focus = lambda _ids: None

    visualizer._focus_selected_object()

    assert np.allclose(client.camera.up_direction, up)
    delta = client.camera.position - visualizer._latest_box_centers[0]
    elevation = float(np.dot(delta, up))
    horizontal = delta - elevation * up
    assert elevation > 0.0
    assert abs(float(np.dot(horizontal, up))) < 1.0e-5


def test_world_up_cli_priority_over_resolved_context(tmp_path):
    context = tmp_path / "resolved_context.json"
    context.write_text('{"resolved_up": [0, 0, 1]}', encoding="utf-8")
    assert np.allclose(resolve_world_up(None, context), [0.0, 0.0, 1.0])
    assert np.allclose(resolve_world_up([1.0, 0.0, 0.0], context), [1.0, 0.0, 0.0])
    with pytest.raises(ValueError, match="finite and non-zero"):
        resolve_world_up([0.0, 0.0, 0.0], None)


def test_initial_object_prefers_validated_compound_without_class_or_id_hint():
    state = {
        "active": torch.tensor([True, True, False]),
        "object_id": torch.tensor([12, 37, 901]),
        "object_evidence_count": torch.tensor([30.0, 18.0, 12.0]),
        "object_geometry_inside_rate": torch.tensor([1.0, 0.9, 0.8]),
        "object_geometry_projected_box_iou": torch.tensor([0.9, 0.8, 0.02]),
        "object_geometry_orientation_confidence": torch.tensor([0.9, 0.8, 0.95]),
        "object_box_dimensions_m": torch.ones((3, 3)),
        "object_geometry_status": [
            "refined",
            "refined",
            "assembly_compound_geometry_probable",
        ],
    }
    assert choose_initial_object(state) == 901


def test_initial_object_falls_back_to_best_active_geometry_evidence():
    state = {
        "active": torch.tensor([True, True, False]),
        "object_id": torch.tensor([12, 37, 901]),
        "object_evidence_count": torch.tensor([3.0, 18.0, 100.0]),
        "object_geometry_inside_rate": torch.tensor([0.5, 1.0, 1.0]),
        "object_geometry_projected_box_iou": torch.tensor([0.5, 0.9, 1.0]),
        "object_geometry_orientation_confidence": torch.tensor([0.5, 0.9, 1.0]),
        "object_box_dimensions_m": torch.ones((3, 3)),
        "object_geometry_status": ["refined", "refined", "not_evaluated"],
    }
    assert choose_initial_object(state) == 37
