#!/usr/bin/env python3
"""Serve a read-only FARM scene cloud with metric MV-SAM3D Gaussian objects."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import sys
import threading
import time
import warnings
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np


PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from farm_runtime.unified_viewer import (  # noqa: E402
    REQUIRED_VISER_VERSION,
    UnifiedViewerError,
    _instance_color,
    camera_presets,
    focus_pose,
    load_gaussian_splat_arrays,
    open_binary_ply,
)


DEFAULT_CLOUD = Path(
    "/home/splatica/workspace/3dgs_work/output/farm_pipeline/knaack/runs/"
    "knaack-production-v8-quality/final/cloud.npz"
)
DEFAULT_RECON = Path(
    "/home/splatica/workspace/splatica_demo_app/outputs/farm_recon"
)


@dataclass(frozen=True)
class OverlayData:
    manifest: Mapping[str, Any]
    object_rows: Mapping[int, Mapping[str, Any]]
    centers: np.ndarray
    covariances: np.ndarray
    natural_rgbs: np.ndarray
    instance_rgbs: np.ndarray
    opacities: np.ndarray
    sampled_object_ids: np.ndarray
    bounds_by_id: Mapping[int, np.ndarray]
    total_rows: int


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while block := handle.read(8 * 1024 * 1024):
            digest.update(block)
    return digest.hexdigest()


def _load_json(path: Path) -> Mapping[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, Mapping):
        raise UnifiedViewerError(f"JSON root must be an object: {path.name}")
    return payload


def stable_stratified_indices(
    labels: np.ndarray,
    max_rows: int,
    *,
    minimum_per_object: int = 4_096,
) -> np.ndarray:
    """Choose deterministic row-order samples while retaining every object ID."""

    values = np.asarray(labels)
    if values.ndim != 1 or values.dtype.kind not in "iu":
        raise UnifiedViewerError("farm_object_id must be one integer vector")
    if int(max_rows) < 1 or int(minimum_per_object) < 1:
        raise UnifiedViewerError("sampling limits must be positive")
    object_ids, counts = np.unique(values.astype(np.int64, copy=False), return_counts=True)
    if len(object_ids) == 0 or np.any(object_ids < 0):
        raise UnifiedViewerError("reconstruction PLY has invalid or missing object IDs")
    if int(max_rows) < len(object_ids):
        raise UnifiedViewerError("max splats must retain at least one row per object")
    target = min(int(max_rows), len(values))
    minimum = min(int(minimum_per_object), max(1, target // len(object_ids)))
    quotas = np.minimum(counts, minimum).astype(np.int64)
    remaining = target - int(quotas.sum())
    capacity = counts - quotas
    if remaining > 0 and int(capacity.sum()) > 0:
        exact = remaining * capacity.astype(np.float64) / float(capacity.sum())
        extra = np.minimum(capacity, np.floor(exact).astype(np.int64))
        quotas += extra
        remaining -= int(extra.sum())
        if remaining:
            residual = exact - extra
            order = np.lexsort((object_ids, -residual))
            for index in order:
                if remaining == 0:
                    break
                if quotas[index] < counts[index]:
                    quotas[index] += 1
                    remaining -= 1
    selected: list[np.ndarray] = []
    for object_id, quota in zip(object_ids, quotas, strict=True):
        candidates = np.flatnonzero(values == object_id)
        take = int(quota)
        if take == len(candidates):
            chosen = candidates
        else:
            chosen = candidates[np.linspace(0, len(candidates) - 1, take, dtype=np.int64)]
        selected.append(chosen.astype(np.int64, copy=False))
    result = np.sort(np.concatenate(selected))
    if len(result) != target or len(np.unique(result)) != target:
        raise UnifiedViewerError("stratified sampler produced an invalid selection")
    if set(np.unique(values[result]).tolist()) != set(object_ids.tolist()):
        raise UnifiedViewerError("stratified sampler dropped an object")
    return np.ascontiguousarray(result)


def _object_bounds(table: Any, object_ids: Sequence[int]) -> dict[int, np.ndarray]:
    source = table.memmap()
    wanted = {int(value) for value in object_ids}
    minimum = {value: np.full(3, np.inf, dtype=np.float64) for value in wanted}
    maximum = {value: np.full(3, -np.inf, dtype=np.float64) for value in wanted}
    for start in range(0, table.count, 524_288):
        rows = source[start : min(table.count, start + 524_288)]
        labels = np.asarray(rows["farm_object_id"], dtype=np.int64)
        points = np.column_stack((rows["x"], rows["y"], rows["z"])).astype(
            np.float64, copy=False
        )
        for object_id in np.unique(labels):
            value = int(object_id)
            if value not in wanted:
                raise UnifiedViewerError(f"PLY contains undeclared object ID {value}")
            local = points[labels == value]
            if not np.isfinite(local).all():
                raise UnifiedViewerError(f"object {value} contains non-finite centres")
            minimum[value] = np.minimum(minimum[value], local.min(axis=0))
            maximum[value] = np.maximum(maximum[value], local.max(axis=0))
    bounds: dict[int, np.ndarray] = {}
    for object_id in sorted(wanted):
        box = np.stack((minimum[object_id], maximum[object_id])).astype(np.float32)
        if not np.isfinite(box).all() or np.any(box[1] <= box[0]):
            raise UnifiedViewerError(f"object {object_id} has invalid metric bounds")
        bounds[object_id] = box
    return bounds


def load_overlay(recon_dir: Path, *, max_splats: int) -> OverlayData:
    root = recon_dir.expanduser().resolve(strict=True)
    manifest_path = root / "manifest.json"
    marker = _load_json(root / "_SUCCESS.json")
    manifest = _load_json(manifest_path)
    if marker.get("status") != "PASS" or marker.get("schema") != "splatica.farm-mv-sam3d.v1":
        raise UnifiedViewerError("FARM reconstruction completion marker is not PASS")
    if marker.get("manifest_sha256") != _sha256(manifest_path):
        raise UnifiedViewerError("FARM reconstruction manifest hash mismatch")
    if manifest.get("schema") != "splatica.farm-mv-sam3d.v1" or manifest.get("scene_id") != "knaack":
        raise UnifiedViewerError("unexpected FARM reconstruction manifest contract")
    assembly = manifest.get("assembly")
    if not isinstance(assembly, Mapping) or assembly.get("status") not in {"PASS", "WARN"}:
        raise UnifiedViewerError("FARM reconstruction assembly is not terminal PASS/WARN")
    rows = assembly.get("objects")
    if not isinstance(rows, list) or len(rows) != 21:
        raise UnifiedViewerError("FARM reconstruction must contain exactly 21 requested objects")
    assembly_rows = {int(row["id"]): row for row in rows if isinstance(row, Mapping)}
    input_rows = {
        int(row["id"]): row
        for row in (manifest.get("objects") or [])
        if isinstance(row, Mapping)
    }
    if len(assembly_rows) != len(rows) or set(assembly_rows) != set(input_rows):
        raise UnifiedViewerError("FARM reconstruction object IDs are not unique")
    object_rows = {
        object_id: {**input_rows[object_id], **assembly_rows[object_id]}
        for object_id in sorted(assembly_rows)
    }
    orientation_report = assembly.get("orientation_refinement_report")
    if not isinstance(orientation_report, Mapping):
        raise UnifiedViewerError("gravity/silhouette orientation report is missing")
    orientation_path = (root / str(orientation_report.get("path"))).resolve(strict=True)
    if orientation_path.parent != root or _sha256(orientation_path) != orientation_report.get("sha256"):
        raise UnifiedViewerError("gravity/silhouette orientation report hash mismatch")
    for object_id, row in object_rows.items():
        transform = row.get("transform")
        orientation = transform.get("orientation_refinement") if isinstance(transform, Mapping) else None
        if not isinstance(orientation, Mapping) or orientation.get("schema") != "splatica.farm-recon-orientation.v1":
            raise UnifiedViewerError(f"object {object_id} has no validated orientation refinement")
    ply_record = assembly.get("combined_world_metric_ply")
    if not isinstance(ply_record, Mapping) or ply_record.get("status") != "PASS":
        raise UnifiedViewerError("world-metric Gaussian PLY is not declared PASS")
    ply_path = (root / str(ply_record.get("path"))).resolve(strict=True)
    if ply_path.parent != root:
        raise UnifiedViewerError("world-metric Gaussian PLY escapes reconstruction root")
    if int(ply_record.get("bytes", -1)) != ply_path.stat().st_size:
        raise UnifiedViewerError("world-metric Gaussian PLY size mismatch")
    if str(ply_record.get("sha256")) != _sha256(ply_path):
        raise UnifiedViewerError("world-metric Gaussian PLY hash mismatch")
    table = open_binary_ply(ply_path)
    if "farm_object_id" not in (table.dtype.names or ()):
        raise UnifiedViewerError("world-metric Gaussian PLY lacks farm_object_id")
    if int(ply_record.get("rows", -1)) != table.count:
        raise UnifiedViewerError("world-metric Gaussian PLY row-count mismatch")
    source = table.memmap()
    labels = np.asarray(source["farm_object_id"], dtype=np.int32)
    declared_ids = set(object_rows)
    if set(np.unique(labels).tolist()) != declared_ids:
        raise UnifiedViewerError("world-metric Gaussian PLY IDs differ from manifest objects")
    selected = stable_stratified_indices(labels, max_splats)
    selected_ids = np.ascontiguousarray(labels[selected])
    centers, covariances, natural_rgbs, opacities = load_gaussian_splat_arrays(
        table, meters_per_scene_unit=1.0, source_indices=selected
    )
    palette = np.asarray([_instance_color(int(value)) for value in selected_ids], dtype=np.uint8)
    return OverlayData(
        manifest=manifest,
        object_rows=object_rows,
        centers=centers,
        covariances=covariances,
        natural_rgbs=natural_rgbs,
        instance_rgbs=np.ascontiguousarray(palette.astype(np.float32) / 255.0),
        opacities=opacities,
        sampled_object_ids=selected_ids,
        bounds_by_id=_object_bounds(table, sorted(declared_ids)),
        total_rows=table.count,
    )


def load_scene_cloud(path: Path, *, max_points: int) -> tuple[np.ndarray, np.ndarray]:
    resolved = path.expanduser().resolve(strict=True)
    with np.load(resolved, allow_pickle=False) as archive:
        xyz = np.asarray(archive["xyz"], dtype=np.float32)
        rgb = np.asarray(archive["rgb"], dtype=np.uint8)
    if xyz.ndim != 2 or xyz.shape[1] != 3 or rgb.shape != xyz.shape:
        raise UnifiedViewerError("scene cloud must contain matching N x 3 xyz/rgb arrays")
    if not np.isfinite(xyz).all() or len(xyz) < 3:
        raise UnifiedViewerError("scene cloud contains invalid points")
    if len(xyz) > int(max_points):
        use = np.linspace(0, len(xyz) - 1, int(max_points), dtype=np.int64)
        xyz, rgb = xyz[use], rgb[use]
    # Preserve real RGB hue but keep the context subordinate to reconstructed objects.
    neutral = np.clip(rgb.astype(np.float32) * 0.48 + 18.0, 0.0, 255.0).astype(np.uint8)
    return np.ascontiguousarray(xyz), np.ascontiguousarray(neutral)


class ReconOverlayViewer:
    def __init__(
        self,
        *,
        cloud: tuple[np.ndarray, np.ndarray],
        overlay: OverlayData,
        host: str,
        port: int,
        point_size: float,
    ) -> None:
        try:
            import viser
        except Exception as exc:  # pragma: no cover - Docker runtime dependency
            raise UnifiedViewerError("viser is required to serve the reconstruction") from exc
        if getattr(viser, "__version__", None) != REQUIRED_VISER_VERSION:
            raise UnifiedViewerError(
                f"viewer requires viser=={REQUIRED_VISER_VERSION}; found {getattr(viser, '__version__', None)}"
            )
        if host not in {"127.0.0.1", "localhost", "::1"}:
            warnings.warn("viewer is unauthenticated and exposed beyond loopback", RuntimeWarning)
        self.overlay = overlay
        self.lock = threading.RLock()
        self.previous: dict[Any, tuple[np.ndarray, np.ndarray, np.ndarray, float]] = {}
        self.server = viser.ViserServer(host=host, port=int(port), label="Knaack · FARM recon")
        self.server.gui.configure_theme(
            control_layout="fixed", control_width="medium", dark_mode=True,
            show_logo=False, show_share_button=False, brand_color=(45, 212, 191),
        )
        background = np.zeros((2, 2, 3), dtype=np.uint8)
        background[...] = [7, 13, 23]
        self.server.scene.set_background_image(background, format="png")
        object_rows = overlay.object_rows
        first_row = next(iter(object_rows.values()))
        transform = first_row.get("transform")
        if not isinstance(transform, Mapping):
            raise UnifiedViewerError("object transform metadata is missing")
        self.world_up = np.asarray(transform.get("resolved_up"), dtype=np.float32).reshape(3)
        self.world_up /= np.linalg.norm(self.world_up)
        self.server.scene.set_up_direction(tuple(float(value) for value in self.world_up))
        points, point_colours = cloud
        self.presets = camera_presets(points, self.world_up)
        self.context = self.server.scene.add_point_cloud(
            "/knaack/context_points", points=points, colors=point_colours,
            point_size=float(point_size), point_shape="circle", point_shading="flat",
            precision="float16",
        )
        self.splats = self.server.scene.add_gaussian_splats(
            "/knaack/mv_sam3d_objects", centers=overlay.centers,
            covariances=overlay.covariances, rgbs=overlay.natural_rgbs,
            opacities=overlay.opacities,
        )
        self.selection_box: Any | None = None
        self.labels_to_ids = {
            f"{object_id:06d} — {str(row.get('category') or 'object').replace('_', ' ')}": object_id
            for object_id, row in sorted(object_rows.items())
        }
        with self.server.gui.add_folder("Knaack — реконструкция FARM", expand_by_default=True):
            self.server.gui.add_markdown(
                f"**ГОТОВО** — {len(points):,} опорных точек сцены; "
                f"показано {len(overlay.centers):,} splat-ов из {overlay.total_rows:,} строк "
                f"для всех {len(object_rows)} объектов MV-SAM3D."
            )
            self.show_context = self.server.gui.add_checkbox("Точки исходной сцены", initial_value=True)
            self.show_splats = self.server.gui.add_checkbox("Плотные splat-объекты MV-SAM3D", initial_value=True)
            self.instance_colours = self.server.gui.add_checkbox(
                "Цвет по instance ID", initial_value=False,
                hint=(
                    "Выкл.: восстановленный DC-цвет splat-а. Вкл.: детерминированный "
                    "категориальный RGB только из farm_object_id; это не confidence, "
                    "не quality score и не новая маска."
                ),
            )
            self.point_size = self.server.gui.add_slider(
                "Размер точек сцены (м)", min=0.001, max=0.012, step=0.001,
                initial_value=float(point_size),
            )
            first_label = next(iter(self.labels_to_ids))
            self.object_select = self.server.gui.add_dropdown(
                "Цель вращения камеры", options=tuple(self.labels_to_ids), initial_value=first_label
            )
            self.object_info = self.server.gui.add_markdown("")
        with self.server.gui.add_folder("Камера и вращение", expand_by_default=True):
            self.server.gui.add_markdown(
                "**ЛКМ:** вращение · **ПКМ:** сдвиг · **колесо/щипок:** масштаб. "
                "Выбор объекта переносит только центр вращения; фокус также кадрирует объект."
            )
            self.overview = self.server.gui.add_button("Обзор всей сцены")
            self.focus = self.server.gui.add_button("Фокус на выбранном объекте")
            self.front = self.server.gui.add_button("Спереди")
            self.side = self.server.gui.add_button("Сбоку")
            self.top = self.server.gui.add_button("Сверху")
            self.previous_button = self.server.gui.add_button("Предыдущая камера")
        @self.show_context.on_update
        def _context(_event: Any = None) -> None:
            self.context.visible = bool(self.show_context.value)

        @self.show_splats.on_update
        def _splats(_event: Any = None) -> None:
            self.splats.visible = bool(self.show_splats.value)

        @self.instance_colours.on_update
        def _colours(_event: Any = None) -> None:
            with self.lock:
                self.splats.rgbs = (
                    self.overlay.instance_rgbs
                    if bool(self.instance_colours.value)
                    else self.overlay.natural_rgbs
                )

        @self.point_size.on_update
        def _size(_event: Any = None) -> None:
            self.context.point_size = float(self.point_size.value)

        @self.object_select.on_update
        def _object(_event: Any = None) -> None:
            self._select(self.labels_to_ids[str(self.object_select.value)])

        @self.overview.on_click
        def _overview(_event: Any = None) -> None:
            self._apply_pose(self.presets["overview"])

        @self.front.on_click
        def _front(_event: Any = None) -> None:
            self._apply_pose(self.presets["front"])

        @self.side.on_click
        def _side(_event: Any = None) -> None:
            self._apply_pose(self.presets["side"])

        @self.top.on_click
        def _top(_event: Any = None) -> None:
            self._apply_pose(self.presets["top"])

        @self.focus.on_click
        def _focus(_event: Any = None) -> None:
            object_id = self.labels_to_ids[str(self.object_select.value)]
            for client in self.server.get_clients().values():
                pose = focus_pose(
                    self.overlay.bounds_by_id[object_id], client.camera.position,
                    client.camera.look_at, self.world_up,
                )
                self._set_camera(client, pose)

        @self.previous_button.on_click
        def _previous(_event: Any = None) -> None:
            with self.lock:
                for client in self.server.get_clients().values():
                    key = getattr(client, "client_id", id(client))
                    state = self.previous.get(key)
                    if state is None:
                        continue
                    current = self._state(client)
                    self._set_camera(client, state[:3], remember=False)
                    client.camera.fov = state[3]
                    self.previous[key] = current

        @self.server.on_client_connect
        def _connect(client: Any) -> None:
            def apply() -> None:
                time.sleep(0.2)
                self._set_camera(client, self.presets["overview"], remember=False)
            threading.Thread(target=apply, daemon=True).start()

        self._select(self.labels_to_ids[first_label], move_camera=False)

    @staticmethod
    def _state(client: Any) -> tuple[np.ndarray, np.ndarray, np.ndarray, float]:
        return (
            np.asarray(client.camera.position, dtype=np.float32).copy(),
            np.asarray(client.camera.look_at, dtype=np.float32).copy(),
            np.asarray(client.camera.up_direction, dtype=np.float32).copy(),
            float(client.camera.fov),
        )

    def _set_camera(
        self, client: Any, pose: tuple[np.ndarray, np.ndarray, np.ndarray], *, remember: bool = True
    ) -> None:
        with self.lock:
            key = getattr(client, "client_id", id(client))
            if remember:
                self.previous[key] = self._state(client)
            position, look_at, up = pose
            client.camera.position = np.asarray(position, dtype=np.float32)
            client.camera.look_at = np.asarray(look_at, dtype=np.float32)
            client.camera.up_direction = np.asarray(up, dtype=np.float32)
            client.camera.fov = math.radians(52.0)

    def _apply_pose(self, pose: tuple[np.ndarray, np.ndarray, np.ndarray]) -> None:
        for client in self.server.get_clients().values():
            self._set_camera(client, pose)

    def _select(self, object_id: int, *, move_camera: bool = True) -> None:
        with self.lock:
            row = self.overlay.object_rows[int(object_id)]
            box = self.overlay.bounds_by_id[int(object_id)]
            center = box.mean(axis=0)
            dimensions = box[1] - box[0]
            if self.selection_box is not None:
                self.selection_box.remove()
            self.selection_box = self.server.scene.add_box(
                "/knaack/selected_object_bounds", color=(45, 212, 191),
                dimensions=dimensions, position=center, wireframe=True, opacity=0.95,
                side="double", cast_shadow=False, receive_shadow=False,
            )
            status = str(row.get("status") or "UNKNOWN")
            views = row.get("views") if isinstance(row.get("views"), list) else []
            reverse = row.get("reverse_projection")
            summary = reverse.get("summary", {}) if isinstance(reverse, Mapping) else {}
            transform = row.get("transform")
            orientation = (
                transform.get("orientation_refinement", {})
                if isinstance(transform, Mapping)
                else {}
            )
            self.object_info.content = (
                f"**#{object_id} · {row.get('category', 'object')} · {status}**  \n"
                f"{len(views)} реальных PINHOLE-кадра · {int(row.get('vertices', 0)):,} вершин · "
                f"{int(row.get('faces', 0)):,} граней  \n"
                f"full-mask silhouette IoU `{float(summary.get('silhouette_iou', 0.0)):.3f}` · "
                f"bbox IoU `{float(summary.get('bbox_iou', 0.0)):.3f}` · "
                f"ошибка центра `{float(summary.get('center_error_px', 0.0)):.1f}px`  \n"
                f"наклон canonical-up к gravity: "
                f"`{float(orientation.get('angle_before_degrees', 0.0)):.1f}° → "
                f"{float(orientation.get('angle_after_degrees', 0.0)):.1f}°` · "
                f"принята доля коррекции `{float(orientation.get('accepted_fraction', 0.0)):.1f}`"
            )
            if move_camera:
                for client in self.server.get_clients().values():
                    self.previous[getattr(client, "client_id", id(client))] = self._state(client)
                    client.camera.look_at = center.copy()
                    client.camera.up_direction = self.world_up.copy()

    def run_forever(self) -> None:
        while True:
            time.sleep(1.0)


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cloud", type=Path, default=DEFAULT_CLOUD)
    parser.add_argument("--recon", type=Path, default=DEFAULT_RECON)
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8082)
    parser.add_argument("--max-context-points", type=int, default=1_500_000)
    parser.add_argument("--max-splats", type=int, default=1_800_000)
    parser.add_argument("--point-size", type=float, default=0.003)
    parser.add_argument("--validate", action="store_true")
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    try:
        cloud = load_scene_cloud(args.cloud, max_points=args.max_context_points)
        overlay = load_overlay(args.recon, max_splats=args.max_splats)
        ready = {
            "status": "READY", "scene": "knaack", "context_points": len(cloud[0]),
            "displayed_splats": len(overlay.centers), "source_splat_rows": overlay.total_rows,
            "objects": len(overlay.object_rows), "renderer": "viser-quantized-3dgs-dc",
        }
        print(json.dumps(ready, ensure_ascii=False, indent=2), flush=True)
        if args.validate:
            return 0
        viewer = ReconOverlayViewer(
            cloud=cloud, overlay=overlay, host=args.host, port=args.port,
            point_size=args.point_size,
        )
        viewer.run_forever()
        return 0
    except (UnifiedViewerError, FileNotFoundError, OSError, ValueError) as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
