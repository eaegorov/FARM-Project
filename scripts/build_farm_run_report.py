#!/usr/bin/env python3
"""Build the final source-backed FARM run report and visual summary video."""

from __future__ import annotations

import argparse
import json
import os
from collections import Counter
from pathlib import Path
from typing import Any, Mapping, Sequence

import cv2
import numpy as np


def load_json(path: Path, *, required: bool = True) -> Any:
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except FileNotFoundError:
        if required:
            raise
        return None


def atomic_write_text(path: Path, value: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    try:
        with temporary.open("w", encoding="utf-8") as handle:
            handle.write(value)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)


def check_metrics(report: Mapping[str, Any], name: str) -> Mapping[str, Any]:
    for row in report.get("checks", []):
        if isinstance(row, Mapping) and row.get("name") == name:
            metrics = row.get("metrics")
            return metrics if isinstance(metrics, Mapping) else {}
    return {}


def metric_value(quality: Mapping[str, Any], name: str, default: Any = None) -> Any:
    row = (quality.get("metrics") or {}).get(name)
    return row.get("value", default) if isinstance(row, Mapping) else default


def human_duration(value: Any) -> str:
    if value is None:
        return "н/д"
    seconds = float(value)
    if seconds < 60:
        return f"{seconds:.2f} с"
    minutes, remainder = divmod(seconds, 60)
    if minutes < 60:
        return f"{int(minutes)} мин {remainder:.1f} с"
    hours, minutes = divmod(minutes, 60)
    return f"{int(hours)} ч {int(minutes)} мин {remainder:.0f} с"


def human_measure(value: Any) -> str:
    if value is None:
        return "н/д"
    if isinstance(value, bool):
        return str(value).lower()
    if isinstance(value, (int, np.integer)):
        return f"{int(value):,}".replace(",", " ")
    if isinstance(value, (float, np.floating)):
        number = float(value)
        if 0.0 <= number <= 1.0:
            return f"{number:.1%}"
        return f"{number:.3g}"
    return str(value)


def warning_table(retention: Mapping[str, Any]) -> str:
    rows = []
    for warning in retention.get("warnings") or []:
        if not isinstance(warning, Mapping):
            continue
        rows.append(
            f"| `{warning.get('code', 'warning')}` | "
            f"{str(warning.get('detail') or 'н/д').replace('|', '\\|')} | "
            f"{human_measure(warning.get('value'))} | {human_measure(warning.get('threshold'))} |"
        )
    if not rows:
        return "Предупреждений качества нет."
    return (
        "| Предупреждение | Что измерено | Наблюдение | Порог |\n"
        "|---|---|---:|---:|\n" + "\n".join(rows)
    )


def describe_hidden_breakdown(
    catalog: Sequence[Mapping[str, Any]],
    presentation: Sequence[Mapping[str, Any]],
) -> str:
    """Describe hidden objects as disjoint sets and disclose suppression overlap."""
    by_id = {
        int(row["id"]): row
        for row in catalog
        if isinstance(row, Mapping) and row.get("id") is not None
    }
    presentation_ids = {
        int(row["id"])
        for row in presentation
        if isinstance(row, Mapping) and row.get("id") is not None
    }
    hidden_ids = set(by_id) - presentation_ids
    geometry_only_ids = {
        object_id for object_id, row in by_id.items()
        if str(row.get("semantic_tier") or "").strip().lower() == "geometry_only"
    }
    suppressed_ids = {
        object_id for object_id, row in by_id.items()
        if str(row.get("display_status") or "").strip().lower().endswith("_suppressed")
    }
    geometry_only_hidden = hidden_ids & geometry_only_ids
    resolved_suppressed = (hidden_ids & suppressed_ids) - geometry_only_ids
    other_hidden = hidden_ids - geometry_only_hidden - resolved_suppressed
    overlap = hidden_ids & geometry_only_ids & suppressed_ids
    parts = [
        f"{len(geometry_only_hidden)} geometry-only",
        f"{len(resolved_suppressed)} resolved-suppressed",
    ]
    if other_hidden:
        parts.append(f"{len(other_hidden)} other-hidden")
    return (
        f"{len(hidden_ids)} = {' + '.join(parts)} "
        f"({len(suppressed_ids)} suppression records total; "
        f"{len(overlap)} overlap geometry-only)"
    )


def _slide_title(path: Path) -> str:
    return path.stem.replace("_4k", "").replace("_", " ").upper()


def compose_slide(image: np.ndarray, title: str, index: int, total: int) -> np.ndarray:
    canvas = np.full((1080, 1920, 3), (12, 18, 27), dtype=np.uint8)
    height, width = image.shape[:2]
    scale = min(1840 / max(width, 1), 950 / max(height, 1))
    resized = cv2.resize(
        image,
        (max(1, round(width * scale)), max(1, round(height * scale))),
        interpolation=cv2.INTER_AREA if scale < 1 else cv2.INTER_LINEAR,
    )
    y0 = 92 + (950 - resized.shape[0]) // 2
    x0 = (1920 - resized.shape[1]) // 2
    canvas[y0:y0 + resized.shape[0], x0:x0 + resized.shape[1]] = resized
    cv2.putText(canvas, title, (42, 57), cv2.FONT_HERSHEY_SIMPLEX, 1.12,
                (238, 243, 249), 2, cv2.LINE_AA)
    label = f"{index:02d} / {total:02d}"
    cv2.putText(canvas, label, (1735, 57), cv2.FONT_HERSHEY_SIMPLEX, 0.78,
                (104, 207, 255), 2, cv2.LINE_AA)
    progress = max(1, round(1840 * index / max(total, 1)))
    cv2.rectangle(canvas, (40, 1060), (1880, 1068), (38, 51, 68), -1)
    cv2.rectangle(canvas, (40, 1060), (40 + progress, 1068), (54, 190, 229), -1)
    return canvas


def rendered_video_timing(
    slide_count: int, fps: int, slide_seconds: float,
) -> dict[str, int | float]:
    """Return the exact frame count and duration emitted by ``build_video``."""
    if slide_count < 1 or fps < 1 or not np.isfinite(slide_seconds) or slide_seconds <= 0:
        raise ValueError("slide_count, fps and slide_seconds must be positive")
    frames_per_slide = max(1, round(fps * slide_seconds))
    fade_frames = min(max(1, round(fps * 0.25)), frames_per_slide // 3)
    frame_count = slide_count * frames_per_slide + max(0, slide_count - 1) * fade_frames
    return {
        "frames_per_slide": frames_per_slide,
        "fade_frames": fade_frames,
        "frame_count": frame_count,
        "duration_seconds": frame_count / fps,
    }


def build_video(run_dir: Path, output: Path, *, fps: int, slide_seconds: float) -> dict[str, Any]:
    index = load_json(run_dir / "visuals/index.json")
    image_paths = [
        run_dir / "visuals" / row["path"]
        for row in index.get("images", [])
        if isinstance(row, Mapping) and isinstance(row.get("path"), str)
    ]
    if not image_paths:
        raise RuntimeError("visuals/index.json contains no images")
    if any(not path.is_file() or path.stat().st_size == 0 for path in image_paths):
        raise RuntimeError("visual summary input is missing or empty")
    output.parent.mkdir(parents=True, exist_ok=True)
    temporary = output.with_name(f".{output.stem}.{os.getpid()}.tmp{output.suffix}")
    writer = cv2.VideoWriter(
        str(temporary), cv2.VideoWriter_fourcc(*"mp4v"), float(fps), (1920, 1080)
    )
    if not writer.isOpened():
        raise RuntimeError("OpenCV could not open an MP4 writer")
    timing = rendered_video_timing(len(image_paths), fps, slide_seconds)
    frames_per_slide = int(timing["frames_per_slide"])
    fade_frames = int(timing["fade_frames"])
    previous: np.ndarray | None = None
    try:
        for index_value, path in enumerate(image_paths, 1):
            image = cv2.imread(str(path), cv2.IMREAD_COLOR)
            if image is None or image.size == 0:
                raise RuntimeError(f"cannot decode visual: {path}")
            slide = compose_slide(image, _slide_title(path), index_value, len(image_paths))
            if previous is not None:
                for fade in range(1, fade_frames + 1):
                    alpha = fade / (fade_frames + 1)
                    writer.write(cv2.addWeighted(previous, 1.0 - alpha, slide, alpha, 0.0))
            for _ in range(frames_per_slide):
                writer.write(slide)
            previous = slide
    finally:
        writer.release()
    if not temporary.is_file() or temporary.stat().st_size < 10_000:
        temporary.unlink(missing_ok=True)
        raise RuntimeError("generated MP4 is missing or unexpectedly small")
    os.replace(temporary, output)
    return {
        "path": output.relative_to(run_dir).as_posix(),
        "bytes": output.stat().st_size,
        "width": 1920,
        "height": 1080,
        "fps": fps,
        "slides": len(image_paths),
        "frame_count": int(timing["frame_count"]),
        "duration_seconds": round(float(timing["duration_seconds"]), 6),
        "codec": "mp4v",
    }


def build_report(run_dir: Path, video: Mapping[str, Any] | None) -> str:
    manifest = load_json(run_dir / "manifest.json")
    preflight = load_json(run_dir / "input/scene_preflight.json")
    selection = load_json(run_dir / "selection/selection_manifest.json")
    prep = load_json(run_dir / "rgbd/prep_summary.json")
    quality = load_json(run_dir / "qa/final_analysis/quality_summary.json")
    retention = load_json(run_dir / "qa/retention_funnel.json", required=False)
    if not isinstance(retention, Mapping):
        retention = {}
    acceptance = load_json(run_dir / "qa/acceptance/result.json", required=False)
    if not isinstance(acceptance, Mapping):
        acceptance = {}
    structural = load_json(run_dir / "qa/geometry_qa/structural.json")
    resources = load_json(run_dir / "qa/resource_summary.json")
    timing = load_json(run_dir / "timing/summary.json", required=False)
    if not isinstance(timing, Mapping):
        timing = load_json(run_dir / "timing/qa_snapshot.json")
    catalog = load_json(run_dir / "final/catalog.json")
    presentation = load_json(run_dir / "final/presentation_catalog.json")
    if not isinstance(catalog, list) or not isinstance(presentation, list):
        raise RuntimeError("final catalogs must be JSON arrays")

    source_config: Mapping[str, Any] = {}
    resolved_plan = load_json(run_dir / "config/resolved-plan.json", required=False)
    run_source_path = run_dir / "config/source.redacted.yaml"
    if run_source_path.is_file():
        try:
            import yaml

            value = yaml.safe_load(run_source_path.read_text(encoding="utf-8"))
            if isinstance(value, Mapping):
                source_config = value
        except (ImportError, OSError, ValueError):
            source_config = {}
    conversion: Mapping[str, Any] = {}
    pinhole_root: Path | None = None
    if isinstance(source_config, Mapping):
        scene_source = source_config.get("inputs") or source_config.get("source")
        if isinstance(scene_source, Mapping):
            colmap_path = scene_source.get("colmap_model") or scene_source.get("colmap")
            if isinstance(colmap_path, str):
                candidate = Path(colmap_path).expanduser()
                for parent in (candidate, *candidate.parents):
                    manifest_path = parent / "conversion_manifest.json"
                    if manifest_path.is_file():
                        value = load_json(manifest_path)
                        if isinstance(value, Mapping):
                            conversion = value
                            pinhole_root = parent
                        break

    lineage_dir = run_dir / "reports/semantic-lineage-repair"
    catalog_before = load_json(lineage_dir / "catalog.before.json", required=False)
    unresolved_before = None
    if isinstance(catalog_before, list):
        unresolved_before = sum(
            1 for row in catalog_before
            if isinstance(row, Mapping)
            and str(row.get("category") or "").strip().lower() in {"", "unresolved object"}
        )
    unresolved_final = sum(
        1 for row in catalog
        if str(row.get("category") or "").strip().lower() in {"", "unresolved object"}
    )
    unresolved_presentation = sum(
        1 for row in presentation
        if str(row.get("category") or "").strip().lower() in {"", "unresolved object"}
    )

    scene_id = str(manifest.get("scene_id") or quality.get("scene_id") or "scene")
    qa_status = str(quality.get("qa_status", "unknown")).upper()
    selection_stats = selection.get("selection") or {}
    thresholds = selection.get("thresholds") or {}
    colmap = check_metrics(preflight, "colmap")
    ply = check_metrics(preflight, "gaussian_ply")
    metric = check_metrics(preflight, "metric_scale")
    gravity = check_metrics(preflight, "gravity")
    alignment = check_metrics(preflight, "colmap_3dgs_alignment")
    image_stats = check_metrics(preflight, "images")
    scene = quality.get("scene") or {}
    stages = timing.get("stages") or []
    category_counts = Counter(str(row.get("category") or "unresolved object") for row in presentation)
    tier_counts = Counter(str(row.get("semantic_tier") or "unavailable") for row in catalog)
    top_categories = ", ".join(
        f"{name} ({count})" for name, count in category_counts.most_common(12)
    ) or "нет"
    checks = quality.get("checks") or {}
    failed_checks = [name for name, passed in checks.items() if passed is not True]
    retention_warnings = [
        str(row.get("code"))
        for row in (retention.get("warnings") or [])
        if isinstance(row, Mapping) and row.get("code")
    ]
    retention_details = retention.get("details") or {}
    acceptance_counts = acceptance.get("counts") or {}
    release_status = str(
        (load_json(run_dir / "qa/summary.json", required=False) or {}).get(
            "release_status",
            acceptance.get("status", retention.get("quality_status", "UNKNOWN")),
        )
    ).upper()
    funnel = retention.get("funnel") or {}
    hidden_breakdown = describe_hidden_breakdown(catalog, presentation)
    funnel_text = " → ".join(
        f"{name.replace('_', ' ')} {human_measure(value)}"
        for name, value in funnel.items()
    ) or "н/д"
    warning_markdown = warning_table(retention)
    total_wall = timing.get("total_wall_seconds", timing.get("total_seconds"))
    stage_rows = []
    for row in stages:
        stage_id = str(row.get("stage_id") or row.get("stage") or "unknown")
        telemetry = row.get("telemetry_summary") or {}
        selected_uuid = str(resources.get("selected_gpu_uuid") or "")
        peak_mb = (telemetry.get("gpu_device_peak_used_by_uuid_mb") or {}).get(selected_uuid)
        delta_mb = (telemetry.get("gpu_device_peak_delta_by_uuid_mb") or {}).get(selected_uuid)
        if peak_mb is None or delta_mb is None:
            resource_row = next(
                (item for item in resources.get("stages", []) if item.get("stage") == stage_id),
                {},
            )
            peak_mb = resource_row.get("peak_used_mb")
            delta_mb = resource_row.get("peak_delta_mb")
        stage_rows.append(
            f"| `{stage_id}` | {row.get('status', 'unknown')} | "
            f"{human_duration(row.get('duration_seconds'))} | "
            f"{float(peak_mb) / 1024:.2f} | {float(delta_mb) / 1024:.2f} |"
            if peak_mb is not None and delta_mb is not None else
            f"| `{stage_id}` | {row.get('status', 'unknown')} | "
            f"{human_duration(row.get('duration_seconds'))} | н/д | н/д |"
        )

    report = f"""# Отчёт обработки сцены {scene_id}

## Техническое резюме

- Итог выпуска: **{release_status}**; structural QA **{retention.get('structural_status', qa_status)}**; качество результата **{retention.get('quality_status', 'н/д')}**; geometry structure **{structural.get('status', 'UNKNOWN')}**.
- Полный метрический каталог: **{len(catalog)}**; default presentation: **{len(presentation)}**; скрыто: **{hidden_breakdown}**.
- Final acceptance: **{acceptance.get('status', 'н/д')}**; автоматически скрыто **{acceptance_counts.get('held_objects', 'н/д')}**; осталось auto/blocking overlap **{acceptance_counts.get('remaining_auto_clusters', 'н/д')}**/**{acceptance_counts.get('remaining_blocking_clusters', 'н/д')}**; label hard errors **{acceptance_counts.get('label_hard_errors', 'н/д')}**.
- Из {selection_stats.get('candidate_observations', 'н/д')} физических timestamp’ов выбрано **{selection_stats.get('selected_observations', 'н/д')}** ({selection_stats.get('selected_views', 'н/д')} pinhole-view); residual flagged edges: **{selection_stats.get('residual_flagged_edges', 'н/д')}**.
- Без итогового label: **{unresolved_final}** в metric catalog / **{unresolved_presentation}** в presentation. Geometry-only объекты сохраняются для аудита, но не показываются по умолчанию.
- Полное wall time: **{human_duration(total_wall)}**. Пиковая занятость выбранного GPU: **{human_measure(resources.get('device_peak_used_gib'))} GiB**; максимальный прирост от baseline этапа: **{human_measure(resources.get('incremental_peak_delta_gib'))} GiB**.
- VRAM измеряется как `{resources.get('measurement_scope', 'unavailable')}`: это счётчик всего выбранного GPU, а не память одного PID.

### Что требует внимания

{warning_markdown}

Evidence funnel: {funnel_text}.

## Основные результаты

![Отбор кадров](visuals/01_selection_dashboard_4k.jpg)

Отбор строится в метрической системе по траектории, вращению и COLMAP-связности. Пороговые значения сохранены в `selection/selection_manifest.json`; итог не содержит необработанных разрывов графа.

![RGB-D и alignment](visuals/02_rgbd_alignment_dashboard_4k.jpg)

RGB-D подготовлен из 3DGS expected-depth и проверен против исходного RGB/COLMAP. Alignment gate: **{prep.get('alignment_qa', {}).get('passed', False)}**; metric scale gate: **{prep.get('metric_scale_check', {}).get('passed', False)}**.

![Финальная геометрия](visuals/04_obb_reprojections_4k.jpg)

OBB строятся в метрах с exact world-up; финальный structural gate не допускает NaN, неположительные размеры и некорректные quaternion.

![Финальный QA](visuals/06_final_quality_dashboard_4k.jpg)

Надёжных объектов (не менее трёх наблюдений): **{metric_value(quality, 'reliable_objects', 'н/д')}**. Полнота captions и Qwen-VL embeddings для активного набора: **{scene.get('captioned_active', 'н/д')}/{scene.get('objects_active_raw', 'н/д')}** и **{scene.get('qwen_vl_embedded_active', 'н/д')}/{scene.get('objects_active_raw', 'н/д')}**.

![Retention и качество результата](visuals/07_retention_quality_dashboard_4k.jpg)

Этот dashboard отделяет структурную корректность от фактического качества: показывает путь detections → tracks → geometry → semantics → presentation, диагностические surface flags, orphan masks и оставшиеся сильные duplicate-кандидаты. `WARN` не маскируется под полный recall и не блокирует сохранение структурно корректного результата.

![Final acceptance](visuals/08_final_acceptance_dashboard_4k.jpg)

Финальный model-free gate выполняет один ограниченный по времени проход: автоматически скрывает только высокоуверенные presentation-дубликаты, блокирует неоднозначные overlap-кластеры и проверяет structured label provenance без изменения metric geometry, active state или mask evidence.

![Overlap review](visuals/final/04_duplicate_overlap_review_4k.jpg)

Отдельный locator разделяет presentation-visible duplicate-like пары (**{retention_details.get('residual_duplicate_like_visible_count', 0)}**) и визуально похожие, но классифицированные distinct пары (**{retention_details.get('residual_similar_distinct_visible_count', 0)}**). Строки со скрытым endpoint (**{retention_details.get('residual_hidden_endpoint_count', 0)}**) не выдаются за оставшиеся видимые дубликаты.

![Label uncertainty sample](visuals/final/06_label_uncertainty_sample_4k.jpg)

Детерминированная uncertainty-first выборка показывает реальные crop evidence, semantic tier, число независимых групп/видов и причины неопределённости. Полный машинный аудит: `qa/acceptance/label_uncertainty.json`.

### Интерактивный и видео-результат

- Viewer: `viewer/launch.sh`
- Scene state: `final/scene_state.pt`
- Каталоги: `final/catalog.json`, `final/presentation_catalog.json`
- Visual summary: `{video.get('path') if video else 'не создан'}`{f" ({video.get('duration_seconds')} с, {video.get('width')}×{video.get('height')})" if video else ''}

## Scope и входные данные

| Источник | Значение |
|---|---:|
| Registered pinhole images | {colmap.get('registered_images', 'н/д')} |
| Проверенные image files | {image_stats.get('checked', 'н/д')} |
| COLMAP sparse points | {colmap.get('sparse_points', 'н/д')} |
| COLMAP observations | {colmap.get('observations', 'н/д')} |
| 3DGS Gaussians | {ply.get('vertex_count', 'н/д')} |
| RGB-D views | {prep.get('view_count', 'н/д')} |
| Physical cameras/sensors | {len(selection.get('grouping', {}).get('selected_sensors', [])) or prep.get('camera_count', 'н/д')} |
| Metres per scene unit | {metric.get('inferred_meters_per_colmap_unit', metric.get('meters_per_colmap_unit', 'н/д'))} |
| Resolved world-up | `{gravity.get('resolved_up', 'н/д')}` |

COLMAP↔3DGS gate: robust bbox IoU **{alignment.get('robust_bbox_iou', 'н/д')}**, centroid-offset ratio **{alignment.get('centroid_offset_ratio', 'н/д')}**, camera-inside ratio **{alignment.get('camera_centers_inside_expanded_gaussian_bounds', 'н/д')}**.

### Подготовка fisheye KB4

| Параметр | Значение |
|---|---:|
| Source camera model | {', '.join((conversion.get('input_validation') or {}).get('camera_models', [])) or 'н/д'} |
| Source registered images | {(conversion.get('input_validation') or {}).get('registered_images', 'н/д')} |
| Полные физические timestamp-группы | {(conversion.get('rig') or {}).get('complete_groups', 'н/д')} |
| Virtual pinhole images | {conversion.get('output_image_count', 'н/д')} |
| Virtual views / lens | {len((conversion.get('config') or {}).get('views', [])) or 'н/д'} |
| Pinhole resolution | {(conversion.get('config') or {}).get('width', 'н/д')}×{(conversion.get('config') or {}).get('height', 'н/д')} |
| Конвертация | {human_duration(conversion.get('runtime_seconds'))} |
| Geometry reprojection max | {(conversion.get('geometry_validation') or {}).get('reprojection_max_px', 'н/д')} px |
| Naming collisions | {conversion.get('naming_collisions', 'н/д')} |

Конвертация — отдельный подготовительный шаг и не включена в wall time 17 стадий FARM. Manifest: `{(pinhole_root / 'conversion_manifest.json') if pinhole_root else 'не найден'}`.

## Методика отбора кадров

1. Файлы объединяются в физические timestamp’ы по configurable regex; virtual views не считаются независимыми позами.
2. Камерные центры переводятся в метры по preflight scale.
3. Начальный набор равномерно покрывает hybrid arc: translation + `{thresholds.get('rotation_weight_m_per_rad', 'н/д')}` м/рад × rotation.
4. Bridge-pass добавляет timestamp’ы, пока выполняются motion/translation/rotation и COLMAP shared-track/overlap gates.
5. Пайплайн fail-closed, если после бюджета остаются flagged edges.

Канонический селектор: `FARM-Project/scripts/select_colmap_keyframes.py`. Стабильная точка запуска для других сцен: `data/utils/select_colmap_keyframes.py`. Для каждого запуска обязательно строится та же шестипанельная QA-картина: trajectory, hybrid arc, COLMAP support, pair motion, connectivity и sampling intervals.

## Время и VRAM по этапам

| Этап | Статус | Время | Device peak, GiB | Incremental peak, GiB |
|---|---:|---:|---:|---:|
{chr(10).join(stage_rows)}

## Семантика и каталог

- Semantic tiers: {', '.join(f'{name}: {count}' for name, count in sorted(tier_counts.items())) or 'нет'}.
- Наиболее частые presentation labels: {top_categories}.
- Suppressed duplicate/part/assembly-member tracks: **{metric_value(quality, 'duplicate_losers_resolved', 'н/д')}** documented losers.
- Unresolved labels: **{unresolved_final}** metric / **{unresolved_presentation}** presentation.
- Acceptance overlap: **{acceptance_counts.get('held_objects', 0)}** presentation objects held; **{retention_details.get('residual_duplicate_like_visible_count', 0)}** open duplicate/blocking; **{retention_details.get('residual_similar_distinct_visible_count', 0)}** similar-but-distinct; **{retention_details.get('residual_hidden_endpoint_count', 0)}** hidden context.
{f'- До исправления generic semantic lineage unresolved labels было **{unresolved_before}**; после передачи cross-pass consensus в финализацию — **{unresolved_final}**. Повторялись только дешёвые finalize/QA стадии, model inference не подменялся.' if unresolved_before is not None else ''}

## Ограничения и robustness

- Детектор использует статический open-vocabulary YOLOE profile: маленькие, закрытые и out-of-vocabulary объекты могут быть пропущены. Результат evidence-backed, но не является гарантированно полным inventory.
- Названия и описания получены моделями без ручных object prompts; ошибки semantic class возможны и должны оцениваться по crop gallery в viewer.
- 3DGS/COLMAP alignment — robust geometric gate; он не заменяет ручную проверку нескольких RGB/depth/OBB overlays.
- Не прошедшие checks: {', '.join(failed_checks) if failed_checks else 'нет'}.
- Result-quality warnings: {', '.join(retention_warnings) if retention_warnings else 'нет'}.

## Следующие шаги

1. Просмотреть `visuals/scene_summary.mp4`, затем OBB reprojections и saved-mask contact sheet.
2. В viewer проверить наиболее крупные/составные объекты и соответствие crop gallery.
3. Для production зафиксировать clean commit/image digest и хранить этот отчёт вместе с `config/source.redacted.yaml` и stage logs.

## Открытые вопросы

- Требуется ли scene-specific target recall по мелким объектам, который оправдывает увеличение бюджета кадров/разрешения?
- Нужна ли downstream-валидация labels по собственному производственному словарю без изменения open-vocabulary inference?
"""
    return report


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument("--report", type=Path)
    parser.add_argument("--video", type=Path)
    parser.add_argument("--fps", type=int, default=12)
    parser.add_argument("--slide-seconds", type=float, default=1.8)
    parser.add_argument("--no-video", action="store_true")
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    run_dir = args.run_dir.expanduser().resolve(strict=True)
    if args.fps < 1 or not np.isfinite(args.slide_seconds) or args.slide_seconds <= 0:
        raise ValueError("fps and slide-seconds must be positive")
    report_path = (args.report or run_dir / "REPORT.md").resolve(strict=False)
    video_path = (args.video or run_dir / "visuals/scene_summary.mp4").resolve(strict=False)
    video: Mapping[str, Any] | None = None
    if not args.no_video:
        video = build_video(run_dir, video_path, fps=args.fps, slide_seconds=args.slide_seconds)
        report_meta = run_dir / "reports/run_report.json"
        report_meta.parent.mkdir(parents=True, exist_ok=True)
        atomic_write_text(report_meta, json.dumps({
            "schema": "farm.run-report.v1", "status": "PASS",
            "release_status": (
                (load_json(run_dir / "qa/summary.json", required=False) or {}).get(
                    "release_status", "UNKNOWN"
                )
            ),
            "video": video,
            "report": report_path.relative_to(run_dir).as_posix(),
        }, indent=2, sort_keys=True) + "\n")
    else:
        report_meta = run_dir / "reports/run_report.json"
        previous = load_json(report_meta, required=False)
        if isinstance(previous, Mapping) and isinstance(previous.get("video"), Mapping):
            video = previous["video"]
    atomic_write_text(report_path, build_report(run_dir, video))
    if report_path.stat().st_size < 1_000:
        raise RuntimeError("generated REPORT.md is unexpectedly small")
    print(json.dumps({
        "status": "PASS", "report": str(report_path),
        "video": str(video_path) if video else None,
    }, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
