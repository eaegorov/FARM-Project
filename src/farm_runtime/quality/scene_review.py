"""Readable review of saved, camera-aligned masks; no geometry or inference.

Supply original-camera RGB, binary projected memberships and projected OBB
segments on ONE pixel grid. Display roll is applied once, to every layer, with
an expanded canvas. Mask holes, distant components and empty results are kept.
An optional original-resolution photo may supply object-card photography; it
must have the same field of view and aspect ratio as the mask grid. No object
or scene names enter selection. GPU rendering belongs to the caller.
"""

from __future__ import annotations

import colorsys
import math
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np
from PIL import Image, ImageDraw, ImageFont, ImageOps


@dataclass(frozen=True)
class ReviewFrame:
    rgb: np.ndarray
    native_masks: Mapping[int, np.ndarray]
    catalog_objects: Mapping[int, Mapping[str, Any]] = field(default_factory=dict)
    input_masks: Mapping[int, np.ndarray] = field(default_factory=dict)
    reference_masks: Mapping[int, np.ndarray] = field(default_factory=dict)
    projected_obbs: Mapping[int, np.ndarray] = field(default_factory=dict)
    instance_ids: np.ndarray | None = None
    reference_instance_ids: np.ndarray | None = None
    rotation_degrees_ccw: float = 0.0
    source_label: str = ""
    object_photo_rgb: np.ndarray | None = None


@dataclass(frozen=True)
class ReviewOptions:
    panel_width: int = 900
    panel_max_height: int = 1000
    object_columns: int = 2
    object_panel_width: int = 360
    object_panel_height: int = 420
    max_obbs: int = 6
    min_obb_mask_pixels: int = 50
    overlay_alpha: float = 0.38
    crop_padding_fraction: float = 0.12
    font_path: str | None = None
    source_title: str = "Фотография"
    input_title: str = "Входная 2D-маска"
    reference_title: str = "3D-маска до изменения"
    native_title: str = "Текущая 3D-маска"


@dataclass(frozen=True)
class ReviewSheet:
    image: Image.Image
    metadata: dict[str, Any]


_BG = (246, 248, 250)
_INK = (30, 44, 56)
_MUTED = (88, 101, 113)
_MASK = (236, 155, 38)


def gravity_display_rotation(
    rotation_world_to_camera: np.ndarray,
    world_up: Sequence[float],
    intrinsics: np.ndarray | None = None,
    image_point: Sequence[float] | None = None,
) -> float | None:
    """CCW roll making projected world-up vertical; None at a gravity pole.

    With intrinsics, use the projection derivative at ``image_point`` (the
    principal point by default). This is display roll, not camera rectification.
    It must not be applied to poses or saved mask memberships.
    """
    r = np.asarray(rotation_world_to_camera, dtype=float)
    up = np.asarray(world_up, dtype=float)
    if (
        r.shape != (3, 3)
        or not np.isfinite(r).all()
        or not np.allclose(r @ r.T, np.eye(3), atol=1e-5)
        or not np.isclose(np.linalg.det(r), 1.0, atol=1e-5)
        or up.shape != (3,)
        or not np.isfinite(up).all()
        or np.linalg.norm(up) < 1e-12
    ):
        raise ValueError("proper camera rotation and finite nonzero world-up required")
    g = r @ (up / np.linalg.norm(up))
    projected = g[:2]
    if intrinsics is not None:
        k = np.asarray(intrinsics, dtype=float)
        if k.shape != (3, 3) or not np.isfinite(k).all() or min(k[0, 0], k[1, 1]) <= 0:
            raise ValueError("finite pinhole intrinsics required")
        uv = np.asarray(image_point if image_point is not None else k[:2, 2], float)
        if uv.shape != (2,) or not np.isfinite(uv).all():
            raise ValueError("finite image point required")
        projected = np.array([
            k[0, 0] * g[0] + k[0, 1] * g[1] - (uv[0] - k[0, 2]) * g[2],
            k[1, 1] * g[1] - (uv[1] - k[1, 2]) * g[2],
        ])
    elif image_point is not None:
        raise ValueError("image_point requires intrinsics")
    if np.linalg.norm(projected) < 1e-6:
        return None
    return float(np.degrees(np.arctan2(projected[0], -projected[1])))


def rotate_layer(array: np.ndarray, degrees_ccw: float, *, mask: bool = False) -> np.ndarray:
    """One shared expanded-canvas image transform; never interpolate masks."""
    a = np.asarray(array)
    if not math.isfinite(degrees_ccw):
        raise ValueError("finite display rotation required")
    if mask:
        if a.ndim != 2 or a.dtype != bool:
            raise ValueError("mask must be a two-dimensional bool array")
        image = Image.fromarray(a.astype(np.uint8) * 255)
    else:
        if a.ndim != 3 or a.shape[2] != 3 or a.dtype != np.uint8:
            raise ValueError("RGB must be an HxWx3 uint8 array")
        image = Image.fromarray(a)
    rotated = image.rotate(
        degrees_ccw,
        resample=Image.Resampling.NEAREST if mask else Image.Resampling.BICUBIC,
        expand=True,
        fillcolor=0 if mask else _BG,
    )
    result = np.asarray(rotated)
    return result > 0 if mask else result.copy()


def _validate(frame: ReviewFrame, options: ReviewOptions) -> None:
    rgb = np.asarray(frame.rgb)
    if rgb.ndim != 3 or rgb.shape[2] != 3 or rgb.dtype != np.uint8 or min(rgb.shape[:2]) == 0:
        raise ValueError("RGB must be a nonempty HxWx3 uint8 array")
    if frame.object_photo_rgb is not None:
        photo = np.asarray(frame.object_photo_rgb)
        if (
            photo.ndim != 3 or photo.shape[2] != 3 or photo.dtype != np.uint8
            or min(photo.shape[:2]) == 0
            or photo.shape[0] * rgb.shape[1] != photo.shape[1] * rgb.shape[0]
        ):
            raise ValueError("object photo must be uint8 RGB with the same camera field of view and aspect ratio")
    if not math.isfinite(frame.rotation_degrees_ccw):
        raise ValueError("finite display rotation required")
    for bank in (frame.native_masks, frame.input_masks, frame.reference_masks):
        for object_id, value in bank.items():
            mask = np.asarray(value)
            if not isinstance(object_id, (int, np.integer)) or object_id < 0:
                raise ValueError("nonnegative integer object IDs required")
            if mask.shape != rgb.shape[:2] or mask.dtype != bool:
                raise ValueError("all masks must be bool arrays on the RGB pixel grid")
    for owner, bank in ((frame.instance_ids, frame.native_masks), (frame.reference_instance_ids, frame.reference_masks)):
        if owner is None:
            continue
        ids = np.asarray(owner)
        if ids.shape != rgb.shape[:2] or not np.issubdtype(ids.dtype, np.integer) or (ids < -1).any():
            raise ValueError("instance_ids must be integer HxW with -1 for background")
        for object_id in np.unique(ids[ids >= 0]):
            if int(object_id) not in bank:
                raise ValueError("instance_ids refer to an absent native mask")
            if np.any((ids == object_id) & ~bank[int(object_id)]):
                raise ValueError("instance ownership extends beyond its native mask")
        union = np.zeros(rgb.shape[:2], bool)
        for mask in bank.values():
            union |= mask
        if np.any((ids >= 0) != union):
            raise ValueError("instance ownership must represent the union of every supplied mask")
    for segments in frame.projected_obbs.values():
        a = np.asarray(segments)
        if a.ndim != 3 or a.shape[1:] != (2, 2) or not np.isfinite(a).all():
            raise ValueError("projected OBBs require finite Ex2x2 image-space segments")
    if not 0 < options.overlay_alpha <= 1 or not 0 <= options.crop_padding_fraction <= 1:
        raise ValueError("invalid display alpha or crop padding")
    for key in ("panel_width", "panel_max_height", "object_columns", "object_panel_width", "object_panel_height"):
        value = getattr(options, key)
        if not isinstance(value, int) or value < 1:
            raise ValueError(f"{key} must be a positive integer")
    if options.max_obbs < 0 or options.min_obb_mask_pixels < 0:
        raise ValueError("OBB limits cannot be negative")


def _font(size: int, options: ReviewOptions, bold: bool = False) -> ImageFont.ImageFont:
    name = "DejaVuSans-Bold.ttf" if bold else "DejaVuSans.ttf"
    paths = [Path(options.font_path)] if options.font_path else [
        Path("/usr/share/fonts/truetype/dejavu") / name,
        Path("/usr/share/fonts/dejavu-sans-fonts") / name,
    ]
    for path in paths:
        if path.is_file():
            return ImageFont.truetype(str(path), size)
    raise FileNotFoundError("A Unicode TrueType font is required; set ReviewOptions.font_path")


def _wrapped(text: str, font: ImageFont.ImageFont, width: int) -> list[str]:
    lines = []
    for paragraph in str(text).split("\n"):
        line = ""
        for word in paragraph.split():
            trial = f"{line} {word}".strip()
            if line and font.getlength(trial) > width:
                lines.append(line)
                line = ""
            for char in ((" " if line else "") + word):
                if line and font.getlength(line + char) > width:
                    lines.append(line)
                    line = ""
                line += char
        lines.append(line)
    return lines


def _text_height(text: str, width: int, size: int, options: ReviewOptions, bold=False) -> int:
    return len(_wrapped(text, _font(size, options, bold), width)) * (size + 8)


def _text(draw: ImageDraw.ImageDraw, xy, text, width, size, options, *, bold=False, fill=_INK):
    font = _font(size, options, bold)
    for i, line in enumerate(_wrapped(text, font, width)):
        draw.text((xy[0], xy[1] + i * (size + 8)), line, font=font, fill=fill)


def _paste(canvas: Image.Image, image: Image.Image, box: tuple[int, int, int, int]):
    x, y, width, height = box
    contained = ImageOps.contain(image, (width, height), Image.Resampling.LANCZOS)
    canvas.paste(contained, (x + (width - contained.width) // 2, y + (height - contained.height) // 2))


def _original_photo_crop(photo, grid_hw, scale, box, panel_wh):
    """Sample the same expanded-grid crop directly from the original photo.

    Both rotations use image centres. Expanded canvas sizes can round
    differently, so matching their centres avoids a scaled rounding offset.
    No low-resolution intermediate is made for the photographic column.
    """
    x0, y0, x1, y1 = box
    grid_h, grid_w = grid_hw
    extent = (
        (x0 - grid_w / 2) * scale + photo.width / 2,
        (y0 - grid_h / 2) * scale + photo.height / 2,
        (x1 - grid_w / 2) * scale + photo.width / 2,
        (y1 - grid_h / 2) * scale + photo.height / 2,
    )
    factor = min(panel_wh[0] / (x1 - x0), panel_wh[1] / (y1 - y0))
    size = (max(1, round((x1 - x0) * factor)), max(1, round((y1 - y0) * factor)))
    tile = photo.transform(size, Image.Transform.EXTENT, extent,
                           Image.Resampling.BICUBIC, fillcolor=_BG)
    return tile, list(extent)


def _color(object_id: int) -> tuple[int, int, int]:
    return tuple(round(v * 255) for v in colorsys.hsv_to_rgb((object_id * 0.61803398875 + 0.08) % 1, 0.66, 0.91))


def _overlay(rgb: np.ndarray, mask: np.ndarray, alpha: float, color=_MASK) -> np.ndarray:
    result = rgb.copy()
    result[mask] = np.rint((1 - alpha) * rgb[mask] + alpha * np.asarray(color)).astype(np.uint8)
    return result


def _base_metadata(frame: ReviewFrame) -> dict[str, Any]:
    return {
        "source_label": frame.source_label,
        "source_hw": list(frame.rgb.shape[:2]),
        "rotation_degrees_ccw": float(frame.rotation_degrees_ccw),
        "display_only": True,
        "geometry_or_membership_modified": False,
        "mask_resampling": "nearest; expanded canvas; no morphology",
        "native_mask_ids": sorted(int(i) for i in frame.native_masks),
    }


def render_scene_sheet(
    frame: ReviewFrame,
    title: str,
    options: ReviewOptions | None = None,
) -> ReviewSheet:
    """Full camera RGB / all native projections, with a separate bounded OBB row.

    The OBB limit is a display budget only. Every mask remains visible in the
    full overlay; skipped boxes and masks with zero projected area are reported.
    ``instance_ids`` must already represent the caller's visibility arbitration.
    Without that evidence, this function displays a single-color mask union.
    """
    options = options or ReviewOptions()
    _validate(frame, options)
    rgb = np.asarray(frame.rgb)
    counts = {int(i): int(np.count_nonzero(m)) for i, m in frame.native_masks.items()}
    coverage = np.zeros(rgb.shape[:2], np.uint32)
    for mask in frame.native_masks.values():
        coverage += mask
    source_rgb = rotate_layer(rgb, frame.rotation_degrees_ccw)
    source = Image.fromarray(source_rgb)
    use_owners = frame.instance_ids is not None and (not frame.reference_masks or frame.reference_instance_ids is not None)

    def projected_image(bank, owners):
        if owners is None:
            union = np.zeros(rgb.shape[:2], bool)
            for mask in bank.values():
                union |= mask
            mask = rotate_layer(union, frame.rotation_degrees_ccw, mask=True)
            return Image.fromarray(_overlay(source_rgb, mask, options.overlay_alpha))
        result = source_rgb.copy()
        for object_id in sorted(bank):
            mask = rotate_layer(owners == object_id, frame.rotation_degrees_ccw, mask=True)
            result = _overlay(result, mask, options.overlay_alpha, _color(object_id))
        return Image.fromarray(result)

    projected = projected_image(frame.native_masks, frame.instance_ids if use_owners else None)
    caption = "Цвет различает маски, а не качество. Неокрашенные участки сохранены." if use_owners else "Оранжевым — все видимые 3D-маски; отверстия и фон сохранены."
    panels = [(options.source_title, source)]
    reference_counts = {}
    if frame.reference_masks:
        reference_counts = {int(i): int(np.count_nonzero(m)) for i, m in frame.reference_masks.items()}
        # Use identical color semantics when the old version has no owner map.
        if not use_owners:
            caption = "Оранжевым — 3D-маски в обеих версиях; одинаковые цвет, прозрачность и камера."
        panels.append((options.reference_title, projected_image(frame.reference_masks, frame.reference_instance_ids if use_owners else None)))
    panels.append((options.native_title, projected))
    candidates = sorted(
        (int(i) for i in frame.projected_obbs if counts.get(int(i), 0) >= max(1, options.min_obb_mask_pixels)),
        key=lambda i: (-counts[i], i),
    )
    selected = candidates[:options.max_obbs]
    omitted = sorted(set(frame.projected_obbs) - set(selected))
    boxes = Image.fromarray(rgb.copy())
    pen = ImageDraw.Draw(boxes)
    for object_id in selected:
        for segment in np.asarray(frame.projected_obbs[object_id]):
            pen.line([tuple(segment[0]), tuple(segment[1])], fill=_color(object_id), width=max(2, rgb.shape[1] // 500))
    box_image = Image.fromarray(rotate_layer(np.asarray(boxes), frame.rotation_degrees_ccw))
    pad, gap = 30, 24
    width = len(panels) * options.panel_width + gap * (len(panels) - 1) + 2 * pad
    image_height = min(options.panel_max_height, round(source.height * options.panel_width / source.width))
    image_height = max(1, image_height)
    sub = f"{frame.source_label}\n{caption} Показано масок в этом ракурсе: {sum(c > 0 for c in counts.values())} из {len(counts)}."
    head = 34 + _text_height(title, width - 2 * pad, 34, options, True)
    top = head + _text_height(sub, width - 2 * pad, 22, options) + 28
    panel_label_height = max(_text_height(label, options.panel_width, 25, options, True) for label, _ in panels) + 12
    row_height = image_height + panel_label_height + 15
    legend_width = width - (pad + options.panel_width + gap) - pad - 32
    box_labels = [(i, f"#{i}: {frame.catalog_objects.get(i, {}).get('label') or 'название не подтверждено'}") for i in selected]
    legend_height = sum(_text_height(label, legend_width, 25, options) + 12 for _, label in box_labels)
    legend_height += 24 + _text_height("Названия предложены моделями и могут быть ошибочными.", legend_width, 21, options)
    box_row_height = max(image_height, legend_height) + 30
    footer = "Поворот исправляет только наклон изображения; фотография, маски и рамки преобразованы одинаково."
    box_note = (
        f"Рамки по наблюдаемой геометрии: {len(selected)} из {len(frame.projected_obbs)}. "
        "Для читаемости выбраны наибольшие видимые проекции; все маски остаются в верхней панели. "
        "Линии рамок не учитывают перекрытие передними предметами; это не подтверждение физических размеров."
    )
    box_note_height = _text_height(box_note, width - 2 * pad, 21, options) + 18
    height = top + row_height + (_text_height(footer, width - 2 * pad, 20, options) + 35)
    if options.max_obbs and frame.projected_obbs:
        height += box_row_height + box_note_height
    canvas = Image.new("RGB", (width, height), _BG)
    draw = ImageDraw.Draw(canvas)
    _text(draw, (pad, 22), title, width - 2 * pad, 34, options, bold=True)
    _text(draw, (pad, head), sub, width - 2 * pad, 22, options, fill=_MUTED)
    for col, (label, panel) in enumerate(panels):
        x = pad + col * (options.panel_width + gap)
        _text(draw, (x, top), label, options.panel_width, 25, options, bold=True)
        _paste(canvas, panel, (x, top + panel_label_height, options.panel_width, image_height))
    y = top + row_height
    if options.max_obbs and frame.projected_obbs:
        _text(draw, (pad, y), box_note, width - 2 * pad, 21, options, fill=_MUTED)
        y += box_note_height
        _paste(canvas, box_image, (pad, y, options.panel_width, image_height))
        legend_x = pad + options.panel_width + gap
        legend_y = y + 8
        legend_width = width - legend_x - pad - 32
        for object_id, label in box_labels:
            draw.rectangle((legend_x, legend_y + 5, legend_x + 18, legend_y + 23), fill=_color(object_id))
            _text(draw, (legend_x + 30, legend_y), label, legend_width, 25, options)
            legend_y += _text_height(label, legend_width, 25, options) + 12
        _text(draw, (legend_x, legend_y + 8), "Названия предложены моделями и могут быть ошибочными.", legend_width, 21, options, fill=_MUTED)
        y += box_row_height
    _text(draw, (pad, y + 10), footer, width - 2 * pad, 20, options, fill=_MUTED)
    metadata = _base_metadata(frame)
    metadata.update(
        visible_mask_ids=sorted(i for i, count in counts.items() if count),
        zero_projection_ids=sorted(i for i, count in counts.items() if not count),
        projected_mask_pixels=counts,
        reference_projected_mask_pixels=reference_counts,
        overlapping_binary_mask_pixels=int(np.count_nonzero(coverage > 1)),
        displayed_union_pixels=int(np.count_nonzero(coverage)),
        instance_display="caller_supplied_owner_map" if frame.instance_ids is not None and (not frame.reference_masks or frame.reference_instance_ids is not None) else "single_color_union",
        obb_selected_ids=selected,
        obb_omitted_ids=omitted,
        obb_selection="descending visible mask pixels, object ID tie-break; no quality filter",
        pixel_counts_are_not_quality_metrics=True,
    )
    return ReviewSheet(canvas, metadata)


def render_object_sheet(
    frame: ReviewFrame,
    object_ids: Sequence[int],
    title: str,
    options: ReviewOptions | None = None,
) -> ReviewSheet:
    """Equal paired crops from the union of ALL supplied stages for each object.

    Passing every catalog ID gives complete pages when paginated by the caller.
    Missing stage data is labelled unavailable, never silently shown as empty.
    A supplied empty mask is shown unchanged and explicitly labelled empty.
    """
    options = options or ReviewOptions()
    _validate(frame, options)
    ids = [int(i) for i in object_ids]
    if not ids or len(ids) != len(set(ids)):
        raise ValueError("a nonempty list of distinct object IDs is required")
    known = set(frame.catalog_objects) | set(frame.native_masks) | set(frame.input_masks) | set(frame.reference_masks)
    if set(ids) - known:
        raise ValueError("requested object IDs are absent from the frame and catalog")
    stages = [(options.source_title, None, "photo")]
    if frame.input_masks:
        stages.append((options.input_title, frame.input_masks, "input"))
    if frame.reference_masks:
        stages.append((options.reference_title, frame.reference_masks, "reference"))
    stages.append((options.native_title, frame.native_masks, "native"))
    source = rotate_layer(frame.rgb, frame.rotation_degrees_ccw)
    original_photo = None if frame.object_photo_rgb is None else Image.fromarray(
        rotate_layer(frame.object_photo_rgb, frame.rotation_degrees_ccw)
    )
    photo_scale = None if original_photo is None else frame.object_photo_rgb.shape[1] / frame.rgb.shape[1]
    pad, gap = 24, 20
    card_width = len(stages) * options.object_panel_width + (len(stages) - 1) * 10
    columns = min(options.object_columns, len(ids))
    width = 2 * pad + columns * card_width + (columns - 1) * gap
    sub = f"{frame.source_label}\nОдинаковые границы кадрирования во всех колонках; показаны и пропуски, и лишние участки. Названия требуют проверки."
    head = 30 + _text_height(title, width - 2 * pad, 32, options, True)
    top = head + _text_height(sub, width - 2 * pad, 21, options) + 25
    records = []
    cards = []
    for object_id in ids:
        rotated = {}
        union = np.zeros(source.shape[:2], bool)
        areas = {}
        for _, bank, key in stages:
            if bank is not None and object_id in bank:
                mask = rotate_layer(bank[object_id], frame.rotation_degrees_ccw, mask=True)
                rotated[key] = mask
                union |= mask
                areas[key] = int(np.count_nonzero(bank[object_id]))
            elif bank is not None:
                areas[key] = None
        yy, xx = np.nonzero(union)
        if len(xx):
            margin = max(8, math.ceil(max(np.ptp(xx) + 1, np.ptp(yy) + 1) * options.crop_padding_fraction))
            box = (max(0, int(xx.min()) - margin), max(0, int(yy.min()) - margin),
                   min(source.shape[1], int(xx.max()) + margin + 1), min(source.shape[0], int(yy.max()) + margin + 1))
        else:
            box = (0, 0, source.shape[1], source.shape[0])
        x0, y0, x1, y1 = box
        crop = source[y0:y1, x0:x1]
        name = frame.catalog_objects.get(object_id, {}).get("label") or "название не подтверждено"
        label = f"#{object_id} · {name}"
        label_height = _text_height(label, card_width - 20, 23, options, True) + 16
        stage_label_height = max(_text_height(s[0], options.object_panel_width, 19, options) for s in stages) + 14
        card = Image.new("RGB", (card_width, label_height + stage_label_height + options.object_panel_height + 43), (255, 255, 255))
        draw = ImageDraw.Draw(card)
        _text(draw, (10, 6), label, card_width - 20, 23, options, bold=True)
        photo_extent = None
        for col, (caption, bank, key) in enumerate(stages):
            x = col * (options.object_panel_width + 10)
            _text(draw, (x, label_height), caption, options.object_panel_width, 19, options)
            image = crop if bank is None or key not in rotated else _overlay(crop, rotated[key][y0:y1, x0:x1], options.overlay_alpha)
            tile = Image.fromarray(image)
            if bank is None and original_photo is not None:
                tile, photo_extent = _original_photo_crop(
                    original_photo, source.shape[:2], photo_scale, box,
                    (options.object_panel_width, options.object_panel_height),
                )
            _paste(card, tile, (x, label_height + stage_label_height, options.object_panel_width, options.object_panel_height))
            note = "" if bank is None else "нет данных" if key not in rotated else "маска пустая" if areas[key] == 0 else ""
            if note:
                _text(draw, (x + 8, card.height - 33), note, options.object_panel_width - 16, 18, options, fill=_MUTED)
        records.append({"object_id": object_id, "upright_crop_xyxy": list(box), "source_mask_pixels": areas, "all_stages_empty_or_missing": not bool(len(xx))})
        if photo_extent is not None:
            records[-1]["original_photo_crop_xyxy"] = photo_extent
        cards.append(card)
    row_heights = [max(c.height for c in cards[start:start + columns]) for start in range(0, len(cards), columns)]
    canvas = Image.new("RGB", (width, top + sum(row_heights) + gap * len(row_heights) + 18), _BG)
    draw = ImageDraw.Draw(canvas)
    _text(draw, (pad, 16), title, width - 2 * pad, 32, options, bold=True)
    _text(draw, (pad, head), sub, width - 2 * pad, 21, options, fill=_MUTED)
    y = top
    for row, height in enumerate(row_heights):
        for col, card in enumerate(cards[row * columns:(row + 1) * columns]):
            canvas.paste(card, (pad + col * (card_width + gap), y))
        y += height + gap
    metadata = _base_metadata(frame)
    if original_photo is not None:
        metadata.update(
            object_photo_source_hw=list(frame.object_photo_rgb.shape[:2]),
            object_photo_sampling="Original RGB, same expanded-centre roll and crop extent; no downsample to mask grid before crop. Only photographic column.",
            object_mask_grid_hw=list(frame.rgb.shape[:2]),
        )
    metadata.update(
        displayed_object_ids=ids,
        requested_object_ids=ids,
        object_records=records,
        stages=[key for _, _, key in stages],
        crop_policy="union of every available input/reference/native mask after shared roll; no component or quality filtering",
        object_selection="explicit caller sequence; no sorting, sampling, dropping or relabelling",
    )
    return ReviewSheet(canvas, metadata)
