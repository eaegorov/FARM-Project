# Входные данные новой сцены

Production FARM не угадывает соответствие между COLMAP, RGB и 3DGS. Эти три
источника должны описывать одну реконструкцию и одну систему координат.

## Обязательные входы

1. Зарегистрированная sparse COLMAP-модель (`cameras`, `images`, `points3D`).
2. Все RGB-файлы по именам из COLMAP.
3. Binary little-endian vertex-only Graphdeco 3DGS PLY.
4. Обоснованный metric scale.
5. Обоснованный gravity/up.
6. Для rig/virtual views — grouping физического timestamp и camera family.

Минимальные поля PLY для lift:

```text
x y z
f_dc_0 f_dc_1 f_dc_2
opacity
scale_0 scale_1 scale_2
rot_0 rot_1 rot_2 rot_3
```

Higher-order SH сохраняются побайтно в полном labelled PLY.

## PINHOLE обязателен для production evidence

MV-SAM3D и camera/mask QA используют реальные зарегистрированные PINHOLE
кадры. Допускается crop, но запрещены non-uniform stretch и blur. RGB
ресемплируется с сохранением aspect ratio; бинарная маска — только nearest.

Если исходная камера KB4/fisheye, сначала создайте виртуальную PINHOLE-модель:

```bash
"$FARM_PY" scripts/convert_colmap_kb4_to_pinhole.py \
  --input-model /data/scene/colmap \
  --input-images /data/scene/images \
  --output-model /data/scene/colmap_pinhole \
  --output-images /data/scene/pinhole_images
```

Конвертер использует COLMAP pixel-centre convention и корректный
COLMAP↔OpenCV half-pixel переход. После конвертации проверьте регистрацию,
размеры и sample remap; не переименовывайте fisheye intrinsics в PINHOLE.

Для Knaack фактические пути:

```text
data/knaack_colmap/          исходный COLMAP
data/knaack_colmap_pinhole/  production PINHOLE
```

## Конфиг сцены

Начинайте с `configs/scenes/example_3dgs_colmap.yaml`:

```yaml
schema_version: farm.scene.v1
scene_id: my_scene
inputs:
  colmap_model: /absolute/data/my_scene/sparse/0
  image_root: /absolute/data/my_scene/images
  gaussian_ply: /absolute/data/my_scene/scene.ply
output_root: /absolute/output/farm_pipeline
```

Для rig дополнительно задаются `camera_grouping`, `selection`, `metric_scale`,
`gravity`, `resources.model_manifest` и `resources.secrets_file`.

## Scale и gravity

- Scale должен происходить из известной базовой линии, sensor alignment или
  другого проверяемого метрического источника.
- Gravity измеряется для этой сцены; Factory/Knaack-вектор нельзя копировать.
- `resolved_up` записывается в `input/resolved_context.json` и является
  авторитетом FARM-world.
- glTF экспорт получает единственное праворукое преобразование FARM-world в
  +Y-up. Повторный ручной flip/rotation запрещён.

## Группировка кадров

Build/held-out lift делится по физическим timestamp, а не по отдельным lens или
virtual view. Один физический момент не должен попадать в обе части. Проверяйте
биекцию `frame_id ↔ timestamp_ns` и явные camera families.

Текущий selector учитывает движение камеры, COLMAP tracks, связность графа и
bridge timestamps. Он не является learned semantic set-cover optimiser. Не
заявляйте coverage-balanced 480–640 views без отдельной реализации и замера.

## Fail-closed preflight

```bash
"$FARM_PY" scripts/farm_preflight.py \
  --config configs/scenes/my_scene.yaml --strict

"$FARM_PY" scripts/farm_resource_preflight.py \
  --config configs/scenes/my_scene.yaml --verify-full-hashes
```

Preflight проверяет:

- все зарегистрированные изображения и exact names;
- sparse points/observations;
- PLY schema/count/full SHA;
- camera model и размеры;
- scale/gravity/alignment;
- model files/revisions/hashes;
- Docker tag→image ID;
- secrets existence/mode без печати токена.

Нулевое число ошибок и warnings в strict режиме — обязательное условие cold run.
