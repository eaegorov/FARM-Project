# FARM — объектная память и реконструкция сцен из COLMAP + 3DGS

FARM строит объектное представление сцены из зарегистрированных изображений,
COLMAP-камер и 3D Gaussian Splatting. Production-контур этого репозитория:

```text
COLMAP + RGB + 3DGS
        │
        ├─ 17 стадий FARM → scene_state, каталог, OBB, QA
        ├─ Gaussian lift → точный gaussian_index → instance_id
        ├─ ShapeR → опциональные mesh-гипотезы
        └─ MV-SAM3D → реконструкция выбранных объектов по реальным кадрам
```

Исходный PLY не изменяется. Каждый production-run исполняется из собственной
проверяемой snapshot-копии кода; конфиги, Docker image ID, модели и результаты
связаны SHA-256. Семантическая подпись, lift, ShapeR и MV-SAM3D — разные уровни
доказательности и не подменяют друг друга.

> Upstream-проект и статья: [arXiv:2606.15476](https://arxiv.org/abs/2606.15476),
> [страница проекта](https://goldengait.github.io/farm/), лицензия
> [AGPL-3.0](LICENSE).

## Куда смотреть

Документация сведена к пяти русским файлам; это весь основной маршрут:

1. **[Развёртывание](docs/DEPLOYMENT.md)** — новый сервер, модели, Docker,
   cold run, lift, ShapeR и viewer.
2. **[Входные данные](docs/INPUTS.md)** — COLMAP/PINHOLE, RGB, 3DGS, scale,
   gravity и конфиг новой сцены.
3. **[Качество](docs/QUALITY.md)** — смысл слоёв, acceptance, recall lift,
   MV-SAM3D, ориентация и текущие метрики Factory/Knaack.
4. **[Исходный FARM и текущий модуль](docs/ORIGINAL_VS_CURRENT.md)** — что было
   в research-коде, что добавлено здесь и какой проверяемый эффект получен.
5. **[Справочник](docs/REFERENCE.md)** — 17 стадий, CLI, выходы, порты и
   диагностика.

Если нужно просто перенести FARM на другой сервер — начинайте с
[docs/DEPLOYMENT.md](docs/DEPLOYMENT.md). Если нужно понять, можно ли отдавать
результат руководству — с [docs/QUALITY.md](docs/QUALITY.md).

Текущий общий профиль, роли моделей, измеренное время и ограничения: [3DGS + COLMAP quality pipeline](docs/QUALITY_SCENE_PROFILE_RU.md). История V12: [FARM V12](docs/FARM_QUALITY_V12_RU.md).
Структура инструментов: [scripts/README.md](scripts/README.md).

Состояние на 2026-08-24:

| Сцена | Cold run | Final / presentation | Release lift |
| --- | --- | ---: | ---: |
| Factory | `factory-production-v4-quality` | 85 / 36 | 15/21 strict, 188 051 splat, PASS |
| Knaack | `knaack-production-v8-quality` | 96 / 52 | 31/36 strict, 163 271 splat, PASS |

Это последние полностью завершённые и криптографически проверенные runs, но не
безусловная визуальная приёмка. Factory v4 отклонён пользовательским review как регрессия по
labels/OBB/recall и остаётся диагностическим baseline. Новый REST3D-inspired
full-COLMAP A/B уже дал 38 mask-pass → 27 geometry-pass → 23 whole-object
acceptance (8 label-verified, 15 `geometry_only`) за 61.26 с SAM3 stage. Это ещё
не release: до замены Factory нужен новый signed cold run.

Старый Knaack давал 323 активных объекта и визуальный лес больших OBB. Новый
surface gate снял 227 неподдержанных треков до публикации. Это повысило precision,
но не означает исчерпывающий recall всех мелких и закрытых объектов.

Отдельный Knaack MV-SAM3D deliverable находится в
`splatica_demo_app/outputs/farm_recon` и содержит 21 выбранный объект из 72
реальных PINHOLE-кадров:

- общий `combined_metric_objects.glb` в glTF +Y-up и метрах;
- `combined_world_metric_objects.npz` в исходном FARM-world;
- `combined_world_metric_gaussians.ply` в FARM-world;
- `combined_viewer_y_up_gaussians.ply` в viewer/glTF +Y-up;
- per-object GLB/PLY, исходные кадры, маски и JPEG QA-листы.

После gravity + full-mask refinement: 20 PASS, 1 REVIEW; все 21 объекта
сохранены. Общий GLB содержит 6 083 340 вершин и 12 167 794 граней; общий PLY —
10 320 512 splat-строк. REVIEW не скрывается и не удаляется.

## Быстрая проверка существующей установки

```bash
cd /home/splatica/workspace/3dgs_work/FARM-Project
export FARM_PY="$PWD/.venv-control/bin/python"

git status --short
"$FARM_PY" scripts/pin_farm_runtime_images.py
"$FARM_PY" scripts/farm_pipeline.py validate-plan \
  --config configs/scenes/knaack_3dgs_colmap.yaml
"$FARM_PY" scripts/serve_farm_unified_viewer.py \
  --registry configs/viewer/scenes.local.v1.yaml \
  --validate --verify-full-source
```

Новый cold run запускается только из чистого commit и с новым `run-id`:

```bash
"$FARM_PY" scripts/farm_pipeline.py run \
  --config configs/scenes/knaack_3dgs_colmap.yaml \
  --run-id knaack-production-NEW
```

Не используйте `--resume` после изменения кода, конфигов, моделей, image ID или
входных данных.

## Viewer

- `8080` — единый Factory/Knaack viewer: FARM, OBB, source 3DGS, dense lift,
  ShapeR.
- `8082` — Knaack FARM-world + MV-SAM3D splat overlay.

Viewer — одно-пользовательский диагностический инструмент без аутентификации.
Для удалённого просмотра безопаснее оставить loopback и использовать:

```bash
ssh -L 8080:localhost:8080 -L 8082:localhost:8082 user@server
```

В 8082 переключатель «Цвет по instance ID» означает только
детерминированную категориальную палитру по `farm_object_id`. Цвет не кодирует
confidence, качество, класс или новую сегментацию. При выключенном режиме
показывается DC-цвет реконструированных Gaussian splat-ов.

## Структура репозитория

```text
configs/                 сцены, модели, lift, ShapeR, viewer registry
src/farm_runtime/        оркестратор, snapshot/integrity, viewer
src/farm_pipeline/       preflight и production-контракты
scripts/                 CLI и 17 стадий FARM
tools/farm_shaper_bridge Gaussian lift и ShapeR bridge
docker/                  production Dockerfile
requirements/            точные host lock-файлы
tests/                   first-party regression suite
docs/                    пять русских руководств
```

## Главные правила качества

- OBB — геометрическая гипотеза, а не точная поверхность.
- Чистый цветной lift может быть неполным: неизвестные splat-ы остаются `-1`.
- ShapeR и MV-SAM3D достраивают невидимые поверхности и не являются ground truth.
- Для MV-SAM3D используются реальные PINHOLE-кадры без растяжения и blur;
  маска ресемплируется только nearest-neighbour.
- Ориентация реконструкции принимается не по PCA/OBB, а по gravity и полной
  2D-маске во всех выбранных камерах.
- Результат считается готовым только после root `_SUCCESS.json`, QA, сверки
  hash/ID/coordinate-frame и визуальной проверки сложных объектов.

Полный checklist: [docs/QUALITY.md](docs/QUALITY.md).
