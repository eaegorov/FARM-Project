# FARM V12: воспроизводимость и независимая оценка качества

Документ содержит инфраструктуру F0/F1 и обновление реализации FARM от 2026-09-05.
Текущий отчёт и изображения: `output/farm_pipeline/factory/experiments/factory-universal-v12-quality-v1/03_development/REVIEW_RU.md` относительно `3dgs_work`.
Реализованы rig-preserving RGB↔3DGS registration, angular/upright discovery ablation,
расширение exact-lift кандидатов по build surface, native-support OBB и финальный CSR allowlist export.
Шесть объектов имеют проверяемые masks/labels/captions/scope. Универсальная точность,
physical-size accuracy и общий production release пока не подтверждены.
Ручная pixel-разметка далее описана как дополнительная независимая оценка; она не блокирует разработку.

## 1. Где результат

Эксперимент:
`/home/ubuntu/workspace/3dgs_work/output/farm_pipeline/factory/experiments/factory-universal-v12-quality-v1`.

Точка входа — `REPORT_RU.md`, итоговый пакет разметки — **`02_manual_gold/packet.json`**.
`02_gold_pilot` и `02_gold_pilot_v2` — ранние диагностические версии, не gold и не inputs для обучения.

Используется исходный
`output/inpainting_research_3dgimp_semanticfoam_2026-09-04/NEXT_STEPS_PLAN_RU.md`, только раздел A.

## 2. Реализованные инструменты

| Инструмент | Что проверяет / создаёт |
|---|---|
| `scripts/quality/audit_farm_quality_baseline.py` | source files/modes, hashes inputs, реальный V11 inventory, внешние QC решения |
| `scripts/quality/smoke_farm_contributor_renderer.py` | аналитический GPU test alpha, feature VJP и depth=3 m |
| `scripts/quality/prepare_farm_manual_gold.py` | timestamp splits, exposure ledger, original-RGB observation packet, annotation templates |
| `scripts/quality/visualize_farm_gold_navigation.py` | train/dev RGB sheets; запрет декодирования test |
| `scripts/quality/prepare_farm_gaussian_inspection.py` | воспроизводимая выборка source Gaussian IDs для независимой 3D проверки |
| `scripts/quality/convert_farm_gold_labelme.py` | импорт/экспорт ручных polygon annotations без зависимости от GUI |
| `scripts/quality/review_farm_manual_gold.py` | validation, comparison, adjudication, pilot freeze и dev scoring |

Все runtime-команды выполняются в FARM Docker. Новых inference dependencies нет.

## 3. Runtime

Основной image:
`sha256:52369591dfc734774319319a5c7fcdb513c716361d186d52932307e9f8659a8e`.
Python: `/home/scene_graph/.venv/bin/python`.

Renderer image:
`sha256:cc040a3d19c048a828a927fe62f4157acbfdab88a437d0cb664afa7fa4cc958d`.
Python: `/opt/conda/envs/rest3d/bin/python`.

Для unit tests задавать `PYTEST_DISABLE_PLUGIN_AUTOLOAD=1`: системный pytest 6.2.5
конфликтует с автоматически подключаемым anyio plugin. Нужные тестам плагины не исключаются:
first-party suite не требует anyio. Это явно записанный runtime параметр.

Для JIT CUDA extension временный каталог должен разрешать загрузку shared libraries:
`--tmpfs /tmp:rw,exec,nosuid,nodev,size=2g`. Обычный Docker tmpfs может иметь `noexec`.
`MAX_JOBS=2` ограничивает стоимость первой компиляции. Kernel smoke не является сценовым benchmark.

Рабочий пример оболочки запуска из корня репозитория:

```bash
FARM_ROOT=/home/ubuntu/workspace/3dgs_work
FARM_V12="$FARM_ROOT/output/farm_pipeline/factory/experiments/factory-universal-v12-quality-v1"
FARM_IMAGE=sha256:52369591dfc734774319319a5c7fcdb513c716361d186d52932307e9f8659a8e

farm_quality_python() {
  docker run --rm --pull never --network none --read-only \
    --tmpfs /tmp:rw,nosuid,nodev,size=2g \
    --entrypoint /home/scene_graph/.venv/bin/python \
    -e PYTHONDONTWRITEBYTECODE=1 -e PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 \
    -e PYTHONPATH=/workspace/src:/workspace \
    -v "$FARM_ROOT:$FARM_ROOT:ro" \
    -v "$FARM_ROOT/projects/FARM-Project:/workspace:ro" \
    -v "$FARM_V12:/results:rw" -w /workspace "$FARM_IMAGE" "$@"
}

farm_quality_python -m pytest tests/ -q -p no:cacheprovider
```

## 4. Пакет F1

Пилот: #37/#27/#2/#108/#0/#47. Для каждого отобраны два train, два dev и два test
physical timestamps. На timestamp включены обе линзы и все пять virtual directions.
Результат — 360 object-view observations; повторные изображения разных объектов
используют один и тот же source hash.

Отбор — navigation, а не ground truth. Он использует прежние object voxels, COLMAP tracks
и проекцию support. Человек независимо определяет actual identity, видимую маску и scope.
Текущие detector/SAM/3D masks не подмешиваются в polygon templates.

Split фиксируется глобально по физическому timestamp. Две линзы и virtual crops одного
момента нельзя разнести в train/dev/test. Старые materialized RGBD frames и output images,
а также просмотренные V12 sheets входят в exposure ledger. Все оставшиеся 34 timestamp
зарезервированы для test. На них не создаются preview и не выполняется model inference.

Гарантия ограничена сохранённой историей FARM: удалённые запуски и незалогированные
просмотры других задач невозможно исключить автоматически. Новый independently captured
scene test необходим для утверждения об универсальности.

## 5. Ручная разметка

Два человека работают в независимых копиях templates, не видя разметку друг друга.
Не использовать готовые FARM/SAM masks как начальные polygon annotations этого pilot.
Оригинальные full-resolution RGB находятся по `source_image.path`; preview — только навигация.

Для каждого изображения:

1. Подтвердить object identity; проверить, что ID не переключился на соседа.
2. Выбрать состояние: `visible`, `partly_occluded`, `fully_occluded`, `out_of_view`, `absent`, `ambiguous`.
3. Разметить видимый whole-object extent. Не дорисовывать скрытый backside.
4. Отдельно разметить core, uncertain boundary, thin parts, attachments и protected neighbours.
5. Явно разметить известный background вокруг объекта. Всё остальное остаётся `unknown`.
6. Для carriers/content отдельно решить scope; полости и прозрачные участки не считать сплошным объёмом.

Слои native annotation JSON:

`whole_object`, `core`, `boundary`, `thin_parts`, `attachment`, `protected`, `carrier`, `content`, `background`.

Core/boundary/thin должны быть подмножествами whole object; core и boundary не пересекаются.
Protected/background не могут пересекаться с положительными слоями.
Незаполненные пиксели автоматически получают unknown; пустой шаблон не равен empty-object GT.
Attachment/carrier/content не становятся отрицательными пикселями автоматически.

Polygon vertices заданы в исходных пикселях `[x,y]`. Native format поддерживает `holes`:

```json
{"layer":"whole_object","points":[[10,10],[40,10],[40,40],[10,40]],
 "holes":[[[20,20],[30,20],[30,30],[20,30]]]}
```

Эти координаты — пример формата, не реальная разметка Factory.

### LabelMe

В `02_manual_gold/labelme_dev` уже подготовлены пустые файлы для dev. Они содержат ссылки
на исходные RGB без копий изображений. В редакторе использовать polygon shapes с именами
слоёв выше. Поставить `reviewed`, `identity_confirmed` и ровно один visibility flag.
Native `holes` требуют отдельных polygon strips в простом LabelMe export либо native JSON.

```bash
farm_quality_python scripts/quality/convert_farm_gold_labelme.py \
  --packet /results/02_manual_gold/packet.json \
  --output /results/reviewer_a_dev.json import \
  --folder /results/reviewer_a_labelme_dev \
  --annotation /results/reviewer_a_metadata_dev.json
```

`reviewer_a_metadata_dev.json` — копия `annotation_templates/dev.json`, в которой человек
заполнил reviewer ID, independence и object-level scope. Import polygons не выставляет эти поля.

### Scope, labels, captions, размеры

Для объекта указать `attachment_policy` (`include/exclude/separate/unknown`) и
`carrier_content_policy` (`carrier_only/content_only/whole_assembly/not_applicable/unknown`).
Неуверенный noun оставить неизвестным с пояснением. Caption должен описывать видимые
диагностические признаки и связи, не обещать неподтверждённую принадлежность деталей.

`physical_dimensions_m` можно заполнить только с `dimension_evidence`. Размеры из старого
OBB — navigation, а не измеренный reference. Указать measurement method, scale source,
object scope и uncertainty. `dimension_evidence` — объект с непустыми строками
`measurement_method`, `scale_source`, `object_scope`, `axis_definition` и тремя конечными
неотрицательными `uncertainty_m`, соответствующими осям dimensions. Это проверка полноты
записи измерения; достоверность reference подтверждают люди. Неподтверждённые dimensions
оставлять `null`. Разногласия captions и measurement evidence тоже требуют разрешения.
По одному monocular crop абсолютные метры не восстанавливаются.

## 6. Validation, disagreement и freeze

```bash
farm_quality_python scripts/quality/review_farm_manual_gold.py \
  --packet /results/02_manual_gold/packet.json \
  --output /results/review_a_validation.json validate \
  --annotation /results/reviewer_a_dev.json

farm_quality_python scripts/quality/review_farm_manual_gold.py \
  --packet /results/02_manual_gold/packet.json \
  --output /results/dev_disagreements.json compare \
  --review-a /results/reviewer_a_dev.json --review-b /results/reviewer_b_dev.json
```

Разногласия не усредняются. Сохраняются обе исходные разметки и послойное число различающихся
пикселей. Для каждого расхождения допускается отдельная human adjudication:
`schema=farm.manual-gold-adjudication.v1`, `comparison_sha256`, `split`, `adjudicator_id`,
`reviewer_kind=human`, финальная `annotation` и `resolutions` с
`disagreement_sha256` и непустым `reason`. Исходные рецензии остаются независимыми;
финальная согласованная разметка не выдаётся за третий blind pass.

Freeze принимает пары A/B для всех трёх splits, проверяет hashes исходных RGB/PLY,
не менее пяти visible physical timestamps на объект и представление каждого split.
Пилот получает `PILOT_FROZEN`, только когда всё это выполнено. Любой пропуск даёт BLOCKED.

Даже frozen six-object pilot не закрывает весь F1: нужны 20–30 стратифицированных объектов,
в том числе отсутствующие в текущем catalog, и независимые 3D inspection labels.
`cohort_expansion.json` содержит существующие 22 IDs как стартовую очередь, не полный GT.

## 7. Метрики

`evaluate-dev` требует frozen pilot, SHA-bound prediction manifest и точное покрытие всего
выбранного split. Нельзя пропустить трудный кадр или добавить test ID.

- Precision/recall/IoU считаются по подтверждённому whole object без uncertain boundary
  и по явно размеченным background/protected pixels.
- Unknown не считается negative и отчёт содержит число predicted unknown pixels.
- Boundary coverage, thin-part recall и protected-region leakage считаются отдельно.
- Сначала усредняются направления одного timestamp, затем timestamps объекта.
- None при пустом знаменателе сохраняется; оно не превращается в perfect score.

Это **conditional precision на размеченных negatives**, не обещание precision по всему кадру.
Boundary F-score, dense 3D recall, identity AP и calibrated production thresholds остаются
следующими задачами после появления actual annotations.

## 8. Текущее состояние алгоритмов и дальнейшая работа

Первый шаг теперь — проверка исходного RGB против 3DGS render, до discovery. `camera.rgb_alignment: rig`
передаётся в стандартный RGB-D ingress; joint pose correction сохраняет intrinsics, baseline и относительные камеры.
Недоверенные timestamps исключаются целиком. Cache fingerprint включает код регистрации, depth checkpoint — pose/K.
В Factory проверено 68 → 60 кадров; QA passed. Исходные COLMAP/PLY не меняются.

`farm quality --help` содержит discovery, camera-refinement, alignment-review, lift-ablation, lift-review, export-review.
CPU helpers находятся в `farm_runtime`, эксперименты — в `farm_runtime.quality`, exact contributor lift — в bridge.
Финальный gate экспортирует `mask_verified_bank.npz` по окончательному allowlist с проверкой SHA и dense ownership.
Реальный replay исключил #185; остальные production gates при этом не обходятся.

Дальше нужны новые observations на исправленных камерах, boundary/attachment/content layers, устранение
внеповерхностных выбросов перед OBB и multiview semantic evidence. Surface growth и angular selection
не объявлены универсальными defaults: пилот выявил как улучшения, так и регрессии.
Полный набор тестов: 1098 passed; после cache-правки выполнены дополнительные ingress tests и реальный RGB-D прогон.
