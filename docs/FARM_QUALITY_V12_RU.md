# FARM V12: воспроизводимость и независимая оценка качества

Документ содержит инфраструктуру F0/F1 и обновление реализации FARM по 2026-09-06.
Текущий отчёт и изображения: `output/farm_pipeline/factory/experiments/factory-universal-v12-quality-v1/45_resolved_view_support/REVIEW_RU.md` относительно `3dgs_work`.
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


## 9. Object refinement и вторая сцена

`farm quality refinement` объединяет `prepare`, `orient`, `sam`, `vlm`, `materialize`.
`lift-ablation --mask-refinement` подключает только SHA-bound build observations;
`lift-review --baseline-ablation --domain` сравнивает одинаковые источник/камеры/split.
`lift-ablation --graph-pruning` — экспериментальная локальная очистка, выключена по умолчанию.

`04_refinement/REVIEW_RU.md` в V12 содержит результаты 103 кропов, сравнение Qwen3-VL-8B
с Qwen3.5-9B, оба graph-cut пилота и Knaack ingress (48/48 кадров, 59,49 с).
SAM/VLM и graph cut пока не улучшают маски универсально; текущий reviewed bank сохранён.
Новая геометрия экспортирует `extent_stability`: чувствительность размеров к хвостам
распределения native centers не является физическим доверительным интервалом.

Исправленная вертикаль принимает список и NumPy-вектор. Для кропов учитывается pinhole
projection в луче самого объекта; обратный поворот маски сохраняет исходные координаты.
Qwen checkpoint 3.5-9B доступен локально, но автоматический object scope пока не подтверждён.
Следующий этап отделяет semantic RGB evidence от выбора масок и расширяет discovery.


## 10. Scene-wide discovery и исправление startup

Из исходного DINO import убрано глобальное включение cuDNN autotuning. YOLOE готовит текстовые признаки на выбранном устройстве пакетами по 64 категории и освобождает временную модель. В прямом L4 smoke startup + первые два кадра: 53,83 → 6,70 с; whole-process Torch peak: 21 443,98 → 1 355,62 MiB. На 48 development views сохранены 313 labels и масок (min paired IoU 0,98166). Изменение DINO включено в fingerprint mapping stage.

Добавлены `farm quality scene-vocabulary`, `discovery --variant`, `discovery-review`, `refinement vlm --semantic-only`. Узкий VLM vocabulary потерял большую часть исходных proposals; union сохранил их, но новые наблюдения нуждаются в geometry/scope validation. Новый обзор 12 RGB и всех 29 изменений хранится в `05_discovery/REVIEW_RU.md` и SHA-bound `visual_review.json`. Полный checkpoint и тесты — в V12 `STATUS.json`. Default labels и native mask bank не подменены экспериментальным словарём.

## 11. Concept discovery, геометрия и отрицательные результаты native refinement

Продолжение: `06_concept_discovery/REVIEW_RU.md` в V12 output. Новые команды `farm quality concept-discovery` и `farm quality proposal-geometry`; refinement поддерживает `sam --backend concept --prompts`. SAM3 image/text cache, depth/occlusion-aware association через FARM union-find, transient exclusion, SHA-bound build-only masks. На 12 одинаковых RGB: SAM3 1108 raw proposals → 88 multiview groups; YOLOE 84 → 7. Все группы просмотрены; это не recall. Scope частей и whole assemblies остаётся отдельной задачей.

103 concept-refinement crops и exact native lift показали регрессии на части объектов. `mask.negative_guard_pixels` и `build.minimum_visible_share` доступны только как opt-in ablation; прежние defaults сохранены. Boundary guard восстанавливает края #27, но добавляет фон #108; visible-share 0,1 это не исправляет. Основной native bank не заменён. Исправлен JSON config numeric roundtrip. Полный suite и артефакты — `06_concept_discovery/tests_final.log`, `native_ablation_summary.json`, `visual_review.json`. Следующий этап: scope/semantics новых пространственных групп до назначения окончательных object labels и OBB.

## 12. Spatial scope и learned OBB pilot

Новые команды: `scope-evidence`, `refinement vlm --scope`, `boxer`, `surface-obb`. Отчёты V12 — `07_object_scope/REVIEW_RU.md`, `08_obb/REVIEW_RU.md`. 16 пространственных групп проверены по 32 листам; Qwen3-VL inference 160,00 с против Qwen3.5 234,41 с при идентичных inputs/prompt. Labels/captions полезны как hypotheses, но mask comments и ownership ненадёжны.

Official external Boxer: 37 boxes / 10 views за 5,41 с, однако часть centers расходится на метры. Optional adapter сохраняет metric units, world-up, exact camera/image rotation, strict weight loading. Model head 5 см–4 м не подходит для физической толщины материала. RGB-D surface refit переиспользует FARM orientation/extent code без display padding; LOTO исключает и геометрию, и learned orientation контрольного timestamp. Boxer yaw не показал общего преимущества над existing FARM gravity-PCA. Native bank не заменён, скрытые размеры остаются unknown. Следующий этап — per-point cross-view visibility/conflict evidence и добор слабых осей.


## 13. Доказательства по отдельным точкам и добор ракурсов

`farm quality surface-evidence` выбирает ограниченное число новых зарегистрированных train/dev timestamps для непроверенных частей. `surface-validation` сопоставляет реальные SAM3 outputs по видимой 3D-опоре и обратному spatial overlap. Геометрическое identity matching использует существующий FARM допуск 1 px; point votes отдельно используют exact foreground и visible background дальше 3 px. Occlusion, transient, missing depth и no-instance остаются unknown. Камеры одного timestamp не множат голоса; conflicted votes сохраняются отдельно.

На трёх новых spatial groups 25/47/201 выбраны шесть RGB. У шкафа 107/4608 surface samples, у тележки 35/2648 получают минимум два отрицательных timestamp и отрицательный перевес. По RGB удаляемые группы лежат на отдельном верхнем ящике и окружающем полу/оборудовании. Fixed-orientation envelope тележки: 1,928×2,060×0,772 → 1,347×0,936×0,772 м; после diagnostic FARM PCA refit — 1,318×0,800×0,772 м. Физическая полнота этих размеров не подтверждена. У крана corroborated samples растут 1553→2528, но далёкие выбросы остаются unknown и не обрезаются.

SAM3 6-view batch: 37,81 с, включая cold load 26,01 с; peak 3892,69 MiB. Дополнительный lexical probe 2 views: 7,85 с, без улучшения маски целой конструкции. Final CPU validation + 6 RGB sheets: 3,82 с. Проверены 17 сохранённых изображений/фигур; полный набор 1162 tests passed, затем 20 targeted tests после depth-integrity guard. Отчёт: V12 `09_surface_evidence/REVIEW_RU.md`. Native bank и production defaults не заменены; следующий шаг — адаптер этих observations к существующему exact Gaussian lift с exclusion/unknown слоями.


## 14. Native observations и mixed-depth evidence

Команды native-observations / native-review / native-evaluation переиспользуют exact FARM VJP, CSR, connected growth и heldout QC. Frozen input содержит source RGB/depth/camera/mask hashes. Observation exclusions убирают transient pixels из FG/BG/visibility/QC; полностью скрытая цель остаётся unobservable. Source row order сохранён. Cross-bundle split использует source frame_id, поскольку nominal timestamp_ns перенумеровывается.

Opt-in policies: mask.positive_depth_policy / negative_depth_policy (stable default, valid experiment); mask.negative_domain (ring default, visible_background experiment); candidate.depth_gate_policy (surface default, not_behind experiment). Backprojection seeds всегда требуют stable depth. Два прежних depth gates удаляли видимую спираль тали. Согласованный rendered-contribution режим её восстанавливает; простое ослабление FG отдельно усиливает фон и отклонено.

Новые groups 25/47/201: native build 4,069→4,888 с. Build-reference IoU 0,802→0,841 / 0,916→0,918 / 0,680→0,706. Тележка на двух новых timestamps: 0,568→0,629. Шкаф на одном виде со scope mismatch control housing: 0,629→0,582. Whole-crane control SAM не прошёл association. Это automatic consistency, не human gold. Пол/груз и scope остаются нерешёнными; default bank не заменён.

V12 10_native_observations/REVIEW_RU.md содержит причинную диагностику, варианты, observed OBB, frozen control plan и визуальный вердикт. 16 новых RGBD views 23,20 с; SAM 5 controls 46,61 с с cold load 26,07 с; reverse QC двух banks 7,42 с. 1175 tests passed, затем 27 targeted checks и реальный control run после metadata fingerprint compatibility. Следующий этап — bounded build-crop refinement whole/part/content, затем legacy objects/Knaack и scene-wide runtime budget.

## 15. Crop refinement по другим physical timestamps — 2026-09-05

Отчёт: output/farm_pipeline/factory/experiments/factory-universal-v12-quality-v1/11_scope_refinement/REVIEW_RU.md.
Добавлена команда farm quality native-refinement. Она использует существующие SAM crop proposals и exact FARM VJP, исключает проверяемый physical timestamp вместе со sibling views, сравнивает foreground coverage и background leakage, сохраняет ambiguous scope как отказ. Отсутствующие и конфликтующие native votes не становятся фоном.

При недостаточном абсолютном покрытии исходного RGB новая маска должна сохранять исходное покрытие с отдельным identity floor. Сильно противоречивое object/view observation без подходящей замены исключается как unknown; другие объекты и минимум два timestamps сохраняются. Изменение crop не переписывает область за его пределами. Новый режим доступен только отдельной development-командой; глобальные параметры не изменены.

Девять crops, 138 proposals: две замены и один исключённый вид. Тележка: native count 16923→14638; на двух уже открытых development controls IoU 0,6292→0,6553, precision 0,7221→0,7852, recall 0,8524→0,8155. Визуально очищены просветы рамы и верхний груз, но нижний груз/часть границ остаются. Шкаф 14788→14681 с почти неизменным control IoU; принадлежность левого housing нерешена. Кран сохраняет все 39114 native IDs и спираль. Физические размеры не валидированы.

Tracker девять crops 18,00 с; concept 57,48 с (альтернативные диагностические ветви). Выбор по сохранённым proposals 9,77 с, native rebuild около 4,3 с. 46 изображений фактически просмотрены; 1182 tests passed, затем 34 targeted после provenance checks и проверка всех реальных proposals. Human gold и scene-wide runtime этими результатами не заменяются.

Следующий шаг: перенос на legacy шесть Factory объектов и Knaack, затем общий scene-wide профиль с ограниченным бюджетом refinement; не расширять число эвристик на единственном примере.

## 16. Перенос на Knaack и границы общих параметров

Отчёт V12: `12_scene_validation/REVIEW_RU.md`. `native-observations --geometry --world-up` принимает multi-timestamp группы напрямую, сохраняя existing FARM geometry/lift. Общий writer проверен на прежнем Factory validation input: 12 масок по массивам, 11 exclusions, split/OBB совпадают. Старый size/mtime PLY fingerprint проверяется перед новым SHA binding; историческая степень проверки сохранена явно.

На шести Factory targets rendered contribution даёт mixed result: пожарный шкаф median IoU 0,8488→0,9488 с лучшим исключением трубы; огнетушитель 0,7762→0,7445 с небольшой потерей тела. 2px negative guard повышает и покрытие, и фон. Глобальные defaults/старый reviewed bank не изменены.

Knaack: Qwen4B 8 RGB 35,71 с; SAM3 12 RGB / 56 prompts 88,72 с; 540 proposals → 342 static surface nodes → только 6 multi-timestamp групп (7,01 с). Native build шести групп 4,60 с, 1 790 MiB peak. Все 6 geometry sheets и 12 native crops просмотрены: коробчатые поверхности плотные, колонны/подъёмник частично неполные, тележка из-за окклюзии подтверждена только фрагментом. Физические размеры не валидированы; это не полный cold scene runtime.

VLM иногда принимает cyan overlay за защитную плёнку. Scope evidence теперь показывает отдельную бинарную схему и original RGB; copied schema placeholders и scope enums вместо label отклоняются. Бинарная схема устраняет эту цветовую примесь, но 4B/8B всё ещё ошибаются в смысле/материале малых crops. 8B не показала достаточного преимущества для автоматического принятия captions. Final full suite: 1191 passed / 70,53 с.

Следующий приоритет: ограниченный добор зарегистрированных ракурсов по видимости уже найденных объектов → подтверждение singleton/occluded targets → contextual RGB semantics → единый scene profile с общим бюджетом. Не лечить недостаток наблюдений ослаблением native voting.

## 17. Добор ракурсов с фиксированным бюджетом

Отчёт V12: `13_adaptive_coverage/REVIEW_RU.md`. Новая команда `quality discovery-coverage` выбирает ограниченный batch ещё не сегментированных RGB по приросту геометрической видимости, с одним видом на physical timestamp. Она планирует segmentation, не создаёт foreground votes. Старые proposals переиспользуются.

Knaack при одинаковых 8 дополнительных RGB: adaptive 40 multi-timestamp groups против uniform 18; исходно 6. SAM3 55,58 / 53,65 с, association 17,66 / 14,61 с; selection 2,35 с. Native build 40 кандидатов 17,78 с, после gates 36. Исходные 342 nodes сохранены. На тех же старых автоматических 2D references IoU группы 10 0,7871→0,9189 и 13 0,9011→0,9322; uniform их не меняет. Это согласие с build masks, не независимая accuracy. Тележка улучшена обоими доборами одинаково; фон и груз остаются.

67 изображений просмотрены, точный список в visual_review.json. Новые проблемы: основание конуса и тонкие части кронштейна неполны; окно имеет неоднозначную физическую depth ownership; верх трубы 202 теряется в общем банке с конкурирующей part-гипотезой 210. Следующий шаг — изолировать конкуренцию whole/part до exclusive native ownership, затем contextual semantics и единый ограниченный scene profile. Full suite 1198 passed / 66,64 с; параметры production и reviewed Factory bank не заменены.

## 18. Вложенные scope и отрицательное свидетельство конкретного кандидата

Отчёт V12: `14_scope_ownership/REVIEW_RU.md`. Для трубы Knaack 202 изолированный lift даёт 572 native IDs, добавление одной вложенной гипотезы 210 оставляет 184: 275 общих strong claims переходят в unknown. Добавлен `native-observations --scope-alternative-budget N`: отдельные пересекающиеся гипотезы из cached VJP, без объявления physical part-of и без замены основного exclusive bank. Четыре альтернативы занимают 1,28–1,30 с.

Новая opt-in настройка `mask.other_mask_negative_policy: background` не позволяет большой proposal mask выключать отрицательные свидетельства вокруг маленькой. На наклейке подъёмника IoU к прежней build mask 0,1454→0,8463, precision 0,1454→0,9107; проекция перестаёт захватывать корпус. Труба/подъёмник в alternative scopes сохранены. Это build references, не независимая accuracy; default exclude пока сохранён для проверки переноса на Factory.

`quality refinement vlm --scope --scope-context` показывает общий исходный RGB, enlarged crop и отдельную binary mask. 4B лучше называет часть объектов, но long-form scope ненадёжен (1/12 valid schema, ошибки parent/whole). Короткий pair-specific pilot правильно распознаёт две маски трубы как один объект, но нет negative controls и путается тип связи наклейки. Автоматическое merging не включено. Прозрачное окно имеет некорректный для физического объекта native envelope (~24 м), остаётся geometry uncertainty. Full suite 1205 passed / 63,64 с. Следующий batch — вложенные объекты Factory и единый scene profile.

## 19. Перенос candidate-specific background на Factory

Отчёт V12: `15_scope_transfer/REVIEW_RU.md`. Девять вложенных geometric targets, одни masks/7 timestamps, legacy exclude vs background. Все 24 сравнения просмотрены. Ручки 173/180: mean build-reference IoU 0,0615→0,7945 и 0,0856→0,7913; исчезает большая примесь дверей. Корпус 47 неизменён, панели немного чище. Кран 0,7601→0,7562 (recall немного ниже, precision выше); это проверка на его двух исходных observations, не замена позднего adaptive refinement.

Native stage с девятью scope alternatives: 6,42 / 6,97 с, 1437 MiB; alternatives из cached VJP 1,19 / 1,29 с. Рекомендуемый режим для нового quality profile — candidate-specific background + transient/depth unknown + сохранение nested scope hypotheses. Исторический default сохранён. Выпуск exclusive bank требует согласованного выбора physical scope. Код не менялся; применимы 1205 tests из этапа 14.

## 20. Короткое описание объекта отдельно от physical scope

Отчёт V12: `16_compact_semantics/REVIEW_RU.md`. `quality refinement vlm --compact-semantics` использует contextual RGB + PHOTO + binary mask и узкую схему из 4 полей, max180 tokens. На одинаковых 24 canvases 4B: 12/12 valid против длинного scope 1/12, inference 32,10 против 80,24 с. Проверены все 24 изображения. Это улучшение формата и полезности черновых описаний, не доказанная semantic accuracy. Ошибки колонна/window frame, sign/decal и неподтверждённые material/orientation сохраняются. Empty uncertainty не означает уверенность.

8B при том же запросе: 53,06 с inference, 17310 MiB; два strict ID-order failures, содержательные регрессии crate→concrete pillar и pallet→box. Замена на 8B не принимается. Model-load time зависит от cache и отдельно указан в отчёте. Исправлен SAM writer с ошибочным VLM-only полем; 1215 tests passed / 69,40 с. Следующий шаг — связный bounded quality run и каталог с явным native/scope/physical-size статусом.

## 21. Общий quality scene profile и каталог

Отчёт V12: `17_scene_profile/REVIEW_RU.md`; руководство [QUALITY_SCENE_PROFILE_RU.md](QUALITY_SCENE_PROFILE_RU.md). Существующий FARM orchestrator выполняет 13 bounded этапов, snapshots и provenance сохранены. Knaack: 40 multi-timestamp candidates, 36 primary native masks, 4 nested alternatives. Native IDs и support точно совпали с этапом 14. Сумма успешных stages 517,41 с после готового RGBD; два запуска с исправлением identity adapter между ними, не uninterrupted cold run. 1224 tests passed / 85,13 с.

Все 40 compact JSON валидны, но визуально обнаружены context distraction (потолочная труба→каска человека) и wall/crate ошибки. Каталог хранит модельные proposals, observed OBB и unresolved physical size/scope. Следующий шаг — local-context comparison на той же когорте, затем bounded crop refinement. Universal release ещё не подтверждён.

## 22. Local context и masked target RGB

Отчёт V12: `18_local_context/REVIEW_RU.md`. Два opt-in ablations на тех же40 candidates не дали устойчивого улучшения: 322 helmet→shoe→tool; тонкая деталь109 всё ещё wall. Masked RGB помогает ящикам/некоторым малым компонентам, но не обосновывает общий default. Просмотрены6 страниц/16 объектов, неизменённые панели проверены на80 реальных canvases. GPU contention исключает честное сравнение новых timings с baseline.

Validator appearance теперь канонизирует порядок только полного множества уникальных evidence IDs; raw сохранён. 1232 tests passed /69,60с. Следующий шаг — автоматический bounded crop scheduler по other-timestamp contradiction, без ручных ID списков.

## 23. Автоматический выбор crop refinement

Отчёт V12: `19_refinement_schedule/REVIEW_RU.md`. Новый `quality refinement-schedule`: >=3 physical timestamps, current timestamp целиком исключён из scoring, общий crop budget12/per-object2, приоритет по возможному gain того же selector score. Исправлен приём больших расширений в unknown:96,32% добавленного участка трубы не поддерживались OTHER foreground. Теперь такая proposal отклоняется. Factory replay сохраняет прежние полезные замены.

Factory6crops автоматически воспроизвели очистку cabinet/cart и quarantine; cart exposed-control IoU0,62920→0,65851 (прежний targeted0,65535), сумма дополнительной работы55,36с. Knaack12crops дали3 небольших изменения; precision немного выше, recall к old build masks ниже. У lift сохраняется выступ уже в >=2TS core, cart неполна. Следующая проверка — alpha/occlusion footprint и native ownership, затем интеграция с актуальными semantic masks.1244tests /64,70с.

## 24. Проверка alpha footprint

Отчёт V12: `20_native_footprint/REVIEW_RU.md`.22 object/view projections на неизменённых Gaussian IDs,8 sheets просмотрены, включая все4lift views. Повышение общего alpha .04→.25 уменьшает выступ lift, но снижает его recall к прежним автоматическим masks .9396→.8283; у crane .8900→.8021. Conditional share практически совпадает с mass. На видахlift2/7 большая часть excess лежит перед expected scene depth, поэтому скрытый объект за окклюдером не объясняет весь выступ. Defaults и membership сохранены; нужен адресный contributor audit.

## 25. Refinement внутри quality DAG

`scene-profile plan --refinement-crops 12` добавляет5 stages к прежним13. Budget0 сохраняет прежнюю последовательность. Пустые schedules пропускают SAM/CUDA; unchanged selection не пересобирает native bank. Appearance получает актуальный native input, пропускает quarantine, использует изменённые masks и дополнительные source-bound views. Разные mask grids сравниваются по доле foreground. Catalog проверяет input binding и сохраняет native timestamp count отдельно от исходной геометрии. Реальные прогоны и проверки фиксируются в `21_refined_profile/`.


## 26. Непрерывные прогоны Factory и Knaack

Чистый snapshot4784c3d: оба18-stage runs успешны. После готового registered RGBD Knaack10:39 (40candidates/36primary/4alternatives/40descriptions), Factory17:25 (174/117/16/64). Это не cold/raw-input timing и не semantic accuracy. Knaack final bank точно воспроизводит Stage19, все80mask bindings проверены. Factory audit локализовал потерю poster в association, container/fire cabinet в недостатке observations, wall unit27 в discovery. Отчёт V12/21_refined_profile/REVIEW_RU.md.

## 27. DAM-3B с прямой маской

12objects×2views finalKnaack: все24captions и12pages просмотрены. 50,13с total,7074MiB; устойчивого выигрыша над Qwen нет (airplane, plant branch, forklift, barrel и cardboard hallucinations). Default replacement отклонён. Весовая лицензия Noncommercial, backend в FARM не добавлен. Отчёт V12/22_region_captioning/REVIEW_RU.md.

## 28. Частичная видимость и адресное заполнение 3D-маски

Opt-in `quality proposal-geometry --partial-view-association`: только для реального пересечения FOV edge учитывает in-frame/covisible support, сохраняя depth/negative/mutual/cannot-link ограничения. Nested clipped part не определяет whole-object identity при наличии larger same-frame proposal. V1 whole-pallet/plank ошибка выявлена визуально и исключена вv2. 1255tests/66,27с.

На фиксированных20RGB Factory174→184multi-TS, Knaack40→43; это counts, не recall. Poster group0 восстановлен из двух односторонне полных observations. Адресный добор2views существующими surface-evidence/SAM/validation/native этапами за17,12с даёт4162→32672Gaussians, mean build-reference IoU0,4610→0,9381, recall0,4709→0,9727. Все4crop comparisons просмотрены; небольшая wall leakage сохраняется. Physical thickness не подтверждена. Whole-scene default не менялся. Отчёт V12/23_partial_view_association/REVIEW_RU.md; следующий приоритет — автоматический выбор слабопокрытых objects для bounded followup.

## 29. Дополнительные предложения без разрушения primary groups

Отчёт V12/24_complementary_discovery/REVIEW_RU.md. `scene-profile union --max-primary-iou 0.5` сохраняет68из148 дополнительных YOLOE proposals на тех же20RGB. Source priority действует на mask representation, correspondence и component merge; primary person exclusions сохранены. Все998primary nodes/points и656исходных групп совпадают; итог1063nodes/699groups/192multi-TS. Все17изменившихсяmulti-TS groups просмотрены. Raw union, раздробивший огнетушитель, отклонён.

`quality surface-tracker` подтверждает найденный YOLOE настенный блок в двух новых ракурсах:622/627points,1252nativeIDs/3timestamps. CLI18candidate masks совпали с pilot,3crops5.17с. Но третий native render теряет часть чёрного основания: build IoU0.898 не означает полный объект. Общий source bank сохранён. Full-frame semantic queries0/3matches; expanded prompt boxes захватывают стену и отвергнуты. Local concept с расширенным RGB context выделяет целый пожарный шкаф, но reciprocal surface support0.267 отражает неполную глубину исходного observation. Шкаф остаётся unresolved; global threshold не снижен. Следующий приоритет — plane-aware identity/whole-object scope, затем перенос bounded recovery на Knaack и общий DAG.1268tests/65.30с.

## 30. Partial FOV в дополнительной валидации

`quality surface-validation --partial-view-association` использует Stage23 guarded partial-view comparison для кандидатов, не прошедших обычную reciprocal проверку. Default сохранён; FOV edge, visibility, negative и nested-scope guards обязательны.1271tests/67.27с.

Пересмотр исходного RGB показал, что пожарный шкаф обрезан границей камеры, а не просто неполон в depth. Имеющийся full-frame SAM кандидат теперь принят без дополнительной сегментации,905points corroborated. Native2timestamps/5120IDs чисто покрывает правую часть, но левая дверца неполна; обе проекции просмотрены. Новых допустимых views в текущем48-frame RGBD pool не найдено. В исходном Factory6310images: следующий шаг — bounded selection из полного COLMAP pool с регистрацией выбранных камер, исключая закрытые timestamps. Plane/SIFT pilots не приняты. Отчёт V12/25_planar_identity/REVIEW_RU.md.


## 31. Полный camera pool и разрешение scope ambiguity

V12/26_camera_completion/REVIEW_RU.md: новый camera-completion выбирает2 timestamps из6310 COLMAP metadata до RGB; регистрация8views сохраняет anchors/исходную поверхность. Из двух новых направлений одно окклюдировано: depth screening должен идти до SAM. Geometric crop + Qwen scope-review с двумя порядками кандидатов восстанавливают целый fire cabinet. VLM допускается только при согласии с уже лучшим геометрическим кандидатом; immutable inputs/candidate evidence проверяются. Native3timestamps/13872IDs вместо5120: на тех же3build views meanIoU0,67918→0,95375,recall0,69541→0,99005. Все3сравнения просмотрены, внешняя труба исключена, небольшие погрешности границ остаются. Это не independent accuracy; observed thickness0,265m чувствительна на40%, physical extent неизвестен.1282tests/64,86с. Эти opt-in utilities ещё не входят в общий18-stage DAG; далее перенос на unit/Knaack и глобальный recovery budget.


## 32. Повторное использование observations и полнота unit

`scene-profile cohort --validation PATH --group-id ID --output NEW` сохраняет logits/person exclusions/flattened source indices и связывает следующий recovery с принятыми observations. Replayunit воспроизвёл1252Gaussian IDs точно. Четыре новых теста;1286fulltests/64,22с.

Stage27показал, что1387/1393corroborated points могут описывать лишь неполную светлую часть unit. Ещё2готовых RGBD views и projection-tracker повторяют ошибку;YOLOE не обнаруживает unit. Targeted VLM bbox с сохранением уже подтверждённого bbox дал2полные SAM masks, принятые существующим geometry validator. Варианты с VLM point на стене не выбраны. Native5timestamps/2087IDs восстанавливает чёрную грань на всех5просмотренных projections; новые whole-mask build references IoU0,67629→0,91737,recall0,70816→0,97026. Это не independent gold, старые2D references неполны. VLM completion остаётся experiment, default не меняется.

Штатные semantics/catalog используют обновлённые masks:unit `wall-mounted device`, fire cabinet `fire hose box`. Каталоги V12/27_recovery_transfer/catalog_v1 и26_camera_completion/catalog_v1; observed OBB, physical-size unknown. Далее распространение полного scope на старые2D masks и Knaack; общий recovery budget в DAG пока не включён.


## 33. Полный scope перенесён в старые2D masks

V12/28_scope_propagation/REVIEW_RU.md. `refinement-schedule --balance-error-modes` резервирует второй crop объекта для другого типа foreground/background ошибки, сохраняя бюджет и defaults. В общем18-stage профиле доступно `--balance-refinement-errors`. Unit:вместо000348+004192 выбраны000348+003285; две tracker замены приняты прежними other-timestamp gates. Чёрная грань появилась в старой2D маске, native2041IDs/5TS сохраняет полноту. Все5reverse comparisons просмотрены; изменения3D metrics малы и не трактуются как independent accuracy. Semantic sheets/ответ совпали с Stage27, итоговый catalog в28_scope_propagation/catalog_v1.1288tests/64,41с. ReplayFactory6crops не меняет, Knaack заменяет1из12; новый Knaack crop требует проверки.

## 34. Проверка Knaack и общий visible-scope completion

Knaack balanced scheduling не снял ambiguity88/image8. Все14candidates просмотрены; исходный bank сохранён. Crop и модельные arrays совпадают с ранним Stage19: нужен reuse, не повторный inference того же запроса. V12/29_knaack_balance/REVIEW_RU.md.

Новый `quality scope-completion` работает только с SHA-bound accepted surface-tracker masks, учитывает flattened supplement indices и несколько объектов в одном кадре. Бюджет4crops/2distinct timestamps per object; VLM→SAM только при high-confidence видимой недостающей части, bbox текущего объекта сохраняется. Новые masks должны пройти прежнюю geometry validation. Cabinet control complete→SAM не загружается. Unit12logits точны кStage27; native2087IDs/5TS точны, float confidence отличается≤1,2e-7. Все3новые sheets просмотрены.1308tests/65,23с. V12/30_scope_completion/REVIEW_RU.md; актуальный refined unit catalog остаётся Stage28. Модуль ещё не включён в18-stage DAG; следующий этап — global recovery budget/reuse.

## 35. Reuse SAM и лёгкие geometry imports

`refinement sam --reuse-proposals` и `scene-profile plan --refinement-reuse` связывают cache с actual crop/seed/model/processor/code/library fingerprints. Идентичные запросы выполняются без моделей/CUDA, но native scoring использует текущие observations. Все14Knaack candidates и selector metrics совпали с Stage29. Eager services/worker imports в captioning package заменены lazy public API; geometry/evidence helpers не загружают inference libraries. CPU replay двух SAM stages13,71→5,33с,1320tests/65,68с. V12/31_refinement_reuse/REVIEW_RU.md; полный scene timing требует отдельного прогона.

## 36. Полный Knaack regression после Stage31

Все 18 этапов прошли за 581,934 с от готового registered RGBD. Состав 36 основных масок, все 43 666 назначений гауссиан и timestamp support точно совпали со Stage21. Различия confidence не превышают 1,8e-7, размеров OBB — 2,7e-8 м. Совпали 40 labels/captions и 161 appearance/native image; в каждом SAM backend использован один cache hit. Прежние семантические ошибки остались. Отчёт: V12/32_common_regression/REVIEW_RU.md; исследовательские выводы: RESEARCH_UPDATE_RU.md рядом.

## 37. Дополнительные предложения в общем профиле

Флаги `--partial-view-association --complementary-model-root PATH` включают 20 этапов с refinement, сохраняя бюджет RGB. После adaptive SAM выполняются YOLOE и объединение с приоритетом основного источника и порогом IoU 0,5. Проверяются RGB, сетка, ориентация и checkpoint; поддержаны небольшие исходные изображения и пустая cohort. Записываются фактические веса словарных голов и MobileCLIP.

На Factory совпали все 148 массивов масок, 1063 nodes, 699 groups, 192 multi-timestamp groups и 192 обзорных изображения со Stage24. На Knaack получены 114 предложений, сохранены 652 основных узла и 599 их групп; подтверждена ещё одна группа. Её 368 native IDs и пять новых изображений проверены: тонкие опоры неполны, две оси OBB чувствительны к хвостам, название Qwen `concrete slab` не принято как подтверждённый материал. Прошли 1330 тестов за 65,28 с. Следующий контроль — полный Factory run: initial partial-view association может изменить выбор дополнительных снимков. Отчёт: V12/33_complementary_profile/REVIEW_RU.md.

## 38. Полный Factory run и оставшиеся ошибки

Stage34 прошёл все 20 этапов из snapshot400c251 за 1301,40 с. Из них около 232 с заняла ошибочно широкая проверка model root. Исправлениеf446556 проверяет только три checkpoint, 810 МБ; отдельное измерение заняло 3,63 с. Это не новый полный benchmark. Прошли 1333 теста за 68,85 с.

Каталог содержит 128 кандидатов, 114 непустых native-масок и 64 описания при 192 multi-timestamp groups. Набор RGB сохранился; итоговые группы совпали с компонентным replay после сопоставления IDs. Из 117 прежних native-масок 67 точны, семь стали пустыми, семь отложены бюджетом, два случая не имеют однозначного соответствия. Whole/part/collection и глобальная конкуренция масок требуют отдельного решения.

Контейнер, настенный блок и пожарный шкаф остаются без native из-за singleton-групп; постер неполон. Просмотр всех 64 paired PHOTO examples и двух полных canvases подтвердил также ошибки VLM: пол назван тележкой, окно — дверной панелью. Далее проверяются target-grounded VLM inputs и общий recovery с корректной передачей восстановленных наблюдений в native, semantics и catalog. Новые flags не объявлены универсальным release. V12/34_full_complementary_factory/REVIEW_RU.md.

## 39. Привязка описания к целевой маске

Stage35 проверил masked RGB на восьми Factory-кандидатах, по два timestamp. Три ошибочных описания окон исправились, но floor mask по-прежнему описана как посторонний объект: cart → pallet. Контрольные объекты узнаваемы. Просмотрены все парные виды; 42,49 с. Глобальный masked default не принят. V12/35_semantic_grounding/REVIEW_RU.md.

## 40. Recovery observations в общем native-входе и каталоге

Коммиты ce84004/c6e7ca5 добавляют `--recovery-validation`, сохранение исходного native-бюджета и до 16 дополнительных подтверждённых групп. Semantic selection учитывает фактические timestamp после recovery/quarantine и получает отдельный ограниченный бюджет для восстановленных объектов. Исходный и фактический счётчики в каталоге разделены.

Factory: 128 → 130 кандидатов, 114 → 116 непустых native masks, 64 → 66 описаний. Все прежние 114 Gaussian memberships сохранены при точном baseline-конфиге. Устройство 517: 1252 IDs/3 timestamp; шкаф 634: 5120/2. Просмотрены все пять native проекций и новые semantic canvas: недобраны чёрная часть устройства и левая часть шкафа. Контейнеру нужен новый ракурс. Старые более полные локальные Stage28/26 результаты остаются development-контролями. 1343 tests / 66,16 с; Knaack сохраняет выбор 40 кандидатов без inference. V12/36_recovery_bridge/REVIEW_RU.md.

Автоматический recovery scheduler ещё не включён в общий DAG. Следующий контроль — приоритет предложений дополнения корпуса перед старой частичной маской при неизменных геометрических gates; labels и физические размеры не объявлены проверенными.

## 41. Приоритет полноты корпуса и выборочное обновление описаний

Stage37 исправляет выбор между допустимыми tracker/completion masks:high-confidence incomplete, прежние identity gates, покрытие≥90% старой маски, ambiguity fallback. Общий Factory result:устройство1252→1731IDs, все старые1252 сохранены, остальные115 memberships точны. Три native views и увеличенное сравнение подтверждают восстановление противоположной чёрной боковины без проводов.

432 decoded mask archives финального input совпали с контролем; повторный lift не нужен. Разные SHA NPZ между runtime объяснены одинаковыми массивами. semantics --retain-stage сохранил65 аннотаций и обновил517 с другим ракурсом; appearance15,57с. Каталог130/116/66. Это сохранение ответов с provenance, не model cache и не semantic acceptance.1375tests/68,34с. Подробности:V12/37_scope_priority/REVIEW_RU.md. Далее автоматический recovery schedule.

## 42. Автоматический recovery выполнен на двух сценах (2026-09-06)

Stage38, код109ea11: scene-profile plan --recovery-groups добавляет семь стадий перед native. Выбор без ручных ID: source/size strata, static-depth support, до64 кандидатов,12 новых RGB и2 timestamp на объект. Отсекаются неразрешённые кадры и чужие tracker/completion proposals. Default остаётся0.1384tests/68,25с; пустая очередь проходит все7 стадий без модели/depth.

Обычные continuations от готовой geometry: Factory909,150с, Knaack462,097с; это не cold raw-scene timing. Factory132кандидата/117масок/68описаний: новые159(126IDs),517(1731),614(366);490 подтверждён в2D, но native пуст.113/114 старых memberships точны,294 теряет2IDs в unassigned. Устройство совпадает со Stage37. Knaack42/37/42:399 получает480IDs/3TS,63 имеет пустойnative; все36старых memberships точны.

Просмотрены native projections и semantic evidence.159 — плоская панель, не доказанный целый шкаф;614 — содержимое чёрного контейнера, не весь контейнер;399 — светлая строительная деталь, concrete не доказан. Автоматический выбор не включает пожарный шкаф634 и не является superset ручного Stage37. Пустые490/63 требуют отдельного диагноза. Rich scope review новых4Factory объектов дал0/4valid и не применён. Отчёт V12/38_automatic_recovery/REVIEW_RU.md.

## 43. Дополнительные scope поля и layout не дали общего улучшения (2026-09-06)

Stage39:8cases/2TS, Factory14/153/159/490/517/614,Knaack63/399.5-field scope:16calls/60,788с; diagram7/8valid,masked8/8valid, но все8masked ответы присвоили surface_region, включая устройство/ленту/трубу. Такой тип нельзя использовать для ownership. Исходные16standard JPEG воспроизведены побайтно.

Две панели CONTEXT+MASKED PHOTO и4поля:8calls/31,364с,8/8valid,1360input tokens. Исправлены окно14 и пол153, но панель490 стала целым industrial cabinet, плоская399 — box. Общего улучшения нет; production defaults и каталоги сохранены. На просмотренных изображениях вырезы159 действительно входят в target, материал399 не установлен. Отчёт V12/39_compact_scope/REVIEW_RU.md.

Дальше: контроль альтернативной VLM на этих же пикселях с учётом прошлых Qwen3.5-9B тестов, диагноз пустыхnative63/490, затем coverage/ranking и ограниченный добор зарегистрированных ракурсов. Physical dimensions остаютсяnull, универсальное качество не заявлено, closed test не открыт. Только Project A.

## 44. Сравнение другой VLM на коротком описании

Stage40: Qwen3.5-9B проверена на тех же8cases×2layouts,32 JPEG побайтно совпали с4B inputs.16запросов/248,930с, из них153,935с загрузка.15ответов обёрнуты в JSON fence, один проходит исходный validator; raw сохранены, payload разобран отдельно только для анализа. Есть исправления окна и некоторых панелей, но появились tape dispenser/white plastic pipe и неподтверждённые материалы. Общего выигрыша нет, модель/default не заменены. Веса уже имелись после Stage04/07. Отчёт V12/40_semantic_model_transfer/REVIEW_RU.md.

## 45. Scope relations используют текущий native input

Stage41,345d4e6,1391tests/76,62с: вложенность пересчитывается после recovery/refinement/quarantine по действующим build masks. Heldout не читается, transient pixels исключены, один физический timestamp — один голос. Factory45→47 связей без потерь старых; Knaack2→2. Пересчёт1,668/0,340с. Primary memberships и timestamp support обеих сцен точны, max confidence delta1,788e-7. Factory alternatives134:+32IDs,338:+1227/-4;13остальных общих точны, все4Knaack точны. Три old/new boards просмотрены.

Пустая490 — в основном конкуренция с уже существующей294:1474из1819supported IDs принадлежат294. На двух просмотренных видах обе выбирают одну панель; создавать второй физический объект неверно. Диагностическое исключение294 даёт2112IDs, не применено. Knaack63 имеет другую причину:333из334spatial candidates без положительного timestamp, один имеет только один. Отчёт V12/41_current_scope/REVIEW_RU.md.

## 46. Пустая труба: contributors и смесь rendered depth

Stage42: global contributor search334→2016 не помогает при старом center-depth gate. Без этого gate получены679IDs; с сохранением проверки и допуском max(старый,2*depth_sigma) —622IDs. Все четыре native views просмотрены: маска следует висящей трубе, без заметного захвата стены/пола; верхний/скрытый участок не полностью восстановлен. Пересечения с прежним primary bank нет, но конкуренция всех объектов ещё должна быть повторена.

ED сохранена корректно и уже нормализована по opacity; source render совпал точно. Scene alpha≈1, вклад трубы≈0,44/0,54, средняя depth≈6,8/6,6м, глубина её Gaussian≈8,1/7,9м. Из E[z²] измерен sigma≈2м: среднее смеси не является точной поверхностью.2*sigma — эвристика, не калиброванный confidence interval.

Следующий конкретный этап — bounded contributor fallback для групп с0supported claims до конкуренции: сейчас Factory0таких случаев,Knaack1. Реальныеcontributors + depth spread, прежние multiview/purity/exclusion gates, затем полная общая конкуренция и проверка старых masks. Budget≤4objects,≤4TS/8views, bounded candidates. Пока prototype, defaults не менялись. Отчёт V12/42_contributor_candidates/REVIEW_RU.md. Closed test не открыт, physical dimensions не валидированы.

## 47. Contributor fallback включён в общий native pipeline (2026-09-06)

Stage43,b41225c,1404tests/71,59с. Quality-config budget4; только0supported claims до конкуренции,2–4TS/≤8views,≤65536candidates, visible-background negatives. Exact contributors и measured depth spread проходят затем общую конкуренцию всех объектов и прежний connected refinement. Неизвестные/heldout pixels и budgets не обходятся.

Knaack native46,488с:37→38масок,63получил622IDs; все37старых memberships точны. Fallback2VJP/0,288с. Factory160,878с:все117масок точны,0fallback VJP. Новая труба и оба native JPEG совпали с просмотренным Stage42 prototype. Gaussian/PLY исходники не менялись.

Semantics retain-stage сохранил42/68ответов без VLM; refresh0,402/0,609с, catalogs2,258/2,704с. Каталоги V12/43_contributor_recovery/knaack/catalog_v1 иfactory/catalog_v1:42/38/42 и132/117/68. Labels/captions прежние, включая известные ошибки. Observed OBB трубы1,738×0,665×1,384м tail-sensitive, физические размеры не приняты. Отчёт V12/43_contributor_recovery/REVIEW_RU.md.

Следующий приоритет — novelty при выборе recovery объектов: не тратить ранние слоты на уже покрытую панель490, проверить coverage для634и новых физических объектов без ручных ID. Затем ограниченный добор зарегистрированных ракурсов для неполных масок. Closed test не открыт, Project B/C не меняются.

## 48. Новизна recovery-областей проверена в общем пайплайне (2026-09-06)

Stage44,bc6256e,1412tests/67,21с. В прежних source/size strata score умножается на1−maximum same-frame multiview mask IoU; это приоритет, не merge/exclusion.490опущена1→232, Factory9дополнительных RGB вместо11.634поднялась219→174 и пока вне64кандидатов. Для Knaack очередь8групп,plan/concepts и72point arrays точны; новый inference не нужен.

Factory15стадий/663,467с от готовой geometry,132/117/68.116из117прежних nonempty memberships точны,294возвращены2неназначенных IDs.392native JPEG побитово совпали соStage43.24SAM refinement requests reused,67annotations retained,1новый VLM для170. У170 native0/geometry reject и ошибочное wall corner; прирост принятого качества не заявлен.

Просмотрены3source pages/11строк и сравнение слабых target masks. В17049pxмаски дают лишь4пригодных depth samples, в159—18; плотные source points ошибочно позволяли принимать такой target. Stage45 исправляет минимум на обоих концах и повторно проверяет готовые proposals. Затем: общий bounded camera recovery/coverage и устранение лишних промежуточных native visuals (измерено85,65с на Factory). Каталог и отчёт V12/44_novelty_schedule/factory/runs/novelty-v1/quality/catalog иREVIEW_RU.md. Физические размеры/универсальный release не приняты, closed test не открыт.

## 49. Минимальная target-surface опора и проверка исправления (2026-09-06)

Stage45,950bc59,1414tests/68,75с. Обычный recovery match теперь требует≥20пригодных target samples, как source geometry и partial-view ветвь. Плотные source points и completion preference не обходят минимум. На готовых proposals Factory170(4samples) перестал подтверждаться;159перешёл с87px/18samples на218px/44samples. Новых segmentation/VLM calls при повторной проверке0. Knaack accepted masks и56point arrays точны; новый native не запускался.

Factory8stage continuation506,729с от готовых proposals:131/117/67.159:126→165IDs, все126сохранены; остальные116 memberships точны.390из392native JPEG точны. Обе изменившиеся native проекции иtarget mask comparison просмотрены: охват панели больше, но у нижнего края близкого вида есть небольшая примесь соседней панели. Полная маска не принята.24SAM requests reused,66annotations retained,1новый VLM; cabinet door/dark base всё ещё недостаточно подтверждены.

Каталог V12/45_resolved_view_support/factory/runs/support-v1/quality/catalog, отчётREVIEW_RU.md. Физические размеры null, closed test не открыт. Следующий46: добор наблюдений для неполных multi-timestamp groups. У постера0timestamp-balanced unconfirmed fraction44,66%, но singleton-only очередь его исключает. CPU audit192Factory/40Knaack groups6,45/1,29с; автоматический выбор/дополнительные masks ещё не выполнены. Затем общий bounded raw-camera fallback для шкафа/контейнера.

## 50. Добор неполных многовидовых объектов (2026-09-06)

Автоматический добор для уже многовидовых, но неполных групп включён в общую recovery-цепочку (fe9e3ab). Очереди singleton/coverage имеют отдельные квоты и общий лимит: 8+4 группы, до 20 RGB, 24 tracker crops и 8 completion crops. Приоритет coverage учитывает недостающую поддержку и подтверждённую опору с равным весом физических timestamp. Default coverage=0; ручные ID не используются.

Общие 15-стадийные прогоны завершены: Factory 948,292 с, каталог 131 кандидатов / 117 непустых масок / 67 описаний; Knaack 420,786 с, 42/39/42. Время измерено от готовой geometry, не от сырой сцены. Factory: постер 0 вырос с 4162 до 32672 Gaussian IDs, механизм 40 — с 1702 до 5785; все 117 итоговых memberships точно совпали с ранее просмотренным v2 prototype. Но стена 3 уменьшилась с 8490 до 6376, а whole crane 17 конкурирует с вложенным 40. Поэтому общий прирост ещё не считается принятым качеством сцены.

Knaack: труба 210 получила 114 IDs; 265:572→586, 273:1072→1396. Refinement дополнительно изменил 84:8160→8252 и 89:1510→1523. Все семь изменившихся native views последних двух групп просмотрены: границы всё ещё частичны, значимый выигрыш не заявлен. Трубы 210/202 имеют пересекающийся scope. Постер описан корректно; стена ошибочно названа fire extinguisher из-за контекста. Полные физические размеры остаются null.

1429 tests / 69,96 с. Отчёт V12/46_multiview_coverage/REVIEW_RU.md и combined_audit_v1.json. Следующие исправления: сохранность исходной видимой поверхности, whole/part relations, общий bounded raw-camera fallback для шкафа и контейнера.

## 51. Скорость preview и сохранение аннотаций (2026-09-06)

Код 1747674 ускоряет native preview и сохраняет неизменившиеся аннотации при перенумерации локальных кадров. Preview группируется по камере, не более 8 объектов на renderer call, с прежними пикселями, alpha gate и порядком. На настоящих сценах все 488 JPEG побайтно совпали: Factory 397→63 вызова и 86,277→28,649 с; Knaack 91→24 вызова и 20,548→8,319 с. Это измерение стадии визуализации, не обещанный speedup всей сцены.

Повторный VLM теперь требуется при изменении RGB, маски, кропа, порядка свидетельств, геометрического namespace, модели или prompt. Только смена локального image_id сохраняет label/caption и исторический raw с явным image_id_rebindings; parsed.observed_image_ids переведены в текущие ID.

Штатные semantics→catalog прошли за 30,913 с для Factory и 29,900 с для Knaack. Сохранены 65/67 и 39/42 аннотации; новых запросов 2 и 3 вместо прежних 66 и 12. Все пять новых ответов и десять JPEG совпали с Stage46; банки масок побайтно прежние. Ошибки captions этим не исправляются. 1431 tests / 68,16 с; отчёт V12/47_pipeline_efficiency/REVIEW_RU.md.

Stage48 уже реализовал защиту исходной поверхности (d3674f9; 1449 tests / 67,47 с). Production revalidation сохранила полезные расширения и отвергла два плохих наблюдения Factory3; Knaack accepted masks и 84 массива точечных свидетельств точны. Полная Factory native/refinement/catalog пересборка выполняется; возврат утраченных Gaussian IDs пока не заявлен.

## 52. Защита исходной поверхности проверена в полном пайплайне (2026-09-06)

Stage48 (d3674f9) проверяет сохранность исходного foreground при добавлении наблюдений. Для каждого исходного облака отдельно учитывается только видимый фон дальше 3 px от новой маски. Используются прежние GeometryPolicy: ≥20 видимых samples, ≥15% видимости источника и >35% definite-background для veto. Окклюзия, отсутствующая depth и выход за кадр остаются unknown. Проверка применяется после обычного/partial match, до completion preference; новый foreground вне неполной исходной маски разрешён.

Production revalidation пяти сохранённых наборов точно совпала с прототипом. Для Factory отвергнуты два неполных наблюдения стены 3 и ошибочный составной кандидат 240 из раннего варианта. Полезные дополнения постера, подъёмника, розеточного блока и настенного устройства сохранены. Knaack common: выбор масок и все 84 point-evidence arrays точны, повторная проверка 17,874 с; новый native/inference не нужен.

Полный Factory native→refinement→semantics→catalog занял 394,522 с по host wall time (392,833 с суммарно в восьми стадиях), дополнительно revalidation 28,373 с. Каталог 131/117/67. Стена 3:6376→8490 IDs (+2558/−444), ровно прежний Stage45 состав; все остальные 116 memberships совпали со Stage46. Постер 0 сохранил 32672 IDs, механизм 40 — 5785, устройство 517 — 1731. Относительно Stage45 неизменны 114 из 117 масок; остаются изменения 0/17/40. 392 из 395 общих JPEG со Stage46 точны, все три изменившихся проекции стены просмотрены.

Дополнительно просмотрены OBB и masks на двух ракурсах каждого из 0/17/40/517 (8 строк). Корпус устройства покрыт, провода исключены. Постер почти заполнен, но observed OBB имеет завышенную толщину и fringe; его dimensions 2,616×0,124×1,360 м не являются физическим измерением. Стена остаётся неполной даже после восстановления. У механизма/крана сохраняются пробелы и конкуренция whole/part; существующие alternative banks 17:33610 и 40:10434 IDs пока не приняты как физический scope.

66 аннотаций сохранены, один VLM-запрос для изменившейся 3. Её название вернулось к прежнему ошибочному ceiling panel — правильную семантику это исправление не доказывает. 1449 tests / 67,47 с. Каталог V12/48_source_support_preservation/factory/runs/preservation-v1/quality/catalog; Knaack актуален V12/47_pipeline_efficiency/knaack/runs/retention-v1/quality/catalog. Отчёт и визуализации в 48_source_support_preservation. Physical dimensions null; coverage opt-in, closed test не открыт.

Следующие практические приоритеты: общий добор raw COLMAP-камер для неполных/пропущенных объектов (особенно текущие geometry-группы 634 и 46), качественный whole/part scope и подписи, привязанные к foreground. Текущие ID относятся только к Stage34 geometry namespace, не к legacy FARM. Повторные полные прогоны запускать при изменении входов; проверенное точное сохранение не требует нового inference.
