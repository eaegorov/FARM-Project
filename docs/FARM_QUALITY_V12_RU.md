# FARM V12: воспроизводимость и независимая оценка качества

Документ содержит инфраструктуру F0/F1 и обновление реализации FARM от 2026-09-05.
Текущий отчёт и изображения: `output/farm_pipeline/factory/experiments/factory-universal-v12-quality-v1/11_scope_refinement/REVIEW_RU.md` относительно `3dgs_work`.
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
