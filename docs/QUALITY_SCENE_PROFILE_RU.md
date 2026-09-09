# Общий профиль обработки 3DGS + COLMAP

Профиль собирает подготовку наблюдений, поиск объектов, согласование ракурсов,
перенос масок в исходные Gaussian и описания в стандартный DAG. Его выполняет
обычный orchestrator: снимок исходников, журналы, статусы и время каждого этапа
сохраняются в каталоге запуска.

Это **профиль разработки**. Проверка 9 сентября 2026 на Factory, Knaack и
Industrial подтвердила полезное исправление конкуренции с окружением, но
выявила пропуски предметов, дубликаты и неправильные названия. Готовность
автоматического каталога для произвольной новой сцены не подтверждена.

## Текущая рабочая конфигурация

Для дальнейших сравнений используется явный профиль **YOLOE-26X + Qwen3-VL-4B,
24 начальных + до 24 дополнительных ракурсов**. Это рекомендация для разработки,
а не смена всех CLI defaults или разрешение заменить проверенный каталог viewer.

| Шаг | Назначение |
|---|---|
| Зарегистрированные RGBD | Фотографии, камеры и глубина согласованы с исходным PLY |
| Словарь | Qwen по 8 контекстным кадрам дополняет общий словарь поиска |
| Сегментация | YOLOE-26X предлагает 2D-маски и гипотезы категорий |
| Согласование | Геометрия и глубина связывают наблюдения; считаются физические моменты съёмки |
| Дополнительные камеры | Подтверждение кандидатов, затем исследование новых позиций и направлений |
| Native lifting | Исходные Gaussian распределяются между предметными группами с учётом видимости |
| Описания | Qwen рассматривает выделенную область в двух реальных видах |
| Каталог | Маски, наблюдаемые OBB, описания, альтернативы состава и происхождение данных |

MobileCLIP2 кодирует текстовые категории для YOLOE-26L/X. Этот адаптер не передаёт
современные визуальные embeddings в старый feature-based association backend:
в текущем профиле ассоциация геометрическая. Финальные label/caption предлагает
Qwen по конкретным областям, а не непосредственно детектор. Ни одна из этих
моделей сама по себе не удостоверяет правильность названия.

### Команда для новой сцены

Пример для установленного control plane и доступных inference runtimes:

    python3 -m farm_runtime.cli quality scene-profile plan \
      --scene-id my_scene \
      --project-root /absolute/FARM-Project \
      --output-root /absolute/quality_outputs \
      --rgbd /absolute/registered_rgbd \
      --ply /absolute/scene.ply \
      --runtimes /absolute/runtime_prefixes.json \
      --detector yoloe \
      --yoloe-model-root /absolute/models \
      --yoloe-checkpoint /absolute/models/yoloe/yoloe-26x-seg.pt \
      --vlm-model /absolute/qwen3_vl_4b_checkpoint \
      --detector-confidence 0.25 \
      --depth-consistent-association --explore-uncovered \
      --initial-views 24 --adaptive-views 24 --vocabulary-views 8 \
      --native-groups 128 --semantic-groups 128 --alternatives 16 \
      --recovery-groups 0 --coverage-groups 0 --refinement-crops 0 \
      --world-up 0 -1 0 \
      --output /absolute/quality_pipeline.json

**Ось world-up нужно получить из данных конкретной сцены.** Значение в примере
подходит проверенным входам, но не является универсальным. Путь RGBD должен
содержать полный доступный зарегистрированный пул: профиль сам выбирает кадры.

В runtime JSON задаются списки аргументов запуска Python: main, geometry и,
для современного YOLOE, detector. Последний изолирует актуальный Ultralytics
от окружений Qwen и Gaussian renderer. Требуется локальный
yoloe/mobileclip2_b.ts. Пути входов, выходов и снимка исходников должны быть
доступны в соответствующих контейнерах. Подробности установки —
[DEPLOYMENT.md](DEPLOYMENT.md), контракты адаптера — [QUALITY.md](QUALITY.md).

    farm validate-plan --config /absolute/quality_pipeline.json
    farm run --config /absolute/quality_pipeline.json --run-id quality-v1
    farm status --run /absolute/quality_outputs/my_scene/runs/quality-v1

Для показанного YOLOE-26X требуется явный runtime с ключом detector, даже если
все зависимости установлены вместе. Режим без runtimes с Python control plane
доступен для SAM3/legacy при наличии их зависимостей. Явный checkpoint обязателен
для воспроизведения YOLOE-26X: без него адаптер использует legacy YOLOE-v8L.
Исторический SAM3 backend остаётся отдельной возможностью CLI.

## Отделение непрерывного окружения

Собранный план по умолчанию включает separate-scene-surfaces на native этапе
и передаёт его selection в каталог. Непрерывные поверхности не конкурируют
с предметами за одни и те же Gaussian. Правило допускает только согласованные
точные категории ceiling, floor, wall, ground, roof во всех наблюдениях,
минимум два физических timestamp и непротиворечивый состав labels.
Смесь предметных и структурных категорий остаётся среди объектов; смесь только структурных категорий может выводиться в окружение. Группы, принятые отдельной
проверкой recovery, не удаляются этим правилом.

В quality scene-profile native эта политика включается явно флагом
--separate-scene-surfaces. У quality native-observations такого флага нет. В plan флаг --include-scene-surfaces отключает её.
Каталог хранит scene_context и происхождение решения. Отдельный native-банк
для окружения не создаётся; physical_role_validated остаётся false.
Поэтому ошибочно названный пол может остаться предметом, а согласованная
категория не равнозначна доказанной физической роли.

Контроль commit 4edbd46: 101 исходный кандидат, три потолочные гипотезы вынесены
из предметной конкуренции. Из 98 оставшихся улучшились две маски; 92 непустых
и четыре пустых сохранили точно тот же состав Gaussian. Это проверка регрессий
на данных разработки, не измерение полноты сцены.

## Время и пределы качества

Свежие 13 этапов с профилем 24 + 24 на одной NVIDIA L4:

| Сцена | Пул RGBD | Обработано кадров | Время от RGBD | Кандидаты / непустые маски |
|---|---:|---:|---:|---:|
| Factory | 48 | 48 | 8:31 | 80 / 76 |
| Knaack | 48 | 48 | 4:42 | 30 / 30 |
| Industrial | 88 | 48 | 7:04 | 61 / 58 |

Модели и CUDA-расширения уже установлены; загрузка моделей и новый inference
включены. Подготовка этих пулов из существующих pinhole COLMAP, RGB и PLY
измерена отдельно: 0:44, 0:45 и 1:01. Извлечение кадров, fisheye conversion,
обучение 3DGS и создание обзорных галерей в эти времена не входят.

Число кандидатов не измеряет recall. Быстрый профиль 12 + 8 потерял десять
прежних целей Knaack; 24 + 24 вернул их, но разделил конус на две группы.
В Industrial даже больший бюджет потерял оба подтверждающих направления
двух контейнеров и светильника. Увеличение бюджета сейчас меняет исходную
выборку, а не гарантирует её расширение. Дубликаты и целое/часть остаются
открытыми проблемами до exclusive распределения точек.

Артефакты контроля в рабочем дереве данных:
output/farm_pipeline/research/quality_2026-09-09/pipeline_validation/bounded_cycle.
Начальная точка — RESULT_RU.md; полный фиксированный набор — OBJECT_REVIEW_RU.md;
воспроизводимые конфигурации и метрики — coverage48-4edbd46-v1.

## Что означает каталог

В quality/catalog/catalog.json связаны:

- object_masks.npz: принадлежность исходных строк PLY основным группам;
- отдельные scope_alternatives: пересекающиеся гипотезы состава объекта;
- observed OBB, чувствительность к удалённым точкам и независимые timestamps;
- label/caption со статусом unverified_model_proposal и исходными изображениями;
- source PLY, namespace геометрии, причины неопределённости и отложенные группы.

physical_dimensions_m = null не скрывает наблюдаемую геометрию: она доступна
в observed_obb.dimensions_m. Но размер наблюдаемой оболочки, скрытая толщина
и реальные физические размеры — разные утверждения. Масштаб исходной сцены
требует подтверждения. Альтернативы не объединяются молча в основной банк.

Целый шкаф, тележка с грузом или составное оборудование допустимы как одна
единица. Принудительно делить предмет на все видимые детали не требуется.
Автоматическое согласование целого и частей пока не считается решённым.

## Дополнительные команды разработки

Ниже сохранены справочные контракты предыдущих экспериментов. Они **не входят
в рекомендуемый профиль 24 + 24**, пока отдельное сравнение не подтвердит
пользу на всём фиксированном наборе. Исторические времена и ID иллюстрируют
конкретные опыты и не определяют правила обработки новых сцен.

`--refinement-crops 12` включает пять дополнительных этапов того же DAG:
scheduler, SAM tracker, SAM concepts, other-timestamp selection и conditional
native rebuild. Допустимый бюджет0..32, максимум2 crops на объект. Default0
сохраняет профиль из13 stages; opt-in12 даёт18. Кандидатам нужны минимум3
независимых timestamps, чтобы проверять crop по двум OTHER timestamps.
Противоречивый ракурс может стать unknown; unsupported expansion отклоняется.
Пустой crop schedule не загружает SAM/CUDA, отсутствие принятых изменений
сохраняет прежний bank/input без повторного native build.

Appearance всегда получает effective native input. Удалённые object/view
наблюдения не участвуют, принятые replacements используют изменённые masks,
дополнительные зарегистрированные views разрешены. Неизменённые исходные masks
сохраняют прежние pixels; выбор между сетками разных разрешений сравнивает
foreground fraction. Связь идёт по имени и SHA256 RGB, а не по local image ID.
Каталог отклоняет stale appearance после mask refinement и отдельно хранит
`native_independent_timestamps`, включая дополнительные/исключённые наблюдения.

Автоматическое physical scope merging пока не включено. Новый каталог не
заменяет проверенный production bank и не означает завершённый universal
quality release.


Проверенные18-stage прогоны из snapshot4784c3d: Knaack10:39, Factory17:25 после готового registered RGBD. Это wall time данного development run, без model download/raw ingress; budgets ограничивают полноту. Подробный аудит — V12/21_refined_profile/REVIEW_RU.md.

Экспериментальный `quality proposal-geometry --partial-view-association` предназначен для отдельного geometry ablation. Общий scene profile пока не включает его: положительный poster followup проверен, но whole/part ambiguity и broader quality gates остаются. См. V12/23_partial_view_association/REVIEW_RU.md.

Дополнительные экспериментальные stages:

- `quality scene-profile union --primary SAM_MANIFEST --supplement OTHER_MANIFEST --max-primary-iou 0.5 --output NEW_DIR` объединяет предложения на одинаковых зарегистрированных RGB. Первичный источник сохраняет приоритет и person exclusions; logits разных моделей не калибруются друг против друга. Без threshold сохраняются все proposals, что в Factory ухудшало association.
- `quality surface-tracker --audit SURFACE_AUDIT --proposals EXTRA_SAM_MANIFEST --validation FAILED_VALIDATION --model SAM_DIR --crop-budget 12 --output NEW_DIR` предлагает masks по проекции видимой поверхности в локальные RGB. `--validation` необязателен; с ним обрабатываются только unresolved matches. Выход передаётся как `--supplement` в `quality surface-validation`, затем принятые observations — в `native-observations`.

Эти stages ещё не включены автоматически в18-stage scene profile. Они решают пропуски observations, но не гарантируют полноту физического объекта. Factory unit восстановлен в3timestamps, однако его основание частично отсутствует; fire cabinet depth support недостаточен для целой маски. Не принимать build-mask IoU или model score за whole-object acceptance. Подробности и runtime — V12/24_complementary_discovery/REVIEW_RU.md.

Для дополнительных observations доступен opt-in `quality surface-validation --partial-view-association`: применяет те же guarded FOV правила, что proposal geometry, до признания source/candidate correspondence отсутствующим. Используйте его перед назначением tracker retries для edge-clipped объектов. Factory fire cabinet подтверждён без новых inference calls;3D completeness всё ещё требует дополнительных независимых камер. См. V12/25_planar_identity/REVIEW_RU.md.


Адресный recovery за пределами текущего RGBD pool (V12/26_camera_completion):

1. `quality camera-completion --audit AUDIT --split-packet PACKET --group-id ID --timestamp-budget 2 --output NEW_DIR` выбирает новые камеры по metadata. Исходный COLMAP, metric scale и filename identity берутся из preparation provenance. `PACKET.split_by_timestamp` разрешает только train/dev; команда не декодирует RGB. Подайте `preparation_names.txt` в существующий RGBD ingress. Frustum coverage ещё не visibility.
2. После rig registration/depth используйте `surface-evidence` для исключения окклюдированных кандидатов, затем SAM и при необходимости `surface-tracker`/`surface-validation`.
3. При `ambiguous_competing_scope`: `quality scope-review --validation VALIDATION --model QWEN_DIR --object-budget 4 --output NEW_DIR`. Максимум2VLM calls на объект/ракурс и2..4 eligible masks; пустая очередь без model load. Нужен независимый reference без FOV clipping. В `surface-validation` добавьте `--scope-review NEW_DIR/manifest.json`, сохранив те же audit/proposals/supplements/policy. Только high consensus в двух порядках, совпавший с geometric best, может снять ambiguity. Прочие решения остаются unknown.
4. Принятые observations передаются в `native-observations`; проверяйте reverse renders и extent stability. Повторное использование устаревшего review отклоняется.

Это отдельные bounded development commands. Автоматический global recovery budget/reuse в scene-profile пока не реализован. Положительный результат для fire cabinet не означает доказанный перенос на все сцены или подтверждение физических размеров.


Для следующей итерации recovery: `quality scene-profile cohort --validation VALIDATION --group-id ID --output NEW_DIR`. Выходы `manifest.json` и `transients.json` можно передать в `proposal-geometry` на том же зарегистрированном RGBD, затем снова в `surface-evidence`. Исходные и дополнительные accepted masks/person exclusions сохраняются без inference и без изменения logits. Новое group-ID namespace нужно читать из geometry результата: не переносите старый integer ID без проверки provenance. Требуются хотя бы2timestamps, accepted geometry selection и неизменённые source artifacts. Unit replay проверен на полном совпадении1252Gaussian IDs. VLM completeness/bbox pilot Stage27 пока не встроен в scene-profile.


При включённом refinement можно добавить `scene-profile plan --balance-refinement-errors`: второй crop одного объекта, если он доступен в том же бюджете, выбирается по другому типу ошибки (foreground deficit/background contamination). Standalone эквивалент: `refinement-schedule --balance-error-modes`. Default и18-stage структура сохранены. Опция протестирована на переносе полной маски unit в старый ракурс; это не замена gates и не увеличение crop budget. Knaack replay меняет1из12crop requests; Stage29 подтвердил abstention из-за разного scope конкурирующих масок. Native bank сохранён.

Для проверки **видимых пропущенных интегральных частей** в уже принятой tracker-маске:

```bash
python -m farm_runtime.cli quality scope-completion \
  --tracker /absolute/surface_tracker/manifest.json \
  --validation /absolute/surface_validation/manifest.json \
  --model /absolute/qwen3_vl_4b_checkpoint \
  --sam-model /absolute/sam3_checkpoint \
  --crop-budget 4 --crops-per-object 2 \
  --output /absolute/new_scope_proposals
```

Validation должна связывать тот же audit/base proposals и содержать tracker в supplements. Проверяются принятый detection, его принадлежность объекту и crop coordinates; ambiguous identity пропускается. Бюджет1..16crops/1..3distinct timestamps на объект, очередь разделяется между объектами. VLM видит RGB и бинарную схему; её bbox не может исключить bbox уже выбранной маски. Она может ошибаться в точках, поэтому выход — дополнительные proposals для `surface-validation --supplement`, затем accepted native lift. Если нужен fallback на прежние tracker masks, передавайте их отдельным supplement наряду с новыми proposals. Старые thresholds сохраняются.

Модели загружаются один раз на этап. Empty/unresolved очередь обходится без весов; при complete/uncertain ответе SAM не запускается. V12/30_scope_completion:unit2RGB17,81с, все12logits точны кпрототипу; complete cabinet8,08с/0SAM calls. Это отдельный opt-in этап; глобальный recovery budget в scene-profile ещё не реализован.

Для повторной обработки можно передать `scene-profile plan --refinement-reuse /absolute/old_proposals/manifest.json` (флаг повторяемый). Он попадает в оба SAM refinement stages, каждый использует только совместимый backend/model/preprocessing contract. Standalone: `quality refinement sam --reuse-proposals MANIFEST`. Crop/seed/RGB/grid/orientation/queries обязаны совпасть; старый native selection не переносится. Текущие OTHER-timestamp votes пересчитываются штатно. Старые manifests без fingerprint обрабатываются обычным inference. V12/31_refinement_reuse:14Knaack candidates точны, повтор двухэтапного CPU запроса13,71→5,33с после устранения eager captioning imports; это не scene-wide speedup.

## Частичная видимость и дополнительные предложения

`--partial-view-association` включает co-visible surface policy в initial и final geometry. Согласованные наблюдения, обрезанные границей кадра, могут объединяться при сохранении проверок вложенности и отрицательных свидетельств. Флаг может изменить adaptive sampling и пока включается явно.

`--complementary-model-root /absolute/models` добавляет два этапа после combined SAM proposals: `complementary-discovery` на тех же RGB и `scene-profile union --max-primary-iou 0.5`. С refinement получается 20 этапов, без него — 15. Root содержит YOLOE/MobileCLIP и соответствует runtime model root. `--complementary-vocabulary` задаёт словарь YOLOE; по умолчанию используется `configs/yoloe_vocabulary.txt`.

Дополнительные маски с IoU ≥ 0,5 к основной маске исключаются. Сохраняются приоритет основной геометрической ассоциации и person exclusions от SAM. Оценки уверенности разных детекторов не смешиваются. Максимум — 48 уже выбранных RGB; новые снимки этап не открывает. Подтверждение группы в нескольких ракурсах ещё не доказывает полноту объекта или правильность его названия.

На фиксированных снимках Factory результат точно совпал с прежним pilot; Knaack сохранил основные группы и подтвердил ещё одну. Её проверка выявила неполноту тонких элементов и недоказанный материал в VLM-описании. Время полного профиля и полнота сцены требуют следующего контроля.

Проверка содержимого зависимостей ограничена тремя checkpoint-файлами: YOLOE prompt-free, YOLOE vocabulary-head и MobileCLIP BLT. Весь model root не хешируется. Runtime отклоняет подмену этих путей через внешние MobileCLIP overrides, чтобы фактически использованные веса всегда входили в fingerprint этапа.


## Передача восстановленных объектов в общий native-вход

`quality scene-profile native --recovery-validation VALIDATION` сохраняет исходный выбор `--groups` и добавляет подтверждённые объекты из recovery cohort (не более 16 групп сверх основного бюджета). Остальные аргументы native этапа прежние. В `native-observations --geometry GEOMETRY` доступен тот же флаг; его нельзя сочетать с `--validation` или `--input`.

Validation должна ссылаться на тот же SHA геометрии. Добавляются только выбранные геометрически допустимые маски с проверенными RGB, регистрацией и person query. Объект должен иметь минимум два различных физических timestamp; два сенсора одного момента не повышают его статус. Исходные ID, маски остальных выбранных объектов и их опорные OBB сохраняются. После объединения выполняется один общий native lift с разрешением конфликтов, а не склейка отдельных CSR банков.

Semantic selection учитывает фактически оставшиеся timestamp после recovery и quarantine. Объекты с сохранёнными validated additional observations получают отдельный бюджет описаний до 16 групп сверх основного выбора, поэтому низкое исходное число ракурсов не вытесняет результат recovery. Потерявший второй timestamp объект в описания не проходит. В каталоге сохраняются оба счётчика — исходный `independent_timestamps` и фактический `native_independent_timestamps`; `recovered_from_single_timestamp` отмечает переход. Геометрическая поддержка не удостоверяет label, физические размеры или полноту маски.

Это интеграция уже проверенных recovery observations. Автоматический выбор объектов для recovery и его включение в `scene-profile plan` пока остаются следующим этапом.


## Приоритет проверяемого дополнения объекта

При передаче `surface-validation --scope-completion COMPLETION` дополнительные маски из `scope-completion` проверяются раньше прежней tracker-маски. Передайте в обычных `--supplement` ровно тот набор и порядок, на котором основана родительская validation; completion manifest добавляется автоматически. Audit, source proposals, partial-view policy, принятый tracker detection и source artifacts должны совпадать.

Приоритет возникает только после `scope_complete=false` и `confidence=high`. Кандидат обязан пройти прежние geometric identity gates и покрывать не менее 90% прежней маски; pixel union не выполняется. Выбирается однозначный geometric best среди таких completion candidates. Если дополнение не проходит проверку или неоднозначно, сохраняется прежний detection только при его текущей geometric eligibility. Иначе результат остаётся unknown. В выходе есть `completion_selection` и provenance выбранной политики.

Режим включается явно и не совмещается со `--scope-review`. Он разделяет подтверждение идентичности и выбор более полного видимого scope; VLM не получает права отменять геометрию. В Stage37 устройство в общем Factory bank выросло с1252 до1731IDs с сохранением всех прежних точек; остальные115 масок не изменились. Видимая противоположная боковина восстановлена; полнота скрытой поверхности неизвестна. Физические размеры не валидированы; общий recovery DAG ещё предстоит.

## Сохранение описаний при неизменных входных данных

В standalone scene-profile semantics доступен --retain-stage /absolute/previous_semantics/manifest.json. Evidence формируется заново из текущего native input. Прежние object records сохраняются только при совпадении двух ordered views: source/crop/mask hashes, timestamp, local image IDs, crop coordinates, orientation и остальных полей. Geometry namespace, обе native bindings и артефакты проверяются. Изменённые/новые объекты проходят обычный compact VLM; пустой список не загружает модель.

Поддерживается стандартный compact profile с теми же prompt и model config. Это сохранение существующих аннотаций, не model cache для сравнения весов или версий. annotation_retention хранит происхождение, retained/refreshed IDs и число новых requests. Top-level timing относится к обновлению, timings/tokens/sheets сохранённых records — к прежнему запуску. Качество ошибочной подписи от сохранения не повышается.

Stage37:65 из66 Factory описаний сохранены точно, устройство с другим вторым ракурсом пересчитано за15,57с вместе с моделью. Каталог содержит1731Gaussian IDs устройства. Флаг не меняет обычный полный запуск и пока не добавлен в generated plan.
