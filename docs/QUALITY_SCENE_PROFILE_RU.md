# Ограниченный quality-профиль FARM

`quality scene-profile plan` собирает существующие проверенные этапы в обычный
FARM DAG. Его выполняет `farm run`: snapshot исходников, отдельные журналы,
статусы этапов и время остаются в стандартном orchestrator.

Это development-профиль для **уже подготовленного зарегистрированного RGBD** из
3DGS+COLMAP. Исходный FARM ingress отвечает за масштаб, gravity и RGB↔GS
регистрацию. Профиль не объявляет unregistered depth метрическим наблюдением.
Обработка исходных PLY/COLMAP/RGB и время ingress пока считаются отдельно.

```mermaid
flowchart LR
    A[Registered metric RGBD] --> B[8 context RGB: scene vocabulary]
    B --> C[Core + scene queries, максимум 80]
    C --> D[12 RGB: SAM3 proposals]
    D --> E[Geometric association]
    E --> F[До 8 дополнительных RGB по видимости]
    F --> G[Новые proposals + прежние masks]
    G --> H[Geometric association]
    H --> I[Native lift и nested alternatives]
    I --> R[Опционально: до 12 crops по противоречиям]
    R --> S[SAM tracker + concepts, проверка по OTHER timestamps]
    S --> T[Пересчёт native только при принятых изменениях]
    T --> J[2 актуальных contextual crops: compact appearance]
    T --> K[Общий каталог и source-order masks]
    J --> K
```

## Запуск

В inference-контейнере с FARM source:

```bash
python -m farm_runtime.cli quality scene-profile plan \
  --scene-id my_scene \
  --rgbd /absolute/registered_rgbd \
  --ply /absolute/scene.ply \
  --sam-model /absolute/sam3_checkpoint \
  --vlm-model /absolute/qwen3_vl_4b_checkpoint \
  --world-up 0 -1 0 \
  --refinement-crops 12 \
  --project-root /absolute/FARM-Project \
  --output-root /absolute/quality_outputs \
  --runtimes /absolute/runtime_prefixes.json \
  --output /absolute/quality_pipeline.json
```

`world-up` должен быть получен из конкретной сцены; пример оси не универсален.
`runtime_prefixes.json` содержит ровно `main` и `geometry`: каждый — список
аргументов команды, запускающей Python. Для раздельных Docker-образов это
`docker run ... --entrypoint /path/to/python IMAGE`, для единого подготовленного
контейнера — путь Python. Shell interpolation не используется. Пути входов,
outputs и `${execution_project_root}` должны быть доступны в обоих runtime;
последний — snapshot, созданный существующим orchestrator. Runtime pins и
mounts задаются конфигурацией установки, а не названиями Factory/Knaack.

На документированном FARM control plane:

```bash
farm validate-plan --config /absolute/quality_pipeline.json
farm run --config /absolute/quality_pipeline.json --run-id quality-v1
farm status --run /absolute/quality_outputs/my_scene/runs/quality-v1
```

GPU inference выполняется в контейнерах. Host control plane и его подготовка
описаны в [DEPLOYMENT.md](DEPLOYMENT.md). Если `--runtimes` пропущен, команды
используют Python текущего orchestrator; это пригодно только для контейнера,
в котором доступны обе группы зависимостей.

## Бюджеты и смысл результата

По умолчанию 12 первоначальных + до 8 дополнительных сегментируемых RGB,
8 context RGB для словаря, до 128 native candidates, до 64 appearance
candidates × 2 timestamps, до 16 nested alternatives. При превышении
object-бюджета порядок задаётся числом независимых timestamps и стабильным ID;
необработанные IDs явно перечислены. Это ограничивает работу, но не доказывает
полноту сцены. Если новых полезных видов нет, SAM3 повторно не загружается.
Отсутствие multi-timestamp groups останавливает native stage с явной причиной.

Native profile использует `configs/quality/native_rendered_v1.json`: валидная
rendered contribution, отрицательное свидетельство конкретного кандидата,
сохранение transient/depth unknown. Выполняется только `exclusions_on`;
`native-observations --mode both` сохраняет прежний режим ablation.
Калиброванного release этот конфиг не заявляет.

Выход `quality/catalog/catalog.json` связывает:

- `object_masks.npz`: exclusive primary membership исходных строк PLY;
- `scope_alternatives/*.npz`: отдельные пересекающиеся варианты physical scope;
- observed OBB и tail-sensitivity, независимые timestamps;
- `label`/`caption` как **unverified model proposals**, с исходными canvases;
- SHA256 namespace геометрии, исходный PLY, его масштаб, причины неопределённости
  и отложенные IDs.

`physical_dimensions_m=null` не скрывает оценку observed geometry: она доступна
в `observed_obb.dimensions_m`. Это разные утверждения. Глубина за прозрачным
окном и скрытая толщина панели не становятся физическим измерением объекта.
Маски альтернатив не объединяются молча в exclusive bank.

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
