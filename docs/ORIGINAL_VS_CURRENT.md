# От исходного FARM к текущему production-модулю

Этот документ отделяет возможности исходного исследовательского проекта от
доработок этого репозитория. Он нужен для ревью, переноса и честной оценки:
название стадии не считается доказательством качества — ниже указаны её
входы, выходы и проверяемый эффект.

## Что представлял собой исходный проект

Исходный FARM строил объектно-ориентированную память сцены поверх RGB-D
последовательности: сегментировал наблюдения, связывал их в 3D-объекты,
получал признаки/подписи и сохранял scene graph. Это исследовательская основа,
а не готовый универсальный контур для произвольной пары COLMAP + 3DGS.

В исходном варианте не было единого строгого контракта, который одновременно:

- принимает обычную COLMAP-реконструкцию и Graphdeco PLY;
- доказывает metric scale, gravity и COLMAP↔3DGS alignment;
- исполняет каждую стадию из подписанного snapshot исходников;
- фиксирует Docker image ID, модели и все входные SHA-256;
- отделяет точный dense Gaussian lift от preview cloud;
- выдаёт held-out QA и fail-closed release marker;
- собирает ShapeR/MV-SAM3D объекты в общей метрической системе;
- показывает несколько сцен и независимые слои в одном проверяемом viewer.

## Что добавлено сейчас

### 1. Воспроизводимый production-контур

- 17 стадий от preflight до QA bundle;
- immutable source snapshot с точным file set, mode и SHA-256;
- полный hash входного PLY и content inventory RGB-D/mapping;
- immutable Docker image ID и versioned model manifest;
- CPU/RSS/GPU telemetry и terminal cleanup;
- cold-run и resume/force правила без скрытой подмены исходников.

### 2. COLMAP + 3DGS как явный вход

- PINHOLE/SIMPLE_PINHOLE camera contracts;
- корректный KB4→PINHOLE half-pixel conversion;
- связный selector физических timestamp;
- native-resolution RGB-D render из 3DGS;
- scale/gravity/alignment preflight и render QA;
- исходный PLY никогда не перезаписывается.

### 3. Более строгая объектная геометрия

- mask/depth consensus и gravity-aligned OBB;
- visual consistency и surface-support gates;
- part/whole, assembly, compound geometry и exhaustive final overlap audit;
- неподдержанные огромные OBB не публикуются только потому, что track существует;
- при неопределённости исходный объект сохраняется, а сомнительное refinement
  не заменяет его.

### 4. Open-vocabulary семантика

- mask-grounded реальные crops вместо неподтверждённого scene context;
- VLM сначала классифицирует topology: whole/carrier/component/mixed;
- functional label требует видимых диагностических деталей или текста;
- независимый verification-fold не может сам повысить confidence до confirmed;
- hard label errors и high uncertainty остаются в QA.

### 5. REST3D-inspired VLM + SAM3 refinement

Из REST3D перенесена полезная логика, а не его scene-specific список классов:

1. Неуверенные объекты ищут до шести дополнительных полностью видимых ракурсов
   по всей зарегистрированной PINHOLE COLMAP-модели.
2. Независимые 2D seeds backproject-ятся через metric render-depth, обрезаются
   OBB и образуют multi-view voxel consensus.
3. Metric support reproject-ится как positive prompts; competing masks и область
   вне OBB дают negatives.
4. Лучший независимо принятый seed распространяется SAM3 вперёд и назад. Оба
   направления запускаются из отдельных одинаково seeded sessions, как в
   проверенном REST3D workflow.
5. Propagated mask не принимается по tracker confidence: она заново проходит
   depth, OBB, connected-component, seed-coverage и multi-view metric gates.
6. Geometry refit и слепой VLM решают существование целого объекта; отдельный
   candidate-conditioned audit проверяет label, но не стирает доказанную
   геометрию. Сомнительный label публикуется как `unresolved object`.
7. Scale `meters_per_scene_unit`, depth units и pose units сверяются plan↔RGB-D
   до загрузки SAM3; перепутать rig baseline 0.0295 м с scene scale больше нельзя.

Budget увеличен до 48 объектов, шести видов и 288 кадров. Реальный Factory A/B
назначил всем 48 кандидатам по шесть видов (251 уникальный RGB): 38 mask-pass,
27 geometry-pass, 23 whole-object acceptance; 8 labels подтверждены двумя
проходами, 15 объектов сохранены как `geometry_only`. SAM3 stage занял 61.26 с,
а весь A/B после готового VLM service — около 6.8 мин. Это измеренный
промежуточный A/B; production-статус появится только после нового signed cold run.

### 6. Точный Gaussian lift

- exact contributor VJP по исходному порядку всех Gaussian rows;
- build/held-out split по физическим timestamp;
- conflict→unknown, без runner-up promotion;
- dense arrays, verified CSR и полный field-preserving labelled PLY;
- connected refinement растёт только по build evidence и снова проходит
  held-out QC;
- release и calibration/nonrelease markers разделены.

### 7. Реконструкция объектов и viewer

- ShapeR bridge с pinned offline runtime и metric assembly;
- MV-SAM3D использует реальные PINHOLE RGB и nearest-neighbour masks;
- gravity/full-silhouette orientation QA вместо слепого PCA↔OBB axis matching;
- общий world-metric NPZ/PLY и glTF +Y-up GLB;
- один read-only viewer для Factory/Knaack, реальные object crops по клику,
  понятные основные слои и технические сведения внизу.

## Что получено и что пока не гарантируется

Получено:

- существенно более высокая precision опубликованных объектов и OBB;
- проверяемая воспроизводимость от входного файла до viewer layer;
- точное соответствие `gaussian_index → instance_id` для verified lift;
- переносимый workflow для новой сцены и другого сервера;
- отдельные QA-границы для семантики, lift и генеративной реконструкции.

Не гарантируется:

- полный recall закрытых, очень мелких или никогда не увиденных объектов;
- истинная невидимая поверхность ShapeR/MV-SAM3D;
- photorealistic native 3DGS в Viser (это quantized DC preview);
- автоматическая production-приёмка новой сцены без visual review сложных
  объектов и frozen blind thresholds.

## Куда смотреть дальше

1. Установка и перенос: [DEPLOYMENT.md](DEPLOYMENT.md).
2. Подготовка новой сцены: [INPUTS.md](INPUTS.md).
3. Acceptance и интерпретация качества: [QUALITY.md](QUALITY.md).
4. CLI, стадии и артефакты: [REFERENCE.md](REFERENCE.md).

