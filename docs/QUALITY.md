# Качество, acceptance и смысл слоёв

Этот файл отвечает на два вопроса: «что действительно стало лучше?» и «можно
ли отдавать результат?». Красивый слой сам по себе не является доказательством.

## Что означает каждый слой

| Слой | Что доказано | Что не доказано |
| --- | --- | --- |
| FARM preview | финальный объектный state и sampled context | точная поверхность и полный recall |
| OBB | метрическая геометрическая гипотеза после gates | совпадение коробки с каждым пикселем объекта |
| Dense lift | исходные 3DGS-строки, прошедшие held-out mask QC | что `-1` — фон или что маска полная |
| ShapeR | сгенерированный mesh, прошедший QA | измеренная топология/текстура |
| MV-SAM3D | реконструкция из реальных PINHOLE RGB+mask | истинная невидимая поверхность |

Слои независимы. Viewer не имеет права показывать calibration/nonrelease как
canonical release.

## Почему старый Knaack выглядел плохо

Старый результат оставлял 323 geometry/visual-valid tracks: стены, фрагменты,
дубли и огромные неподдержанные OBB. Чистый lift скрывал проблему, потому что
показывал только небольшой high-precision subset.

Новый surface gate выполняется до semantic captioning и публикации. Из 323
кандидатов Knaack 227 демотированы; осталось 96 metric и 52 presentation.
Factory: 85 metric и 36 presentation.

| Сцена/run | До surface gate | Final metric | Presentation | Release lift |
| --- | ---: | ---: | ---: | --- |
| Factory `factory-production-v4-quality` | 90 | 85 | 36 | 15/21 strict, 188 051, PASS |
| Knaack `knaack-production-v8-quality` | 323 | 96 | 52 | 31/36 strict, 163 271, PASS |

Precision вырос измеримо. Полный recall не гарантирован: маленький, закрытый
или не наблюдавшийся объект может отсутствовать.

## Как работает lift

Lift использует exact contributor VJP pinned `gsplat` при native resolution,
а не nearest-neighbour от preview cloud.

1. Глобальный split физических timestamp на build/held-out.
2. Положительная evidence внутри mask, negative/boundary/depth gates.
3. Конфликты разных объектов остаются unknown.
4. Provisional membership замораживается.
5. Held-out masks открываются только для reverse-render QC.
6. Fail возвращает строки в `-1`; runner-up не повышается.

Connected refinement повышает полноту консервативно:

- seed только из multi-timestamp strong core;
- растут только unknown-строки с положительной build evidence;
- рост обязан быть пространственно связным и ограниченным;
- cross-object conflict остаётся unknown;
- held-out masks не участвуют в росте;
- затем повторяются geometry и held-out gates.

Для новой сцены thresholds не подстраиваются по объектам. Сначала calibration
на одной сцене, commit конфигурации, затем blind validation на другой.


## REST3D-inspired уточнение 2D-масок

Для сложных объектов основной subset кадров недостаточен: маска может покрывать
только видимую грань, а label — случайный компонент. Поэтому для ограниченной
очереди неопределённых объектов используется отдельный fail-closed проход:

1. Планировщик ищет до шести хорошо видимых ракурсов по всей зарегистрированной
   PINHOLE COLMAP-модели; исходные RGB не растягиваются и не размываются.
2. Независимые seed-маски backproject-ятся через render-depth в метрический мир,
   обрезаются расширенным OBB и объединяются только при поддержке минимум двух
   ракурсов.
3. Согласованные 3D-точки reproject-ятся с depth-consistency и становятся
   positive prompts для SAM3; competing masks дают negative prompts.
4. Лучший независимо принятый seed распространяется вперёд и назад по кадрам.
   Как в локальном REST3D, направления используют отдельные одинаково seeded
   sessions: forward-cache не может повлиять на reverse.
5. Каждая propagated-маска заново проходит confidence, component, rendered-depth
   и metric-OBB gates. Одного tracker output недостаточно; нужен хотя бы один
   независимо принятый seed и общая 3D-поддержка минимум двух видов.
6. Повторный geometry refit проверяет метрический центр, размеры, reprojection,
   box support и voxel support.
7. Слепой VLM решает, является ли маска целым физическим объектом. Отдельная
   candidate-conditioned проверка решает только, можно ли публиковать label.
   Неуверенный label становится `geometry_only / unresolved object`, но уже
   доказанная геометрия не удаляется. Неуверенность в существовании по-прежнему
   сохраняет исходный объект без замены.

Production budget: до 48 объектов, шесть видов на объект и 288 уникальных кадров
с round-robin распределением. На Factory план дал всем 48 объектам по шесть
ракурсов (251 уникальный RGB); planning занял 52.77 с, RGB-D — 60.82 с.

Измеренный A/B 2026-08-24:

- предыдущий 192-view budget: 27 mask-pass → 20 geometry-pass → 17
  whole-object acceptance (4 label-verified, 13 `geometry_only`);
- текущий 6-view/288 budget: всем 48 кандидатам назначено по шесть ракурсов,
  251 уникальный реальный RGB;
- mapping: 251/251, dropped 0, 124.88 с;
- SAM3 + независимые directional propagation sessions: 38 mask-pass,
  123 refined mask sidecars, 61.26 с (13.40 с model inference);
- strict metric geometry: 27 pass / 11 reject;
- VLM: 27 initial + 27 verification запросов, 0 transport errors, 102.93 с;
- итоговый existence gate: 23 объекта;
- из них 8 label-verified и 15 честно `geometry_only`.

Полный измеренный A/B-контур после готового VLM service занимает около 6.8 мин:
52.77 с planning + 60.82 с RGB-D + 124.88 с mapping + 61.26 с SAM3 +
102.93 с VLM и короткие merge/geometry/acceptance. Холодный старт VLM учитывается
отдельно и зависит от GPU/runtime cache.

Это не cold-run acceptance и не повод подменять текущий Factory release: новый
механизм должен пройти полный signed cold run. Но A/B доказывает три исправленные
ошибки: scale теперь связан между plan/render/OBB fail-closed, а неуверенность
candidate-conditioned label-аудита больше не уничтожает достоверный 3D-объект,
а расширение до шести full-COLMAP видов повышает mask-pass 27→38 и итоговый
whole-object acceptance 17→23 без ослабления geometry/VLM gates.

## Knaack MV-SAM3D

Deliverable: `splatica_demo_app/outputs/farm_recon`.

- 21 отфильтрованный объект;
- 72 реальные фотографии из `data/knaack_colmap_pinhole`;
- 2–5 кадров на объект, приоритет целого незакрытого силуэта;
- native lossless crop без растяжения/blur;
- connected masks, nearest-neighbour resize;
- per-view reverse projection;
- общий GLB, world NPZ и два PLY.

После ориентационного исправления: 20 PASS, bucket #17 REVIEW; все 21 включены.
Общий GLB: 21 geometry, 6 083 340 вершин, 12 167 794 граней. Общий PLY:
10 320 512 строк. World↔Y-up согласован с ошибкой меньше 4 микрометров.

### Причина наклонённых объектов

Старая сборка сопоставляла PCA-оси с OBB dimensions. Для симметричной или
частичной генерации PCA-ось не означает «верх объекта». Поэтому OBB мог быть
правильным, а реконструированный splat — наклонённым.

`scripts/geometry/refine_farm_recon_orientation.py` теперь:

1. восстанавливает canonical Z-up MV-SAM3D из pose contract;
2. вычисляет минимальную rotation к `resolved_up` FARM;
3. проверяет 0–100% коррекции во всех реальных PINHOLE-камерах;
4. сравнивает полную silhouette mask, не только bbox;
5. выбирает максимальную коррекцию в пределах 0.005 от лучшего IoU;
6. для углов <3° не вращает объект из-за numerical noise;
7. применяет ту же affine rotation к mesh, Gaussian centres, covariance,
   normals и quaternions;
8. сохраняет before/after angle и per-view QA JPEG.

Примеры измерений:

| Object | Было | Стало | Median silhouette IoU |
| --- | ---: | ---: | ---: |
| vehicle #352 | 8.6° | 0.0° | .676 → .690 |
| scissor lift #445 | 20.1° | 2.0° | .716 → .727 |
| cart #993 | 38.7° | 0.0° | .663 → .707 |
| bucket #17 | 75.7° | 30.3° | .629 → .714 |

Box #326 и cart #329 имеют ambiguous canonical axis; полный 90° flip ухудшает
маски. Поэтому применена только evidence-supported часть, а не визуально
удобная, но ложная принудительная ориентация.

## Instance colours на 8082

`farm_object_id` каждой PLY-строки уже авторитетно записан при сборке. Viewer
не сегментирует заново. При включении palette вычисляется:

```text
hue = (0.11 + farm_object_id × golden_ratio_fraction) mod 1
RGB = HSV(hue, saturation=0.72, value=0.96)
```

То есть одинаковый ID всегда имеет один цвет. Цвет не означает class,
confidence, mask strength, PASS/REVIEW или качество формы. При выключении
виден восстановленный DC RGB splat-а.

## Acceptance checklist

1. Clean commit, exact image/model/config pins.
2. Strict scene/resource preflight: 0 errors/warnings.
3. Новый 17-stage cold run, root `_SUCCESS.json`.
4. Zero hard label errors и blocking overlap после final acceptance.
5. Просмотр OBB на контексте и реальных object crops, не OBB-only overview.
6. Retention/surface/uncertainty dashboard без скрытого массового мусора.
7. Lift без retune, release strict fraction пройдена.
8. Dense arrays == CSR; полный PLY сохраняет все исходные строки/поля.
9. MV-SAM3D: 2–5 хороших кадров, целый объект, connected masks, per-view
   silhouette и bbox QA.
10. Gravity/camera/coordinate transform записаны; ручных flip нет.
11. GLB/NPZ/PLY finite, ID/count/hash совпадают.
12. REVIEW-объекты явно перечислены, не скрыты.

## Ограничения

- Selector не является learned semantic coverage optimiser.
- OBB строится по upstream evidence и может быть условным на всех views.
- Viser передаёт covariance как float16, RGBA как uint8 и показывает только DC;
  это QA preview, не native photorealistic 3DGS.
- ShapeR/MV-SAM3D hallucinate unseen surfaces.
- Полная 2D silhouette лучше bbox, но не заменяет ground-truth 3D.

## Explicit YOLOE checkpoint comparison (development)

The legacy pipeline default remains unchanged. `farm quality discovery` accepts
an explicit segmentation `--checkpoint` in an isolated Ultralytics >=8.4 runtime.
For YOLOE-26L/26X, put the official checkpoint and `mobileclip2_b.ts` under the
local model root. The adapter never downloads these implicitly.

```bash
farm quality discovery --plan /data/development_plan.json \
  --model-root /models --vocabulary /config/yoloe_vocabulary.txt \
  --checkpoint /models/yoloe/yoloe-26x-seg.pt \
  --confidence 0.25 --world-up 0 -1 0 --output /output/new_comparison
```

The prepared plan lists unique development source frames with hashes, camera
rotations and capture timestamps. Alternatively supply `--packet`, `--colmap`
and `--images` to build the ordinary plan. Do not combine both input modes.

The adapter preserves native suppression and real mask logits in source camera
orientation. A startup parity check must exercise the custom predictor. Person
exclusions are emitted only when the vocabulary actually contains `person`;
zero detections without this query are not evidence that the image has no people.
The native geometry adapter accepts both per-frame query metadata and the
explicit all-frame detector query declaration.

This is a detector-to-geometry integration, not a drop-in replacement for the
legacy visual embedding/association backend. Detector category hypotheses remain
separate from final appearance annotations. `quality scene-catalog` takes labels
and captions from the namespace-bound Qwen appearance stage; absent annotations
remain unavailable. MobileCLIP text embeddings do not establish an object's
physical identity, ownership, completeness or measured dimensions.

### Depth-consistent association experiment

`quality proposal-geometry --depth-consistent-association` lets independently
sampled surfaces match within the same depth tolerance already used by their
bidirectional projections. Mask agreement, visibility, ambiguity, same-frame
conflicts and cannot-links still apply. The flag is off by default. Tolerances
use the units of prepared geometry; the experiment does not validate physical
scene scale.

Frozen-node replays on Factory, Knaack and Industrial preserve all previous
cannot-links and join additional independent observations. These are association
results, not object recall: building surfaces also acquire more observations.
On Industrial, newly supported ceiling masks compete with light fixtures for
Gaussian ownership, reducing several fixture masks and increasing lift time.
Do not enable this flag as a quality preset before separating background context
from object ownership and reviewing both recovered and degraded objects.

Validation: 1,803 tests passed, 6 modern-runtime tests skipped in the legacy
container; modern decoder parity is checked separately. Experiment sources,
full-cohort membership comparisons and visual reviews are under
`output/farm_pipeline/research/quality_2026-09-09/` outside the repository.

## YOLOE as the primary detector in the common profile

The measured YOLOE adapter can now feed the existing scene pipeline, including
adaptive registered-view selection, native Gaussian masks, nested alternatives,
Qwen appearance and the final candidate catalog. Opt in with
`quality scene-profile plan --detector yoloe --yoloe-model-root ...`.
The historical SAM3 profile remains available. With zero recovery/refinement
budgets, a YOLOE profile does not require or load SAM3.

For YOLOE-26L/X, supply `--yoloe-checkpoint` and a `detector` command prefix
in the runtime JSON, alongside `main` and `geometry`. This isolates modern
Ultralytics from the Qwen and native-rendering environments. The local
`yoloe/mobileclip2_b.ts` checkpoint is required and verified against the loaded
encoder. Without an explicit checkpoint the pinned legacy YOLOE-v8L backend is
used. Model weights, text weights, vocabulary and inference configuration are
bound to the stage; mixed batches cannot silently merge.

The YOLOE search vocabulary retains the broad repository vocabulary, adds the
explicit person exclusion query and scene-specific object proposals, with a
2,048-term cap. `--yoloe-vocabulary` can supply another broad core. Scene query
roles and conflicting role hypotheses remain metadata; they do not authorize
object deletion or determine final labels. Qwen independently describes actual
selected regions. Modern visual embeddings are not exported to the original
feature-based association backend; this profile uses geometric association.

`--depth-consistent-association` is available in the compiled profile as a
separate experiment. Its improved view association also strengthens large
structural masks, so it does not establish useful-object recall or safe
automatic inventory admission. Nested alternatives remain separate hypotheses.
An empty adaptive schedule records zero inference and merges with the initial
batch without reloading the model. All budgets remain explicit.


The optional `--explore-uncovered` profile flag spends only the unused adaptive
view budget on camera positions/directions not covered by observed or already
scheduled frames. The previous visibility-driven choices retain priority.
Exploration supplies no visibility votes or object admission; it only schedules
segmentation. Physical timestamps still count once downstream. The trajectory
scale uses O(N) storage and is invariant to scene translation, rotation and
uniform scale; scoring is bounded by the adaptive view budget. Camera novelty
is a scheduling heuristic, not proof of a newly discovered object.
