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

`scripts/refine_farm_recon_orientation.py` теперь:

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
