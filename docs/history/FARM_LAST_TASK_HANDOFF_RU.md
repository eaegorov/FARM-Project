# FARM V11 — текущий handoff

Дата среза: 2026-09-04

Рабочая директория: `/home/ubuntu/workspace/3dgs_work`

Репозиторий: `/home/ubuntu/workspace/3dgs_work/projects/FARM-Project`

Эксперимент: `/home/ubuntu/workspace/3dgs_work/output/farm_pipeline/factory/experiments/factory-universal-v11-development-v1`

## Статус

Factory Universal V11 остаётся **DEVELOPMENT / BLOCKED / НЕ PRODUCTION**:

- production-ready V11 objects: **0**;
- inpainting-ready objects: **0**;
- production catalog: **пуст**;
- scene release: **BLOCKED**;
- positive `mask_verified`: #0/#16/#37/#47/#108;
- positive `obb_verified`: #16/#37/#108;
- allowlist manifest: `release_eligible=false`, `experimental_nonrelease=true`.

V11 улучшает fail-closed evidence pipeline, но не заменяет действующий Factory V10. Clean-GS, Gaussian Grouping, Inpaint360GS, InstaInpaint и AuraFusion360 не являются внутренними стадиями V11; их результаты живут во внешнем benchmark:

`/home/ubuntu/workspace/3dgs_work/output/factory_object_inpainting_benchmark_v1/REPORT_RU.md`

## SAM3 против SAM1+DEVA

Контролируемый A/B использовал одинаковые 200 Full-COLMAP кадров: 20 episodes × 10, train/heldout 100/100, tracker reset на каждый camera × view-family episode, physical-timestamp overlap 0.

| Метрика | SAM1+DEVA | SAM3 |
|---|---:|---:|
| Cold model load | 10,915 с | 16,015 с |
| Inference | 552,014 с | **383,029 с** |
| End-to-end suite | 617,951 с | **448,202 с** |
| External wall | 619,565 с | **450,265 с** |
| Скорость | 0,362 fps | **0,522 fps** |
| Peak torch allocated | 15 724,84 MiB | **4 948,22 MiB** |
| Peak torch reserved | 21 396 MiB | **5 682 MiB** |
| Peak whole-device VRAM | 21 648 MiB | **5 930 MiB** |

SAM3 быстрее по inference в 1,44 раза и требует в 3,18 раза меньше allocated VRAM. При этом strict association дал 0/4 heldout matches у SAM1+DEVA и 0/5 у SAM3. Вывод ограничен: SAM3 — предпочтительный bounded proposal/tracking backend, но scene-wide cross-fold identity не доказана.

Object-paired SAM3 повтор: 10 объектов, 173 RGB (87 train / 86 heldout), 83 / 79 timestamps, overlap 0; external wall 356,306 с; peak allocated/reserved 4 569,02 / 5 120 MiB; whole-device delta 5 368 MiB. Anchor reconciliation принял только #2 и #37; strict v1 применил 0 heldout identities, шесть shadow associations diagnostic-only.

## Full-COLMAP и frozen evidence

- 6310 registered images; рассмотрены 26 объектов; 24 получили retrieval plan, 2 fail-closed без OBB fallback.
- RGB-D union: 140 views; train 100 images / 70 timestamps, heldout 40 / 26; image и timestamp overlap 0.
- 38 accepted train views для 10 объектов, но accepted view не равен released object.
- Gaussian frozen QC: mask PASS #0/#16/#37/#47/#108; #185 rejected, включая depth p90 0,219 м при лимите 0,20 м.
- Strict train OBB release: #13/#16/#37/#108; pre-lift selector допустил #16/#37/#108.
- Post-lift positive OBB allowlist: #16/#37/#108.

## #37: только OBB gain

Universal planar-yaw correction на полном 10-object scope приняла только #37; baseline остальных девяти сохранён. Для тех же 40 827 Gaussian:

| Метрика | До | После |
|---|---:|---:|
| Yaw/normal-axis error | 6,8717° | **0,6940°** |
| Train median IoU | 0,835591 | **0,868143** |
| Train q25 IoU | 0,751464 | **0,853589** |
| Gaussian inside OBB | 0,459451 | **0,917775** |
| OBB/support volume ratio | 0,410340 | **0,780894** |
| Normalized support center offset | 0,297315 | **0,071098** |
| Post-lift OBB gate | FAIL | **PASS** |

Коррекция −6,836745°, materiality angle 2,953142°. Это **OBB-only improvement**: mask, semantic identity, whole-object scope и production readiness не менялись. Slab-pruning, удалявший 11,24% Gaussian, отклонён: heldout IoU 0,85388→0,85336, recall/depth ухудшились.

## Semantic gate

Authoritative semantic layer — `semantic_scene_wide_resolved_v3` поверх immutable V8 outputs:

- exact identity принята только у #2 (`fire extinguisher`) и #37 (`sign`);
- пять объектов остаются quarantined;
- whole-object audit оставляет **0 inpainting-ready**;
- #2 требует доказать полный hose/nozzle scope;
- #37 имеет подтверждённый head noun, но `complete_bounded=false`;
- #108 сохраняет quarantined carrier/content/attached-pipe ambiguity.

SAM3 concept second-chance остаётся незавершённым: CPU/code contract и 154 regression tests готовы, но финального acceptance-grade frozen A/B нет. Viewed diagnostic #37 имел external wall 22,519 с, whole-device peak delta 2 476 MiB и итог 0 accepted objects / 0 newly accepted views. `authorization=diagnostic_only`; allowlists не менялись.

## Финальный Docker

Финальный runtime image собран и проверен:

- tag: `scene_graph:farm-v11-universal-final`;
- digest: `sha256:52369591dfc734774319319a5c7fcdb513c716361d186d52932307e9f8659a8e`;
- build: `rc=0`, 621,614 с, peak process-tree RSS 110 504 KiB, GPU delta 0 MiB;
- offline smoke с `--network none`: `rc=0`, 36,141 с, peak RSS 29 912 KiB, GPU 0 MiB;
- smoke imports: `farm_runtime.standard`, `farm_runtime.second_chance_refinement`, `farm_runtime.semantic_evidence_resolver`.

Это подтверждает runtime/import surface, но не является scene release.

## V10 не изменён

Действующий baseline остаётся Factory V10:

- контейнер `farm-factory-viewer-8080` использует image `scene_graph:farm-v10-relation-v1-final`;
- image ID: `sha256:b1d5b7f642869409ef24016dcda32c107f2339ef225e9063e3662a81f91f57ce`;
- registry: `/home/ubuntu/workspace/3dgs_work/output/farm_pipeline/factory/experiments/factory-production-v10-relation-v1/viewer/scenes.v10.yaml`;
- registry SHA-256: `2f2938ec856fd20f9d548b3c4c82f7f5f2cdde05b506161be305975e8a6c72c8`;
- registry mtime: `2026-09-02 17:53:51 UTC`; running container started `2026-09-02T20:07:45.766843119Z`;
- output/data mounts контейнера read-only.

Порт 8080 в V11 work не переключался и не проверялся запросами. Исторического registry checksum для байтового before/after сравнения нет; неизменность поддерживается pre-V11 mtime/start time, сохранённым V10 digest/tag, read-only data mounts и отсутствием V11 release action.

## Ближайшие обязательные действия

1. Выполнить acceptance-grade second-chance только на заранее frozen train/heldout targets; viewed diagnostic #37 не использовать для state change.
2. Закрыть whole-object scope для #37, #2 и #108; semantic head noun не считать доказательством mask completeness.
3. Для #0/#47 получить независимое train OBB evidence без heldout fit/candidate selection.
4. Повторить SHA-bound mask/OBB/semantic/scope gates и scene conflict audit; публиковать только их пересечение.
5. Не менять V10 и viewer registry, пока V11 production catalog пуст и release `BLOCKED`.

## Authoritative пути

- Итоговый V11 report: `/home/ubuntu/workspace/3dgs_work/output/farm_pipeline/factory/experiments/factory-universal-v11-development-v1/REPORT_RU.md`.
- Tracker A/B: `.../02_tracker_ab/REPORT_RU.md`.
- Full-COLMAP folds: `.../06_full_colmap_refinement/REPORT_RU.md`.
- Strict train OBB: `.../06_full_colmap_refinement/obb_release_gate_v1/REPORT_RU.md`.
- Semantic V3: `.../06_full_colmap_refinement/semantic_scene_wide_resolved_v3/REPORT_RU.md`.
- Frozen QC: `.../07_gaussian_lift/area_cap_planar_v5_dedup_upright_candidate/frozen_heldout_v1/REPORT_RU.md`.
- Full planar-yaw/post-lift audit: `.../07_gaussian_lift/area_cap_planar_v5_dedup_upright_candidate/post_lift_obb_consensus_yaw_full_queue_v1/REPORT_RU.md`.
- Docker build/smoke: `.../08_container_build/measurement.json`, `.../08_container_build/smoke_measurement.json`, `.../08_container_build/image_identity.txt`.
- External benchmark: `/home/ubuntu/workspace/3dgs_work/output/factory_object_inpainting_benchmark_v1/REPORT_RU.md`.
