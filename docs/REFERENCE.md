# Справочник FARM

## 17 production-стадий

```text
preflight → selection → rgbd → mapping → mapping_qa → presentation
→ geometry → visual_consistency → semantics → part_whole → assemblies
→ surface_support → compound_geometry → geometry_qa → dedup
→ finalize → qa_bundle
```

Стадии и их code fingerprints объявлены в `src/farm_runtime/standard.py`.

## Основной CLI

```bash
farm validate-plan --config SCENE.yaml
farm run --config SCENE.yaml --run-id UNIQUE_ID
farm status --config SCENE.yaml --attempt --json
farm serve --config SCENE.yaml
farm stop --config SCENE.yaml
```

Эквивалент без установленного entry point:

```bash
.venv-control/bin/python scripts/farm_pipeline.py <command> ...
```

## Структура run

```text
<output_root>/<scene_id>/runs/<run_id>/
  _SUCCESS.json
  manifest.json
  config/source_snapshot/FARM-Project/
  input/
  selection/
  rgbd/
  mapping/
  final/{scene_state.pt,catalog.json,presentation_catalog.json,cloud.npz}
  qa/{acceptance,summary,retention_funnel,...}
  stages/<stage>/attempts/<n>/
  timing/summary.json
  visuals/
  viewer/
```

`scene_state.pt` — trusted PyTorch pickle. Открывайте только локально созданные
или доверенные файлы.

## Lift outputs

```text
per_gaussian_object_id.npy          int32; -1 = unknown
per_gaussian_confidence.npy         float32
per_gaussian_timestamp_support.npy  uint16
per_gaussian_status.npy             uint8
verified_instance_bank.npz          CSR verified membership
split_manifest.json
build_manifest.json
heldout_qc.json
result.json
instance_labeled_full.ply
labeled_ply_manifest.json
```

## ShapeR outputs

```text
inputs/result.json + PKL
batch/result.json + per-object GLB/QA
scene/scene_manifest.json
scene/viewer_assets.json
scene/combined_metric_objects.{glb,npz}
```

У каждого уровня ровно один marker: `_SUCCESS.json` или
`_NONRELEASE_SUCCESS.json`.

## MV-SAM3D outputs

```text
farm_recon/
  object_<id>_<class>/
    snapshots/<rank>/{image-source.png,mask.png,...}
    reconstruction/{GLB,PLY,NPZ,GIF,result.json}
    metric/{model_metric.glb,result.json,*-qa.jpg}
  combined_metric_objects.glb
  combined_metric_objects.npz
  combined_world_metric_objects.npz
  combined_world_metric_gaussians.ply
  combined_viewer_y_up_gaussians.ply
  manifest.json
  orientation_refinement.json
  alignment_audit.json
  _SUCCESS.json
```

## Viewer и порты

| Порт | Назначение |
| ---: | --- |
| 8080 | unified Factory/Knaack FARM/lift/ShapeR |
| 8082 | Knaack scene cloud + MV-SAM3D Gaussian overlay |
| 8000/8002/8006 | временные model services production-run |

Model-service ports после terminal run должны быть свободны. Viewer-контейнеры
не удаляются pipeline cleanup, если они не относятся к run scope.

## Важные скрипты

- `farm_pipeline.py` — validate/run/status/serve/stop.
- `farm_standard_stage.py` — snapshot-bound stage wrapper.
- `farm_preflight.py`, `farm_resource_preflight.py` — inputs/resources.
- `pin_farm_runtime_images.py` — tag→immutable image ID audit/repin.
- `serve_farm_unified_viewer.py` — основной viewer.
- `serve_farm_recon_overlay.py` — Knaack MV-SAM3D overlay.
- `refine_farm_recon_orientation.py` — gravity + full-mask refinement.
- `tools/farm_shaper_bridge/run_gaussian_lift.py` — dense lift.
- `tools/farm_shaper_bridge/run_bridge_stage.py` — ShapeR prepare/assemble.
- `tools/farm_shaper_bridge/run_shaper.py` — pinned ShapeR inference.

## Диагностика

### Run упал сразу на import

Пересоздайте `.venv-control` строго из `requirements/control-plane.lock.txt` и
запустите `scripts/farm_runtime_inventory.py validate --lock
requirements/control-plane.lock.txt`.

### Mapping не импортирует YOLOE/MobileCLIP

Проверьте recursive submodule и source snapshot closure. Production snapshot
должен содержать runtime package roots, иначе editable install внутри image
будет скрыт read-only mount.

### Viewer показывает белую страницу

```bash
docker ps
docker logs --tail 100 <viewer-container>
curl -I http://127.0.0.1:8080/
```

Проверьте UID/GID, read permission marker/artifacts, WebSocket и firewall.

### Объект наклонён, а OBB правильный

Не вращайте PLY вручную. Запустите `refine_farm_recon_orientation.py`, проверьте
`orientation-silhouette-qa.jpg`, before/after angle и full-mask IoU. Затем
повторите `farm_recon_validate.py` и перезапустите viewer.

### Release lift не появился

Читайте `result.json` и marker. `PASS/WARN` algorithm может корректно дать
`_NONRELEASE_SUCCESS.json`, если strict verified fraction ниже frozen floor.
