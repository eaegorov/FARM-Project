# Развёртывание FARM на новом сервере

Это единственная production-инструкция установки. Команды выполняются из
корня `FARM-Project`. Для release используйте чистый commit, новый cold run и
точно проверенные image/model pins.

## 1. Требования

- Linux x86_64;
- NVIDIA GPU и драйвер, совместимый с выбранными CUDA-образами;
- Docker + NVIDIA Container Toolkit;
- Git с submodule;
- Python 3.11+;
- место для исходного 3DGS, моделей, RGBD и immutable outputs.

Измеренные пики whole-device H100: Factory около 28.3 GiB, Knaack около
30.2 GiB. Это не гарантия для новой сцены: число кадров/объектов и размер PLY
меняют требования.

## 2. Клонирование и host runtime

```bash
git clone --recursive <URL> FARM-Project
cd FARM-Project

python3.11 -m venv .venv-control
.venv-control/bin/python -m pip install \
  --requirement requirements/control-plane.lock.txt
export FARM_PY="$PWD/.venv-control/bin/python"
"$FARM_PY" scripts/farm_runtime_inventory.py validate \\
  --lock requirements/control-plane.lock.txt
```

GitHub source archive не содержит `third_party/yoloe`; нужен recursive clone.
Host runtime выполняет CPU preflight/selection/report/orchestration и поэтому
содержит NumPy, SciPy, pycolmap, OpenCV, Pillow и CPU Torch. GPU inference
остаётся в контейнерах.

## 3. Модели

Примите лицензию gated DINOv3-S+/16 и подготовьте файл секретов с mode `0600`:

```json
{"HF_TOKEN": "hf_..."}
```

```bash
FARM_MODEL_PYTHON="$FARM_PY" ./bootstrap_models.sh \
  --secrets-file /absolute/private/secrets.json

# Полный offline-аудит уже скачанного:
FARM_MODEL_PYTHON="$FARM_PY" ./bootstrap_models.sh \
  --secrets-file /absolute/private/secrets.json \
  --local-files-only --verify-full-hashes
```

Ревизии и SHA берутся только из `configs/models/farm_models.v1.json`.

## 4. Docker: build → audit → pin → commit

```bash
FARM_UID="$(id -u)" FARM_GID="$(id -g)" ./run.sh build

docker build --check --file docker/Dockerfile.prep .
docker build --file docker/Dockerfile.prep \
  --build-arg FARM_UID="$(id -u)" \
  --build-arg FARM_GID="$(id -g)" \
  --tag rest3d:factory-baseline .

"$FARM_PY" scripts/pin_farm_runtime_images.py
```

Если audit сообщает drift, проверьте образы, затем:

```bash
"$FARM_PY" scripts/pin_farm_runtime_images.py --write
git diff -- configs/models/farm_models.v1.json
git add configs/models/farm_models.v1.json
git commit -m "build: pin reviewed FARM runtime images"
"$FARM_PY" scripts/pin_farm_runtime_images.py
```

Tag — не identity. Release identity — локальный immutable `sha256:` image ID,
зафиксированный в manifest.

## 5. Новая сцена

Подготовьте входы по [INPUTS.md](INPUTS.md), скопируйте
`configs/scenes/example_3dgs_colmap.yaml`, задайте абсолютные пути, scale,
gravity, grouping и resource manifest. Не переносите Factory/Knaack gravity
или scale на другую сцену.

```bash
"$FARM_PY" scripts/farm_preflight.py \
  --config configs/scenes/my_scene.yaml --strict

"$FARM_PY" scripts/farm_resource_preflight.py \
  --config configs/scenes/my_scene.yaml --verify-full-hashes

"$FARM_PY" scripts/farm_pipeline.py validate-plan \
  --config configs/scenes/my_scene.yaml
```

## 6. Cold FARM run

```bash
git status --short --untracked-files=all

"$FARM_PY" scripts/farm_pipeline.py run \
  --config configs/scenes/my_scene.yaml \
  --run-id my-scene-production-v1
```

Run создаёт `config/source_snapshot/FARM-Project` с точным file set, mode,
size и SHA-256. Все 17 стадий, finalizer и report запускаются из snapshot.
Изменение snapshot до/во время child process блокирует success.

Статус:

```bash
"$FARM_PY" scripts/farm_pipeline.py status \
  --config configs/scenes/my_scene.yaml --attempt --json
```

`--resume` допустим только для того же terminal/interrupted run, если snapshot,
конфиг и входы не изменились. Для acceptance после изменения чего-либо всегда
используйте новый `run-id`.

## 7. Gaussian lift

```bash
RUN=/absolute/output/farm_pipeline/my_scene/runs/my-scene-production-v1
PLY=/absolute/data/my_scene/scene.ply
LIFT=/absolute/output/farm_gaussian_lift/my_scene/my-scene-production-v1

"$FARM_PY" tools/farm_shaper_bridge/run_gaussian_lift.py \
  --run "$RUN" --ply "$PLY" --output "$LIFT" \
  --config configs/gaussian_lift.v1.yaml --gpu 0 --plan-only

"$FARM_PY" tools/farm_shaper_bridge/run_gaussian_lift.py \
  --run "$RUN" --ply "$PLY" --output "$LIFT" \
  --config configs/gaussian_lift.v1.yaml --gpu 0
```

`LIFT` должен быть новым/пустым. Конфиг обязан побайтно совпасть с копией в
run snapshot. Output сохраняет исходный порядок и все поля PLY; неизвестные и
не прошедшие held-out объекты остаются `-1`.

## 8. ShapeR (опционально)

ShapeR — mesh-гипотеза, не измеренная поверхность. Host launcher имеет
отдельный lock:

```bash
python3.11 -m venv .venv-bridge-control
.venv-bridge-control/bin/python -m pip install \
  --requirement requirements/bridge-control.lock.txt
export FARM_BRIDGE_PY="$PWD/.venv-bridge-control/bin/python"
export SHAPER_REPO=/absolute/ShapeR
export HF_CACHE=/absolute/hf-cache

"$FARM_BRIDGE_PY" tools/farm_shaper_bridge/manage_runtime.py validate \
  --config configs/shaper_bridge.v1.yaml \
  --shaper-repo "$SHAPER_REPO" --hf-cache "$HF_CACHE"
```

Release chain без override-флагов:

```bash
"$FARM_PY" tools/farm_shaper_bridge/run_bridge_stage.py prepare \
  --run "$RUN" --ply "$PLY" --lift "$LIFT" --output /output/shaper/inputs

"$FARM_BRIDGE_PY" tools/farm_shaper_bridge/run_shaper.py \
  --run "$RUN" --inputs /output/shaper/inputs --output /output/shaper/batch \
  --shaper-repo "$SHAPER_REPO" --hf-cache "$HF_CACHE" \
  --config configs/shaper_bridge.v1.yaml --profile balance --gpu 0

"$FARM_PY" tools/farm_shaper_bridge/run_bridge_stage.py assemble \
  --run "$RUN" --shaper-outputs /output/shaper/batch \
  --output /output/shaper/scene
```

## 9. Viewer

Сначала измените `configs/viewer/scenes.local.v1.yaml`: run, source PLY,
полный SHA/size и roots lift/ShapeR. Затем:

```bash
"$FARM_PY" scripts/serve_farm_unified_viewer.py \
  --registry configs/viewer/scenes.local.v1.yaml \
  --validate --verify-full-source
```

Viewer удобнее запускать в pinned main image с read-only mounts. Текущий
образ должен сохранять доступ к своей venv; добавьте host artifact group:

```bash
docker run --rm --network host --read-only \
  --cap-drop ALL --security-opt no-new-privileges \
  --user 1000:1000 --group-add "$(id -g)" \
  --tmpfs /tmp:rw,nosuid,nodev,size=2g \
  --mount type=bind,src=/absolute/FARM-Project,dst=/workspace/FARM-Project,readonly \
  --mount type=bind,src=/absolute/output,dst=/workspace/output,readonly \
  --mount type=bind,src=/absolute/data,dst=/workspace/data,readonly \
  --workdir /workspace/FARM-Project \
  --entrypoint /home/scene_graph/.venv/bin/python \
  sha256:<PINNED_MAIN_IMAGE_ID> \
  scripts/serve_farm_unified_viewer.py \
  --registry configs/viewer/scenes.local.v1.yaml \
  --host 127.0.0.1 --port 8080
```

Viewer не имеет аутентификации. Для remote используйте SSH tunnel или
защищённый reverse proxy; не публикуйте `0.0.0.0` напрямую.

## 10. Перенос артефактов

Переносите целиком:

- чистый Git commit FARM;
- scene YAML и model manifest;
- исходный COLMAP/RGB/PLY;
- run directory с `_SUCCESS.json`;
- lift/ShapeR directories со своими markers;
- viewer registry;
- Docker images или воспроизводимый build + последующий repin.

После копирования повторите runtime pin audit, full resource hash, run marker,
viewer `--validate --verify-full-source` и checklist из [QUALITY.md](QUALITY.md).
