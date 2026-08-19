# Universal COLMAP + 3DGS scene preflight

`scripts/farm_preflight.py` is the CPU-only gate before keyframe selection,
depth rendering, model startup or mapping. It reads its inputs without changing
them. Supplying `--report` is the only operation that writes a file.

```bash
python scripts/farm_preflight.py \
  --config configs/scenes/my_scene.yaml \
  --report output/my_scene/preflight.json \
  --strict
```

Start from `configs/scenes/example_3dgs_colmap.yaml`. Paths are relative to the
config file. The formal contract is `configs/schema/farm_scene.schema.json`.
The loader also accepts the early flat aliases `colmap`, `images`, `ply` and
`output`, but new configs should use the nested schema.

## What is checked

- a complete standard COLMAP text or binary model;
- valid cameras, poses, sparse points, unique image IDs/names and image files;
- downstream-supported camera models (normally virtual `PINHOLE`);
- regex rig grouping and complete member/view combinations;
- 3DGS PLY header, exact vertex count/file size, Gaussian properties and a
  bounded deterministic finite/bounds sample;
- metric-scale evidence separately from coordinate-frame alignment;
- explicit gravity/up or camera-roll consensus;
- robust COLMAP sparse-point versus Gaussian centroid, extent, AABB overlap
  and sampled nearest-neighbour agreement;
- stable full SHA-256 hashes for small inputs and explicitly labelled
  head/middle/tail sampled hashes for large inputs.

`pass` is ready. `warn` has no hard failure but requires attention; use
`--strict` for production so it returns exit code 3. `fail` returns exit code 1.
A configuration parse error returns 2.

## Important limits

The lightweight alignment gate catches wrong PLY files, transforms and gross
scale offsets. It cannot prove pixel-level registration. The run must still
perform a small rendered RGB/depth agreement QA before processing all selected
views. Likewise, sharing COLMAP coordinates does not prove metres: use a known
rig baseline or an auditable `declared_metric` scale.

Original KB4/fisheye COLMAP is expected to fail the default camera-model gate.
Convert it to virtual pinhole views first; do not merely add KB4 to
`accepted_models` unless the dense-depth adapter has matching distortion math.
