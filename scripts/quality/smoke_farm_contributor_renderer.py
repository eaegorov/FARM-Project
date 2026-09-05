#!/usr/bin/env python3
"""Analytic F0 smoke for pinned gsplat alpha/feature VJP and metric depth."""
from __future__ import annotations

# Allow direct execution from the source checkout.
import sys as _entry_sys
from pathlib import Path as _EntryPath
_entry_root = _EntryPath(__file__).resolve().parents[2]
_entry_sys.path[:0] = [str(_entry_root), str(_entry_root / "src")]

import argparse
import json
from pathlib import Path
import platform
import resource
import time


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--output", required=True, type=Path)
    args = p.parse_args()
    if args.output.exists():
        raise ValueError("output must be new")
    import gsplat
    import torch
    if not torch.cuda.is_available():
        raise RuntimeError("FARM renderer smoke requires the pinned CUDA prep runtime")
    started = time.monotonic()
    device = "cuda"
    torch.cuda.reset_peak_memory_stats()
    features = torch.ones((1, 1), device=device, requires_grad=True)
    kwargs = dict(
        means=torch.tensor([[0., 0., 3.]], device=device),
        quats=torch.tensor([[1., 0., 0., 0.]], device=device),
        scales=torch.tensor([[0.10, 0.10, 0.02]], device=device),
        opacities=torch.tensor([0.85], device=device),
        viewmats=torch.eye(4, device=device)[None],
        Ks=torch.tensor([[[48., 0., 32.], [0., 48., 32.], [0., 0., 1.]]], device=device),
        width=64, height=64, packed=True, near_plane=0.01, far_plane=20.0,
    )
    rgb, alpha, _ = gsplat.rasterization(colors=features, render_mode="RGB", **kwargs)
    mask = torch.zeros_like(alpha)
    mask[:, 24:40, 24:40] = 1
    gradient, = torch.autograd.grad((rgb*mask).sum(), features)
    expected = (alpha*mask).sum()
    depth, depth_alpha, _ = gsplat.rasterization(colors=features.detach(), render_mode="ED", **kwargs)
    visible = depth_alpha > 0.01
    torch.cuda.synchronize()
    errors = {
        "feature_one_vs_alpha_max": float((rgb-alpha).abs().max()),
        "vjp_vs_masked_alpha_sum": float((gradient.sum()-expected).abs()),
        "metric_depth_vs_3m_max": float((depth[visible]-3.0).abs().max()),
    }
    passed = errors["feature_one_vs_alpha_max"] < 1e-6 and errors["vjp_vs_masked_alpha_sum"] < 1e-4 and errors["metric_depth_vs_3m_max"] < 1e-5
    result = {"schema": "farm.contributor-renderer-smoke.v1", "status": "PASS" if passed else "FAIL",
              "errors": errors, "gaussians": 1, "image_size": [64, 64],
              "masked_contributor_weight": float(gradient.sum()), "python": platform.python_version(),
              "torch": torch.__version__, "cuda": torch.version.cuda, "gsplat": gsplat.__version__,
              "gpu": torch.cuda.get_device_name(), "wall_seconds": time.monotonic()-started,
              "peak_allocated_mib": torch.cuda.max_memory_allocated()/2**20,
              "peak_reserved_mib": torch.cuda.max_memory_reserved()/2**20,
              "peak_rss_kib": resource.getrusage(resource.RUSAGE_SELF).ru_maxrss,
              "scope": "Analytic kernel smoke; not a full-scene mask accuracy test."}
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2)+"\n")
    print(json.dumps(result))
    return 0 if passed else 1


if __name__ == "__main__":
    raise SystemExit(main())
