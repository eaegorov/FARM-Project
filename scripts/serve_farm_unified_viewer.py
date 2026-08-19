#!/usr/bin/env python3
"""Validate or serve the read-only multi-scene FARM viewer on one port."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from farm_runtime.unified_viewer import (  # noqa: E402
    UnifiedViewerError,
    serve_registry,
    validate_registry,
)


DEFAULT_REGISTRY = PROJECT_ROOT / "configs" / "viewer" / "scenes.local.v1.yaml"


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""Current scene_graph image permissions contract:
  keep the image user as --user 1000:1000 so /home/scene_graph/.venv is readable;
  add --group-add \"$(id -g)\" so host scene artifacts with mode 0660 are readable.

Mount scene inputs read-only and publish a port only when remote access is intended.
This is an unauthenticated single-user inspection service. The default bind
address is loopback (127.0.0.1); non-loopback binds emit a runtime warning.
""",
    )
    parser.add_argument("--registry", type=Path, default=DEFAULT_REGISTRY)
    parser.add_argument("--validate", action="store_true", help="Validate all required scene/run/source contracts and exit")
    parser.add_argument(
        "--verify-full-source",
        action="store_true",
        help="Also stream and SHA-256 source PLYs for any available canonical dense lifts",
    )
    parser.add_argument(
        "--host",
        default="127.0.0.1",
        help="Bind address; loopback by default. Pass 0.0.0.0 only for deliberate network exposure.",
    )
    parser.add_argument("--port", type=int, default=8080)
    parser.add_argument("--scene", help="Initial scene_id (defaults to the first registry scene)")
    parser.add_argument("--point-size", type=float, default=0.004)
    parser.add_argument("--max-preview-points", type=int, default=1_500_000)
    parser.add_argument("--max-instance-points", type=int, default=700_000)
    parser.add_argument("--max-dense-points", type=int, default=700_000)
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    try:
        validation = validate_registry(
            args.registry,
            verify_full_source=bool(args.verify_full_source),
        )
        if args.validate:
            print(json.dumps(validation.as_dict(), ensure_ascii=False, indent=2))
            return 0 if validation.ok else 2
        if not validation.ready_scenes:
            print(json.dumps(validation.as_dict(), ensure_ascii=False, indent=2), file=sys.stderr)
            return 2
        if not validation.ok:
            print(json.dumps(validation.as_dict(), ensure_ascii=False, indent=2), file=sys.stderr)
        if args.scene is not None and args.scene not in {
            scene.spec.scene_id for scene in validation.ready_scenes
        }:
            raise UnifiedViewerError(f"unknown --scene {args.scene!r}")
        viewer = serve_registry(
            args.registry,
            validation=validation,
            host=args.host,
            port=args.port,
            initial_scene=args.scene,
            point_size=args.point_size,
            max_preview_points=args.max_preview_points,
            max_instance_points=args.max_instance_points,
            max_dense_points=args.max_dense_points,
        )
        print(json.dumps({
            "status": "READY",
            "host": args.host,
            "port": args.port,
            "registry": str(args.registry.expanduser().resolve()),
            "scenes": [scene.spec.scene_id for scene in validation.ready_scenes],
            "unavailable_scenes": [
                item.scene_id for item in validation.scenes if item not in validation.ready_scenes
            ],
            "renderer": "viser-one-canvas",
            "source_splats": "all-N Viser float16/uint8 quantized DC preview (no higher-order SH)",
            "security": "single-user; unauthenticated; loopback by default",
            "viser_required": "1.0.30",
        }, ensure_ascii=False, indent=2), flush=True)
        viewer.run_forever()
        return 0
    except (UnifiedViewerError, FileNotFoundError, OSError, ValueError) as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
