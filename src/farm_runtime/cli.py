from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

from .config import Plan, PlanError, load_plan, plan_to_public_dict
from .orchestrator import PipelineRunError, RunOrchestrator, resolve_run_dir
from .viewer import ViewerError, serve_viewer, stop_viewer, viewer_status


def _add_config(parser: argparse.ArgumentParser, *, required: bool = False) -> None:
    parser.add_argument("--config", type=Path, required=required, help="Generic FARM pipeline YAML")
    parser.add_argument("--project-root", type=Path, help="Override project_root from YAML")


def _add_run_lookup(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--run", "--run-dir", dest="run_dir", type=Path, help="Immutable FARM run directory")
    _add_config(parser)
    parser.add_argument("--attempt", action="store_true", help="Use latest-attempt instead of latest")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="farm-pipeline", description="Reproducible generic FARM pipeline runtime")
    commands = parser.add_subparsers(dest="subcommand", required=True)
    validate = commands.add_parser("validate-plan", help="Validate and print the resolved stage DAG")
    _add_config(validate, required=True)
    validate.add_argument("--json", action="store_true")
    run = commands.add_parser("run", help="Create or resume an immutable scene run")
    _add_config(run, required=True)
    run.add_argument("--run-id")
    run.add_argument("--resume", action="store_true")
    run.add_argument("--force", action="store_true")
    run.add_argument("--force-stage", action="append", default=[])
    status = commands.add_parser("status", help="Show pipeline and managed viewer status")
    _add_run_lookup(status)
    status.add_argument("--json", action="store_true")
    serve = commands.add_parser("serve", help="Explicitly start the configured viewer")
    _add_run_lookup(serve)
    serve.add_argument("--host")
    serve.add_argument("--port", type=int)
    serve.add_argument(
        "--runtime",
        choices=("auto", "host", "docker"),
        default="auto",
        help="Viewer execution runtime; auto honors the bundle and falls back to Docker",
    )
    serve.add_argument("--json", action="store_true")
    stop = commands.add_parser("stop", help="Stop only the viewer managed for this run")
    _add_run_lookup(stop)
    stop.add_argument("--timeout", type=float, default=10.0)
    stop.add_argument("--json", action="store_true")
    return parser


def _plan(args: argparse.Namespace) -> Plan | None:
    return load_plan(args.config, project_root=args.project_root) if args.config else None


def _print(value: Any, as_json: bool) -> None:
    if as_json:
        print(json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True))
    elif isinstance(value, dict):
        for key, item in value.items():
            print(f"{key}: {item}")
    else:
        print(value)


def _status(run_dir: Path) -> dict[str, Any]:
    manifest = json.loads((run_dir / "manifest.json").read_text(encoding="utf-8"))
    stages = {}
    for stage_id in manifest.get("stage_order", []):
        try:
            state = json.loads((run_dir / "stages" / stage_id / "state.json").read_text(encoding="utf-8"))
            stages[stage_id] = {"status": state.get("status"), "attempt": state.get("attempt"), "duration_seconds": state.get("duration_seconds")}
        except (FileNotFoundError, OSError, json.JSONDecodeError):
            stages[stage_id] = {"status": "not-run"}
    return {
        "run_dir": str(run_dir), "pipeline": manifest.get("status", "unknown"),
        "success_marker": (run_dir / "_SUCCESS.json").exists(),
        "failure_marker": (run_dir / "_FAILED.json").exists(),
        "stages": stages, "viewer": viewer_status(run_dir),
    }


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    try:
        if args.subcommand == "validate-plan":
            plan = _plan(args)
            assert plan is not None
            value = plan_to_public_dict(plan, plan.output_root / plan.scene_id / "runs" / "<RUN_ID>")
            if args.json:
                _print(value, True)
            else:
                print(f"Scene: {plan.scene_id}\nOutput: {plan.output_root}\nDAG: " + " -> ".join(plan.stage_order))
                for stage in value["stages"]:
                    print(f"  {stage['id']}: {' '.join(stage['command'])}")
            return 0
        if args.subcommand == "run":
            plan = _plan(args)
            assert plan is not None
            print(RunOrchestrator(plan).run(resume=args.resume, run_id=args.run_id, force=args.force, force_stages=args.force_stage))
            return 0
        plan = _plan(args)
        run_dir = resolve_run_dir(plan, args.run_dir, attempt=args.attempt)
        if args.subcommand == "status":
            _print(_status(run_dir), args.json)
        elif args.subcommand == "serve":
            _print(
                serve_viewer(
                    run_dir,
                    host=args.host,
                    port=args.port,
                    runtime=args.runtime,
                ),
                args.json,
            )
        elif args.subcommand == "stop":
            if args.timeout <= 0:
                raise ViewerError("--timeout must be positive")
            _print(stop_viewer(run_dir, timeout_seconds=args.timeout), args.json)
        return 0
    except (PlanError, PipelineRunError, ViewerError, OSError, ValueError, json.JSONDecodeError) as exc:
        print(f"farm-pipeline: {exc}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
