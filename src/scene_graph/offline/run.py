"""Offline driver for the mapping pipeline.

Reuses the online ``StreamingMapper`` node unchanged — the only difference from
``mapping_five_cam.launch.py`` is the ingress: instead of ROS subscriptions, we
pull frames from a dataset on disk or a rosbag and feed them to
``mapper._run_mapping_batch(...)`` directly. The full algorithm (segmentation,
IoU filtering, correspondence, Gaussian fusion, covisibility, captioning,
pruning, snapshot) runs exactly as online.

Typical usage inside the Docker container::

    # ScanNet .sens
    python -m scene_graph.offline.run \
        --source sens \
        --sens-path /data/scannet/scene0000_00/scene0000_00.sens \
        --stride 5 \
        --save-path /data/out/scene0000_00/scene_state.pt

    # Rosbag
    python -m scene_graph.offline.run \
        --source rosbag \
        --bag-path /data/bags/office \
        --save-path /data/out/office/scene_state.pt
"""

from __future__ import annotations

import argparse
import contextlib
import json
import logging
import os
import sys
import time
from pathlib import Path
from typing import List, Optional

LOGGER = logging.getLogger("scene_graph.offline.run")


def _prefer_repo_mapping_source() -> None:
    """Prefer the checked-out ROS mapping package for offline runs.

    The long-lived Docker containers may have a stale colcon-built copy of the
    ``mapping`` Python package under ``/tmp/colcon_ws``.  Offline evaluation is
    driven from this repository checkout, so importing the source tree avoids a
    rebuild requirement when mapper code changes.
    """

    repo_root = Path(__file__).resolve().parents[3]
    mapping_src = repo_root / "ros" / "mapping"
    if mapping_src.exists():
        src = str(mapping_src)
        if src not in sys.path:
            sys.path.insert(0, src)
        pythonpath = os.environ.get("PYTHONPATH", "")
        if src not in pythonpath.split(os.pathsep):
            os.environ["PYTHONPATH"] = src + (os.pathsep + pythonpath if pythonpath else "")


def parse_args(argv: Optional[List[str]] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Offline runner for the mapping pipeline (shares StreamingMapper with online).",
    )
    parser.add_argument(
        "--source",
        choices=("sens", "rosbag", "npz", "frames-json"),
        required=True,
        help="Frame source type.",
    )
    # Source-specific
    parser.add_argument("--sens-path", type=Path, help="(sens) path to .sens archive.")
    parser.add_argument(
        "--camera",
        type=str,
        default="scannet",
        help="(sens|npz) camera id to tag frames with.",
    )
    parser.add_argument(
        "--stride",
        type=int,
        default=1,
        help="(sens|npz|frames-json) take every Nth frame.",
    )
    parser.add_argument(
        "--start",
        type=int,
        default=0,
        help="(sens|npz|frames-json) first frame index.",
    )
    parser.add_argument(
        "--end",
        type=int,
        default=-1,
        help="(sens|npz|frames-json) exclusive end frame (-1 for all).",
    )

    parser.add_argument("--bag-path", type=Path, help="(rosbag) path to rosbag2 directory or .mcap.")
    parser.add_argument(
        "--cameras",
        nargs="+",
        default=None,
        help="(rosbag) restrict to these cameras.",
    )
    parser.add_argument(
        "--every-nth",
        type=int,
        default=1,
        help="(rosbag) keep every Nth frame per camera.",
    )

    parser.add_argument(
        "--npz-dir",
        type=Path,
        help="(npz) directory of NPZ archives produced by an offline renderer.",
    )
    parser.add_argument(
        "--npz-pattern",
        type=str,
        default="*.npz",
        help="(npz) glob pattern relative to --npz-dir.",
    )

    parser.add_argument(
        "--frames-json-dir",
        type=Path,
        help="(frames-json) scene directory containing frames.json + image/depth subdirs.",
    )
    parser.add_argument(
        "--frames-json-camera",
        action="append",
        default=None,
        metavar="CAMERA",
        help="(frames-json) restrict to these cameras (repeat for multiple). Default: all.",
    )
    parser.add_argument(
        "--frames-json-depth-clip-m",
        type=float,
        default=None,
        help="(frames-json) clip depth values above this distance to 0. Default: no clip.",
    )
    parser.add_argument(
        "--frames-json-nominal-hz",
        type=float,
        default=10.0,
        help="(frames-json) nominal Hz used to synthesise stamp_ns when timestamp_ns is missing.",
    )

    # Driver/shared
    parser.add_argument(
        "--batch-size",
        type=int,
        default=1,
        help="Frames per _run_mapping_batch call. Online uses 1; larger may batch better.",
    )
    parser.add_argument(
        "--target-fps",
        type=float,
        default=40.0,
        help="Cap frame ingest rate to this many frames per second (0 disables). "
        "Captioning is async; pacing the mapper lets caption results land before "
        "downstream merges/canonicalizations would otherwise drop them.",
    )
    parser.add_argument(
        "--drop-when-late",
        action="store_true",
        help="Simulate real-deployment frame dropping: when processing takes "
        "longer than 1/target-fps per batch, advance the source iterator "
        "(drop frames) so the mapper stays synced to the wall-clock schedule. "
        "Without this, slow processing just stretches out the recon in wall-"
        "clock time without simulating queue-1 'latest-only' subscriber behavior.",
    )
    parser.add_argument(
        "--warmup-frames",
        type=int,
        default=1,
        help="Process this many frames *before* starting the wall-clock timer "
        "and `--drop-when-late` schedule. Their data still ends up in the "
        "scene_state, but their cost (YOLOE compile, CUDA warmup, first "
        "vLLM-tunnel round-trip, SigLIP2 warm-up) doesn't count against the "
        "deadline — otherwise the first-frame spike (~30-60s on HM3D) would "
        "cause 100+ subsequent frames to be dropped before the loop "
        "stabilises. Set 0 to disable.",
    )
    parser.add_argument(
        "--save-path",
        type=Path,
        default=None,
        help="Where to write scene_state.pt. Defaults to $SCENE_GRAPH_MAPPING_DATA_DIR/scene_state.pt.",
    )
    parser.add_argument(
        "--caption",
        action="store_true",
        help="Enable the caption manager (requires vLLM servers on default ports).",
    )
    parser.add_argument(
        "--caption-batch-size",
        type=int,
        default=10,
        help="Caption worker batch size when --caption is enabled.",
    )
    parser.add_argument(
        "--viser",
        action="store_true",
        help="Enable viser visualization.",
    )
    parser.add_argument("--viser-host", default="127.0.0.1", help="Host/interface for --viser.")
    parser.add_argument("--viser-port", type=int, default=8080, help="Port for --viser.")
    parser.add_argument(
        "--keep-viser-after-run",
        action="store_true",
        help="After the frame source is exhausted, save/drain captions but keep the embedded Viser server alive.",
    )
    parser.add_argument(
        "--viser-live-rgb",
        dest="viser_live_rgb",
        action="store_true",
        default=True,
        help="Show the latest RGB frame in the Viser GUI side panel.",
    )
    parser.add_argument(
        "--no-viser-live-rgb",
        dest="viser_live_rgb",
        action="store_false",
        help="Disable the latest-RGB side panel in Viser.",
    )
    parser.add_argument(
        "--viser-live-rgb-max-side",
        type=int,
        default=320,
        help="Resize the live RGB side-panel image to this longest side.",
    )
    parser.add_argument(
        "--viser-live-rgb-max-fps",
        type=float,
        default=5.0,
        help="Maximum update rate for the live RGB side panel.",
    )
    parser.add_argument(
        "--viser-record-dir",
        type=Path,
        default=None,
        help=(
            "Write per-batch Viser replay snapshots and mapper-only durations to this directory. "
            "This does not require --viser and records after mapper timing is measured."
        ),
    )
    parser.add_argument(
        "--viser-record-every-n",
        type=int,
        default=1,
        help="Record every Nth processed batch when --viser-record-dir is set.",
    )
    parser.add_argument(
        "--image-saving",
        action="store_true",
        help="Let StreamingMapper copy frames to image_store (default: off, references only).",
    )
    parser.add_argument(
        "--no-mask-observations",
        dest="mask_observations",
        action="store_false",
        default=True,
        help="Disable saving per-object 2D detector masks next to the scene_state.",
    )
    parser.add_argument(
        "--mask-observation-dir",
        type=Path,
        default=None,
        help="Override sidecar directory for per-object detector mask observations.",
    )
    parser.add_argument(
        "--mask-observation-max-per-object",
        type=int,
        default=256,
        help="Maximum saved mask observations referenced per object.",
    )
    parser.add_argument(
        "--covisibility",
        action="store_true",
        help="Enable covisibility graph updates (off by default, matching online).",
    )
    parser.add_argument(
        "--regions",
        action="store_true",
        help="Enable periodic region clustering/labeling in StreamingMapper.",
    )
    parser.add_argument(
        "--log-every",
        type=int,
        default=50,
        help="Log progress every N frames.",
    )
    parser.add_argument(
        "--logger-level",
        default="INFO",
        choices=("DEBUG", "INFO", "WARN", "ERROR"),
        help="ROS logger severity.",
    )
    parser.add_argument(
        "--offline-debug",
        action="store_true",
        help="Enable StreamingMapper's offline-debug capture (extra fields in debug_info).",
    )
    parser.add_argument(
        "--debug-trace-path",
        type=Path,
        default=None,
        help=(
            "If set, StreamingMapper writes a structured per-frame JSONL trace to this path. "
            "See scene_graph.debug.tracer / scripts/inspect_pipeline_trace.py."
        ),
    )
    parser.add_argument(
        "--timing-report-path",
        type=Path,
        default=None,
        help="Write an end-to-end JSON timing report for this offline run.",
    )
    parser.add_argument(
        "--preload-scene-state",
        action="store_true",
        help="Allow loading the default scene_state.pt under ~/.ros at startup. "
        "Off by default for offline runs so each scene starts empty.",
    )
    parser.add_argument(
        "--extra-param",
        action="append",
        default=[],
        metavar="KEY:=VALUE",
        help="Additional --ros-args -p override; may be repeated.",
    )
    return parser.parse_args(argv)


def build_ros_args(args: argparse.Namespace) -> List[str]:
    save_path = args.save_path.expanduser() if args.save_path else None
    params = {
        "logger_level": args.logger_level,
        "expected_batch": str(max(1, args.batch_size)),
        "viser_enabled": "true" if args.viser else "false",
        "viser_host": args.viser_host,
        "viser_port": str(int(args.viser_port)),
        "viser_live_rgb_enabled": "true" if args.viser_live_rgb else "false",
        "viser_live_rgb_max_side": str(max(1, int(args.viser_live_rgb_max_side))),
        "viser_live_rgb_max_fps": str(max(0.0, float(args.viser_live_rgb_max_fps))),
        "offline_debug": "true" if args.offline_debug else "false",
        "caption_enabled": "true" if args.caption else "false",
        "caption_batch_size": str(max(1, int(args.caption_batch_size))),
        "image_saving_enabled": "true" if args.image_saving else "false",
        "object_mask_saving_enabled": "true" if args.mask_observations else "false",
        "object_mask_observation_max_per_object": str(max(1, int(args.mask_observation_max_per_object))),
        "covisibility_enabled": "true" if args.covisibility else "false",
        "region_enabled": "true" if args.regions else "false",
        "publish_local_captions_enabled": "false",
        # Persist on destroy_node() so the driver just has to call it.
        "scene_state_save_on_shutdown": "true",
        # Captioning is async + bursty — the live-ROS default of 5s is far too
        # short for offline reconstruction; bump so the .pt actually contains
        # the captions the worker generated.
        "caption_drain_timeout_sec": "600.0",
    }
    # By default start each offline run from a clean state — pre-existing
    # scene_state.pt under ~/.ros pollutes per-scene reconstructions otherwise.
    # ROS2 -p override can't take an empty string, so use a YAML empty string.
    if not args.preload_scene_state:
        params["scene_state_load_path"] = "''"
    if save_path is not None:
        save_path.parent.mkdir(parents=True, exist_ok=True)
        params["scene_state_save_path"] = str(save_path)
        if args.mask_observations:
            mask_dir = args.mask_observation_dir or (save_path.parent / f"{save_path.stem}_masks")
            mask_dir.mkdir(parents=True, exist_ok=True)
            params["object_mask_storage_dir"] = str(mask_dir)
        # When image-saving is on, anchor the per-scene image_store next to
        # the save_path under a scene-stem-specific subdir; otherwise multiple
        # scenes saving to a shared default dir overwrite each other's frames.
        if args.image_saving:
            per_scene_image_dir = save_path.parent / f"{save_path.stem}_images"
            per_scene_image_dir.mkdir(parents=True, exist_ok=True)
            params["storage_image_dir"] = str(per_scene_image_dir)
    if args.debug_trace_path is not None:
        trace_path = args.debug_trace_path.expanduser()
        trace_path.parent.mkdir(parents=True, exist_ok=True)
        params["debug_trace_path"] = str(trace_path)
    ros_args: List[str] = ["--ros-args"]
    for key, value in params.items():
        ros_args.extend(["-p", f"{key}:={value}"])
    for extra in args.extra_param:
        ros_args.extend(["-p", str(extra)])
    return ros_args


def _wait_for_viser_streaming_resume(mapper: object) -> None:
    visualizer = getattr(mapper, "_viser_visualizer", None)
    if visualizer is None:
        return

    was_paused = False
    while bool(getattr(visualizer, "streaming_paused", False)):
        if not was_paused:
            LOGGER.info("Streaming paused from Viser.")
            was_paused = True
        time.sleep(0.05)

    if was_paused:
        LOGGER.info("Streaming resumed from Viser.")


def _finalize_before_viser_idle(mapper: object) -> None:
    """Best-effort final caption drain/save while keeping the mapper node alive."""
    caption_manager = getattr(mapper, "_caption_manager", None)
    if caption_manager is not None and getattr(caption_manager, "enabled", False):
        with contextlib.suppress(Exception):
            pending = getattr(mapper, "_pending_caption_indices", [])
            if pending:
                caption_manager.enqueue_objects(pending)
                pending.clear()
            caption_manager.wait_until_idle(
                timeout=float(getattr(mapper, "_caption_drain_timeout_sec", 0.0) or 0.0),
                poll_interval=0.05,
            )
            caption_manager.drain_results()

    visualizer = getattr(mapper, "_viser_visualizer", None)
    if visualizer is not None and getattr(visualizer, "enabled", False):
        with contextlib.suppress(Exception):
            poses = getattr(visualizer, "_latest_poses", []) or []
            visualizer.update([], [], [], poses, getattr(mapper, "_scene_state", {}))

    if bool(getattr(mapper, "_scene_state_save_on_shutdown", False)):
        with contextlib.suppress(Exception):
            lock = getattr(mapper, "_scene_state_save_lock", None)
            ctx = lock if lock is not None else contextlib.nullcontext()
            with ctx:
                saved_path = mapper._save_scene_state_now(reason="offline_done_keep_viser")
                LOGGER.info("Saved scene state before Viser idle: %s", saved_path)


def _keep_viser_alive(mapper: object, *, host: str, port: int) -> None:
    LOGGER.info("Mapping complete; keeping Viser alive at http://localhost:%d", int(port))
    LOGGER.info("Use Ctrl+C or terminate this process when finished inspecting.")
    while True:
        _wait_for_viser_streaming_resume(mapper)
        time.sleep(1.0)


def _record_viser_snapshot(
    recorder: object | None,
    mapper: object,
    *,
    duration_s: float,
    batch_size: int,
    warmup: bool,
) -> None:
    if recorder is None:
        return
    snapshot = None
    export_fn = getattr(mapper, "export_visualization_snapshot", None)
    if callable(export_fn):
        snapshot = export_fn()
    append_fn = getattr(recorder, "append", None)
    if callable(append_fn):
        append_fn(
            snapshot,
            duration_s=float(duration_s),
            batch_size=int(batch_size),
            warmup=bool(warmup),
        )


def make_frame_source(args: argparse.Namespace):
    if args.source == "sens":
        if args.sens_path is None:
            raise SystemExit("--sens-path is required when --source sens")
        from scene_graph.offline.frame_sources.sens import SensFrameSource

        return SensFrameSource(
            args.sens_path,
            camera=args.camera,
            stride=args.stride,
            start=args.start,
            end=None if args.end < 0 else args.end,
        )
    if args.source == "rosbag":
        if args.bag_path is None:
            raise SystemExit("--bag-path is required when --source rosbag")
        from scene_graph.offline.frame_sources.rosbag import RosbagFrameSource

        return RosbagFrameSource(
            args.bag_path,
            cameras=args.cameras,
            every_nth=args.every_nth,
        )
    if args.source == "npz":
        if args.npz_dir is None:
            raise SystemExit("--npz-dir is required when --source npz")
        from scene_graph.offline.frame_sources.npz import NPZFrameSource

        return NPZFrameSource(
            args.npz_dir,
            camera=args.camera,
            stride=args.stride,
            start=args.start,
            end=None if args.end < 0 else args.end,
            pattern=args.npz_pattern,
        )
    if args.source == "frames-json":
        if args.frames_json_dir is None:
            raise SystemExit("--frames-json-dir is required when --source frames-json")
        from scene_graph.offline.frame_sources.frames_json import FramesJsonFrameSource

        return FramesJsonFrameSource(
            args.frames_json_dir,
            cameras=args.frames_json_camera,
            stride=args.stride,
            start=args.start,
            end=None if args.end < 0 else args.end,
            nominal_hz=args.frames_json_nominal_hz,
            depth_clip_m=args.frames_json_depth_clip_m,
        )
    raise SystemExit(f"Unknown source: {args.source}")


def _duration_summary(values: list[float]) -> dict[str, float | int | None]:
    finite = sorted(float(value) for value in values if value >= 0.0)
    if not finite:
        return {"count": 0, "total": 0.0, "mean": None, "p50": None, "p95": None, "max": None}

    def percentile(fraction: float) -> float:
        position = fraction * (len(finite) - 1)
        left = int(position)
        right = min(left + 1, len(finite) - 1)
        alpha = position - left
        return finite[left] * (1.0 - alpha) + finite[right] * alpha

    total = float(sum(finite))
    return {
        "count": len(finite),
        "total": total,
        "mean": total / len(finite),
        "p50": percentile(0.50),
        "p95": percentile(0.95),
        "max": finite[-1],
    }


def _atomic_write_json(path: Path, value: object) -> None:
    """Durably replace a JSON report without exposing a partial file."""
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    try:
        with temporary.open("w", encoding="utf-8") as handle:
            json.dump(value, handle, indent=2)
            handle.write("\n")
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, path)
    finally:
        with contextlib.suppress(FileNotFoundError):
            temporary.unlink()


def _run_warmup(
    mapper: object,
    src_iter: object,
    *,
    warmup_frames: int,
    batch_size: int,
    recorder: object | None,
    batch_durations: list[float],
    progress: dict[str, int],
) -> int:
    """Process warmup frames; mapping/recording failures deliberately propagate."""
    warmup_batch: list = []
    warmup_count = 0
    progress["frames"] = 0
    while warmup_count < warmup_frames:
        _wait_for_viser_streaming_resume(mapper)
        try:
            item = next(src_iter)  # type: ignore[arg-type]
        except StopIteration:
            break
        warmup_batch.append(item)
        if len(warmup_batch) >= batch_size:
            t_batch_start = time.perf_counter()
            result = mapper._run_mapping_batch(warmup_batch)  # type: ignore[attr-defined]
            t_batch_end = time.perf_counter()
            batch_durations.append(t_batch_end - t_batch_start)
            if result is not None:
                _record_viser_snapshot(
                    recorder,
                    mapper,
                    duration_s=t_batch_end - t_batch_start,
                    batch_size=len(warmup_batch),
                    warmup=True,
                )
            warmup_count += len(warmup_batch)
            progress["frames"] = warmup_count
            warmup_batch = []
    if warmup_batch:
        t_batch_start = time.perf_counter()
        result = mapper._run_mapping_batch(warmup_batch)  # type: ignore[attr-defined]
        t_batch_end = time.perf_counter()
        batch_durations.append(t_batch_end - t_batch_start)
        if result is not None:
            _record_viser_snapshot(
                recorder,
                mapper,
                duration_s=t_batch_end - t_batch_start,
                batch_size=len(warmup_batch),
                warmup=True,
            )
        warmup_count += len(warmup_batch)
        progress["frames"] = warmup_count
    return warmup_count


def _destroy_mapper(mapper: object, *, run_succeeded: bool) -> object:
    """Release mapper resources, persisting state only for a successful run."""
    return mapper.destroy_node(save_scene_state=bool(run_succeeded))  # type: ignore[attr-defined]


def main(argv: Optional[List[str]] = None) -> int:
    logging.basicConfig(level=logging.INFO, format="[%(levelname)s] %(name)s: %(message)s")
    args = parse_args(argv)
    overall_started = time.perf_counter()
    runtime_init_started = time.perf_counter()

    import rclpy

    _prefer_repo_mapping_source()
    from mapping.nodes.streaming_mapper import StreamingMapper

    ros_args = build_ros_args(args)
    LOGGER.info("rclpy.init args: %s", " ".join(ros_args))
    rclpy.init(args=ros_args)
    runtime_init_seconds = time.perf_counter() - runtime_init_started

    source_init_started = time.perf_counter()
    source = make_frame_source(args)
    source_init_seconds = time.perf_counter() - source_init_started
    batch_size = max(1, args.batch_size)
    recorder = None
    if args.viser_record_dir is not None:
        from scene_graph.offline.viser_recording import ViserRecordingWriter

        scene_id = ""
        if args.save_path is not None:
            scene_id = args.save_path.expanduser().stem
        elif args.npz_dir is not None:
            scene_id = args.npz_dir.expanduser().name
        elif args.frames_json_dir is not None:
            scene_id = args.frames_json_dir.expanduser().name
        recorder = ViserRecordingWriter(
            args.viser_record_dir,
            every_n=max(1, int(args.viser_record_every_n)),
            scene_id=scene_id,
        )
        LOGGER.info("Viser replay recording enabled: %s", recorder.root)

    mapper = None
    mapper_init_seconds = 0.0
    warmup_seconds = 0.0
    steady_seconds = 0.0
    finalize_seconds = 0.0
    warmup_frames_consumed = 0
    frame_count = 0
    n_dropped = 0
    warmup_batch_seconds: list[float] = []
    steady_batch_seconds: list[float] = []
    run_status = "failed"
    run_error: str | None = None
    source_closed = False
    try:
        mapper_init_started = time.perf_counter()
        mapper = StreamingMapper()
        mapper_init_seconds = time.perf_counter() - mapper_init_started
        LOGGER.info(
            "StreamingMapper ready (caption=%s, viser=%s, image_saving=%s)",
            args.caption,
            args.viser,
            args.image_saving,
        )

        target_fps = float(args.target_fps)
        period_per_frame = (1.0 / target_fps) if target_fps > 0.0 else 0.0
        drop_when_late = bool(args.drop_when_late) and period_per_frame > 0.0
        warmup_frames = max(0, int(args.warmup_frames))

        batch: list = []
        frame_count = 0
        n_dropped = 0

        src_iter = iter(source)
        # Warm-up phase: process N frames before the wall-clock timer starts so
        # one-time costs (YOLOE compile, CUDA warmup, first vLLM tunnel round-
        # trip, SigLIP2 first inference) don't blow the deadline budget and
        # cascade-drop hundreds of subsequent frames.
        if warmup_frames > 0:
            t_warmup_start = time.perf_counter()
            warmup_progress = {"frames": 0}
            try:
                warmup_count = _run_warmup(
                    mapper,
                    src_iter,
                    warmup_frames=warmup_frames,
                    batch_size=batch_size,
                    recorder=recorder,
                    batch_durations=warmup_batch_seconds,
                    progress=warmup_progress,
                )
            finally:
                warmup_seconds = time.perf_counter() - t_warmup_start
                warmup_frames_consumed = int(warmup_progress["frames"])
            LOGGER.info(
                "Warmup: %d frames in %.1fs (timer now starts)",
                warmup_count,
                warmup_seconds,
            )

        # Reset frame counters after warmup so reported fps reflects steady-state
        # post-warmup performance only.
        warmup_frames_consumed = int(warmup_frames_consumed)
        frame_count = 0
        t_start = time.perf_counter()
        # Wall-clock deadline for the next batch to *finish* by.
        # Each batch consumes batch_size frames of schedule time.
        next_deadline = period_per_frame * batch_size if drop_when_late else None
        try:
            while True:
                _wait_for_viser_streaming_resume(mapper)
                try:
                    item = next(src_iter)
                except StopIteration:
                    break
                batch.append(item)
                if len(batch) >= batch_size:
                    t_batch_start = time.perf_counter()
                    result = mapper._run_mapping_batch(batch)
                    t_batch_end = time.perf_counter()
                    steady_batch_seconds.append(t_batch_end - t_batch_start)
                    frame_count += len(batch)
                    batch_size_actual = len(batch)
                    if result is not None:
                        _record_viser_snapshot(
                            recorder,
                            mapper,
                            duration_s=t_batch_end - t_batch_start,
                            batch_size=batch_size_actual,
                            warmup=False,
                        )
                    batch = []
                    if period_per_frame > 0.0:
                        if drop_when_late:
                            elapsed = t_batch_end - t_start
                            # If we missed the deadline, skip ahead in the source
                            # to catch up. Prefer the source's cheap ``skip(n)``
                            # method (advances cursor without decoding) if it
                            # provides one — full ``next()`` can cost 100s of ms
                            # on NPZ/jpeg-backed sources, which would entirely
                            # defeat the purpose of dropping.
                            if elapsed > next_deadline + period_per_frame * batch_size_actual:
                                periods_behind = int(
                                    (elapsed - next_deadline) // (period_per_frame * batch_size_actual)
                                )
                                want = periods_behind * batch_size_actual
                                skip_fn = getattr(source, "skip", None)
                                if callable(skip_fn) and want > 0:
                                    actually_skipped = int(skip_fn(want))
                                else:
                                    actually_skipped = 0
                                    while actually_skipped < want:
                                        try:
                                            next(src_iter)
                                        except StopIteration:
                                            break
                                        actually_skipped += 1
                                n_dropped += actually_skipped
                                next_deadline += periods_behind * period_per_frame * batch_size_actual
                            # Sleep until the deadline if we're ahead
                            sleep_for = next_deadline - (time.perf_counter() - t_start)
                            if sleep_for > 0.0:
                                time.sleep(sleep_for)
                            next_deadline += period_per_frame * batch_size_actual
                        else:
                            budget = period_per_frame * batch_size_actual
                            sleep_for = budget - (t_batch_end - t_batch_start)
                            if sleep_for > 0.0:
                                time.sleep(sleep_for)
                    if args.log_every > 0 and frame_count % args.log_every == 0:
                        dt = time.perf_counter() - t_start
                        drop_msg = f" dropped={n_dropped}" if drop_when_late else ""
                        LOGGER.info(
                            "Processed %d frames (%.2f fps avg)%s",
                            frame_count,
                            frame_count / max(dt, 1e-9),
                            drop_msg,
                        )
            if batch:
                _wait_for_viser_streaming_resume(mapper)
                t_batch_start = time.perf_counter()
                result = mapper._run_mapping_batch(batch)
                t_batch_end = time.perf_counter()
                steady_batch_seconds.append(t_batch_end - t_batch_start)
                if result is not None:
                    _record_viser_snapshot(
                        recorder,
                        mapper,
                        duration_s=t_batch_end - t_batch_start,
                        batch_size=len(batch),
                        warmup=False,
                    )
                if period_per_frame > 0.0 and not drop_when_late:
                    budget = period_per_frame * len(batch)
                    sleep_for = budget - (t_batch_end - t_batch_start)
                    if sleep_for > 0.0:
                        time.sleep(sleep_for)
                frame_count += len(batch)
        finally:
            source.close()
            source_closed = True

        dt = time.perf_counter() - t_start
        steady_seconds = dt
        drop_msg = f" dropped={n_dropped}" if drop_when_late else ""
        LOGGER.info("Done. %d frames in %.1fs (%.2f fps avg)%s", frame_count, dt, frame_count / max(dt, 1e-9), drop_msg)
        if recorder is not None:
            recorder.close()
            LOGGER.info("Viser replay manifest written: %s", recorder.manifest_path)
        if args.keep_viser_after_run and args.viser:
            _finalize_before_viser_idle(mapper)
            _keep_viser_alive(mapper, host=args.viser_host, port=args.viser_port)
        run_status = "completed"
    except BaseException as exc:
        run_error = f"{type(exc).__name__}: {exc}"
        raise
    finally:
        finalize_started = time.perf_counter()
        primary_failure = run_error is not None
        cleanup_error: BaseException | None = None
        if recorder is not None:
            with contextlib.suppress(Exception):
                recorder.close()
        if not source_closed:
            with contextlib.suppress(Exception):
                source.close()
        if mapper is not None:
            # Always release workers/tracers. A failed or interrupted run must
            # never replace a previously valid scene_state with partial data.
            try:
                _destroy_mapper(mapper, run_succeeded=(run_status == "completed"))
            except BaseException as exc:
                cleanup_error = exc
                LOGGER.exception("StreamingMapper cleanup failed")
                if run_error is None:
                    run_status = "failed"
                    run_error = f"{type(exc).__name__}: {exc}"
        try:
            rclpy.shutdown()
        except BaseException as exc:
            LOGGER.exception("rclpy shutdown failed")
            if cleanup_error is None:
                cleanup_error = exc
            if run_error is None:
                run_status = "failed"
                run_error = f"{type(exc).__name__}: {exc}"
        finalize_seconds = time.perf_counter() - finalize_started
        if args.timing_report_path is not None:
            timing_path = args.timing_report_path.expanduser()
            timing_path.parent.mkdir(parents=True, exist_ok=True)
            total_seconds = time.perf_counter() - overall_started
            total_frames = int(warmup_frames_consumed + frame_count)
            report = {
                "schema": "farm.offline-timing.v1",
                "created_unix_s": time.time(),
                "status": run_status,
                "error": run_error,
                "configuration": {
                    "source": args.source,
                    "frames_json_dir": str(args.frames_json_dir.expanduser()) if args.frames_json_dir else None,
                    "batch_size": batch_size,
                    "target_fps": float(args.target_fps),
                    "warmup_frames_requested": int(args.warmup_frames),
                    "caption_enabled": bool(args.caption),
                    "covisibility_enabled": bool(args.covisibility),
                    "regions_enabled": bool(args.regions),
                    "extra_params": list(args.extra_param),
                },
                "frames": {
                    "warmup": int(warmup_frames_consumed),
                    "steady": int(frame_count),
                    "total": total_frames,
                    "dropped": int(n_dropped),
                },
                "stages_seconds": {
                    "runtime_initialization": runtime_init_seconds,
                    "frame_source_initialization": source_init_seconds,
                    "mapper_and_model_initialization": mapper_init_seconds,
                    "warmup_mapping": warmup_seconds,
                    "steady_mapping": steady_seconds,
                    "finalize_and_save": finalize_seconds,
                    "end_to_end": total_seconds,
                },
                "batches_seconds": {
                    "warmup": _duration_summary(warmup_batch_seconds),
                    "steady": _duration_summary(steady_batch_seconds),
                },
                "throughput_views_per_second": {
                    "steady": frame_count / max(steady_seconds, 1.0e-9),
                    "end_to_end": total_frames / max(total_seconds, 1.0e-9),
                },
            }
            _atomic_write_json(timing_path, report)
            LOGGER.info("Timing report written: %s", timing_path)
        if cleanup_error is not None and not primary_failure:
            raise cleanup_error
    return 0


if __name__ == "__main__":
    sys.exit(main())
