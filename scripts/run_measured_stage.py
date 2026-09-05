#!/usr/bin/env python3
"""Run one stage while sampling wall time, process RSS, and GPU memory."""

from __future__ import annotations

import argparse
import datetime as dt
import json
import os
import subprocess
import sys
import time
from pathlib import Path


def _gpu_used_mib() -> int | None:
    try:
        output = subprocess.check_output(
            ["nvidia-smi", "--query-gpu=memory.used", "--format=csv,noheader,nounits"],
            text=True,
            stderr=subprocess.DEVNULL,
            timeout=5,
        )
        return sum(int(line.strip()) for line in output.splitlines() if line.strip())
    except (OSError, ValueError, subprocess.SubprocessError):
        return None


def _parse_gpu_process_table(output: str, selected_pids: set[int]) -> int:
    """Sum per-process NVIDIA memory for an exact PID allowlist."""

    total = 0
    for line in output.splitlines():
        fields = [value.strip() for value in line.split(",")]
        if len(fields) != 2:
            continue
        try:
            pid = int(fields[0])
            used = int(fields[1])
        except ValueError:
            continue
        if pid in selected_pids:
            total += used
    return total


def _gpu_process_used_mib(selected_pids: set[int]) -> int | None:
    try:
        output = subprocess.check_output(
            [
                "nvidia-smi",
                "--query-compute-apps=pid,used_gpu_memory",
                "--format=csv,noheader,nounits",
            ],
            text=True,
            stderr=subprocess.DEVNULL,
            timeout=5,
        )
        return _parse_gpu_process_table(output, selected_pids)
    except (OSError, ValueError, subprocess.SubprocessError):
        return None


def _descendants(root_pid: int) -> set[int]:
    parents: dict[int, int] = {}
    proc = Path("/proc")
    for entry in proc.iterdir():
        if not entry.name.isdigit():
            continue
        try:
            fields = (entry / "stat").read_text().split()
            parents[int(entry.name)] = int(fields[3])
        except (OSError, ValueError, IndexError):
            continue
    selected = {root_pid}
    changed = True
    while changed:
        changed = False
        for pid, parent in parents.items():
            if parent in selected and pid not in selected:
                selected.add(pid)
                changed = True
    return selected


def _rss_kib(root_pid: int) -> int:
    total = 0
    for pid in _descendants(root_pid):
        try:
            for line in Path(f"/proc/{pid}/status").read_text().splitlines():
                if line.startswith("VmRSS:"):
                    total += int(line.split()[1])
                    break
        except (OSError, ValueError, IndexError):
            continue
    return total


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--name", required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--stdout-log", type=Path, required=True)
    parser.add_argument("--poll-seconds", type=float, default=0.25)
    parser.add_argument("command", nargs=argparse.REMAINDER)
    args = parser.parse_args()
    command = list(args.command)
    if command[:1] == ["--"]:
        command = command[1:]
    if not command:
        parser.error("a command is required after --")
    if args.poll_seconds <= 0.0:
        parser.error("--poll-seconds must be positive")
    output = args.output.expanduser().resolve()
    log = args.stdout_log.expanduser().resolve()
    output.parent.mkdir(parents=True, exist_ok=True)
    log.parent.mkdir(parents=True, exist_ok=True)
    started_utc = dt.datetime.now(dt.timezone.utc).isoformat()
    started = time.perf_counter()
    baseline_gpu = _gpu_used_mib()
    peak_gpu = baseline_gpu
    peak_process_gpu: int | None = 0
    process_gpu_samples = 0
    peak_rss = 0
    samples = 0
    with log.open("wb") as stream:
        process = subprocess.Popen(command, stdout=stream, stderr=subprocess.STDOUT)
        while process.poll() is None:
            gpu = _gpu_used_mib()
            if gpu is not None:
                peak_gpu = gpu if peak_gpu is None else max(peak_gpu, gpu)
            descendants = _descendants(process.pid)
            process_gpu = _gpu_process_used_mib(descendants)
            if process_gpu is not None:
                peak_process_gpu = (
                    process_gpu
                    if peak_process_gpu is None
                    else max(peak_process_gpu, process_gpu)
                )
                process_gpu_samples += 1
            peak_rss = max(peak_rss, _rss_kib(process.pid))
            samples += 1
            time.sleep(float(args.poll_seconds))
        returncode = int(process.wait())
    final_gpu = _gpu_used_mib()
    duration = time.perf_counter() - started
    report = {
        "schema": "farm.measured-stage.v1",
        "name": args.name,
        "started_utc": started_utc,
        "finished_utc": dt.datetime.now(dt.timezone.utc).isoformat(),
        "duration_seconds": duration,
        "returncode": returncode,
        "command": command,
        "stdout_log": str(log),
        "samples": samples,
        "poll_seconds": float(args.poll_seconds),
        "peak_process_tree_rss_kib": peak_rss,
        "gpu_memory_used_mib_before": baseline_gpu,
        "gpu_memory_used_mib_peak": peak_gpu,
        "gpu_memory_used_mib_peak_delta": (
            max(0, peak_gpu - baseline_gpu)
            if peak_gpu is not None and baseline_gpu is not None
            else None
        ),
        "gpu_memory_used_mib_after": final_gpu,
        "gpu_process_tree_memory_used_mib_peak": peak_process_gpu,
        "gpu_process_tree_samples": process_gpu_samples,
        "gpu_metric_scope": {
            "whole_device": (
                "all visible GPUs; use peak delta only under exclusive sequential execution"
            ),
            "process_tree": (
                "nvidia-smi compute-app PIDs descending from the measured command; "
                "may be unavailable when the command is only a Docker client"
            ),
            "preferred_in_process": (
                "torch.cuda.max_memory_allocated/reserved emitted by the model runner"
            ),
        },
    }
    output.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(report, indent=2))
    return returncode


if __name__ == "__main__":
    raise SystemExit(main())
