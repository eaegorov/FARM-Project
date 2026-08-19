from __future__ import annotations

import csv
import json
import os
import shutil
import signal
import socket
import subprocess
import time
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import IO, Any, Mapping, Sequence
from urllib.error import HTTPError, URLError
from urllib.request import urlopen


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")


def atomic_write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    with temporary.open("w", encoding="utf-8") as handle:
        json.dump(value, handle, ensure_ascii=False, indent=2, sort_keys=True)
        handle.write("\n")
        handle.flush()
        os.fsync(handle.fileno())
    os.replace(temporary, path)


def atomic_write_text(path: Path, value: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    with temporary.open("w", encoding="utf-8") as handle:
        handle.write(value)
        handle.flush()
        os.fsync(handle.fileno())
    os.replace(temporary, path)


def process_start_token(pid: int) -> str | None:
    """Return Linux process start ticks, used to defend against PID reuse."""

    try:
        value = Path(f"/proc/{pid}/stat").read_text(encoding="utf-8")
        tail = value[value.rfind(")") + 2 :].split()
        return tail[19]
    except (FileNotFoundError, PermissionError, IndexError, OSError):
        return None


def process_identity_matches(pid: int, token: str | None) -> bool:
    return pid > 1 and token is not None and process_start_token(pid) == str(token)


def _process_children(pid: int) -> set[int]:
    found: set[int] = set()
    pending = [pid]
    while pending:
        parent = pending.pop()
        try:
            raw = Path(f"/proc/{parent}/task/{parent}/children").read_text(encoding="utf-8").strip()
        except (FileNotFoundError, PermissionError, OSError):
            continue
        for value in raw.split():
            try:
                child = int(value)
            except ValueError:
                continue
            if child not in found:
                found.add(child)
                pending.append(child)
    return found


def _rss_bytes(pid: int) -> int:
    try:
        for line in Path(f"/proc/{pid}/status").read_text(encoding="utf-8").splitlines():
            if line.startswith("VmRSS:"):
                return int(line.split()[1]) * 1024
    except (FileNotFoundError, PermissionError, OSError, ValueError, IndexError):
        pass
    return 0


def _gpu_usage_mb(pids: set[int]) -> dict[str, int]:
    executable = shutil.which("nvidia-smi")
    if not executable or not pids:
        return {}
    try:
        result = subprocess.run(
            [
                executable,
                "--query-compute-apps=pid,used_gpu_memory,gpu_uuid",
                "--format=csv,noheader,nounits",
            ],
            stdout=subprocess.PIPE,
            stderr=subprocess.DEVNULL,
            text=True,
            timeout=5,
            check=False,
        )
    except (OSError, subprocess.TimeoutExpired):
        return {}
    if result.returncode != 0:
        return {}
    usage: dict[str, int] = {}
    for row in csv.reader(result.stdout.splitlines(), skipinitialspace=True):
        if len(row) != 3:
            continue
        try:
            pid = int(row[0].strip())
            used_mb = int(float(row[1].strip()))
        except ValueError:
            continue
        if pid in pids:
            uuid = row[2].strip()
            usage[uuid] = usage.get(uuid, 0) + used_mb
    return usage


def _gpu_device_memory_mb() -> dict[str, dict[str, int | str]]:
    """Return whole-device memory counters.

    Docker GPU processes are descendants of the Docker daemon, not of the
    monitored wrapper.  These counters therefore complement, and are kept
    explicitly separate from, PID-attributed telemetry.
    """

    executable = shutil.which("nvidia-smi")
    if not executable:
        return {}
    try:
        result = subprocess.run(
            [
                executable,
                "--query-gpu=index,uuid,memory.used,memory.free,memory.total",
                "--format=csv,noheader,nounits",
            ],
            stdout=subprocess.PIPE,
            stderr=subprocess.DEVNULL,
            text=True,
            timeout=5,
            check=False,
        )
    except (OSError, subprocess.TimeoutExpired):
        return {}
    if result.returncode != 0:
        return {}
    devices: dict[str, dict[str, int | str]] = {}
    for row in csv.reader(result.stdout.splitlines(), skipinitialspace=True):
        if len(row) != 5:
            continue
        try:
            index = int(row[0].strip())
            used = int(float(row[2].strip()))
            free = int(float(row[3].strip()))
            total = int(float(row[4].strip()))
        except ValueError:
            continue
        uuid = row[1].strip()
        devices[uuid] = {
            "index": index,
            "uuid": uuid,
            "used_mb": used,
            "free_mb": free,
            "total_mb": total,
        }
    return devices


@dataclass
class TelemetrySummary:
    sample_count: int = 0
    rss_tree_peak_bytes: int = 0
    gpu_peak_total_mb: int = 0
    gpu_peak_by_uuid_mb: dict[str, int] = field(default_factory=dict)
    gpu_device_baseline_used_by_uuid_mb: dict[str, int] = field(default_factory=dict)
    gpu_device_peak_used_by_uuid_mb: dict[str, int] = field(default_factory=dict)
    gpu_device_min_free_by_uuid_mb: dict[str, int] = field(default_factory=dict)
    gpu_device_total_by_uuid_mb: dict[str, int] = field(default_factory=dict)

    def set_device_baseline(self, devices: Mapping[str, Mapping[str, int | str]]) -> None:
        for uuid, value in devices.items():
            used = int(value["used_mb"])
            self.gpu_device_baseline_used_by_uuid_mb[uuid] = used
            self.gpu_device_peak_used_by_uuid_mb[uuid] = used
            self.gpu_device_min_free_by_uuid_mb[uuid] = int(value["free_mb"])
            self.gpu_device_total_by_uuid_mb[uuid] = int(value["total_mb"])

    def update(
        self,
        rss_bytes: int,
        gpu_usage: Mapping[str, int],
        devices: Mapping[str, Mapping[str, int | str]],
    ) -> None:
        self.sample_count += 1
        self.rss_tree_peak_bytes = max(self.rss_tree_peak_bytes, rss_bytes)
        total = sum(gpu_usage.values())
        self.gpu_peak_total_mb = max(self.gpu_peak_total_mb, total)
        for uuid, value in gpu_usage.items():
            self.gpu_peak_by_uuid_mb[uuid] = max(self.gpu_peak_by_uuid_mb.get(uuid, 0), value)
        for uuid, value in devices.items():
            used = int(value["used_mb"])
            free = int(value["free_mb"])
            if uuid not in self.gpu_device_baseline_used_by_uuid_mb:
                self.gpu_device_baseline_used_by_uuid_mb[uuid] = used
            self.gpu_device_peak_used_by_uuid_mb[uuid] = max(
                self.gpu_device_peak_used_by_uuid_mb.get(uuid, used), used
            )
            self.gpu_device_min_free_by_uuid_mb[uuid] = min(
                self.gpu_device_min_free_by_uuid_mb.get(uuid, free), free
            )
            self.gpu_device_total_by_uuid_mb[uuid] = int(value["total_mb"])

    def as_dict(self) -> dict[str, Any]:
        return {
            "sample_count": self.sample_count,
            "rss_tree_peak_bytes": self.rss_tree_peak_bytes,
            "rss_tree_peak_gib": round(self.rss_tree_peak_bytes / (1024**3), 4),
            "gpu_pid_attributed_peak_total_mb": self.gpu_peak_total_mb,
            "gpu_pid_attributed_peak_by_uuid_mb": dict(sorted(self.gpu_peak_by_uuid_mb.items())),
            "gpu_device_baseline_used_by_uuid_mb": dict(sorted(self.gpu_device_baseline_used_by_uuid_mb.items())),
            "gpu_device_peak_used_by_uuid_mb": dict(sorted(self.gpu_device_peak_used_by_uuid_mb.items())),
            "gpu_device_min_free_by_uuid_mb": dict(sorted(self.gpu_device_min_free_by_uuid_mb.items())),
            "gpu_device_total_by_uuid_mb": dict(sorted(self.gpu_device_total_by_uuid_mb.items())),
            "gpu_device_peak_delta_by_uuid_mb": {
                uuid: max(0, peak - self.gpu_device_baseline_used_by_uuid_mb.get(uuid, peak))
                for uuid, peak in sorted(self.gpu_device_peak_used_by_uuid_mb.items())
            },
        }


@dataclass
class ProcessResult:
    returncode: int
    started_at: str
    finished_at: str
    duration_seconds: float
    pid: int
    process_start_token: str | None
    telemetry: TelemetrySummary
    timed_out: bool = False
    interrupted: bool = False


def _terminate_group(process: subprocess.Popen[Any], grace_seconds: float = 10.0) -> None:
    if process.poll() is not None:
        return
    try:
        os.killpg(process.pid, signal.SIGTERM)
    except (ProcessLookupError, PermissionError, OSError):
        process.terminate()
    try:
        process.wait(timeout=grace_seconds)
        return
    except subprocess.TimeoutExpired:
        pass
    try:
        os.killpg(process.pid, signal.SIGKILL)
    except (ProcessLookupError, PermissionError, OSError):
        process.kill()
    process.wait()


def run_monitored_process(
    command: Sequence[str],
    *,
    cwd: Path,
    env: Mapping[str, str],
    stdout_path: Path,
    stderr_path: Path,
    telemetry_path: Path,
    interval_seconds: float,
    timeout_seconds: float | None,
) -> ProcessResult:
    if not command:
        raise ValueError("command must not be empty")
    stdout_path.parent.mkdir(parents=True, exist_ok=True)
    telemetry_path.parent.mkdir(parents=True, exist_ok=True)
    started_at = utc_now()
    start_monotonic = time.monotonic()
    timed_out = False
    interrupted = False
    summary = TelemetrySummary()
    summary.set_device_baseline(_gpu_device_memory_mb())
    with stdout_path.open("ab", buffering=0) as stdout_handle, stderr_path.open("ab", buffering=0) as stderr_handle:
        process = subprocess.Popen(
            list(command),
            cwd=str(cwd),
            env=dict(env),
            stdin=subprocess.DEVNULL,
            stdout=stdout_handle,
            stderr=stderr_handle,
            shell=False,
            start_new_session=True,
        )
        token = process_start_token(process.pid)
        try:
            with telemetry_path.open("a", encoding="utf-8") as telemetry_handle:
                while process.poll() is None:
                    elapsed = time.monotonic() - start_monotonic
                    pids = {process.pid, *_process_children(process.pid)}
                    rss = sum(_rss_bytes(pid) for pid in pids)
                    gpu = _gpu_usage_mb(pids)
                    devices = _gpu_device_memory_mb()
                    summary.update(rss, gpu, devices)
                    telemetry_handle.write(
                        json.dumps(
                            {
                                "timestamp": utc_now(),
                                "elapsed_seconds": round(elapsed, 3),
                                "process_count": len(pids),
                                "rss_tree_bytes": rss,
                                "gpu_pid_attributed_memory_mb": gpu,
                                "gpu_device_total_memory": devices,
                            },
                            sort_keys=True,
                        )
                        + "\n"
                    )
                    telemetry_handle.flush()
                    if timeout_seconds is not None and elapsed >= timeout_seconds:
                        timed_out = True
                        _terminate_group(process)
                        break
                    time.sleep(interval_seconds)
        except KeyboardInterrupt:
            interrupted = True
            _terminate_group(process)
        returncode = process.wait()
    duration = time.monotonic() - start_monotonic
    return ProcessResult(
        returncode=returncode,
        started_at=started_at,
        finished_at=utc_now(),
        duration_seconds=duration,
        pid=process.pid,
        process_start_token=token,
        telemetry=summary,
        timed_out=timed_out,
        interrupted=interrupted,
    )


def port_is_available(host: str, port: int) -> bool:
    bind_host = host
    if host in {"0.0.0.0", "::"}:
        bind_host = host
    family = socket.AF_INET6 if ":" in bind_host else socket.AF_INET
    try:
        with socket.socket(family, socket.SOCK_STREAM) as sock:
            sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
            sock.bind((bind_host, port))
    except OSError:
        return False
    return True


def wait_for_http(url: str, process: subprocess.Popen[Any], timeout_seconds: float) -> bool:
    deadline = time.monotonic() + timeout_seconds
    while time.monotonic() < deadline:
        if process.poll() is not None:
            return False
        try:
            with urlopen(url, timeout=1.0) as response:
                if 200 <= response.status < 500:
                    return True
        except (HTTPError, URLError, TimeoutError, OSError):
            pass
        time.sleep(0.25)
    return False


def open_detached_process(
    command: Sequence[str],
    *,
    cwd: Path,
    env: Mapping[str, str],
    stdout: IO[bytes],
    stderr: IO[bytes],
) -> subprocess.Popen[Any]:
    return subprocess.Popen(
        list(command),
        cwd=str(cwd),
        env=dict(env),
        stdin=subprocess.DEVNULL,
        stdout=stdout,
        stderr=stderr,
        shell=False,
        start_new_session=True,
        close_fds=True,
    )
