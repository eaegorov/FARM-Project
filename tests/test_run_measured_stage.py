from __future__ import annotations

import json
import os
import sys

from scripts.run_measured_stage import (
    _descendants,
    _parse_gpu_process_table,
    _rss_kib,
    main,
)


def test_gpu_process_parser_uses_only_selected_valid_rows() -> None:
    table = """101, 200
202, 75 MiB
malformed
303, 9
101, 5
"""
    assert _parse_gpu_process_table(table, {101, 303}) == 214
    assert _parse_gpu_process_table(table, set()) == 0


def test_process_tree_sampling_contains_self_and_positive_rss() -> None:
    assert os.getpid() in _descendants(os.getpid())
    assert _rss_kib(os.getpid()) > 0


def test_cpu_stage_writes_reproducible_measurement_contract(
    tmp_path, monkeypatch
) -> None:
    report_path = tmp_path / "measurement.json"
    log_path = tmp_path / "stdout.log"
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "run_measured_stage.py",
            "--name",
            "cpu-smoke",
            "--output",
            str(report_path),
            "--stdout-log",
            str(log_path),
            "--poll-seconds",
            "0.01",
            "--",
            sys.executable,
            "-c",
            "import time; print('ok', flush=True); time.sleep(0.04)",
        ],
    )
    assert main() == 0
    report = json.loads(report_path.read_text(encoding="utf-8"))
    assert report["schema"] == "farm.measured-stage.v1"
    assert report["name"] == "cpu-smoke"
    assert report["returncode"] == 0
    assert report["duration_seconds"] > 0.0
    assert report["samples"] >= 1
    assert report["peak_process_tree_rss_kib"] > 0
    assert "preferred_in_process" in report["gpu_metric_scope"]
    assert log_path.read_text(encoding="utf-8").strip() == "ok"
