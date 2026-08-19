from __future__ import annotations

import os
import stat
import subprocess
from pathlib import Path

import pytest

from scripts.farm_standard_stage import Context


ROOT = Path(__file__).resolve().parents[1]


def _mode(path: Path) -> int:
    return stat.S_IMODE(path.stat().st_mode)


@pytest.mark.parametrize(
    ("entrypoint", "extra_env", "command"),
    [
        ("entrypoint.sh", {}, ["/bin/true"]),
        ("python-entrypoint.sh", {"FARM_PIPELINE_PYTHON": "/bin/true"}, []),
    ],
)
def test_run_permission_normalization_preserves_source_snapshot_modes(
    tmp_path: Path,
    entrypoint: str,
    extra_env: dict[str, str],
    command: list[str],
) -> None:
    # A symlink plus whitespace exercises canonical path handling without
    # coupling the contract to a particular container mount spelling.
    run_dir = tmp_path / "run with spaces"
    snapshot = run_dir / "config/source_snapshot/FARM-Project"
    ordinary_dir = run_dir / "mapping/output"
    snapshot.mkdir(parents=True)
    ordinary_dir.mkdir(parents=True)
    snapshot_file = snapshot / "tool.py"
    snapshot_file.write_text("print('frozen')\n", encoding="utf-8")
    ordinary_file = ordinary_dir / "result.json"
    ordinary_file.write_text("{}\n", encoding="utf-8")
    os.chmod(snapshot, 0o755)
    os.chmod(snapshot_file, 0o755)
    os.chmod(ordinary_dir, 0o755)
    os.chmod(ordinary_file, 0o644)
    run_link = tmp_path / "run-link"
    run_link.symlink_to(run_dir, target_is_directory=True)

    environment = os.environ.copy()
    environment.update(extra_env)
    environment["FARM_RUN_DIR"] = f"{run_link}/"
    completed = subprocess.run(
        ["/bin/bash", str(ROOT / "docker" / entrypoint), *command],
        cwd=ROOT,
        env=environment,
        check=False,
        capture_output=True,
        text=True,
    )

    assert completed.returncode == 0, completed.stderr
    assert _mode(snapshot) == 0o755
    assert _mode(snapshot_file) == 0o755
    assert _mode(ordinary_dir) == 0o775
    assert _mode(ordinary_file) == 0o664


@pytest.mark.parametrize("entrypoint", ["entrypoint.sh", "python-entrypoint.sh"])
def test_entrypoint_permission_contract_prunes_exact_snapshot_tree(
    entrypoint: str,
) -> None:
    source = (ROOT / "docker" / entrypoint).read_text(encoding="utf-8")
    assert 'pwd -P' in source
    assert '[ "$run_dir" != "/" ]' in source
    assert 'source_snapshot_dir="$run_dir/config/source_snapshot"' in source
    assert '-path "$source_snapshot_dir"' in source
    assert '-path "$source_snapshot_dir/*"' in source
    assert ') -prune -o' in source


def test_standard_stage_permission_normalization_preserves_snapshot(
    tmp_path: Path,
) -> None:
    run_dir = tmp_path / "run"
    snapshot = run_dir / "config/source_snapshot/FARM-Project"
    ordinary = run_dir / "mapping"
    snapshot.mkdir(parents=True)
    ordinary.mkdir(parents=True)
    snapshot_file = snapshot / "tool.py"
    ordinary_file = ordinary / "result.json"
    snapshot_file.write_text("print('frozen')\n", encoding="utf-8")
    ordinary_file.write_text("{}\n", encoding="utf-8")
    os.chmod(snapshot, 0o755)
    os.chmod(snapshot_file, 0o755)
    os.chmod(ordinary, 0o755)
    os.chmod(ordinary_file, 0o644)

    context = object.__new__(Context)
    context.run_dir = run_dir
    context.prepare_main_mount()

    assert _mode(snapshot) == 0o755
    assert _mode(snapshot_file) == 0o755
    assert _mode(ordinary) == 0o775
    assert _mode(ordinary_file) == 0o664
