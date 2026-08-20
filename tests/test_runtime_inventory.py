from __future__ import annotations

import ast
import importlib
import sys
from pathlib import Path

import pytest
import yaml


from scripts.farm_runtime_inventory import (
    InventoryError,
    canonical_name,
    compare_inventory,
    parse_inventory,
)


ROOT = Path(__file__).resolve().parents[1]
CONTROL_PLANE_EXECUTION_FILES = (
    ROOT / "scripts/farm_standard_stage.py",
    ROOT / "scripts/farm_preflight.py",
    ROOT / "src/farm_pipeline/preflight.py",
    ROOT / "src/farm_pipeline/scene_config.py",
    ROOT / "scripts/select_colmap_keyframes.py",
    ROOT / "scripts/build_farm_run_report.py",
)
CONTROL_IMPORT_DISTRIBUTIONS = {
    "PIL": "pillow",
    "cv2": "opencv-python",
    "matplotlib": "matplotlib",
    "numpy": "numpy",
    "pycolmap": "pycolmap",
    "scipy": "scipy",
    "torch": "torch",
    "yaml": "pyyaml",
}


def test_parse_inventory_accepts_comments_and_index_directive(tmp_path: Path) -> None:
    lock = tmp_path / "runtime.txt"
    lock.write_text(
        "# exact\n--extra-index-url https://example.invalid/simple\n"
        "Torch==2.5.1+cu121\nPy_YAML==6.0.3\n",
        encoding="utf-8",
    )
    assert parse_inventory(lock) == {
        "torch": "2.5.1+cu121",
        "py-yaml": "6.0.3",
    }


@pytest.mark.parametrize(
    "entry",
    ("torch>=2", "torch", "torch @ file:///tmp/wheel.whl", "torch==2; python_version>'3'"),
)
def test_parse_inventory_rejects_non_exact_entries(tmp_path: Path, entry: str) -> None:
    lock = tmp_path / "runtime.txt"
    lock.write_text(entry + "\n", encoding="utf-8")
    with pytest.raises(InventoryError):
        parse_inventory(lock)


def test_compare_inventory_is_fail_closed_for_missing_and_mismatch() -> None:
    result = compare_inventory(
        {"torch": "2.5.1", "numpy": "1.26.0", "pyyaml": "6.0.3"},
        {"torch": "2.5.1", "numpy": "2.0.0"},
    )
    assert result["status"] == "fail"
    assert result["missing"] == ["pyyaml"]
    assert result["mismatched"] == {
        "numpy": {"expected": "1.26.0", "installed": "2.0.0"}
    }


def test_committed_runtime_profiles_are_exact_and_prep_closure_is_present() -> None:
    control = parse_inventory(ROOT / "requirements/control-plane.lock.txt")
    bridge_control = parse_inventory(ROOT / "requirements/bridge-control.lock.txt")
    prep = parse_inventory(ROOT / "requirements/prep-runtime.lock.txt")
    main = parse_inventory(ROOT / "requirements/main-runtime.observed.txt")
    assert control == {
        "huggingface-hub": "1.26.0",
        "matplotlib": "3.10.9",
        "numpy": "2.4.6",
        "opencv-python": "4.11.0.86",
        "pillow": "11.0.0",
        "pycolmap": "3.11.1",
        "pyyaml": "6.0.2",
        "scipy": "1.15.1",
        "torch": "2.12.0",
    }
    assert bridge_control == {
        "numpy": "1.26.0",
        "opencv-python": "4.11.0.86",
        "pyyaml": "6.0.2",
        "scipy": "1.16.3",
        "torch": "2.12.0",
    }
    assert {
        "torch", "torchvision", "numpy", "pycolmap", "gsplat", "plyfile",
        "opencv-python", "pillow", "pyyaml", "scipy", "matplotlib",
    } <= set(prep)
    assert {"torch", "numpy", "transformers", "ultralytics", "viser"} <= set(main)


def test_control_plane_lock_covers_static_host_execution_imports() -> None:
    roots: set[str] = set()
    for path in CONTROL_PLANE_EXECUTION_FILES:
        tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
        for node in ast.walk(tree):
            if isinstance(node, ast.Import):
                roots.update(alias.name.split(".", 1)[0] for alias in node.names)
            elif isinstance(node, ast.ImportFrom) and node.level == 0 and node.module:
                roots.add(node.module.split(".", 1)[0])
    third_party = roots - set(sys.stdlib_module_names) - {
        "__future__", "farm_pipeline", "farm_runtime",
    }
    assert third_party == set(CONTROL_IMPORT_DISTRIBUTIONS)
    control = parse_inventory(ROOT / "requirements/control-plane.lock.txt")
    assert set(CONTROL_IMPORT_DISTRIBUTIONS.values()) <= set(control)


@pytest.mark.parametrize("module_name", sorted(CONTROL_IMPORT_DISTRIBUTIONS))
def test_control_plane_host_execution_import_smoke(module_name: str) -> None:
    assert importlib.import_module(module_name) is not None


def test_docker_contract_has_no_machine_specific_paths_or_viewer_port() -> None:
    compose_path = ROOT / "docker/compose.pipeline.yml"
    compose_text = compose_path.read_text(encoding="utf-8")
    compose = yaml.safe_load(compose_text)
    assert set(compose["services"]) == {
        "scene-preflight", "prep-runtime-validate", "prep", "main-runtime-validate"
    }
    assert "/home/splatica" not in compose_text
    assert "/var/run/docker.sock" not in compose_text
    assert "8080" not in compose_text
    assert "network_mode: host" not in compose_text
    dockerfile = (ROOT / "docker/Dockerfile.prep").read_text(encoding="utf-8")
    assert "requirements/prep-runtime.lock.txt" in dockerfile
    assert "farm_runtime_inventory.py validate" in dockerfile
    assert "/opt/conda/envs/rest3d/bin/python" in dockerfile
    assert "TORCH_EXTENSIONS_DIR=/opt/farm-torch-extensions" in dockerfile
    assert "from gsplat.cuda._backend import _C" in dockerfile
    assert "chmod -R a+rX" in dockerfile


def test_canonical_name_matches_python_packaging_normalization() -> None:
    assert canonical_name("opencv_python") == "opencv-python"
    assert canonical_name("Qwen.VL-Utils") == "qwen-vl-utils"
