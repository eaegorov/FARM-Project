from __future__ import annotations

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
    prep = parse_inventory(ROOT / "requirements/prep-runtime.lock.txt")
    main = parse_inventory(ROOT / "requirements/main-runtime.observed.txt")
    assert {
        "torch", "torchvision", "numpy", "pycolmap", "gsplat", "plyfile",
        "opencv-python", "pillow", "pyyaml", "scipy", "matplotlib",
    } <= set(prep)
    assert {"torch", "numpy", "transformers", "ultralytics", "viser"} <= set(main)


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


def test_canonical_name_matches_python_packaging_normalization() -> None:
    assert canonical_name("opencv_python") == "opencv-python"
    assert canonical_name("Qwen.VL-Utils") == "qwen-vl-utils"
