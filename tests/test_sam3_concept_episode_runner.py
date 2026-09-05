from __future__ import annotations

import importlib.util
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest


ROOT = Path(__file__).resolve().parents[1]
SCRIPT = ROOT / "scripts/tracking/run_farm_sam3_concept_episode.py"
SPEC = importlib.util.spec_from_file_location("farm_sam3_episode_runner", SCRIPT)
assert SPEC is not None and SPEC.loader is not None
MODULE = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(MODULE)


def test_overlap_is_resolved_by_score_and_ids_persist() -> None:
    masks = np.zeros((2, 4, 5), dtype=bool)
    masks[0, 1:3, 1:4] = True
    masks[1, 2:4, 2:5] = True
    identity_map = {99: 1}

    output, audit = MODULE._compose_short_id_mask(
        object_ids=np.array([99, 101]),
        scores=np.array([0.70, 0.90]),
        masks=masks,
        local_id_by_object_id=identity_map,
        minimum_score=0.45,
    )

    assert identity_map == {99: 1, 101: 2}
    assert output[1, 1] == 1
    assert output[2, 2] == 2
    assert audit["overlap_pixels_before_resolution"] == 2
    assert audit["accepted_objects"] == 2


def test_low_score_proposal_is_not_published() -> None:
    masks = np.ones((1, 2, 3), dtype=bool)

    output, audit = MODULE._compose_short_id_mask(
        object_ids=np.array([5]),
        scores=np.array([0.2]),
        masks=masks,
        local_id_by_object_id={},
        minimum_score=0.45,
    )

    assert not output.any()
    assert audit["accepted_objects"] == 0


def test_shape_mismatch_and_identity_overflow_fail_closed() -> None:
    with pytest.raises(ValueError, match="inconsistent"):
        MODULE._compose_short_id_mask(
            object_ids=np.array([1, 2]),
            scores=np.array([0.8]),
            masks=np.ones((2, 2, 2), dtype=bool),
            local_id_by_object_id={},
            minimum_score=0.4,
        )

    with pytest.raises(ValueError, match="identity budget"):
        MODULE._compose_short_id_mask(
            object_ids=np.array([999]),
            scores=np.array([0.8]),
            masks=np.ones((1, 2, 2), dtype=bool),
            local_id_by_object_id={index: index for index in range(1, 255)},
            minimum_score=0.4,
        )


def test_prompt_file_is_nonempty_and_casefold_unique(tmp_path: Path) -> None:
    prompt_file = tmp_path / "prompts.txt"
    prompt_file.write_text("machine\nMachine\n", encoding="utf-8")
    with pytest.raises(ValueError, match="duplicate"):
        MODULE._read_lines(prompt_file, label="prompt file")

    prompt_file.write_text("# comment\nfire extinguisher\nvacuum cleaner\n", encoding="utf-8")
    assert MODULE._read_lines(prompt_file, label="prompt file") == [
        "fire extinguisher",
        "vacuum cleaner",
    ]


def test_cv_utils_kernel_is_loaded_from_exact_pinned_revision() -> None:
    calls = []
    kernel = SimpleNamespace(generic_nms=lambda *args: None, cc_2d=lambda *args: None)
    modeling = SimpleNamespace(cv_utils_kernel=None)

    def loader(*args, **kwargs):
        calls.append((args, kwargs))
        return kernel

    report = MODULE._load_pinned_cv_utils_kernel(
        package_version_resolver=lambda name: "0.16.1",
        kernel_loader=loader,
        modeling_module=modeling,
    )

    assert calls == [
        (
            ("kernels-community/cv-utils",),
            {
                "lockfile": None,
                "revision": "8bc5d583ad41502e37647763437b4881b87e5110",
                "backend": "cuda",
            },
        )
    ]
    assert modeling.cv_utils_kernel is kernel
    assert report["resolver_version"] == "0.16.1"
    assert report["load_mode"] == "exact-revision-local-cache-fail-closed"


def test_cv_utils_kernel_rejects_wrong_resolver_version_before_load() -> None:
    with pytest.raises(RuntimeError, match="requires kernels==0.16.1"):
        MODULE._load_pinned_cv_utils_kernel(
            package_version_resolver=lambda name: "0.17.0",
            kernel_loader=lambda *args, **kwargs: pytest.fail("must not load"),
            modeling_module=SimpleNamespace(cv_utils_kernel=None),
        )
