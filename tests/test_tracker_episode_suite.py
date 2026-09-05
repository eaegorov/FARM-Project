from __future__ import annotations

import importlib.util
import json
import sys
from types import SimpleNamespace
from pathlib import Path

import numpy as np
import pytest
from PIL import Image

ROOT = Path(__file__).resolve().parents[1]
MODULE_PATH = ROOT / "src" / "farm_runtime" / "tracker_episode_suite.py"
SPEC = importlib.util.spec_from_file_location("farm_tracker_episode_suite", MODULE_PATH)
assert SPEC is not None and SPEC.loader is not None
MODULE = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = MODULE
SPEC.loader.exec_module(MODULE)

RUNNER_PATH = ROOT / "scripts/tracking/run_farm_tracker_episode_suite.py"
RUNNER_SPEC = importlib.util.spec_from_file_location(
    "farm_tracker_suite_runner", RUNNER_PATH
)
assert RUNNER_SPEC is not None and RUNNER_SPEC.loader is not None
RUNNER = importlib.util.module_from_spec(RUNNER_SPEC)
RUNNER_SPEC.loader.exec_module(RUNNER)


def _make_materialized_suite(tmp_path: Path) -> Path:
    source = tmp_path / "source"
    source.mkdir()
    (source / "train.png").write_bytes(b"train-rgb")
    (source / "heldout.png").write_bytes(b"heldout-rgb")
    plan = tmp_path / "plan.json"
    plan.write_text('{"schema":"farm.tracker-subset-plan.v1"}\n', encoding="utf-8")
    plan_hash = MODULE.sha256_file(plan)
    root = tmp_path / "episodes"
    root.mkdir()
    rows = []
    for episode_id, split, source_name, timestamp in (
        ("cam00__center__train__000", "train", "train.png", "000010"),
        ("cam01__center__heldout__000", "heldout", "heldout.png", "000020"),
    ):
        episode_root = root / episode_id
        frame_root = episode_root / "frames"
        frame_root.mkdir(parents=True)
        frame_name = f"000000__{source_name}"
        source_path = (source / source_name).resolve()
        (frame_root / frame_name).symlink_to(source_path)
        (episode_root / "frames.txt").write_text(frame_name + "\n", encoding="utf-8")
        payload = {
            "schema": "farm.materialized-tracker-episode.v1",
            "episode_id": episode_id,
            "camera": episode_id.split("__", 1)[0],
            "view_family": "center",
            "split": split,
            "frame_count": 1,
            "source_plan": str(plan),
            "source_plan_sha256": plan_hash,
            "frames": [
                {
                    "episode_index": 0,
                    "materialized_name": frame_name,
                    "source_name": source_name,
                    "source_path": str(source_path),
                    "bytes": source_path.stat().st_size,
                    "sha256": MODULE.sha256_file(source_path),
                    "physical_timestamp": timestamp,
                }
            ],
        }
        (episode_root / "episode.json").write_text(
            json.dumps(payload), encoding="utf-8"
        )
        rows.append(
            {
                "episode_id": episode_id,
                "camera": payload["camera"],
                "view_family": "center",
                "split": split,
                "frame_count": 1,
                "relative_path": episode_id,
            }
        )
    manifest = {
        "schema": "farm.materialized-tracker-episodes.v1",
        "source_plan": str(plan),
        "source_plan_sha256": plan_hash,
        "hashes_verified": True,
        "episode_count": 2,
        "frame_count": 2,
        "episodes": rows,
    }
    (root / "manifest.json").write_text(json.dumps(manifest), encoding="utf-8")
    return root


def test_suite_input_verifies_hashes_and_preserves_split_provenance(
    tmp_path: Path,
) -> None:
    root = _make_materialized_suite(tmp_path)

    suite = MODULE.load_materialized_suite(root)

    assert [episode.split for episode in suite.episodes] == ["train", "heldout"]
    assert suite.source_plan_sha256 == MODULE.sha256_file(tmp_path / "plan.json")
    assert all(
        len(episode.ordered_frame_content_sha256) == 64 for episode in suite.episodes
    )


def test_suite_selection_unknown_id_and_changed_frame_fail_closed(
    tmp_path: Path,
) -> None:
    root = _make_materialized_suite(tmp_path)
    with pytest.raises(ValueError, match="unknown episode"):
        MODULE.load_materialized_suite(root, episode_ids={"missing"})

    (tmp_path / "source" / "train.png").write_bytes(b"changed")
    with pytest.raises(ValueError, match="size changed|checksum changed"):
        MODULE.load_materialized_suite(root)


def test_manifest_declared_counts_are_enforced(tmp_path: Path) -> None:
    root = _make_materialized_suite(tmp_path)
    manifest_path = root / "manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    manifest["frame_count"] = 99
    manifest_path.write_text(json.dumps(manifest), encoding="utf-8")

    with pytest.raises(ValueError, match="frame_count mismatch"):
        MODULE.load_materialized_suite(root)


def test_timestamp_cannot_leak_between_train_and_heldout(tmp_path: Path) -> None:
    root = _make_materialized_suite(tmp_path)
    payload_path = root / "cam01__center__heldout__000" / "episode.json"
    payload = json.loads(payload_path.read_text(encoding="utf-8"))
    payload["frames"][0]["physical_timestamp"] = "000010"
    payload_path.write_text(json.dumps(payload), encoding="utf-8")

    with pytest.raises(ValueError, match="leaks across"):
        MODULE.load_materialized_suite(root)


def test_materialized_frame_must_remain_declared_symlink(tmp_path: Path) -> None:
    root = _make_materialized_suite(tmp_path)
    link = root / "cam00__center__train__000" / "frames" / "000000__train.png"
    link.unlink()
    link.write_bytes(b"train-rgb")

    with pytest.raises(ValueError, match="must remain a symlink"):
        MODULE.load_materialized_suite(root)


def test_atomic_output_publishes_or_removes_all_staging(tmp_path: Path) -> None:
    destination = tmp_path / "success"
    with MODULE.atomic_output_directory(destination) as staging:
        (staging / "complete.txt").write_text("done", encoding="utf-8")
    assert (destination / "complete.txt").read_text(encoding="utf-8") == "done"

    failed = tmp_path / "failed"
    with pytest.raises(RuntimeError, match="boom"):
        with MODULE.atomic_output_directory(failed) as staging:
            (staging / "partial.txt").write_text("partial", encoding="utf-8")
            raise RuntimeError("boom")
    assert not failed.exists()
    assert not list(tmp_path.glob(".failed.tmp-*"))


def test_mask_audit_enforces_exact_uint8_set_and_254_budget(tmp_path: Path) -> None:
    masks = tmp_path / "masks"
    masks.mkdir()
    Image.fromarray(np.array([[0, 1], [254, 0]], dtype=np.uint8)).save(
        masks / "frame.png"
    )
    audit = MODULE.audit_8bit_masks(masks, ["frame.jpg"])
    assert audit["local_identity_count"] == 2
    assert audit["maximum_local_id"] == 254

    Image.fromarray(np.array([[0, 255]], dtype=np.uint8)).save(masks / "frame.png")
    with pytest.raises(ValueError, match="1..254 budget"):
        MODULE.audit_8bit_masks(masks, ["frame.jpg"])

    Image.fromarray(np.zeros((2, 2, 3), dtype=np.uint8)).save(masks / "frame.png")
    with pytest.raises(ValueError, match="single-channel"):
        MODULE.audit_8bit_masks(masks, ["frame.jpg"])


def test_mask_audit_matches_source_rgb_shape(tmp_path: Path) -> None:
    masks = tmp_path / "masks"
    frames = tmp_path / "frames"
    masks.mkdir()
    frames.mkdir()
    Image.fromarray(np.zeros((3, 4, 3), dtype=np.uint8)).save(frames / "frame.jpg")
    Image.fromarray(np.zeros((2, 4), dtype=np.uint8)).save(masks / "frame.png")

    with pytest.raises(ValueError, match="mask/RGB shape mismatch"):
        MODULE.audit_8bit_masks(masks, ["frame.jpg"], reference_frame_root=frames)


def test_suite_rejects_backend_owned_paths_and_hashes_checkpoint_directory(
    tmp_path: Path,
) -> None:
    with pytest.raises(ValueError, match="owns per-episode"):
        RUNNER._reject_owned_options(
            ["--model", "weights", "--output-dir=/tmp/unsafe"], ["--output-dir"]
        )
    model = tmp_path / "model"
    model.mkdir()
    (model / "model.safetensors").write_bytes(b"weights")
    (model / "config.json").write_text("{}", encoding="utf-8")
    first = RUNNER._checkpoint_directory(model)
    second = RUNNER._checkpoint_directory(model)
    assert first["aggregate_sha256"] == second["aggregate_sha256"]
    assert {row["relative_path"] for row in first["files"]} == {
        "config.json",
        "model.safetensors",
    }


def test_peak_rss_sampler_reports_local_values() -> None:
    with MODULE.PeakRssSampler(interval_seconds=0.001) as sampler:
        allocation = bytearray(1024 * 1024)
        assert allocation
    report = sampler.report()
    assert report["start_rss_mib"] > 0
    assert report["peak_rss_mib"] >= report["start_rss_mib"]
    assert report["end_rss_mib"] > 0


def test_orchestrator_constructs_backend_once_and_publishes_atomically(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    episodes = _make_materialized_suite(tmp_path)
    output = tmp_path / "suite-output"
    calls = {"loads": 0, "episodes": 0}

    class FakeBackend:
        name = "fake-deva"

        def __init__(self, arguments: list[str]) -> None:
            assert arguments == ["--quality", "test"]
            calls["loads"] += 1
            self.checkpoints = {"fake": {"sha256": "a" * 64}}
            stage = {
                "seconds": 1.0,
                "peak_allocated_mib": 10.0,
                "peak_reserved_mib": 12.0,
            }
            self.model_load = {"fake": stage}

        def configuration(self) -> dict[str, object]:
            return {"maximum_objects": 254}

        def run_episode(
            self,
            episode: object,
            output_dir: Path,
            final_dir: Path,
            run_id: str,
        ) -> dict[str, object]:
            calls["episodes"] += 1
            output_dir.mkdir()
            (output_dir / "complete.txt").write_text("ok", encoding="utf-8")
            report = {
                "status": "pass",
                "input": {"episode_id": episode.episode_id},
                "measurement": {
                    "wall_seconds": 0.1,
                    "peak_cuda_allocated_mib": 11.0,
                    "peak_cuda_reserved_mib": 13.0,
                    "stages": {
                        "automatic_segmentation_and_tracking": {"seconds": 0.05}
                    },
                },
                "output": {"mask_count": episode.frame_count},
            }
            RUNNER._write_json(output_dir / "measurement.json", report)
            return report

    fake_torch = SimpleNamespace(
        cuda=SimpleNamespace(
            is_available=lambda: True,
            get_device_name=lambda _index: "fake-cuda",
        ),
        version=SimpleNamespace(cuda="fake-runtime"),
        __version__="fake-torch",
    )
    monkeypatch.setitem(sys.modules, "torch", fake_torch)
    monkeypatch.setattr(RUNNER, "DevaBackend", FakeBackend)

    assert (
        RUNNER.main(
            [
                "--backend",
                "deva",
                "--episodes-root",
                str(episodes),
                "--output-root",
                str(output),
                "--run-id",
                "cpu-test",
                "--quality",
                "test",
            ]
        )
        == 0
    )
    assert calls == {"loads": 1, "episodes": 2}
    report = json.loads((output / "suite_measurement.json").read_text(encoding="utf-8"))
    assert report["execution"]["model_loads"] == 1
    assert report["input"]["episode_count_by_split"] == {
        "heldout": 1,
        "train": 1,
    }
    assert report["input"]["frame_count_by_split"] == {"heldout": 1, "train": 1}
    assert not list(tmp_path.glob(".suite-output.tmp-*"))


def _encode_long_ids(values: np.ndarray) -> np.ndarray:
    values = np.asarray(values, dtype=np.uint32)
    return np.stack(
        [
            (values & 255).astype(np.uint8),
            ((values >> 8) & 255).astype(np.uint8),
            ((values >> 16) & 255).astype(np.uint8),
        ],
        axis=-1,
    )


def test_deva_long_ids_are_losslessly_compacted_without_uint8_wrap(
    tmp_path: Path,
) -> None:
    masks = tmp_path / "Annotations"
    masks.mkdir()
    Image.fromarray(
        _encode_long_ids(np.array([[0, 269], [66000, 269]], dtype=np.uint32)),
        mode="RGB",
    ).save(masks / "000000.png")
    Image.fromarray(
        _encode_long_ids(np.array([[66000, 0], [269, 0]], dtype=np.uint32)),
        mode="RGB",
    ).save(masks / "000001.png")
    video_json = {
        "annotations": [
            {"segments_info": [{"id": 269, "area": 2}, {"id": 66000, "area": 1}]},
            {"segments_info": [{"id": 66000, "area": 1}, {"id": 269, "area": 1}]},
        ]
    }

    compact_json, mapping = RUNNER._compact_deva_long_id_outputs(
        masks, video_json, maximum_local_id=254
    )

    assert mapping == [
        {"deva_original_id": 269, "local_id": 1},
        {"deva_original_id": 66000, "local_id": 2},
    ]
    first = np.asarray(Image.open(masks / "000000.png"))
    assert first.dtype == np.uint8 and first.ndim == 2
    assert first.tolist() == [[0, 1], [2, 1]]
    segments = compact_json["annotations"][0]["segments_info"]
    assert [(row["id"], row["deva_original_id"]) for row in segments] == [
        (1, 269),
        (2, 66000),
    ]
    assert video_json["annotations"][0]["segments_info"][0]["id"] == 269


def test_deva_long_id_compaction_rejects_pred_mask_disagreement(
    tmp_path: Path,
) -> None:
    masks = tmp_path / "Annotations"
    masks.mkdir()
    Image.fromarray(
        _encode_long_ids(np.array([[0, 300]], dtype=np.uint32)), mode="RGB"
    ).save(masks / "000000.png")
    video_json = {"annotations": [{"segments_info": [{"id": 269, "area": 1}]}]}

    with pytest.raises(ValueError, match="absent from pred.json"):
        RUNNER._compact_deva_long_id_outputs(
            masks, video_json, maximum_local_id=254
        )
    assert masks.is_dir()
    assert not (tmp_path / "Annotations.compact").exists()
