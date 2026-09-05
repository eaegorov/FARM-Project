from copy import deepcopy
import json
from pathlib import Path

import numpy as np
import pytest

from farm_runtime.quality_baseline import describe_file, json_digest, sha256_file
from farm_runtime.quality_benchmark import annotation_template, compare_reviews
from scripts.quality.review_farm_manual_gold import evaluate_development, freeze_gold


def fixture_packet(tmp_path):
    ply = tmp_path / "scene.ply"
    ply.write_bytes(b"test gaussian row order")
    p = {"schema": "farm.manual-gold-packet.v1", "objects": [{"object_id": 1}],
         "historically_observed_timestamps": [], "split_by_timestamp": {}, "observations": [],
         "source_ply": describe_file(ply), "selection": {"deficits": []}}
    for i, split in enumerate(["train", "train", "dev", "dev", "test", "test"]):
        ts = f"{i:06d}"
        name = f"cam00_{ts}_center.png"
        source = tmp_path / name
        source.write_bytes(f"original image {i}".encode())
        p["split_by_timestamp"][ts] = split
        p["observations"].append({"observation_id": name, "object_id": 1,
                                  "image_name": name, "source_image": describe_file(source),
                                  "physical_timestamp": ts, "camera": "cam00", "direction": "center",
                                  "shape_hw": [8, 8], "split": split})
    pairs = []
    for split in ("train", "dev", "test"):
        a = annotation_template(p, split)
        a.update(reviewer_id="alice", independent=True)
        for row in a["observations"]:
            row.update(reviewed=True, state="visible", object_identity="1", regions=[
                {"layer": "whole_object", "points": [[1, 1], [3, 1], [3, 3], [1, 3]]},
                {"layer": "protected", "points": [[5, 5], [7, 5], [7, 7], [5, 7]]},
            ])
        a["objects"][0].update(scope_reviewed=True, attachment_policy="exclude",
                               carrier_content_policy="not_applicable")
        b = deepcopy(a)
        b["reviewer_id"] = "bob"
        pairs.append((a, b))
    return p, pairs


def predictions(p, tmp_path):
    rows = []
    for obs in p["observations"]:
        if obs["split"] != "dev":
            continue
        path = tmp_path / (obs["observation_id"] + ".npy")
        arr = np.zeros((8, 8), dtype=bool)
        arr[1:4, 1:4] = True
        np.save(path, arr)
        rows.append({"observation_id": obs["observation_id"], "path": str(path), "sha256": sha256_file(path)})
    return {"packet_sha256": json_digest(p), "observations": rows}


def test_freeze_evaluate_round_trip_never_opens_test_masks(tmp_path, monkeypatch):
    p, pairs = fixture_packet(tmp_path)
    frozen = tmp_path / "frozen"
    report = freeze_gold(p, pairs, frozen)
    assert report["status"] == "PILOT_FROZEN"
    # A pilot is not a 20-30 object scene-wide quality claim.
    assert report["algorithm_changes_allowed"] is False
    pred = predictions(p, tmp_path)
    original = np.load
    def guarded(path, *args, **kwargs):
        assert "test" not in Path(path).parts
        return original(path, *args, **kwargs)
    monkeypatch.setattr(np, "load", guarded)
    result = evaluate_development(p, frozen, pred, "dev")
    assert result["per_object"][0]["iou"] == 1
    assert result["test_opened"] is False
    with pytest.raises(ValueError, match="test is sealed"):
        evaluate_development(p, frozen, pred, "test")
    pred["observations"].append({"observation_id": p["observations"][-1]["observation_id"]})
    with pytest.raises(ValueError, match="exactly"):
        evaluate_development(p, frozen, pred, "dev")


def test_source_change_blocks_freeze_and_does_not_create_mask_files(tmp_path):
    p, pairs = fixture_packet(tmp_path)
    Path(p["observations"][0]["source_image"]["path"]).write_bytes(b"changed")
    output = tmp_path / "rejected"
    result = freeze_gold(p, pairs, output)
    assert result["status"] == "BLOCKED"
    assert not list(output.rglob("*.npz"))


def test_disagreement_needs_explicit_bound_adjudication(tmp_path):
    p, pairs = fixture_packet(tmp_path)
    pairs[1][1]["observations"][0]["regions"][0]["points"][1] = [4, 1]
    comparison = compare_reviews(p, *pairs[1])
    final = deepcopy(pairs[1][0])
    final.update(reviewer_id="charlie", independent=False)
    decision = {"schema": "farm.manual-gold-adjudication.v1", "split": "dev",
                "comparison_sha256": json_digest(comparison), "adjudicator_id": "charlie",
                "reviewer_kind": "human", "annotation": final,
                "resolutions": [{"disagreement_sha256": json_digest(d), "reason": "Reviewed full resolution edge; use left polygon."}
                                for d in comparison["disagreements"]]}
    blocked = freeze_gold(p, pairs, tmp_path / "blocked")
    assert blocked["status"] == "BLOCKED"
    result = freeze_gold(p, pairs, tmp_path / "resolved", [decision])
    assert result["status"] == "PILOT_FROZEN"
    assert (tmp_path / "resolved/reviews/adjudication_0.json").exists()
    # Both original independent votes survive unchanged.
    assert json.loads((tmp_path / "resolved/reviews/1_review_b.json").read_text()) == pairs[1][1]
    decision["resolutions"] = []
    assert freeze_gold(p, pairs, tmp_path / "bad_log", [decision])["status"] == "BLOCKED"


def test_frozen_mask_tampering_fails_before_scoring(tmp_path):
    p, pairs = fixture_packet(tmp_path)
    frozen = tmp_path / "frozen"
    report = freeze_gold(p, pairs, frozen)
    path = frozen / next(r["path"] for r in report["artifacts"] if r["split"] == "dev")
    path.write_bytes(b"changed")
    with pytest.raises(ValueError, match="hash mismatch"):
        evaluate_development(p, frozen, predictions(p, tmp_path), "dev")
