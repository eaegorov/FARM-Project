from __future__ import annotations

import pytest

from farm_runtime.anchor_cross_fold_reconciliation import reconcile_object_pair


ANCHOR = {
    "point_count": 128,
    "center_m": [0.0, 0.0, 0.0],
    "shape_extents_m": [2.0, 1.0, 0.5],
    "shape_diagonal_m": 2.291287847,
    "obb_used": False,
}


def _identity(
    object_id: int,
    split: str,
    local_id: int,
    global_id: int,
    *,
    center: list[float],
    shape: list[float] | None = None,
    prompt: str = "target",
) -> dict:
    episode = f"object-{object_id:06d}__cam00__center__{split}"
    return {
        "identity_key": f"{episode}::local:{local_id}",
        "episode_id": episode,
        "split": split,
        "global_id": global_id,
        "passed_shadow_gate": True,
        "center_m": center,
        "shape_extents_m": shape or [1.0, 0.55, 0.30],
        "prompts": [prompt],
    }


def _reconcile(object_id: int, identities: list[dict]) -> dict:
    return reconcile_object_pair(
        object_id=object_id,
        train_episode_id=f"object-{object_id:06d}__cam00__center__train",
        heldout_episode_id=f"object-{object_id:06d}__cam00__center__heldout",
        train_physical_timestamps=["000100", "000101"],
        heldout_physical_timestamps=["000200", "000201"],
        anchor=ANCHOR,
        identities=identities,
    )


def test_unique_shared_id_with_joint_anchor_support_is_accepted() -> None:
    result = _reconcile(
        2,
        [
            _identity(2, "train", 3, 1, center=[0.08, 0.01, 0.0], prompt="fire extinguisher"),
            _identity(2, "heldout", 15, 1, center=[-0.04, 0.02, 0.01], prompt="fire extinguisher"),
        ],
    )
    assert result["status"] == "accepted"
    assert result["accepted_mapping"]["raw_global_id"] == 1
    assert result["timestamp_independence"]["verified"] is True


def test_near_anchor_different_ids_stay_unresolved_and_distant_shared_id_is_rejected() -> None:
    result = _reconcile(
        47,
        [
            _identity(47, "train", 4, 9, center=[0.09, 0.0, 0.0], prompt="server"),
            _identity(47, "heldout", 21, 0, center=[0.12, 0.01, 0.0], prompt="server"),
            _identity(47, "train", 1, 8, center=[3.0, 0.0, 0.0], prompt="fire hose cabinet"),
            _identity(47, "heldout", 22, 8, center=[3.1, 0.0, 0.0], prompt="fire hose cabinet"),
        ],
    )
    assert result["status"] == "unresolved"
    assert result["accepted_mapping"] is None
    assert result["reason"] == "metric_anchor_pair_exists_but_raw_cross_fold_id_is_not_shared"
    assert [row["raw_global_id"] for row in result["candidates"]] == [8]
    assert result["candidates"][0]["passed"] is False
    assert "train_metric_anchor_support" in result["candidates"][0]["failed_checks"]
    assert len(result["near_anchor_unreconciled"]) == 1


def test_distant_shared_distractor_never_counts_as_anchor_success() -> None:
    result = _reconcile(
        16,
        [
            _identity(16, "train", 6, 5, center=[4.4, 0.0, 0.0], prompt="poster"),
            _identity(16, "heldout", 4, 5, center=[4.3, 0.0, 0.0], prompt="poster"),
        ],
    )
    assert result["status"] == "rejected"
    assert result["eligible_shared_candidate_count"] == 0
    assert result["near_anchor_unreconciled"] == []


def test_two_anchor_supported_shared_ids_are_ambiguous_fail_closed() -> None:
    result = _reconcile(
        37,
        [
            _identity(37, "train", 1, 5, center=[0.04, 0.00, 0.00], prompt="poster"),
            _identity(37, "heldout", 3, 5, center=[0.06, 0.01, 0.00], prompt="poster"),
            _identity(37, "train", 7, 6, center=[-0.07, 0.00, 0.00], prompt="sign"),
            _identity(37, "heldout", 9, 6, center=[-0.05, 0.01, 0.00], prompt="sign"),
        ],
    )
    assert result["status"] == "ambiguous"
    assert result["accepted_mapping"] is None
    assert result["eligible_shared_candidate_count"] == 2


def test_physical_timestamp_overlap_is_rejected_before_reconciliation() -> None:
    identity = _identity(2, "train", 3, 1, center=[0.0, 0.0, 0.0])
    with pytest.raises(ValueError, match="physical timestamps overlap"):
        reconcile_object_pair(
            object_id=2,
            train_episode_id="object-000002__cam00__center__train",
            heldout_episode_id="object-000002__cam00__center__heldout",
            train_physical_timestamps=["000100"],
            heldout_physical_timestamps=["000100"],
            anchor=ANCHOR,
            identities=[identity],
        )
