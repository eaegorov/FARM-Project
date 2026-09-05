"""Selection must reject parts, contradicted cargo and unknowable scope ties."""

import numpy as np
import pytest

from farm_runtime.quality.native_refinement import (
    choose_proposal,
    other_timestamp_memberships,
    proposal_metrics,
)


def test_other_timestamp_votes_exclude_all_siblings_and_keep_conflict_unknown():
    votes = {
        "100": (np.array([1, 1, 1, 0], np.uint16), np.array([0, 0, 0, 1], np.uint16)),
        "200": (np.array([0, 1, 1, 0], np.uint16), np.array([1, 0, 0, 0], np.uint16)),
        "300": (np.array([0, 1, 0, 0], np.uint16), np.array([1, 0, 1, 0], np.uint16)),
    }
    foreground, background = other_timestamp_memberships(votes, "100")
    np.testing.assert_array_equal(foreground, [False, True, False, False])
    np.testing.assert_array_equal(background, [True, False, False, False])
    assert not (foreground & background).any()
    with pytest.raises(ValueError, match="absent"):
        other_timestamp_memberships(votes, "missing")


def evidence():
    fg = np.zeros((20, 20), np.float32)
    bg = np.zeros_like(fg)
    fg[5:15, 5:15] = 1
    bg[15:20, 5:15] = 1
    return fg, bg


def test_clean_frame_beats_attached_cargo_but_single_door_is_not_eligible():
    fg, bg = evidence()
    source = (fg + bg) > 0
    part = fg > 0
    part[:, 10:] = False
    result = choose_proposal(dict(source=source, frame=fg > 0, door=part), fg, bg)
    assert result["selected"] == "frame"
    assert not result["metrics"]["door"]["eligible"]
    assert result["metrics"]["frame"]["foreground_recall"] == 1
    assert result["metrics"]["source"]["background_mass_in_mask"] == 50


def test_unobserved_region_cannot_resolve_whole_object_scope():
    fg, bg = evidence()
    compact = fg > 0
    extended = compact.copy()
    extended[:5, :] = True
    result = choose_proposal(
        dict(source=(fg + bg) > 0, compact=compact, extended=extended), fg, bg
    )
    assert result["decision"] == "ambiguous_scope"
    assert result["selected"] is None


def test_no_other_view_mass_is_unknown_and_tiny_gain_does_not_replace():
    fg, bg = evidence()
    result = choose_proposal(dict(source=fg > 0, clean=fg > 0), fg, bg)
    assert result["decision"] == "no_material_gain"
    result = choose_proposal(dict(source=fg > 0, clean=fg > 0), fg * 0.001, bg)
    assert result["decision"] == "insufficient_other_view_evidence"
    assert result["selected"] is None
    result = choose_proposal(dict(source=fg > 0, clean=fg > 0), fg * 0, bg)
    assert result["metrics"]["clean"]["foreground_recall"] is None


def test_all_contaminated_candidates_abstain_and_invalid_evidence_fails():
    fg, bg = evidence()
    result = choose_proposal(
        dict(source=(fg + bg) > 0, contaminated=(fg + bg) > 0), fg, bg
    )
    assert result["decision"] == "no_consistent_candidate"
    with pytest.raises(ValueError, match="finite"):
        proposal_metrics(fg > 0, fg + np.nan, bg)
    with pytest.raises(ValueError, match="non-negative"):
        proposal_metrics(fg > 0, fg, -bg)
    with pytest.raises(ValueError, match="boolean"):
        proposal_metrics(fg, fg, bg)


def test_partial_source_can_improve_without_losing_existing_foreground():
    fg, bg = evidence()
    source = (fg + bg) > 0
    source[5, 5:15] = False
    source[6, 5:10] = False  # source covers only 85% of other-view native mass
    clean = source & (fg > 0)
    result = choose_proposal(dict(source=source, clean=clean), fg, bg)
    assert result["selected"] == "clean"
    smaller = clean.copy()
    smaller[6, 10] = False
    result = choose_proposal(dict(source=source, smaller=smaller), fg, bg)
    assert result["selected"] is None
    source[7:10] = False
    result = choose_proposal(dict(source=source, tiny=source & (fg > 0)), fg, bg)
    assert result["selected"] is None


def test_quarantine_is_object_specific_unknown_and_preserves_two_timestamps():
    from farm_runtime.quality.native_refinement import (
        quarantine_inconsistent_observations,
    )

    masks = [
        dict(image_id=i, physical_timestamp_ns=i * 100, mask={"original": i})
        for i in range(4)
    ]
    manifest = dict(
        objects=[dict(object_id=7, masks=masks[:]), dict(object_id=8, masks=masks[:])],
        split=dict(objects=[dict(object_id=7), dict(object_id=8)], build_timestamps=[]),
    )
    decisions = [
        dict(
            object_id=7,
            image_id=i,
            decision="no_consistent_candidate",
            metrics=dict(
                source=dict(
                    foreground_mass=100, foreground_recall=0.7, background_fraction=0.6
                )
            ),
        )
        for i in range(3)
    ]
    rejected = quarantine_inconsistent_observations(manifest, decisions)
    assert len(rejected) == 2
    assert [m["image_id"] for m in manifest["objects"][0]["masks"]] == [2, 3]
    assert manifest["objects"][1]["masks"] == masks
    assert decisions[2]["quarantine"] == "insufficient_remaining_timestamps"
    assert manifest["split"]["objects"][0]["build_timestamps"] == ["200", "300"]
    assert rejected[0]["observation"]["mask"] == {"original": 0}
