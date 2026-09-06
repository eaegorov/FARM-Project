import numpy as np
import pytest

from farm_runtime.quality.refinement_schedule import crop_bounds, select_observations


def row(oid, image, timestamp, recall=0.7, background=0.2, mass=50):
    return dict(
        object_id=oid,
        image_id=image,
        timestamp=timestamp,
        metrics=dict(
            foreground_mass=mass,
            foreground_recall=recall,
            background_fraction=background,
            score=(
                (recall - recall * background / (1 - background))
                if recall is not None and background is not None
                else None
            ),
        ),
    )


def test_scheduler_skips_unknown_and_consistent_evidence_without_filling_budget():
    rows = [
        row(1, 0, "a", mass=19),
        row(2, 1, "b", recall=0.995, background=0.002),
        row(3, 2, "c", recall=None, background=None, mass=0),
        row(4, 3, "d"),
    ]
    selected = select_observations(rows, 12)
    assert [(r["object_id"], r["image_id"]) for r in selected] == [(4, 3)]
    assert "priority" not in rows[-1]


def test_scheduler_is_diverse_and_stereo_siblings_do_not_exhaust_budget():
    rows = [
        row(1, 0, "a", recall=0.2),
        row(1, 1, "a", recall=0.1),
        row(1, 2, "b", recall=0.3),
        row(1, 3, "c", recall=0.4),
        row(2, 4, "d", recall=0.6),
        row(3, 5, "e", recall=0.8),
    ]
    selected = select_observations(rows, 4, 2)
    assert [(r["object_id"], r["image_id"]) for r in selected] == [
        (1, 1),
        (2, 4),
        (3, 5),
        (1, 2),
    ]
    assert select_observations(list(reversed(rows)), 4, 2) == selected


@pytest.mark.parametrize(
    "budget, per_object", [(0, 2), (33, 2), (True, 2), (12, 0), (12, 4)]
)
def test_scheduler_rejects_unbounded_work(budget, per_object):
    with pytest.raises(ValueError):
        select_observations([], budget, per_object)


def test_scheduler_rejects_duplicate_or_invalid_observation_metrics():
    r = row(1, 0, "a")
    with pytest.raises(ValueError, match="duplicate"):
        select_observations([r, r], 12)
    with pytest.raises(ValueError, match="invalid"):
        select_observations([row(1, 0, "a", recall=float("nan"))], 12)


def test_evidence_crop_preserves_border_pixels_and_rejects_empty_region():
    region = np.zeros((100, 120), bool)
    region[0:20, 100:120] = True
    assert crop_bounds(region) == [80, 0, 120, 40]
    with pytest.raises(ValueError, match="nonempty"):
        crop_bounds(np.zeros_like(region))


def test_scheduler_uses_attainable_gain_when_recall_and_background_pass_absolute_gates():
    observation = row(47, 10, "a", recall=0.97, background=0.097)
    result = select_observations([observation], 12)
    assert len(result) == 1
    assert result[0]["priority"] > 0.13
