from __future__ import annotations

from scripts.evaluation.select_farm_heldout_geometry_improvements import evaluate_object


def _row(*, status="verified", heldouts=3, good=3, iou=0.75, precision=0.8, recall=0.8, component=0.99):
    def metric(value):
        return {"minimum": value, "median": value}
    return {
        "object_id": 1,
        "status": status,
        "heldout_timestamps": heldouts,
        "good_timestamps": good,
        "provisional_gaussians": 4000,
        "summaries": {
            "iou": metric(iou),
            "precision": metric(precision),
            "recall": metric(recall),
            "largest_component_fraction": metric(component),
        },
    }


def _evaluate(baseline, candidate):
    return evaluate_object(
        1,
        baseline,
        candidate,
        minimum_holdouts=3,
        minimum_good_holdouts=3,
        minimum_iou=0.65,
        minimum_precision=0.65,
        minimum_recall=0.70,
        minimum_component_fraction=0.95,
        maximum_iou_regression=0.03,
        maximum_precision_regression=0.15,
        maximum_recall_regression=0.05,
        maximum_component_regression=0.03,
        minimum_iou_gain=0.01,
    )


def test_accepts_absolute_quality_with_measurable_heldout_gain() -> None:
    baseline = _row(heldouts=2, good=2, iou=0.78, precision=0.86, recall=0.84)
    candidate = _row(heldouts=3, good=3, iou=0.80, precision=0.76, recall=0.83)
    result = _evaluate(baseline, candidate)
    assert result["accepted"]
    assert result["benefit_reasons"] == [
        "additional_independent_holdout",
        "median_iou_gain",
    ]


def test_rejects_candidate_that_regresses_and_loses_component_coherence() -> None:
    baseline = _row(iou=0.77, precision=0.88, recall=0.74, component=1.0)
    candidate = _row(
        status="rejected", heldouts=4, iou=0.53, precision=0.34,
        recall=0.61, component=0.71,
    )
    result = _evaluate(baseline, candidate)
    assert not result["accepted"]
    assert "candidate_status_not_verified" in result["reasons"]
    assert "iou_regression_exceeds_tolerance" in result["reasons"]
    assert "component_regression_exceeds_tolerance" in result["reasons"]


def test_rejects_identical_candidate_without_measurable_benefit() -> None:
    baseline = _row()
    result = _evaluate(baseline, _row())
    assert not result["accepted"]
    assert "no_measurable_heldout_benefit" in result["reasons"]
