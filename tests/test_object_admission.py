import copy
import json

import pytest

from farm_runtime.quality.object_admission import (
    aggregate_admission,
    validate_admission_response,
)


def answer(scope="single_whole_object", fit="fits_target", confidence="high"):
    return dict(scope=scope, mask_fit=fit, confidence=confidence,
                reason="The selection follows the visible enclosure and its fitted doors.")


def view(image_id, timestamp, response=None, **metadata):
    return dict(view_id=image_id, physical_timestamp=timestamp,
                response=answer() if response is None else response, **metadata)


def independent_views(response=None):
    return [view(10, "time-a", response), view(20, "time-b", response)]


@pytest.mark.parametrize("change", [
    lambda r: r.update(label="cabinet"),
    lambda r: r.update(physical_timestamp="invented"),
    lambda r: r.update(admitted=True),
    lambda r: r.pop("mask_fit"),
    lambda r: r.update(scope="cabinet"),
    lambda r: r.update(scope=[]),
    lambda r: r.update(mask_fit={}),
    lambda r: r.update(mask_fit="probably_fits"),
    lambda r: r.update(confidence=True),
    lambda r: r.update(confidence=0.99),
    lambda r: r.update(reason=" \n "),
    lambda r: r.update(reason=["whole object"]),
])
def test_response_requires_only_bounded_physical_scope_fields(change):
    response = answer()
    change(response)
    with pytest.raises(ValueError):
        validate_admission_response(response)
    with pytest.raises(ValueError):
        validate_admission_response(json.dumps(response))


@pytest.mark.parametrize("raw", [
    "not JSON", "null", "[]", "42", b"{}", None,
    '```json\n{"scope":"single_whole_object"}\n```',
    '{"scope":"object_part","scope":"single_whole_object",'
    '"mask_fit":"fits_target","confidence":"high","reason":"Enclosure."}',
])
def test_invalid_json_and_duplicate_fields_do_not_become_valid_responses(raw):
    with pytest.raises(ValueError):
        validate_admission_response(raw)


def test_validated_response_is_fresh_and_preserves_visual_reason():
    source = answer()
    parsed = validate_admission_response(source)
    assert parsed == source == validate_admission_response(json.dumps(source))
    parsed["scope"] = "object_part"
    assert source["scope"] == "single_whole_object"


@pytest.mark.parametrize("reason", [
    "One integrated conveyor; its visible frame and belt form one assembly.",
    "One enclosure including fitted doors and controls.",
    "One small sign mounted on a wall, with its visible extent selected.",
    "One container, partly occluded; all of its visible extent is selected.",
])
def test_whole_object_admission_does_not_depend_on_names_or_size(reason):
    response = answer()
    response["reason"] = reason
    result = aggregate_admission(independent_views(response))
    assert result["decision"] == "admit_whole_object" and result["admitted"]
    assert result["supporting_timestamps"] == ["time-a", "time-b"]
    assert result["reason_codes"] == []
    assert result["evidence_status"] == "model_review_not_ground_truth"


def test_many_cameras_or_calls_at_one_physical_time_cannot_admit():
    views = [view(i, "same-panorama") for i in range(8)]
    views.extend(copy.deepcopy(views))
    result = aggregate_admission(views)
    assert result["decision"] == "defer"
    assert result["independent_timestamp_count"] == result["supporting_timestamp_count"] == 1
    assert result["reason_codes"] == ["insufficient_independent_timestamps"]
    assert result["timestamp_audit"][0]["observation_count"] == 16


def test_repeated_calls_do_not_inflate_real_independent_support():
    observations = independent_views()
    observations.extend([copy.deepcopy(observations[0]), view(11, "time-a")])
    result = aggregate_admission(observations)
    assert result["admitted"]
    assert result["supporting_timestamp_count"] == 2
    assert result["timestamp_audit"][0]["view_ids"] == ["10", "11"]


def test_same_view_cannot_claim_to_be_two_physical_times():
    result = aggregate_admission([view(10, "a"), view("10", "b")])
    assert not result["admitted"]
    assert "view_timestamp_conflict" in result["reason_codes"]
    assert result["supporting_timestamp_count"] == 0
    assert all("view_timestamp_conflict" in t["reason_codes"] for t in result["timestamp_audit"])


def test_timestamp_whitespace_does_not_create_independence():
    result = aggregate_admission([view(10, "a"), view(20, " a ")])
    assert not result["admitted"] and result["supporting_timestamps"] == ["a"]


@pytest.mark.parametrize("scope,fit,expected", [
    ("structural_surface", "fits_target", "route_context"),
    ("object_part", "fits_target", "route_part"),
    ("multiple_independent_objects", "fits_target", "route_collection"),
    ("multiple_independent_objects", "crosses_objects", "route_collection"),
])
def test_nonprimary_scopes_remain_routed_without_becoming_whole_objects(scope, fit, expected):
    result = aggregate_admission(independent_views(answer(scope, fit)))
    assert result["decision"] == expected
    assert not result["admitted"]
    assert result["scope"] == scope
    assert result["reason_codes"] == []


def test_partial_mask_of_a_whole_object_defers_for_recovery_not_part_relabeling():
    result = aggregate_admission(independent_views(answer(fit="partial_target")))
    assert result["scope"] == "single_whole_object"
    assert result["decision"] == "defer"
    assert "partial_target_requires_recovery" in result["reason_codes"]


@pytest.mark.parametrize("response,reason", [
    (answer(scope="unclear"), "unclear_response"),
    (answer(fit="unclear"), "unclear_response"),
    (answer(fit="crosses_objects"), "mask_crosses_objects"),
    (answer(confidence="medium"), "confidence_below_high"),
    (answer(confidence="low"), "confidence_below_high"),
])
def test_agreement_cannot_rescue_unclear_unfit_or_low_confidence_evidence(response, reason):
    result = aggregate_admission(independent_views(response))
    assert result["decision"] == "defer" and reason in result["reason_codes"]


@pytest.mark.parametrize("response,reason", [
    (answer(scope="object_part"), "scope_conflict"),
    (answer(scope="structural_surface"), "scope_conflict"),
    (answer(scope="multiple_independent_objects"), "scope_conflict"),
    (answer(fit="partial_target"), "mask_fit_conflict"),
    (answer(scope="unclear"), "unclear_response"),
    (answer(confidence="medium"), "confidence_below_high"),
])
def test_one_contrary_observation_blocks_a_large_high_confidence_majority(response, reason):
    observations = [view(i, f"time-{i}") for i in range(5)]
    observations.append(view(100, "time-0", response))
    result = aggregate_admission(observations)
    assert result["decision"] == "defer" and reason in result["reason_codes"]
    assert reason in result["timestamp_audit"][0]["reason_codes"]


@pytest.mark.parametrize("metadata,reason", [
    ({"dependency_valid": False}, "dependency_invalid"),
    ({"dependency_valid": None}, "invalid_dependency_status"),
    ({"dependency_valid": 1}, "invalid_dependency_status"),
    ({"fallback_used": True}, "fallback_used"),
    ({"fallback_used": "false"}, "invalid_fallback_status"),
    ({"fallback_used": 0}, "invalid_fallback_status"),
    ({"validation_error": "mask source hash changed"}, "runner_validation_error"),
])
def test_failed_dependencies_or_fallback_cannot_be_dropped_after_two_good_views(metadata, reason):
    result = aggregate_admission(independent_views() + [view(30, "time-c", **metadata)])
    assert result["decision"] == "defer" and reason in result["reason_codes"]
    assert result["supporting_timestamp_count"] == 2


@pytest.mark.parametrize("extra,reason", [
    (dict(view_id=30, physical_timestamp="time-c", response="bad JSON"), "invalid_response"),
    (dict(view_id=True, physical_timestamp="time-c", response=answer()), "invalid_view_id"),
    (dict(view_id=30, physical_timestamp=3, response=answer()), "invalid_physical_timestamp"),
    (dict(view_id=30, physical_timestamp=" ", response=answer()), "invalid_physical_timestamp"),
    (dict(view_id=30, response=answer()), "invalid_physical_timestamp"),
    (dict(physical_timestamp="time-c", response=answer()), "invalid_view_id"),
    ([], "invalid_observation"),
])
def test_bad_observation_is_a_group_blocker_even_with_two_valid_timestamps(extra, reason):
    result = aggregate_admission(independent_views() + [extra])
    assert result["decision"] == "defer" and reason in result["reason_codes"]


def test_cross_time_conflict_is_visible_even_when_each_timestamp_agrees_locally():
    result = aggregate_admission([view(10, "a"), view(20, "b", answer(scope="object_part"))])
    assert result["decision"] == "defer" and result["scope"] == "unclear"
    assert result["reason_codes"] == ["scope_conflict"]
    assert [t["scope"] for t in result["timestamp_audit"]] == ["single_whole_object", "object_part"]


def test_empty_evidence_cannot_be_a_primary_or_a_context_decision():
    result = aggregate_admission([])
    assert result["decision"] == "defer" and result["scope"] == "unclear"
    assert result["reason_codes"] == ["insufficient_independent_timestamps", "no_observations"]


@pytest.mark.parametrize("minimum", [0, 1, True, 2.0, "2"])
def test_independence_policy_cannot_be_relaxed_to_one_view(minimum):
    with pytest.raises(ValueError, match="at least two"):
        aggregate_admission(independent_views(), minimum)


def test_stronger_independence_policy_requires_additional_physical_time():
    assert not aggregate_admission(independent_views(), 3)["admitted"]
    assert aggregate_admission(independent_views() + [view(30, "time-c")], 3)["admitted"]


def test_aggregation_is_nonmutating_and_order_does_not_change_verdict():
    observations = independent_views() + [view(30, "time-c", answer(fit="partial_target"))]
    original = copy.deepcopy(observations)
    first = aggregate_admission(observations)
    second = aggregate_admission(reversed(observations))
    for key in ("decision", "admitted", "scope", "mask_fit", "reason_codes",
                "supporting_timestamps", "timestamp_audit"):
        assert first[key] == second[key]
    first["view_audit"][0]["response"]["scope"] = "unclear"
    assert observations == original


def witness_answer(target="independent_item", parent="none", relation="none", evidence=""):
    return dict(
        target_kind=target, mask_fit="fits_target", confidence="high",
        reason="The selection follows one independently bounded item's visible body.",
        parent_witness=dict(kind=parent, relation=relation, visual_evidence=evidence),
    )


@pytest.mark.parametrize("change", [
    lambda r: r.pop("parent_witness"),
    lambda r: r.update(target_kind="cabinet"),
    lambda r: r.update(target_kind=[]),
    lambda r: r.update(scope="single_whole_object"),
    lambda r: r.update(parent_witness=None),
    lambda r: r["parent_witness"].pop("visual_evidence"),
    lambda r: r["parent_witness"].update(parent_id=123),
    lambda r: r["parent_witness"].update(kind="wall"),
    lambda r: r["parent_witness"].update(kind=[]),
    lambda r: r["parent_witness"].update(relation="touching"),
    lambda r: r["parent_witness"].update(visual_evidence=True),
])
def test_witness_schema_requires_bounded_fields_without_model_parent_ids(change):
    from farm_runtime.quality.object_admission import validate_parent_witness_response

    response = witness_answer()
    change(response)
    with pytest.raises(ValueError):
        validate_parent_witness_response(response)
    with pytest.raises(ValueError):
        validate_parent_witness_response(json.dumps(response))


@pytest.mark.parametrize("raw", [None, "null", "[]", "bad JSON", b"{}"])
def test_witness_schema_rejects_nonobjects(raw):
    from farm_runtime.quality.object_admission import validate_parent_witness_response

    with pytest.raises(ValueError):
        validate_parent_witness_response(raw)


def test_nested_duplicate_witness_field_cannot_override_physical_relation():
    from farm_runtime.quality.object_admission import validate_parent_witness_response

    raw = json.dumps(witness_answer()).replace(
        '"relation": "none"', '"relation": "integral_component", "relation": "none"'
    )
    with pytest.raises(ValueError):
        validate_parent_witness_response(raw)


@pytest.mark.parametrize("parent,relation,evidence", [
    ("none", "none", ""),
    ("building_support", "mounted_separate", "A bracket attaches the complete device to the wall behind it."),
    ("bounded_manufactured_object", "mounted_separate", "A separate sign is affixed to the outside of the enclosure."),
])
def test_independent_item_remains_admissible_when_mounted(parent, relation, evidence):
    from farm_runtime.quality.object_admission import aggregate_witness_admission

    result = aggregate_witness_admission(independent_views(witness_answer(
        parent=parent, relation=relation, evidence=evidence,
    )))
    assert result["admitted"] and result["decision"] == "admit_whole_object"
    assert result["target_kind"] == "independent_item"
    assert result["supporting_timestamp_count"] == 2
    assert not result["parent_witness_is_ground_truth"]
    assert result["reason_codes"] == []


def test_component_routes_as_part_only_with_asserted_bounded_integral_parent():
    from farm_runtime.quality.object_admission import aggregate_witness_admission

    result = aggregate_witness_admission(independent_views(witness_answer(
        "intrinsic_component", "bounded_manufactured_object", "integral_component",
        "The selected door is hinged into the larger enclosure's visible frame and side body.",
    )))
    assert result["decision"] == "route_part" and not result["admitted"]
    assert result["target_kind"] == "intrinsic_component"
    assert result["reason_codes"] == []
    # The validated claim does not identify a parent instance or authorize merging.
    assert result["evidence_status"] == "model_review_not_ground_truth"
    assert "parent_id" not in result


@pytest.mark.parametrize("parent,relation,evidence,expected_reason", [
    ("none", "none", "", "intrinsic_component_requires_bounded_integral_parent"),
    ("building_support", "integral_component", "The item attaches to a wall.",
     "intrinsic_component_requires_bounded_integral_parent"),
    ("building_support", "mounted_separate", "A wall bracket supports the item's own body.",
     "intrinsic_component_requires_bounded_integral_parent"),
    ("bounded_manufactured_object", "mounted_separate", "The item is mounted on an enclosure.",
     "intrinsic_component_requires_bounded_integral_parent"),
    ("bounded_manufactured_object", "integral_component", " \n ", "missing_parent_visual_evidence"),
    ("unclear", "unclear", "No larger bounded item is visible.", "unclear_parent_witness"),
])
def test_unsupported_component_is_deferred_never_promoted_or_suppressed(parent, relation, evidence, expected_reason):
    from farm_runtime.quality.object_admission import aggregate_witness_admission

    result = aggregate_witness_admission(independent_views(witness_answer(
        "intrinsic_component", parent, relation, evidence,
    )))
    assert result["decision"] == "defer" and not result["admitted"]
    assert result["target_kind"] == "intrinsic_component"
    assert expected_reason in result["reason_codes"]
    assert result["supporting_timestamp_count"] == 0
    assert all(expected_reason in row["reason_codes"] for row in result["timestamp_audit"])


@pytest.mark.parametrize("target,parent,relation,expected", [
    ("independent_item", "bounded_manufactured_object", "integral_component",
     "independent_item_conflicts_with_integral_relation"),
    ("independent_item", "building_support", "none", "inconsistent_parent_witness"),
    ("independent_item", "none", "mounted_separate", "inconsistent_parent_witness"),
    ("independent_item", "unclear", "unclear", "unclear_parent_witness"),
    ("collection", "bounded_manufactured_object", "integral_component",
     "collection_conflicts_with_integral_relation"),
    ("building_structure", "bounded_manufactured_object", "mounted_separate",
     "building_structure_conflicts_with_equipment_parent"),
])
def test_cross_field_witness_conflicts_cannot_authorize_any_catalog_route(target, parent, relation, expected):
    from farm_runtime.quality.object_admission import aggregate_witness_admission

    result = aggregate_witness_admission(independent_views(witness_answer(
        target, parent, relation, "A visible attachment is asserted by the model.",
    )))
    assert result["decision"] == "defer"
    assert expected in result["reason_codes"]


@pytest.mark.parametrize("target,fit,decision", [
    ("building_structure", "fits_target", "route_context"),
    ("collection", "crosses_objects", "route_collection"),
    ("unclear", "fits_target", "defer"),
    ("independent_item", "partial_target", "defer"),
])
def test_witness_protocol_preserves_context_collection_and_mask_recovery_paths(target, fit, decision):
    from farm_runtime.quality.object_admission import aggregate_witness_admission

    response = witness_answer(target)
    response["mask_fit"] = fit
    result = aggregate_witness_admission(independent_views(response))
    assert result["decision"] == decision and not result["admitted"]


def test_witness_cannot_inflate_independence_with_two_views_of_one_capture():
    from farm_runtime.quality.object_admission import aggregate_witness_admission

    result = aggregate_witness_admission([view(1, "a", witness_answer()), view(2, "a", witness_answer())])
    assert not result["admitted"]
    assert result["supporting_timestamp_count"] == 1


@pytest.mark.parametrize("failure", ["missing_witness", "different_target", "fallback", "dependency", "uncertain"])
def test_two_good_witness_views_cannot_overrule_an_invalid_or_contrary_third(failure):
    from farm_runtime.quality.object_admission import aggregate_witness_admission

    observations = independent_views(witness_answer())
    third = view(30, "time-c", witness_answer())
    if failure == "missing_witness":
        third["response"].pop("parent_witness")
    elif failure == "different_target":
        third["response"] = witness_answer("intrinsic_component", "bounded_manufactured_object",
                                           "integral_component", "Door in the larger visible enclosure frame.")
    elif failure == "fallback":
        third["fallback_used"] = True
    elif failure == "dependency":
        third["dependency_valid"] = False
    else:
        third["response"]["confidence"] = "medium"
    result = aggregate_witness_admission(observations + [third])
    assert result["decision"] == "defer"


def test_parent_witness_validation_and_aggregation_copy_nested_input():
    from farm_runtime.quality.object_admission import aggregate_witness_admission, validate_parent_witness_response

    response = witness_answer()
    observations = independent_views(response)
    original = copy.deepcopy(observations)
    validated = validate_parent_witness_response(response)
    validated["parent_witness"]["kind"] = "unclear"
    result = aggregate_witness_admission(observations)
    result["view_audit"][0]["response"]["parent_witness"]["kind"] = "unclear"
    assert observations == original
    assert aggregate_admission(independent_views())["schema"] == "farm.object-admission.v1"
