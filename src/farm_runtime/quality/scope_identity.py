"""Review physical scope identity independently of geometric mask completeness.

Identity is symmetric. Extent direction must come from separately validated
geometric evidence and is deliberately absent from this semantic response.
"""
from __future__ import annotations

import json

PROTOCOL = "farm.scope-identity-only.v1"
PROMPT = """Judge whether segmentation hypotheses A and B refer to the same single independent physical object. Answer only physical identity, not which mask is larger, more complete, whole, or partial. Do not infer an answer from the order or the letters.

There are three sheets. Sheet 0 contains full scene context and two aligned crops of the SAME photograph: A is highlighted in blue and B in orange. Sheet 1 contains an independent real photograph with A's raw mask only. Sheet 2 contains an independent real photograph with B's raw mask only. The extra sheets are individual references, NOT a second photograph with both masks. Follow the actual highlighted pixels and object boundaries, not the rectangular crop or unselected surroundings. Annotation colors do not describe the materials.

First describe what A actually selects, then what B actually selects. Check the two independent references against the paired photograph. Incomplete visibility or missing segmented surfaces can still depict the same independent object. However, a separately targeted component, a structural surface, a collection of neighboring machines, and an independent device attached to another object are not aliases of a whole independent object. Shared parts, proximity and similar appearance do not establish identity.

For scope_a and scope_b use one_independent_object only when the selected scope represents one identifiable independent object, even if some of its surfaces are missing or occluded. Use only_part when the selected scope is just a component rather than an independent object. Use multiple_independent_objects if the selected scope also includes another independent neighboring object; this is an explicit veto even when much of the mask overlaps the same main object. Use structural_surface or unclear when appropriate. If your observation describes an included neighboring independent object, do not label that scope one_independent_object. Do not invent unseen continuations or assume that all surfaces in the crop are selected.

Choose same_independent_object only if BOTH scopes identify the SAME one independent object across the actual references and neither includes a separate neighbor. Choose different_objects for different independent objects. Choose part_or_collection when one or both scopes instead describe a component, multiple independent objects, or a structural surface. Choose unclear when the photographs do not resolve identity. High confidence requires consistent visible physical identity in the paired photograph and both individual references. Do not use category labels or numeric object IDs as evidence.

Return ONLY JSON with exactly these fields: observation_a, observation_b, scope_a, scope_b, relation, confidence, reason. observation_a, observation_b and reason must be nonempty descriptions grounded in visible structure. scope_a and scope_b must each be one of: one_independent_object, only_part, multiple_independent_objects, structural_surface, unclear. relation must be one of: different_objects, same_independent_object, part_or_collection, unclear. confidence must be high, medium or low. Do not return extent, whole/partial, larger/smaller or direction fields."""

_FIELDS = {"observation_a", "observation_b", "scope_a", "scope_b", "relation", "confidence", "reason"}
_SCOPES = {"one_independent_object", "only_part", "multiple_independent_objects", "structural_surface", "unclear"}
_RELATIONS = {"different_objects", "same_independent_object", "part_or_collection", "unclear"}


def validate_response(raw: str) -> dict:
    """Validate exact schema and explicit objectness veto; never rewrite answers."""
    def unique_fields(pairs):
        result = {}
        for key, value in pairs:
            if key in result:
                raise ValueError("duplicate identity response field")
            result[key] = value
        return result

    result = json.loads(raw, object_pairs_hook=unique_fields)
    if not isinstance(result, dict) or set(result) != _FIELDS:
        raise ValueError("exact identity-only response fields required")
    for key in ("observation_a", "observation_b", "reason"):
        if not isinstance(result[key], str) or not result[key].strip():
            raise ValueError("visible observations and an identity reason required")
    for key in ("scope_a", "scope_b"):
        if not isinstance(result[key], str) or result[key] not in _SCOPES:
            raise ValueError("invalid physical scope")
    if not isinstance(result["relation"], str) or result["relation"] not in _RELATIONS:
        raise ValueError("invalid symmetric identity relation")
    if not isinstance(result["confidence"], str) or result["confidence"] not in {"high", "medium", "low"}:
        raise ValueError("invalid confidence")
    if result["relation"] == "same_independent_object" and any(
        result[key] != "one_independent_object" for key in ("scope_a", "scope_b")
    ):
        raise ValueError("same-object identity contradicts explicit physical-scope veto")
    return result


def consensus(responses: list[dict], scope_ids: list[int] | tuple[int, int]) -> bool:
    """Require high identity agreement in both orders; contains no size decision.

    The caller must independently enforce candidate/source bindings, geometric
    eligibility, previously reviewed separate-object vetoes and disjoint aliases.
    This validator cannot establish narrative truth from a fluent explanation.
    """
    if len(scope_ids) != 2 or any(type(oid) is not int for oid in scope_ids) or scope_ids[0] == scope_ids[1]:
        raise ValueError("two distinct integer scope IDs required")
    if len(responses) != 2:
        return False
    seen = set()
    for response in responses:
        order = response.get("scope_order")
        if not isinstance(order, (list, tuple)) or len(order) != 2 or any(type(oid) is not int for oid in order) or set(order) != set(scope_ids):
            raise ValueError("response display order does not bind to the candidate pair")
        seen.add(tuple(order))
        if response.get("validation_error") is not None or response.get("parsed") is None:
            return False
        parsed = validate_response(response["raw"])
        if parsed != response["parsed"]:
            raise ValueError("parsed identity response differs from immutable raw output")
        if parsed["relation"] != "same_independent_object" or parsed["confidence"] != "high":
            return False
    return seen == {tuple(scope_ids), tuple(reversed(scope_ids))}


GEOMETRY_POLICY = dict(
    maximum_pairs=8, minimum_paired_containment=0.90,
    minimum_smaller_strong_claim_coverage=0.85,
    minimum_individual_physical_timestamps=2, minimum_strong_gaussians=64,
    minimum_weighted_small_inside_large_obb=0.85, observed_obb_padding_m=0.04,
)


def validate_pair_geometry(proof):
    """Recheck cached proof arithmetic; source measurements are not recomputed.

    The application must bind the unchanged proof to its verified source plan.
    This is one paired raw-mask view plus independent individual references,
    not two paired observations or a new depth measurement.
    """
    import math

    pair = proof["scope_ids"]
    if len(pair) != 2 or any(type(oid) is not int for oid in pair) or pair[0] == pair[1]:
        raise ValueError("two distinct integer geometry scope IDs required")
    if proof.get("eligible") is not True:
        return False
    whole, partial = proof["geometric_whole_candidate"], proof["geometric_partial_candidate"]
    if {whole, partial} != set(pair) or whole == partial:
        raise ValueError("geometric direction changed candidate IDs")
    eligible = []
    for observation in proof["paired_observations"]:
        areas, intersection = observation["areas"], observation["intersection"]
        if (len(areas) != 2 or any(type(n) is not int or n <= 0 for n in areas)
                or type(intersection) is not int or not 0 <= intersection <= min(areas)):
            raise ValueError("invalid paired-mask area evidence")
        coverage = [intersection / n for n in areas]
        if len(observation["coverage"]) != 2 or any(
            not math.isclose(actual, recorded, rel_tol=1e-9, abs_tol=1e-9)
            for actual, recorded in zip(coverage, observation["coverage"])
        ):
            raise ValueError("paired-mask coverage differs from pixel counts")
        bigger = pair[int(areas[1] > areas[0])]
        if observation["bigger_scope"] != bigger:
            raise ValueError("paired-mask direction differs from pixel counts")
        if max(coverage) >= GEOMETRY_POLICY["minimum_paired_containment"]:
            eligible.append(observation)
    if not eligible or {row["bigger_scope"] for row in eligible} != {whole}:
        return False
    selected = max(eligible, key=lambda row: (row["intersection"], -row["frame_id"]))
    if selected["frame_id"] != proof["paired_frame_id"]:
        raise ValueError("selected paired observation changed")
    counts = {oid: proof["strong_counts"][str(oid)] for oid in pair}
    common = proof["shared_strong_gaussians"]
    if (any(type(n) is not int or n < 0 for n in counts.values())
            or type(common) is not int or not 0 <= common <= min(counts.values())):
        raise ValueError("invalid strong-Gaussian counts")
    coverage = common / max(1, min(counts.values()))
    if not math.isclose(coverage, proof["smaller_strong_claim_coverage"], rel_tol=1e-9, abs_tol=1e-9):
        raise ValueError("strong support coverage differs from row counts")
    contained = proof["weighted_small_inside_large_obb"]
    if not isinstance(contained, (int, float)) or not math.isfinite(contained) or not 0 <= contained <= 1:
        raise ValueError("invalid weighted extent containment")
    if (min(counts.values()) < GEOMETRY_POLICY["minimum_strong_gaussians"]
            or counts[whole] < counts[partial]
            or coverage < GEOMETRY_POLICY["minimum_smaller_strong_claim_coverage"]
            or contained < GEOMETRY_POLICY["minimum_weighted_small_inside_large_obb"]):
        return False
    for oid in pair:
        timestamps = proof["individual_physical_timestamps"][str(oid)]
        if (any(not isinstance(t, str) or not t for t in timestamps)
                or len(set(timestamps)) < GEOMETRY_POLICY["minimum_individual_physical_timestamps"]):
            return False
    return True
