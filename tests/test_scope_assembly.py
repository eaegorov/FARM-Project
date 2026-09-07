
from types import SimpleNamespace
from pathlib import Path
import numpy as np
import pytest
from tools.farm_shaper_bridge import gaussian_lift as lift
from farm_runtime.quality.scope_assembly import collapse_family_evidence, select_families, FIELDS

def config():
    c,_=lift.load_config(Path("configs/quality/native_rendered_v1.json"))
    c["refinement"]["enabled"]=False
    c["build"].update(minimum_object_gaussians=1,maximum_object_fraction=.9)
    return c

def evidence(g,indices,positive=2,negative=0):
    n=len(indices)
    return lift.ObjectEvidence(SimpleNamespace(object_id=g,category="object"),
        np.array(indices,np.int64),np.full(n,.01,np.float32),max(positive,2),
        np.ones(n,np.float32),np.zeros(n,np.float32),np.ones(n,np.float32),
        np.full(n,positive,np.uint16),np.full(n,negative,np.uint16))

def test_internal_conflict_recovers_but_unrelated_stronger_claim_wins():
    c=config();e={1:evidence(1,[0,1,2]),2:evidence(2,[0,1]),3:evidence(3,[2,3],3)}
    prior=lift.make_claims(e,20,c)[0].copy()
    pooled,audit=collapse_family_evidence(e,{1:[1,2]},c,{1:{"t1","t2"},2:{"t1","t2"},3:{"t1","t2","t3"}})
    after=lift.make_claims(pooled,20,c)[0]
    assert prior[:4].tolist()==[-1,-1,3,3]
    assert after[:4].tolist()==[1,1,3,3]
    assert pooled[1].positive_timestamps.tolist()==[2,2,2]
    assert pooled[1].build_timestamp_count==2
    assert audit[0]["timestamps_summed"] is False

def test_whole_row_selection_keeps_negative_evidence_and_acceptance_tiers():
    c=config();c["build"]["minimum_global_purity"]=.8
    a=evidence(1,[0,1],3,2);a.negative_weight[:]=4
    b=evidence(2,[0,1],1,0);b.positive_timestamps[0]=2
    # First child row is strong; second is growth eligible. Both beat rejected
    # higher-support parent rows; fields must be copied from that same source.
    pooled,_=collapse_family_evidence({1:a,2:b},{1:[1,2]},c,{1:{"1","2","3"},2:{"1","2"}})
    for field in FIELDS:
        np.testing.assert_array_equal(getattr(pooled[1],field),getattr(b,field))
    assert pooled[1].build_timestamp_count==3
    np.testing.assert_array_equal(a.negative_timestamps,[2,2])

def test_shared_part_does_not_bridge_two_objects_or_transitively_merge():
    review=dict(schema="farm.scope-family-review.v1",decisions=[
        dict(parent_id=1,accepted=True,parent_scope="single_whole_object",integral_ids=[3,4]),
        dict(parent_id=2,accepted=True,parent_scope="single_whole_object",integral_ids=[4,5])])
    families,audit=select_families(review,range(1,6))
    assert families=={1:[1,3],2:[2,5]}
    assert audit["ambiguous_child_ids"]==[4]
    review["decisions"][1]["integral_ids"]=[1]
    families,audit=select_families(review,range(1,6))
    assert families=={2:[1,2,3,4]}
    assert audit["nested_parent_ids_composed"]==[1]

def test_invalid_and_rejected_families_cannot_change_scope():
    d=dict(schema="farm.scope-family-review.v1",decisions=[
        dict(parent_id=1,accepted=True,parent_scope="multiple_independent_objects",integral_ids=[2])])
    assert select_families(d,{1,2})[0]=={}
    d["decisions"][0]["integral_ids"]=[3]
    with pytest.raises(ValueError):select_families(d,{1,2})
    e={1:evidence(1,[0]),2:evidence(2,[0])}
    with pytest.raises(ValueError):collapse_family_evidence(e,{1:[1,2],2:[2]},config(),{1:{"a","b"},2:{"a","b"}})

def test_no_family_preserves_evidence_identity_and_duplicate_order_is_stable():
    c=config();e={1:evidence(1,[0,2]),2:evidence(2,[0,1])}
    exact,_=collapse_family_evidence(e,{},c,{1:{"a","b"},2:{"a","b"}})
    assert exact[1] is e[1] and exact[2] is e[2]
    a,_=collapse_family_evidence(e,{1:[1,2]},c,{1:{"a","b"},2:{"b","c"}})
    b,_=collapse_family_evidence(dict(reversed(list(e.items()))),{1:[2,1]},c,{1:{"a","b"},2:{"b","c"}})
    for field in FIELDS:np.testing.assert_array_equal(getattr(a[1],field),getattr(b[1],field))
    assert a[1].build_timestamp_count==3


def test_shared_ancestor_paths_compose_but_separate_veto_and_cycles_do_not():
    def row(p,children,separate=()):
        return dict(parent_id=p,integral_ids=children,separate_ids=list(separate),
                    accepted=True,parent_scope="single_whole_object")
    review=dict(schema="farm.scope-family-review.v1",decisions=[
        row(1,[2,3]),row(2,[3,4]),row(3,[4])])
    assert select_families(review,{1,2,3,4})[0]=={1:[1,2,3,4]}
    review["decisions"][0]["separate_ids"]=[4]
    families,audit=select_families(review,{1,2,3,4})
    assert families=={1:[1,2,3]}
    assert audit["separate_veto_ids"]==[4]
    review["decisions"][2]["integral_ids"]=[1,4]
    families,audit=select_families(review,{1,2,3,4})
    assert families=={}
    assert audit["cyclic_or_downstream_ids_deferred"]==[1,2,3,4]


def relation(outer, inner, times=("a", "b"), status="multiview_containment"):
    return dict(outer_group=outer, inner_group=inner, timestamps=list(times),
                independent_timestamps=len(times), status=status,
                relation="observed_mask_containment", physical_relation="unresolved",
                evidence=[dict(frame=f"frame_{t}", timestamp=t, parent_node=outer,
                               child_node=inner, child_coverage=0.99, parent_coverage=0.2)
                          for t in times])


def test_remapped_relations_deduplicate_edges_and_count_real_times_once():
    from farm_runtime.quality.scope_assembly import remap_scope_relations
    original = [relation(1, 9), relation(2, 9), relation(1, 9), relation(1, 2)]
    mapped = remap_scope_relations(original, {1: 1, 2: 1})
    assert len(mapped) == 1
    row = mapped[0]
    assert (row["outer_group"], row["inner_group"]) == (1, 9)
    assert row["independent_timestamps"] == 2  # not 2 + 2 + 2
    assert row["timestamps"] == ["a", "b"]
    assert len(row["evidence"]) == 4  # two different source masks, each in two times
    assert [r["source_relation_index"] for r in row["source_relations"]] == [0, 1, 2]
    assert {r["source_outer_group"] for r in row["evidence"]} == {1, 2}
    assert row["assembled_mask_containment_remeasured"] is False
    assert row["timestamp_support_summed"] is False
    assert original[1]["outer_group"] == 2


def test_remapped_direction_conflict_stays_explicit_and_single_views_do_not_promote():
    from farm_runtime.quality.scope_assembly import remap_scope_relations
    rows = remap_scope_relations([relation(1, 9), relation(9, 2)], {2: 1})
    assert len(rows) == 2
    assert all(r["status"] == "conflicting_directions" for r in rows)
    partial = [relation(1, 9, ["a"], "single_timestamp"),
               relation(2, 9, ["b"], "single_timestamp")]
    row, = remap_scope_relations(partial, {2: 1})
    assert row["independent_timestamps"] == 2
    assert row["status"] == "unresolved_after_family_mapping"


def test_current_obb_and_external_relations_replace_stale_quality_flags():
    from farm_runtime.quality.scope_assembly import refresh_object_quality
    stable = dict(extent_stability=dict(sensitive_axes=[]))
    sensitive = dict(extent_stability=dict(sensitive_axes=[1]))
    rows = [
        dict(object_id=1, observed_obb=stable, competing_scope_ids=[2],
             quality=dict(flags=["no_accepted_native_mask", "observed_extent_tail_sensitive", "unresolved_nested_scope"])),
        dict(object_id=3, observed_obb=None, quality=dict(flags=[])),
        dict(object_id=4, observed_obb=sensitive, quality=dict(flags=[])),
    ]
    refresh_object_quality(rows, [relation(3, 4)])
    assert rows[0]["quality"]["flags"] == []
    assert rows[0]["quality"]["observed_obb_available"]
    assert rows[0]["competing_scope_ids"] == []
    assert rows[1]["quality"]["flags"] == ["no_accepted_native_mask", "unresolved_nested_scope"]
    assert rows[2]["quality"]["flags"] == ["observed_extent_tail_sensitive", "unresolved_nested_scope"]
    assert rows[1]["competing_scope_ids"] == [4]
    assert rows[2]["competing_scope_ids"] == [3]
    assert all(r["quality"]["physical_size_validated"] is False for r in rows)
    with pytest.raises(ValueError, match="absent object"):
        refresh_object_quality(rows, [relation(1, 2)])


def test_cyclic_branch_does_not_remove_an_unrelated_valid_branch():
    def row(parent, children):
        return dict(parent_id=parent, accepted=True, parent_scope="single_whole_object",
                    integral_ids=children, separate_ids=[])
    review = dict(schema="farm.scope-family-review.v1", decisions=[
        row(1, [2, 4]), row(2, [3]), row(3, [2, 5])])
    families, audit = select_families(review, {1, 2, 3, 4, 5})
    assert families == {1: [1, 4]}
    assert audit["cyclic_or_downstream_ids_deferred"] == [2, 3, 5]


def test_explicit_veto_of_the_root_defers_the_contradictory_family():
    review = dict(schema="farm.scope-family-review.v1", decisions=[
        dict(parent_id=1, accepted=True, parent_scope="single_whole_object",
             integral_ids=[2], separate_ids=[]),
        dict(parent_id=2, accepted=True, parent_scope="single_whole_object",
             integral_ids=[3], separate_ids=[1]),
    ])
    families, audit = select_families(review, {1, 2, 3})
    assert families == {}
    assert audit["separate_veto_ids"] == [1]


def identity_fixture():
    """Two overlapping whole hypotheses share two reviewed integral parts."""
    import json
    from copy import deepcopy
    from farm_runtime.quality.scope_identity import GEOMETRY_POLICY
    def parent(oid):
        return dict(parent_id=oid, accepted=True, parent_scope="single_whole_object",
                    integral_ids=[3, 4], separate_ids=[])
    catalog = dict(sha256="a" * 64)
    prior_source = dict(sha256="b" * 64)
    prior = dict(schema="farm.scope-family-review.v1", source_catalog=catalog,
                 decisions=[parent(1), parent(2)], closed_test_opened=False)
    proof = dict(scope_ids=[1, 2], shared_child_ids=[3, 4], eligible=True,
                 geometric_whole_candidate=2, geometric_partial_candidate=1,
                 paired_observations=[dict(frame_id=7, physical_timestamp="t1",
                     areas=[100, 180], intersection=100, coverage=[1.0, 100 / 180], bigger_scope=2)],
                 paired_frame_id=7, strong_counts={"1": 100, "2": 180},
                 shared_strong_gaussians=90, smaller_strong_claim_coverage=.9,
                 weighted_small_inside_large_obb=.99,
                 individual_physical_timestamps={"1": ["t1", "t2"], "2": ["t1", "t3"]})
    parsed = dict(observation_a="One identifiable machine enclosure.",
                  observation_b="The same enclosure with additional visible casing.",
                  scope_a="one_independent_object", scope_b="one_independent_object",
                  relation="same_independent_object", confidence="high",
                  reason="Matching fixed structure; no selected independent neighbor.")
    responses = [dict(scope_order=order, raw=json.dumps(parsed), parsed=deepcopy(parsed),
                      validation_error=None, sheets=[dict(image=dict(sha256=str(i) * 64))])
                 for i, order in enumerate(([1, 2], [2, 1]))]
    plan = dict(schema="farm.shared-root-pair-review.v1", source_catalog=deepcopy(catalog),
                source_prior_review=deepcopy(prior_source), policy=deepcopy(GEOMETRY_POLICY),
                scheduled_pairs=[dict(scope_ids=[1, 2])],
                pairs=[dict(proof=deepcopy(proof), responses=deepcopy(responses))],
                closed_test_opened=False)
    identity = dict(schema="farm.symmetric-identity-experiment.v1",
                    source_catalog=deepcopy(catalog), source_prior_review=deepcopy(prior_source),
                    source_pair_plan=dict(sha256="c" * 64), closed_test_opened=False,
                    pairs=[dict(proof=proof, responses=responses, accepted=True)])
    return prior, identity, plan


def test_reviewed_alias_recovers_parts_without_changing_neighbors_or_source_review():
    from copy import deepcopy
    from farm_runtime.quality.scope_assembly import merge_reviewed_same_object_edges
    prior, identity, plan = identity_fixture()
    original = deepcopy((prior, identity, plan))
    known = {1, 2, 3, 4, 9}
    prior["decisions"][0]["separate_ids"] = [9]  # a protected external neighbor
    expected_prior = deepcopy(prior)
    updated, audit = merge_reviewed_same_object_edges(prior, identity, known, pair_plan=plan)
    assert prior == expected_prior and (identity, plan) == original[1:]
    assert [(r["whole_scope_id"], r["partial_scope_id"]) for r in audit["applied_same_object_scope_edges"]] == [(2, 1)]
    families, selection = select_families(updated, known)
    assert families == {2: [1, 2, 3, 4]}
    assert selection["ambiguous_child_ids"] == []
    assert updated["decisions"][0]["separate_ids"] == [9]
    c = config()
    rows = {1: evidence(1, [0, 1, 2]), 2: evidence(2, [0, 1, 2]),
            3: evidence(3, [0], 3), 4: evidence(4, [1], 3), 9: evidence(9, [2, 3], 4)}
    before = lift.make_claims(rows, 20, c)[0]
    pooled, _ = collapse_family_evidence(rows, families, c, {g: {"t1", "t2"} for g in rows})
    after = lift.make_claims(pooled, 20, c)[0]
    assert before[:4].tolist() == [3, 4, 9, 9]
    assert after[:4].tolist() == [2, 2, 9, 9]
    assert pooled[2].positive_timestamps.tolist() == [3, 3, 2]
    assert audit["accepted_flag_used_as_evidence"] is False


@pytest.mark.parametrize("field", ["namespace", "prior_review", "proof", "sheets", "unscheduled", "unknown", "policy"])
def test_identity_application_rejects_changed_sources_and_evidence(field):
    from farm_runtime.quality.scope_assembly import merge_reviewed_same_object_edges
    prior, identity, plan = identity_fixture()
    known = {1, 2, 3, 4}
    if field == "namespace": identity["source_catalog"]["sha256"] = "d" * 64
    elif field == "prior_review": identity["source_prior_review"]["sha256"] = "d" * 64
    elif field == "proof": identity["pairs"][0]["proof"]["shared_strong_gaussians"] = 99
    elif field == "sheets": identity["pairs"][0]["responses"][0]["sheets"] = []
    elif field == "unscheduled": plan["scheduled_pairs"] = []
    elif field == "unknown": known = {1, 3, 4}
    elif field == "policy": plan["policy"]["minimum_paired_containment"] = .1
    with pytest.raises(ValueError):
        merge_reviewed_same_object_edges(prior, identity, known, pair_plan=plan)


@pytest.mark.parametrize("cause", ["semantic", "parent_rejected", "one_shared_part", "direct_veto", "descendant_veto", "weak_3d", "one_timestamp"])
def test_identity_application_defers_even_when_accepted_flag_is_true(cause):
    import json
    from copy import deepcopy
    from farm_runtime.quality.scope_assembly import merge_reviewed_same_object_edges
    prior, identity, plan = identity_fixture()
    if cause == "semantic":
        response = identity["pairs"][0]["responses"][1]
        response["parsed"]["relation"] = "different_objects"
        response["raw"] = json.dumps(response["parsed"])
    elif cause == "parent_rejected": prior["decisions"][0]["accepted"] = False
    elif cause == "one_shared_part": prior["decisions"][0]["integral_ids"] = [3]
    elif cause == "direct_veto": prior["decisions"][0]["separate_ids"] = [2]
    elif cause == "descendant_veto":
        prior["decisions"].append(dict(parent_id=3, integral_ids=[], separate_ids=[2], accepted=False, parent_scope="unclear"))
    elif cause == "weak_3d":
        identity["pairs"][0]["proof"].update(shared_strong_gaussians=80, smaller_strong_claim_coverage=.8)
        plan["pairs"][0]["proof"] = deepcopy(identity["pairs"][0]["proof"])
    elif cause == "one_timestamp":
        identity["pairs"][0]["proof"]["individual_physical_timestamps"]["1"] = ["t1", "t1"]
        plan["pairs"][0]["proof"] = deepcopy(identity["pairs"][0]["proof"])
    updated, audit = merge_reviewed_same_object_edges(prior, identity, {1, 2, 3, 4}, pair_plan=plan)
    assert not audit["applied_same_object_scope_edges"]
    assert len(audit["deferred_pairs"]) == 1
    assert updated["decisions"] == prior["decisions"]


def test_identity_application_recomputes_arithmetic_and_does_not_bridge_pair_chains():
    from copy import deepcopy
    from farm_runtime.quality.scope_assembly import merge_reviewed_same_object_edges
    prior, identity, plan = identity_fixture()
    proof = identity["pairs"][0]["proof"]
    proof["smaller_strong_claim_coverage"] = .99  # unchanged plan cannot hide contradictory arithmetic
    plan["pairs"][0]["proof"] = deepcopy(proof)
    with pytest.raises(ValueError, match="row counts"):
        merge_reviewed_same_object_edges(prior, identity, {1, 2, 3, 4}, pair_plan=plan)
    prior, identity, plan = identity_fixture()
    third = deepcopy(prior["decisions"][1]); third["parent_id"] = 5
    prior["decisions"].append(third)
    extra = deepcopy(identity["pairs"][0])
    extra["proof"].update(scope_ids=[1, 5], geometric_whole_candidate=5)
    extra["proof"]["paired_observations"][0]["bigger_scope"] = 5
    for key in ("strong_counts", "individual_physical_timestamps"):
        extra["proof"][key]["5"] = extra["proof"][key].pop("2")
    for response in extra["responses"]:
        response["scope_order"] = [5 if g == 2 else g for g in response["scope_order"]]
    identity["pairs"].append(extra)
    plan["pairs"].append(deepcopy(extra))
    plan["scheduled_pairs"].append(dict(scope_ids=[1, 5]))
    updated, audit = merge_reviewed_same_object_edges(prior, identity, {1, 2, 3, 4, 5}, pair_plan=plan)
    assert not audit["applied_same_object_scope_edges"]
    assert audit["overlapping_pair_scope_ids"] == [1]
    assert [r["reason"] for r in audit["deferred_pairs"]] == ["overlapping_accepted_pairs"] * 2
    assert updated["decisions"] == prior["decisions"]


def identity_files(tmp_path):
    from farm_runtime.quality_baseline import describe_file, write_json
    prior, identity, plan = identity_fixture()
    catalog_path = tmp_path / "catalog.json"
    write_json(catalog_path, dict(schema="farm.quality-scene-catalog.v1", objects=[]))
    prior["source_catalog"] = describe_file(catalog_path)
    family_path = tmp_path / "families.json"
    write_json(family_path, prior)
    plan["source_catalog"] = describe_file(catalog_path)
    plan["source_prior_review"] = describe_file(family_path)
    plan_path = tmp_path / "pair_plan.json"
    write_json(plan_path, plan)
    identity["source_pair_plan"] = describe_file(plan_path)
    identity.pop("source_catalog")
    identity.pop("source_prior_review")  # actual inference result binds through its plan
    identity_path = tmp_path / "identity.json"
    write_json(identity_path, identity)
    return family_path, identity_path, plan_path, catalog_path


def test_identity_loader_binds_real_files_without_rewriting_inference(tmp_path):
    from farm_runtime.quality.scope_assembly import load_reviewed_same_object_edges
    from farm_runtime.quality_baseline import describe_file
    paths = identity_files(tmp_path)
    before = {p: p.read_bytes() for p in paths}
    updated, audit = load_reviewed_same_object_edges(paths[0], paths[1], {1, 2, 3, 4})
    assert select_families(updated, {1, 2, 3, 4})[0] == {2: [1, 2, 3, 4]}
    assert audit["identity_review"] == describe_file(paths[1])
    assert audit["family_review"] == describe_file(paths[0])
    assert audit["source_pair_plan"] == describe_file(paths[2])
    assert audit["application_code"]["path"].endswith("scope_assembly.py")
    assert audit["identity_protocol_code"]["path"].endswith("scope_identity.py")
    assert {p: p.read_bytes() for p in paths} == before
    assert set(tmp_path.iterdir()) == set(paths)


@pytest.mark.parametrize("target", ["catalog", "plan", "prior", "different_family", "conflicting_namespace"])
def test_identity_loader_rejects_changed_source_files(tmp_path, target):
    import json
    from farm_runtime.quality.scope_assembly import load_reviewed_same_object_edges
    from farm_runtime.quality_baseline import write_json
    family, identity, plan, catalog = identity_files(tmp_path)
    if target in {"catalog", "plan", "prior"}:
        path = {"catalog": catalog, "plan": plan, "prior": family}[target]
        path.write_text(path.read_text() + " ")
    elif target == "different_family":
        other = tmp_path / "other_families.json"
        other.write_text(family.read_text() + " ")
        family = other
    else:
        row = json.loads(identity.read_text())
        row["source_catalog"] = dict(sha256="d" * 64)
        write_json(identity, row)
    with pytest.raises(ValueError):
        load_reviewed_same_object_edges(family, identity, {1, 2, 3, 4})


@pytest.mark.parametrize("enable_identity", [False, True])
def test_optional_identity_cli_applies_only_after_loading_known_objects(tmp_path, monkeypatch, enable_identity):
    import json
    import farm_runtime.quality.scope_assembly as assembly
    import farm_runtime.quality.native_observations as native_observations
    from farm_runtime.quality_baseline import describe_file, write_json
    family, identity, plan_path, catalog_path = identity_files(tmp_path)
    input_path, config_path, native_path = (tmp_path / name for name in ("input.json", "config.json", "native.json"))
    write_json(input_path, {})
    write_json(config_path, {})
    write_json(native_path, dict(input=describe_file(input_path), config=describe_file(config_path)))
    catalog = json.loads(catalog_path.read_text())
    catalog["native_output"] = describe_file(native_path)
    write_json(catalog_path, catalog)
    prior = json.loads(family.read_text()); prior["source_catalog"] = describe_file(catalog_path)
    write_json(family, prior)
    plan = json.loads(plan_path.read_text())
    plan.update(source_catalog=describe_file(catalog_path), source_prior_review=describe_file(family))
    write_json(plan_path, plan)
    inference = json.loads(identity.read_text()); inference["source_pair_plan"] = describe_file(plan_path)
    write_json(identity, inference)
    known = {1, 2, 3, 4}
    run = SimpleNamespace(objects=[SimpleNamespace(object_id=g) for g in known])
    monkeypatch.setattr(native_observations, "load_prepared", lambda path: (run, {}, {}))
    monkeypatch.setattr(lift, "load_config", lambda path: ({}, None))
    calls = []
    original_select = assembly.select_families
    original_loader = assembly.load_reviewed_same_object_edges
    def loader(family_path, identity_path, known_ids):
        calls.append(set(known_ids))
        return original_loader(family_path, identity_path, known_ids)
    class Prepared(Exception):
        pass
    def select(review, known_ids):
        result = original_select(review, known_ids)
        # The helper internally validates the original prior; stop only after
        # main receives the enriched review or the optional path was skipped.
        if "same_object_identity_application" in review or not enable_identity:
            assert result[0] == ({2: [1, 2, 3, 4]} if enable_identity else {})
            raise Prepared
        return result
    monkeypatch.setattr(assembly, "load_reviewed_same_object_edges", loader)
    monkeypatch.setattr(assembly, "select_families", select)
    args = ["--catalog", str(catalog_path), "--families", str(family), "--output", str(tmp_path / "result")]
    if enable_identity:
        args += ["--identity-review", str(identity)]
    with pytest.raises(Prepared):
        assembly.main(args)
    assert calls == ([known] if enable_identity else [])
    assert not (tmp_path / "result").exists()
