
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
