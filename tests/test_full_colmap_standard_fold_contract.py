from __future__ import annotations

import inspect

from scripts.farm_standard_stage import Context


def test_standard_rescue_never_uses_union_or_heldout_for_state_fitting() -> None:
    source = inspect.getsource(Context.full_colmap_rescue)
    builder = source.index('"scripts/evaluation/build_farm_full_colmap_folds.py"')
    mapping = source.index('docker_base("full-colmap-mapping"')
    assert builder < mapping
    train_frames = "/farm-run/qa/full_colmap_rescue/rgbd_folds/train/frames.json"
    train_directory = "/farm-run/qa/full_colmap_rescue/rgbd_folds/train"
    assert source.count(train_directory) == 3
    assert '"--frames-json-dir", "' + train_directory + '"' in source
    assert '"--rescue-frames-json", "' + train_frames + '"' in source
    assert '"--rescue-frames", "' + train_frames + '"' in source
    assert '"--view-role", "train"' in source
    fitting_tail = source[mapping:]
    assert "/rgbd_folds/heldout" not in fitting_tail
    assert '"--minimum-accepted-views", "3"' in source
    assert '"--minimum-independent-views", "2"' in source
    assert '"--minimum-accepted-views", "2"' not in source
    assert '"--minimum-preferred-observations", "3"' in source


def test_standard_rescue_apply_uses_canonicalized_source_state() -> None:
    source = inspect.getsource(Context.full_colmap_rescue)
    apply = source[source.index('self.post("apply_farm_full_colmap_rescue.py"') :]
    assert (
        '"--source-state", '
        '"/farm-run/qa/full_colmap_rescue/combined/scene_state_source_canonical.pt"'
        in apply
    )
    assert '"--source-state", "/farm-run/mapping/scene_state_semantic.pt"' not in apply
