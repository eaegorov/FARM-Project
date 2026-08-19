import ast
import json
import sys
from pathlib import Path

import pytest


ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from scene_graph.offline.run import _atomic_write_json, _destroy_mapper, _run_warmup  # noqa: E402


class _FailingWarmupMapper:
    def __init__(self):
        self.calls = 0

    def _run_mapping_batch(self, _batch):
        self.calls += 1
        if self.calls == 2:
            raise RuntimeError("synthetic warmup failure")
        return None


class _DestroyRecorder:
    def __init__(self):
        self.save_flags = []

    def destroy_node(self, *, save_scene_state=True):
        self.save_flags.append(save_scene_state)
        return True


def test_warmup_exception_propagates_and_keeps_partial_progress():
    mapper = _FailingWarmupMapper()
    durations = []
    progress = {}
    with pytest.raises(RuntimeError, match="synthetic warmup failure"):
        _run_warmup(
            mapper,
            iter(["first", "second", "third"]),
            warmup_frames=3,
            batch_size=1,
            recorder=None,
            batch_durations=durations,
            progress=progress,
        )
    assert mapper.calls == 2
    assert progress == {"frames": 1}
    assert len(durations) == 1


def test_offline_cleanup_passes_explicit_success_state():
    mapper = _DestroyRecorder()
    assert _destroy_mapper(mapper, run_succeeded=False) is True
    assert _destroy_mapper(mapper, run_succeeded=True) is True
    assert mapper.save_flags == [False, True]


def test_timing_json_is_atomically_replaced_without_temp_leak(tmp_path):
    path = tmp_path / "timing" / "summary.json"
    _atomic_write_json(path, {"status": "failed", "attempt": 1})
    _atomic_write_json(path, {"status": "completed", "attempt": 2})
    assert json.loads(path.read_text(encoding="utf-8")) == {"status": "completed", "attempt": 2}
    assert list(path.parent.glob(f".{path.name}.*.tmp")) == []


def test_streaming_mapper_destroy_signature_and_save_guard():
    path = ROOT / "ros" / "mapping" / "mapping" / "nodes" / "streaming_mapper.py"
    source = path.read_text(encoding="utf-8")
    tree = ast.parse(source)
    functions = [
        node for node in ast.walk(tree)
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)) and node.name == "destroy_node"
    ]
    assert len(functions) == 1
    destroy = functions[0]
    keyword_names = [argument.arg for argument in destroy.args.kwonlyargs]
    assert "save_scene_state" in keyword_names
    default = destroy.args.kw_defaults[keyword_names.index("save_scene_state")]
    assert isinstance(default, ast.Constant) and default.value is True
    body = ast.get_source_segment(source, destroy)
    assert body is not None
    assert "save_scene_state and self._scene_state_save_on_shutdown" in body
