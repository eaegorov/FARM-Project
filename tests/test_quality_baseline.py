from pathlib import Path

from farm_runtime.quality_baseline import sha256_file, verify_source_snapshot


def test_snapshot_verifies_content_modes_and_unexpected_files(tmp_path: Path):
    p = tmp_path / "code.py"
    p.write_text("original")
    p.chmod(0o755)
    rows = [{"path": "code.py", "sha256": sha256_file(p), "mode": "0o755"}]
    assert verify_source_snapshot(tmp_path, rows)["status"] == "PASS"
    p.write_text("changed")
    p.chmod(0o644)
    (tmp_path / "unexpected.py").write_text("pass")
    result = verify_source_snapshot(tmp_path, rows)
    assert set(result["errors"]) == {"content:code.py", "mode:code.py", "unexpected:unexpected.py"}


def test_json_digest_is_stable_after_integer_key_serialization():
    import json
    from farm_runtime.quality_baseline import json_digest
    value = {"objects": {2: "two", 108: "one hundred eight", 37: "thirty seven"}}
    assert json_digest(value) == json_digest(json.loads(json.dumps(value)))
