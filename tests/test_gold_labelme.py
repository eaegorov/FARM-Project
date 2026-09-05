import json
from pathlib import Path

import pytest

from farm_runtime.quality_baseline import describe_file, json_digest
from farm_runtime.quality_benchmark import annotation_template
from scripts.quality.convert_farm_gold_labelme import export_templates, import_annotations


def test_labelme_import_needs_explicit_identity_and_preserves_reviewer_metadata(tmp_path):
    rgb = tmp_path / "cam00_000001_center.png"
    rgb.write_bytes(b"original")
    p = {"schema": "farm.manual-gold-packet.v1", "objects": [{"object_id": 5}],
         "split_by_timestamp": {"000001": "dev"}, "historically_observed_timestamps": [],
         "observations": [{"observation_id": "five", "object_id": 5, "image_name": rgb.name,
                           "physical_timestamp": "000001", "camera": "cam00", "direction": "center",
                           "split": "dev", "source_image": describe_file(rgb), "shape_hw": [8, 8]}]}
    output = tmp_path / "labelme"
    export_templates(p, "dev", output)
    base = annotation_template(p, "dev")
    with pytest.raises(ValueError, match="identity_confirmed"):
        import_annotations(p, base, output)
    binding = json.loads((output / "_farm_binding.json").read_text())
    path = output / next(iter(binding["files"]))
    data = json.loads(path.read_text())
    data["flags"].update(reviewed=True, identity_confirmed=True, visible=True)
    data["shapes"] = [{"shape_type": "polygon", "label": "whole_object", "points": [[1, 1], [3, 1], [3, 3]]}]
    path.write_text(json.dumps(data))
    result = import_annotations(p, base, output)
    assert result["observations"][0]["object_identity"] == "5"
    assert result["reviewer_id"] is None
    assert result["independent"] is None
    assert result["objects"][0]["scope_reviewed"] is False
    rgb.write_bytes(b"other image")
    with pytest.raises(ValueError, match="image identity"):
        import_annotations(p, base, output)


def test_canonical_hash_rejects_json_key_collisions():
    with pytest.raises(ValueError, match="duplicate JSON key"):
        json_digest({1: "one", "1": "different"})
