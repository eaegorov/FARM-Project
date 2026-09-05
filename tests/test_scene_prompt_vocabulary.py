from __future__ import annotations

import importlib.util
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
SCRIPT = ROOT / "scripts/semantics/build_farm_scene_prompt_vocabulary.py"
SPEC = importlib.util.spec_from_file_location("scene_prompt_vocabulary", SCRIPT)
assert SPEC is not None and SPEC.loader is not None
MODULE = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(MODULE)


def _row(object_id: int, category: str, **overrides):
    contract = {
        "contract_valid": True,
        "decision": "keep",
        "context_sufficient": True,
        "complete_bounded": True,
        "head_noun_is_primitive": False,
        "specificity": "specific_identity",
        "confidence": 0.95,
        "identity_basis": "diagnostic_geometry",
        "form_hypernym": category,
    }
    contract.update(overrides)
    return {"id": object_id, "category": category, "label_contract": contract}


def test_only_context_backed_specific_qwen_terms_become_prompts() -> None:
    rows = MODULE.select_prompt_rows(
        [
            _row(2, "Fire Extinguisher"),
            _row(3, "fire extinguisher", confidence=0.99),
            _row(4, "machine"),
            _row(5, "monitor", context_sufficient=False),
            _row(6, "cabinet", confidence=0.70),
            _row(7, "box", head_noun_is_primitive=True),
        ],
        minimum_confidence=0.90,
    )

    assert [row["prompt"] for row in rows] == ["fire extinguisher"]
    assert [source["object_id"] for source in rows[0]["sources"]] == [2, 3]


def test_normalization_does_not_invent_synonyms() -> None:
    rows = MODULE.select_prompt_rows(
        [_row(16, "Cleaner")], minimum_confidence=0.90
    )
    assert rows[0]["prompt"] == "cleaner"
    assert "vacuum" not in rows[0]["prompt"]
