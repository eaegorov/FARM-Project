from __future__ import annotations

import importlib.util
import json
import urllib.error
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


def _load(name: str, relative: str):
    spec = importlib.util.spec_from_file_location(name, ROOT / relative)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_review_payload_caps_each_crop_without_dropping_views(monkeypatch) -> None:
    review = _load("farm_review_budget", "scripts/semantics/review_farm_object_crops.py")
    captured: dict = {}

    class Response:
        def __enter__(self):
            return self

        def __exit__(self, *_args):
            return False

        def read(self) -> bytes:
            content = json.dumps({
                "category": "box",
                "description": "rectangular box",
                "attributes": ["rectangular"],
                "confidence": 0.9,
                "decision": "keep",
            })
            return json.dumps({"choices": [{"message": {"content": content}}]}).encode()

    def fake_urlopen(request, timeout):
        assert timeout == 180
        captured.update(json.loads(request.data.decode("utf-8")))
        return Response()

    monkeypatch.setattr(review.urllib.request, "urlopen", fake_urlopen)
    crops = [(Path(f"crop-{index}.jpg"), b"jpeg") for index in range(6)]
    result = review.request_review("http://127.0.0.1:1/v1", "model", {}, crops)

    assert result["review_category"] == "box"
    assert captured["mm_processor_kwargs"] == {"max_pixels": 262144}
    images = captured["messages"][0]["content"][1:]
    assert len(images) == 6


def test_vl_embedding_payload_fits_pinned_context() -> None:
    embeddings = _load("farm_embedding_budget", "scripts/semantics/enrich_farm_embeddings.py")
    payload = embeddings._vl_payload("model", "data:image/jpeg;base64,AA==")

    assert payload["mm_processor_kwargs"] == {"max_pixels": 262144}
    assert payload["encoding_format"] == "float"
    # 16x16 patches and 2x2 spatial merge: at most 256 image tokens/crop.
    assert (262144 // (16 * 16 * 2 * 2)) + 220 < 600


def test_transport_failure_is_not_reinterpreted_as_semantic_unknown() -> None:
    review = _load("farm_review_transport", "scripts/semantics/review_farm_object_crops.py")
    report = {
        "initial_request_error_ids": [17],
        "verification_request_error_ids": [],
    }
    try:
        review.require_successful_transport(report)
    except RuntimeError as error:
        assert "semantic unknown is reserved" in str(error)
    else:
        raise AssertionError("transport failures must fail the review stage")

    review.require_successful_transport({
        "initial_request_error_ids": [],
        "verification_request_error_ids": [],
    })


def test_review_retries_one_temporary_transport_failure(monkeypatch) -> None:
    review = _load("farm_review_retry", "scripts/semantics/review_farm_object_crops.py")
    calls = []

    class Response:
        def __enter__(self):
            return self

        def __exit__(self, *_args):
            return False

        def read(self) -> bytes:
            content = json.dumps({
                "category": "box", "description": "box", "attributes": [],
                "confidence": 0.9, "decision": "keep",
            })
            return json.dumps({"choices": [{"message": {"content": content}}]}).encode()

    def fake_urlopen(request, timeout):
        calls.append(request)
        if len(calls) == 1:
            raise urllib.error.URLError("temporary")
        return Response()

    monkeypatch.setattr(review.urllib.request, "urlopen", fake_urlopen)
    monkeypatch.setattr(review.time, "sleep", lambda _seconds: None)
    result = review.request_review(
        "http://127.0.0.1:1/v1", "model", {}, [(Path("crop.jpg"), b"jpeg")]
    )
    assert result["review_category"] == "box"
    assert len(calls) == 2
