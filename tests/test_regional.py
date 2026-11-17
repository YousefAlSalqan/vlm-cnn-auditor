import json
from uuid import uuid4

import httpx
import pytest

from auditor.contracts import CupDetection, FrameKey
from auditor.export import _verify_review_evidence
from auditor.regional import (
    PROMPT_SHA256,
    PROTOCOL,
    RegionalReviewer,
    region_observations,
    validate_stored_review,
)
from tests.test_models import jpeg


def test_region_plan_preserves_valid_thin_frame_dimensions():
    from auditor.regional import region_plan

    assert region_plan(160, 2, []) == [
        {"index": 0, "native_xyxy": [0, 0, 160, 1]},
        {"index": 1, "native_xyxy": [0, 1, 160, 2]},
    ]


def fixture_input():
    return FrameKey(session_id=uuid4(), frame_id="frame_000000005", timestamp_ms=167), jpeg(
        (40, 70, 100)
    )


def test_default_regional_review_preserves_raw_observations_and_host_identity():
    key, raw = fixture_input()
    calls = []

    def handler(request):
        payload = json.loads(request.content)
        calls.append(payload)
        assert len(payload["messages"][1]["images"]) == 1
        assert str(key.session_id) not in json.dumps(payload["messages"])
        assert payload["options"] == {"temperature": 0, "num_ctx": 8192, "num_predict": 1024}
        content = {
            "cup_present": "yes" if len(calls) == 1 else "no",
            "evidence": "A white mug with a handle.",
        }
        return httpx.Response(200, json={"done": True, "message": {"content": json.dumps(content)}})

    reviewer = RegionalReviewer("http://127.0.0.1:11435", transport=httpx.MockTransport(handler))
    _, prepared = reviewer.prepare(key, raw, raw, [])
    result = reviewer.review(key, raw, raw, [])
    assert len(calls) == 3 and result.status == "validated"
    assert result.request_sha256 == prepared.request_sha256
    assert result.response.frame_id == key.frame_id
    assert result.response.findings[0].detection_ids == []
    envelope = json.loads(result.raw_response)
    assert len(envelope["regions"]) == 3
    assert json.loads(envelope["regions"][0]["raw_response"])["cup_present"] == "yes"
    assert validate_stored_review(result.raw_response, key, [], result.transform) == result.response
    assert "Top area: yes" in region_observations(result.raw_response)
    envelope["regions"][0]["native_xyxy"][1] = 1
    with pytest.raises(ValueError):
        validate_stored_review(json.dumps(envelope), key, [], result.transform)


def test_occupied_frame_is_still_sent_to_qwen_but_is_not_cleared():
    key, raw = fixture_input()
    detections = [CupDetection(ordinal=0, score=0.9, bbox_xyxy=(0.0, 0.0, 160.0, 120.0))]
    calls = []

    def handler(request):
        calls.append(request)
        return httpx.Response(
            200,
            json={
                "done": True,
                "message": {
                    "content": json.dumps(
                        {"cup_present": "yes", "evidence": "A white cup is visible."}
                    )
                },
            },
        )

    reviewer = RegionalReviewer("http://127.0.0.1:11435", transport=httpx.MockTransport(handler))
    result = reviewer.review(key, raw, raw, detections)
    assert len(calls) == 1 and result.status == "validated"
    assert result.response.outcome == "insufficient_evidence"
    assert not result.response.findings


def test_uncertain_region_stops_before_any_later_request():
    key, raw = fixture_input()
    calls = []

    def handler(request):
        calls.append(request)
        raise httpx.ReadTimeout("uncertain", request=request)

    reviewer = RegionalReviewer("http://127.0.0.1:11435", transport=httpx.MockTransport(handler))
    _, prepared = reviewer.prepare(key, raw, raw, [])
    result = reviewer.review(key, raw, raw, [])
    assert len(calls) == 1 and result.status == "timeout" and result.uncertain
    assert result.response is None and result.request_sha256 == prepared.request_sha256


def test_regional_validation_keeps_legacy_raw_responses_readable():
    key, _ = fixture_input()
    old = key.model_dump(mode="json") | dict(
        outcome="no_issue", estimated_count=0, needs_review=False, findings=[]
    )
    assert validate_stored_review(json.dumps(old), key, [], {}).outcome == "no_issue"
    assert region_observations(json.dumps(old)) == ""


def test_export_replays_regional_raw_evidence_against_actual_source_pixels():
    key, raw = fixture_input()

    def handler(request):
        return httpx.Response(
            200,
            json={
                "done": True,
                "message": {
                    "content": json.dumps(
                        {"cup_present": "yes", "evidence": "Synthetic cup for export validation."}
                    )
                },
            },
        )

    reviewer = RegionalReviewer("http://127.0.0.1:11435", transport=httpx.MockTransport(handler))
    result = reviewer.review(key, raw, raw, [])
    review_id = str(uuid4())
    review = key.model_dump(mode="json") | {
        "review_id": review_id,
        "attempt_no": 1,
        "response_schema_version": 1,
        "rule_version": "cup_rules_v1",
        "model_tag": reviewer.model,
        "model_digest": None,
        "runtime_version": None,
        "prompt_version": PROTOCOL,
        "prompt_sha256": PROMPT_SHA256,
        "request_sha256": result.request_sha256,
        "input_transform": result.transform,
        "raw_response": result.raw_response,
        "raw_response_truncated": False,
        "status": "validated",
        "outcome": result.response.outcome,
        "needs_review": True,
        "estimated_count": None,
        "findings": [
            key.model_dump(mode="json")
            | {
                "finding_id": str(uuid4()),
                "review_id": review_id,
                "ordinal": 0,
                **result.response.findings[0].model_dump(mode="json"),
            }
        ],
    }
    payload = key.model_dump(mode="json") | {
        "image": {"width": 160, "height": 120},
        "yolo": {"detections": []},
        "vlm_reviews": [review],
        "review_decisions": [],
    }
    configuration = {"rule_version": "cup_rules_v1", "qwen": reviewer.metadata}
    _verify_review_evidence(payload, configuration, set(), raw)
    with pytest.raises(ValueError, match="source image"):
        _verify_review_evidence(payload, configuration, set(), jpeg((0, 0, 0)))
