import json
from uuid import uuid4

import cv2
import httpx
import numpy as np
import pytest

from auditor.contracts import FrameKey
from auditor.models import OllamaReviewer, render_overlay


def jpeg(color):
    image = np.full((120, 160, 3), color, dtype=np.uint8)
    return cv2.imencode(".jpg", image)[1].tobytes()


def test_two_image_schema_request_produces_valid_abstention():
    key = FrameKey(session_id=uuid4(), frame_id="frame_000000001", timestamp_ms=33)
    raw = jpeg((0, 0, 255))
    annotated = jpeg((255, 0, 0))

    def handler(request):
        assert request.url.path == "/api/chat"
        payload = json.loads(request.content)
        assert payload["stream"] is False
        assert payload["format"]["type"] == "object"
        images = payload["messages"][1]["images"]
        assert len(images) == 2 and images[0] != images[1]
        content = key.model_dump(mode="json") | dict(
            outcome="insufficient_evidence", estimated_count=None, needs_review=True, findings=[]
        )
        return httpx.Response(
            200,
            json={
                "done": True,
                "model": "qwen3-vl:2b-instruct",
                "message": {"role": "assistant", "content": json.dumps(content)},
                "total_duration": 1000000,
                "eval_count": 10,
            },
        )

    reviewer = OllamaReviewer("http://127.0.0.1:11435", transport=httpx.MockTransport(handler))
    result = reviewer.review(key, raw, annotated, [])
    assert result.status == "validated"
    assert result.response.outcome == "insufficient_evidence"
    assert result.response.findings == []
    assert result.request_sha256 and result.wall_ms >= 0


@pytest.mark.parametrize("mode", ["invalid_json", "invalid_reference", "timeout", "oversize"])
def test_invalid_model_outputs_remain_diagnostic_attempts(mode):
    key = FrameKey(session_id=uuid4(), frame_id="frame_000000000", timestamp_ms=0)

    def handler(request):
        if mode == "timeout":
            raise httpx.ReadTimeout("slow model", request=request)
        content = "{" if mode == "invalid_json" else "x" * 66000
        if mode == "invalid_reference":
            content = json.dumps(
                key.model_dump(mode="json")
                | dict(
                    outcome="findings",
                    estimated_count=0,
                    needs_review=True,
                    findings=[
                        dict(
                            rule_id="R2",
                            detection_ids=[str(uuid4())],
                            evidence="box",
                            explanation="not cup",
                        )
                    ],
                )
            )
        return httpx.Response(200, json={"done": True, "message": {"content": content}})

    reviewer = OllamaReviewer("http://127.0.0.1:11435", transport=httpx.MockTransport(handler))
    result = reviewer.review(key, jpeg((0, 0, 0)), jpeg((255, 255, 255)), [])
    assert result.response is None
    assert result.status == ("invalid_json" if mode == "oversize" else mode)
    assert result.error_code
    if mode == "oversize":
        assert result.truncated is True
    if mode == "invalid_reference":
        assert "R2" in result.raw_response


def test_overlay_preserves_unannotated_bytes_and_native_geometry():
    from auditor.contracts import CupDetection

    raw = jpeg((0, 0, 0))
    before = bytes(raw)
    overlay = render_overlay(raw, [CupDetection(ordinal=0, score=0.8, bbox_xyxy=(10, 10, 80, 90))])
    assert raw == before
    decoded = cv2.imdecode(np.frombuffer(overlay, np.uint8), cv2.IMREAD_COLOR)
    assert decoded.shape == (120, 160, 3)
    assert decoded[10, 10].sum() > 0
