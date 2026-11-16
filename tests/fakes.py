import json

from auditor.contracts import CupDetection, validate_review
from auditor.models import ReviewAttempt


class FakeDetector:
    metadata = {
        "model": "synthetic-test-only",
        "weights_sha256": "a" * 64,
        "ultralytics": "fake",
        "torch": "fake",
        "device": "test",
    }

    def detect(self, raw_jpeg):
        assert raw_jpeg.startswith(b"\xff\xd8")
        return [CupDetection(ordinal=0, score=0.8, bbox_xyxy=(10, 10, 80, 90))]


class FakeReviewer:
    metadata = {
        "model": "synthetic-test-only",
        "model_digest": "test",
        "runtime_version": "test",
        "prompt_sha256": "b" * 64,
        "prompt_version": "cup_review_v1",
    }

    def __init__(self, fail=False):
        self.fail = fail
        self.calls = 0

    def identity(self):
        return self.metadata

    def prepare(self, key, raw, annotated, detections):
        return {}, ReviewAttempt(status="running", request_sha256="c" * 64)

    def review(self, key, raw, annotated, detections):
        self.calls += 1
        assert raw != annotated and len(detections) == 1
        if self.fail:
            return ReviewAttempt(
                status="timeout",
                error_code="synthetic_timeout",
                request_sha256="c" * 64,
                wall_ms=10,
            )
        finding = [
            {
                "rule_id": "R2",
                "detection_ids": [str(detections[0].detection_id)],
                "evidence": "Synthetic test evidence",
                "explanation": "Injected test finding",
            }
        ]
        payload = key.model_dump(mode="json") | dict(
            outcome="findings" if key.timestamp_ms == 0 else "no_issue",
            estimated_count=1,
            needs_review=key.timestamp_ms == 0,
            findings=finding if key.timestamp_ms == 0 else [],
        )
        raw_response = json.dumps(payload)
        response = validate_review(raw_response, key, {detections[0].detection_id}, 1)
        return ReviewAttempt(
            status="validated",
            raw_response=raw_response,
            response=response,
            request_sha256="c" * 64,
            wall_ms=10,
        )


class TwoFindingReviewer(FakeReviewer):
    def review(self, key, raw, annotated, detections):
        result = super().review(key, raw, annotated, detections)
        if key.timestamp_ms == 0:
            payload = json.loads(result.raw_response)
            payload["estimated_count"] = 2
            payload["findings"].append(
                dict(
                    rule_id="R1",
                    detection_ids=[],
                    evidence="Second synthetic finding",
                    explanation="Separate synthetic finding",
                )
            )
            result.raw_response = json.dumps(payload)
            result.response = validate_review(
                result.raw_response, key, {d.detection_id for d in detections}, len(detections)
            )
        return result
