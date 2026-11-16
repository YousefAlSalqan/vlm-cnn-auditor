from uuid import uuid4

import pytest
from pydantic import ValidationError

from auditor.contracts import CupDetection, FrameKey, validate_review


def test_frame_identity_rejects_unsafe_or_incomplete_keys():
    with pytest.raises(ValidationError):
        FrameKey(session_id=uuid4(), frame_id="../original", timestamp_ms=0)
    with pytest.raises(ValidationError):
        FrameKey(session_id=uuid4(), frame_id="frame_000000001", timestamp_ms=-1)


def test_detection_rejects_non_cup_nonfinite_and_inverted_boxes():
    base = dict(detection_id=uuid4(), ordinal=0, score=0.8, bbox_xyxy=(1.0, 2.0, 9.0, 10.0))
    for changes in [dict(class_id=0), dict(score=float("nan")), dict(bbox_xyxy=(9, 2, 1, 10))]:
        with pytest.raises(ValidationError):
            CupDetection(**(base | changes))


def response(key, **changes):
    return (
        key.model_dump(mode="json")
        | dict(outcome="no_issue", estimated_count=1, needs_review=False, findings=[])
        | changes
    )


def test_review_checks_identity_references_and_outcome_consistency():
    import json

    key = FrameKey(session_id=uuid4(), frame_id="frame_000000120", timestamp_ms=4000)
    detection_id = uuid4()
    valid = response(key)
    assert validate_review(json.dumps(valid), key, {detection_id}, 1).outcome == "no_issue"
    bad_outputs = [
        response(key, timestamp_ms=4001),
        response(key, estimated_count=0),
        response(key, needs_review=True),
        response(key, outcome="insufficient_evidence", needs_review=True, estimated_count=1),
        response(key, unexpected="field"),
        response(
            key,
            outcome="findings",
            needs_review=True,
            findings=[
                dict(
                    rule_id="R2",
                    detection_ids=[str(uuid4())],
                    evidence="box",
                    explanation="not a cup",
                )
            ],
        ),
        response(
            key,
            outcome="findings",
            needs_review=True,
            findings=[
                dict(
                    rule_id="R3",
                    detection_ids=[str(detection_id), str(detection_id)],
                    evidence="boxes",
                    explanation="duplicate",
                )
            ],
        ),
    ]
    for invalid in bad_outputs:
        with pytest.raises((ValidationError, ValueError)):
            validate_review(json.dumps(invalid), key, {detection_id}, 1)


def test_abstention_and_missed_cup_are_valid_but_not_verified_truth():
    import json

    key = FrameKey(session_id=uuid4(), frame_id="frame_000000000", timestamp_ms=0)
    abstain = response(
        key, outcome="insufficient_evidence", estimated_count=None, needs_review=True
    )
    assert validate_review(json.dumps(abstain), key, set(), 0).findings == []
    miss = response(
        key,
        outcome="findings",
        estimated_count=1,
        needs_review=True,
        findings=[
            dict(
                rule_id="R1", detection_ids=[], evidence="cup visible", explanation="possible miss"
            )
        ],
    )
    assert validate_review(json.dumps(miss), key, set(), 0).findings[0].rule_id == "R1"
