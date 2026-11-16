"""Final independent review: decision revisions must not depend on wall-clock ordering."""

from datetime import UTC, datetime
from uuid import UUID

import pytest

from auditor.contracts import FrameKey
from auditor.db import Decision
from auditor.evidence import frame_bundle, record_decision
from auditor.export import export_session, verify_export
from tests.test_workflow import processed


@pytest.mark.integration
@pytest.mark.parametrize("check", ("revision_order", "export"))
def test_equal_decision_timestamps_preserve_revision_order_and_export(
    engine, app_settings, small_clip, monkeypatch, check
):
    session_id, _, _ = processed(engine, app_settings, small_clip)
    key = FrameKey(session_id=session_id, frame_id="frame_000000000", timestamp_ms=0)
    finding = frame_bundle(engine, app_settings.data_root, key)["vlm_reviews"][0]["findings"][0]
    # A tied wall clock is valid. UUID order is deliberately opposite to revision order.
    monkeypatch.setattr(
        Decision.__table__.c.created_at.default,
        "arg",
        lambda context: datetime(2026, 11, 16, 12, tzinfo=UTC),
    )
    for number, token in enumerate((UUID(int=2), UUID(int=1))):
        record_decision(
            engine,
            app_settings.data_root,
            key,
            UUID(finding["finding_id"]),
            "accept" if number == 0 else "reject",
            "Synthetic tied clock",
            number,
            token,
        )
    if check == "revision_order":
        bundle = frame_bundle(engine, app_settings.data_root, key)
        assert [decision["decision_number"] for decision in bundle["review_decisions"]] == [1, 2]
    else:
        assert verify_export(export_session(engine, app_settings.data_root, session_id))
