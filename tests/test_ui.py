import pytest

from auditor.app import build_app, frame_review_status


@pytest.mark.parametrize(
    ("selected", "reviews", "expected"),
    [
        (False, [], "not_selected"),
        (True, [], "not_reviewed"),
        (True, [{"status": "timeout", "outcome": None}], "failed"),
        (True, [{"status": "validated", "outcome": "no_issue"}], "no_issue"),
        (
            True,
            [{"status": "validated", "outcome": "insufficient_evidence"}],
            "insufficient_evidence",
        ),
        (True, [{"status": "validated", "outcome": "findings"}], "findings"),
    ],
)
def test_unreviewed_and_failed_frames_never_appear_clean(selected, reviews, expected):
    bundle = {"vlm_selection": {"selected": selected}, "vlm_reviews": reviews}
    assert frame_review_status(bundle) == expected


def test_ui_constructs_with_the_pinned_gradio_api(engine, app_settings):
    app = build_app(app_settings, engine)
    image_components = [c for c in app.get_config_file()["components"] if c["type"] == "image"]
    assert len(image_components) == 2


def test_saved_timestamps_are_serialized_in_utc():
    from datetime import datetime, timedelta, timezone

    from auditor.db import AuditSession
    from auditor.evidence import row_dict

    record = AuditSession(
        created_at=datetime(2026, 11, 16, 12, 0, tzinfo=timezone(timedelta(hours=-5)))
    )
    assert row_dict(record)["created_at"] == "2026-11-16T17:00:00+00:00"


def test_fast_finding_switch_cannot_save_against_unloaded_evidence(engine, app_settings):
    from uuid import uuid4

    import gradio as gr

    app = build_app(app_settings, engine)
    handler = next(function.fn for function in app.fns.values() if function.fn.__name__ == "handle")
    session_id, displayed_finding, selected_finding = (str(uuid4()) for _ in range(3))
    key = dict(
        session_id=session_id,
        frame_id="frame_000000000",
        timestamp_ms=0,
        finding_id=displayed_finding,
    )
    with pytest.raises(gr.Error, match="selected finding"):
        handler(
            key,
            selected_finding,
            0,
            str(uuid4()),
            "Displayed finding note",
            session_id,
            key["frame_id"],
        )


def test_history_shows_only_the_displayed_findings_decisions(engine, app_settings, small_clip):
    from uuid import UUID, uuid4

    from auditor.contracts import FrameKey
    from auditor.evidence import frame_bundle, record_decision
    from tests.fakes import TwoFindingReviewer
    from tests.test_workflow import processed

    session_id, _, _ = processed(engine, app_settings, small_clip, TwoFindingReviewer())
    key = FrameKey(session_id=session_id, frame_id="frame_000000000", timestamp_ms=0)
    findings = frame_bundle(engine, app_settings.data_root, key)["vlm_reviews"][0]["findings"]
    record_decision(
        engine,
        app_settings.data_root,
        key,
        UUID(findings[0]["finding_id"]),
        "accept",
        "Only the first finding",
        0,
        uuid4(),
    )
    app = build_app(app_settings, engine)
    load = next(
        function.fn for function in app.fns.values() if function.fn.__name__ == "load_frame"
    )
    for finding, expected_rows in ((findings[0], 1), (findings[1], 0)):
        result = load(str(session_id), key.frame_id, preferred=finding["finding_id"])
        history = next(
            value for component, value in result.items() if component.elem_id == "decision-history"
        )
        assert len(history) == expected_rows
