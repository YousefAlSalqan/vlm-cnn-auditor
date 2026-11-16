import json
from uuid import UUID, uuid4
from zipfile import ZipFile

import pytest
from sqlalchemy import func, select
from sqlalchemy.orm import Session

from auditor.contracts import ConflictError, FrameKey
from auditor.db import AuditSession, Decision, Detection, Frame
from auditor.evidence import create_upload, frame_bundle, record_decision
from auditor.export import export_session, verify_export
from auditor.pipeline import run_analysis
from tests.fakes import FakeDetector, FakeReviewer

pytestmark = pytest.mark.integration


def processed(engine, app_settings, small_clip, reviewer=None):
    session_id = create_upload(engine, app_settings, small_clip)
    reviewer = reviewer or FakeReviewer()
    progress = list(
        run_analysis(engine, app_settings, session_id, detector=FakeDetector(), reviewer=reviewer)
    )
    return session_id, reviewer, progress


def test_upload_only_saves_original_until_analysis_is_started(engine, app_settings, small_clip):
    session_id = create_upload(engine, app_settings, small_clip)
    original = app_settings.data_root / f"session_{session_id}" / "original.mp4"
    assert original.read_bytes() == small_clip.read_bytes()
    with Session(engine) as db:
        assert db.get(AuditSession, session_id).state == "uploaded"
        assert db.scalar(select(func.count()).select_from(Frame)) == 0


def test_end_to_end_identity_decisions_restart_and_export(engine, app_settings, small_clip):
    session_id, reviewer, progress = processed(engine, app_settings, small_clip)
    assert reviewer.calls == 12
    assert "review_ready" in progress[-1]
    with Session(engine) as db:
        assert db.scalar(select(func.count()).select_from(Frame)) == 15
        assert db.scalar(select(func.count()).select_from(Detection)) == 15
    key = FrameKey(session_id=session_id, frame_id="frame_000000000", timestamp_ms=0)
    bundle = frame_bundle(engine, app_settings.data_root, key)
    assert bundle["timestamp_ms"] == 0
    assert bundle["source_frame_index"] == 0
    assert bundle["yolo"]["detections"][0]["class_id"] == 41
    finding_id = UUID(bundle["vlm_reviews"][0]["findings"][0]["finding_id"])
    token = uuid4()
    record_decision(engine, app_settings.data_root, key, finding_id, "accept", "", 0, token)
    record_decision(engine, app_settings.data_root, key, finding_id, "accept", "", 0, token)
    with pytest.raises(ConflictError):
        record_decision(engine, app_settings.data_root, key, finding_id, "reject", "", 0, uuid4())
    record_decision(
        engine,
        app_settings.data_root,
        key,
        finding_id,
        "reject",
        "Human revised judgment",
        1,
        uuid4(),
    )
    with Session(engine) as db:
        assert db.scalar(select(func.count()).select_from(Decision)) == 2
        assert db.scalar(select(func.count()).select_from(Detection)) == 15
    reopened = frame_bundle(engine, app_settings.data_root, key)
    assert [d["action"] for d in reopened["review_decisions"]] == ["accept", "reject"]
    exported = export_session(engine, app_settings.data_root, session_id)
    assert verify_export(exported)
    with ZipFile(exported) as archive:
        manifest = json.loads(archive.read("manifest.json"))
        assert manifest["complete"] is True
        assert len(manifest["frames"]) == 15
        assert len(archive.namelist()) == 47
        assert archive.read("original.mp4") == small_clip.read_bytes()
        frame_json = json.loads(archive.read("json/frame_000000000.json"))
        assert frame_json["session_id"] == str(session_id)
        assert frame_json["frame_id"] == "frame_000000000"
        assert frame_json["timestamp_ms"] == 0
        assert [d["action"] for d in frame_json["review_decisions"]] == ["accept", "reject"]
        assert not any(".env" in name or ".staging" in name for name in archive.namelist())


def test_qwen_failure_keeps_manual_evidence_and_does_not_retry_timeouts(
    engine, app_settings, small_clip
):
    session_id, reviewer, progress = processed(
        engine, app_settings, small_clip, FakeReviewer(fail=True)
    )
    assert reviewer.calls == 1
    assert "partial" in progress[-1]
    key = FrameKey(session_id=session_id, frame_id="frame_000000000", timestamp_ms=0)
    bundle = frame_bundle(engine, app_settings.data_root, key)
    assert bundle["vlm_reviews"][0]["status"] == "timeout"
    assert len(bundle["yolo"]["detections"]) == 1
    assert bundle["artifacts"]["raw"]["sha256"]


def test_corrupt_media_blocks_export_instead_of_overwriting_evidence(
    engine, app_settings, small_clip
):
    session_id, _, _ = processed(engine, app_settings, small_clip)
    raw = app_settings.data_root / f"session_{session_id}" / "frames/frame_000000000.jpg"
    raw.write_bytes(b"corrupt")
    with pytest.raises(ValueError, match="[Cc]hecksum|[Cc]orrupt|[Ii]ntegrity"):
        export_session(engine, app_settings.data_root, session_id)
    assert raw.read_bytes() == b"corrupt"
