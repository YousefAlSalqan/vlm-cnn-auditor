from uuid import uuid4

import pytest
from sqlalchemy import inspect
from sqlalchemy.exc import IntegrityError
from sqlalchemy.orm import Session

from auditor.db import AuditSession, Detection, Finding, FindingDetection, Frame, Video, VLMReview

pytestmark = pytest.mark.integration


def seed_frame(db, session_id=None, frame_index=0):
    session_id = session_id or uuid4()
    if db.get(AuditSession, session_id) is None:
        db.add(AuditSession(session_id=session_id, config={}, config_sha256="a" * 64))
        db.flush()
        video = Video(
            session_id=session_id,
            original_path="original.mp4",
            original_sha256="a" * 64,
            byte_count=100,
            container="mp4",
            codec="h264",
            stream_index=0,
            duration_us=15_000_000,
            time_base_num=1,
            time_base_den=90000,
            first_pts=0,
            rotation_degrees=0,
            timestamp_source="pts",
            probe_version="test",
            probe_data={},
        )
        db.add(video)
        db.flush()
    else:
        video = db.query(Video).filter_by(session_id=session_id).one()
    frame = Frame(
        session_id=session_id,
        frame_id=f"frame_{frame_index:09d}",
        video_id=video.video_id,
        source_frame_index=frame_index,
        source_pts=frame_index * 3000,
        timestamp_ms=0,
        width=320,
        height=240,
        json_path=f"json/frame_{frame_index:09d}.json",
    )
    db.add(frame)
    db.flush()
    return frame


def test_migration_creates_core_tables_with_foreign_keys(engine):
    inspector = inspect(engine)
    tables = set(inspector.get_table_names()) - {"alembic_version"}
    assert tables == {
        "sessions",
        "videos",
        "frames",
        "yolo_detections",
        "vlm_reviews",
        "vlm_findings",
        "finding_detections",
        "review_decisions",
    }
    assert inspector.get_foreign_keys("frames")
    assert len(inspector.get_foreign_keys("finding_detections")) == 2


def test_duplicate_timestamps_are_allowed_but_source_indices_are_unique(engine):
    with Session(engine) as db, db.begin():
        first = seed_frame(db)
        second = seed_frame(db, first.session_id, 1)
        assert first.timestamp_ms == second.timestamp_ms == 0
        session_id = first.session_id
    with pytest.raises(IntegrityError), Session(engine) as db, db.begin():
        seed_frame(db, session_id, 1)


def test_detection_and_ready_state_constraints(engine):
    with Session(engine) as db, db.begin():
        frame = seed_frame(db)
        identity = (frame.session_id, frame.frame_id)
    with pytest.raises(IntegrityError), Session(engine) as db, db.begin():
        db.add(
            Detection(
                session_id=identity[0],
                frame_id=identity[1],
                ordinal=0,
                class_id=0,
                label="person",
                score=0.8,
                x1=1,
                y1=2,
                x2=10,
                y2=20,
            )
        )
        db.flush()
    with pytest.raises(IntegrityError), Session(engine) as db, db.begin():
        frame = db.get(Frame, identity)
        frame.media_state = "ready"
        db.flush()


def test_finding_cannot_reference_a_detection_from_another_frame(engine):
    with Session(engine) as db, db.begin():
        first = seed_frame(db)
        other = seed_frame(db)
        detection = Detection(
            session_id=other.session_id,
            frame_id=other.frame_id,
            ordinal=0,
            score=0.8,
            x1=1,
            y1=2,
            x2=10,
            y2=20,
        )
        review = VLMReview(
            session_id=first.session_id,
            frame_id=first.frame_id,
            attempt_no=1,
            status="validated",
            outcome="findings",
            needs_review=True,
            model_tag="test",
            prompt_sha256="a" * 64,
            request_sha256="a" * 64,
        )
        db.add_all([detection, review])
        db.flush()
        finding = Finding(
            review_id=review.review_id,
            session_id=first.session_id,
            frame_id=first.frame_id,
            ordinal=0,
            rule_id="R2",
            evidence="suspected",
            explanation="review",
        )
        db.add(finding)
        db.flush()
        values = (finding.finding_id, detection.detection_id, first.session_id, first.frame_id)
    with pytest.raises(IntegrityError), Session(engine) as db, db.begin():
        db.add(
            FindingDetection(
                finding_id=values[0],
                detection_id=values[1],
                session_id=values[2],
                frame_id=values[3],
            )
        )
        db.flush()
