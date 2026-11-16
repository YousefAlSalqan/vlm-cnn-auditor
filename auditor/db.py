from __future__ import annotations

from datetime import UTC, datetime
from uuid import UUID, uuid4

from sqlalchemy import (
    BigInteger,
    Boolean,
    CheckConstraint,
    DateTime,
    Float,
    ForeignKey,
    ForeignKeyConstraint,
    Index,
    Integer,
    String,
    Text,
    UniqueConstraint,
    create_engine,
    text,
)
from sqlalchemy.dialects.postgresql import JSONB
from sqlalchemy.orm import DeclarativeBase, Mapped, mapped_column

from auditor.config import Settings
from auditor.contracts import AuditorError


def utcnow():
    return datetime.now(UTC)


class Base(DeclarativeBase):
    pass


class AuditSession(Base):
    __tablename__ = "sessions"
    session_id: Mapped[UUID] = mapped_column(primary_key=True, default=uuid4)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=utcnow)
    updated_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=utcnow)
    state: Mapped[str] = mapped_column(String(24), default="uploaded")
    revision: Mapped[int] = mapped_column(BigInteger, default=1)
    config: Mapped[dict] = mapped_column(JSONB)
    config_sha256: Mapped[str] = mapped_column(String(64))
    pipeline_version: Mapped[str] = mapped_column(String(64), default="0.1.0")
    cancel_requested: Mapped[bool] = mapped_column(Boolean, default=False)
    processing_started_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True))
    processing_finished_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True))
    stage_metrics: Mapped[dict] = mapped_column(JSONB, default=dict)
    last_error: Mapped[str | None] = mapped_column(Text)
    __table_args__ = (
        CheckConstraint(
            "state IN ('uploaded','processing','review_ready','partial','cancelled','interrupted','failed')",
            name="session_state",
        ),
        CheckConstraint("revision >= 1", name="session_revision"),
        Index("ix_sessions_state_updated", "state", "updated_at"),
    )


class Video(Base):
    __tablename__ = "videos"
    video_id: Mapped[UUID] = mapped_column(primary_key=True, default=uuid4)
    session_id: Mapped[UUID] = mapped_column(
        ForeignKey("sessions.session_id", ondelete="RESTRICT"), unique=True
    )
    original_path: Mapped[str] = mapped_column(Text)
    original_sha256: Mapped[str] = mapped_column(String(64))
    byte_count: Mapped[int] = mapped_column(BigInteger)
    container: Mapped[str] = mapped_column(String(8))
    codec: Mapped[str] = mapped_column(String(32))
    stream_index: Mapped[int] = mapped_column(Integer)
    duration_us: Mapped[int] = mapped_column(BigInteger)
    time_base_num: Mapped[int] = mapped_column(BigInteger)
    time_base_den: Mapped[int] = mapped_column(BigInteger)
    first_pts: Mapped[int] = mapped_column(BigInteger)
    rotation_degrees: Mapped[int] = mapped_column(Integer, default=0)
    timestamp_source: Mapped[str] = mapped_column(String(64))
    probe_version: Mapped[str] = mapped_column(Text)
    probe_data: Mapped[dict] = mapped_column(JSONB)
    __table_args__ = (
        UniqueConstraint("video_id", "session_id", name="uq_video_scope"),
        CheckConstraint(
            "byte_count > 0 AND duration_us > 0 AND time_base_num > 0 AND time_base_den > 0",
            name="video_positive_values",
        ),
    )


class Frame(Base):
    __tablename__ = "frames"
    session_id: Mapped[UUID] = mapped_column(primary_key=True)
    frame_id: Mapped[str] = mapped_column(String(32), primary_key=True)
    video_id: Mapped[UUID]
    source_frame_index: Mapped[int] = mapped_column(BigInteger)
    source_pts: Mapped[int] = mapped_column(BigInteger)
    timestamp_ms: Mapped[int] = mapped_column(BigInteger)
    width: Mapped[int] = mapped_column(Integer)
    height: Mapped[int] = mapped_column(Integer)
    raw_path: Mapped[str | None] = mapped_column(Text)
    annotated_path: Mapped[str | None] = mapped_column(Text)
    json_path: Mapped[str] = mapped_column(Text)
    raw_sha256: Mapped[str | None] = mapped_column(String(64))
    annotated_sha256: Mapped[str | None] = mapped_column(String(64))
    media_state: Mapped[str] = mapped_column(String(16), default="pending")
    vlm_selected: Mapped[bool] = mapped_column(Boolean, default=False)
    vlm_skip_reason: Mapped[str | None] = mapped_column(Text)
    revision: Mapped[int] = mapped_column(BigInteger, default=1)
    snapshot_revision: Mapped[int] = mapped_column(BigInteger, default=0)
    snapshot_sha256: Mapped[str | None] = mapped_column(String(64))
    timings: Mapped[dict] = mapped_column(JSONB, default=dict)
    error_code: Mapped[str | None] = mapped_column(Text)
    __table_args__ = (
        ForeignKeyConstraint(
            ["video_id", "session_id"],
            ["videos.video_id", "videos.session_id"],
            ondelete="RESTRICT",
            name="fk_frame_video_scope",
        ),
        UniqueConstraint("session_id", "source_frame_index", name="uq_frame_source"),
        CheckConstraint(
            "source_frame_index >= 0 AND timestamp_ms >= 0 AND width > 0 AND height > 0 AND revision >= 1 AND snapshot_revision >= 0",
            name="frame_positive_values",
        ),
        CheckConstraint(
            "frame_id = 'frame_' || lpad(source_frame_index::text, 9, '0') AND source_frame_index < 1000000000",
            name="frame_source_identity",
        ),
        CheckConstraint(
            "media_state IN ('pending','ready','failed','corrupt')", name="frame_media_state"
        ),
        CheckConstraint(
            "media_state != 'ready' OR (raw_path IS NOT NULL AND annotated_path IS NOT NULL AND raw_sha256 IS NOT NULL AND annotated_sha256 IS NOT NULL)",
            name="ready_frame_complete",
        ),
        Index("ix_frames_timeline", "session_id", "timestamp_ms", "source_frame_index"),
    )


class Detection(Base):
    __tablename__ = "yolo_detections"
    detection_id: Mapped[UUID] = mapped_column(primary_key=True, default=uuid4)
    session_id: Mapped[UUID]
    frame_id: Mapped[str] = mapped_column(String(32))
    ordinal: Mapped[int] = mapped_column(Integer)
    class_id: Mapped[int] = mapped_column(Integer, default=41)
    label: Mapped[str] = mapped_column(String(16), default="cup")
    score: Mapped[float] = mapped_column(Float)
    x1: Mapped[float] = mapped_column(Float)
    y1: Mapped[float] = mapped_column(Float)
    x2: Mapped[float] = mapped_column(Float)
    y2: Mapped[float] = mapped_column(Float)
    __table_args__ = (
        ForeignKeyConstraint(
            ["session_id", "frame_id"],
            ["frames.session_id", "frames.frame_id"],
            ondelete="RESTRICT",
            name="fk_detection_frame",
        ),
        UniqueConstraint("session_id", "frame_id", "ordinal", name="uq_detection_ordinal"),
        UniqueConstraint("detection_id", "session_id", "frame_id", name="uq_detection_scope"),
        CheckConstraint("class_id = 41 AND label = 'cup' AND ordinal >= 0", name="cup_only"),
        CheckConstraint(
            "score >= 0 AND score <= 1 AND x1 >= 0 AND y1 >= 0 AND x1 < x2 AND y1 < y2 AND x2 < 'Infinity'::float8 AND y2 < 'Infinity'::float8",
            name="detection_finite_geometry",
        ),
    )


class VLMReview(Base):
    __tablename__ = "vlm_reviews"
    review_id: Mapped[UUID] = mapped_column(primary_key=True, default=uuid4)
    session_id: Mapped[UUID]
    frame_id: Mapped[str] = mapped_column(String(32))
    attempt_no: Mapped[int] = mapped_column(Integer)
    status: Mapped[str] = mapped_column(String(32), default="running")
    outcome: Mapped[str | None] = mapped_column(String(32))
    needs_review: Mapped[bool | None] = mapped_column(Boolean)
    estimated_count: Mapped[int | None] = mapped_column(Integer)
    model_tag: Mapped[str] = mapped_column(Text)
    model_digest: Mapped[str | None] = mapped_column(Text)
    runtime_version: Mapped[str | None] = mapped_column(Text)
    prompt_version: Mapped[str] = mapped_column(Text, default="cup_review_v1")
    rule_version: Mapped[str] = mapped_column(Text, default="cup_rules_v1")
    prompt_sha256: Mapped[str] = mapped_column(String(64))
    request_sha256: Mapped[str] = mapped_column(String(64))
    input_transform: Mapped[dict] = mapped_column(JSONB, default=dict)
    runtime_metrics: Mapped[dict] = mapped_column(JSONB, default=dict)
    started_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=utcnow)
    completed_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True))
    wall_ms: Mapped[int | None] = mapped_column(BigInteger)
    raw_response: Mapped[str | None] = mapped_column(Text)
    raw_response_truncated: Mapped[bool] = mapped_column(Boolean, default=False)
    error_code: Mapped[str | None] = mapped_column(Text)
    runtime_uncertain: Mapped[bool] = mapped_column(
        Boolean, default=False, server_default=text("false")
    )
    runtime_restart_confirmed_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True))
    __table_args__ = (
        ForeignKeyConstraint(
            ["session_id", "frame_id"],
            ["frames.session_id", "frames.frame_id"],
            ondelete="RESTRICT",
            name="fk_review_frame",
        ),
        UniqueConstraint("session_id", "frame_id", "attempt_no", name="uq_review_attempt"),
        UniqueConstraint("review_id", "session_id", "frame_id", name="uq_review_scope"),
        CheckConstraint("attempt_no IN (1,2)", name="bounded_review_attempts"),
        CheckConstraint(
            "status IN ('running','validated','invalid_json','invalid_reference','timeout','transport_error','oom','cancelled')",
            name="review_status",
        ),
        CheckConstraint(
            "outcome IS NULL OR outcome IN ('findings','no_issue','insufficient_evidence')",
            name="review_outcome",
        ),
        CheckConstraint(
            "status != 'validated' OR (outcome IS NOT NULL AND needs_review IS NOT NULL)",
            name="validated_review_outcome",
        ),
        CheckConstraint(
            "estimated_count IS NULL OR estimated_count BETWEEN 0 AND 100", name="review_count"
        ),
        Index(
            "uq_validated_review",
            "session_id",
            "frame_id",
            unique=True,
            postgresql_where=text("status = 'validated'"),
        ),
    )


class Finding(Base):
    __tablename__ = "vlm_findings"
    finding_id: Mapped[UUID] = mapped_column(primary_key=True, default=uuid4)
    review_id: Mapped[UUID]
    session_id: Mapped[UUID]
    frame_id: Mapped[str] = mapped_column(String(32))
    ordinal: Mapped[int] = mapped_column(Integer)
    rule_id: Mapped[str] = mapped_column(String(2))
    evidence: Mapped[str] = mapped_column(Text)
    explanation: Mapped[str] = mapped_column(Text)
    __table_args__ = (
        ForeignKeyConstraint(
            ["review_id", "session_id", "frame_id"],
            ["vlm_reviews.review_id", "vlm_reviews.session_id", "vlm_reviews.frame_id"],
            ondelete="RESTRICT",
            name="fk_finding_review_scope",
        ),
        UniqueConstraint("review_id", "ordinal", name="uq_finding_ordinal"),
        UniqueConstraint("finding_id", "session_id", "frame_id", name="uq_finding_scope"),
        CheckConstraint(
            "rule_id IN ('R1','R2','R3','R4') AND ordinal >= 0 AND length(evidence) BETWEEN 1 AND 1000 AND length(explanation) BETWEEN 1 AND 1000",
            name="finding_values",
        ),
        Index("ix_finding_rule", "rule_id", "review_id"),
    )


class FindingDetection(Base):
    __tablename__ = "finding_detections"
    finding_id: Mapped[UUID] = mapped_column(primary_key=True)
    detection_id: Mapped[UUID] = mapped_column(primary_key=True)
    session_id: Mapped[UUID]
    frame_id: Mapped[str] = mapped_column(String(32))
    __table_args__ = (
        ForeignKeyConstraint(
            ["finding_id", "session_id", "frame_id"],
            ["vlm_findings.finding_id", "vlm_findings.session_id", "vlm_findings.frame_id"],
            ondelete="RESTRICT",
            name="fk_reference_finding_scope",
        ),
        ForeignKeyConstraint(
            ["detection_id", "session_id", "frame_id"],
            [
                "yolo_detections.detection_id",
                "yolo_detections.session_id",
                "yolo_detections.frame_id",
            ],
            ondelete="RESTRICT",
            name="fk_reference_detection_scope",
        ),
        Index("ix_reference_detection", "detection_id"),
    )


class Decision(Base):
    __tablename__ = "review_decisions"
    decision_id: Mapped[UUID] = mapped_column(primary_key=True, default=uuid4)
    finding_id: Mapped[UUID] = mapped_column(
        ForeignKey("vlm_findings.finding_id", ondelete="RESTRICT")
    )
    decision_number: Mapped[int] = mapped_column(Integer)
    action: Mapped[str] = mapped_column(String(8))
    note: Mapped[str | None] = mapped_column(Text)
    reviewer_label: Mapped[str] = mapped_column(String(64), default="local")
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=utcnow)
    supersedes_decision_id: Mapped[UUID | None]
    __table_args__ = (
        UniqueConstraint("decision_id", "finding_id", name="uq_decision_scope"),
        ForeignKeyConstraint(
            ["supersedes_decision_id", "finding_id"],
            ["review_decisions.decision_id", "review_decisions.finding_id"],
            ondelete="RESTRICT",
            name="fk_previous_decision_same_finding",
        ),
        UniqueConstraint("finding_id", "decision_number", name="uq_decision_number"),
        CheckConstraint(
            "action IN ('accept','reject','defer') AND decision_number >= 1 AND (note IS NULL OR length(note) <= 2000)",
            name="decision_values",
        ),
    )


def create_engine_for(settings: Settings):
    return create_engine(settings.database_url.get_secret_value(), pool_pre_ping=True)


def require_schema(engine):
    try:
        with engine.connect() as connection:
            revision = connection.scalar(text("SELECT version_num FROM alembic_version"))
    except Exception as exc:
        raise AuditorError(
            "Database is unavailable or unmigrated. Run python -m alembic upgrade head."
        ) from exc
    if revision != "0002_review_recovery":
        raise AuditorError(
            "Database schema differs from this app. Check migrations before continuing."
        )
