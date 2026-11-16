from __future__ import annotations

import hashlib
import os
import shutil
from datetime import UTC, datetime
from pathlib import Path
from uuid import UUID, uuid4

from sqlalchemy import func, select, update
from sqlalchemy.orm import Session

from auditor.config import Settings
from auditor.contracts import AuditorError, ConflictError, FrameKey
from auditor.db import (
    AuditSession,
    Decision,
    Detection,
    Finding,
    FindingDetection,
    Frame,
    Video,
    VLMReview,
    utcnow,
)
from auditor.storage import (
    atomic_write,
    canonical_json,
    copy_original,
    safe_path,
    session_dir,
    session_lock,
    sha256_file,
)
from auditor.video import validate_and_probe


def row_dict(row) -> dict:
    values = {}
    for column in row.__table__.columns:
        value = getattr(row, column.name)
        if isinstance(value, UUID):
            value = str(value)
        elif isinstance(value, datetime):
            value = value.astimezone(UTC).isoformat()
        values[column.name] = value
    return values


def bump_revision(db: Session, session_id: UUID, frame: Frame | None = None):
    session = db.scalar(
        select(AuditSession).where(AuditSession.session_id == session_id).with_for_update()
    )
    session.revision += 1
    session.updated_at = utcnow()
    if frame is not None:
        frame.revision += 1


def create_upload(engine, settings: Settings, source: Path) -> UUID:
    source = Path(source)
    probe = validate_and_probe(source, settings)
    settings.data_root.mkdir(parents=True, exist_ok=True)
    if shutil.disk_usage(settings.data_root).free < 1024**3 + 2 * source.stat().st_size:
        raise AuditorError("Not enough free disk space to preserve and export this session")
    session_id = uuid4()
    directory = session_dir(settings.data_root, session_id)
    with session_lock(settings.data_root, session_id):
        name, digest, size = copy_original(source, directory, probe.container)
        probe = validate_and_probe(safe_path(directory, name), settings)
        config = settings.analysis_config()
        with Session(engine) as db, db.begin():
            db.add(
                AuditSession(
                    session_id=session_id,
                    config=config,
                    config_sha256=hashlib.sha256(canonical_json(config)).hexdigest(),
                )
            )
            db.flush()
            db.add(
                Video(
                    session_id=session_id,
                    original_path=name,
                    original_sha256=digest,
                    byte_count=size,
                    container=probe.container,
                    codec=probe.codec,
                    stream_index=probe.stream_index,
                    duration_us=probe.duration_us,
                    time_base_num=probe.time_base_num,
                    time_base_den=probe.time_base_den,
                    first_pts=probe.first_pts,
                    rotation_degrees=probe.rotation_degrees,
                    timestamp_source=probe.timestamp_source,
                    probe_version=probe.probe_version,
                    probe_data=probe.model_dump(mode="json"),
                )
            )
    return session_id


def list_sessions(engine, limit: int = 50, offset: int = 0) -> list[dict]:
    frame_counts = (
        select(Frame.session_id, func.count().label("count")).group_by(Frame.session_id).subquery()
    )
    finding_counts = (
        select(Finding.session_id, func.count().label("count"))
        .group_by(Finding.session_id)
        .subquery()
    )
    query = (
        select(
            AuditSession,
            func.coalesce(frame_counts.c.count, 0),
            func.coalesce(finding_counts.c.count, 0),
        )
        .outerjoin(frame_counts, frame_counts.c.session_id == AuditSession.session_id)
        .outerjoin(finding_counts, finding_counts.c.session_id == AuditSession.session_id)
        .order_by(AuditSession.created_at.desc())
        .limit(min(max(limit, 1), 100))
        .offset(max(offset, 0))
    )
    with Session(engine) as db:
        return [
            dict(
                session_id=str(session.session_id),
                created_at=session.created_at.isoformat(),
                state=session.state,
                frames=frames,
                findings=findings,
                last_error=session.last_error,
            )
            for session, frames, findings in db.execute(query)
        ]


def session_summary(engine, session_id: UUID) -> dict:
    with Session(engine) as db:
        session = db.get(AuditSession, session_id)
        if session is None:
            raise AuditorError("Session does not exist")
        video = db.scalar(select(Video).where(Video.session_id == session_id))
        return row_dict(session) | {"video": row_dict(video)}


def list_frames(engine, session_id: UUID) -> list[dict]:
    counts = (
        select(Finding.frame_id, func.count().label("count"))
        .where(Finding.session_id == session_id)
        .group_by(Finding.frame_id)
        .subquery()
    )
    query = (
        select(Frame, func.coalesce(counts.c.count, 0))
        .outerjoin(counts, counts.c.frame_id == Frame.frame_id)
        .where(Frame.session_id == session_id)
        .order_by(
            func.coalesce(counts.c.count, 0).desc(), Frame.timestamp_ms, Frame.source_frame_index
        )
    )
    with Session(engine) as db:
        return [
            dict(
                frame_id=f.frame_id,
                timestamp_ms=f.timestamp_ms,
                findings=count,
                media_state=f.media_state,
                selected=f.vlm_selected,
            )
            for f, count in db.execute(query)
        ]


def analysis_running(engine) -> bool:
    with Session(engine) as db:
        return (
            db.scalar(
                select(AuditSession.session_id).where(AuditSession.state == "processing").limit(1)
            )
            is not None
        )


def _payload(db: Session, frame: Frame) -> dict:
    session = db.get(AuditSession, frame.session_id)
    video = db.get(Video, frame.video_id)
    detections = db.scalars(
        select(Detection)
        .where(Detection.session_id == frame.session_id, Detection.frame_id == frame.frame_id)
        .order_by(Detection.ordinal)
    ).all()
    reviews = db.scalars(
        select(VLMReview)
        .where(VLMReview.session_id == frame.session_id, VLMReview.frame_id == frame.frame_id)
        .order_by(VLMReview.attempt_no)
    ).all()
    review_ids = [r.review_id for r in reviews]
    findings = (
        db.scalars(
            select(Finding).where(Finding.review_id.in_(review_ids)).order_by(Finding.ordinal)
        ).all()
        if reviews
        else []
    )
    finding_ids = [f.finding_id for f in findings]
    references = (
        db.scalars(
            select(FindingDetection).where(FindingDetection.finding_id.in_(finding_ids))
        ).all()
        if findings
        else []
    )
    decisions = (
        db.scalars(
            select(Decision)
            .where(Decision.finding_id.in_(finding_ids))
            .order_by(Decision.finding_id, Decision.decision_number)
        ).all()
        if findings
        else []
    )
    reference_map = {f.finding_id: [] for f in findings}
    for ref in references:
        reference_map[ref.finding_id].append(str(ref.detection_id))
    finding_map = {review_id: [] for review_id in review_ids}
    for finding in findings:
        finding_map[finding.review_id].append(
            row_dict(finding) | {"detection_ids": sorted(reference_map[finding.finding_id])}
        )
    yolo = session.config.get("yolo", {})
    return {
        "schema_version": 1,
        "snapshot_revision": frame.revision,
        "session_id": str(frame.session_id),
        "video_id": str(frame.video_id),
        "frame_id": frame.frame_id,
        "source_frame_index": frame.source_frame_index,
        "timestamp_ms": frame.timestamp_ms,
        "presentation": {
            "source_pts": frame.source_pts,
            "first_pts": video.first_pts,
            "time_base_num": video.time_base_num,
            "time_base_den": video.time_base_den,
            "timestamp_source": video.timestamp_source,
            "rotation_degrees": video.rotation_degrees,
        },
        "image": {"width": frame.width, "height": frame.height, "bbox_format": "xyxy_pixels"},
        "artifacts": {
            "raw": {"path": frame.raw_path, "sha256": frame.raw_sha256},
            "annotated": {"path": frame.annotated_path, "sha256": frame.annotated_sha256},
        },
        "yolo": {
            "model_name": yolo.get("model"),
            "weights_sha256": yolo.get("weights_sha256"),
            "package_version": yolo.get("ultralytics"),
            "settings": yolo,
            "detections": [
                dict(
                    detection_id=str(d.detection_id),
                    ordinal=d.ordinal,
                    class_id=d.class_id,
                    label=d.label,
                    score=d.score,
                    bbox_xyxy=[d.x1, d.y1, d.x2, d.y2],
                )
                for d in detections
            ],
        },
        "vlm_selection": {
            "selected": frame.vlm_selected,
            "reason": frame.vlm_skip_reason or "uniform bounded selection",
        },
        "vlm_reviews": [
            row_dict(r) | {"response_schema_version": 1, "findings": finding_map[r.review_id]}
            for r in reviews
        ],
        "review_decisions": [row_dict(d) for d in decisions],
        "human_annotations": [],
        "reference_adjudications": [],
    }


def verify_media(directory: Path, frame: Frame):
    for path, digest in [
        (frame.raw_path, frame.raw_sha256),
        (frame.annotated_path, frame.annotated_sha256),
    ]:
        if not path or not digest:
            raise AuditorError("Frame evidence is incomplete")
        file = safe_path(directory, path)
        if not file.is_file() or sha256_file(file) != digest:
            raise AuditorError(
                f"Checksum/integrity failure for {frame.frame_id}. Evidence was not overwritten."
            )


def frame_bundle(engine, root: Path, key: FrameKey) -> dict:
    with session_lock(root, key.session_id), Session(engine) as db:
        frame = db.get(Frame, (key.session_id, key.frame_id))
        if frame is None or frame.timestamp_ms != key.timestamp_ms:
            raise AuditorError("Frame identity does not match this session and timestamp")
        if frame.media_state != "ready":
            raise AuditorError("Frame evidence is not ready or has been marked corrupt")
        verify_media(session_dir(root, key.session_id), frame)
        return _payload(db, frame)


def publish_snapshot(engine, root: Path, key: FrameKey) -> dict:
    with session_lock(root, key.session_id):
        with Session(engine) as db:
            frame = db.get(Frame, (key.session_id, key.frame_id))
            if (
                frame is None
                or frame.timestamp_ms != key.timestamp_ms
                or frame.media_state != "ready"
            ):
                raise AuditorError("Cannot publish an incomplete or mismatched frame")
            payload = _payload(db, frame)
            revision, relative = frame.revision, frame.json_path
        path = safe_path(session_dir(root, key.session_id), relative)
        digest = atomic_write(path, canonical_json(payload))
        with Session(engine) as db, db.begin():
            changed = db.execute(
                update(Frame)
                .where(
                    Frame.session_id == key.session_id,
                    Frame.frame_id == key.frame_id,
                    Frame.revision == revision,
                )
                .values(snapshot_revision=revision, snapshot_sha256=digest)
            ).rowcount
            if changed != 1:
                raise ConflictError(
                    "Evidence changed during JSON publication. Retry to reconcile it."
                )
        return {"path": relative, "sha256": digest, "revision": revision}


def reconcile_session(engine, root: Path, session_id: UUID):
    with session_lock(root, session_id):
        directory = session_dir(root, session_id)
        with Session(engine) as db:
            video = db.scalar(select(Video).where(Video.session_id == session_id))
            if (
                video is None
                or sha256_file(safe_path(directory, video.original_path)) != video.original_sha256
            ):
                raise AuditorError("Original video checksum/integrity failure")
            keys = [
                (f.frame_id, f.timestamp_ms)
                for f in db.scalars(
                    select(Frame)
                    .where(Frame.session_id == session_id)
                    .order_by(Frame.source_frame_index)
                )
            ]
        for frame_id, timestamp in keys:
            with Session(engine) as db, db.begin():
                frame = db.get(Frame, (session_id, frame_id))
                if frame.media_state == "pending":
                    for kind, relative, digest in [
                        ("raw", frame.raw_path, frame.raw_sha256),
                        ("annotated", frame.annotated_path, frame.annotated_sha256),
                    ]:
                        if not relative or not digest:
                            raise AuditorError("Pending frame has incomplete preparation metadata")
                        final = safe_path(directory, relative)
                        staged = safe_path(directory, f".staging/{kind}_{frame_id}.jpg")
                        if not final.exists():
                            if not staged.is_file() or sha256_file(staged) != digest:
                                raise AuditorError("Pending evidence is missing or corrupt")
                            final.parent.mkdir(parents=True, exist_ok=True)
                            os.replace(staged, final)
                    verify_media(directory, frame)
                    frame.media_state = "ready"
                    bump_revision(db, session_id, frame)
                elif frame.media_state != "ready":
                    raise AuditorError("Corrupt or failed evidence requires explicit repair")
                verify_media(directory, frame)
            publish_snapshot(
                engine,
                root,
                FrameKey(session_id=session_id, frame_id=frame_id, timestamp_ms=timestamp),
            )


def record_decision(
    engine,
    root: Path,
    key: FrameKey,
    finding_id: UUID,
    action: str,
    note: str,
    expected_number: int,
    decision_id: UUID,
) -> dict:
    if action not in ("accept", "reject", "defer") or len(note) > 2000 or expected_number < 0:
        raise AuditorError("Choose Accept, Reject or Defer and keep notes under 2000 characters")
    with session_lock(root, key.session_id):
        with Session(engine) as db, db.begin():
            frame = db.get(Frame, (key.session_id, key.frame_id))
            finding = db.scalar(
                select(Finding).where(Finding.finding_id == finding_id).with_for_update()
            )
            if (
                frame is None
                or frame.timestamp_ms != key.timestamp_ms
                or finding is None
                or (finding.session_id, finding.frame_id) != (key.session_id, key.frame_id)
            ):
                raise AuditorError("Decision references a different frame or finding")
            existing = db.get(Decision, decision_id)
            if existing:
                if (existing.finding_id, existing.action, existing.note or "") != (
                    finding_id,
                    action,
                    note.strip(),
                ):
                    raise ConflictError(
                        "This decision token was already used for a different action"
                    )
                result = row_dict(existing)
            else:
                latest = db.scalar(
                    select(Decision)
                    .where(Decision.finding_id == finding_id)
                    .order_by(Decision.decision_number.desc())
                    .limit(1)
                )
                current = latest.decision_number if latest else 0
                if current != expected_number:
                    raise ConflictError(
                        "This finding changed. Reload it before recording another decision."
                    )
                decision = Decision(
                    decision_id=decision_id,
                    finding_id=finding_id,
                    decision_number=current + 1,
                    action=action,
                    note=note.strip() or None,
                    supersedes_decision_id=latest.decision_id if latest else None,
                )
                db.add(decision)
                bump_revision(db, key.session_id, frame)
                db.flush()
                result = row_dict(decision)
        try:
            publish_snapshot(engine, root, key)
            result["snapshot_pending"] = False
        except OSError:
            result["snapshot_pending"] = True
        return result
