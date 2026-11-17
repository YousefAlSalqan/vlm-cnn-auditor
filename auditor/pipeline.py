from __future__ import annotations

import hashlib
import time
from pathlib import Path
from uuid import UUID

import cv2
from filelock import FileLock, Timeout
from sqlalchemy import func, select, update
from sqlalchemy.orm import Session

from auditor.config import Settings
from auditor.contracts import AuditorError, CupDetection, FrameKey, VideoProbe
from auditor.db import (
    AuditSession,
    Detection,
    Finding,
    FindingDetection,
    Frame,
    Video,
    VLMReview,
    utcnow,
)
from auditor.evidence import bump_revision, publish_snapshot, reconcile_session, verify_media
from auditor.models import PROMPT_SHA256, OllamaReviewer, YoloDetector, render_overlay
from auditor.regional import PROTOCOL, RegionalReviewer, validate_stored_review
from auditor.storage import (
    atomic_write,
    canonical_json,
    safe_path,
    session_dir,
    session_lock,
    sha256_file,
)
from auditor.video import iter_sampled_frames


def select_review_frames(keys: list[FrameKey], budget: int | None = None) -> list[FrameKey]:
    if budget is None:
        return list(keys)
    if not keys:
        return []
    if not 1 <= budget <= 12:
        raise AuditorError("Qwen review budget must be between 1 and 12")
    if len(keys) <= budget:
        return list(keys)
    if budget == 1:
        return [keys[0]]
    return [keys[i * (len(keys) - 1) // (budget - 1)] for i in range(budget)]


def request_cancel(engine, session_id: UUID) -> bool:
    with Session(engine) as db, db.begin():
        return bool(
            db.execute(
                update(AuditSession)
                .where(AuditSession.session_id == session_id, AuditSession.state == "processing")
                .values(cancel_requested=True)
            ).rowcount
        )


def recover_interrupted(engine):
    with Session(engine) as db, db.begin():
        db.execute(
            update(AuditSession)
            .where(AuditSession.state == "processing")
            .values(
                state="interrupted",
                last_error="App stopped during analysis. Completed evidence was retained.",
                revision=AuditSession.revision + 1,
                updated_at=utcnow(),
            )
        )
        reviews = db.scalars(select(VLMReview).where(VLMReview.status == "running")).all()
        for review in reviews:
            review.status, review.error_code = "transport_error", "interrupted"
            review.runtime_uncertain = True
            review.completed_at = utcnow()
            frame = db.get(Frame, (review.session_id, review.frame_id))
            bump_revision(db, review.session_id, frame)


def _cancelled(engine, session_id):
    with Session(engine) as db:
        return bool(db.get(AuditSession, session_id).cancel_requested)


def _terminal(engine, session_id, state, message=None, elapsed=None):
    with Session(engine) as db, db.begin():
        session = db.get(AuditSession, session_id)
        if session is None:
            return
        session.state = state
        session.last_error = message
        session.processing_finished_at = utcnow()
        if elapsed is not None:
            session.stage_metrics = session.stage_metrics | {
                "last_run_wall_ms": round(elapsed * 1000)
            }
        bump_revision(db, session_id)


def _code_digest():
    digest = hashlib.sha256()
    for name in ("contracts.py", "video.py", "models.py", "regional.py", "pipeline.py"):
        digest.update((Path(__file__).parent / name).read_bytes())
    return digest.hexdigest()


def _confirm_runtime_restart(engine, root):
    with Session(engine) as db:
        session_ids = db.scalars(
            select(VLMReview.session_id)
            .where(
                VLMReview.runtime_uncertain.is_(True),
                VLMReview.runtime_restart_confirmed_at.is_(None),
            )
            .distinct()
            .order_by(VLMReview.session_id)
        ).all()
    for session_id in session_ids:
        with session_lock(root, session_id):
            keys = []
            with Session(engine) as db, db.begin():
                reviews = db.scalars(
                    select(VLMReview).where(
                        VLMReview.session_id == session_id,
                        VLMReview.runtime_uncertain.is_(True),
                        VLMReview.runtime_restart_confirmed_at.is_(None),
                    )
                ).all()
                for review in reviews:
                    review.runtime_restart_confirmed_at = utcnow()
                    if review.status == "running":
                        review.status, review.error_code = "transport_error", "interrupted"
                        review.completed_at = utcnow()
                    frame = db.get(Frame, (session_id, review.frame_id))
                    bump_revision(db, session_id, frame)
                    keys.append(
                        FrameKey(
                            session_id=session_id,
                            frame_id=frame.frame_id,
                            timestamp_ms=frame.timestamp_ms,
                        )
                    )
            for key in keys:
                publish_snapshot(engine, root, key)


def _commit_frame(
    engine, root, session_id, video_id, sampled, detections, yolo_ms, *, select_for_review=False
):
    directory = session_dir(root, session_id)
    frame_id = sampled.key.frame_id
    raw_path, annotated_path = f"frames/{frame_id}.jpg", f"annotated/{frame_id}.jpg"
    annotated = render_overlay(sampled.raw_jpeg, detections)
    raw_hash = atomic_write(
        safe_path(directory, f".staging/raw_{frame_id}.jpg"), sampled.raw_jpeg, immutable=True
    )
    annotated_hash = atomic_write(
        safe_path(directory, f".staging/annotated_{frame_id}.jpg"), annotated, immutable=True
    )
    with Session(engine) as db, db.begin():
        frame = Frame(
            session_id=session_id,
            frame_id=frame_id,
            video_id=video_id,
            source_frame_index=sampled.source_frame_index,
            source_pts=sampled.source_pts,
            timestamp_ms=sampled.key.timestamp_ms,
            width=sampled.width,
            height=sampled.height,
            raw_path=raw_path,
            annotated_path=annotated_path,
            json_path=f"json/{frame_id}.json",
            raw_sha256=raw_hash,
            annotated_sha256=annotated_hash,
            timings={"yolo_wall_ms": yolo_ms},
            vlm_selected=select_for_review,
            vlm_skip_reason="same source frames as YOLO" if select_for_review else None,
        )
        db.add(frame)
        db.flush()
        for detection in detections:
            x1, y1, x2, y2 = detection.bbox_xyxy
            if x2 > sampled.width or y2 > sampled.height:
                raise AuditorError("Detector box does not fit the canonical frame")
            db.add(
                Detection(
                    detection_id=detection.detection_id,
                    session_id=session_id,
                    frame_id=frame_id,
                    ordinal=detection.ordinal,
                    class_id=detection.class_id,
                    label=detection.label,
                    score=detection.score,
                    x1=x1,
                    y1=y1,
                    x2=x2,
                    y2=y2,
                )
            )
        bump_revision(db, session_id)
    for kind, relative in [("raw", raw_path), ("annotated", annotated_path)]:
        staged = safe_path(directory, f".staging/{kind}_{frame_id}.jpg")
        final = safe_path(directory, relative)
        final.parent.mkdir(parents=True, exist_ok=True)
        if final.exists():
            expected = raw_hash if kind == "raw" else annotated_hash
            if sha256_file(final) != expected:
                raise AuditorError("Existing immutable frame has different bytes")
            staged.unlink(missing_ok=True)
        else:
            staged.replace(final)
    with Session(engine) as db, db.begin():
        frame = db.get(Frame, (session_id, frame_id))
        verify_media(directory, frame)
        frame.media_state = "ready"
        bump_revision(db, session_id, frame)
    publish_snapshot(engine, root, sampled.key)


def _begin_review(engine, key, metadata, attempt_no, prepared):
    with Session(engine) as db, db.begin():
        review = VLMReview(
            session_id=key.session_id,
            frame_id=key.frame_id,
            attempt_no=attempt_no,
            status="running",
            runtime_uncertain=True,
            model_tag=metadata.get("model", "unknown"),
            model_digest=metadata.get("model_digest"),
            runtime_version=metadata.get("runtime_version"),
            prompt_sha256=metadata.get("prompt_sha256", PROMPT_SHA256),
            prompt_version=metadata.get("prompt_version", "cup_review_v1"),
            request_sha256=prepared.request_sha256,
            input_transform=prepared.transform,
        )
        db.add(review)
        db.flush()
        review_id = review.review_id
        bump_revision(db, key.session_id, db.get(Frame, (key.session_id, key.frame_id)))
    return review_id


def _save_review(engine, root, key, detections, result, review_id):
    if result.status == "validated":
        result.response = validate_stored_review(
            result.raw_response, key, detections, result.transform
        )
    completed = utcnow()
    with Session(engine) as db, db.begin():
        review = db.get(VLMReview, review_id)
        if (
            review is None
            or review.status != "running"
            or review.request_sha256 != result.request_sha256
            or review.input_transform != result.transform
            or (review.prompt_version == PROTOCOL) != (result.transform.get("protocol") == PROTOCOL)
        ):
            raise AuditorError("Model request identity differs from its durable dispatch record")
        review.status = result.status
        review.runtime_metrics = result.metrics
        review.completed_at = completed
        review.wall_ms = result.wall_ms
        review.raw_response = result.raw_response
        review.raw_response_truncated = result.truncated
        review.error_code = result.error_code
        review.runtime_uncertain = result.uncertain or result.status == "timeout"
        if result.response is not None:
            review.outcome = result.response.outcome
            review.needs_review = result.response.needs_review
            review.estimated_count = result.response.estimated_count
        db.flush()
        if result.status == "validated":
            for ordinal, proposed in enumerate(result.response.findings):
                finding = Finding(
                    review_id=review.review_id,
                    session_id=key.session_id,
                    frame_id=key.frame_id,
                    ordinal=ordinal,
                    rule_id=proposed.rule_id,
                    evidence=proposed.evidence,
                    explanation=proposed.explanation,
                )
                db.add(finding)
                db.flush()
                for detection_id in proposed.detection_ids:
                    db.add(
                        FindingDetection(
                            finding_id=finding.finding_id,
                            detection_id=detection_id,
                            session_id=key.session_id,
                            frame_id=key.frame_id,
                        )
                    )
        frame = db.get(Frame, (key.session_id, key.frame_id))
        bump_revision(db, key.session_id, frame)
    publish_snapshot(engine, root, key)


def run_analysis(
    engine,
    settings: Settings,
    session_id: UUID,
    *,
    detector=None,
    reviewer=None,
    retry_after_restart: bool = False,
):
    root = settings.data_root
    root.mkdir(parents=True, exist_ok=True)
    global_lock = FileLock(root / ".analysis.lock", timeout=0)
    try:
        global_lock.acquire()
    except Timeout as exc:
        raise AuditorError(
            "Another analysis is running. Wait for it to finish or request cancellation."
        ) from exc
    if (root / ".dataset-qwen-pause.json").exists():
        global_lock.release()
        raise AuditorError(
            "A dataset Qwen request may still be running. Restart Qwen and clear the pause in the dataset panel first."
        )
    started = time.monotonic()
    try:
        with session_lock(root, session_id):
            with Session(engine) as db:
                session = db.get(AuditSession, session_id)
                if session is None:
                    raise AuditorError("Choose an uploaded session first")
                if session.state == "review_ready":
                    yield "review_ready: Analysis is already complete. Existing evidence was preserved."
                    return
                video = db.scalar(select(Video).where(Video.session_id == session_id))
                probe = VideoProbe.model_validate(video.probe_data)
                original_path = video.original_path
                video_id = video.video_id
                frozen = dict(session.config)
                prior_frames = db.scalar(
                    select(Frame.frame_id).where(Frame.session_id == session_id).limit(1)
                )
            reconcile_session(engine, root, session_id)
            with Session(engine) as db, db.begin():
                session = db.get(AuditSession, session_id)
                session.state, session.cancel_requested, session.last_error = (
                    "processing",
                    False,
                    None,
                )
                session.processing_started_at = utcnow()
                bump_revision(db, session_id)
            yield "Loading detector. The original video is saved and protected."
            detector = detector or YoloDetector(settings.weights_path)
            if reviewer is None:
                if frozen.get("frame_stride") is not None:
                    reviewer = RegionalReviewer(
                        settings.ollama_base_url,
                        frozen["qwen"]["model"],
                        frozen["qwen"]["timeout_seconds"],
                        cancel_check=lambda: _cancelled(engine, session_id),
                    )
                else:
                    reviewer = OllamaReviewer(
                        settings.ollama_base_url,
                        frozen["qwen"]["model"],
                        frozen["qwen"]["timeout_seconds"],
                    )
            reviewer_metadata = reviewer.identity()
            if prior_frames:
                if frozen.get("pipeline_code_sha256") != _code_digest():
                    raise AuditorError(
                        "Analysis code changed since this session began. Upload a new session to rerun models."
                    )
                if (
                    any(
                        frozen.get("yolo", {}).get(name) != value
                        for name, value in detector.metadata.items()
                    )
                    or frozen.get("opencv_version") != cv2.__version__
                ):
                    raise AuditorError(
                        "YOLO checkpoint or detector/decoder runtime changed. Upload a new session. Existing predictions were preserved."
                    )
                old_prompt = frozen.get("qwen", {})
                if any(
                    old_prompt.get(name) != reviewer_metadata.get(name)
                    for name in ("prompt_sha256", "prompt_version")
                ):
                    raise AuditorError(
                        "Prompt identity changed. Use a new session for a different prompt."
                    )
                if any(
                    frozen.get("qwen", {}).get(name) != reviewer_metadata.get(name)
                    for name in ("model", "model_digest", "runtime_version")
                ):
                    raise AuditorError(
                        "Qwen model identity changed or is unavailable. Upload a new session for different models or runtimes."
                    )
            else:
                frozen["yolo"] = frozen["yolo"] | detector.metadata
                frozen["qwen"] = frozen["qwen"] | reviewer_metadata
                frozen["pipeline_code_sha256"] = _code_digest()
                frozen["opencv_version"] = cv2.__version__
                with Session(engine) as db, db.begin():
                    session = db.get(AuditSession, session_id)
                    session.config = frozen
                    session.config_sha256 = hashlib.sha256(canonical_json(frozen)).hexdigest()
                    bump_revision(db, session_id)
            directory = session_dir(root, session_id)
            count = 0
            for sampled in iter_sampled_frames(
                safe_path(directory, original_path),
                session_id,
                probe,
                **(
                    {"frame_stride": frozen["frame_stride"]}
                    if frozen.get("frame_stride") is not None
                    else {"sample_hz": frozen["sample_hz"]}
                ),
            ):
                if _cancelled(engine, session_id):
                    _terminal(
                        engine,
                        session_id,
                        "cancelled",
                        "Stopped between frames; completed evidence was retained.",
                    )
                    yield "cancelled: Completed evidence was preserved. You can resume later."
                    return
                with Session(engine) as db:
                    existing = db.get(Frame, (session_id, sampled.key.frame_id))
                    if existing:
                        if (
                            existing.timestamp_ms != sampled.key.timestamp_ms
                            or existing.source_pts != sampled.source_pts
                            or existing.raw_sha256 != hashlib.sha256(sampled.raw_jpeg).hexdigest()
                        ):
                            raise AuditorError(
                                "Decoded evidence differs from the frozen session. Original files were preserved."
                            )
                        count += 1
                        continue
                before = time.monotonic()
                detections = detector.detect(sampled.raw_jpeg)
                if len(detections) > 100:
                    raise AuditorError("Detector exceeded the per-frame detection limit")
                _commit_frame(
                    engine,
                    root,
                    session_id,
                    video_id,
                    sampled,
                    detections,
                    round((time.monotonic() - before) * 1000),
                    select_for_review=frozen.get("frame_stride") is not None,
                )
                count += 1
                yield f"YOLO: {count} sampled frames saved with synchronized evidence."
            detector = None
            with Session(engine) as db, db.begin():
                frames = db.scalars(
                    select(Frame)
                    .where(Frame.session_id == session_id)
                    .order_by(Frame.source_frame_index)
                ).all()
                keys = [
                    FrameKey(
                        session_id=session_id, frame_id=f.frame_id, timestamp_ms=f.timestamp_ms
                    )
                    for f in frames
                ]
                selected = {
                    k.frame_id
                    for k in select_review_frames(
                        keys,
                        None if frozen.get("frame_stride") is not None else frozen["vlm_budget"],
                    )
                }
                for frame in frames:
                    reason = (
                        (
                            "same source frames as YOLO"
                            if frozen.get("frame_stride") is not None
                            else "uniform bounded selection"
                        )
                        if frame.frame_id in selected
                        else "outside Qwen review budget"
                    )
                    if (
                        frame.vlm_selected != (frame.frame_id in selected)
                        or frame.vlm_skip_reason != reason
                    ):
                        frame.vlm_selected = frame.frame_id in selected
                        frame.vlm_skip_reason = reason
                        bump_revision(db, session_id, frame)
            for key in keys:
                publish_snapshot(engine, root, key)
            blocked = None
            if retry_after_restart:
                _confirm_runtime_restart(engine, root)
            with Session(engine) as db:
                uncertain = db.scalar(
                    select(VLMReview.review_id)
                    .where(
                        VLMReview.runtime_uncertain.is_(True),
                        VLMReview.runtime_restart_confirmed_at.is_(None),
                    )
                    .limit(1)
                )
            if uncertain:
                blocked = "Qwen may still be running a previous request. Restart Qwen and explicitly enable recovery retry."
            if blocked:
                _terminal(engine, session_id, "partial", blocked, time.monotonic() - started)
                yield f"partial: {blocked} Manual evidence remains available."
                return
            for ordinal, key in enumerate([k for k in keys if k.frame_id in selected], 1):
                if _cancelled(engine, session_id):
                    _terminal(engine, session_id, "cancelled", "Stopped before the next Qwen call.")
                    yield "cancelled: Completed evidence and model responses were retained."
                    return
                with Session(engine) as db:
                    attempts = db.scalars(
                        select(VLMReview)
                        .where(
                            VLMReview.session_id == session_id, VLMReview.frame_id == key.frame_id
                        )
                        .order_by(VLMReview.attempt_no)
                    ).all()
                    if any(a.status == "validated" for a in attempts) or len(attempts) >= 2:
                        continue
                    frame = db.get(Frame, (session_id, key.frame_id))
                    raw = safe_path(directory, frame.raw_path).read_bytes()
                    annotated = safe_path(directory, frame.annotated_path).read_bytes()
                    stored = db.scalars(
                        select(Detection)
                        .where(
                            Detection.session_id == session_id, Detection.frame_id == key.frame_id
                        )
                        .order_by(Detection.ordinal)
                    ).all()
                    detections = [
                        CupDetection(
                            detection_id=d.detection_id,
                            ordinal=d.ordinal,
                            score=d.score,
                            bbox_xyxy=(d.x1, d.y1, d.x2, d.y2),
                        )
                        for d in stored
                    ]
                    attempt_no = len(attempts) + 1
                mode = (
                    "frame (one to three regional checks)"
                    if frozen.get("frame_stride") is not None
                    else "pair"
                )
                yield f"Qwen: Reviewing {mode} {ordinal} of {len(selected)}. Each model call may take up to {frozen['qwen']['timeout_seconds']:g} seconds."
                while attempt_no <= 2:
                    payload, prepared = reviewer.prepare(key, raw, annotated, detections)
                    del payload
                    review_id = _begin_review(engine, key, reviewer_metadata, attempt_no, prepared)
                    result = reviewer.review(key, raw, annotated, detections)
                    _save_review(engine, root, key, detections, result, review_id)
                    if result.status in ("timeout", "oom", "transport_error") or result.uncertain:
                        blocked = result.error_code or result.status
                        break
                    if (
                        result.status == "validated"
                        or not result.retryable
                        or _cancelled(engine, session_id)
                    ):
                        break
                    attempt_no += 1
                if blocked:
                    break
            with Session(engine) as db:
                valid = db.scalar(
                    select(func.count())
                    .select_from(VLMReview)
                    .where(VLMReview.session_id == session_id, VLMReview.status == "validated")
                )
            state = "review_ready" if valid == len(selected) else "partial"
            _terminal(engine, session_id, state, blocked, time.monotonic() - started)
            yield f"{state}: {count} frames saved, {valid}/{len(selected)} Qwen reviews validated. Human decisions do not alter predictions."
    except GeneratorExit:
        raise
    except Exception as exc:
        message = (
            str(exc)
            if isinstance(exc, AuditorError)
            else f"Analysis stopped ({type(exc).__name__}). Completed evidence was preserved."
        )
        _terminal(engine, session_id, "failed", message, time.monotonic() - started)
        raise AuditorError(message) from exc
    finally:
        with Session(engine) as db, db.begin():
            session = db.get(AuditSession, session_id)
            if session and session.state == "processing":
                session.state = "cancelled" if session.cancel_requested else "interrupted"
                session.last_error = "Analysis was interrupted. Completed evidence was retained."
                bump_revision(db, session_id)
        global_lock.release()
