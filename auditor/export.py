from __future__ import annotations

import hashlib
import json
import os
import shutil
import tempfile
from pathlib import Path
from uuid import UUID, uuid4
from zipfile import ZIP_DEFLATED, ZIP_STORED, BadZipFile, ZipFile

from sqlalchemy import select
from sqlalchemy.orm import Session

from auditor.contracts import AuditorError, CupDetection, FrameKey, validate_review
from auditor.db import AuditSession, Frame, Video, VLMReview, utcnow
from auditor.evidence import reconcile_session, row_dict
from auditor.storage import (
    atomic_write,
    canonical_json,
    safe_path,
    session_dir,
    session_lock,
    sha256_file,
)


def export_session(engine, root: Path, session_id: UUID) -> Path:
    with Session(engine) as db:
        state = db.get(AuditSession, session_id)
        if state is None:
            raise AuditorError("Choose a saved session first")
        if state.state == "processing":
            raise AuditorError("Wait for analysis to finish or stop before exporting")
    with session_lock(root, session_id):
        reconcile_session(engine, root, session_id)
        directory = session_dir(root, session_id)
        with Session(engine) as db:
            session = db.get(AuditSession, session_id)
            video = db.scalar(select(Video).where(Video.session_id == session_id))
            frames = db.scalars(
                select(Frame)
                .where(Frame.session_id == session_id)
                .order_by(Frame.source_frame_index)
            ).all()
            reviews = db.scalars(
                select(VLMReview)
                .where(VLMReview.session_id == session_id)
                .order_by(VLMReview.attempt_no)
            ).all()
            by_frame = {f.frame_id: [] for f in frames}
            for review in reviews:
                by_frame[review.frame_id].append(review)
            files = [
                {
                    "path": video.original_path,
                    "kind": "original",
                    "byte_count": video.byte_count,
                    "sha256": video.original_sha256,
                }
            ]
            entries = []
            for frame in frames:
                key = FrameKey(
                    session_id=session_id, frame_id=frame.frame_id, timestamp_ms=frame.timestamp_ms
                )
                for kind, relative, digest in [
                    ("raw", frame.raw_path, frame.raw_sha256),
                    ("annotated", frame.annotated_path, frame.annotated_sha256),
                    ("frame_json", frame.json_path, frame.snapshot_sha256),
                ]:
                    path = safe_path(directory, relative)
                    files.append(
                        key.model_dump(mode="json")
                        | dict(
                            path=relative, kind=kind, byte_count=path.stat().st_size, sha256=digest
                        )
                    )
                attempts = by_frame[frame.frame_id]
                latest = attempts[-1] if attempts else None
                entries.append(
                    key.model_dump(mode="json")
                    | dict(
                        source_frame_index=frame.source_frame_index,
                        snapshot_revision=frame.snapshot_revision,
                        raw_path=frame.raw_path,
                        annotated_path=frame.annotated_path,
                        json_path=frame.json_path,
                        vlm_selected=frame.vlm_selected,
                        review_status=latest.status
                        if latest
                        else ("not_reviewed" if frame.vlm_selected else "not_selected"),
                    )
                )
            complete = bool(frames) and session.state == "review_ready"
            manifest = {
                "schema_version": 1,
                "export_id": str(uuid4()),
                "session_id": str(session_id),
                "session_revision": session.revision,
                "exported_at": utcnow().isoformat(),
                "session_state": session.state,
                "complete": complete,
                "configuration": session.config,
                "configuration_sha256": session.config_sha256,
                "pipeline_version": session.pipeline_version,
                "video": row_dict(video),
                "frames": entries,
                "files": files,
                "diagnostics": {
                    "last_error": session.last_error,
                    "evaluation": "Independent annotations and business evaluation are deferred in V1.",
                    "human_decisions_are_ground_truth": False,
                },
            }
        size = sum(f["byte_count"] for f in files)
        if shutil.disk_usage(root).free < size + 64 * 1024**2:
            raise AuditorError("Not enough disk space to create a safe session export")
        output_dir = safe_path(root, f"exports/{session_id}")
        output_dir.mkdir(parents=True, exist_ok=True)
        target = output_dir / f"{manifest['export_id']}.zip"
        with tempfile.NamedTemporaryFile(dir=output_dir, suffix=".tmp", delete=False) as temporary:
            temporary_path = Path(temporary.name)
        try:
            with ZipFile(temporary_path, "w", allowZip64=True) as archive:
                for item in files:
                    path = safe_path(directory, item["path"])
                    if sha256_file(path) != item["sha256"]:
                        raise AuditorError(
                            "Checksum changed during export. No archive was published."
                        )
                    compression = ZIP_DEFLATED if item["kind"] == "frame_json" else ZIP_STORED
                    archive.write(path, item["path"], compress_type=compression)
                archive.writestr(
                    "manifest.json", canonical_json(manifest), compress_type=ZIP_DEFLATED
                )
            verify_export(temporary_path)
            atomic_write(safe_path(directory, "manifest.json"), canonical_json(manifest))
            os.replace(temporary_path, target)
        finally:
            temporary_path.unlink(missing_ok=True)
        return target


def verify_export(path: Path) -> bool:
    try:
        return _verify_export(path)
    except AuditorError:
        raise
    except (BadZipFile, KeyError, IndexError, TypeError, ValueError) as exc:
        raise AuditorError("Export contains malformed or inconsistent evidence metadata") from exc


def _verify_review_evidence(payload, configuration, detection_ids):
    key = FrameKey.model_validate(
        {name: payload[name] for name in ("session_id", "frame_id", "timestamp_ms")}
    )
    reviews = payload["vlm_reviews"]
    if not isinstance(reviews, list) or len(reviews) > 2:
        raise AuditorError("Export exceeds the per-frame review attempt limit")
    finding_ids, review_ids = set(), set()
    validated_count = 0
    for attempt, review in enumerate(reviews, 1):
        review_id = str(UUID(review["review_id"]))
        if (
            review_id in review_ids
            or review["attempt_no"] != attempt
            or review["session_id"] != payload["session_id"]
            or review["frame_id"] != payload["frame_id"]
            or review["response_schema_version"] != 1
            or review["rule_version"] != configuration["rule_version"]
        ):
            raise AuditorError("Exported review has inconsistent scope or attempt identity")
        review_ids.add(review_id)
        if any(
            review[review_field] != configuration["qwen"].get(config_field)
            for review_field, config_field in (
                ("model_tag", "model"),
                ("model_digest", "model_digest"),
                ("runtime_version", "runtime_version"),
                ("prompt_version", "prompt_version"),
                ("prompt_sha256", "prompt_sha256"),
            )
        ):
            raise AuditorError("Review provenance differs from the frozen session configuration")
        if review["status"] != "validated":
            if review["findings"] or any(
                review[field] is not None
                for field in ("outcome", "needs_review", "estimated_count")
            ):
                raise AuditorError("An invalid review contains normalized suggestions")
            continue
        validated_count += 1
        response = validate_review(review["raw_response"], key, detection_ids, len(detection_ids))
        if (
            review["raw_response_truncated"]
            or any(
                review[field] != getattr(response, field)
                for field in ("outcome", "needs_review", "estimated_count")
            )
            or len(review["findings"]) != len(response.findings)
        ):
            raise AuditorError("Validated review differs from its original model response")
        for ordinal, (finding, proposed) in enumerate(
            zip(review["findings"], response.findings, strict=True)
        ):
            finding_id = str(UUID(finding["finding_id"]))
            if (
                finding_id in finding_ids
                or finding["review_id"] != review_id
                or finding["session_id"] != payload["session_id"]
                or finding["frame_id"] != payload["frame_id"]
                or finding["ordinal"] != ordinal
                or any(
                    finding[field] != getattr(proposed, field)
                    for field in ("rule_id", "evidence", "explanation")
                )
                or sorted(finding["detection_ids"])
                != sorted(str(value) for value in proposed.detection_ids)
            ):
                raise AuditorError(
                    "Finding scope or detection references differ from its validated response"
                )
            finding_ids.add(finding_id)
    if validated_count > 1:
        raise AuditorError("Export contains multiple validated reviews for one frame")
    previous = {}
    decision_ids = set()
    for decision in payload["review_decisions"]:
        finding_id = decision["finding_id"]
        number, previous_id = previous.get(finding_id, (0, None))
        decision_id = str(UUID(decision["decision_id"]))
        if (
            finding_id not in finding_ids
            or decision_id in decision_ids
            or decision["decision_number"] != number + 1
            or decision["supersedes_decision_id"] != previous_id
            or decision["action"] not in ("accept", "reject", "defer")
            or (decision["note"] is not None and len(decision["note"]) > 2000)
        ):
            raise AuditorError(
                "Decision history has an invalid finding reference or revision chain"
            )
        decision_ids.add(decision_id)
        previous[finding_id] = (number + 1, decision_id)


def _verify_export(path: Path) -> bool:
    with ZipFile(path) as archive:
        names = archive.namelist()
        if len(names) != len(set(names)) or "manifest.json" not in names:
            raise AuditorError("Export contains duplicate entries or has no manifest")
        info = archive.getinfo("manifest.json")
        if info.file_size > 2 * 1024**2:
            raise AuditorError("Export manifest exceeds the size limit")
        manifest = json.loads(archive.read("manifest.json"))
        if (
            not isinstance(manifest, dict)
            or type(manifest.get("schema_version")) is not int
            or manifest["schema_version"] != 1
        ):
            raise AuditorError("Unsupported export schema version")
        session_id = str(UUID(manifest["session_id"]))
        if session_id != manifest["session_id"]:
            raise AuditorError("Export session identity is not canonical")
        revision = manifest["session_revision"]
        if type(revision) is not int or revision < 1:
            raise AuditorError("Invalid exported session revision")
        if (
            hashlib.sha256(canonical_json(manifest["configuration"])).hexdigest()
            != manifest["configuration_sha256"]
        ):
            raise AuditorError("Export configuration checksum does not match")
        video = manifest["video"]
        if video["session_id"] != session_id or video["original_path"] not in (
            "original.mp4",
            "original.mov",
        ):
            raise AuditorError("Original video has an invalid session or path")
        entries = manifest["frames"]
        if not isinstance(entries, list) or len(entries) > 60:
            raise AuditorError("Invalid exported frame inventory")
        allowlist = {video["original_path"]: "original"}
        frame_ids = set()
        for entry in entries:
            key = FrameKey.model_validate(
                {name: entry[name] for name in ("session_id", "frame_id", "timestamp_ms")}
            )
            if str(key.session_id) != session_id or key.frame_id in frame_ids:
                raise AuditorError("Duplicate or cross-session frame identity")
            frame_ids.add(key.frame_id)
            canonical_paths = {
                "raw_path": f"frames/{key.frame_id}.jpg",
                "annotated_path": f"annotated/{key.frame_id}.jpg",
                "json_path": f"json/{key.frame_id}.json",
            }
            for field, kind in (
                ("raw_path", "raw"),
                ("annotated_path", "annotated"),
                ("json_path", "frame_json"),
            ):
                if entry[field] != canonical_paths[field]:
                    raise AuditorError("Frame paths are not scoped to their exact identity")
                allowlist[entry[field]] = kind
        files = manifest["files"]
        expected = {f["path"] for f in files}
        if len(expected) != len(files) or set(names) != expected | {"manifest.json"}:
            raise AuditorError("Export inventory contains unexpected or missing files")
        if expected != set(allowlist):
            raise AuditorError("Frame inventory does not account for every exported artifact")
        inventory = {item["path"]: item for item in files}
        original = inventory[video["original_path"]]
        if (
            original["sha256"] != video["original_sha256"]
            or original["byte_count"] != video["byte_count"]
        ):
            raise AuditorError("Original video metadata does not match the file inventory")
        for item in files:
            safe_path(Path.cwd(), item["path"])
            if item["kind"] != allowlist[item["path"]]:
                raise AuditorError("Export contains an unexpected artifact kind")
            if archive.getinfo(item["path"]).file_size != item["byte_count"]:
                raise AuditorError("Export file size does not match the manifest")
            digest = hashlib.sha256()
            with archive.open(item["path"]) as stream:
                for chunk in iter(lambda: stream.read(1024 * 1024), b""):
                    digest.update(chunk)
            if digest.hexdigest() != item["sha256"]:
                raise AuditorError("Export checksum verification failed")
        if type(manifest["complete"]) is not bool:
            raise AuditorError("Export completeness must be explicit")
        if manifest["complete"] and (not entries or manifest["session_state"] != "review_ready"):
            raise AuditorError("Export claims completeness without a completed analysis")
        for entry in entries:
            # shortcut: cap a loaded snapshot at16MiB; stream histories if review volumes reach this limit.
            if archive.getinfo(entry["json_path"]).file_size > 16 * 1024**2:
                raise AuditorError("Frame history exceeds the 16 MiB verification limit")
            payload = json.loads(archive.read(entry["json_path"]))
            if (
                any(
                    payload.get(key) != entry.get(key)
                    for key in (
                        "session_id",
                        "frame_id",
                        "timestamp_ms",
                        "source_frame_index",
                        "snapshot_revision",
                    )
                )
                or payload["session_id"] != manifest["session_id"]
            ):
                raise AuditorError("Export frame identity/revision does not match the manifest")
            frame_revision = payload["snapshot_revision"]
            if type(frame_revision) is not int or not 1 <= frame_revision <= revision:
                raise AuditorError("Frame snapshot revision exceeds the session revision")
            if payload["video_id"] != video["video_id"] or payload["schema_version"] != 1:
                raise AuditorError("Frame belongs to another video or schema")
            index = payload["source_frame_index"]
            if type(index) is not int or index < 0 or payload["frame_id"] != f"frame_{index:09d}":
                raise AuditorError("Frame source index does not match its identifier")
            presentation = payload["presentation"]
            if any(
                presentation[field] != video[field]
                for field in (
                    "first_pts",
                    "time_base_num",
                    "time_base_den",
                    "timestamp_source",
                    "rotation_degrees",
                )
            ):
                raise AuditorError("Frame presentation evidence differs from its source video")
            if presentation["source_pts"] != video["probe_data"]["timestamps"][index]:
                raise AuditorError("Frame presentation timestamp does not match its source index")
            numerator = (
                (presentation["source_pts"] - presentation["first_pts"])
                * presentation["time_base_num"]
                * 1000
            )
            denominator = presentation["time_base_den"]
            if (
                denominator <= 0
                or (2 * numerator + denominator) // (2 * denominator) != payload["timestamp_ms"]
            ):
                raise AuditorError("Frame timestamp does not match its original presentation time")
            if payload["yolo"]["settings"] != manifest["configuration"]["yolo"]:
                raise AuditorError(
                    "Frame model provenance differs from the frozen session configuration"
                )
            detection_ids = set()
            for ordinal, detection in enumerate(payload["yolo"]["detections"]):
                parsed = CupDetection.model_validate(detection)
                if (
                    parsed.detection_id in detection_ids
                    or parsed.ordinal != ordinal
                    or parsed.bbox_xyxy[2] > payload["image"]["width"]
                    or parsed.bbox_xyxy[3] > payload["image"]["height"]
                ):
                    raise AuditorError("Exported detection box exceeds native frame dimensions")
                detection_ids.add(parsed.detection_id)
            _verify_review_evidence(payload, manifest["configuration"], detection_ids)
            if payload["vlm_selection"]["selected"] != entry["vlm_selected"]:
                raise AuditorError("Frame review selection differs between snapshot and manifest")
            reviews = payload["vlm_reviews"]
            expected_status = (
                reviews[-1]["status"]
                if reviews
                else ("not_reviewed" if entry["vlm_selected"] else "not_selected")
            )
            if entry["review_status"] != expected_status or (reviews and not entry["vlm_selected"]):
                raise AuditorError("Manifest review status contradicts its frame evidence")
            if (
                manifest["complete"]
                and entry["vlm_selected"]
                and not any(review["status"] == "validated" for review in payload["vlm_reviews"])
            ):
                raise AuditorError("Complete export contains an unreviewed selected frame")
            for kind, name in (("raw", "raw_path"), ("annotated", "annotated_path")):
                artifact = payload["artifacts"][kind]
                if (
                    artifact["path"] != entry[name]
                    or artifact["path"] not in expected
                    or artifact["sha256"] != inventory[artifact["path"]]["sha256"]
                ):
                    raise AuditorError("Export frame pair paths do not match")
                if any(
                    inventory[artifact["path"]][field] != entry[field]
                    for field in ("session_id", "frame_id", "timestamp_ms")
                ):
                    raise AuditorError("Artifact inventory references a different frame")
        return True
