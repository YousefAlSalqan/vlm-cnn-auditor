"""Read-only Qwen experiments. This module never imports the application or database."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Literal

import cv2
from filelock import FileLock
from pydantic import Field, ValidationError

from auditor.config import Settings
from auditor.contracts import CupDetection, FrameKey, StrictRecord, validate_review
from auditor.models import OllamaReviewer, decode_image
from auditor.storage import atomic_write, canonical_json, safe_path

WORKSPACE = Path(__file__).resolve().parents[1]


class CupPresence(StrictRecord):
    cup_present: Literal["yes", "no", "unclear"]
    evidence: str = Field(min_length=1, max_length=400)


class _ExperimentReviewer(OllamaReviewer):
    def __init__(self, settings, prompt: str, *, probe: bool = False):
        super().__init__(settings.ollama_base_url, settings.ollama_model, settings.request_timeout)
        self.prompt = prompt
        self.probe = probe
        self.metadata |= {
            "prompt_version": "cup_presence_probe" if probe else "experimental_audit",
            "prompt_sha256": hashlib.sha256(prompt.encode()).hexdigest(),
        }

    def prepare(self, key, raw, annotated, detections):
        payload, attempt = super().prepare(key, raw, annotated, detections)
        prompt = self.prompt
        if self.probe:
            payload["format"] = CupPresence.model_json_schema()
            payload["messages"][1] = {
                "role": "user",
                "content": "Inspect only this image.",
                "images": payload["messages"][1]["images"][:1],
            }
            attempt.transform["image_sha256"] = attempt.transform["image_sha256"][:1]
        elif "{{frame_examples}}" in prompt:
            header = key.model_dump(mode="json")
            examples = [
                header
                | {
                    "outcome": "findings",
                    "estimated_count": None,
                    "needs_review": True,
                    "findings": [
                        {
                            "rule_id": "R1",
                            "detection_ids": [],
                            "evidence": "<actual location and appearance of an unboxed cup>",
                            "explanation": "<why this visible cup has no corresponding box>",
                        }
                    ],
                },
                header
                | {
                    "outcome": "no_issue",
                    "estimated_count": None,
                    "needs_review": False,
                    "findings": [],
                },
                header
                | {
                    "outcome": "insufficient_evidence",
                    "estimated_count": None,
                    "needs_review": True,
                    "findings": [],
                },
            ]
            prompt = prompt.replace(
                "{{frame_examples}}", "\n".join(json.dumps(e) for e in examples)
            )
        if "{{" in prompt:
            raise ValueError("Unrendered prompt placeholder")
        payload["messages"][0]["content"] = prompt
        attempt.request_sha256 = hashlib.sha256(canonical_json(payload)).hexdigest()
        return payload, attempt


def _case(session_dir: Path, frame_id: str):
    source = safe_path(session_dir, f"json/{frame_id}.json")
    snapshot_bytes = source.read_bytes()
    snapshot = json.loads(snapshot_bytes)
    key = FrameKey.model_validate(
        {name: snapshot[name] for name in ("session_id", "frame_id", "timestamp_ms")}
    )
    if key.frame_id != frame_id or session_dir.name != f"session_{key.session_id}":
        raise ValueError("Snapshot does not belong to the requested frame/session")
    detections = [CupDetection.model_validate(item) for item in snapshot["yolo"]["detections"]]
    raw = safe_path(session_dir, snapshot["artifacts"]["raw"]["path"]).read_bytes()
    annotated = safe_path(session_dir, snapshot["artifacts"]["annotated"]["path"]).read_bytes()
    return (
        key,
        detections,
        raw,
        annotated,
        {
            "snapshot_sha256": hashlib.sha256(snapshot_bytes).hexdigest(),
            "raw_sha256": hashlib.sha256(raw).hexdigest(),
            "annotated_sha256": hashlib.sha256(annotated).hexdigest(),
        },
    )


def _output(output_dir: Path, session_dir: Path):
    output_dir = output_dir.resolve()
    if not output_dir.is_relative_to(WORKSPACE.resolve()) or output_dir.is_relative_to(
        session_dir.resolve()
    ):
        raise ValueError("Output must stay in the repository and outside the source session")
    output_dir.mkdir(parents=True, exist_ok=False)
    return output_dir


def _reviewer(prompt: str, *, probe=False):
    reviewer = _ExperimentReviewer(Settings(), prompt, probe=probe)
    identity = reviewer.identity()
    if not identity.get("model_digest"):
        raise RuntimeError("Existing local Qwen model is unavailable")
    return reviewer, reviewer.metadata | identity


def _record(reviewer, identity, inputs, hashes):
    payload, prepared = reviewer.prepare(*inputs)
    return {
        "frame": inputs[0].model_dump(mode="json"),
        "model": identity,
        "generation": payload["options"],
        "rendered_prompt": payload["messages"][0]["content"],
        "request_sha256": prepared.request_sha256,
        "image_transform": prepared.transform,
        "source_hashes": hashes,
        "detector_count": len(inputs[3]),
        "status": "running",
    }


def _finish(record, result):
    record.update(
        status=result.status,
        error_code=result.error_code,
        wall_ms=result.wall_ms,
        raw_response=result.raw_response,
        uncertain=result.uncertain,
        metrics=result.metrics,
        response=result.response.model_dump(mode="json") if result.response else None,
    )


def run_batch(
    session_dir: Path,
    prompt_path: Path,
    frame_ids: list[str],
    output_dir: Path,
    max_calls: int = 20,
) -> Path:
    """Audit saved frame pairs with a new prompt, preserving strict V1 validation."""
    if not 1 <= max_calls <= 20:
        raise ValueError("A batch permits 1 to 20 calls")
    if len(frame_ids) != len(set(frame_ids)):
        raise ValueError("Do not repeat a deterministic frame request inside one batch")
    output_dir = _output(output_dir, session_dir)
    lock = WORKSPACE / ".local-runtime/qwen-yolo-loop/inference.lock"
    lock.parent.mkdir(parents=True, exist_ok=True)
    records = []
    report = output_dir / "report.json"
    with FileLock(lock, timeout=0):
        reviewer, identity = _reviewer(prompt_path.read_text(encoding="utf-8"))
        for frame_id in frame_ids[:max_calls]:
            key, detections, raw, annotated, hashes = _case(session_dir, frame_id)
            inputs = (key, raw, annotated, detections)
            record = _record(reviewer, identity, inputs, hashes)
            path = output_dir / f"{frame_id}.json"
            atomic_write(path, canonical_json(record))
            result = reviewer.review(*inputs)
            _finish(record, result)
            atomic_write(path, canonical_json(record))
            records.append(record)
            atomic_write(report, json.dumps(records, indent=2).encode())
            if result.uncertain or result.status in ("timeout", "oom", "transport_error"):
                break
    if not report.exists():
        atomic_write(report, b"[]")
    return report


def run_probe(
    session_dir: Path,
    frame_id: str,
    prompt: str,
    output_dir: Path,
    *,
    crop_xyxy: tuple[int, int, int, int] | None = None,
) -> Path:
    """Inspect one raw view using a diagnostic schema. Never emit a trusted V1 review."""
    key, detections, raw, annotated, hashes = _case(session_dir, frame_id)
    image = decode_image(raw)
    height, width = image.shape[:2]
    crop = crop_xyxy or (0, 0, width, height)
    x1, y1, x2, y2 = crop
    if not (
        all(type(value) is int for value in crop)
        and 0 <= x1 < x2 <= width
        and 0 <= y1 < y2 <= height
    ):
        raise ValueError("Invalid native-coordinate crop")
    encoded_ok, encoded = cv2.imencode(".jpg", image[y1:y2, x1:x2], [cv2.IMWRITE_JPEG_QUALITY, 95])
    if not encoded_ok:
        raise ValueError("Cannot encode crop")
    view = encoded.tobytes()
    output_dir = _output(output_dir, session_dir)
    lock = WORKSPACE / ".local-runtime/qwen-yolo-loop/inference.lock"
    lock.parent.mkdir(parents=True, exist_ok=True)
    path = output_dir / f"{frame_id}.json"
    with FileLock(lock, timeout=0):
        reviewer, identity = _reviewer(prompt, probe=True)
        inputs = (key, view, view, [])
        record = _record(reviewer, identity, inputs, hashes)
        record.update(
            scope="region" if crop_xyxy else "full_raw_diagnostic",
            crop_native_xyxy=list(crop),
            source_width=width,
            source_height=height,
            detector_count=len(detections),
            overlapping_detection_ids=[
                str(d.detection_id)
                for d in detections
                if max(x1, d.bbox_xyxy[0]) < min(x2, d.bbox_xyxy[2])
                and max(y1, d.bbox_xyxy[1]) < min(y2, d.bbox_xyxy[3])
            ],
        )
        atomic_write(output_dir / "view.jpg", view)
        atomic_write(path, canonical_json(record))
        result = reviewer.review(*inputs)
        _finish(record, result)
        record["diagnostic_status"] = "unavailable"
        record["diagnostic"] = None
        # The adapter still applies V1 validation. Probe success is a separate, regional result.
        if not result.uncertain and result.error_code == "invalid_response_schema":
            try:
                diagnostic = CupPresence.model_validate_json(result.raw_response)
                record["diagnostic_status"] = "validated"
                record["diagnostic"] = diagnostic.model_dump(mode="json")
            except ValidationError:
                record["diagnostic_status"] = "invalid_json"
        atomic_write(path, canonical_json(record))
    return path


def make_review_draft(session_dir: Path, probe_path: Path) -> dict | None:
    """Build a host-scoped R1 draft. This does not turn the raw probe into a V1 model review."""
    record = json.loads(probe_path.read_text(encoding="utf-8"))
    if (
        record.get("uncertain")
        or record.get("diagnostic_status") != "validated"
        or record.get("scope") != "region"
    ):
        return None
    diagnostic = CupPresence.model_validate_json(record["raw_response"])
    if diagnostic.model_dump(mode="json") != record["diagnostic"]:
        raise ValueError("Parsed diagnostic differs from the raw model response")
    if diagnostic.cup_present != "yes":
        return None
    key, detections, raw, _, hashes = _case(session_dir, record["frame"]["frame_id"])
    if hashes != record["source_hashes"] or key.model_dump(mode="json") != record["frame"]:
        raise ValueError("Probe source identity or evidence changed")
    image = decode_image(raw)
    height, width = image.shape[:2]
    x1, y1, x2, y2 = record["crop_native_xyxy"]
    if not (
        all(type(v) is int for v in (x1, y1, x2, y2))
        and 0 <= x1 < x2 <= width
        and 0 <= y1 < y2 <= height
    ):
        raise ValueError("Invalid source crop")
    # shortcut: zero-overlap regions propose review only; verify object matching before accepting labels.
    if any(
        max(x1, d.bbox_xyxy[0]) < min(x2, d.bbox_xyxy[2])
        and max(y1, d.bbox_xyxy[1]) < min(y2, d.bbox_xyxy[3])
        for d in detections
    ):
        return None
    view = cv2.imencode(".jpg", image[y1:y2, x1:x2], [cv2.IMWRITE_JPEG_QUALITY, 95])[1].tobytes()
    # prepare() performs local image processing only, including the adapter's resize rules.
    verifier = OllamaReviewer(base_url="", model=record["model"]["model"])
    _, prepared = verifier.prepare(key, view, view, [])
    prepared.transform["image_sha256"] = prepared.transform["image_sha256"][:1]
    if record["image_transform"] != prepared.transform:
        raise ValueError("Model input image does not match the source crop")
    draft = key.model_dump(mode="json") | {
        "outcome": "findings",
        "estimated_count": None,
        "needs_review": True,
        "findings": [
            {
                "rule_id": "R1",
                "detection_ids": [],
                "evidence": diagnostic.evidence,
                "explanation": f"Qwen reports cups in region [{x1}, {y1}, {x2}, {y2}], which intersects none of YOLO's boxes.",
            }
        ],
    }
    return validate_review(
        json.dumps(draft), key, {d.detection_id for d in detections}, len(detections)
    ).model_dump(mode="json")
