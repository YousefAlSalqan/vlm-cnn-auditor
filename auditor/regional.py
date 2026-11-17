"""Regional Qwen observations with host-owned identity and replayable derivation."""

from __future__ import annotations

import hashlib
import time
from pathlib import Path
from typing import Literal

import cv2
from pydantic import Field, StrictBool, StrictInt, ValidationError

from auditor.contracts import FrameKey, ReviewResponse, StrictRecord, validate_review
from auditor.models import OllamaReviewer, ReviewAttempt, decode_image
from auditor.storage import canonical_json

PROTOCOL = "cup_regions_v1"
PROMPT = (Path(__file__).parent / "prompts/cup_presence_v2.txt").read_text(encoding="utf-8")
PROMPT_SHA256 = hashlib.sha256(PROMPT.encode()).hexdigest()


class CupPresence(StrictRecord):
    cup_present: Literal["yes", "no", "unclear"]
    evidence: str = Field(min_length=1, max_length=400)


class RegionReceipt(StrictRecord):
    index: StrictInt
    native_xyxy: tuple[StrictInt, StrictInt, StrictInt, StrictInt]
    request_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    image_transform: dict
    status: str = Field(max_length=32)
    error_code: str | None
    truncated: StrictBool
    raw_response: str = Field(max_length=65536)


class RegionalEnvelope(FrameKey):
    protocol: Literal["cup_regions_v1"]
    width: StrictInt = Field(ge=1, le=1920)
    height: StrictInt = Field(ge=1, le=1920)
    regions: list[RegionReceipt] = Field(min_length=1, max_length=3)


def region_observations(raw):
    try:
        envelope = RegionalEnvelope.model_validate_json(raw or "")
    except ValueError:
        return ""
    labels = {0: "Top area", 1: "Middle area", 2: "Bottom area", 3: "Whole-frame context"}
    lines = []
    for region in envelope.regions:
        try:
            observation = CupPresence.model_validate_json(region.raw_response)
            lines.append(
                f"{labels.get(region.index, 'Area')}: {observation.cup_present}. {observation.evidence}"
            )
        except ValueError:
            lines.append(
                f"{labels.get(region.index, 'Area')}: observation unavailable ({region.status})."
            )
    return "\n\n".join(lines)


def region_plan(width, height, detections):
    if not 1 <= width <= 1920 or not 1 <= height <= 1920:
        raise ValueError("Invalid regional image dimensions")
    regions = []
    band_count = min(3, height)
    for index in range(band_count):
        box = [0, index * height // band_count, width, (index + 1) * height // band_count]
        if not any(
            max(box[0], d.bbox_xyxy[0]) < min(box[2], d.bbox_xyxy[2])
            and max(box[1], d.bbox_xyxy[1]) < min(box[3], d.bbox_xyxy[3])
            for d in detections
        ):
            regions.append({"index": index, "native_xyxy": box})
    # Every analyzed source frame reaches Qwen, even when every band touches a YOLO box.
    return regions or [{"index": 3, "native_xyxy": [0, 0, width, height]}]


class _RegionClient(OllamaReviewer):
    def prepare(self, key, raw, annotated, detections):
        payload, attempt = super().prepare(key, raw, annotated, detections)
        payload["format"] = CupPresence.model_json_schema()
        payload["messages"] = [
            {"role": "system", "content": PROMPT},
            {
                "role": "user",
                "content": "Inspect only this image.",
                "images": payload["messages"][1]["images"][:1],
            },
        ]
        attempt.transform["image_sha256"] = attempt.transform["image_sha256"][:1]
        attempt.request_sha256 = hashlib.sha256(canonical_json(payload)).hexdigest()
        return payload, attempt


def validate_stored_review(raw, key, detections, transform):
    if transform.get("protocol") != PROTOCOL:
        return validate_review(raw, key, {d.detection_id for d in detections}, len(detections))
    if len(raw.encode("utf-8")) > 2 * 1024**2:
        raise ValueError("Regional response exceeds its bounded envelope size")
    envelope = RegionalEnvelope.model_validate_json(raw)
    if (
        FrameKey.model_validate(
            envelope.model_dump(include={"session_id", "frame_id", "timestamp_ms"})
        )
        != key
    ):
        raise ValueError("Regional response references a different frame")
    if (envelope.width, envelope.height) != (transform["native_width"], transform["native_height"]):
        raise ValueError("Regional dimensions changed")
    plan = region_plan(envelope.width, envelope.height, detections)
    if len(plan) != len(envelope.regions) or len(plan) != len(transform["regions"]):
        raise ValueError("Regional review is incomplete")
    candidates = []
    for expected, receipt, prepared in zip(
        plan, envelope.regions, transform["regions"], strict=True
    ):
        if (
            receipt.index != expected["index"]
            or list(receipt.native_xyxy) != expected["native_xyxy"]
            or {
                "index": receipt.index,
                "native_xyxy": list(receipt.native_xyxy),
                "request_sha256": receipt.request_sha256,
                "image_transform": receipt.image_transform,
            }
            != prepared
            or receipt.status != "validated"
            or receipt.error_code is not None
            or receipt.truncated
        ):
            raise ValueError("Regional evidence differs from its prepared request")
        observation = CupPresence.model_validate_json(receipt.raw_response)
        if observation.cup_present == "yes" and receipt.index != 3:
            candidates.append((receipt, observation))
    findings = []
    if candidates:
        evidence = "\n".join(f"Region {r.index + 1}: {o.evidence}" for r, o in candidates)
        if len(evidence) > 1000:
            evidence = f"{candidates[0][1].evidence}\nAdditional regional observations are preserved in the raw review."
        locations = "; ".join(str(list(r.native_xyxy)) for r, _ in candidates)
        findings = [
            {
                "rule_id": "R1",
                "detection_ids": [],
                "evidence": evidence,
                "explanation": f"Qwen reports cups in native regions {locations}. No saved YOLO box intersects these regions. Verify each suggestion.",
            }
        ]
    response = key.model_dump(mode="json") | {
        "outcome": "findings" if findings else "insufficient_evidence",
        "estimated_count": None,
        "needs_review": True,
        "findings": findings,
    }
    return ReviewResponse.model_validate(response)


class RegionalReviewer(OllamaReviewer):
    def __init__(self, *args, cancel_check=None, **kwargs):
        super().__init__(*args, **kwargs)
        self.cancel_check = cancel_check or (lambda: False)
        self.metadata |= {"prompt_version": PROTOCOL, "prompt_sha256": PROMPT_SHA256}

    def _prepare(self, key, raw, annotated, detections):
        image = decode_image(raw)
        if image.shape != decode_image(annotated).shape:
            raise ValueError("Regional frame pair dimensions differ")
        height, width = image.shape[:2]
        transform = {
            "protocol": PROTOCOL,
            "native_width": width,
            "native_height": height,
            "regions": [],
        }
        calls = []
        for region in region_plan(width, height, detections):
            x1, y1, x2, y2 = region["native_xyxy"]
            ok, encoded = cv2.imencode(".jpg", image[y1:y2, x1:x2], [cv2.IMWRITE_JPEG_QUALITY, 95])
            if not ok:
                raise ValueError("Cannot encode regional view")
            view = encoded.tobytes()
            client = _RegionClient(
                self.base_url, self.model, self.timeout, transport=self.transport
            )
            _, prepared = client.prepare(key, view, view, [])
            item = region | {
                "request_sha256": prepared.request_sha256,
                "image_transform": prepared.transform,
            }
            transform["regions"].append(item)
            calls.append((client, view, item))
        bundle = key.model_dump(mode="json") | {
            "protocol": PROTOCOL,
            "regions": transform["regions"],
        }
        prepared = ReviewAttempt(
            status="running",
            request_sha256=hashlib.sha256(canonical_json(bundle)).hexdigest(),
            transform=transform,
        )
        return bundle, prepared, calls

    def prepare(self, key, raw, annotated, detections):
        bundle, prepared, _ = self._prepare(key, raw, annotated, detections)
        return bundle, prepared

    def review(self, key, raw, annotated, detections):
        started = time.monotonic()
        _, result, calls = self._prepare(key, raw, annotated, detections)
        envelope = key.model_dump(mode="json") | {
            "protocol": PROTOCOL,
            "width": result.transform["native_width"],
            "height": result.transform["native_height"],
            "regions": [],
        }
        result.metrics = {"protocol": PROTOCOL, "regions": []}
        for client, view, item in calls:
            if self.cancel_check():
                result.status, result.error_code = "cancelled", "stopped_between_regions"
                break
            attempt = client.review(key, view, view, [])
            if attempt.request_sha256 != item["request_sha256"]:
                raise ValueError("Regional dispatch identity changed")
            status, error = attempt.status, attempt.error_code
            if not attempt.uncertain and error == "invalid_response_schema":
                try:
                    CupPresence.model_validate_json(attempt.raw_response)
                    status, error = "validated", None
                except ValidationError:
                    pass
            envelope["regions"].append(
                item
                | {
                    "status": status,
                    "error_code": error,
                    "truncated": attempt.truncated,
                    "raw_response": attempt.raw_response,
                }
            )
            result.metrics["regions"].append(
                {"index": item["index"], "wall_ms": attempt.wall_ms, **attempt.metrics}
            )
            if status != "validated":
                result.status, result.error_code = status, error
                result.uncertain, result.truncated = attempt.uncertain, attempt.truncated
                result.retryable = attempt.retryable and not attempt.uncertain
                break
        else:
            result.status = "validated"
        result.raw_response = canonical_json(envelope).decode("utf-8")
        result.wall_ms = round((time.monotonic() - started) * 1000)
        if result.status == "validated":
            result.response = validate_stored_review(
                result.raw_response, key, detections, result.transform
            )
        return result
