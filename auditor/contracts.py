from __future__ import annotations

from typing import Literal
from uuid import UUID, uuid4

from pydantic import BaseModel, ConfigDict, Field, StrictBool, StrictInt, model_validator


class AuditorError(ValueError):
    """An actionable input, integrity, or workflow error safe to show locally."""


class ConflictError(AuditorError):
    """The evidence changed since the reviewer loaded it."""


class StrictRecord(BaseModel):
    model_config = ConfigDict(extra="forbid", allow_inf_nan=False)


class FrameKey(StrictRecord):
    session_id: UUID
    frame_id: str = Field(pattern=r"^frame_[0-9]{9}$")
    timestamp_ms: StrictInt = Field(ge=0)


class CupDetection(StrictRecord):
    detection_id: UUID = Field(default_factory=uuid4)
    ordinal: StrictInt = Field(ge=0)
    class_id: Literal[41] = 41
    label: Literal["cup"] = "cup"
    score: float = Field(ge=0, le=1, strict=True)
    bbox_xyxy: tuple[float, float, float, float]

    @model_validator(mode="after")
    def ordered_box(self):
        x1, y1, x2, y2 = self.bbox_xyxy
        if not (0 <= x1 < x2 and 0 <= y1 < y2):
            raise ValueError("Bounding box must have positive area in native xyxy pixels")
        return self


class ProposedFinding(StrictRecord):
    rule_id: Literal["R1", "R2", "R3", "R4"]
    detection_ids: list[UUID] = Field(max_length=100)
    evidence: str = Field(min_length=1, max_length=1000)
    explanation: str = Field(min_length=1, max_length=1000)

    @model_validator(mode="after")
    def reference_cardinality(self):
        count = len(self.detection_ids)
        if count != len(set(self.detection_ids)):
            raise ValueError("Detection references must be unique")
        if self.rule_id == "R1" and count != 0:
            raise ValueError("R1 must not reference an existing detection")
        if self.rule_id == "R2" and count != 1:
            raise ValueError("R2 requires exactly one detection")
        if self.rule_id == "R3" and count < 2:
            raise ValueError("R3 requires at least two detections")
        return self


class ReviewResponse(FrameKey):
    outcome: Literal["findings", "no_issue", "insufficient_evidence"]
    estimated_count: StrictInt | None = Field(ge=0, le=100)
    needs_review: StrictBool
    findings: list[ProposedFinding] = Field(max_length=8)

    @model_validator(mode="after")
    def consistent_outcome(self):
        if self.outcome == "findings" and (not self.findings or not self.needs_review):
            raise ValueError("Findings require evidence and human review")
        if self.outcome != "findings" and self.findings:
            raise ValueError("Only a findings outcome may contain findings")
        if self.outcome == "no_issue" and self.needs_review:
            raise ValueError("No-issue outcome cannot claim findings need review")
        if self.outcome == "insufficient_evidence" and (
            not self.needs_review or self.estimated_count is not None
        ):
            raise ValueError("Abstention requires review and an unknown count")
        seen = [(f.rule_id, tuple(sorted(str(d) for d in f.detection_ids))) for f in self.findings]
        if len(seen) != len(set(seen)):
            raise ValueError("Duplicate rule/reference proposals are not allowed")
        return self


def validate_review(
    raw: str, key: FrameKey, detection_ids: set[UUID], detector_count: int
) -> ReviewResponse:
    result = ReviewResponse.model_validate_json(raw)
    if (result.session_id, result.frame_id, result.timestamp_ms) != (
        key.session_id,
        key.frame_id,
        key.timestamp_ms,
    ):
        raise ValueError("Model response references a different frame")
    for finding in result.findings:
        if not set(finding.detection_ids).issubset(detection_ids):
            raise ValueError("Model referenced an unknown or cross-frame detection")
        if finding.rule_id == "R4" and (
            result.estimated_count is None or result.estimated_count == detector_count
        ):
            raise ValueError("R4 requires a count different from YOLO's count")
    if result.outcome == "no_issue" and result.estimated_count not in (None, detector_count):
        raise ValueError("No-issue count contradicts the detector count")
    return result


class VideoProbe(StrictRecord):
    container: Literal["mp4", "mov"]
    codec: str
    duration_us: int = Field(gt=0)
    width: int = Field(gt=0)
    height: int = Field(gt=0)
    stream_index: int = Field(ge=0)
    time_base_num: int = Field(gt=0)
    time_base_den: int = Field(gt=0)
    first_pts: int
    timestamps: list[int] = Field(min_length=1, max_length=3600)
    rotation_degrees: Literal[0, 90, 180, 270] = 0
    timestamp_source: str
    probe_version: str


class SampledFrame(StrictRecord):
    key: FrameKey
    source_frame_index: int
    source_pts: int
    width: int
    height: int
    raw_jpeg: bytes
