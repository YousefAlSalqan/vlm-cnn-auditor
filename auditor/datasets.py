"""Local dataset drafts, reviewed annotations and immutable training releases."""

from __future__ import annotations

import hashlib
import json
from datetime import UTC, datetime
from pathlib import Path
from typing import Literal
from uuid import UUID, uuid4
from zipfile import ZipFile

from filelock import FileLock
from pydantic import Field, model_validator

from auditor.contracts import AuditorError, ConflictError, FrameKey, StrictRecord
from auditor.export import verify_export
from auditor.models import decode_image
from auditor.regional import CupPresence, RegionalEnvelope
from auditor.storage import atomic_write, canonical_json, safe_path, sha256_file

MAX_RECORD_BYTES = 32 * 1024**2
DEFAULT_TRAINING = dict(epochs=10, batch=2, imgsz=640, workers=0, device="cpu", seed=0)


def _now():
    return datetime.now(UTC).isoformat()


class Box(StrictRecord):
    box_id: UUID = Field(default_factory=uuid4)
    class_id: Literal[41] = 41
    bbox_xyxy: tuple[float, float, float, float]
    origin: Literal["yolo", "qwen", "human"] = "human"
    source_ids: list[str] = Field(default_factory=list, max_length=32)
    edited: bool = False

    @model_validator(mode="after")
    def geometry(self):
        x1, y1, x2, y2 = self.bbox_xyxy
        if not 0 <= x1 < x2 or not 0 <= y1 < y2:
            raise AuditorError("A box needs positive area and ordered coordinates")
        return self


class DatasetFrame(StrictRecord):
    token: str = Field(pattern=r"^[a-f0-9]{64}$")
    key: FrameKey
    source_frame_index: int = Field(ge=0)
    source_group: str = Field(pattern=r"^[a-f0-9]{64}$")
    raw_sha256: str = Field(pattern=r"^[a-f0-9]{64}$")
    image_path: str
    snapshot_path: str
    snapshot_sha256: str
    width: int = Field(ge=1, le=1920)
    height: int = Field(ge=1, le=1920)
    split: Literal["train", "val"]
    status: Literal["pending", "complete", "reviewed_negative", "exclude"] = "pending"
    proposals: list[Box] = Field(default_factory=list, max_length=256)
    boxes: list[Box] = Field(default_factory=list, max_length=256)
    hints: list[str] = Field(default_factory=list, max_length=64)
    source_regions: list[tuple[int, int, int, int]] = Field(default_factory=list, max_length=3)
    localization_receipts: list[dict] = Field(default_factory=list, max_length=64)

    @model_validator(mode="after")
    def bounds(self):
        for boxes in (self.proposals, self.boxes):
            if len({b.box_id for b in boxes}) != len(boxes):
                raise AuditorError("Box identifiers must be unique within an image")
            if any(b.bbox_xyxy[2] > self.width or b.bbox_xyxy[3] > self.height for b in boxes):
                raise AuditorError("Box exceeds the image bounds")
        if self.status == "complete" and not self.boxes:
            raise AuditorError("Use reviewed negative to explicitly approve an empty cup image")
        if self.status == "reviewed_negative" and self.boxes:
            raise AuditorError("A reviewed negative cannot contain cup boxes")
        return self


class DraftDataset(StrictRecord):
    schema_version: Literal[1] = 1
    dataset_id: UUID
    revision: int = Field(ge=1)
    created_at: str
    updated_at: str
    sources: list[dict]
    frames: list[DatasetFrame] = Field(min_length=1)
    group_splits: dict[str, Literal["train", "val"]]
    frozen_splits: dict[str, Literal["train", "val"]] = Field(default_factory=dict)
    history: list[dict] = Field(default_factory=list)
    approval: dict | None = None
    localization_uncertain: bool = False

    @model_validator(mode="after")
    def identities(self):
        if len({f.token for f in self.frames}) != len(self.frames):
            raise AuditorError("Duplicate frame identity in dataset")
        if any(self.group_splits.get(f.source_group) != f.split for f in self.frames):
            raise AuditorError("A source video cannot cross dataset splits")
        if any(
            self.group_splits.get(group) != split for group, split in self.frozen_splits.items()
        ):
            raise AuditorError("A previously approved source video cannot change splits")
        return self


def dataset_dir(root: Path, dataset_id) -> Path:
    return safe_path(Path(root) / "datasets", str(UUID(str(dataset_id))))


def _read(path: Path):
    if path.stat().st_size > MAX_RECORD_BYTES:
        raise AuditorError("Dataset record exceeds the local record size limit")
    return json.loads(path.read_text(encoding="utf-8"))


def _write_draft(directory, draft):
    parsed = DraftDataset.model_validate(draft).model_dump(mode="json")
    data = canonical_json(parsed)
    if len(data) > MAX_RECORD_BYTES:
        raise AuditorError("Dataset is too large for this local review workspace")
    atomic_write(directory / "draft.json", data)
    return parsed


def load_draft(root: Path, dataset_id=None) -> dict | None:
    if dataset_id is None:
        pointer = Path(root) / "datasets/current.json"
        if not pointer.exists():
            return None
        dataset_id = _read(pointer)["dataset_id"]
    directory = dataset_dir(root, dataset_id)
    draft = DraftDataset.model_validate(_read(directory / "draft.json")).model_dump(mode="json")
    if draft["dataset_id"] != str(UUID(str(dataset_id))):
        raise AuditorError("Dataset identity differs from its folder")
    return draft


def _lock(root):
    path = Path(root) / "datasets"
    path.mkdir(parents=True, exist_ok=True)
    return FileLock(path / ".dataset.lock", timeout=10)


def _regions(snapshot, width, height):
    reviews = [r for r in snapshot["vlm_reviews"] if r["status"] == "validated"]
    if not reviews or not any(f["rule_id"] == "R1" for f in reviews[-1]["findings"]):
        return []
    try:
        envelope = RegionalEnvelope.model_validate_json(reviews[-1]["raw_response"])
        return [
            list(r.native_xyxy)
            for r in envelope.regions
            if r.index != 3 and CupPresence.model_validate_json(r.raw_response).cup_present == "yes"
        ]
    except (ValueError, KeyError):
        return [[0, 0, width, height]]


def build_draft(root: Path, source_zips: list[Path], *, enabled=False, dataset_id=None):
    if not enabled:
        return None
    if not source_zips:
        raise AuditorError("Choose at least one saved session export")
    with _lock(root):
        draft = load_draft(root, dataset_id)
        if draft is None:
            draft = dict(
                dataset_id=str(uuid4()),
                revision=1,
                created_at=_now(),
                updated_at=_now(),
                sources=[],
                frames=[],
                group_splits={},
                frozen_splits={},
                history=[],
                approval=None,
                localization_uncertain=False,
            )
        directory = dataset_dir(root, draft["dataset_id"])
        by_token = {f["token"]: f for f in draft["frames"]}
        changed = False
        for source in source_zips:
            digest = sha256_file(Path(source))
            verify_export(Path(source))
            with ZipFile(source) as archive:
                manifest = json.loads(archive.read("manifest.json"))
                group = manifest["video"]["original_sha256"]
                split = draft["group_splits"].get(group)
                if split is None:
                    split = (
                        "val"
                        if draft["group_splits"] and "val" not in draft["group_splits"].values()
                        else "train"
                    )
                imported = 0
                for entry in manifest["frames"]:
                    snapshot_bytes = archive.read(entry["json_path"])
                    snapshot = json.loads(snapshot_bytes)
                    key = FrameKey.model_validate(
                        {k: snapshot[k] for k in ("session_id", "frame_id", "timestamp_ms")}
                    )
                    token = hashlib.sha256(canonical_json(key.model_dump(mode="json"))).hexdigest()
                    raw = archive.read(entry["raw_path"])
                    raw_hash = hashlib.sha256(raw).hexdigest()
                    if token in by_token:
                        previous = by_token[token]
                        if (
                            previous["raw_sha256"] != raw_hash
                            or previous["source_group"] != group
                            or previous["source_frame_index"] != snapshot["source_frame_index"]
                        ):
                            raise AuditorError(
                                "Source evidence changed for an existing frame identity"
                            )
                        continue
                    image = decode_image(raw)
                    height, width = image.shape[:2]
                    if (width, height) != (snapshot["image"]["width"], snapshot["image"]["height"]):
                        raise AuditorError("Source image dimensions differ from saved evidence")
                    proposals = [
                        Box(
                            box_id=UUID(d["detection_id"]),
                            bbox_xyxy=d["bbox_xyxy"],
                            origin="yolo",
                            source_ids=[d["detection_id"]],
                        ).model_dump(mode="json")
                        for d in snapshot["yolo"]["detections"]
                    ]
                    frame = DatasetFrame(
                        token=token,
                        key=key,
                        source_frame_index=snapshot["source_frame_index"],
                        source_group=group,
                        raw_sha256=raw_hash,
                        image_path=f"images/{token[:24]}.jpg",
                        snapshot_path=f"sources/{token[:24]}.json",
                        snapshot_sha256=hashlib.sha256(snapshot_bytes).hexdigest(),
                        width=width,
                        height=height,
                        split=split,
                        proposals=proposals,
                        boxes=proposals,
                        hints=[
                            f["evidence"]
                            for review in snapshot["vlm_reviews"]
                            if review["status"] == "validated"
                            for f in review["findings"]
                        ],
                        source_regions=_regions(snapshot, width, height),
                    ).model_dump(mode="json")
                    atomic_write(safe_path(directory, frame["image_path"]), raw, immutable=True)
                    atomic_write(
                        safe_path(directory, frame["snapshot_path"]), snapshot_bytes, immutable=True
                    )
                    draft["frames"].append(frame)
                    by_token[token] = frame
                    imported += 1
                if sha256_file(Path(source)) != digest:
                    raise AuditorError("Session export changed while preparing the dataset")
                if imported:
                    draft["group_splits"][group] = split
                    draft["sources"].append(
                        dict(
                            archive_sha256=digest,
                            source_group=group,
                            session_id=manifest["session_id"],
                            complete=manifest["complete"],
                        )
                    )
                    changed = True
        if changed:
            if (directory / "draft.json").exists():
                draft["revision"] += 1
            draft.update(approval=None, updated_at=_now())
            draft = _write_draft(directory, draft)
            atomic_write(
                Path(root) / "datasets/current.json",
                canonical_json({"dataset_id": draft["dataset_id"]}),
            )
        if not draft["frames"]:
            raise AuditorError("These exports contain no saved frames for a dataset")
        return draft


def _frame(draft, token):
    try:
        return next(frame for frame in draft["frames"] if frame["token"] == token)
    except StopIteration as exc:
        raise AuditorError("Choose a frame from this dataset") from exc


def _revision(draft, expected):
    if draft["revision"] != expected:
        raise ConflictError("Dataset changed. Reload the current revision before saving")


def _save_edit(directory, draft, event):
    draft["revision"] += 1
    event.update(
        event_id=str(uuid4()), revision=draft["revision"], recorded_at=_now(), reviewer="local"
    )
    atomic_write(
        directory / "revisions" / f"{event['event_id']}.json", canonical_json(event), immutable=True
    )
    draft["history"].append(event)
    draft.update(approval=None, updated_at=_now())
    return _write_draft(directory, draft)


def save_frame_review(
    root, dataset_id, token, expected_revision, boxes, disposition, *, split=None
):
    with _lock(root):
        draft = load_draft(root, dataset_id)
        _revision(draft, expected_revision)
        frame = _frame(draft, token)
        if disposition not in ("pending", "complete", "reviewed_negative", "exclude"):
            raise AuditorError("Choose pending, complete, reviewed negative or exclude")
        if not isinstance(boxes, list) or len(boxes) > 256:
            raise AuditorError("This image has too many annotation boxes")
        originals = {b["box_id"]: b for b in frame["proposals"]}
        originals |= {b["box_id"]: b for b in frame["boxes"] if b["origin"] == "human"}
        parsed = []
        for box in boxes:
            previous = originals.get(str(box.get("box_id", "")))
            candidate = Box(
                box_id=UUID(previous["box_id"]) if previous else uuid4(),
                bbox_xyxy=box["bbox_xyxy"],
                origin=previous["origin"] if previous else "human",
                source_ids=previous["source_ids"] if previous else [],
                edited=bool(previous and list(box["bbox_xyxy"]) != previous["bbox_xyxy"]),
            ).model_dump(mode="json")
            parsed.append(candidate)
        if split is not None:
            if split not in ("train", "val"):
                raise AuditorError("Choose a training or validation split")
            group = frame["source_group"]
            if group in draft["frozen_splits"] and draft["frozen_splits"][group] != split:
                raise AuditorError("An approved source video cannot change dataset split")
            draft["group_splits"][group] = split
            for item in draft["frames"]:
                if item["source_group"] == group:
                    item["split"] = split
        frame.update(boxes=parsed, status=disposition)
        # Validate before writing either the revision receipt or the current draft.
        DraftDataset.model_validate(draft)
        return _save_edit(
            dataset_dir(root, dataset_id),
            draft,
            dict(
                frame_token=token,
                disposition=disposition,
                boxes=parsed,
                split=frame["split"],
            ),
        )


def yolo_line(box, width, height):
    parsed = Box(bbox_xyxy=box)
    x1, y1, x2, y2 = parsed.bbox_xyxy
    if width <= 0 or height <= 0 or x2 > width or y2 > height:
        raise AuditorError("Box exceeds the image bounds")
    values = [
        (x1 + x2) / (2 * width),
        (y1 + y2) / (2 * height),
        (x2 - x1) / width,
        (y2 - y1) / height,
    ]
    return "0 " + " ".join(f"{value:.8g}" for value in values)


def _signature(boxes):
    return sorted(tuple(b["bbox_xyxy"]) for b in boxes)


def box_iou(box, target):
    intersection = max(0, min(box[2], target[2]) - max(box[0], target[0])) * max(
        0, min(box[3], target[3]) - max(box[1], target[1])
    )
    union = (
        (box[2] - box[0]) * (box[3] - box[1])
        + (target[2] - target[0]) * (target[3] - target[1])
        - intersection
    )
    return intersection / union if union else 0.0


def readiness(draft):
    reasons = []
    if any(f["status"] == "pending" for f in draft["frames"]):
        reasons.append("Review or exclude every pending frame before approving the dataset")
    frames = [f for f in draft["frames"] if f["status"] != "exclude"]
    if {f["split"] for f in frames} != {"train", "val"}:
        reasons.append("Use separate source videos for training and validation")
    if any(
        not any(f["boxes"] for f in frames if f["split"] == split) for split in ("train", "val")
    ):
        reasons.append("Training and validation each need a reviewed cup example")
    seen = {}
    for frame in frames:
        previous = seen.setdefault(frame["raw_sha256"], frame)
        if previous["split"] != frame["split"]:
            reasons.append("Identical images cannot cross training and validation splits")
        if _signature(previous["boxes"]) != _signature(frame["boxes"]):
            reasons.append("Identical images have conflicting reviewed annotations")
    return list(dict.fromkeys(reasons))


def freeze_release(root, dataset_id, expected_revision, base_weights: Path, *, enabled=False):
    if not enabled:
        raise AuditorError("Enable the dataset workflow before approving training")
    base_weights = Path(base_weights).resolve()
    if not base_weights.is_file() or base_weights.suffix != ".pt":
        raise AuditorError("Choose an existing local YOLO checkpoint")
    training = DEFAULT_TRAINING | dict(
        base_weights=str(base_weights), base_weights_sha256=sha256_file(base_weights)
    )
    with _lock(root):
        draft = load_draft(root, dataset_id)
        _revision(draft, expected_revision)
        reasons = readiness(draft)
        if reasons:
            raise AuditorError(". ".join(reasons))
        frames = [f for f in draft["frames"] if f["status"] != "exclude"]
        directory = dataset_dir(root, dataset_id)
        if (
            draft["approval"]
            and draft["approval"]["revision"] == expected_revision
            and draft["approval"]["training"] == training
        ):
            release = safe_path(directory, draft["approval"]["release"])
            verify_release(release, expected_sha=draft["approval"]["manifest_sha256"])
            return release
        seen = {}
        for frame in frames:
            digest = frame["raw_sha256"]
            seen.setdefault(digest, frame)
        versions = [
            int(p.name[1:])
            for p in (directory / "releases").glob("v[0-9]*")
            if p.name[1:].isdigit()
        ]
        release = directory / "releases" / f"v{max(versions, default=0) + 1:06d}"
        release.mkdir(parents=True)
        files = {}

        def publish(path, data):
            files[path] = atomic_write(safe_path(release, path), data, immutable=True)

        exported = []
        for frame in seen.values():
            raw = safe_path(directory, frame["image_path"]).read_bytes()
            if (
                hashlib.sha256(raw).hexdigest() != frame["raw_sha256"]
                or sha256_file(safe_path(directory, frame["snapshot_path"]))
                != frame["snapshot_sha256"]
            ):
                raise AuditorError("Dataset source image or snapshot checksum changed")
            image_path = f"images/{frame['split']}/{frame['token'][:24]}.jpg"
            label_path = f"labels/{frame['split']}/{frame['token'][:24]}.txt"
            lines = [
                yolo_line(b["bbox_xyxy"], frame["width"], frame["height"]) for b in frame["boxes"]
            ]
            publish(image_path, raw)
            publish(label_path, ("\n".join(lines) + ("\n" if lines else "")).encode())
            exported.append(frame | dict(image_path=image_path, label_path=label_path))
        publish(
            "data.yaml",
            (
                f"path: {json.dumps(str(release.resolve()))}\ntrain: images/train\nval: images/val\nnames:\n  0: cup\n"
            ).encode(),
        )
        publish("history.json", canonical_json({"history": draft["history"]}))
        manifest = dict(
            schema_version=1,
            dataset_id=str(dataset_id),
            revision=expected_revision,
            approved=True,
            approved_at=_now(),
            class_names={"0": "cup"},
            canonical_class_id=41,
            frames=exported,
            history=draft["history"],
            training=training,
            files=files,
            sources=draft["sources"],
            group_splits=draft["group_splits"],
        )
        digest = atomic_write(release / "manifest.json", canonical_json(manifest), immutable=True)
        atomic_write(
            release / "approval.json",
            canonical_json({"manifest_sha256": digest, "approved": True}),
            immutable=True,
        )
        verify_release(release, expected_sha=digest)
        draft["approval"] = dict(
            release=release.relative_to(directory).as_posix(),
            manifest_sha256=digest,
            revision=expected_revision,
            training=training,
        )
        draft["frozen_splits"] = dict(draft["group_splits"])
        _write_draft(directory, draft)
        return release


def verify_release(release: Path, *, expected_sha=None):
    release = Path(release)
    approval = _read(release / "approval.json")
    digest = sha256_file(release / "manifest.json")
    if (
        approval.get("approved") is not True
        or digest != approval["manifest_sha256"]
        or (expected_sha and digest != expected_sha)
    ):
        raise AuditorError("Approved dataset manifest checksum changed")
    manifest = _read(release / "manifest.json")
    if manifest["approved"] is not True or manifest["class_names"] != {"0": "cup"}:
        raise AuditorError("Dataset is not an approved cup release")
    expected = set(manifest["files"]) | {"manifest.json", "approval.json"}
    actual = {p.relative_to(release).as_posix() for p in release.rglob("*") if p.is_file()}
    if actual != expected:
        raise AuditorError("Approved release inventory changed")
    for relative, expected_hash in manifest["files"].items():
        if sha256_file(safe_path(release, relative)) != expected_hash:
            raise AuditorError("Approved dataset file checksum changed")
    groups = {}
    for frame in manifest["frames"]:
        parsed = DatasetFrame.model_validate({k: v for k, v in frame.items() if k != "label_path"})
        if parsed.status not in ("complete", "reviewed_negative"):
            raise AuditorError("Unreviewed frame in approved dataset")
        if parsed.source_group in groups and groups[parsed.source_group] != parsed.split:
            raise AuditorError("Source video crosses dataset splits")
        groups[parsed.source_group] = parsed.split
        expected_labels = "\n".join(
            yolo_line(b.bbox_xyxy, parsed.width, parsed.height) for b in parsed.boxes
        )
        if safe_path(release, frame["label_path"]).read_text().strip() != expected_labels:
            raise AuditorError("Training labels differ from approved annotation boxes")
    return manifest
