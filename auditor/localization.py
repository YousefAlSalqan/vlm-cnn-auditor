"""Optional dataset-only cup grounding, using the existing bounded Ollama transport."""

from __future__ import annotations

import hashlib
from pathlib import Path
from typing import Literal
from uuid import uuid4

import cv2
from filelock import FileLock
from pydantic import Field, StrictInt, model_validator

from auditor import datasets
from auditor.contracts import AuditorError, FrameKey, StrictRecord
from auditor.models import OllamaReviewer, decode_image
from auditor.storage import atomic_write, canonical_json, safe_path, sha256_file

PROMPT = (Path(__file__).parent / "prompts/cup_boxes_v1.txt").read_text(encoding="utf-8")
PROMPT_SHA256 = hashlib.sha256(PROMPT.encode()).hexdigest()


class CupBoxes(StrictRecord):
    units: Literal["normalized_1000"]
    boxes: list[tuple[StrictInt, StrictInt, StrictInt, StrictInt]] = Field(max_length=16)

    @model_validator(mode="after")
    def geometry(self):
        for x1, y1, x2, y2 in self.boxes:
            if not (0 <= x1 < x2 <= 1000 and 0 <= y1 < y2 <= 1000):
                raise AuditorError("Invalid normalized localization box")
        return self


class BoxClient(OllamaReviewer):
    def prepare(self, key, raw, annotated, detections):
        payload, attempt = super().prepare(key, raw, annotated, detections)
        payload["format"] = CupBoxes.model_json_schema()
        payload["messages"] = [
            {"role": "system", "content": PROMPT},
            {
                "role": "user",
                "content": "Locate cups in this image.",
                "images": payload["messages"][1]["images"][:1],
            },
        ]
        attempt.transform["image_sha256"] = attempt.transform["image_sha256"][:1]
        attempt.request_sha256 = hashlib.sha256(canonical_json(payload)).hexdigest()
        return payload, attempt


def propose_frame(settings, dataset_id, token, *, enabled=False, cancel_check=None, transport=None):
    if not enabled:
        return None
    cancelled = cancel_check or (lambda: False)
    root = settings.data_root
    # Share the process lock with analysis and training. Never dispatch simultaneous model jobs.
    with FileLock(root / ".analysis.lock", timeout=0), datasets._lock(root):
        draft = datasets.load_draft(root, dataset_id)
        frame = datasets._frame(draft, token)
        pause = root / ".dataset-qwen-pause.json"
        if (
            cancelled()
            or draft["localization_uncertain"]
            or pause.exists()
            or frame["status"] != "pending"
        ):
            return draft
        directory = datasets.dataset_dir(root, dataset_id)
        image_path = safe_path(directory, frame["image_path"])
        if sha256_file(image_path) != frame["raw_sha256"]:
            raise AuditorError("Dataset image checksum changed")
        image = decode_image(image_path.read_bytes())
        tried = {tuple(r["native_xyxy"]) for r in frame["localization_receipts"]}
        for region in frame["source_regions"]:
            if cancelled() or tuple(region) in tried:
                continue
            x1, y1, x2, y2 = region
            if not (0 <= x1 < x2 <= frame["width"] and 0 <= y1 < y2 <= frame["height"]):
                raise AuditorError("Localization crop exceeds the source image")
            ok, encoded = cv2.imencode(".jpg", image[y1:y2, x1:x2])
            if not ok:
                raise AuditorError("Cannot encode the localization image")
            receipt = dict(
                receipt_id=str(uuid4()),
                native_xyxy=region,
                status="running",
                raw_response="",
                model=settings.ollama_model,
                model_digest=None,
                prompt_sha256=PROMPT_SHA256,
            )
            frame["localization_receipts"].append(receipt)
            # A crash or uncertain HTTP exit must block further requests on restart.
            draft["localization_uncertain"] = True
            atomic_write(
                pause,
                canonical_json(
                    {"dataset_id": str(dataset_id), "receipt_id": receipt["receipt_id"]}
                ),
            )
            datasets._save_edit(directory, draft, dict(action="localization_started", token=token))
            client = BoxClient(
                settings.ollama_base_url,
                settings.ollama_model,
                settings.request_timeout,
                transport=transport,
            )
            view = encoded.tobytes()
            attempt = client.review(FrameKey.model_validate(frame["key"]), view, view, [])
            receipt.update(
                status=attempt.status,
                raw_response=attempt.raw_response,
                error_code=attempt.error_code,
                request_sha256=attempt.request_sha256,
                image_transform=attempt.transform,
                uncertain=attempt.uncertain,
                truncated=attempt.truncated,
                wall_ms=attempt.wall_ms,
            )
            if (
                not attempt.uncertain
                and not attempt.truncated
                and attempt.error_code == "invalid_response_schema"
            ):
                try:
                    parsed = CupBoxes.model_validate_json(attempt.raw_response)
                except ValueError:
                    pass
                else:
                    receipt.update(status="validated", error_code=None)
                    for box in parsed.boxes:
                        native = [
                            x1 + box[0] * (x2 - x1) / 1000,
                            y1 + box[1] * (y2 - y1) / 1000,
                            x1 + box[2] * (x2 - x1) / 1000,
                            y1 + box[3] * (y2 - y1) / 1000,
                        ]
                        original = next(
                            (b for b in frame["proposals"] if b["bbox_xyxy"] == native), None
                        )
                        if original:
                            original["source_ids"] = list(
                                dict.fromkeys(original["source_ids"] + [receipt["receipt_id"]])
                            )
                            continue
                        proposal = datasets.Box(
                            bbox_xyxy=native, origin="qwen", source_ids=[receipt["receipt_id"]]
                        ).model_dump(mode="json")
                        frame["proposals"].append(proposal)
                        frame["boxes"].append(dict(proposal))
            draft["localization_uncertain"] = attempt.uncertain
            datasets._save_edit(
                directory,
                draft,
                dict(action="localization_finished", token=token, receipt_id=receipt["receipt_id"]),
            )
            if not attempt.uncertain:
                pause.unlink(missing_ok=True)
            if attempt.uncertain or cancelled():
                break
        return draft


def clear_model_pause(settings, confirmed):
    if not confirmed:
        raise AuditorError("Confirm you stopped or restarted Qwen before clearing the pause")
    root = settings.data_root
    with FileLock(root / ".analysis.lock", timeout=0), datasets._lock(root):
        pause = root / ".dataset-qwen-pause.json"
        if not pause.exists():
            return
        dataset_id = datasets._read(pause)["dataset_id"]
        draft = datasets.load_draft(root, dataset_id)
        draft["localization_uncertain"] = False
        datasets._save_edit(
            datasets.dataset_dir(root, dataset_id), draft, dict(action="model_restart_acknowledged")
        )
        pause.unlink()
