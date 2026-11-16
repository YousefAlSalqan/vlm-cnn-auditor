from __future__ import annotations

import base64
import hashlib
import importlib.metadata
import json
import os
import queue
import threading
import time
from dataclasses import dataclass, field
from pathlib import Path

import cv2
import httpx
import numpy as np
from pydantic import ValidationError

from auditor.contracts import AuditorError, CupDetection, FrameKey, ReviewResponse, validate_review
from auditor.storage import canonical_json, sha256_file

PROMPT = (Path(__file__).parent / "prompts" / "cup_review_v1.txt").read_text(encoding="utf-8")
PROMPT_SHA256 = hashlib.sha256(PROMPT.encode()).hexdigest()


def decode_image(raw: bytes):
    image = cv2.imdecode(np.frombuffer(raw, np.uint8), cv2.IMREAD_COLOR)
    if image is None:
        raise AuditorError("Saved evidence is not a decodable JPEG")
    return image


def render_overlay(raw_jpeg: bytes, detections: list[CupDetection]) -> bytes:
    image = decode_image(raw_jpeg)
    height, width = image.shape[:2]
    for detection in detections:
        x1, y1, x2, y2 = detection.bbox_xyxy
        if x2 > width or y2 > height:
            raise AuditorError("Detection box exceeds the canonical image dimensions")
        start = (round(x1), round(y1))
        end = (min(width - 1, round(x2)), min(height - 1, round(y2)))
        cv2.rectangle(image, start, end, (40, 180, 90), 2)
        label = f"D{detection.ordinal + 1} cup {detection.score:.2f}"
        cv2.putText(
            image,
            label,
            (max(0, start[0]), max(14, start[1] - 5)),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.45,
            (40, 180, 90),
            1,
            cv2.LINE_AA,
        )
    ok, encoded = cv2.imencode(".jpg", image, [cv2.IMWRITE_JPEG_QUALITY, 95])
    if not ok:
        raise AuditorError("Could not encode an annotated frame")
    return encoded.tobytes()


class YoloDetector:
    def __init__(self, weights: Path):
        if not weights.is_file():
            raise AuditorError(
                "YOLO weights are missing. Run python -m auditor.models --download-yolo."
            )
        os.environ.setdefault("YOLO_CONFIG_DIR", str(weights.parent / "ultralytics-config"))
        os.environ.setdefault("YOLO_AUTOINSTALL", "false")
        from ultralytics import YOLO, settings

        settings.update({"sync": False})
        self.metadata = {
            "model": weights.name,
            "weights_sha256": sha256_file(weights),
            "ultralytics": importlib.metadata.version("ultralytics"),
            "torch": importlib.metadata.version("torch"),
            "device": "cpu",
        }
        self.model = YOLO(str(weights), task="detect")
        if self.model.names.get(41) != "cup":
            raise AuditorError("This checkpoint does not map COCO class 41 to cup")

    def detect(self, raw_jpeg: bytes) -> list[CupDetection]:
        image = decode_image(raw_jpeg)
        result = self.model.predict(
            image,
            classes=[41],
            conf=0.25,
            iou=0.7,
            imgsz=640,
            max_det=100,
            device="cpu",
            verbose=False,
        )[0]
        detections = []
        for ordinal, (box, score, class_id) in enumerate(
            zip(
                result.boxes.xyxy.cpu().tolist(),
                result.boxes.conf.cpu().tolist(),
                result.boxes.cls.cpu().tolist(),
                strict=True,
            )
        ):
            if class_id != 41:
                raise AuditorError("Detector returned an unexpected object class")
            detection = CupDetection(ordinal=ordinal, score=float(score), bbox_xyxy=tuple(box))
            if box[2] > image.shape[1] or box[3] > image.shape[0]:
                raise AuditorError("Detector returned an out-of-frame box")
            detections.append(detection)
        return detections


@dataclass
class ReviewAttempt:
    status: str
    raw_response: str = ""
    response: ReviewResponse | None = None
    error_code: str | None = None
    wall_ms: int = 0
    request_sha256: str = ""
    transform: dict = field(default_factory=dict)
    metrics: dict = field(default_factory=dict)
    truncated: bool = False
    retryable: bool = False
    uncertain: bool = False


class OllamaReviewer:
    def __init__(
        self,
        base_url: str,
        model: str = "qwen3-vl:2b-instruct",
        timeout: float = 120,
        *,
        transport=None,
    ):
        self.base_url = base_url
        self.model = model
        self.timeout = timeout
        self.transport = transport
        self.metadata = {
            "model": model,
            "model_digest": None,
            "runtime_version": None,
            "prompt_sha256": PROMPT_SHA256,
            "prompt_version": "cup_review_v1",
        }

    def identity(self) -> dict:
        try:
            with httpx.Client(
                base_url=self.base_url, timeout=5, transport=self.transport, trust_env=False
            ) as client:
                tags = client.get("/api/tags")
                tags.raise_for_status()
                envelope = tags.json()
                if not isinstance(envelope, dict) or not isinstance(envelope.get("models"), list):
                    raise ValueError("Invalid model inventory")
                match = next(
                    (
                        m
                        for m in envelope["models"]
                        if isinstance(m, dict) and m.get("name") == self.model
                    ),
                    None,
                )
                if match is None:
                    return self.metadata | {
                        "availability_error": "Requested Qwen model is not installed"
                    }
                version = client.get("/api/version")
                version.raise_for_status()
                version_data = version.json()
                if not isinstance(version_data, dict):
                    raise ValueError("Invalid runtime identity")
                self.metadata |= {
                    "model_digest": match["digest"],
                    "runtime_version": version_data.get("version"),
                }
        except (httpx.HTTPError, ValueError, KeyError):
            return self.metadata | {"availability_error": "Local Ollama is unavailable"}
        return dict(self.metadata)

    def prepare(
        self, key: FrameKey, raw_jpeg: bytes, annotated_jpeg: bytes, detections: list[CupDetection]
    ) -> tuple[dict, ReviewAttempt]:
        raw = decode_image(raw_jpeg)
        annotated = decode_image(annotated_jpeg)
        if raw.shape != annotated.shape:
            raise AuditorError("Review images do not share canonical dimensions")
        height, width = raw.shape[:2]
        scale = min(1.0, 1024 / max(width, height))
        view_width, view_height = max(1, round(width * scale)), max(1, round(height * scale))
        images = []
        hashes = []
        for image in (raw, annotated):
            if scale < 1:
                image = cv2.resize(image, (view_width, view_height), interpolation=cv2.INTER_AREA)
            encoded = cv2.imencode(".jpg", image, [cv2.IMWRITE_JPEG_QUALITY, 95])[1].tobytes()
            images.append(base64.b64encode(encoded).decode("ascii"))
            hashes.append(hashlib.sha256(encoded).hexdigest())
        transform = {
            "native_width": width,
            "native_height": height,
            "view_width": view_width,
            "view_height": view_height,
            "scale_x": view_width / width,
            "scale_y": view_height / height,
            "image_sha256": hashes,
        }
        metadata = key.model_dump(mode="json") | {
            "image_width": view_width,
            "image_height": view_height,
            "detector_count": len(detections),
            "detections": [
                d.model_dump(mode="json")
                | {
                    "display_id": f"D{d.ordinal + 1}",
                    "bbox_xyxy": [
                        d.bbox_xyxy[0] * transform["scale_x"],
                        d.bbox_xyxy[1] * transform["scale_y"],
                        d.bbox_xyxy[2] * transform["scale_x"],
                        d.bbox_xyxy[3] * transform["scale_y"],
                    ],
                }
                for d in detections
            ],
        }
        payload = {
            "model": self.model,
            "stream": False,
            "format": ReviewResponse.model_json_schema(),
            "options": {"temperature": 0, "num_ctx": 8192, "num_predict": 1024},
            "messages": [
                {"role": "system", "content": PROMPT},
                {"role": "user", "content": json.dumps(metadata), "images": images},
            ],
        }
        request_hash = hashlib.sha256(canonical_json(payload)).hexdigest()
        return payload, ReviewAttempt(
            status="running", request_sha256=request_hash, transform=transform
        )

    @staticmethod
    def _request(client, payload, deadline):
        with client.stream("POST", "/api/chat", json=payload) as response:
            body = bytearray()
            for chunk in response.iter_bytes():
                if time.monotonic() >= deadline:
                    raise httpx.ReadTimeout("Absolute request deadline exceeded")
                room = 65536 - len(body)
                body.extend(chunk[:room])
                if len(chunk) > room:
                    return response.status_code, bytes(body), True
            return response.status_code, bytes(body), False

    def review(
        self, key: FrameKey, raw_jpeg: bytes, annotated_jpeg: bytes, detections: list[CupDetection]
    ) -> ReviewAttempt:
        started = time.monotonic()
        deadline = started + self.timeout
        payload, result = self.prepare(key, raw_jpeg, annotated_jpeg, detections)
        result.status = "transport_error"
        body = b""
        client = httpx.Client(
            base_url=self.base_url,
            transport=self.transport,
            trust_env=False,
            timeout=httpx.Timeout(
                self.timeout,
                connect=min(5, self.timeout),
                write=min(15, self.timeout),
                pool=min(5, self.timeout),
            ),
        )
        outcome = queue.Queue(maxsize=1)

        def dispatch():
            try:
                outcome.put(self._request(client, payload, deadline))
            except Exception as exc:
                outcome.put(exc)
            finally:
                client.close()

        try:
            remaining = deadline - time.monotonic()
            if remaining <= 0:
                raise httpx.ReadTimeout("Request preparation exceeded the deadline")
            threading.Thread(target=dispatch, daemon=True).start()
            try:
                reply = outcome.get(timeout=remaining)
            except queue.Empty as exc:
                # Closing HTTP does not prove the model stopped. The pipeline gates future calls.
                client.close()
                raise httpx.ReadTimeout("Absolute request deadline exceeded") from exc
            if isinstance(reply, Exception):
                raise reply
            status_code, body, truncated = reply
            if truncated:
                result.status, result.error_code = "invalid_json", "response_exceeds_64_kib"
                result.raw_response = body.decode("utf-8", errors="replace")
                result.truncated, result.uncertain = True, True
                return result
            if status_code >= 300:
                result.raw_response = body.decode("utf-8", errors="replace")
                result.status = (
                    "oom" if "out of memory" in result.raw_response.lower() else "transport_error"
                )
                result.error_code = f"ollama_http_{status_code}"
                return result
            data = json.loads(body)
            if (
                not isinstance(data, dict)
                or not isinstance(data.get("message"), dict)
                or not isinstance(data["message"].get("content"), str)
            ):
                raise TypeError("Invalid API envelope")
            if data.get("done") is not True:
                result.status = "invalid_json"
                result.raw_response = bytes(body).decode("utf-8", errors="replace")
                result.error_code = "generation_incomplete"
                result.uncertain = True
                return result
            result.raw_response = data["message"]["content"]
            result.metrics = {
                name: data[name]
                for name in (
                    "total_duration",
                    "load_duration",
                    "prompt_eval_count",
                    "eval_count",
                    "prompt_eval_duration",
                    "eval_duration",
                )
                if isinstance(data.get(name), int)
                and not isinstance(data[name], bool)
                and data[name] >= 0
            }
            try:
                result.response = validate_review(
                    result.raw_response, key, {d.detection_id for d in detections}, len(detections)
                )
                result.status = "validated"
            except ValidationError:
                result.status, result.error_code = "invalid_json", "invalid_response_schema"
                result.retryable = True
            except ValueError:
                result.status, result.error_code = (
                    "invalid_reference",
                    "invalid_frame_or_detection_reference",
                )
                result.retryable = True
        except httpx.TimeoutException:
            result.status, result.error_code = "timeout", "request_timed_out_no_automatic_retry"
            result.uncertain = True
        except httpx.HTTPError as exc:
            result.error_code = "local_ollama_unavailable"
            result.uncertain = not isinstance(exc, httpx.ConnectError)
        except (json.JSONDecodeError, KeyError, TypeError):
            result.status, result.error_code = "invalid_json", "invalid_api_envelope"
            result.raw_response = bytes(body).decode("utf-8", errors="replace")
        finally:
            result.wall_ms = round((time.monotonic() - started) * 1000)
        return result


def main():
    import argparse
    import urllib.request

    from auditor.config import Settings

    parser = argparse.ArgumentParser(description="Explicitly acquire the official YOLO checkpoint")
    parser.add_argument("--download-yolo", action="store_true", required=True)
    parser.parse_args()
    settings = Settings()
    path = settings.weights_path
    if path.exists():
        print("YOLO checkpoint already exists. It was not replaced.")
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(".download")
    try:
        urllib.request.urlretrieve(
            "https://github.com/ultralytics/assets/releases/download/v8.3.0/yolo11n.pt", temporary
        )
        if not 1_000_000 <= temporary.stat().st_size <= 20_000_000:
            raise AuditorError("Unexpected checkpoint size")
        os.replace(temporary, path)
        print(
            "Official YOLO11n checkpoint saved. Review Ultralytics licensing before redistribution."
        )
    finally:
        temporary.unlink(missing_ok=True)


if __name__ == "__main__":
    main()
