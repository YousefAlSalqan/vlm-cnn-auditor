import json
from types import SimpleNamespace
from uuid import uuid4

import cv2
import numpy as np
import pytest

from auditor.contracts import FrameKey
from auditor.models import OllamaReviewer, ReviewAttempt
from experiments.qwen_yolo_loop import make_review_draft, run_batch, run_probe


@pytest.fixture
def saved_frames(tmp_path, monkeypatch):
    monkeypatch.setattr("experiments.qwen_yolo_loop.WORKSPACE", tmp_path)
    session_id = uuid4()
    session = tmp_path / f"session_{session_id}"
    for folder in ("frames", "annotated", "json"):
        (session / folder).mkdir(parents=True)
    image = cv2.imencode(".jpg", np.zeros((32, 48, 3), dtype=np.uint8))[1].tobytes()
    keys = []
    for index in (0, 30):
        key = FrameKey(
            session_id=session_id, frame_id=f"frame_{index:09d}", timestamp_ms=index * 1000 // 30
        )
        keys.append(key)
        for folder in ("frames", "annotated"):
            (session / folder / f"{key.frame_id}.jpg").write_bytes(image)
        snapshot = key.model_dump(mode="json") | {
            "yolo": {"detections": []},
            "artifacts": {
                "raw": {"path": f"frames/{key.frame_id}.jpg"},
                "annotated": {"path": f"annotated/{key.frame_id}.jpg"},
            },
        }
        (session / "json" / f"{key.frame_id}.json").write_text(json.dumps(snapshot))
    monkeypatch.setattr(
        "experiments.qwen_yolo_loop.Settings",
        lambda: SimpleNamespace(
            ollama_base_url="http://127.0.0.1:11435",
            ollama_model="qwen3-vl:2b-instruct",
            request_timeout=1,
        ),
    )
    monkeypatch.setattr(
        OllamaReviewer,
        "identity",
        lambda self: {
            "model": self.model,
            "model_digest": "unchanged-test-digest",
            "runtime_version": "test",
        },
    )
    return session, keys, tmp_path


def envelope(content):
    return (
        200,
        json.dumps({"done": True, "message": {"content": json.dumps(content)}}).encode(),
        False,
    )


def test_existing_run_is_rejected_before_model_dispatch(saved_frames, monkeypatch):
    session, keys, folder = saved_frames
    output = folder / "existing"
    output.mkdir()
    sentinel = output / "keep.json"
    sentinel.write_text("keep")
    monkeypatch.setattr(OllamaReviewer, "identity", lambda self: pytest.fail("Model contacted"))
    prompt = folder / "prompt.txt"
    prompt.write_text("Audit the image.")
    with pytest.raises(FileExistsError):
        run_batch(session, prompt, [keys[0].frame_id], output)
    assert sentinel.read_text() == "keep"


def test_uncertain_request_stops_batch_without_retry(saved_frames, monkeypatch):
    session, keys, folder = saved_frames
    calls = []

    def uncertain(self, *args):
        calls.append(args[0])
        return ReviewAttempt(status="timeout", uncertain=True, raw_response="partial")

    monkeypatch.setattr(OllamaReviewer, "review", uncertain)
    prompt = folder / "prompt.txt"
    prompt.write_text("Audit the image.")
    report = run_batch(session, prompt, [key.frame_id for key in keys], folder / "uncertain")
    records = json.loads(report.read_text())
    assert len(calls) == len(records) == 1
    assert records[0]["status"] == "timeout" and records[0]["uncertain"]
    assert records[0]["raw_response"] == "partial"
    assert records[0]["request_sha256"]


def test_unknown_detection_reference_stays_rejected(saved_frames, monkeypatch):
    session, keys, folder = saved_frames
    content = keys[0].model_dump(mode="json") | {
        "outcome": "findings",
        "estimated_count": None,
        "needs_review": True,
        "findings": [
            {
                "rule_id": "R2",
                "detection_ids": [str(uuid4())],
                "evidence": "A box covers a jar",
                "explanation": "Not a cup",
            }
        ],
    }
    monkeypatch.setattr(OllamaReviewer, "_request", staticmethod(lambda *args: envelope(content)))
    prompt = folder / "prompt.txt"
    prompt.write_text("Audit the image.")
    records = json.loads(
        run_batch(session, prompt, [keys[0].frame_id], folder / "references").read_text()
    )
    assert records[0]["status"] == "invalid_reference"
    assert records[0]["response"] is None
    assert json.loads(records[0]["raw_response"]) == content


def test_crop_diagnostic_cannot_become_whole_frame_no_issue(saved_frames, monkeypatch):
    session, keys, folder = saved_frames
    payloads = []

    def reply(client, payload, deadline):
        payloads.append(payload)
        return envelope({"cup_present": "no", "evidence": "The region is empty."})

    monkeypatch.setattr(OllamaReviewer, "_request", staticmethod(reply))
    record_path = run_probe(
        session, keys[0].frame_id, "Inspect this region.", folder / "crop", crop_xyxy=(8, 6, 40, 28)
    )
    record = json.loads(record_path.read_text())
    assert record["diagnostic_status"] == "validated"
    assert record["diagnostic"]["cup_present"] == "no"
    assert record["scope"] == "region"
    assert record["response"] is None
    assert record["crop_native_xyxy"] == [8, 6, 40, 28]
    assert len(payloads[0]["messages"][1]["images"]) == 1
    assert str(keys[0].session_id) not in payloads[0]["messages"][1]["content"]
    assert payloads[0]["options"] == {"temperature": 0, "num_ctx": 8192, "num_predict": 1024}


def test_invalid_crop_is_rejected_before_request(saved_frames, monkeypatch):
    session, keys, folder = saved_frames
    monkeypatch.setattr(OllamaReviewer, "review", lambda *args: pytest.fail("Model dispatched"))
    with pytest.raises(ValueError, match="crop"):
        run_probe(
            session, keys[0].frame_id, "Inspect.", folder / "bad-crop", crop_xyxy=(0, 0, 100, 40)
        )


def test_draft_uses_source_identity_and_preserves_raw_model_response(saved_frames, monkeypatch):
    session, keys, folder = saved_frames
    content = {"cup_present": "yes", "evidence": "A white mug with a handle is visible."}
    monkeypatch.setattr(OllamaReviewer, "_request", staticmethod(lambda *args: envelope(content)))
    probe_path = run_probe(
        session, keys[0].frame_id, "Inspect.", folder / "positive", crop_xyxy=(0, 0, 48, 32)
    )
    original = probe_path.read_bytes()
    draft = make_review_draft(session, probe_path)
    assert {name: draft[name] for name in ("session_id", "frame_id", "timestamp_ms")} == keys[
        0
    ].model_dump(mode="json")
    assert draft["outcome"] == "findings" and draft["needs_review"] is True
    assert draft["findings"][0]["rule_id"] == "R1"
    assert draft["findings"][0]["detection_ids"] == []
    assert draft["findings"][0]["evidence"] == content["evidence"]
    assert probe_path.read_bytes() == original
    source = session / "json" / f"{keys[0].frame_id}.json"
    source.write_text(source.read_text() + "\n")
    with pytest.raises(ValueError, match="source"):
        make_review_draft(session, probe_path)


def test_draft_rechecks_geometry_instead_of_trusting_overlap_list(saved_frames, monkeypatch):
    session, keys, folder = saved_frames
    source = session / "json" / f"{keys[0].frame_id}.json"
    snapshot = json.loads(source.read_text())
    snapshot["yolo"]["detections"] = [
        {
            "detection_id": str(uuid4()),
            "ordinal": 0,
            "score": 0.9,
            "bbox_xyxy": [1.0, 1.0, 40.0, 28.0],
        }
    ]
    source.write_text(json.dumps(snapshot))
    monkeypatch.setattr(
        OllamaReviewer,
        "_request",
        staticmethod(
            lambda *args: envelope({"cup_present": "yes", "evidence": "A white mug is visible."})
        ),
    )
    probe_path = run_probe(
        session, keys[0].frame_id, "Inspect.", folder / "already-boxed", crop_xyxy=(0, 0, 48, 32)
    )
    record = json.loads(probe_path.read_text())
    record["overlapping_detection_ids"] = []
    probe_path.write_text(json.dumps(record))
    assert make_review_draft(session, probe_path) is None


def test_draft_replays_the_adapter_resize_for_large_views(saved_frames, monkeypatch):
    session, keys, folder = saved_frames
    large = cv2.imencode(".jpg", np.zeros((1080, 1920, 3), dtype=np.uint8))[1].tobytes()
    for kind in ("frames", "annotated"):
        (session / kind / f"{keys[0].frame_id}.jpg").write_bytes(large)
    calls = []

    def reply(client, payload, deadline):
        calls.append(payload)
        return envelope({"cup_present": "yes", "evidence": "A white mug is visible."})

    monkeypatch.setattr(OllamaReviewer, "_request", staticmethod(reply))
    probe_path = run_probe(
        session, keys[0].frame_id, "Inspect.", folder / "large-view", crop_xyxy=(0, 0, 1920, 1080)
    )
    record = json.loads(probe_path.read_text())
    assert record["image_transform"]["view_width"] == 1024
    assert make_review_draft(session, probe_path)["findings"][0]["rule_id"] == "R1"
    assert len(calls) == 1
