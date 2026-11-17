"""Partial-export regression checks without opening any database."""

import hashlib
from io import BytesIO
from uuid import uuid4
from zipfile import ZipFile

import pytest

from auditor.export import verify_export
from auditor.storage import canonical_json
from tests.test_models import jpeg


def partial_archive(*, complete=False, selected=False):
    session_id, video_id = str(uuid4()), str(uuid4())
    key = {"session_id": session_id, "frame_id": "frame_000000000", "timestamp_ms": 0}
    presentation = {
        "first_pts": 0,
        "time_base_num": 1,
        "time_base_den": 30,
        "timestamp_source": "synthetic",
        "rotation_degrees": 0,
    }
    configuration = {"frame_stride": 5, "yolo": {}, "qwen": {}, "rule_version": "cup_rules_v1"}
    raw = jpeg((10, 20, 30))
    digest = hashlib.sha256(raw).hexdigest()
    paths = {"raw": "frames/frame_000000000.jpg", "annotated": "annotated/frame_000000000.jpg"}
    payload = key | {
        "schema_version": 1,
        "video_id": video_id,
        "source_frame_index": 0,
        "snapshot_revision": 1,
        "image": {"width": 160, "height": 120},
        "presentation": presentation | {"source_pts": 0},
        "yolo": {"settings": {}, "detections": []},
        "vlm_selection": {"selected": selected},
        "vlm_reviews": [],
        "review_decisions": [],
        "artifacts": {kind: {"path": path, "sha256": digest} for kind, path in paths.items()},
    }
    files = {
        "original.mp4": b"synthetic structural-export fixture",
        **{p: raw for p in paths.values()},
        "json/frame_000000000.json": canonical_json(payload),
    }
    inventory = []
    for path, data in files.items():
        kind = (
            "original"
            if path == "original.mp4"
            else (
                "raw"
                if path.startswith("frames/")
                else "annotated"
                if path.startswith("annotated/")
                else "frame_json"
            )
        )
        inventory.append(
            {
                "path": path,
                "kind": kind,
                "byte_count": len(data),
                "sha256": hashlib.sha256(data).hexdigest(),
                **({} if kind == "original" else key),
            }
        )
    original = inventory[0]
    entry = key | {
        "source_frame_index": 0,
        "snapshot_revision": 1,
        "raw_path": paths["raw"],
        "annotated_path": paths["annotated"],
        "json_path": "json/frame_000000000.json",
        "vlm_selected": selected,
        "review_status": "not_reviewed" if selected else "not_selected",
    }
    manifest = {
        "schema_version": 1,
        "session_id": session_id,
        "session_revision": 2,
        "complete": complete,
        "session_state": "review_ready" if complete else "cancelled",
        "configuration": configuration,
        "configuration_sha256": hashlib.sha256(canonical_json(configuration)).hexdigest(),
        "video": presentation
        | {
            "session_id": session_id,
            "video_id": video_id,
            "original_path": "original.mp4",
            "original_sha256": original["sha256"],
            "byte_count": original["byte_count"],
            "probe_data": {"timestamps": [0] if complete else list(range(6))},
        },
        "frames": [entry],
        "files": inventory,
    }
    buffer = BytesIO()
    with ZipFile(buffer, "w") as archive:
        for path, data in files.items():
            archive.writestr(path, data)
        archive.writestr("manifest.json", canonical_json(manifest))
    buffer.seek(0)
    return buffer


def test_cancelled_yolo_phase_exports_unreviewed_stride_frames():
    assert verify_export(partial_archive())


def test_complete_export_still_rejects_unselected_stride_frames():
    with pytest.raises(ValueError, match="selection|unreviewed"):
        verify_export(partial_archive(complete=True))
