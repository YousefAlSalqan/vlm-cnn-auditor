"""Dataset review checks with synthetic exports and no database."""

import hashlib
import json
from uuid import uuid4
from zipfile import ZipFile

import pytest

from auditor import datasets
from auditor.storage import canonical_json
from tests.test_export_partial import partial_archive


def source_zip(directory, group="one", *, box=True):
    directory.mkdir(parents=True, exist_ok=True)
    with ZipFile(partial_archive()) as archive:
        files = {name: archive.read(name) for name in archive.namelist()}
    manifest = json.loads(files.pop("manifest.json"))
    frame = manifest["frames"][0]
    snapshot = json.loads(files[frame["json_path"]])
    if box:
        snapshot["yolo"]["detections"] = [
            dict(
                detection_id=str(uuid4()),
                ordinal=0,
                class_id=41,
                label="cup",
                score=0.8,
                bbox_xyxy=[10, 20, 30, 60],
            )
        ]
    # Distinct synthetic footage and pixels represent independent recordings.
    files["original.mp4"] = f"synthetic source {group}".encode()
    from tests.test_models import jpeg

    color = hashlib.sha256(group.encode()).digest()[:3]
    raw = jpeg(tuple(color))
    for kind in ("raw", "annotated"):
        files[frame[f"{kind}_path"]] = raw
        snapshot["artifacts"][kind]["sha256"] = hashlib.sha256(raw).hexdigest()
    files[frame["json_path"]] = canonical_json(snapshot)
    for item in manifest["files"]:
        data = files[item["path"]]
        item.update(byte_count=len(data), sha256=hashlib.sha256(data).hexdigest())
    original = next(item for item in manifest["files"] if item["kind"] == "original")
    manifest["video"].update(original_sha256=original["sha256"], byte_count=original["byte_count"])
    target = directory / f"{uuid4()}.zip"
    with ZipFile(target, "w") as archive:
        for name, data in files.items():
            archive.writestr(name, data)
        archive.writestr("manifest.json", canonical_json(manifest))
    return target


def reviewed_dataset(tmp_path):
    root = tmp_path / "data"
    paths = [source_zip(tmp_path / "sources", name) for name in ("one", "two")]
    draft = datasets.build_draft(root, paths, enabled=True)
    for frame in draft["frames"]:
        draft = datasets.save_frame_review(
            root,
            draft["dataset_id"],
            frame["token"],
            draft["revision"],
            frame["boxes"],
            "complete",
        )
    weights = tmp_path / "existing.pt"
    weights.write_bytes(b"synthetic checkpoint, never loaded in these tests")
    return root, draft, weights, paths


def test_disabled_workflow_does_not_read_sources_or_create_files(tmp_path):
    root = tmp_path / "unused"
    assert datasets.build_draft(root, [tmp_path / "missing.zip"], enabled=False) is None
    assert not root.exists()


def test_draft_preserves_source_predictions_and_starts_unreviewed(tmp_path):
    path = source_zip(tmp_path)
    before = path.read_bytes()
    root = tmp_path / "data"
    draft = datasets.build_draft(root, [path], enabled=True)
    frame = draft["frames"][0]
    assert frame["status"] == "pending"
    assert frame["boxes"][0]["origin"] == "yolo"
    assert frame["boxes"][0]["class_id"] == 41
    assert frame["key"]["timestamp_ms"] == 0
    assert datasets.build_draft(root, [path], enabled=True)["revision"] == draft["revision"]
    assert path.read_bytes() == before


def test_review_rejects_stale_revision_bad_boxes_and_accidental_negatives(tmp_path):
    root = tmp_path / "data"
    draft = datasets.build_draft(root, [source_zip(tmp_path)], enabled=True)
    frame = draft["frames"][0]
    args = (root, draft["dataset_id"], frame["token"], draft["revision"])
    with pytest.raises(ValueError, match="negative|empty|cup"):
        datasets.save_frame_review(*args, [], "complete")
    with pytest.raises(ValueError, match="box|bound|image"):
        datasets.save_frame_review(*args, [{"bbox_xyxy": [0, 0, 999, 999]}], "complete")
    saved = datasets.save_frame_review(*args, frame["boxes"], "complete")
    with pytest.raises(ValueError, match="changed|revision|stale"):
        datasets.save_frame_review(*args, frame["boxes"], "complete")
    assert saved["revision"] == draft["revision"] + 1


def test_approved_release_maps_class_and_keeps_review_history(tmp_path):
    root, draft, weights, paths = reviewed_dataset(tmp_path)
    source_hashes = [hashlib.sha256(path.read_bytes()).hexdigest() for path in paths]
    release = datasets.freeze_release(
        root, draft["dataset_id"], draft["revision"], weights, enabled=True
    )
    manifest = datasets.verify_release(release)
    assert manifest["class_names"] == {"0": "cup"}
    assert len(manifest["frames"]) == 2
    assert {f["split"] for f in manifest["frames"]} == {"train", "val"}
    assert len(manifest["history"]) == 2
    assert (
        manifest["training"]["base_weights_sha256"]
        == hashlib.sha256(weights.read_bytes()).hexdigest()
    )
    assert datasets.yolo_line([10, 20, 30, 60], 100, 100) == "0 0.2 0.4 0.2 0.4"
    assert [hashlib.sha256(p.read_bytes()).hexdigest() for p in paths] == source_hashes
    assert (
        datasets.freeze_release(root, draft["dataset_id"], draft["revision"], weights, enabled=True)
        == release
    )
    image_path = release / manifest["frames"][0]["image_path"]
    image_path.write_bytes(b"corrupted copy")
    with pytest.raises(ValueError, match="hash|checksum|changed"):
        datasets.verify_release(release)


def test_single_video_and_pending_frames_cannot_be_approved(tmp_path):
    root = tmp_path / "data"
    draft = datasets.build_draft(root, [source_zip(tmp_path)], enabled=True)
    weights = tmp_path / "weights.pt"
    weights.write_bytes(b"weights")
    with pytest.raises(ValueError, match="(?i)review|pending"):
        datasets.freeze_release(root, draft["dataset_id"], draft["revision"], weights, enabled=True)
    frame = draft["frames"][0]
    draft = datasets.save_frame_review(
        root, draft["dataset_id"], frame["token"], draft["revision"], frame["boxes"], "complete"
    )
    with pytest.raises(ValueError, match="video|validation|split"):
        datasets.freeze_release(root, draft["dataset_id"], draft["revision"], weights, enabled=True)


def test_new_sources_retain_earlier_reviews_and_edits_revoke_approval(tmp_path):
    root, draft, weights, _ = reviewed_dataset(tmp_path)
    datasets.freeze_release(root, draft["dataset_id"], draft["revision"], weights, enabled=True)
    draft = datasets.build_draft(root, [source_zip(tmp_path, "three")], enabled=True)
    assert len(draft["frames"]) == 3
    assert [f["status"] for f in draft["frames"]].count("complete") == 2
    assert draft["approval"] is None
    frame = draft["frames"][0]
    boxes = [{"box_id": frame["boxes"][0]["box_id"], "bbox_xyxy": [11, 20, 31, 60]}]
    edited = datasets.save_frame_review(
        root, draft["dataset_id"], frame["token"], draft["revision"], boxes, "complete"
    )
    assert edited["frames"][0]["boxes"][0]["edited"] is True
    assert edited["frames"][0]["boxes"][0]["origin"] == "yolo"


def test_repeated_frame_identity_cannot_change_source_video(tmp_path):
    path = source_zip(tmp_path)
    root = tmp_path / "d"
    datasets.build_draft(root, [path], enabled=True)
    with ZipFile(path) as archive:
        files = {name: archive.read(name) for name in archive.namelist()}
    manifest = json.loads(files["manifest.json"])
    files["original.mp4"] = b"different original video with the same frame identity"
    original = next(f for f in manifest["files"] if f["kind"] == "original")
    original.update(
        sha256=hashlib.sha256(files["original.mp4"]).hexdigest(),
        byte_count=len(files["original.mp4"]),
    )
    manifest["video"].update(original_sha256=original["sha256"], byte_count=original["byte_count"])
    files["manifest.json"] = canonical_json(manifest)
    other = tmp_path / "other.zip"
    with ZipFile(other, "w") as archive:
        for name, contents in files.items():
            archive.writestr(name, contents)
    with pytest.raises(ValueError, match="identity|source"):
        datasets.build_draft(root, [other], enabled=True)


def test_reviewed_negative_is_explicit_and_approved_splits_stay_frozen(tmp_path):
    root, draft, weights, _ = reviewed_dataset(tmp_path)
    datasets.freeze_release(root, draft["dataset_id"], draft["revision"], weights, enabled=True)
    frame = draft["frames"][0]
    with pytest.raises(ValueError, match="split"):
        datasets.save_frame_review(
            root,
            draft["dataset_id"],
            frame["token"],
            draft["revision"],
            frame["boxes"],
            "complete",
            split="val",
        )
    draft = datasets.build_draft(root, [source_zip(tmp_path, "negative", box=False)], enabled=True)
    frame = draft["frames"][-1]
    draft = datasets.save_frame_review(
        root, draft["dataset_id"], frame["token"], draft["revision"], [], "reviewed_negative"
    )
    release = datasets.freeze_release(
        root, draft["dataset_id"], draft["revision"], weights, enabled=True
    )
    manifest = datasets.verify_release(release)
    negative = next(f for f in manifest["frames"] if f["status"] == "reviewed_negative")
    assert (release / negative["label_path"]).read_bytes() == b""
