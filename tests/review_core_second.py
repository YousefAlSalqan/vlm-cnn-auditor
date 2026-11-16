"""Second independent core review: frozen model identity and exported scope."""

import hashlib
import json
from uuid import uuid4

import pytest

import auditor.pipeline as pipeline_module
from auditor.contracts import AuditorError
from auditor.export import export_session, verify_export
from auditor.pipeline import run_analysis
from auditor.storage import canonical_json
from tests.fakes import FakeDetector, FakeReviewer
from tests.review_core_integrity import rewrite_archive
from tests.test_workflow import processed


@pytest.mark.integration
@pytest.mark.parametrize("drift", ["review_scope", "finding_detection", "review_response"])
def test_export_rejects_inconsistent_nested_review_evidence(
    engine, app_settings, small_clip, tmp_path, drift
):
    session_id, _, _ = processed(engine, app_settings, small_clip)
    source = export_session(engine, app_settings.data_root, session_id)

    def change(contents, manifest):
        path = manifest["frames"][0]["json_path"]
        snapshot = json.loads(contents[path])
        review = snapshot["vlm_reviews"][0]
        if drift == "review_scope":
            review["session_id"] = str(uuid4())
        elif drift == "finding_detection":
            review["findings"][0]["detection_ids"] = [str(uuid4())]
        else:
            response = json.loads(review["raw_response"])
            response["timestamp_ms"] = 12345
            review["raw_response"] = json.dumps(response)
        contents[path] = canonical_json(snapshot)
        inventory = next(item for item in manifest["files"] if item["path"] == path)
        inventory["sha256"] = hashlib.sha256(contents[path]).hexdigest()
        inventory["byte_count"] = len(contents[path])

    changed = rewrite_archive(source, tmp_path / "inconsistent-review.zip", change)
    with pytest.raises(AuditorError):
        verify_export(changed)


@pytest.mark.integration
def test_resume_refuses_a_different_model_when_initial_digest_was_unavailable(
    engine, app_settings, small_clip
):
    original = FakeReviewer(fail=True)
    original.metadata = original.metadata | {"model": "original-tag", "model_digest": None}
    session_id, _, _ = processed(engine, app_settings, small_clip, original)
    changed = FakeReviewer()
    changed.metadata = changed.metadata | {"model": "different-tag", "model_digest": "new-model"}
    with pytest.raises(AuditorError, match="[Mm]odel|Qwen"):
        list(
            run_analysis(
                engine,
                app_settings,
                session_id,
                detector=FakeDetector(),
                reviewer=changed,
                retry_after_restart=True,
            )
        )
    assert changed.calls == 0


@pytest.mark.integration
def test_resume_uses_the_frozen_request_deadline(engine, app_settings, small_clip, monkeypatch):
    session_id, _, _ = processed(engine, app_settings, small_clip, FakeReviewer(fail=True))
    actual_timeouts = []

    def local_reviewer(base_url, model, timeout):
        actual_timeouts.append(timeout)
        return FakeReviewer()

    monkeypatch.setattr(pipeline_module, "OllamaReviewer", local_reviewer)
    changed_settings = app_settings.model_copy(update={"request_timeout": 321.0})
    list(
        run_analysis(
            engine, changed_settings, session_id, detector=FakeDetector(), retry_after_restart=True
        )
    )
    assert actual_timeouts == [app_settings.request_timeout]
