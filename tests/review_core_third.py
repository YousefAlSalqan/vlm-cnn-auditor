"""Third independent core review reproductions, using synthetic isolated evidence."""

import pytest

from auditor.contracts import AuditorError
from auditor.evidence import create_upload
from auditor.export import export_session, verify_export
from auditor.pipeline import run_analysis
from tests.fakes import FakeDetector, FakeReviewer
from tests.review_core_integrity import rewrite_archive
from tests.test_workflow import processed


@pytest.mark.integration
def test_resume_refuses_a_changed_detector_runtime_version(engine, app_settings, small_clip):
    session_id = create_upload(engine, app_settings, small_clip)
    running = run_analysis(
        engine, app_settings, session_id, detector=FakeDetector(), reviewer=FakeReviewer()
    )
    assert "Loading" in next(running)
    assert "YOLO: 1" in next(running)
    running.close()
    changed = FakeDetector()
    changed.metadata = changed.metadata | {
        "ultralytics": "different-runtime",
        "torch": "new-runtime",
    }
    with pytest.raises(AuditorError, match="[Rr]untime|YOLO|[Dd]etector"):
        list(
            run_analysis(
                engine, app_settings, session_id, detector=changed, reviewer=FakeReviewer()
            )
        )


@pytest.mark.integration
def test_export_rejects_a_manifest_review_status_that_contradicts_its_snapshot(
    engine, app_settings, small_clip, tmp_path
):
    session_id, _, _ = processed(engine, app_settings, small_clip)
    source = export_session(engine, app_settings.data_root, session_id)

    def change(contents, manifest):
        assert manifest["frames"][0]["review_status"] == "validated"
        manifest["frames"][0]["review_status"] = "not_selected"

    changed = rewrite_archive(source, tmp_path / "false-status.zip", change)
    with pytest.raises(AuditorError):
        verify_export(changed)
