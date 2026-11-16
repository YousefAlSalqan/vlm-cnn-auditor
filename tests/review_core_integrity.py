"""Independent core review reproductions; synthetic data and isolated test schemas only."""

import hashlib
import json
from concurrent.futures import ThreadPoolExecutor, TimeoutError
from fractions import Fraction as RealFraction
from uuid import uuid4
from zipfile import ZipFile

import numpy as np
import pytest
from sqlalchemy import func, select
from sqlalchemy.orm import Session

import auditor.video as video_module
from auditor.contracts import AuditorError, VideoProbe
from auditor.db import VLMReview
from auditor.evidence import create_upload
from auditor.export import export_session, verify_export
from auditor.pipeline import _confirm_runtime_restart, recover_interrupted, run_analysis
from auditor.storage import canonical_json, session_lock
from tests.fakes import FakeDetector, FakeReviewer
from tests.test_workflow import processed


def rewrite_archive(source, target, change):
    with ZipFile(source) as archive:
        contents = {name: archive.read(name) for name in archive.namelist()}
    manifest = json.loads(contents["manifest.json"])
    change(contents, manifest)
    contents["manifest.json"] = canonical_json(manifest)
    with ZipFile(target, "w") as archive:
        for name, content in contents.items():
            archive.writestr(name, content)
    return target


@pytest.mark.integration
@pytest.mark.parametrize(
    "inconsistency",
    [
        "raw_digest",
        "configuration_digest",
        "session_revision",
        "unscoped_file",
        "omitted_frame",
        "timestamp_inventory",
    ],
)
def test_export_verification_rejects_internal_evidence_drift(
    engine, app_settings, small_clip, tmp_path, inconsistency
):
    session_id, _, _ = processed(engine, app_settings, small_clip)
    source = export_session(engine, app_settings.data_root, session_id)

    def change(contents, manifest):
        if inconsistency == "raw_digest":
            path = manifest["frames"][0]["raw_path"]
            contents[path] = b"changed bytes with unchanged snapshot artifact digest"
            inventory = next(item for item in manifest["files"] if item["path"] == path)
            inventory["sha256"] = hashlib.sha256(contents[path]).hexdigest()
            inventory["byte_count"] = len(contents[path])
        elif inconsistency == "configuration_digest":
            manifest["configuration"]["sample_hz"] = 2.0
        elif inconsistency == "session_revision":
            manifest["session_revision"] = 1
        elif inconsistency == "unscoped_file":
            contents[".env"] = b"synthetic private configuration, not application evidence"
            manifest["files"].append(
                dict(
                    path=".env",
                    kind="original",
                    byte_count=len(contents[".env"]),
                    sha256=hashlib.sha256(contents[".env"]).hexdigest(),
                )
            )
        elif inconsistency == "omitted_frame":
            manifest["frames"].pop()
        else:
            manifest["video"]["probe_data"]["timestamps"] = []

    changed = rewrite_archive(source, tmp_path / "inconsistent.zip", change)
    with pytest.raises(AuditorError):
        verify_export(changed)


class SimulatedProcessExit(BaseException):
    pass


class InterruptedReviewer(FakeReviewer):
    def review(self, *args):
        self.calls += 1
        raise SimulatedProcessExit("synthetic process loss during the model request")


@pytest.mark.integration
def test_interrupted_model_request_remains_recorded_and_requires_runtime_restart(
    engine, app_settings, small_clip
):
    session_id = create_upload(engine, app_settings, small_clip)
    reviewer = InterruptedReviewer()
    with pytest.raises(SimulatedProcessExit):
        list(
            run_analysis(
                engine, app_settings, session_id, detector=FakeDetector(), reviewer=reviewer
            )
        )
    assert reviewer.calls == 1
    recover_interrupted(engine)
    with Session(engine) as db:
        attempts = db.scalar(
            select(func.count()).select_from(VLMReview).where(VLMReview.session_id == session_id)
        )
    assert attempts == 1, "The dispatched model request disappeared from durable history"
    next_reviewer = FakeReviewer()
    list(
        run_analysis(
            engine, app_settings, session_id, detector=FakeDetector(), reviewer=next_reviewer
        )
    )
    assert next_reviewer.calls == 0, "An uncertain earlier model call needs explicit restart gating"


@pytest.mark.integration
def test_resume_after_interrupted_dispatch_requires_explicit_idle_confirmation(
    engine, app_settings, small_clip
):
    session_id = create_upload(engine, app_settings, small_clip)
    with pytest.raises(SimulatedProcessExit):
        list(
            run_analysis(
                engine,
                app_settings,
                session_id,
                detector=FakeDetector(),
                reviewer=InterruptedReviewer(),
            )
        )
    recover_interrupted(engine)
    next_reviewer = FakeReviewer()
    list(
        run_analysis(
            engine, app_settings, session_id, detector=FakeDetector(), reviewer=next_reviewer
        )
    )
    assert next_reviewer.calls == 0, (
        "The first model request may still be running after the app exits"
    )


@pytest.mark.integration
def test_resume_refuses_a_different_prompt_in_an_existing_session(engine, app_settings, small_clip):
    session_id, _, _ = processed(engine, app_settings, small_clip, FakeReviewer(fail=True))
    reviewer = FakeReviewer()
    reviewer.metadata = reviewer.metadata | {
        "prompt_sha256": "d" * 64,
        "prompt_version": "cup_review_changed",
    }
    with pytest.raises(AuditorError, match="[Pp]rompt"):
        list(
            run_analysis(
                engine,
                app_settings,
                session_id,
                detector=FakeDetector(),
                reviewer=reviewer,
                retry_after_restart=True,
            )
        )
    assert reviewer.calls == 0


@pytest.mark.integration
def test_runtime_restart_confirmation_respects_other_session_evidence_lock(
    engine, app_settings, small_clip
):
    session_id, _, _ = processed(engine, app_settings, small_clip, FakeReviewer(fail=True))
    with ThreadPoolExecutor(max_workers=1) as executor:
        with session_lock(app_settings.data_root, session_id):
            confirmation = executor.submit(_confirm_runtime_restart, engine, app_settings.data_root)
            with pytest.raises(TimeoutError):
                confirmation.result(timeout=0.2)
            with Session(engine) as db:
                review = db.scalar(select(VLMReview).where(VLMReview.session_id == session_id))
                assert review.runtime_restart_confirmed_at is None
        confirmation.result(timeout=5)
    with Session(engine) as db:
        review = db.scalar(select(VLMReview).where(VLMReview.session_id == session_id))
        assert review.runtime_restart_confirmed_at is not None


def test_sampling_advances_across_timestamp_gaps_with_bounded_work(monkeypatch):
    class FakeCapture:
        def __init__(self, *args):
            self.remaining = 2

        def isOpened(self):
            return True

        def set(self, *args):
            return True

        def read(self):
            self.remaining -= 1
            return (
                (True, np.zeros((8, 8, 3), dtype=np.uint8))
                if self.remaining >= 0
                else (False, None)
            )

        def release(self):
            pass

    calls = 0

    def bounded_fraction(*args):
        nonlocal calls
        calls += 1
        assert calls < 100, "Sampling loops through missed target times instead of source frames"
        return RealFraction(*args)

    probe = VideoProbe(
        container="mp4",
        codec="h264",
        duration_us=15_000_000,
        width=8,
        height=8,
        stream_index=0,
        time_base_num=1,
        time_base_den=1,
        first_pts=0,
        timestamps=[0, 100_000],
        timestamp_source="pts",
        probe_version="synthetic",
    )
    monkeypatch.setattr(video_module.cv2, "VideoCapture", FakeCapture)
    monkeypatch.setattr(video_module, "Fraction", bounded_fraction)
    frames = list(video_module.iter_sampled_frames("synthetic", uuid4(), probe, 1))
    assert len(frames) == 2
