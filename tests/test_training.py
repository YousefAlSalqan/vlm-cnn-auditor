import pytest
from filelock import FileLock

from auditor import datasets, training
from tests.test_datasets import reviewed_dataset


def release_fixture(tmp_path):
    root, draft, weights, _ = reviewed_dataset(tmp_path)
    release = datasets.freeze_release(
        root, draft["dataset_id"], draft["revision"], weights, enabled=True
    )
    return root, release, weights


def test_disabled_worker_and_invalid_release_cannot_train(tmp_path):
    worker = training.TrainingWorker(tmp_path, runner=lambda *args: pytest.fail("Training started"))
    with pytest.raises(ValueError, match="Enable|enabled"):
        worker.enqueue(tmp_path / "missing")
    assert not list(tmp_path.iterdir())
    worker.set_enabled(True)
    with pytest.raises((ValueError, OSError)):
        worker.enqueue(tmp_path / "missing")


def test_approval_is_idempotent_and_preserves_base_checkpoint(tmp_path):
    root, release, weights = release_fixture(tmp_path)
    before = weights.read_bytes()
    calls = []

    def fake_run(directory, manifest, cancelled):
        calls.append(directory)
        assert manifest["approved"] and not cancelled()
        return {"candidate": "run/weights/best.pt", "metrics": {"measured": False}}

    worker = training.TrainingWorker(root, runner=fake_run)
    worker.set_enabled(True)
    job = worker.enqueue(release)
    worker.thread.join(5)
    assert worker.status(job)["state"] == "completed"
    assert worker.enqueue(release) == job
    assert len(calls) == 1
    assert weights.read_bytes() == before
    datasets.verify_release(release)


def test_disable_cancels_job_waiting_for_analysis_lock(tmp_path):
    root, release, _ = release_fixture(tmp_path)
    worker = training.TrainingWorker(
        root, runner=lambda *args: pytest.fail("Ran despite cancellation")
    )
    worker.set_enabled(True)
    with FileLock(root / ".analysis.lock", timeout=0):
        job = worker.enqueue(release)
        worker.set_enabled(False)
        worker.thread.join(5)
    assert worker.status(job)["state"] == "cancelled"


def test_changed_base_weights_fail_before_runner(tmp_path):
    root, release, weights = release_fixture(tmp_path)
    weights.write_bytes(b"changed")
    worker = training.TrainingWorker(root, runner=lambda *args: pytest.fail("Used changed weights"))
    worker.set_enabled(True)
    job = worker.enqueue(release)
    worker.thread.join(5)
    assert worker.status(job)["state"] == "failed"
    assert "checkpoint" in worker.status(job)["error"].lower()


def test_disabling_running_worker_marks_candidate_cancelled(tmp_path):
    root, release, _ = release_fixture(tmp_path)

    def stop_during_run(job, manifest, cancelled):
        worker.set_enabled(False)
        assert cancelled()
        return {"activated": False}

    worker = training.TrainingWorker(root, runner=stop_during_run)
    worker.set_enabled(True)
    job = worker.enqueue(release)
    worker.thread.join(5)
    assert worker.status(job)["state"] == "cancelled"
    datasets.verify_release(release)


def test_incompatible_checkpoint_is_rejected():
    from types import SimpleNamespace

    with pytest.raises(ValueError, match="COCO cup"):
        training.cup_class(SimpleNamespace(task="detect", names={0: "bottle"}))


def test_restarted_worker_does_not_resume_interrupted_job(tmp_path):
    root, release, _ = release_fixture(tmp_path)
    first = training.TrainingWorker(root, runner=lambda *args: {})
    first.set_enabled(True)
    job = first.enqueue(release)
    first.thread.join(5)
    first._state(job, "running")  # Simulate a process that exited without a terminal receipt.
    second = training.TrainingWorker(root)
    assert second.status(job)["state"] == "interrupted"
    assert second.thread is None
    second.set_enabled(True)
    assert second.enqueue(release) == job
    assert second.thread is None
    assert second.status()["state"] == "interrupted"


def test_matching_counts_duplicates_and_misses_once():
    counts = training.match_boxes(
        [[0, 0, 10, 10], [1, 1, 10, 10], [80, 80, 90, 90]],
        [[0, 0, 10, 10], [30, 30, 40, 40]],
    )
    assert counts == {"tp": 1, "fp": 2, "fn": 1}


def test_candidate_training_uses_approved_copy_and_evaluates_class_mapping(tmp_path, monkeypatch):
    from pathlib import Path
    from types import SimpleNamespace

    import numpy as np

    root, release, weights = release_fixture(tmp_path)
    before = weights.read_bytes()
    calls = []
    classes = []

    class FakeModel:
        task = "detect"

        def __init__(self, path):
            self.candidate = Path(path).name == "best.pt"
            self.names = (
                {0: "cup"}
                if self.candidate
                else {i: "cup" if i == 41 else str(i) for i in range(80)}
            )
            self.callbacks = {}

        def add_callback(self, event, fn):
            self.callbacks[event] = fn

        def train(self, **config):
            calls.append(config)
            self.callbacks["on_train_start"](self)
            assert Path(config["data"]) != release / "data.yaml"
            candidate = Path(config["project"]) / "run/weights/best.pt"
            candidate.parent.mkdir(parents=True)
            candidate.write_bytes(b"synthetic candidate")

        def predict(self, image, **config):
            classes.append(config["classes"])
            boxes = [[10, 20, 30, 60]] * (1 if self.candidate else 2)
            return [
                SimpleNamespace(
                    boxes=SimpleNamespace(conf=np.array([0.8] * len(boxes)), xyxy=np.array(boxes))
                )
            ]

    monkeypatch.setattr(training.importlib.metadata, "version", lambda name: "test")
    worker = training.TrainingWorker(
        root,
        runner=lambda job, manifest, cancelled: training.train_candidate(
            job,
            manifest,
            cancelled,
            model_factory=FakeModel,
        ),
    )
    worker.set_enabled(True)
    job = worker.enqueue(release)
    worker.thread.join(5)
    status = worker.status(job)
    assert status["state"] == "completed", status
    assert classes == [[41], [0]]
    assert status["result"]["metrics"]["base"]["fp"] == 1
    assert status["result"]["metrics"]["candidate"]["fp"] == 0
    assert calls[0]["epochs"] == 10 and calls[0]["workers"] == 0
    assert status["result"]["activated"] is False
    assert weights.read_bytes() == before
    datasets.verify_release(release)
