"""One optional local training job, bound to an immutable human-approved release."""

from __future__ import annotations

import atexit
import hashlib
import importlib.metadata
import json
import os
import shutil
import subprocess
import sys
import threading
import time
from pathlib import Path

from filelock import FileLock, Timeout

from auditor import datasets
from auditor.contracts import AuditorError
from auditor.storage import atomic_write, canonical_json, safe_path, sha256_file


class TrainingCancelled(AuditorError):
    pass


def check_checkpoint(manifest):
    weights = Path(manifest["training"]["base_weights"])
    if not weights.is_file() or sha256_file(weights) != manifest["training"]["base_weights_sha256"]:
        raise AuditorError("Base checkpoint changed since approval")
    return weights


class TrainingWorker:
    def __init__(self, root, *, runner=None):
        self.root = Path(root)
        self.shutdown = threading.Event()
        self.runner = runner or (
            lambda job, manifest, cancelled: _subprocess_run(
                job, manifest, cancelled, self.shutdown.is_set
            )
        )
        self.enabled = False
        self.stop = threading.Event()
        self.guard = threading.Lock()
        self.thread = None
        self.current = None
        atexit.register(self.close)

    def set_enabled(self, enabled):
        with self.guard:
            self.enabled = bool(enabled)
            if not self.enabled:
                self.stop.set()

    def close(self):
        self.shutdown.set()
        self.set_enabled(False)
        if self.thread and self.thread.is_alive():
            self.thread.join(3)

    def enqueue(self, release):
        with self.guard:
            if not self.enabled:
                raise AuditorError("Enable the dataset workflow before training")
            release = Path(release).resolve()
            manifest = datasets.verify_release(release)
            directory = datasets.dataset_dir(self.root, manifest["dataset_id"])
            if release.parent.name != "releases" or release.parents[1] != directory:
                raise AuditorError("Choose a release inside this dataset workspace")
            spec = dict(
                release=str(release),
                manifest_sha256=sha256_file(release / "manifest.json"),
                training=manifest["training"],
            )
            key = hashlib.sha256(canonical_json(spec)).hexdigest()
            job = directory / "jobs" / key[:24]
            if (job / "spec.json").exists():
                if datasets._read(job / "spec.json") != spec:
                    raise AuditorError("Training job identity collision")
                if self.thread and self.thread.is_alive() and self.current != job:
                    raise AuditorError("A different training job is already active")
                self.current = job
                return job
            if self.thread and self.thread.is_alive():
                raise AuditorError("A training job is already active. Wait for it to finish")
            atomic_write(job / "spec.json", canonical_json(spec), immutable=True)
            self._state(job, "queued")
            self.current = job
            self.stop = threading.Event()
            self.thread = threading.Thread(target=self._run, args=(job,), daemon=True)
            self.thread.start()
            return job

    def _state(self, job, state, **details):
        atomic_write(
            job / "job.json",
            canonical_json(dict(state=state, updated_at=datasets._now(), **details)),
        )

    def status(self, job=None):
        job = job or self.current
        if not job:
            return {"state": "idle"}
        job = Path(job)
        state = datasets._read(job / "job.json")
        if state["state"] in ("queued", "waiting", "running") and not (
            self.thread and self.thread.is_alive()
        ):
            try:
                with FileLock(job / ".job.lock", timeout=0):
                    self._state(job, "interrupted", error="This job will not restart automatically")
                    state = datasets._read(job / "job.json")
            except Timeout:
                pass
        return state | {"job_directory": str(job)}

    def _run(self, job):
        try:
            with FileLock(job / ".job.lock", timeout=0):
                lock = FileLock(self.root / ".analysis.lock", timeout=0)
                self._state(job, "waiting")
                while not self.stop.is_set():
                    try:
                        lock.acquire()
                        break
                    except Timeout:
                        self.stop.wait(0.25)
                else:
                    raise TrainingCancelled("Cancelled before training started")
                try:
                    if self.stop.is_set() or not self.enabled:
                        raise TrainingCancelled("Dataset workflow was disabled")
                    if (self.root / ".dataset-qwen-pause.json").exists():
                        raise AuditorError(
                            "Restart Qwen and clear the dataset model pause before training"
                        )
                    spec = datasets._read(job / "spec.json")
                    manifest = datasets.verify_release(
                        Path(spec["release"]), expected_sha=spec["manifest_sha256"]
                    )
                    check_checkpoint(manifest)
                    self._state(job, "running")
                    result = self.runner(job, manifest, self.stop.is_set)
                    if self.stop.is_set():
                        raise TrainingCancelled(
                            "Cancellation requested. Candidate was not activated"
                        )
                    check_checkpoint(manifest)
                    datasets.verify_release(
                        Path(spec["release"]), expected_sha=spec["manifest_sha256"]
                    )
                    self._state(job, "completed", result=result)
                finally:
                    lock.release()
        except TrainingCancelled as exc:
            self._state(job, "cancelled", error=str(exc))
        except Exception as exc:
            self._state(job, "failed", error=f"{type(exc).__name__}: {exc}"[:1000])


def _subprocess_run(job, manifest, cancelled, shutdown=lambda: False):
    env = os.environ | {
        "YOLO_CONFIG_DIR": str(job / "runtime"),
        "YOLO_OFFLINE": "true",
        "YOLO_AUTOINSTALL": "false",
        "MPLCONFIGDIR": str(job / "runtime/matplotlib"),
        "OMP_NUM_THREADS": "2",
        "MKL_NUM_THREADS": "2",
        "HF_HUB_OFFLINE": "1",
    }
    with (job / "training.log").open("wb") as log:
        process = subprocess.Popen(
            [sys.executable, "-m", "auditor.training", "--run", str(job)],
            cwd=Path(__file__).resolve().parents[1],
            env=env,
            stdout=log,
            stderr=subprocess.STDOUT,
            creationflags=subprocess.CREATE_NO_WINDOW if os.name == "nt" else 0,
        )
        try:
            while process.poll() is None:
                if shutdown():
                    raise TrainingCancelled("Application is shutting down")
                if cancelled():
                    atomic_write(job / "cancel", b"cancel requested")
                time.sleep(0.2)
            if cancelled():
                raise TrainingCancelled("Training stopped at a safe boundary")
            if process.returncode:
                raise AuditorError("Training failed. See training.log in the job folder")
            return datasets._read(job / "result.json")
        finally:
            if process.poll() is None:
                process.terminate()
                process.wait(timeout=10)


def match_boxes(predictions, truth):
    """Greedy one-to-one IoU >= 0.5 matching, with predictions supplied in score order."""
    unmatched = set(range(len(truth)))
    tp = 0
    for box in predictions:
        best, best_iou = None, 0.0
        for index in unmatched:
            iou = datasets.box_iou(box, truth[index])
            if iou > best_iou:
                best, best_iou = index, iou
        if best is not None and best_iou >= 0.5:
            unmatched.remove(best)
            tp += 1
    return dict(tp=tp, fp=len(predictions) - tp, fn=len(unmatched))


def evaluate(model, class_id, manifest, release, cancelled):
    counts = dict(tp=0, fp=0, fn=0)
    for frame in manifest["frames"]:
        if frame["split"] != "val":
            continue
        if cancelled():
            raise TrainingCancelled("Cancelled during evaluation")
        result = model.predict(
            str(safe_path(release, frame["image_path"])),
            device="cpu",
            classes=[class_id],
            conf=0.25,
            iou=0.7,
            imgsz=640,
            max_det=100,
            save=False,
            verbose=False,
        )[0]
        boxes = result.boxes
        ordered = sorted(zip(boxes.conf.tolist(), boxes.xyxy.tolist(), strict=True), reverse=True)
        matched = match_boxes([box for _, box in ordered], [b["bbox_xyxy"] for b in frame["boxes"]])
        counts = {key: counts[key] + matched[key] for key in counts}
    return counts | dict(
        precision=counts["tp"] / (counts["tp"] + counts["fp"])
        if counts["tp"] + counts["fp"]
        else None,
        recall=counts["tp"] / (counts["tp"] + counts["fn"])
        if counts["tp"] + counts["fn"]
        else None,
        undefined_reason="A ratio is null when its denominator is zero",
    )


def cup_class(model):
    names = model.names
    if model.task != "detect":
        raise AuditorError("Use a YOLO object-detection checkpoint")
    if len(names) == 1 and names[0] == "cup":
        return 0
    if len(names) == 80 and names[41] == "cup":
        return 41
    raise AuditorError("Checkpoint must have COCO cup class 41 or a single cup class 0")


def train_candidate(job, manifest, cancelled, *, model_factory=None):
    if cancelled():
        raise TrainingCancelled("Cancelled before loading weights")
    if model_factory is None:
        from ultralytics import YOLO, settings
        from ultralytics.data import utils as data_utils

        settings.update(
            dict(
                sync=False,
                **{
                    k: False
                    for k in (
                        "clearml",
                        "comet",
                        "dvc",
                        "mlflow",
                        "raytune",
                        "tensorboard",
                        "wandb",
                    )
                },
            )
        )
        # Plotting is disabled. Avoid the dataset validator's optional font download.
        data_utils.check_font = lambda *args, **kwargs: None
        model_factory = YOLO
    base = check_checkpoint(manifest)
    spec = datasets._read(job / "spec.json")
    release = Path(spec["release"])
    data = job / "data"
    shutil.copytree(release / "images", data / "images")
    shutil.copytree(release / "labels", data / "labels")
    atomic_write(
        data / "data.yaml",
        (
            f"path: {json.dumps(str(data.resolve()))}\ntrain: images/train\nval: images/val\nnames:\n  0: cup\n"
        ).encode(),
    )
    checkpoint = job / "base.pt"
    shutil.copyfile(base, checkpoint)
    if sha256_file(checkpoint) != manifest["training"]["base_weights_sha256"]:
        raise AuditorError("Checkpoint copy differs from approved weights")
    model = model_factory(str(checkpoint))
    base_class = cup_class(model)

    def boundary(trainer):
        if cancelled():
            raise TrainingCancelled("Training cancelled at a batch boundary")

    model.add_callback("on_train_batch_end", boundary)
    model.add_callback("on_train_start", boundary)
    if cancelled():
        raise TrainingCancelled("Cancelled before training started")
    config = {key: manifest["training"][key] for key in datasets.DEFAULT_TRAINING}
    if config != datasets.DEFAULT_TRAINING:
        raise AuditorError(
            "Approved training settings differ from this version's supported settings"
        )
    model.train(
        data=str(data / "data.yaml"),
        **config,
        project=str(job),
        name="run",
        exist_ok=True,
        cache=False,
        plots=False,
        amp=False,
        save=True,
        resume=False,
        verbose=False,
    )
    candidate = job / "run/weights/best.pt"
    if not candidate.is_file():
        raise AuditorError("Training produced no best checkpoint")
    candidate_model = model_factory(str(candidate))
    if cup_class(candidate_model) != 0:
        raise AuditorError("Candidate checkpoint did not produce the one-class cup model")
    metrics = dict(
        policy=dict(
            conf=0.25,
            nms_iou=0.7,
            match_iou=0.5,
            matching="score-ordered greedy",
            unit="held-out source videos",
        ),
        base=evaluate(model_factory(str(checkpoint)), base_class, manifest, release, cancelled),
        candidate=evaluate(candidate_model, 0, manifest, release, cancelled),
    )
    versions = {
        name: importlib.metadata.version(name) for name in ("ultralytics", "torch", "opencv-python")
    }
    return dict(
        candidate="run/weights/best.pt",
        candidate_sha256=sha256_file(candidate),
        base_weights_sha256=manifest["training"]["base_weights_sha256"],
        manifest_sha256=spec["manifest_sha256"],
        versions=versions,
        canonical_class_id=41,
        candidate_class_id=0,
        metrics=metrics,
        activated=False,
    )


def main():
    import argparse

    parser = argparse.ArgumentParser(description="Run one approved local dataset job")
    parser.add_argument("--run", type=Path, required=True)
    job = parser.parse_args().run.resolve()
    spec = datasets._read(job / "spec.json")
    manifest = datasets.verify_release(Path(spec["release"]), expected_sha=spec["manifest_sha256"])
    result = train_candidate(job, manifest, lambda: (job / "cancel").exists())
    atomic_write(job / "result.json", canonical_json(result), immutable=True)


if __name__ == "__main__":
    main()
