"""Opt-in dataset review, separate from the existing evidence decisions."""

from __future__ import annotations

from functools import wraps
from itertools import combinations
from pathlib import Path

import cv2
import gradio as gr
from filelock import Timeout

from auditor import datasets
from auditor.contracts import AuditorError
from auditor.localization import clear_model_pause, propose_frame
from auditor.models import decode_image
from auditor.storage import safe_path, sha256_file
from auditor.training import TrainingWorker


class DatasetWorkspace:
    def __init__(self, settings, export_provider, *, worker=None):
        self.settings = settings
        self.root = settings.data_root
        self.export_provider = export_provider
        self.worker = worker or TrainingWorker(self.root)
        self.automatic = True

    def set_enabled(self, enabled):
        self.worker.set_enabled(enabled)

    def require_enabled(self):
        if not self.worker.enabled:
            raise AuditorError("Enable the dataset workflow first")

    def build(self, paths, selected_session, automatic):
        self.require_enabled()
        paths = [Path(p) for p in (paths or [])]
        if not paths and selected_session:
            paths = [Path(self.export_provider(selected_session))]
        draft = datasets.build_draft(self.root, paths, enabled=True)
        if automatic and self.automatic:
            for frame in draft["frames"]:
                if not self.worker.enabled or not self.automatic or draft["localization_uncertain"]:
                    break
                if frame["status"] == "pending" and frame["source_regions"]:
                    draft = propose_frame(
                        self.settings,
                        draft["dataset_id"],
                        frame["token"],
                        enabled=True,
                        cancel_check=lambda: not self.worker.enabled or not self.automatic,
                    )
        return draft

    def checked_frame(self, stamp, selected):
        self.require_enabled()
        if not stamp or stamp["token"] != selected:
            raise AuditorError("Wait for the selected frame to finish displaying")
        draft = datasets.load_draft(self.root, stamp["dataset_id"])
        datasets._revision(draft, stamp["revision"])
        return draft, datasets._frame(draft, selected)

    def edit(self, stamp, selected, box_id, coordinates, action):
        draft, frame = self.checked_frame(stamp, selected)
        boxes = frame["boxes"]
        if action == "add":
            boxes.append(datasets.Box(bbox_xyxy=coordinates).model_dump(mode="json"))
        else:
            box = next((b for b in boxes if b["box_id"] == box_id), None)
            if box is None:
                raise AuditorError("Choose a displayed cup box first")
            if action == "delete":
                boxes.remove(box)
            elif action == "update":
                box["bbox_xyxy"] = coordinates
            else:
                raise AuditorError("Unknown box action")
        return datasets.save_frame_review(
            self.root, draft["dataset_id"], selected, draft["revision"], boxes, "pending"
        )

    def mark(self, stamp, selected, disposition, split, checked):
        draft, frame = self.checked_frame(stamp, selected)
        if not checked:
            raise AuditorError("Confirm you checked the entire frame before marking it reviewed")
        return datasets.save_frame_review(
            self.root,
            draft["dataset_id"],
            selected,
            draft["revision"],
            frame["boxes"],
            disposition,
            split=split,
        )

    def approve(self, stamp, selected, base_weights):
        draft, _ = self.checked_frame(stamp, selected)
        base = Path(base_weights).expanduser().resolve()
        if str(base) != stamp["base_weights"] or sha256_file(base) != stamp["base_weights_sha256"]:
            raise AuditorError("Base checkpoint changed. Reload and review the training settings")
        release = datasets.freeze_release(
            self.root, draft["dataset_id"], draft["revision"], base, enabled=True
        )
        return self.worker.enqueue(release)


def corner_coordinates(index, width, height, first):
    if not isinstance(index, (tuple, list)) or len(index) != 2:
        raise AuditorError("Select a corner on the raw image")
    x, y = index
    if not (0 <= x <= width and 0 <= y <= height):
        raise AuditorError("Corner falls outside the native image")
    if first is None:
        return [x, y], [x, y, None, None]
    return None, [min(x, first[0]), min(y, first[1]), max(x, first[0]), max(y, first[1])]


def build_dataset_panel(settings, sessions, source_export_provider):
    workspace = DatasetWorkspace(settings, source_export_provider)
    with gr.Accordion("4. Build a reviewed cup dataset", open=False, elem_id="dataset-workspace"):
        master = gr.Checkbox(
            label="Enable dataset building and fine-tuning",
            value=False,
            info="Optional. Review every frame before approving training. Resets OFF on restart.",
        )
        automatic = gr.Checkbox(
            label="Automatically propose missing boxes",
            value=True,
            interactive=False,
            info="Uses the same Qwen model for separate box proposals. All proposals need review.",
        )
        with gr.Column(visible=False) as panel:
            gr.Markdown(
                "Add the selected saved session, or choose existing session ZIPs. Later imports retain earlier reviews."
            )
            archives = gr.File(
                label="Session ZIPs (optional)",
                file_count="multiple",
                file_types=[".zip"],
                type="filepath",
            )
            with gr.Row():
                build = gr.Button("Build or extend dataset", variant="primary")
                reload_button = gr.Button("Reload dataset")
            location = gr.Textbox(label="Local dataset folder", interactive=False)
            summary = gr.Textbox(label="Dataset readiness", interactive=False, lines=3)
            frames = gr.Dropdown(label="Dataset frame", choices=[], interactive=True)
            identity = gr.Textbox(label="Dataset frame identity", interactive=False)
            hints = gr.Textbox(label="Qwen hints and proposal status", interactive=False, lines=3)
            with gr.Accordion("Recover an uncertain dataset request", open=False):
                restarted = gr.Checkbox(label="I stopped or restarted Qwen", value=False)
                clear_pause = gr.Button("Clear dataset model pause")
            with gr.Row():
                raw = gr.Image(
                    label="Dataset raw image: select two box corners",
                    type="numpy",
                    interactive=False,
                    height=360,
                    elem_id="dataset-raw",
                    buttons=["fullscreen"],
                )
                overlay = gr.Image(
                    label="Draft boxes: verify every visible cup",
                    type="numpy",
                    interactive=False,
                    height=360,
                    buttons=["fullscreen"],
                )
            gr.Markdown(
                "Box the visible extent of each cup, including its handle. Use the raw image to check every cup. Remove false or duplicate boxes."
            )
            box_selector = gr.Dropdown(label="Cup box", choices=[], interactive=True)
            with gr.Row():
                coordinates = [
                    gr.Number(label=label, precision=2)
                    for label in ("Left x", "Top y", "Right x", "Bottom y")
                ]
            with gr.Row():
                add = gr.Button("Add box")
                update = gr.Button("Update selected box")
                delete = gr.Button("Delete selected box")
            with gr.Row():
                split = gr.Dropdown(
                    label="Whole-video split", choices=["train", "val"], value="train"
                )
                disposition = gr.Radio(
                    label="Frame decision",
                    choices=[
                        ("All visible cups boxed", "complete"),
                        ("No visible cups", "reviewed_negative"),
                        ("Exclude this frame", "exclude"),
                    ],
                    value="complete",
                )
            checked = gr.Checkbox(label="I checked the entire frame", value=False)
            mark = gr.Button("Save frame review", variant="primary")
            base = gr.Textbox(
                label="Existing base checkpoint",
                value=str(settings.weights_path),
                info="Local .pt file. Choose the current model or an earlier evaluated cup candidate.",
            )
            gr.Markdown(
                "Training after approval: CPU, 10 epochs, batch 2, image size 640, 0 loader workers. The new cup model is saved as a candidate."
            )
            approve = gr.Button(
                "Approve dataset and fine-tune", variant="primary", interactive=False
            )
            refresh_job = gr.Button("Refresh training status")
            job_status = gr.JSON(label="Training job")
        message = gr.Textbox(
            label="Dataset action",
            value="OFF. Normal analysis does not build a dataset or start training.",
            interactive=False,
        )
        stamp = gr.State(None)
        corner = gr.State(None)
    outputs = [
        frames,
        identity,
        raw,
        overlay,
        hints,
        box_selector,
        *coordinates,
        split,
        disposition,
        checked,
        stamp,
        corner,
        summary,
        location,
        approve,
        message,
        job_status,
    ]
    queue = dict(concurrency_id="dataset-review", concurrency_limit=1, api_visibility="private")

    def guarded(fn):
        @wraps(fn)
        def invoke(*args, **kwargs):
            try:
                return fn(*args, **kwargs)
            except (ValueError, OSError, Timeout, KeyError) as exc:
                raise gr.Error(str(exc)) from exc

        return invoke

    def render(draft, selected, base_weights, notice=""):
        if not draft:
            return {message: "Build a dataset from a saved session or session ZIP first."}
        frame = next((f for f in draft["frames"] if f["token"] == selected), draft["frames"][0])
        selected = frame["token"]
        directory = datasets.dataset_dir(settings.data_root, draft["dataset_id"])
        if workspace.worker.current is None:
            jobs = list((directory / "jobs").glob("*/job.json"))
            if jobs:
                workspace.worker.current = max(jobs, key=lambda p: p.stat().st_mtime_ns).parent
        path = safe_path(directory, frame["image_path"])
        if sha256_file(path) != frame["raw_sha256"]:
            raise AuditorError("Dataset image checksum changed")
        image = decode_image(path.read_bytes())
        drawn = image.copy()
        colors = {"yolo": (80, 180, 60), "qwen": (0, 180, 245), "human": (220, 140, 60)}
        for number, box in enumerate(frame["boxes"], 1):
            x1, y1, x2, y2 = map(round, box["bbox_xyxy"])
            color = colors[box["origin"]]
            cv2.rectangle(drawn, (x1, y1), (x2, y2), color, 2)
            cv2.putText(
                drawn, str(number), (x1, max(12, y1)), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1
            )
        box_choices = [
            (f"{i}. {b['origin']}{' (edited)' if b['edited'] else ''}", b["box_id"])
            for i, b in enumerate(frame["boxes"], 1)
        ]
        first = frame["boxes"][0] if frame["boxes"] else None
        base_path = Path(base_weights).expanduser().resolve()
        digest = (
            sha256_file(base_path) if base_path.is_file() and base_path.suffix == ".pt" else None
        )
        reasons = datasets.readiness(draft)
        if not digest:
            reasons.append("Choose an existing local .pt checkpoint")
        receipts = "\n".join(
            f"Box request: {r['status']} ({len(r.get('raw_response', ''))} response characters)"
            for r in frame["localization_receipts"]
        )
        if draft["localization_uncertain"]:
            receipts += "\nA Qwen request may still be running. Further automatic requests are blocked. Manual review remains available."
        # At most 256 boxes in one frame. Warn without merging potentially distinct cups.
        if any(
            datasets.box_iou(a["bbox_xyxy"], b["bbox_xyxy"]) >= 0.75
            for a, b in combinations(frame["boxes"], 2)
        ):
            receipts += "\nSome boxes overlap heavily. Check for duplicate annotations."
        key = frame["key"]
        result = {
            frames: gr.update(
                choices=[
                    (f"{i}. {f['key']['frame_id']} | {f['status']} | {f['split']}", f["token"])
                    for i, f in enumerate(draft["frames"], 1)
                ],
                value=selected,
            ),
            identity: f"{key['session_id']} | {key['frame_id']} | {key['timestamp_ms']} ms | {frame['width']}x{frame['height']} | {frame['status']}",
            raw: cv2.cvtColor(image, cv2.COLOR_BGR2RGB),
            overlay: cv2.cvtColor(drawn, cv2.COLOR_BGR2RGB),
            hints: "\n".join(frame["hints"]) + "\n" + receipts,
            box_selector: gr.update(choices=box_choices, value=first["box_id"] if first else None),
            split: frame["split"],
            disposition: frame["status"] if frame["status"] != "pending" else "complete",
            checked: False,
            corner: None,
            stamp: dict(
                dataset_id=draft["dataset_id"],
                revision=draft["revision"],
                token=selected,
                base_weights=str(base_path),
                base_weights_sha256=digest,
            ),
            location: str(directory),
            summary: f"Revision {draft['revision']}. {len(draft['frames'])} frames. "
            + ("; ".join(reasons) if reasons else "Ready for approval. Base SHA-256: " + digest),
            approve: gr.update(interactive=workspace.worker.enabled and not reasons),
            message: notice,
            job_status: workspace.worker.status(),
        }
        result.update(zip(coordinates, first["bbox_xyxy"] if first else [None] * 4, strict=True))
        return result

    def toggle(enabled):
        workspace.set_enabled(enabled)
        return (
            gr.update(interactive=enabled),
            gr.update(visible=enabled),
            (
                "ON. Import a session, then check every cup before approving."
                if enabled
                else "OFF. Pending work is cancelled. No new model jobs will start."
            ),
        )

    master.change(
        toggle,
        inputs=master,
        outputs=[automatic, panel, message],
        queue=False,
        api_visibility="private",
    )

    @guarded
    def build_click(paths, session, auto, base_weights):
        draft = workspace.build(paths, session, auto)
        return render(
            draft,
            None,
            base_weights,
            "Draft ready. Model boxes are unapproved until you review each frame.",
        )

    def toggle_automatic(enabled):
        workspace.automatic = bool(enabled)

    automatic.change(toggle_automatic, inputs=automatic, queue=False, api_visibility="private")

    @guarded
    def clear_pause_click(confirmed, base_weights, selected):
        workspace.require_enabled()
        clear_model_pause(settings, confirmed)
        return render(
            datasets.load_draft(settings.data_root),
            selected,
            base_weights,
            "Pause cleared. Previous requests are not retried automatically.",
        )

    clear_pause.click(clear_pause_click, inputs=[restarted, base, frames], outputs=outputs, **queue)

    @guarded
    def reload_view(base_weights, selected):
        workspace.require_enabled()
        return render(
            datasets.load_draft(settings.data_root), selected, base_weights, "Loaded local dataset."
        )

    @guarded
    def box_choice(view, selected, box_id):
        _, frame = workspace.checked_frame(view, selected)
        box = next((b for b in frame["boxes"] if b["box_id"] == box_id), None)
        return box["bbox_xyxy"] if box else [None] * 4

    @guarded
    def choose_corner(view, selected, first, event: gr.SelectData):
        _, frame = workspace.checked_frame(view, selected)
        next_corner, values = corner_coordinates(
            event.index, frame["width"], frame["height"], first
        )
        return [next_corner, *values]

    @guarded
    def mark_frame(view, selected, decision, video_split, confirmed, base_weights):
        draft = workspace.mark(view, selected, decision, video_split, confirmed)
        return render(
            draft,
            selected,
            base_weights,
            "Frame review saved. Editing boxes makes the frame pending again.",
        )

    @guarded
    def approve_click(view, selected, base_weights):
        job = workspace.approve(view, selected, base_weights)
        return render(
            datasets.load_draft(settings.data_root),
            selected,
            base_weights,
            f"Approved release. Training job: {job.name}. Current detector stays active.",
        )

    @guarded
    def refresh_training():
        workspace.require_enabled()
        return workspace.worker.status()

    build.click(build_click, inputs=[archives, sessions, automatic, base], outputs=outputs, **queue)
    reload_button.click(reload_view, inputs=[base, frames], outputs=outputs, **queue)
    frames.input(
        reload_view, inputs=[base, frames], outputs=outputs, trigger_mode="always_last", **queue
    )
    base.change(
        reload_view, inputs=[base, frames], outputs=outputs, trigger_mode="always_last", **queue
    )
    box_selector.input(
        box_choice, inputs=[stamp, frames, box_selector], outputs=coordinates, **queue
    )
    raw.select(
        choose_corner, inputs=[stamp, frames, corner], outputs=[corner, *coordinates], **queue
    )
    for button, action in ((add, "add"), (update, "update"), (delete, "delete")):

        @guarded
        def edit_box(view, selected, box_id, left, top, right, bottom, base_weights, action=action):
            draft = workspace.edit(view, selected, box_id, [left, top, right, bottom], action)
            return render(
                draft,
                selected,
                base_weights,
                "Box edit saved. Review the entire frame before marking it complete.",
            )

        button.click(
            edit_box,
            inputs=[stamp, frames, box_selector, *coordinates, base],
            outputs=outputs,
            **queue,
        )
    mark.click(
        mark_frame,
        inputs=[stamp, frames, disposition, split, checked, base],
        outputs=outputs,
        **queue,
    )
    approve.click(approve_click, inputs=[stamp, frames, base], outputs=outputs, **queue)
    refresh_job.click(refresh_training, outputs=job_status, **queue)

    @guarded
    def after_analysis(selected, auto, base_weights):
        if not workspace.worker.enabled:
            return {component: gr.skip() for component in outputs}
        return build_click(None, selected, auto, base_weights)

    return after_analysis, [sessions, automatic, base], outputs
