from __future__ import annotations

import os
from pathlib import Path
from uuid import UUID, uuid4

os.environ.setdefault("GRADIO_ANALYTICS_ENABLED", "False")
os.environ.setdefault("HF_HUB_DISABLE_TELEMETRY", "1")

import cv2
import gradio as gr
from filelock import FileLock, Timeout

from auditor.config import Settings
from auditor.contracts import FrameKey
from auditor.db import create_engine_for, require_schema
from auditor.evidence import (
    analysis_running,
    create_upload,
    frame_bundle,
    list_frames,
    list_sessions,
    record_decision,
    session_summary,
)
from auditor.export import export_session
from auditor.models import OllamaReviewer, decode_image
from auditor.pipeline import recover_interrupted, request_cancel, run_analysis
from auditor.regional import region_observations
from auditor.storage import safe_path, session_dir

CSS = (Path(__file__).parent / "style.css").read_text(encoding="utf-8")
THEME = gr.themes.Base(
    primary_hue="teal",
    neutral_hue="slate",
    font=[gr.themes.Font("Segoe UI"), gr.themes.Font("Arial")],
    font_mono=[gr.themes.Font("Consolas")],
)


def frame_review_status(bundle: dict) -> str:
    if not bundle["vlm_selection"]["selected"]:
        return "not_selected"
    reviews = bundle["vlm_reviews"]
    if not reviews:
        return "not_reviewed"
    latest = reviews[-1]
    return latest["outcome"] if latest["status"] == "validated" else "failed"


def build_app(settings: Settings, engine, *, detector=None, reviewer=None):
    settings.data_root.mkdir(parents=True, exist_ok=True)

    def session_choices():
        # shortcut: show the latest 50 sessions; add history pagination when this first page fills.
        return [
            (
                f"{s['created_at'][:16].replace('T', ' ')} | {s['session_id'][:8]} | {s['state']}",
                s["session_id"],
            )
            for s in list_sessions(engine)
        ]

    with gr.Blocks(
        title="VLM-CNN Auditor", analytics_enabled=False, delete_cache=(86400, 86400)
    ) as app:
        gr.HTML(
            '<header id="masthead"><div><span class="eyebrow">LOCAL REVIEW WORKSPACE</span>'
            "<h1>VLM-CNN Auditor</h1><p>Inspect cup detections. Keep the evidence. Make the decision.</p>"
            '</div><span class="local-badge">LOCAL ONLY</span></header>'
        )
        system_status = gr.Textbox(
            label="Workspace status",
            value="Checking local models…",
            interactive=False,
            elem_id="system-status",
        )
        with gr.Row(elem_id="workspace"):
            with gr.Column(scale=1, min_width=280, elem_id="sidebar"):
                gr.Markdown("### 1. Add a video")
                upload = gr.File(label="Phone video", file_types=[".mp4", ".mov"], type="filepath")
                gr.Markdown("15–30 seconds · MP4 or MOV · up to 1080p and 200 MiB")
                save = gr.Button("Save video", variant="primary")
                upload_message = gr.Textbox(label="Upload result", interactive=False, lines=2)
                gr.Markdown("### Saved sessions")
                sessions = gr.Dropdown(
                    label="Saved session", choices=session_choices(), interactive=True
                )
                refresh = gr.Button("Refresh sessions", size="sm")
                session_info = gr.Textbox(label="Session", interactive=False, lines=4)
                start = gr.Button("Start analysis", variant="primary", interactive=False)
                cancel = gr.Button("Stop after current step", interactive=False)
                progress = gr.Textbox(
                    label="Analysis progress",
                    interactive=False,
                    lines=4,
                    value="Save a video to begin. Uploading does not start analysis.",
                )
                with gr.Accordion("Recovery", open=False):
                    retry = gr.Checkbox(
                        label="I restarted Qwen. Retry failed reviews.",
                        value=False,
                        info="Use after a timeout. A timed-out model call may still be running.",
                    )
                with gr.Accordion("Original video", open=False):
                    original = gr.Video(
                        label="Preserved original", interactive=False, buttons=["download"]
                    )
            with gr.Column(scale=3, min_width=420, elem_id="evidence-panel"):
                gr.Markdown("### 2. Review evidence")
                frame_selector = gr.Dropdown(
                    label="Frame (flagged frames first)", choices=[], interactive=False
                )
                frame_heading = gr.Textbox(
                    label="Frame identity", value="No frame selected", interactive=False
                )
                with gr.Row(elem_id="frame-pair"):
                    raw_image = gr.Image(
                        label="Raw frame",
                        interactive=False,
                        type="numpy",
                        height=300,
                        buttons=["fullscreen"],
                    )
                    annotated_image = gr.Image(
                        label="YOLO overlay",
                        interactive=False,
                        type="numpy",
                        height=300,
                        buttons=["fullscreen"],
                    )
                review_status = gr.Textbox(
                    label="Qwen review",
                    interactive=False,
                    value="Frames appear after analysis. Unreviewed does not mean correct.",
                )
                regional_notes = gr.Textbox(
                    label="Qwen observations (unverified)", interactive=False, lines=4
                )
                finding_selector = gr.Dropdown(label="Finding", choices=[], interactive=False)
                details = gr.Textbox(label="Evidence and explanation", interactive=False, lines=4)
                note = gr.Textbox(
                    label="Your note (optional)",
                    placeholder="Why you accepted, rejected or deferred this suggestion",
                    interactive=False,
                    lines=2,
                )
                with gr.Row():
                    accept = gr.Button("Accept", variant="primary", interactive=False)
                    reject = gr.Button("Reject", interactive=False)
                    defer = gr.Button("Defer", interactive=False)
                decision_message = gr.Textbox(label="Decision result", interactive=False)
                gr.Markdown(
                    "Decisions record your judgment. They do not alter YOLO predictions or retrain a model."
                )
                with gr.Accordion("Decision history", open=False):
                    history = gr.Dataframe(
                        headers=["Revision", "Action", "Note", "Recorded UTC"],
                        datatype=["number", "str", "str", "str"],
                        interactive=False,
                        elem_id="decision-history",
                    )
                with gr.Accordion("Saved frame metadata", open=False):
                    metadata = gr.JSON(label="Canonical frame record")
                gr.Markdown("### 3. Export the session")
                gr.Markdown(
                    "Includes the original video, frame pairs, JSON records and a checksum manifest. Exports contain your footage."
                )
                export_button = gr.Button("Create session ZIP", interactive=False)
                export_file = gr.File(label="Session ZIP", interactive=False)
                export_message = gr.Textbox(label="Export result", interactive=False)
        view_key = gr.State(None)
        displayed_key = gr.JSON(value=None, visible=False)
        active_job = gr.State(None)
        recovery_ack = gr.State(False)
        expected_number = gr.Number(value=0, precision=0, visible=False)
        decision_token = gr.Textbox(value=str(uuid4()), visible=False)
        view_queue = {"concurrency_id": "review", "concurrency_limit": 1}

        view_outputs = [
            raw_image,
            annotated_image,
            frame_heading,
            review_status,
            regional_notes,
            finding_selector,
            details,
            note,
            view_key,
            displayed_key,
            expected_number,
            decision_token,
            history,
            metadata,
            accept,
            reject,
            defer,
            decision_message,
            export_file,
        ]

        def empty_view(message="No frames yet. Start analysis to create synchronized evidence."):
            return {
                raw_image: None,
                annotated_image: None,
                frame_heading: "No frame selected",
                review_status: message,
                regional_notes: "",
                finding_selector: gr.update(choices=[], value=None, interactive=False),
                details: "",
                note: gr.update(value="", interactive=False),
                view_key: None,
                displayed_key: None,
                expected_number: 0,
                decision_token: str(uuid4()),
                history: [],
                metadata: {},
                accept: gr.update(interactive=False),
                reject: gr.update(interactive=False),
                defer: gr.update(interactive=False),
                decision_message: "",
                export_file: None,
            }

        def load_frame(selected, frame_id, current_key=None, preferred=None):
            if not selected or not frame_id:
                return empty_view()
            try:
                session_id = UUID(selected)
                frames = list_frames(engine, session_id)
                frame = next((f for f in frames if f["frame_id"] == frame_id), None)
                if frame is None:
                    return empty_view("This frame does not belong to the selected session.")
                key = FrameKey(
                    session_id=session_id, frame_id=frame_id, timestamp_ms=frame["timestamp_ms"]
                )
                bundle = frame_bundle(engine, settings.data_root, key)
                directory = session_dir(settings.data_root, session_id)
                raw = decode_image(
                    safe_path(directory, bundle["artifacts"]["raw"]["path"]).read_bytes()
                )
                overlay = decode_image(
                    safe_path(directory, bundle["artifacts"]["annotated"]["path"]).read_bytes()
                )
                findings = [f for review in bundle["vlm_reviews"] for f in review["findings"]]
                if (
                    preferred is None
                    and current_key
                    and current_key["session_id"] == selected
                    and current_key["frame_id"] == frame_id
                ):
                    preferred = current_key.get("finding_id")
                choices = [
                    (f"{f['rule_id']} | {f['evidence'][:80]}", f["finding_id"]) for f in findings
                ]
                chosen = next(
                    (f for f in findings if f["finding_id"] == preferred),
                    findings[0] if findings else None,
                )
                decisions = bundle["review_decisions"]
                context = key.model_dump(mode="json") | {
                    "finding_id": chosen["finding_id"] if chosen else None
                }
                note_update = (
                    gr.update(interactive=bool(chosen))
                    if context == current_key
                    else gr.update(value="", interactive=bool(chosen))
                )
                number = max(
                    (
                        d["decision_number"]
                        for d in decisions
                        if chosen and d["finding_id"] == chosen["finding_id"]
                    ),
                    default=0,
                )
                labels = {
                    "not_selected": "Not selected for Qwen review. This frame has not been cleared.",
                    "not_reviewed": "Selected, but Qwen has not reviewed this frame. See session diagnostics.",
                    "failed": "Qwen review failed. YOLO evidence remains available for manual inspection.",
                    "no_issue": "Qwen found no suspected issue. This is not a verified human label.",
                    "insufficient_evidence": "Qwen did not establish a reviewable error. This frame still needs human inspection.",
                    "findings": "Qwen proposed findings. Verify each against the raw image.",
                }
                finding_update = gr.update(choices=choices, interactive=bool(chosen))
                if context != current_key:
                    finding_update["value"] = chosen["finding_id"] if chosen else None
                return {
                    raw_image: cv2.cvtColor(raw, cv2.COLOR_BGR2RGB),
                    annotated_image: cv2.cvtColor(overlay, cv2.COLOR_BGR2RGB),
                    frame_heading: f"Session {selected[:8]} | {frame_id} | {key.timestamp_ms / 1000:.3f}s | source index {bundle['source_frame_index']} | {len(bundle['yolo']['detections'])} YOLO cup boxes",
                    review_status: labels[frame_review_status(bundle)],
                    regional_notes: region_observations(bundle["vlm_reviews"][-1]["raw_response"])
                    if bundle["vlm_reviews"]
                    else "",
                    finding_selector: finding_update,
                    details: f"{chosen['rule_id']}: {chosen['evidence']}\n\n{chosen['explanation']}\n\nDetection IDs: {', '.join(chosen['detection_ids']) or 'none'}"
                    if chosen
                    else "No finding to decide.",
                    note: note_update,
                    view_key: context,
                    displayed_key: context,
                    expected_number: number,
                    decision_token: str(uuid4()),
                    history: [
                        [d["decision_number"], d["action"], d["note"] or "", d["created_at"]]
                        for d in decisions
                        if chosen and d["finding_id"] == chosen["finding_id"]
                    ],
                    metadata: bundle,
                    accept: gr.update(interactive=bool(chosen)),
                    reject: gr.update(interactive=bool(chosen)),
                    defer: gr.update(interactive=bool(chosen)),
                    decision_message: "",
                    export_file: None,
                }
            except (ValueError, OSError, Timeout) as exc:
                return empty_view(f"Evidence unavailable: {exc}")

        def load_session(selected):
            if not selected:
                return {
                    session_info: "Choose or upload a session.",
                    original: None,
                    frame_selector: gr.update(choices=[], value=None, interactive=False),
                    start: gr.update(interactive=False),
                    export_button: gr.update(interactive=False),
                    export_file: None,
                    export_message: "",
                }
            try:
                summary = session_summary(engine, UUID(selected))
                frames = list_frames(engine, UUID(selected))
                choices = [
                    (
                        f"{f['timestamp_ms'] / 1000:.3f}s | {f['findings']} findings | {f['frame_id']}",
                        f["frame_id"],
                    )
                    for f in frames
                ]
                text = f"{summary['state']} | {summary['video']['duration_us'] / 1_000_000:.2f}s | {len(frames)} sampled frames\nSession {selected}"
                if summary["last_error"]:
                    text += f"\n{summary['last_error']}"
                return {
                    session_info: text,
                    original: str(
                        safe_path(
                            session_dir(settings.data_root, UUID(selected)),
                            summary["video"]["original_path"],
                        )
                    ),
                    frame_selector: gr.update(
                        choices=choices,
                        value=choices[0][1] if choices else None,
                        interactive=bool(choices),
                    ),
                    start: gr.update(
                        interactive=summary["state"] not in ("processing", "review_ready")
                        and not analysis_running(engine)
                    ),
                    export_button: gr.update(interactive=summary["state"] != "processing"),
                    export_file: None,
                    export_message: "",
                }
            except ValueError as exc:
                raise gr.Error(str(exc)) from exc

        session_outputs = [
            session_info,
            original,
            frame_selector,
            start,
            export_button,
            export_file,
            export_message,
        ]

        def save_video(path):
            if not path:
                raise gr.Error("Choose an MP4 or MOV file first.")
            try:
                session_id = create_upload(engine, settings, Path(path))
                return gr.update(
                    choices=session_choices(), value=str(session_id)
                ), "Original saved. Analysis has not started."
            except (ValueError, OSError) as exc:
                raise gr.Error(str(exc)) from exc

        def refresh_sessions(selected):
            choices = session_choices()
            value = selected if selected in {v for _, v in choices} else None
            return gr.update(choices=choices, value=value)

        def start_analysis(selected, retry_confirmed):
            if not selected:
                yield "Choose a saved session first.", gr.update(interactive=False)
                return
            try:
                for message in run_analysis(
                    engine,
                    settings,
                    UUID(selected),
                    detector=detector,
                    reviewer=reviewer,
                    retry_after_restart=bool(retry_confirmed),
                ):
                    running = not any(
                        message.startswith(f"{state}:")
                        for state in ("review_ready", "partial", "cancelled", "failed")
                    )
                    yield f"Session {selected[:8]}: {message}", gr.update(interactive=running)
            except (ValueError, OSError) as exc:
                yield f"Analysis stopped: {exc}", gr.update(interactive=False)

        def mark_start(selected, retry_confirmed):
            if not selected:
                raise gr.Error("Choose a saved session first.")
            return (
                gr.update(interactive=False),
                gr.update(interactive=False),
                selected,
                bool(retry_confirmed),
                False,
            )

        def stop_analysis(selected):
            if selected and request_cancel(engine, UUID(selected)):
                return "Stop requested. The current step may finish before processing stops."
            return "This session is not currently processing."

        def choose_finding(key, finding_id):
            if not key:
                return empty_view()
            output = load_frame(
                key["session_id"], key["frame_id"], current_key=key, preferred=finding_id
            )
            # Preserve a newer browser selection while an older callback finishes.
            output[finding_selector] = gr.skip()
            return output

        def decide(action, key, finding_id, number, token, user_note, selected, selected_frame):
            if not key or not finding_id:
                raise gr.Error("Load a finding before recording a decision.")
            if key["session_id"] != selected or key["frame_id"] != selected_frame:
                raise gr.Error("Wait for the selected frame to load before deciding.")
            if key.get("finding_id") != finding_id:
                raise gr.Error("Wait for the selected finding to load before deciding.")
            try:
                decision = record_decision(
                    engine,
                    settings.data_root,
                    FrameKey.model_validate(
                        {name: key[name] for name in ("session_id", "frame_id", "timestamp_ms")}
                    ),
                    UUID(finding_id),
                    action,
                    user_note or "",
                    int(number),
                    UUID(token),
                )
                output = load_frame(selected, selected_frame, current_key=key, preferred=finding_id)
                output[note] = gr.update(value="", interactive=True)
                output[decision_message] = (
                    f"{action.capitalize()} recorded as revision {decision['decision_number']}. Predictions are unchanged."
                )
                if decision.get("snapshot_pending"):
                    output[decision_message] += (
                        " JSON publication is pending; export will reconcile it."
                    )
                return output
            except (ValueError, OSError, Timeout) as exc:
                raise gr.Error(str(exc)) from exc

        def create_zip(selected):
            if not selected:
                raise gr.Error("Choose a session first.")
            try:
                path = export_session(engine, settings.data_root, UUID(selected))
                return (
                    str(path),
                    "ZIP created and verified. Partial-session limitations are recorded in its manifest.",
                )
            except (ValueError, OSError, Timeout) as exc:
                raise gr.Error(str(exc)) from exc

        def workspace_status():
            detector_status = (
                "YOLO ready" if settings.weights_path.is_file() else "YOLO checkpoint missing"
            )
            identity = OllamaReviewer(settings.ollama_base_url, settings.ollama_model).identity()
            qwen_status = (
                "Qwen ready"
                if identity.get("model_digest")
                else "Qwen unavailable; manual review remains supported"
            )
            cadence = (
                "YOLO and Qwen on every 5th source frame | regional cup review"
                if settings.frame_stride is not None
                else f"{settings.sample_hz:g} frame/s | up to {settings.vlm_budget} Qwen pairs/video"
            )
            return f"{detector_status} | {qwen_status} | {cadence}"

        save_event = save.click(
            save_video,
            inputs=upload,
            outputs=[sessions, upload_message],
            api_visibility="private",
            **view_queue,
        )
        save_event.then(load_session, inputs=sessions, outputs=session_outputs, **view_queue).then(
            load_frame,
            inputs=[sessions, frame_selector, view_key],
            outputs=view_outputs,
            **view_queue,
        )
        sessions.input(
            load_session,
            inputs=sessions,
            outputs=session_outputs,
            trigger_mode="always_last",
            **view_queue,
        ).then(
            load_frame,
            inputs=[sessions, frame_selector, view_key],
            outputs=view_outputs,
            **view_queue,
        )
        refresh.click(refresh_sessions, inputs=sessions, outputs=sessions, **view_queue).then(
            load_session, inputs=sessions, outputs=session_outputs, **view_queue
        ).then(
            load_frame,
            inputs=[sessions, frame_selector, view_key],
            outputs=view_outputs,
            **view_queue,
        )
        frame_selector.input(
            load_frame,
            inputs=[sessions, frame_selector, view_key],
            outputs=view_outputs,
            trigger_mode="always_last",
            **view_queue,
        )
        finding_selector.input(
            choose_finding,
            inputs=[view_key, finding_selector],
            outputs=view_outputs,
            trigger_mode="always_last",
            **view_queue,
        )
        start_event = start.click(
            mark_start,
            inputs=[sessions, retry],
            outputs=[start, cancel, active_job, recovery_ack, retry],
            queue=False,
        ).success(
            start_analysis,
            inputs=[active_job, recovery_ack],
            outputs=[progress, cancel],
            concurrency_id="analysis",
            concurrency_limit=1,
            api_visibility="private",
        )
        start_event.then(lambda: gr.update(interactive=False), outputs=cancel).then(
            refresh_sessions, inputs=sessions, outputs=sessions, **view_queue
        ).then(load_session, inputs=sessions, outputs=session_outputs, **view_queue).then(
            load_frame,
            inputs=[sessions, frame_selector, view_key],
            outputs=view_outputs,
            **view_queue,
        )
        cancel.click(
            stop_analysis,
            inputs=active_job,
            outputs=progress,
            queue=False,
            api_visibility="private",
        )
        for button, action in [(accept, "accept"), (reject, "reject"), (defer, "defer")]:

            def handle(
                key, finding_id, number, token, user_note, selected, selected_frame, action=action
            ):
                return decide(
                    action, key, finding_id, number, token, user_note, selected, selected_frame
                )

            button.click(
                handle,
                inputs=[
                    displayed_key,
                    finding_selector,
                    expected_number,
                    decision_token,
                    note,
                    sessions,
                    frame_selector,
                ],
                outputs=view_outputs,
                **view_queue,
                api_visibility="private",
            )
        export_button.click(
            create_zip,
            inputs=sessions,
            outputs=[export_file, export_message],
            concurrency_id="export",
            concurrency_limit=1,
            api_visibility="private",
        )
        app.load(workspace_status, outputs=system_status)
        app.load(refresh_sessions, inputs=sessions, outputs=sessions)
    return app


def launch(settings: Settings, engine, *, detector=None, reviewer=None, port=None):
    app = build_app(settings, engine, detector=detector, reviewer=reviewer)
    return app.launch(
        server_name="127.0.0.1",
        server_port=port or settings.server_port,
        share=False,
        allowed_paths=[str(settings.data_root)],
        blocked_paths=[
            str(settings.data_root / ".locks"),
            str(settings.data_root / "models"),
            str(settings.data_root / ".app.lock"),
            str(settings.data_root / ".analysis.lock"),
        ],
        max_file_size="200mb",
        enable_monitoring=False,
        show_error=False,
        theme=THEME,
        css=CSS,
        footer_links=[],
    )


def main():
    settings = Settings()
    settings.data_root.mkdir(parents=True, exist_ok=True)
    engine = create_engine_for(settings)
    require_schema(engine)
    try:
        with FileLock(settings.data_root / ".app.lock", timeout=0):
            recover_interrupted(engine)
            launch(settings, engine)
    except Timeout as exc:
        raise SystemExit("Another auditor instance already owns this data folder.") from exc
    finally:
        engine.dispose()


if __name__ == "__main__":
    main()
