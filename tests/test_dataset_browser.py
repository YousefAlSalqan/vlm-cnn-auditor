"""File-backed browser workflow. No PostgreSQL, Ollama, or real training."""

import os
import re
import socket

import gradio as gr
import pytest
from playwright.sync_api import expect, sync_playwright

from auditor import dataset_ui, datasets
from auditor.app import CSS, THEME
from auditor.config import Settings
from auditor.training import TrainingWorker
from tests.test_datasets import source_zip


@pytest.mark.browser
def test_local_dataset_review_on_desktop_and_mobile(tmp_path, monkeypatch):
    if (
        os.environ.get("AUDITOR_DATASET_BROWSER_TESTS") != "1"
        and os.environ.get("AUDITOR_BROWSER_TESTS") != "1"
    ):
        pytest.skip("Set AUDITOR_DATASET_BROWSER_TESTS=1 to run the file-only browser test")
    paths = [source_zip(tmp_path / "sources", name) for name in ("one", "two")]
    weights = tmp_path / "existing.pt"
    weights.write_bytes(b"synthetic weights, never loaded")
    root = tmp_path / "d"
    root.mkdir()
    settings = Settings(
        _env_file=None,
        database_url="postgresql+psycopg://unused",
        data_root=root,
        yolo_weights=weights,
    )
    worker = TrainingWorker(root, runner=lambda *args: {"test_only": True, "activated": False})
    monkeypatch.setattr(dataset_ui, "TrainingWorker", lambda root: worker)
    with gr.Blocks(analytics_enabled=False) as app:
        session = gr.Dropdown(label="Saved session", choices=[])
        dataset_ui.build_dataset_panel(
            settings, session, lambda selected: pytest.fail("Database export requested")
        )
    with socket.socket() as available:
        available.bind(("127.0.0.1", 0))
        port = available.getsockname()[1]
    try:
        app.launch(
            server_name="127.0.0.1",
            server_port=port,
            share=False,
            prevent_thread_lock=True,
            allowed_paths=[str(root)],
            theme=THEME,
            css=CSS,
            quiet=True,
            footer_links=[],
        )
        with sync_playwright() as playwright:
            browser = playwright.chromium.launch(headless=True)
            context = browser.new_context(viewport={"width": 1280, "height": 1000}, has_touch=True)
            page = context.new_page()
            errors = []
            failed = []
            page.on("pageerror", lambda error: errors.append(str(error)))
            page.on(
                "response",
                lambda response: (
                    failed.append(f"{response.status} {response.url}")
                    if response.status >= 400
                    else None
                ),
            )
            try:
                page.goto(f"http://127.0.0.1:{port}", wait_until="domcontentloaded")
                page.get_by_role("button", name="4. Build a reviewed cup dataset").click()
                master = page.get_by_role(
                    "checkbox", name="Enable dataset building and fine-tuning", exact=True
                )
                auto = page.get_by_role(
                    "checkbox", name="Automatically propose missing boxes", exact=True
                )
                expect(master).not_to_be_checked()
                expect(auto).to_be_checked()
                expect(auto).to_be_disabled()
                assert not (root / "datasets").exists()
                master.check()
                auto.uncheck()
                page.locator('input[type="file"]').set_input_files([str(p) for p in paths])
                expect(page.get_by_role("cell", name=paths[1].name, exact=True)).to_be_visible()
                page.get_by_role("button", name="Build or extend dataset", exact=True).click()
                expect(page.get_by_label("Dataset action", exact=True)).to_have_value(
                    re.compile("Draft ready"), timeout=30000
                )
                expect(page.locator("#dataset-raw img")).to_be_visible()

                def check_corner_mapping():
                    image = page.locator("#dataset-raw img")
                    image.scroll_into_view_if_needed()
                    geometry = image.evaluate("""e => {
                        const b=e.getBoundingClientRect();
                        const s=Math.min(b.width/e.naturalWidth,b.height/e.naturalHeight,1);
                        return {x:b.x+(b.width-e.naturalWidth*s)/2,y:b.y+(b.height-e.naturalHeight*s)/2,s};
                    }""")
                    for x, y in ((20, 30), (60, 80)):
                        page.mouse.click(
                            geometry["x"] + x * geometry["s"], geometry["y"] + y * geometry["s"]
                        )
                        if x == 20:
                            expect(page.get_by_label("Top y", exact=True)).to_have_value("30")
                    for label, value in (
                        ("Left x", 20),
                        ("Top y", 30),
                        ("Right x", 60),
                        ("Bottom y", 80),
                    ):
                        expect(page.get_by_label(label, exact=True)).to_have_value(
                            str(value), timeout=10000
                        )

                check_corner_mapping()
                page.get_by_role("button", name="Add box", exact=True).click()
                expect(page.get_by_label("Dataset action", exact=True)).to_have_value(
                    re.compile("Box edit saved")
                )
                boxes = page.get_by_role("combobox", name="Cup box", exact=True)
                boxes.click()
                page.get_by_role("option").filter(has_text="2. human").click()
                expect(page.get_by_label("Left x", exact=True)).to_have_value("20")
                page.get_by_label("Left x", exact=True).fill("21")
                page.get_by_role("button", name="Update selected box", exact=True).click()
                expect(page.get_by_label("Dataset readiness", exact=True)).to_have_value(
                    re.compile("Revision 3")
                )
                boxes.click()
                page.get_by_role("option").filter(has_text="2. human").click()
                page.get_by_role("button", name="Delete selected box", exact=True).click()
                expect(page.get_by_label("Dataset readiness", exact=True)).to_have_value(
                    re.compile("Revision 4")
                )
                page.get_by_role("checkbox", name="I checked the entire frame", exact=True).check()
                page.get_by_role("button", name="Save frame review", exact=True).click()
                expect(page.get_by_label("Dataset action", exact=True)).to_have_value(
                    re.compile("Frame review saved")
                )
                page.get_by_role("combobox", name="Dataset frame", exact=True).click()
                page.get_by_role("option").filter(has_text="2. frame_").click()
                expect(page.get_by_label("Dataset frame identity", exact=True)).to_have_value(
                    re.compile("pending")
                )
                page.get_by_role("checkbox", name="I checked the entire frame", exact=True).check()
                page.get_by_role("button", name="Save frame review", exact=True).click()
                expect(page.get_by_label("Dataset readiness", exact=True)).to_have_value(
                    re.compile("Ready for approval")
                )
                page.get_by_role("button", name="Approve dataset and fine-tune", exact=True).click()
                expect(page.get_by_label("Dataset action", exact=True)).to_have_value(
                    re.compile("Approved release")
                )
                worker.thread.join(5)
                assert worker.status()["state"] == "completed"
                page.get_by_role("button", name="Refresh training status", exact=True).click()
                expect(page.get_by_text('"completed"', exact=True)).to_be_visible()
                page.set_viewport_size({"width": 375, "height": 900})
                check_corner_mapping()
                assert page.evaluate("document.documentElement.scrollWidth <= window.innerWidth")
                assert not errors
                assert not failed
                manifest = datasets.load_draft(root)["approval"]
                directory = datasets.dataset_dir(root, datasets.load_draft(root)["dataset_id"])
                datasets.verify_release(directory / manifest["release"])
                page.screenshot(path=str(tmp_path / "dataset-mobile.png"), full_page=True)
            except Exception:
                page.screenshot(path=str(tmp_path / "dataset-failure.png"), full_page=True)
                raise
            finally:
                context.close()
                browser.close()
    finally:
        worker.close()
        app.close()
