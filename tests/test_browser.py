"""Portable browser smoke test, with synthetic models and an isolated PostgreSQL schema."""

import os
import re
import socket
import threading
import time

import pytest
from playwright.sync_api import expect, sync_playwright
from sqlalchemy import func, select
from sqlalchemy.orm import Session

import auditor.app as app_module
from auditor.app import CSS, THEME, build_app
from auditor.db import Decision
from auditor.export import verify_export
from tests.fakes import FakeDetector, TwoFindingReviewer


@pytest.mark.browser
@pytest.mark.integration
def test_upload_analysis_review_and_export_on_desktop_and_mobile(
    engine, app_settings, small_clip, tmp_path, monkeypatch
):
    if os.environ.get("AUDITOR_BROWSER_TESTS") != "1":
        pytest.skip("Set AUDITOR_BROWSER_TESTS=1 and install Playwright Chromium")
    with socket.socket() as available:
        available.bind(("127.0.0.1", 0))
        port = available.getsockname()[1]
    delay = threading.Event()
    original_bundle = app_module.frame_bundle

    def delayed_bundle(*args, **kwargs):
        if delay.is_set():
            time.sleep(2)
        return original_bundle(*args, **kwargs)

    monkeypatch.setattr(app_module, "frame_bundle", delayed_bundle)
    app = build_app(app_settings, engine, detector=FakeDetector(), reviewer=TwoFindingReviewer())
    try:
        app.launch(
            server_name="127.0.0.1",
            server_port=port,
            share=False,
            prevent_thread_lock=True,
            allowed_paths=[str(app_settings.data_root)],
            theme=THEME,
            css=CSS,
            footer_links=[],
            quiet=True,
        )
        with sync_playwright() as playwright:
            browser = playwright.chromium.launch(headless=True)
            page = browser.new_page(viewport={"width": 1280, "height": 900}, accept_downloads=True)
            errors = []
            page.on("pageerror", lambda error: errors.append(str(error)))
            page.goto(f"http://127.0.0.1:{port}")
            page.locator('input[type="file"]').first.set_input_files(str(small_clip))
            page.get_by_role("button", name="Save video", exact=True).click()
            expect(page.get_by_label("Upload result", exact=True)).to_have_value(
                "Original saved. Analysis has not started."
            )
            expect(page.get_by_label("Frame identity", exact=True)).to_have_value(
                "No frame selected"
            )
            page.get_by_role("button", name=re.compile("^Recovery")).click()
            recovery = page.get_by_role(
                "checkbox", name="I restarted Qwen. Retry failed reviews.", exact=True
            )
            recovery.check()
            page.get_by_role("button", name="Start analysis", exact=True).click()
            expect(recovery).not_to_be_checked()
            expect(page.get_by_label("Session", exact=True)).to_have_value(
                re.compile("review_ready"), timeout=90000
            )
            expect(page.get_by_label("Frame identity", exact=True)).to_have_value(
                re.compile("frame_000000000")
            )
            expect(page.locator("#frame-pair img")).to_have_count(2)
            page.get_by_label("Your note (optional)", exact=True).fill("Stale R2 note")
            delay.set()
            finding = page.get_by_role("combobox", name="Finding", exact=True)
            finding.click()
            page.get_by_role("option").filter(has_text="Second synthetic finding").click()
            page.get_by_role("button", name="Accept", exact=True).click()
            expect(
                page.get_by_text(
                    "Wait for the selected finding to load before deciding.", exact=True
                )
            ).to_be_visible(timeout=30000)
            expect(page.get_by_label("Evidence and explanation", exact=True)).to_have_value(
                re.compile("^R1:"), timeout=30000
            )
            delay.clear()
            with Session(engine) as db:
                assert db.scalar(select(func.count()).select_from(Decision)) == 0
            page.locator(".toast-wrap button").last.click()
            page.get_by_label("Your note (optional)", exact=True).fill("Separate R1 judgment")
            page.get_by_role("button", name="Accept", exact=True).click()
            expect(page.get_by_label("Decision result", exact=True)).to_have_value(
                re.compile("Accept recorded as revision 1"), timeout=30000
            )
            finding.click()
            page.get_by_role("option").filter(has_text="Synthetic test evidence").click()
            expect(page.get_by_label("Evidence and explanation", exact=True)).to_have_value(
                re.compile("^R2:"), timeout=30000
            )
            for revision, action in enumerate(("Accept", "Reject", "Defer"), 1):
                page.get_by_label("Your note (optional)", exact=True).fill(
                    f"Synthetic {action} note"
                )
                page.get_by_role("button", name=action, exact=True).click()
                expect(page.get_by_label("Decision result", exact=True)).to_have_value(
                    re.compile(f"{action} recorded as revision {revision}")
                )
            page.get_by_role("button", name=re.compile("^Decision history")).click()
            expect(page.locator("#decision-history")).not_to_contain_text("Separate R1 judgment")
            for action in ("Accept", "Reject", "Defer"):
                expect(
                    page.locator("#decision-history")
                    .get_by_text(f"Synthetic {action} note", exact=True)
                    .last
                ).to_be_visible()
            page.set_viewport_size({"width": 375, "height": 812})
            expect(
                page.get_by_role("button", name="Create session ZIP", exact=True)
            ).to_be_visible()
            assert page.evaluate("document.documentElement.scrollWidth <= innerWidth")
            page.get_by_role("button", name="Create session ZIP", exact=True).click()
            expect(page.get_by_label("Export result", exact=True)).to_have_value(
                re.compile("verified"), timeout=30000
            )
            with page.expect_download() as download:
                page.locator('a[href*=".zip"]').first.click()
            archive = tmp_path / "browser-export.zip"
            download.value.save_as(archive)
            assert verify_export(archive)
            assert not errors
            browser.close()
    finally:
        app.close()
