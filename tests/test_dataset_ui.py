from pathlib import Path

import gradio as gr
import pytest

from auditor import datasets
from auditor.config import Settings
from auditor.dataset_ui import DatasetWorkspace, build_dataset_panel, corner_coordinates
from tests.test_datasets import source_zip


def test_panel_defaults_off_without_reading_database_or_loading_models(tmp_path):
    settings = Settings(database_url="postgresql+psycopg://unused", data_root=tmp_path)
    with gr.Blocks() as app:
        sessions = gr.Dropdown(choices=[])
        hook, _, _ = build_dataset_panel(
            settings, sessions, lambda selected: pytest.fail("Read database")
        )
        assert hook("unknown session", True, "missing.pt")
    components = {c["props"].get("label"): c["props"] for c in app.get_config_file()["components"]}
    assert components["Enable dataset building and fine-tuning"]["value"] is False
    assert components["Automatically propose missing boxes"]["value"] is True
    assert components["Automatically propose missing boxes"]["interactive"] is False
    assert not (tmp_path / "datasets").exists()


def test_manual_edits_preserve_origins_require_review_and_reject_stale_frame(tmp_path):
    settings = Settings(database_url="postgresql+psycopg://unused", data_root=tmp_path / "d")
    workspace = DatasetWorkspace(settings, lambda selected: pytest.fail("Unexpected export"))
    with pytest.raises(ValueError, match="Enable"):
        workspace.build([tmp_path / "missing"], None, False)
    workspace.set_enabled(True)
    draft = workspace.build([source_zip(tmp_path)], None, False)
    frame = draft["frames"][0]
    stamp = dict(dataset_id=draft["dataset_id"], revision=draft["revision"], token=frame["token"])
    with pytest.raises(ValueError, match="display|frame"):
        workspace.edit(stamp, "different frame", None, [1, 1, 5, 5], "add")
    draft = workspace.edit(stamp, frame["token"], None, [40, 20, 70, 60], "add")
    frame = draft["frames"][0]
    assert [b["origin"] for b in frame["boxes"]] == ["yolo", "human"]
    assert frame["status"] == "pending"
    with pytest.raises(ValueError, match="changed|revision"):
        workspace.edit(stamp, frame["token"], None, [1, 1, 5, 5], "add")
    stamp["revision"] = draft["revision"]
    with pytest.raises(ValueError, match="entire"):
        workspace.mark(stamp, frame["token"], "complete", "train", False)
    draft = workspace.mark(stamp, frame["token"], "complete", "train", True)
    assert draft["frames"][0]["status"] == "complete"


def test_two_corner_editor_uses_native_pixel_coordinates():
    corner, coords = corner_coordinates([20, 30], 100, 100, None)
    assert coords == [20, 30, None, None]
    corner, coords = corner_coordinates([10, 90], 100, 100, corner)
    assert corner is None and coords == [10, 30, 20, 90]
    with pytest.raises(ValueError, match="outside"):
        corner_coordinates([101, 0], 100, 100, None)


def test_disabled_proposals_do_not_dispatch_and_disabling_blocks_edits(tmp_path, monkeypatch):
    settings = Settings(database_url="postgresql+psycopg://unused", data_root=tmp_path / "d")
    workspace = DatasetWorkspace(settings, lambda selected: Path("unused"))
    workspace.set_enabled(True)
    monkeypatch.setattr(
        "auditor.dataset_ui.propose_frame", lambda *a, **k: pytest.fail("Dispatched Qwen")
    )
    draft = workspace.build([source_zip(tmp_path)], None, False)
    workspace.set_enabled(False)
    frame = draft["frames"][0]
    stamp = dict(dataset_id=draft["dataset_id"], revision=draft["revision"], token=frame["token"])
    with pytest.raises(ValueError, match="Enable"):
        workspace.edit(stamp, frame["token"], None, [1, 1, 5, 5], "add")
    assert datasets.load_draft(settings.data_root)["revision"] == draft["revision"]


def test_normal_app_mounts_optional_panel_without_database_calls(tmp_path, monkeypatch):
    import auditor.app as app_module

    monkeypatch.setattr(app_module, "list_sessions", lambda engine: [])
    settings = Settings(
        _env_file=None, database_url="postgresql+psycopg://unused", data_root=tmp_path
    )
    app = app_module.build_app(settings, object())
    config = app.get_config_file()
    labels = [component["props"].get("label") for component in config["components"]]
    assert "Enable dataset building and fine-tuning" in labels
    assert "Raw frame" in labels and "YOLO overlay" in labels
    assert len([c for c in config["components"] if c["type"] == "image"]) == 4
