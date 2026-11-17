import json

import httpx
import pytest

from auditor import datasets, localization
from auditor.config import Settings
from tests.test_datasets import source_zip


def setup_frame(tmp_path):
    settings = Settings(database_url="postgresql+psycopg://unused", data_root=tmp_path / "d")
    draft = datasets.build_draft(settings.data_root, [source_zip(tmp_path)], enabled=True)
    frame = draft["frames"][0]
    # A controlled native crop stands in for a saved Qwen missed-cup lead.
    frame["source_regions"] = [[20, 10, 100, 90]]
    datasets._write_draft(datasets.dataset_dir(settings.data_root, draft["dataset_id"]), draft)
    return settings, draft, frame


def test_localization_off_never_contacts_model(tmp_path):
    def unexpected(request):
        raise AssertionError("Model contacted while OFF")

    assert (
        localization.propose_frame(
            Settings(database_url="postgresql+psycopg://unused", data_root=tmp_path),
            "invalid",
            "invalid",
            enabled=False,
            transport=httpx.MockTransport(unexpected),
        )
        is None
    )
    assert not (tmp_path / "datasets").exists()


def test_valid_boxes_map_to_native_pixels_without_becoming_approved(tmp_path):
    settings, draft, frame = setup_frame(tmp_path)
    requests = []

    def respond(request):
        requests.append(json.loads(request.content))
        return httpx.Response(
            200,
            json={
                "done": True,
                "message": {
                    "content": json.dumps(
                        {
                            "units": "normalized_1000",
                            "boxes": [[0, 0, 500, 1000], [500, 0, 1000, 500]],
                        }
                    )
                },
            },
        )

    result = localization.propose_frame(
        settings,
        draft["dataset_id"],
        frame["token"],
        enabled=True,
        transport=httpx.MockTransport(respond),
    )
    saved = result["frames"][0]
    qwen = [box for box in saved["boxes"] if box["origin"] == "qwen"]
    assert [box["bbox_xyxy"] for box in qwen] == [[20, 10, 60, 90], [60, 10, 100, 50]]
    assert saved["status"] == "pending"
    receipt = saved["localization_receipts"][0]
    assert receipt["status"] == "validated"
    assert receipt["native_xyxy"] == [20, 10, 100, 90]
    assert receipt["raw_response"]
    assert requests[0]["model"] == settings.ollama_model
    assert requests[0]["options"]["temperature"] == 0


@pytest.mark.parametrize(
    "response",
    [
        {"units": "pixels", "boxes": [[0, 0, 20, 20]]},
        {"units": "normalized_1000", "boxes": [[200, 0, 100, 20]]},
        {"units": "normalized_1000", "boxes": [[0, 0, 1001, 1000]]},
        {"units": "normalized_1000", "boxes": [[0, 0, 10, 10]] * 17},
    ],
)
def test_invalid_localization_is_preserved_without_labels(tmp_path, response):
    settings, draft, frame = setup_frame(tmp_path)
    transport = httpx.MockTransport(
        lambda request: httpx.Response(
            200,
            json={
                "done": True,
                "message": {"content": json.dumps(response)},
            },
        )
    )
    saved = localization.propose_frame(
        settings,
        draft["dataset_id"],
        frame["token"],
        enabled=True,
        transport=transport,
    )["frames"][0]
    assert len(saved["boxes"]) == 1  # Original YOLO proposal only.
    assert saved["localization_receipts"][0]["status"] == "invalid_json"


def test_timeout_stops_later_calls_and_persists_uncertainty(tmp_path):
    settings, draft, frame = setup_frame(tmp_path)
    calls = []

    def timeout(request):
        calls.append(request)
        raise httpx.ReadTimeout("synthetic timeout")

    for _ in range(2):
        result = localization.propose_frame(
            settings,
            draft["dataset_id"],
            frame["token"],
            enabled=True,
            transport=httpx.MockTransport(timeout),
        )
    assert len(calls) == 1
    assert result["localization_uncertain"] is True
    assert result["frames"][0]["localization_receipts"][0]["status"] == "timeout"
    assert (settings.data_root / ".dataset-qwen-pause.json").exists()
    from uuid import uuid4

    from auditor.pipeline import run_analysis

    # The pause is checked before touching the supplied database engine.
    with pytest.raises(ValueError, match="dataset Qwen request"):
        list(run_analysis(None, settings, uuid4()))
    with pytest.raises(ValueError, match="Confirm"):
        localization.clear_model_pause(settings, False)
    localization.clear_model_pause(settings, True)
    assert not (settings.data_root / ".dataset-qwen-pause.json").exists()
    assert datasets.load_draft(settings.data_root)["localization_uncertain"] is False
