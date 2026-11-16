"""Independent synthetic checks for the local model HTTP trust boundary."""

import json
import threading
import time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from uuid import uuid4

import httpx
import pytest

from auditor.contracts import FrameKey
from auditor.models import OllamaReviewer
from tests.test_models import jpeg


def key_and_envelope():
    key = FrameKey(session_id=uuid4(), frame_id="frame_000000000", timestamp_ms=0)
    content = key.model_dump(mode="json") | dict(
        outcome="no_issue", estimated_count=0, needs_review=False, findings=[]
    )
    body = json.dumps({"done": True, "message": {"content": json.dumps(content)}}).encode()
    return key, body


@pytest.mark.parametrize("envelope", [[], None, 1, "invalid top-level JSON scalar"])
def test_valid_json_with_invalid_api_shape_stays_a_diagnostic_attempt(envelope):
    key, _ = key_and_envelope()
    reviewer = OllamaReviewer(
        "http://127.0.0.1:11435",
        transport=httpx.MockTransport(
            lambda request: httpx.Response(200, content=json.dumps(envelope))
        ),
    )
    result = reviewer.review(key, jpeg((0, 0, 0)), jpeg((255, 255, 255)), [])
    assert result.status == "invalid_json"
    assert result.error_code == "invalid_api_envelope"
    assert result.raw_response


def local_server(body, *, delay=0):
    requests = []

    class Handler(BaseHTTPRequestHandler):
        def log_message(self, *args):
            pass

        def do_POST(self):
            requests.append(self.rfile.read(int(self.headers["Content-Length"])))
            self.send_response(200)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            try:
                for start in range(0, len(body), 32):
                    if delay:
                        time.sleep(delay)
                    self.wfile.write(body[start : start + 32])
                    self.wfile.flush()
            except (ConnectionError, OSError):
                pass

    server = ThreadingHTTPServer(("127.0.0.1", 0), Handler)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    return server, thread, requests


def test_request_timeout_is_an_absolute_wall_deadline(monkeypatch):
    key, body = key_and_envelope()
    server, thread, _ = local_server(body, delay=0.035)
    try:
        monkeypatch.setenv("NO_PROXY", "127.0.0.1,localhost")
        reviewer = OllamaReviewer(f"http://127.0.0.1:{server.server_port}", timeout=0.15)
        result = reviewer.review(key, jpeg((0, 0, 0)), jpeg((255, 255, 255)), [])
        assert result.status == "timeout", (
            f"A {result.wall_ms} ms response passed a 150 ms request deadline"
        )
    finally:
        server.shutdown()
        server.server_close()
        thread.join(timeout=2)


def test_local_footage_is_not_sent_through_environment_http_proxy(monkeypatch):
    key, body = key_and_envelope()
    server, thread, requests = local_server(body)
    try:
        proxy = f"http://127.0.0.1:{server.server_port}"
        monkeypatch.setenv("HTTP_PROXY", proxy)
        monkeypatch.setenv("http_proxy", proxy)
        monkeypatch.setenv("ALL_PROXY", proxy)
        monkeypatch.setenv("all_proxy", proxy)
        monkeypatch.setenv("NO_PROXY", "")
        monkeypatch.setenv("no_proxy", "")
        reviewer = OllamaReviewer("http://127.0.0.1:1", timeout=0.2)
        reviewer.review(key, jpeg((0, 0, 0)), jpeg((255, 255, 255)), [])
        assert requests == [], "The environment proxy received the original/overlay image payload"
    finally:
        server.shutdown()
        server.server_close()
        thread.join(timeout=2)
