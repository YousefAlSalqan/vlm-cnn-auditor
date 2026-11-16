from pathlib import Path

import pytest

from auditor.storage import atomic_write, safe_path, sha256_file


def test_paths_cannot_escape_the_root(tmp_path):
    for path in ["../secret", "/absolute", "C:/Windows/system.ini", "frames/../../secret"]:
        with pytest.raises(ValueError):
            safe_path(tmp_path, path)
    assert (
        safe_path(tmp_path, "frames/frame_000000000.jpg") == tmp_path / "frames/frame_000000000.jpg"
    )


def test_immutable_write_verifies_existing_content(tmp_path):
    path = tmp_path / "original.mp4"
    atomic_write(path, b"original", immutable=True)
    original_hash = sha256_file(path)
    atomic_write(path, b"original", immutable=True)
    with pytest.raises(ValueError):
        atomic_write(path, b"replacement", immutable=True)
    assert path.read_bytes() == b"original"
    assert sha256_file(path) == original_hash
    assert list(Path(tmp_path).glob("*.tmp")) == []


def test_atomic_write_failure_does_not_replace_good_file(tmp_path, monkeypatch):
    import os

    path = tmp_path / "frame.json"
    atomic_write(path, b"first")

    def fail_replace(*_):
        raise OSError("disk unavailable")

    monkeypatch.setattr(os, "replace", fail_replace)
    with pytest.raises(OSError):
        atomic_write(path, b"second")
    assert path.read_bytes() == b"first"
