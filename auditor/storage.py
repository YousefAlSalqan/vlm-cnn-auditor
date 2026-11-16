from __future__ import annotations

import hashlib
import json
import os
import tempfile
import threading
import weakref
from pathlib import Path, PureWindowsPath
from uuid import UUID

from filelock import FileLock

from auditor.contracts import AuditorError

_locks: weakref.WeakValueDictionary[str, FileLock] = weakref.WeakValueDictionary()
_lock_guard = threading.Lock()


def session_lock(root: Path, session_id: UUID) -> FileLock:
    path = root / ".locks" / f"{session_id}.lock"
    path.parent.mkdir(parents=True, exist_ok=True)
    with _lock_guard:
        lock = _locks.get(str(path))
        if lock is None:
            lock = FileLock(path, timeout=10)
            _locks[str(path)] = lock
        return lock


def session_dir(root: Path, session_id: UUID | str) -> Path:
    return safe_path(root, f"session_{UUID(str(session_id))}")


def safe_path(root: Path, relative: str) -> Path:
    relative_path = Path(relative)
    if (
        not relative
        or relative_path.is_absolute()
        or PureWindowsPath(relative).drive
        or ".." in relative_path.parts
        or "\\" in relative
    ):
        raise AuditorError("Artifact path must stay inside its session directory")
    root = root.resolve()
    result = (root / relative_path).resolve()
    if not result.is_relative_to(root):
        raise AuditorError("Artifact path escapes its session directory")
    return result


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def canonical_json(payload: dict) -> bytes:
    return json.dumps(
        payload, sort_keys=True, ensure_ascii=False, allow_nan=False, separators=(",", ":")
    ).encode("utf-8")


def atomic_write(path: Path, data: bytes, *, immutable: bool = False) -> str:
    digest = hashlib.sha256(data).hexdigest()
    path.parent.mkdir(parents=True, exist_ok=True)
    if immutable and path.exists():
        if sha256_file(path) != digest:
            raise AuditorError("Existing immutable evidence has different bytes")
        return digest
    temporary = None
    try:
        with tempfile.NamedTemporaryFile(
            dir=path.parent, prefix=f".{path.name}.", suffix=".tmp", delete=False
        ) as stream:
            temporary = Path(stream.name)
            stream.write(data)
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary, path)
        return digest
    finally:
        if temporary is not None:
            temporary.unlink(missing_ok=True)


def copy_original(source: Path, directory: Path, container: str, max_bytes: int = 200 * 1024**2):
    if container not in ("mp4", "mov"):
        raise AuditorError("Unsupported video container")
    destination = safe_path(directory, f"original.{container}")
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = None
    digest = hashlib.sha256()
    total = 0
    try:
        with (
            source.open("rb") as incoming,
            tempfile.NamedTemporaryFile(
                dir=directory, prefix=".original.", suffix=".tmp", delete=False
            ) as outgoing,
        ):
            temporary = Path(outgoing.name)
            for chunk in iter(lambda: incoming.read(1024 * 1024), b""):
                total += len(chunk)
                if total > max_bytes:
                    raise AuditorError("Video exceeds the 200 MiB upload limit")
                digest.update(chunk)
                outgoing.write(chunk)
            outgoing.flush()
            os.fsync(outgoing.fileno())
        if destination.exists():
            raise AuditorError("An original video already exists for this session")
        os.replace(temporary, destination)
        return destination.name, digest.hexdigest(), total
    finally:
        if temporary is not None:
            temporary.unlink(missing_ok=True)
