import os
import re
import subprocess
from uuid import uuid4

import pytest
from alembic import command
from alembic.config import Config
from sqlalchemy import create_engine, text
from sqlalchemy.engine import make_url

from auditor.config import Settings


@pytest.fixture
def engine():
    url = os.environ.get("AUDITOR_TEST_DATABASE_URL")
    if not url:
        pytest.skip("Set AUDITOR_TEST_DATABASE_URL to a disposable PostgreSQL test database")
    if not (make_url(url).database or "").startswith("auditor_test"):
        raise RuntimeError("Tests require a separate database named auditor_test*")
    schema = f"test_{uuid4().hex}"
    assert re.fullmatch(r"test_[a-f0-9]{32}", schema)
    admin = create_engine(url)
    with admin.begin() as connection:
        connection.execute(text(f'CREATE SCHEMA "{schema}"'))
    test_engine = create_engine(url, connect_args={"options": f"-csearch_path={schema}"})
    try:
        config = Config("alembic.ini")
        with test_engine.begin() as connection:
            config.attributes["connection"] = connection
            command.upgrade(config, "head")
        yield test_engine
    finally:
        test_engine.dispose()
        with admin.begin() as connection:
            connection.execute(text(f'DROP SCHEMA "{schema}" CASCADE'))
        admin.dispose()


@pytest.fixture
def app_settings(tmp_path, engine):
    return Settings(
        _env_file=None,
        database_url=engine.url.render_as_string(hide_password=False),
        data_root=tmp_path / "data",
        vlm_budget=12,
    )


@pytest.fixture
def small_clip(tmp_path):
    path = tmp_path / "clip.mp4"
    subprocess.run(
        [
            "ffmpeg",
            "-v",
            "error",
            "-f",
            "lavfi",
            "-i",
            "color=c=gray:s=160x120:r=2:d=15",
            "-c:v",
            "libx264",
            "-preset",
            "ultrafast",
            str(path),
        ],
        check=True,
    )
    return path
