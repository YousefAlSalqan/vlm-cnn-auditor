import json
import subprocess
from uuid import uuid4

import pytest

from auditor.config import Settings
from auditor.contracts import AuditorError
from auditor.video import iter_sampled_frames, validate_and_probe


def settings(tmp_path):
    return Settings(
        _env_file=None,
        database_url="postgresql+psycopg://unused@localhost/unused",
        data_root=tmp_path,
    )


def test_cfr_sampling_preserves_exact_source_indices_and_timestamps(phone_clip, tmp_path):
    probe = validate_and_probe(phone_clip, settings(tmp_path))
    session_id = uuid4()
    frames = list(iter_sampled_frames(phone_clip, session_id, probe, 1.0))
    assert [f.source_frame_index for f in frames] == list(range(0, 450, 30))
    assert [f.key.timestamp_ms for f in frames] == list(range(0, 15000, 1000))
    assert frames[4].key.frame_id == "frame_000000120"
    assert all(f.key.session_id == session_id and f.raw_jpeg[:2] == b"\xff\xd8" for f in frames)


def test_stride_five_uses_source_indices_without_the_old_sixty_frame_cap(phone_clip, tmp_path):
    probe = validate_and_probe(phone_clip, settings(tmp_path))
    frames = list(iter_sampled_frames(phone_clip, uuid4(), probe, frame_stride=5))
    assert [f.source_frame_index for f in frames] == list(range(0, 450, 5))
    assert len(frames) == 90
    assert frames[1].key.timestamp_ms == 167


def test_new_session_defaults_align_both_models_on_stride_five(tmp_path):
    config = settings(tmp_path).analysis_config()
    assert config["frame_stride"] == 5
    assert config["prompt_version"] == "cup_regions_v1"
    assert "vlm_budget" not in config


def test_vfr_uses_presentation_time_instead_of_nominal_fps(tmp_path):
    path = tmp_path / "variable.mp4"
    subprocess.run(
        [
            "ffmpeg",
            "-v",
            "error",
            "-f",
            "lavfi",
            "-i",
            "testsrc2=size=160x120:rate=30:duration=16",
            "-vf",
            "select='not(mod(n,if(lt(t,7),30,15)))'",
            "-fps_mode",
            "vfr",
            "-c:v",
            "libx264",
            "-preset",
            "ultrafast",
            str(path),
        ],
        check=True,
    )
    probe = validate_and_probe(path, settings(tmp_path))
    frames = list(iter_sampled_frames(path, uuid4(), probe, 1.0))
    assert [f.source_frame_index for f in frames] == [
        0,
        1,
        2,
        3,
        4,
        5,
        6,
        7,
        9,
        11,
        13,
        15,
        17,
        19,
        21,
        23,
    ]
    assert [f.key.timestamp_ms for f in frames] == list(range(0, 16000, 1000))


def test_rotation_is_applied_once(phone_clip, tmp_path):
    rotated = tmp_path / "rotated.mov"
    subprocess.run(
        [
            "ffmpeg",
            "-v",
            "error",
            "-display_rotation",
            "90",
            "-i",
            str(phone_clip),
            "-c",
            "copy",
            str(rotated),
        ],
        check=True,
    )
    probe = validate_and_probe(rotated, settings(tmp_path))
    assert probe.rotation_degrees == 90
    first = next(iter_sampled_frames(rotated, uuid4(), probe, 1.0))
    assert (first.width, first.height) == (240, 320)


def test_probe_decode_count_mismatch_is_rejected(phone_clip, tmp_path):
    probe = validate_and_probe(phone_clip, settings(tmp_path))
    wrong = probe.model_copy(update={"timestamps": probe.timestamps[:-1]})
    with pytest.raises(ValueError, match="[Tt]imestamp|[Dd]ecode"):
        list(iter_sampled_frames(phone_clip, uuid4(), wrong, 1.0))


@pytest.mark.parametrize("invalid", ["container", "time_base", "timestamp_span"])
def test_probe_rejects_inconsistent_source_metadata(tmp_path, monkeypatch, invalid):
    path = tmp_path / "disguised.mp4"
    path.write_bytes(b"synthetic metadata fixture")
    data = {
        "format": {"duration": "15", "format_name": "mov,mp4,m4a,3gp,3g2,mj2"},
        "streams": [
            {
                "index": 0,
                "codec_name": "h264",
                "width": 160,
                "height": 120,
                "duration": "15",
                "time_base": "1/1000",
            }
        ],
        "frames": [{"pts": 0}, {"pts": 14900}],
    }
    if invalid == "container":
        data["format"]["format_name"] = "matroska,webm"
    elif invalid == "time_base":
        data["streams"][0]["time_base"] = "0/1"
    else:
        data["frames"][-1]["pts"] = 100000000

    def fake_probe(command, **kwargs):
        if "-version" in command:
            return subprocess.CompletedProcess(command, 0, stdout="ffprobe synthetic")
        kwargs["stdout"].write(json.dumps(data).encode())
        return subprocess.CompletedProcess(command, 0)

    monkeypatch.setattr(subprocess, "run", fake_probe)
    with pytest.raises(AuditorError):
        validate_and_probe(path, settings(tmp_path))
