from __future__ import annotations

import json
import subprocess
import tempfile
from fractions import Fraction
from pathlib import Path
from uuid import UUID

import cv2

from auditor.config import Settings
from auditor.contracts import AuditorError, FrameKey, SampledFrame, VideoProbe


def validate_and_probe(path: Path, settings: Settings) -> VideoProbe:
    path = Path(path)
    if not path.is_file() or path.suffix.lower() not in (".mp4", ".mov"):
        raise AuditorError("Choose a local MP4 or MOV video")
    if not 0 < path.stat().st_size <= 200 * 1024**2:
        raise AuditorError("Video must be nonempty and no larger than 200 MiB")
    entries = (
        "format=duration,format_name:format_tags=major_brand:"
        "stream=index,codec_name,width,height,time_base,duration,color_transfer:"
        "stream_tags=rotate:stream_side_data=rotation:frame=pts,best_effort_timestamp"
    )
    try:
        with tempfile.TemporaryFile() as output:
            process = subprocess.run(
                [
                    settings.ffprobe_binary,
                    "-v",
                    "error",
                    "-select_streams",
                    "v:0",
                    "-show_frames",
                    "-show_entries",
                    entries,
                    "-of",
                    "json",
                    str(path),
                ],
                stdout=output,
                stderr=subprocess.PIPE,
                timeout=45,
                check=False,
            )
            if process.returncode:
                raise AuditorError(
                    "The video could not be decoded by FFmpeg. Try an H.264 SDR clip."
                )
            output.seek(0)
            raw = output.read(8 * 1024**2 + 1)
        if len(raw) > 8 * 1024**2:
            raise AuditorError("Video timestamp metadata exceeds the safety limit")
        data = json.loads(raw)
        formats = set(data.get("format", {}).get("format_name", "").split(","))
        if not formats.intersection({"mov", "mp4"}):
            raise AuditorError("The file contents must be an MP4 or MOV container")
        streams = data.get("streams", [])
        if len(streams) != 1:
            raise AuditorError("No supported video stream was found")
        stream = streams[0]
        duration = Fraction(str(stream.get("duration") or data.get("format", {}).get("duration")))
        if not 15 <= duration <= 30:
            raise AuditorError("Use a video between 15 and 30 seconds long")
        width, height = int(stream["width"]), int(stream["height"])
        if min(width, height) < 1 or max(width, height) > 1920 or width * height > 2073600:
            raise AuditorError("Use a video up to 1080p, with a maximum long edge of 1920 pixels")
        if stream.get("color_transfer") in ("smpte2084", "arib-std-b67"):
            raise AuditorError(
                "HDR footage is not supported yet. Export an SDR copy and keep the original."
            )
        time_base = Fraction(stream["time_base"])
        if time_base <= 0:
            raise AuditorError("Video presentation time base must be positive")
        frames = data.get("frames", [])
        if not 1 <= len(frames) <= 3600:
            raise AuditorError("Video must contain between 1 and 3600 decoded frames")
        timestamps = []
        sources = set()
        for frame in frames:
            name = "pts" if "pts" in frame else "best_effort_timestamp"
            if name not in frame:
                raise AuditorError("A source frame has no usable presentation timestamp")
            timestamps.append(int(frame[name]))
            sources.add(name)
        if any(b < a for a, b in zip(timestamps, timestamps[1:], strict=False)):
            raise AuditorError(
                "Presentation timestamps decrease. This codec/edit list is unsupported."
            )
        if (timestamps[-1] - timestamps[0]) * time_base > duration:
            raise AuditorError("Video presentation timestamps exceed the declared duration")
        rotation = stream.get("tags", {}).get("rotate", 0)
        for side_data in stream.get("side_data_list", []):
            if "rotation" in side_data:
                rotation = side_data["rotation"]
        rotation = float(rotation)
        if not rotation.is_integer() or int(rotation) % 90:
            raise AuditorError("Only quarter-turn video rotation is supported")
        version = subprocess.run(
            [settings.ffprobe_binary, "-version"],
            capture_output=True,
            text=True,
            timeout=5,
            check=True,
        ).stdout.splitlines()[0]
        container = "mov" if path.suffix.lower() == ".mov" else "mp4"
        return VideoProbe(
            container=container,
            codec=stream["codec_name"],
            duration_us=int(duration * 1_000_000),
            width=width,
            height=height,
            stream_index=stream["index"],
            time_base_num=time_base.numerator,
            time_base_den=time_base.denominator,
            first_pts=timestamps[0],
            timestamps=timestamps,
            rotation_degrees=int(rotation) % 360,
            timestamp_source="+".join(sorted(sources)),
            probe_version=version,
        )
    except (OSError, subprocess.SubprocessError) as exc:
        raise AuditorError(
            "FFprobe is unavailable or timed out. Install FFmpeg and check this clip."
        ) from exc
    except AuditorError:
        raise
    except (KeyError, TypeError, ValueError, ZeroDivisionError) as exc:
        raise AuditorError("The video has incomplete or unsupported timestamp metadata") from exc


def iter_sampled_frames(
    path: Path,
    session_id: UUID,
    probe: VideoProbe,
    sample_hz: float | None = None,
    *,
    frame_stride: int | None = None,
):
    rate = Fraction(str(sample_hz)) if frame_stride is None else None
    if frame_stride is not None and (type(frame_stride) is not int or frame_stride != 5):
        raise AuditorError("Source-frame stride must be five")
    if rate is not None and not Fraction(1, 5) <= rate <= 2:
        raise AuditorError("Sampling must be between 0.2 and 2 frames per second")
    capture = cv2.VideoCapture(str(path), cv2.CAP_FFMPEG)
    try:
        if not capture.isOpened():
            raise AuditorError("OpenCV cannot decode the original video")
        orientation_disabled = capture.set(cv2.CAP_PROP_ORIENTATION_AUTO, 0)
        if probe.rotation_degrees and not orientation_disabled:
            raise AuditorError("Decoder cannot establish a reliable orientation transform")
        target_number = 0
        emitted = 0
        for index, pts in enumerate(probe.timestamps):
            ok, image = capture.read()
            if not ok:
                raise AuditorError("Decode count does not match the timestamp metadata")
            elapsed = Fraction((pts - probe.first_pts) * probe.time_base_num, probe.time_base_den)
            if frame_stride is not None:
                if index % frame_stride:
                    continue
            else:
                if elapsed < Fraction(target_number, 1) / rate:
                    continue
                target_number = int(elapsed * rate) + 1
            limit = 720 if frame_stride is not None else 60
            if emitted >= limit:
                raise AuditorError(f"Video exceeds the {limit} sampled-frame limit")
            emitted += 1
            rotations = {
                90: cv2.ROTATE_90_COUNTERCLOCKWISE,
                180: cv2.ROTATE_180,
                270: cv2.ROTATE_90_CLOCKWISE,
            }
            if probe.rotation_degrees:
                image = cv2.rotate(image, rotations[probe.rotation_degrees])
            ok, encoded = cv2.imencode(".jpg", image, [cv2.IMWRITE_JPEG_QUALITY, 95])
            if not ok:
                raise AuditorError("Could not encode a canonical raw JPEG")
            millis_numerator = (pts - probe.first_pts) * probe.time_base_num * 1000
            millis = (2 * millis_numerator + probe.time_base_den) // (2 * probe.time_base_den)
            yield SampledFrame(
                key=FrameKey(
                    session_id=session_id, frame_id=f"frame_{index:09d}", timestamp_ms=millis
                ),
                source_frame_index=index,
                source_pts=pts,
                width=image.shape[1],
                height=image.shape[0],
                raw_jpeg=encoded.tobytes(),
            )
        if capture.read()[0]:
            raise AuditorError("Decode count exceeds the available timestamp metadata")
    finally:
        capture.release()
