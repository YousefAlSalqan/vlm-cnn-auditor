# Risks and implementation gates

> **Planning risk register.** This document retains the original implementation gates and sources. Current runtime behavior, recovery limits and experimental evidence are described in [ARCHITECTURE.md](ARCHITECTURE.md), [V1_GUIDE.md](V1_GUIDE.md) and [V1_VERIFICATION.md](V1_VERIFICATION.md). The two-image/full-schema approach below is historical. The new default uses raw regional views and a small observation schema.

This register distinguishes verified documentation from untested integration assumptions. Owners are proposed roles for one developer and the local operator. No risk entry authorizes deployment, model downloads, cloud media upload, or application implementation during planning.

| Risk | Consequence | Mitigation and verification gate | Owner |
|---|---|---|---|
| Qwen hallucinations and overlay anchoring | Plausible but false errors or corrections | Count from raw view first, preserve proposals separately, allow abstention, validate references, and adjudicate against independent human labels. Measure unsafe/accepted-proposal risk. No automatic corrections or retraining | Developer + annotator |
| Strict JSON mistaken for truth | A valid schema appears trustworthy despite wrong content | Pydantic plus frame-context validation protects structure only. UI calls results suggestions. Human labels determine accuracy | Developer |
| Ollama two-image/schema incompatibility | Text-only answers, one ignored image, unsupported options, or invalid replies | Verify the installed runtime and model digest with distinguishable synthetic images, a real raw/overlay pair, and schema validation. HTTP 200 is insufficient. Fail visibly rather than assuming full-video support | Developer + local operator |
| RAM/VRAM/context limits | OOM, swapping, excessive latency | Start with 2B quantization, one request, bounded image size/context/output, CPU YOLO or sequential loading. Measure peak memory locally. An 8 GiB GPU target is unverified, not a purchase requirement | Local operator |
| GPU and package compatibility | PyTorch/CUDA/driver combinations fail or consume unexpected resources | Pin a tested Python/package/runtime set at implementation. Record OS, GPU, driver, model digest and decoder versions. Preserve a CPU/stub path. Do not silently change device/model mid-session | Developer |
| Variable-rate phone video and backend differences | Wrong timestamp attached to the right-looking frame | Preserve rational PTS evidence, source index, chosen stream and rotation. Compare ffprobe presentation order with sequential OpenCV decode, test colored/indexed fixtures and real phone clips. Stop on missing/decreasing timestamps or mismatched frame counts | Developer + local operator |
| Rotation, HEVC, HDR, edit lists, or corrupt frames | Mirrored/rotated images, changed colors, or shifted mapping | H.264 SDR is the initial supported baseline. Probe content, not extensions. Do not silently transcode or normalize timing. Unsupported inputs get actionable errors while the original remains protected | Developer |
| Database/filesystem split failure | Half-published pairs, stale JSON, invalid exports | Pending/ready states, immutable hashes, revision checks, same-filesystem temporary writes, explicit reconciliation, crash injection, and export under a session lock. Full protocol is in DATA_MODEL | Developer |
| Restart, cancelled work, or late response | Duplicate inference, lost decisions, stale UI updates | Stable frame/detection IDs, unique constraints, append-only histories, expected revisions, keyed callbacks, interrupted status, and explicit resume. A timed-out request may still run, so wait for idle before retry | Developer |
| Temporal sampling and NMS | Brief errors missed or no natural duplicates observed | Report sampled-time and Qwen-selection coverage. Annotate all sampled frames. Synthetic duplicates test code only. Do not loosen detector thresholds to manufacture benchmark wins | Experiment owner |
| VLM latency and resource cost | Review overhead exceeds the value of found errors | Cap selected frames/attempts, expose progress/cancel, separate cold/warm time, count failed calls, and compare human-only baseline at matched quality. Report a negative or inconclusive result | Experiment owner |
| Small, correlated dataset | Inflated quality and business conclusions | Separate development/held-out recordings, independent annotations, clip-level reporting, denominator counts, and uncertainty. Four clips establish feasibility only | Annotator + experiment owner |
| Private footage and sensitive raw responses | Personal information reaches logs, public shares, exports, or Git | Default loopback only, Gradio share disabled, local Ollama only, no automatic remote fallback, plain-text model output, redacted logs, restricted file serving, and no telemetry containing media. Exports are sensitive local artifacts | Developer + local operator |
| Malicious upload/path/prompt | Resource exhaustion, unintended file access, or injected instructions | Bounded bytes/duration/pixels/frames, local file inputs only, safe generated paths, no shell interpolation, no model tools, no media URLs, no symlink escape, no rendering model HTML, dependency updates, and ZIP allowlist | Developer |
| Disk full, unexpected shutdown, or backup mismatch | Missing original, corrupt writes, or unusable restore | Preflight disk budget, propagate write errors, verify checksums, retain staging evidence, and back up DB plus media at a consistent point. Test restore to a separate directory/database before claiming recoverability | Local operator |
| License/distribution assumptions | Repository or product distribution conflicts with dependency/model terms | Review exact code and weight versions before redistribution or choosing the repository license. Record notices and dependencies. Public-service or commercial distribution is outside this MVP decision | Project owner |

## Runtime API facts and remaining uncertainty

Ollama documents image arrays with base64 REST input and schema-constrained output. Its current structured-output guide also says Ollama cloud does not support that feature. Those facts do not prove that a specific installed Qwen build correctly reasons over both images. Keep a local capability gate and versioned request/response evidence. Sources: [vision API](https://docs.ollama.com/capabilities/vision), [structured outputs](https://docs.ollama.com/capabilities/structured-outputs), and [Qwen tag](https://ollama.com/library/qwen3-vl:2b-instruct).

The runtime seam is a small callable, not a multi-provider service. Implementing a trusted remote runtime later requires explicit operator configuration, privacy review of the actual destination, and the same image/schema contract tests. It must never be an automatic fallback after local failure.

## Licensing investigation

Ultralytics publishes AGPL-3.0 and commercial Enterprise licensing paths. Review the actual package and YOLO checkpoint license, notices, intended distribution, and repository-license compatibility before distributing a combined application. Do not infer that a private repository automatically resolves licensing obligations. No license is selected and no commercial terms are purchased in this planning delivery. Source: [Ultralytics licensing](https://www.ultralytics.com/license).

The official Qwen3-VL-2B-Instruct model card lists Apache-2.0, and the proposed Ollama tag also exposes license information. Verify the resolved model artifact and notices rather than applying one model family's license to every derivative or runtime. Sources: [official Qwen model card](https://huggingface.co/Qwen/Qwen3-VL-2B-Instruct) and [Ollama artifact](https://ollama.com/library/qwen3-vl:2b-instruct).

At dependency pinning, inventory Gradio, OpenCV, PostgreSQL, Ollama, PyTorch, the Python packages, and the exact FFmpeg build. FFmpeg licensing depends on enabled components, so preserve build configuration and review its [official legal guidance](https://ffmpeg.org/legal.html). This is a release gate, not a claim that every distribution is already cleared.

## Local privacy and retention defaults

Process valid media locally and retain completed session evidence until the operator explicitly removes it. V1 has no automatic deletion job. Keep database credentials and media under user-restricted directories, with OS disk encryption recommended for personal footage. The database is not a media blob store. Exports contain video and model explanations and need the same protection as originals.

Reject unvalidated uploads before creating a canonical session. Remove only app-owned rejected temporary copies when the handler finishes. Bound diagnostic text and never keep the rejected file in logs. If cleanup fails, list the leftover app-owned path for operator attention. After a valid original is saved, a processing failure preserves that session and its evidence. Quarantined orphan/staging data is reported for explicit operator cleanup.

Gradio's upload/cache directory and allowed file-serving paths require a version-specific check. Restrict serving to the intended session/export files, block environment/model/DB directories, and never expose `DATA_ROOT` broadly through a public share. Verify the pinned release against [Gradio's file-access guide](https://www.gradio.app/guides/file-access) and [Blocks implementation](https://github.com/gradio-app/gradio/blob/main/gradio/blocks.py). Guide availability and exact API arguments must be rechecked during implementation.

## Pre-implementation decisions and evidence gaps

The proposed baseline uses Python 3.11, PostgreSQL 16, H.264 SDR phone clips, YOLO11n, and local Qwen 2B. The user's actual CPU, RAM, GPU/driver, Ollama build, phone codecs, free disk, and preferred labor-cost assumptions are not known. They do not block planning. T1 and T10 discover compatibility and measured cost inputs before making live-performance claims.

At the end of planning, no model was downloaded or run, no phone footage was decoded, no application tests existed, no database was provisioned, and no live hardware/resource benchmark was performed. Future test names and acceptance targets describe work to do. They are not passing results.
