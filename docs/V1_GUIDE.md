# V1: setup, workflow and boundaries

This release implements the local upload, analysis, review and export flow approved after planning. Eight operational tables are present. Independent annotations, adjudication and reviewer trials remain future work. Accepting a Qwen suggestion is a judgment, not an independently verified label. The original plans remain design references.

This branch also offers a [local dataset review and fine-tuning workflow](DATASET_REVIEW.md). Its toggle starts OFF. It creates model-assisted cup labels that a person reviews before approval. Independent reference annotation and adjudication remain separate work.

## Manual test

1. Run `start-auditor.cmd` from the checkout. Open http://127.0.0.1:7860.
2. Record one physical cup in four conditions: normal light, partial occlusion, poor light and an empty scene. Use separate 15–30 second SDR clips, up to 1080p and 200 MiB.
3. Transfer each clip by USB. Select it in Phone video and click Save video. Click Start analysis separately.
4. Review flagged frames, then inspect unflagged and unselected samples. Save Accept, Reject or Defer with a note for each finding you inspect.
5. Create a session ZIP. Report the session ID and visible error if anything fails. Keep `.env` and personal footage private.

V1 does not synchronize the phone automatically. The browser runs on the laptop's loopback address and is not exposed to the phone or internet.

The prepared laptop has 16 GiB RAM, an i7-11800H and a 4 GiB RTX 3050 Laptop GPU. YOLO runs on CPU to leave memory for Qwen. Ollama chooses its available compute. New sessions analyze every fifth source frame with both models. Each Qwen frame review uses one to three sequential calls, each with a 120-second default deadline. This is offline analysis, not a real-time speed guarantee. A 27-second clip at 30 fps produces 162 analyzed frames and can take substantially longer than the previous twelve-review default.

## Full flow

Save the video, start analysis, review synchronized evidence, then export the session. The [current architecture](ARCHITECTURE.md) contains the component diagram and complete processing flow.

`FRAME_STRIDE=5` selects source indices 0, 5, 10 and so on, with up to 720 samples from the 3600-source-frame input bound. YOLO completes the sampled-frame stage before Qwen reviews every one of those frames. Each Qwen review uses raw regional views and can derive an unverified R1 missed-cup lead. The app does not automatically create corrected boxes, training labels or an object count.

Variable-frame-rate clips retain their actual presentation timestamps. Legacy sessions retain their recorded time-based sampling and review budget. New sessions use the stride default even if an older `.env` still contains `SAMPLE_HZ` or `VLM_BUDGET`. Restart the app and upload a new session to use the new default.

## Actual database architecture

The app has eight operational PostgreSQL tables with enforced foreign keys. The [database diagram and table guide](ARCHITECTURE.md#postgresql-design) describe the actual schema, composite scope constraints and ordered decision history. Media bytes stay on disk.

Alembic creates eight tables in `0001_core`, then adds uncertain-request recovery in `0002_review_recovery`. The app requires the current revision and never upgrades a schema on startup. `human_annotations`, `reference_adjudications` and `review_trials` are deferred.

## Video storage and synchronized JSON

The default is your home directory plus `vlm-cnn-auditor-data`. On the prepared laptop: `C:\Users\Dell\vlm-cnn-auditor-data`. `DATA_ROOT` can select another local directory outside Git.

```text
DATA_ROOT/
  session_<UUID>/
    original.mp4                 # or original.mov, exact copied bytes
    frames/frame_000000030.jpg   # decoded source frame index 30
    annotated/frame_000000030.jpg
    json/frame_000000030.json
    manifest.json                # latest exported manifest
    .staging/                   # recoverable pending files
  exports/<UUID>/<export_UUID>.zip
```

The original is copied with a streaming SHA-256 hash and probed from its saved location. Decoding is sequential. `frame_id` uses source index, never rounded time. `timestamp_ms` is elapsed presentation time from the first presented frame, using the stored rational time base and half-up rounding. Repeated timestamps remain distinct frames. Decreasing timestamps, unsupported metadata and probe/decode count mismatches stop processing.

Canonical raw JPEGs use quality 95 after the recorded quarter-turn rotation is applied once. These are decoded samples, not camera RAW files. Overlays are derived copies. The default Qwen prompt is `cup_presence_v2.txt`, promoted unchanged from the tested experiment. Qwen receives raw regional views with a maximum long edge of 1024 pixels. Native region coordinates, scale, image hashes and request hashes are recorded. Canonical images retain native dimensions.

The frame is divided into three horizontal bands. Bands intersecting any existing YOLO box are conservatively skipped, because their cup presence cannot establish a missing detection. If all bands intersect boxes, one full-frame context check still sends this source frame to Qwen and cannot generate R1 by itself. There is no cross-frame reuse of a model result.

PostgreSQL owns searchable metadata and decisions. Pairs are staged, metadata records expected hashes, then files move to final paths before the frame becomes ready. JSON is an atomic derived snapshot of committed rows. DB and filesystem writes cannot share one atomic transaction. Reconciliation recovers valid staged files and regenerates pending snapshots. Conflicting or missing evidence blocks publication/export instead of overwriting bytes.

Pair paths and JSON retain identical session/frame/time mapping. Export holds the session lock, reconciles snapshots, streams a temporary ZIP, and checks inventory, hashes, configuration/revisions, timing, nested review references and decision chains before publication. This detects internal inconsistency. It is not a signed authenticity certificate against someone who rewrites all evidence consistently.

Saved session evidence is never automatically deleted. Exports contain your footage. Gradio uses temporary upload/display copies, separate from canonical storage, with a one-day cache cleanup. Back up the session folders and PostgreSQL data together. V1 has no delete or restore UI.

## Qwen rules and recovery

The regional default proposes R1 possible missed cups. R2 through R4 remain readable in older reviews. Positive unboxed regions are grouped into one frame-level R1 finding. The **Qwen observations (unverified)** panel shows the original descriptions by area. A negative or unclear regional response never clears the whole frame. When no R1 candidate is established, the frame receives `insufficient_evidence` and remains available for human review.

| Rule | Meaning |
| --- | --- |
| R1 | A visible cup may be missing from detections. No existing detection reference. |
| R2 | A detection may not depict a physical cup. Reference the relevant detection. |
| R3 | Multiple boxes may describe one cup. Reference at least two detections. |
| R4 | The visible cup count may differ from YOLO's count. |

The host supplies the exact session/frame/timestamp. Qwen supplies only the diagnostic cup-presence response. Its original strings are retained in a versioned `cup_regions_v1` raw envelope. Saving and exporting replay the raw observations and geometric derivation against the prepared requests. Export also reconstructs the requests from actual source pixels. The normalized `ReviewResponse` contract and database tables are unchanged. Legacy raw responses still use their original strict validator. Each finding retains its own decision history.

Frame attempts are saved before the first regional dispatch. Each HTTP response remains capped at 64 KiB. Regional envelopes contain up to three bounded responses and are limited to 2 MiB during successful replay. Invalid schema output can receive one retry, with at most two frame attempts. Timeouts and uncertain transport interruptions stop before any later regional call and never automatically retry. Closing HTTP does not prove the model stopped. Cancellation is checked between regions as well as frames.

After a timeout, restart Ollama. For a portable process started by this launcher:

```powershell
.\scripts\start-local.ps1 -RestartQwen
```

Otherwise restart Ollama manually. Check **I restarted Qwen. Retry failed reviews.** under Recovery, then Start analysis. The acknowledgement is recorded and the checkbox resets for each run. A later uncertain call requires a fresh restart and acknowledgement. Two exhausted attempts require a new session. Startup marks in-flight requests uncertain after an app crash. Stop waits for the current step or its deadline.

Model tag/digest, runtime version, prompt, code identity and sampling settings are frozen. Resume refuses changed or unavailable identity. If Qwen's initial identity was unavailable, upload a new session after restoring it. Changes to analysis settings in `.env` apply to new sessions. Original predictions remain intact.

## Setup on another machine

Use Python 3.11, PostgreSQL 16, FFmpeg with `ffprobe` on PATH, and [local Ollama](https://docs.ollama.com/quickstart). Keep Ollama on loopback. Its [FAQ](https://docs.ollama.com/faq) documents `OLLAMA_HOST`, `OLLAMA_MODELS` and `OLLAMA_NO_CLOUD=1`. Downloads need internet access. Analysis uses installed local weights.

```powershell
py -3.11 -m venv .venv
.\.venv\Scripts\python.exe -m pip install -r requirements-cpu.lock
.\.venv\Scripts\python.exe -m pip install --no-deps -e .
Copy-Item .env.example .env
```

The lock records the tested Windows Python 3.11 CPU environment, without cross-platform wheel hashes. Other platforms can install `.[models,dev]` and validate their environment. CI uses `.[dev]` with fake adapters. CPU Torch wheels use the official PyTorch index in the lock.

Create a dedicated role/database from an administrator's `psql` session:

```sql
CREATE ROLE auditor LOGIN;
\password auditor
CREATE DATABASE vlm_cnn_auditor OWNER auditor;
```

Set `DATABASE_URL` in `.env` to the dedicated `postgresql+psycopg://` connection, with URL-encoded credentials and the actual port. Keep this file private. Apply the migration explicitly:

```powershell
.\.venv\Scripts\python.exe -m alembic upgrade head
.\.venv\Scripts\python.exe -m auditor.models --download-yolo
ollama pull qwen3-vl:2b-instruct
.\scripts\start-local.ps1 -Check
```

`ollama pull` must target the same host and model directory as your configured server. The default is port 11434. The prepared laptop uses portable Ollama on 11435 and PostgreSQL on 55432, outside Git. Optional `POSTGRES_BIN`, `POSTGRES_DATA`, `POSTGRES_PORT`, `OLLAMA_EXE` and `OLLAMA_MODELS` configure the Windows launcher. It starts an existing initialized database. Otherwise start the services yourself.

The [Qwen model card](https://huggingface.co/Qwen/Qwen3-VL-2B-Instruct) lists its license. [Ultralytics licensing](https://www.ultralytics.com/license) needs a separate distribution review. No weights, database volumes, credentials or personal media ship in this repo.

## Tests and business KPIs

Set `AUDITOR_TEST_DATABASE_URL` to a separate database named `auditor_test*`. Tests create disposable schemas there and refuse the normal app database. Set `AUDITOR_BROWSER_TESTS=1` and install Playwright Chromium to include desktop/mobile browser flow.

| Area | Implemented checks |
| --- | --- |
| Contracts/SQL | R1–R4 cardinality, strict JSON, incorrect frame/time/detection references, real migrations/composite foreign keys, invalid boxes and decision chains |
| Video/storage | CFR/VFR time, source indices, rotation, decode-count mismatch, container/time-base/span rejection, traversal, immutable originals, atomic snapshots and staged recovery |
| Models | Two-image transforms, frozen provenance/settings, proxy isolation, malformed envelopes, absolute deadline, output cap, durable attempts and restart gating |
| Workflow/export | Separate Save/Start, partial failure, cancellation/resume, append-only/idempotent decisions, corruption rejection, nested references and independent ZIP hashes |
| UI | Pinned Gradio construction, note/finding-switch safety, scoped history, UTC, session refresh, desktop/mobile workflow and narrow-screen overflow |

CI uses PostgreSQL 16, FFmpeg and fake models. Real local smoke checks establish that YOLO and Qwen can process synthetic evidence and generate a verified export. They do not establish accuracy. Personal phone codecs, physical-device display, cup-error coverage, full review timing and prolonged resource use still need human testing.

The app records counts, states, model identity, attempt latency, Qwen runtime metrics, stage wall time and decisions. These are operational measurements. V1 does not calculate precision, recall, time saved, cost per verified error, energy savings or ROI from acceptance counts.

The later pilot can use up to eight clips across the four conditions, with two people and independent labels. A candidate gate is about 20% less human review time at comparable verified error coverage. Measure paired review time, verified-error precision/recall, false corrections, inference latency/resource cost and cost per verified error using [EVALUATION.md](EVALUATION.md). Money assumptions and the complete comparison protocol remain undecided. A small pilot cannot establish deployment performance.

## Current limits

- One local user, one active analysis, latest 50 sessions in the UI.
- Up to 200 MiB, 15–30 seconds, 1080p long edge 1920 and 3600 source frames. New stride-five sessions contain at most 720 frames, all selected for Qwen. Try H.264 SDR first. HDR and arbitrary-angle rotation are rejected.
- No automatic phone sync, cloud fallback, accounts, independent reference-label editor, business dashboard or automatic evidence deletion. Optional corrected boxes and candidate training require the separate dataset workflow and explicit review.
- Loaded JSON history is capped at 16 MiB per frame during ZIP verification. Larger histories need a later streaming verifier.
- Speed, accuracy and business usefulness depend on actual footage and hardware. Qwen can agree with a wrong YOLO result.
