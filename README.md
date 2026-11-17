# VLM-CNN Auditor

A model-agnostic quality-assurance layer for computer-vision deployments. The first demo audits **YOLO cup detections (COCO class 41)** using **Qwen3-VL**, records suspected errors with frame-level evidence, and allows a human reviewer to accept or reject suggested corrections.

> Status: first working local version. Eight core PostgreSQL tables, Gradio review UI, YOLO/Qwen integration and verified exports are implemented. Independent labels and business evaluation remain deferred.

New sessions use the tested regional cup prompt by default. YOLO runs on source frames **0, 5, 10, ...**, and Qwen reviews those same frames. Qwen checks one to three raw image regions and proposes possible missed cups for human review. Model weights and the database schema are unchanged. Restart the app and upload a new session to use the new flow. A 27-second, 30 fps clip produces 162 analyzed frames, so Qwen processing can take longer than the earlier one-frame-per-second, twelve-review default.

## Run the prepared Windows workspace

Double-click **`start-auditor.cmd`**, or run `.\scripts\start-local.ps1` from the checkout. Open **http://127.0.0.1:7860**.

The launcher checks the schema, FFprobe, YOLO checkpoint and local Qwen installation. It can start existing portable services configured in your private `.env`. It does not install software or initialize databases. Other machines need the [setup and recovery guide](docs/V1_GUIDE.md).

## First demonstration
1. Record a 15–30 second phone video of a cup under normal, occluded, empty-scene, and poor-light conditions.
2. Transfer the video to the laptop by USB. Upload through Gradio, click Save video, then click Start analysis separately.
3. Extract every fifth source frame, retaining source frame index, presentation timestamp, and stable IDs.
4. Run YOLO (cup class 41) and save raw images, rendered annotations, bounding boxes, class labels and confidences.
5. Ask Qwen3-VL to inspect raw regions on each of those frames. Derive possible missed-cup suggestions only from positive regions without intersecting YOLO boxes. Saved older reviews retain their original rule types.
6. Validate each regional observation and the host-derived finding, retain raw responses, and show synchronized evidence plus unverified regional observations in the review UI.
7. Accept, reject or defer each finding with a note. Decisions remain separate from predictions and independent reference labels. Nothing automatically retrains the model.

Media defaults to `~/vlm-cnn-auditor-data/session_<UUID>/`. `DATA_ROOT` changes that location. PostgreSQL owns metadata and decisions. See the [actual V1 database diagram](docs/V1_GUIDE.md#actual-database-architecture).

## Key principle
A VLM's suggestion is **not ground truth**. Separate raw YOLO predictions, VLM proposals, and verified human labels. The original video and raw frames must never be overwritten.

## Stack and verification
- Python, OpenCV and Ultralytics YOLO (check licensing for redistribution)
- Qwen3-VL 2B via Ollama for local image inference (adapter should permit a future Transformers backend)
- Gradio for upload, frame-by-frame comparison and feedback
- PostgreSQL + SQLAlchemy + Pydantic for metadata, normalized findings and audit trail
- FastAPI **optional in the first MVP**; use direct Python functions from Gradio to ship sooner
- Pytest for unit/integration tests, with fake model responses for CI

Direct versions are in [pyproject.toml](pyproject.toml). The tested Windows Python 3.11 CPU environment is frozen in [requirements-cpu.lock](requirements-cpu.lock). Tests require a separate PostgreSQL database named `auditor_test*`.

```powershell
$env:AUDITOR_TEST_DATABASE_URL = 'postgresql+psycopg://TEST_ROLE:TEST_PASSWORD@127.0.0.1:5432/auditor_test_v1'
$env:AUDITOR_BROWSER_TESTS = '1'
.\.venv\Scripts\python.exe -m playwright install chromium
.\.venv\Scripts\python.exe -m pytest -q
.\.venv\Scripts\python.exe -m ruff check .
```

CI uses synthetic footage and fake models. Separate local smoke checks use real YOLO and Qwen. Personal-footage accuracy, physical-phone compatibility and business savings still need human testing.

## Repo docs
- [V1 guide, setup, flow and actual database architecture](docs/V1_GUIDE.md)
- [V1 verification and review results](docs/V1_VERIFICATION.md)
- [Agent instructions](AGENTS.md)
- [Requirements and architecture constraints](docs/PROJECT_BRIEF.md)
- [Cloud agent planning assignment](docs/PLAN_REQUEST.md)
- [Implementation plan, architecture, and dependencies](docs/IMPLEMENTATION_PLAN.md)
- [Data model, frame synchronization, and export contract](docs/DATA_MODEL.md)
- [Evaluation, testing, and business KPIs](docs/EVALUATION.md)
- [Risks, privacy, and licensing gates](docs/RISKS.md)
- [Ordered development tasks and acceptance checks](docs/DEVELOPMENT_TASKS.md)

Videos, frames, model weights, PII, and database credentials must remain out of git. This project is independent of any specific company.
