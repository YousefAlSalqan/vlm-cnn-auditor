# Product and Engineering Brief

## Goal
Build a reusable **VLM-assisted CV quality assurance** application. The user's first experiment focuses only on **cup detection** (COCO class 41). The product's business objective is to **reduce manual review and cost per verified detection error**, not merely to maximize VLM agreement with YOLO.

## Functional requirements
- A user uploads a 15–30 second MP4/MOV video captured with a phone.
- Preserve original uploaded file; create a stable session ID and video record.
- OpenCV extracts frames with precise mapping to original source frame index and `timestamp_ms`. Configure sampling frequency; do not assume constant frame rate if actual timestamps are obtainable.
- Run YOLO detector filtered to cup class (41); persist detections including model version, label, score and bounding box in pixels (explicit `xyxy` convention).
- Save untouched raw frame JPEG and a YOLO-rendered annotated JPEG per reviewed frame, plus a machine-readable JSON snapshot with identical frame ID and timestamp.
- Qwen3-VL reviews selected frame pairs plus detections, not necessarily entire videos. Apply rule IDs:
  - R1 possible false negative: cup not in detections
  - R2 possible false positive: detection may not represent a physical cup
  - R3 duplicate detection: same cup appears boxed twice
  - R4 count mismatch: visible cup count may differ from YOLO count
- Qwen returns strict structured suggestions with rule IDs, evidence, referenced detection IDs, estimated count if useful, explanation and `needs_review`. It may respond `insufficient_evidence` / `no_issue`.
- Validate JSON with Pydantic before database writes; retain raw responses for debugging.
- Reviewer sees original frame and overlay/annotated counterpart synchronized with metadata, findings, and buttons to accept/reject. Store reviewer decision separately from proposed correction.
- Export a session ZIP with original video, frames, annotated frames, JSON and a manifest of timestamps/model versions.
- Design for one local user; no account management required in MVP.

## Data
Propose normalized PostgreSQL tables: `sessions`, `videos`, `frames`, `yolo_detections`, `vlm_reviews`, `review_decisions` (or justify a simpler schema). Store media on the filesystem with relative paths; PostgreSQL owns metadata. Generate JSON snapshots from canonical persisted records or specify an atomic write-and-reconciliation approach to avoid drift. Use stable IDs; maintain referential integrity. Do not store all media as database blobs.

Example session filesystem:

```
data/session_<uuid>/
  original.mp4
  frames/frame_000001.jpg
  annotated/frame_000001.jpg
  json/frame_000001.json
  manifest.json
```

## Architecture decisions
- MVP: Python pipeline + Gradio calling Python functions, SQLAlchemy + PostgreSQL, Ollama API adapter for Qwen3-VL 2B, YOLO, OpenCV.
- FastAPI can be added later. Do not overengineer microservices or message queues.
- Prefer sequential/queued batch processing with observable progress and bounded VLM sampling.
- If video support is unreliable in the chosen Ollama API, use two images per timestamp (raw frame and annotated frame); never silently assume full-video support.
- For model comparisons, human-verified annotations are the only reference labels, not VLM suggestions.
- Avoid treating AI-suggested boxes as precise ground truth; optional corrected bounding boxes require human verification.

## Evaluation + business value
Define/test: verified YOLO errors found, precision/recall of Qwen flags, false-correction rate, reviewer time saved relative to a baseline sampling strategy, cost per verified error, throughput, inference latency, and GPU/CPU resource use. Record model versions and prompt/rule version for traceability. Separate inference cost from downstream business savings. No unverified ROI claims.

## Safety, privacy, licensing
- Default local-only media processing; no public video upload service required.
- Gitignore all videos, images, frames, credentials, local DB volumes, model weights.
- No secret tokens in repo. Provide `.env.example`.
- Investigate YOLO/Ultralytics and model licensing before deciding repository/license distribution.
- Keep changes proposal-only; no unattended model retraining or automated real-world control.
- Do not tailor the project to EclimAI or any specific business.
