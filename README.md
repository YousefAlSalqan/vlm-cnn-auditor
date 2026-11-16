# VLM-CNN Auditor

A model-agnostic quality-assurance layer for computer-vision deployments. The first demo audits **YOLO cup detections (COCO class 41)** using **Qwen3-VL**, records suspected errors with frame-level evidence, and allows a human reviewer to accept or reject suggested corrections.

> Status: implementation plan prepared for owner review. Application coding has not started and requires a separate request.

## First demonstration
1. Record a 15–30 second phone video of a cup under normal, occluded, empty-scene, and poor-light conditions.
2. Upload the video via Gradio; save the original file to local session storage.
3. Use OpenCV to extract sampled frames, retaining source frame index, video timestamp, and stable IDs.
4. Run YOLO (cup class 41) and save raw images, rendered annotations, bounding boxes, class labels and confidences.
5. Ask Qwen3-VL to compare original frames, annotated frames and structured detections against four rules: possible missed cup, false positive, duplicate and count mismatch.
6. Validate the model's structured findings, persist artifacts and metadata, and show synchronized evidence in a review UI.
7. Accept/reject suspected errors. Reviewed decisions become labels for evaluation, not automatic model retraining.

## Key principle
A VLM's suggestion is **not ground truth**. Separate raw YOLO predictions, VLM proposals, and verified human labels. The original video and raw frames must never be overwritten.

## Proposed stack
- Python, OpenCV and Ultralytics YOLO (check licensing for redistribution)
- Qwen3-VL 2B via Ollama for local image inference (adapter should permit a future Transformers backend)
- Gradio for upload, frame-by-frame comparison and feedback
- PostgreSQL + SQLAlchemy + Pydantic for metadata, normalized findings and audit trail
- FastAPI **optional in the first MVP**; use direct Python functions from Gradio to ship sooner
- Pytest for unit/integration tests, with fake model responses for CI

## Repo docs
- [Agent instructions](AGENTS.md)
- [Requirements and architecture constraints](docs/PROJECT_BRIEF.md)
- [Cloud agent planning assignment](docs/PLAN_REQUEST.md)
- [Implementation plan, architecture, and dependencies](docs/IMPLEMENTATION_PLAN.md)
- [Data model, frame synchronization, and export contract](docs/DATA_MODEL.md)
- [Evaluation, testing, and business KPIs](docs/EVALUATION.md)
- [Risks, privacy, and licensing gates](docs/RISKS.md)
- [Ordered development tasks and acceptance checks](docs/DEVELOPMENT_TASKS.md)

Videos, frames, model weights, PII, and database credentials must remain out of git. This project is independent of any specific company.
