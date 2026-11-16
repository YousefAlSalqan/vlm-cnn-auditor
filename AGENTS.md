# Repository agent instructions

- Prioritize a **working, understandable MVP** over infrastructure complexity.
- The user wants a cloud Codex agent to **author a plan first**, synced to this personal GitHub repository.
- Preserve exact mapping `session_id` + `frame_id` + `timestamp_ms` between raw frame, annotated frame, per-frame JSON, YOLO metadata and VLM review.
- YOLO is the baseline detector. Qwen3-VL is a **fallible reviewer**, not authoritative ground truth.
- Do not automatically retrain YOLO or apply Qwen corrections without review.
- Cup only in V1: COCO ID `41`; focus on false negatives, false positives, duplicates and count mismatches.
- Keep video/JPG/JSON on local disk and searchable metadata/decisions in PostgreSQL.
- Prefer Gradio + direct Python pipeline for V1; FastAPI can come later.
- Plan/test for resource constraints and inference latency; keep an optional model adapter to handle Qwen local vs remote runtimes.
- Keep user footage, secrets, DB dumps, model weights and generated exports out of git.
- Avoid claims of energy savings or business gains absent measurement.
- Agent's first deliverable: detailed plan docs in a new branch / pull request, not deployment.
