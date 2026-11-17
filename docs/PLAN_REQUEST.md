# Assignment for cloud Codex planning agent

> **Historical planning request.** The planning and subsequent first implementation have been completed. The instructions below preserve the original assignment. They do not describe the current application status. Start with the [documentation index](README.md) for current architecture, setup, evidence and future work.

You have a clean personal GitHub repository. **First create an implementation plan** for the VLM-CNN Auditor; do not jump into building an overengineered system.

Read `README.md`, `AGENTS.md` and `docs/PROJECT_BRIEF.md` first.

Deliverables to commit on a feature branch (prefer opening a PR for review):
1. `docs/IMPLEMENTATION_PLAN.md`: staged milestones, architecture diagram, component boundaries, dependencies, minimum hardware, setup workflow and first-test acceptance criteria.
2. `docs/DATA_MODEL.md`: PostgreSQL tables and indexes, JSON schema with synced frame IDs/timestamps, stable paths, session export manifest, migration strategy.
3. `docs/EVALUATION.md`: 4 cup-test scenarios with only one physical cup, ground-truth collection method, precise metrics, cost calculations, model-comparison approach and sample size caveats.
4. `docs/RISKS.md`: model hallucinations, hardware/VRAM constraints, Ollama multimodal image API verification, timestamps and frame sync, data privacy, licenses, inference performance and recovery from model errors.
5. `docs/DEVELOPMENT_TASKS.md`: exact ordered small issues/tasks to reach an end-to-end test.
6. Clearly distinguish what can be verified in the cloud (unit/integration tests with stubs) from what requires the user's local phone/GPU.
7. If the environment and access permit, only after planning is complete and visible, propose the minimum scaffold and confirm CI scope.

Do not commit sample personal videos, fake performance results, large model weights or credentials. Use real links for sources where needed. Explicitly state any assumptions about camera formats and local compute. Keep MVP focused on Qwen3-VL, YOLO cup class 41, OpenCV, PostgreSQL, Gradio and synchronized frame/JSON artifacts; FastAPI is optional.

Use a plan-first branch such as `codex/implementation-plan`, push it, and return branch/PR links, what was done, what was tested and blockers.
