# Documentation index

Start with the [project README](../README.md) for the current application. The documents below separate implementation facts, experimental evidence and proposed work.

## Current application

| Document | Purpose |
| --- | --- |
| [Architecture](ARCHITECTURE.md) | Components, frame flow, eight-table SQL diagram, foreign keys, JSON and file storage. |
| [Setup and test guide](V1_GUIDE.md) | Dependencies, running the app, a manual test, export and recovery. |
| [Verification record](V1_VERIFICATION.md) | Automated and real-model checks, what each establishes and what remains unverified. |
| [Prompt iterations and results](PROMPT_LOOP_RESULTS.md) | Why the prompt changed, crop controls, observations and limitations. |
| [Reviewed cup datasets](DATASET_REVIEW.md) | Optional local annotations, whole-frame review, frozen releases and candidate training. |
| [Dataset verification](DATASET_VERIFICATION.md) | Synthetic checks, browser evidence and remaining live-test gaps. |

## Future work

[Future improvements and the HVAC blueprint](ROADMAP.md) covers evaluation, performance, reviewed annotations, additional auditing rules and domain adaptation. These are proposals, not implemented capabilities.

## Original planning archive

These documents preserve the original requirements and design decisions. Their bodies can describe earlier sampling settings, unbuilt evaluation tables and proposed acceptance gates. Use the current architecture and guide for implemented behavior.

- [Product brief](PROJECT_BRIEF.md) and [original planning request](PLAN_REQUEST.md)
- [Implementation plan](IMPLEMENTATION_PLAN.md) and [ordered task list](DEVELOPMENT_TASKS.md)
- [Proposed data model and JSON contracts](DATA_MODEL.md)
- [Evaluation protocol and proposed business KPIs](EVALUATION.md)
- [Risk register](RISKS.md) and [prompt-experiment plan](superpowers/plans/2026-11-17-qwen-yolo-experiment-loop.md)
