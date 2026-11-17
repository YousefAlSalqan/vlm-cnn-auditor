# Qwen and YOLO Experiment Loop Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Status:** Proposed plan only. No new model runs, application changes or GitHub publication are authorized by this document itself. The owner requested this plan before starting the loop.

**Goal:** Find a concrete, visually supported Qwen suggestion that adds value beyond the saved YOLO cup detections, then test whether that behavior repeats while recording wrong suggestions.

**Architecture:** An isolated experiment runner reads existing frame pairs and YOLO JSON without a database connection. It records bounded model attempts and separate visual assessments under the ignored repository runtime folder. The application continues using its current prompt and schema until a useful change is separately reviewed for integration.

**Tech Stack:** Existing Python environment, Qwen3-VL 2B through local Ollama, saved YOLO11n results, OpenCV, Pydantic, httpx, filelock and pytest. No new dependencies or model downloads are proposed.

**Spec:** Owner's request in this chat: continue testing for suggestions that can help YOLO in any useful way, present a plan and possible uses before starting. Repository requirements are in `AGENTS.md` and `docs/PROJECT_BRIEF.md`. Prior local evidence is in `.local-runtime/audit-prompt-20261117/REVIEW.md`.

**Proposed execution:** Sequential work in the current chat, with one local model request at a time. Begin with missed cups, then boxed-object checks. Labels and tracking are later targets if the first targets do not produce a useful signal.

## Global Constraints

- Keep the installed `qwen3-vl:2b-instruct` digest and YOLO weights unchanged. Preserve generation options `temperature=0`, `num_ctx=8192`, `num_predict=1024` for the initial experiment family.
- Do not connect to PostgreSQL, run migrations, launch the application, alter saved sessions, retrain YOLO or change laptop settings.
- Read the supplied 27-second clip and existing session evidence. Write new artifacts only inside this repository. Keep personal media, actual model responses, local paths and credentials out of Git.
- Preserve original `session_id`, `frame_id` and `timestamp_ms`. New crops and views have additional source-coordinate transforms and hashes, never fabricated source frame identities.
- Keep the current `ReviewResponse` and `validate_review` unchanged for any result described as compatible with the application. Diagnostic outputs remain explicitly separate from validated application reviews.
- Qwen suggestions do not overwrite YOLO predictions. They do not become accepted labels or training data automatically.
- Live inference requires this laptop and the existing local Ollama runtime to remain available. This plan does not establish a cloud job.

## Uses to Investigate

| Priority | Use | Question for Qwen | What would count as useful |
|---|---|---|---|
| 1 | Missed cups, R1 | Which visible physical cup is outside all YOLO boxes? | A specific real cup or group with appearance and location, missing from the baseline detections. Start with hanging cups already observed in the clip. |
| 2 | False positives, R2 | Is this particular boxed object a physical cup, another object, or unclear? | A correct warning about a wrong detection, without rejecting correctly boxed cups. |
| 3 | Duplicates and counts, R3/R4 | Do two boxes refer to the same cup? Is a confident visible count inconsistent with YOLO? | A real redundant box or count discrepancy. Do not force a count through blur or occlusion. |
| 4 | Label drafts and hard examples | What object or appearance is visible here, and what needs review? | A useful draft annotation or difficult example for human review and possible later training. Naming a class filtered out by the cup-only configuration does not establish a YOLO failure. |
| 5 | Continuity and tracking review | Is the same object still visible when a detection or track disappears? | A correctly identified detection dropout, or a track problem once a tracker baseline exists. The current application has no tracking baseline, so no tracker improvement claim is possible yet. |

These are hypotheses for this installed model. Published model capabilities are motivation for experiments, not evidence of local accuracy. Relevant primary sources are the [Qwen model card](https://huggingface.co/Qwen/Qwen3-VL-2B-Instruct), [Ollama structured-output guidance](https://docs.ollama.com/capabilities/structured-outputs), and [Ultralytics tracking documentation](https://docs.ultralytics.com/modes/track/).

## Review Focus

1. A cup already boxed by YOLO is falsely called missed. The known text-mug failure must stay in the comparison and count against the prompt.
2. A crop excludes part of the scene. A regional observation must never be interpreted as a whole-frame `no_issue` result.
3. A prompt produces valid JSON with invented visual facts or wrong object references. Structural validation and visual scoring remain separate.
4. Repeated views of one mug inflate the number of discoveries. Record both frame-level proposals and distinct underlying object/error examples where they can be identified.
5. Prompt changes fit the same familiar scene. Reserve unused-for-tuning frames, freeze each candidate before evaluation, and disclose that a single video cannot establish general accuracy.

## Task 1: Freeze Inputs and Define the Scorecard

**Files:** Create private `.local-runtime/qwen-yolo-loop/<run>/baseline.json`, `cases.json` and `assessments.json`. Do not modify the original session or earlier trial artifacts.

**Interfaces:** `cases.json` records source frame keys, image paths/hashes, detection UUIDs, development/evaluation assignment, and any crop transforms. `assessments.json` records visually supported, contradicted, ambiguous and unassessed proposals, plus reviewer provenance. These files are experiment data, not database records or canonical frame snapshots.

- [ ] Read the current checkout, source contracts and existing evidence before programming. Capture a new behavior-guard baseline inside the ignored repository artifact directory, honoring the owner's repository-only write restriction.
- [ ] Select a small development set from the existing 27 sampled frames. Include known missed-cup regions, correctly detected cups and blurred regions. Reserve at least five other frames that are not used to tune the next candidate.
- [ ] Inspect the selected reference views before seeing new candidate outputs. Record specific objects and uncertainty. Assistant visual checks are provisional references, not independent human ground truth.
- [ ] Keep the previous 0-, 13- and 26-second failures in the regression set. Do not silently replace them with easier examples.
- [ ] Define success before running: one clear, actionable extra suggestion is a useful signal. A repeatable integration candidate also needs the unchanged application schema to pass, a second check on reserved inputs, and an honest list of false suggestions. Human confirmation remains necessary before accepting a label.

## Task 2: Implement the Smallest Isolated Runner

**Files:** Create `experiments/qwen_yolo_loop.py`, general experiment prompt files under `experiments/prompts/`, and `tests/test_qwen_yolo_loop.py`. Keep active `auditor/models.py`, `auditor/contracts.py`, database code and application startup unchanged.

**Interfaces:** Provide `run_batch(session_dir: Path, prompt_path: Path, frame_ids: list[str], output_dir: Path, max_calls: int = 20) -> Path`. It returns the result-index path. Use the existing `OllamaReviewer` request preparation, deadline, response bounds and strict validation for full-frame audit trials. Input paths are explicit. The private case manifest supplies any view transformations.

- [ ] Write offline tests before adding logic. A duplicate run name must fail without dispatch or overwrite. An invalid frame identity/reference must remain rejected. A mocked uncertain request must stop the batch. A diagnostic crop must not produce a trusted whole-frame no-issue review.
- [ ] Run only those offline tests first and confirm they fail for the missing runner. Do not supply a database test URL or invoke database fixtures.
- [ ] Add the minimal runner using existing helpers. Persist a receipt before each model request. Save exact prompt text/hash, model identity, settings, input hashes, raw response, parsed response when valid, latency and terminal status. Process sequentially and never repair raw model output to make it pass.
- [ ] Enforce the per-batch call limit and stop after uncertain runtime/transport failures. Do not automatically retry a request whose completion is unknown.
- [ ] Rerun the offline tests and existing model/contract/storage checks. Confirm original application, database and infrastructure definitions remain unchanged against the baseline.

## Task 3: Run Adaptive, Measurable Rounds

**Files:** Private per-run prompts, attempts and assessments under `.local-runtime/qwen-yolo-loop/`. Update a sanitized `docs/PROMPT_LOOP_RESULTS.md` only with supported findings. Do not publish footage, actual session IDs or private responses.

**Interfaces:** Each round consumes frozen cases and one declared hypothesis. It produces a complete result table containing structural validity, useful suggestions, contradicted suggestions, ambiguous/unassessed suggestions, abstentions, latency and distinct-example grouping. Assessments are separate from immutable predictions.

- [ ] Begin with full-frame missed-cup prompts. Ask one narrow question at a time, preserving exact frame identifiers and existing schema. Compare new candidates with the previous prompt on the same inputs.
- [ ] If broad scene comparison still confuses boxed and unboxed cups, test systematic overlapping regions that cover the scene independently of YOLO's boxes. Keep a full-frame control. Record native-coordinate mappings and view hashes. Treat this as a changed input protocol, not a prompt-only gain.
- [ ] Separately test cropped checks of already-boxed objects for false positives. Use correctly boxed cups as controls. If needed, use a small diagnostic schema first, followed by a separately scored full-frame application-schema audit. A diagnostic cannot be counted as an application-schema success.
- [ ] Change one factor per comparison: wording, task decomposition or image region. Do not combine changed settings, different frames and different prompts into one claimed improvement. Repeating an identical deterministic request without a new hypothesis is not a new experiment.
- [ ] Run batches of at most 20 calls and inspect all results after each batch. Initial estimate: 30 to 60 minutes including setup and visual review, subject to model latency and timeouts. Continue with a different hypothesis or next use case when a round fails. Stop for a useful result that needs client review, exhausted meaningful approaches on the available footage, or an external/runtime blocker. Do not manufacture success by discarding failures.

For diagnostic region sweeps with F frames and K regions per frame, request count is bounded by F times K and the batch call limit. Encode and process one view pair at a time. Retained result storage grows with completed attempts. No unbounded parallel jobs, persistent cache or database work is needed.

## Task 4: Verify and Make the Checkpoint Reviewable

- [ ] Freeze a promising candidate before testing reserved frames. Keep the known boxed-text-mug control. Recheck frame/time mapping, strict JSON validation and the actual visual claim independently.
- [ ] Record useful and wrong suggestions separately. Suggestion precision can be calculated only on adjudicated proposals, with ambiguous/unassessed counts reported. Do not claim recall without a sufficiently labeled reference set, or count adjacent views as independent scenes.
- [ ] Prepare a compact proof for each promising suggestion: source time, raw/overlay view, Qwen's exact statement, the corresponding YOLO result and the review disposition. Ask the owner to confirm the useful examples before treating them as accepted labels.
- [ ] Keep an isolated improvement even if the overall method remains unreliable, labeled as a partial experimental result. Do not require every cup to be found before preserving useful progress.
- [ ] After the owner resumes the GitHub checkpoint request, publish only the general prompt/runner, offline tests and sanitized results to the personal feature branch. No main merge, PR, database changes, model-weight changes or automatic prompt activation are part of this plan.

Qwen does not train YOLO during this loop. The immediate benefit being tested is a better review lead, a justified proposed correction, or a useful draft training example. Any later claim that YOLO itself became more accurate requires a separate training or inference change and a measured comparison.
