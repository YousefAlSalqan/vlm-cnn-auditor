# Dataset workflow verification

Verified locally on 2026-11-17 on `codex/reviewed-dataset-builder`. This records software checks for the optional workflow. It does not establish real Qwen box accuracy, training speed or improved detection quality.

## Results

| Check | Result | Scope |
| --- | --- | --- |
| Repository test suite, dataset browser enabled | 82 passed, 34 skipped | Synthetic exports, fake model responses, fake training and existing database-free checks. |
| Ruff lint | Passed | Application, experiments and test source. |
| Ruff formatting | Passed | Application and test source. |
| Browser flow | Passed | Desktop and 375-pixel touch viewport, with native coordinate checks and no recorded browser errors or failed HTTP responses. |
| Real PostgreSQL, Ollama and fine-tuning | Not run | The normal database, installed models and active checkpoint were not used by these checks. |

All 34 skipped checks require a disposable PostgreSQL test database, including the original database-backed browser flow. `AUDITOR_TEST_DATABASE_URL` was deliberately unset. The new dataset browser check uses files and a fake trainer. Thirty warnings came from Gradio's existing pandas compatibility code. They were not suppressed or fixed by changing dependencies.

To reproduce the isolated checks in an environment with the existing development/model dependencies and Playwright Chromium:

```powershell
Remove-Item Env:AUDITOR_TEST_DATABASE_URL -ErrorAction SilentlyContinue
$env:GRADIO_ANALYTICS_ENABLED = 'False'
$env:HF_HUB_DISABLE_TELEMETRY = '1'
$env:AUDITOR_DATASET_BROWSER_TESTS = '1'
python -m pytest -q -rs --basetemp=.local-runtime/test
python -m ruff check auditor experiments tests
python -m ruff format --check auditor tests
git diff --check
```

Use a short local test directory on Windows because deep temporary paths can hit legacy path-length limits. The review panel and browser check require local loopback connections. No real database is needed for the new file-backed tests.

## Coverage

- OFF defaults do not create dataset files, export a session, contact Qwen or start training. The existing app mounts the closed optional panel while retaining its original evidence images and controls.
- Verified exports keep frame/session/timestamp identities. Repeated imports are idempotent. Changed source evidence for an existing identity is rejected. Original export bytes remain unchanged.
- Manual box geometry, accidental negatives, stale revisions and stale frame selections are checked. Model origin survives edits. Earlier reviews survive later imports. Human edits revoke the current approval.
- Releases map canonical cup class 41 to YOLO training class 0, preserve review history, enforce whole-video splits, freeze prior split assignments and require explicit reviewed negatives. Changed release bytes fail checksum verification.
- Optional grounding validates units, multi-object geometry, crop mapping, invalid output and box limits. Timeouts stop subsequent calls and leave a durable pause. The normal analysis preflight rejects that pause before accessing a database engine. Clearing it requires explicit restart acknowledgement.
- The worker rejects OFF mode, invalid release inputs, changed base weights and incompatible class mappings. Duplicate approvals reuse one job. Cancellation while waiting or running preserves the release. Interrupted work does not restart on app launch.
- A fake model exercises the training integration: approved working data, existing checkpoint initialization, candidate saving, class-41/base versus class-0/candidate evaluation, separate output hashes and no activation. Its numbers are synthetic assertions, not experimental accuracy results.
- The browser imports two exports, checks default controls, clicks native image coordinates, adds/updates/deletes a box, reviews both frames, approves a release, observes simulated completion and checks mobile layout. Exact pixel mapping is checked using the pinned viewer's scale-down behavior.

## Preservation and review

The pre-code baseline was captured at planning commit `a26c414` in the isolated worktree. It records tracked-file hashes, original branch state, allowed differences and gaps. Local evidence is stored in `.local-runtime/dataset-implementation/`, which is excluded from Git. It includes test output, the implementation ledger and a file-preservation comparison. The original baseline remains outside the checkout and is not rewritten.

| Surface | Finding | Evidence or limit |
| --- | --- | --- |
| SQL schema, constraints and migration definitions | Preserved by source comparison | No changes to `db.py`, contracts or migrations. Live SQL state was not inspected. |
| Existing detector, Qwen audit prompts and every-fifth-frame configuration | Preserved by source comparison and existing isolated tests | No changes to the model adapters, regional audit prompt or analysis settings. |
| Existing synchronized media/JSON and export contract | Preserved by source comparison and synthetic export checks | Existing storage and export implementations are reused. No user footage was processed. |
| Optional panel and dataset files | Authorized addition | New local draft, review, release and candidate workflow. |
| Analysis preflight after an uncertain dataset Qwen call | Authorized addition | File-only pause guard runs before database access. The ordinary path has no pause marker. |
| Runtime packages, service setup and startup scripts | Preserved by source comparison | No dependency installation, settings rewrite or new service. |
| Git ignore rules | Authorized addition | `datasets/` is ignored alongside existing media/model/export rules. |

| Before | After | Why |
| --- | --- | --- |
| Two evidence images and finding decisions | Same evidence flow plus a closed opt-in dataset panel | Dataset annotations need a separate review boundary. |
| Regional text leads without corrected training boxes | Optional proposed boxes and manual native-coordinate editing | A person can turn supported leads into complete annotations. |
| No candidate training path | Approved releases queue one local candidate job | Explicit approval controls training and leaves the active detector intact. |

Review was performed inline without subagents. The simplification pass reused release-readiness checks, image handling, export verification, locks and shared box-overlap geometry. No generic agent framework, annotation server, new database layer or new runtime dependency was added.

Ingestion indexes frame identities and streams image/file work, with expected linear record lookup plus input-byte verification and copying. Each UI edit reads and rewrites the bounded draft and its history, so an entire review session is not linear in frame count. Metadata is capped at 32 MiB. Overlap warnings cost up to O(B²) for one frame's at-most-256 boxes. Validation matching costs O(P×G) per frame, with at most 100 predictions and 256 reviewed boxes. Images are processed sequentially. These are bounds, not measured performance gains.

## Still to validate with real footage

1. Manually box a small independent pilot, then measure Qwen localization errors, omitted cups and human correction effort. Structured JSON alone cannot establish box quality.
2. Review at least two independent recordings and run a short real fine-tuning job. Measure CPU time, memory, cancellation behavior and compatibility with the installed training environment.
3. Evaluate the candidate on separate held-out recordings before considering activation. A single short video and model-assisted labels cannot establish generalization gains.

Normal PostgreSQL/browser integration, real inference, real checkpoint loading, real training and independent-model accuracy remain unverified for this change. Candidate activation is outside this delivery.
