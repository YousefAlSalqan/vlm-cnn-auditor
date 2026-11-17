# Verification record

## Regional default, 2026-11-17

The integrated regional-review version at commit `7f5c6b4628a0be400c3f6f8aaf60ccde4a913741` passed **86 tests** in [GitHub Actions run 44128518913](https://github.com/YousefAlSalqan/vlm-cnn-auditor/actions/runs/44128518913). The run used synthetic footage, fake model responses, a disposable PostgreSQL service and Chromium. Ruff lint and formatting also passed. These checks do not measure real-model accuracy.

Local checks passed 52 tests and skipped 34 database/browser integration tests to avoid opening the owner's database. The full integration cases ran in GitHub's disposable service. A separate read-only browser fixture passed at 1280px and 375px with synchronized observations, loaded image pairs, no horizontal overflow and no page/console errors.

The 27-second test video contained 810 source frames. The production sampler selected 162 matching YOLO/Qwen keys at stride five. Two real-Qwen integration cases, at 8 and 13 seconds, produced validated R1 suggestions and passed reconstruction from saved raw image evidence. This was not a 162-frame live inference run. The model digest, generation settings, weights, private configuration and 82 existing session files were preserved. No local database connection or schema migration was performed for this integration work.

New checks cover regional schema and identity, source-image replay and tamper rejection, tiny-frame bounds, matched stride selection beyond the legacy 60-frame limit, 90-frame persistence/export with mocked regional calls, interrupted-analysis partial exports and strict complete-export selection. See [the experiment record](PROMPT_LOOP_RESULTS.md) for observed leads and ambiguous results.

Independent reference labels, general precision/recall, a complete live run of the new default and measured review-time/business gains remain unverified. See [current architecture](ARCHITECTURE.md) for today's limits and [future work](ROADMAP.md) for the proposed evaluation.

## Original local V1 record, 2026-11-16

The rest of this document is historical evidence for the first implementation. Its sampling limits, branch status and untouched-document hashes describe that date, not the current release. No database or runtime checks described below were repeated as part of the prospect-documentation update.

The local first version passed **65 tests in 62.68 seconds** on Windows with Python 3.11.18, PostgreSQL 16.16 and FFmpeg/FFprobe 8.1.1. Both `AUDITOR_TEST_DATABASE_URL` and `AUDITOR_BROWSER_TESTS=1` were set. There were no skipped tests. Gradio emitted 30 upstream Pandas deprecation warnings during browser execution.

## Tested behavior

The suite covers real PostgreSQL migrations and scoped foreign keys, strict Qwen responses, CFR/VFR timestamps, rotation, immutable storage, staged recovery, durable model attempts, absolute deadlines, proxy isolation, frozen model/runtime settings, append-only decisions and ZIP integrity. A real Chromium test exercises upload, separate Start, delayed multi-finding navigation, stale-action rejection, note/history persistence, recovery reset, 375px layout and ZIP download.

Separate live checks used the actual local YOLO11n checkpoint and `qwen3-vl:2b-instruct`. A generated 15-second clip produced three sampled pairs, one validated Qwen review and a verified ZIP with the default 120-second deadline. An earlier two-color image check confirmed that the local runtime consumed both distinct images. Neither check measures cup-detection accuracy.

The actual app startup was checked on loopback port 7860 at desktop and mobile widths. Model readiness was confirmed without adding synthetic sessions to the normal app database. The launcher started portable Ollama, restarted its owned Ollama process while leaving the app available, and started the existing PostgreSQL data directory after a clean shutdown. No global service installation was needed.

## Independent review loop

Core and browser reviewers repeatedly reproduced defects, added checks, then reviewed the fixes. The final core review reported zero remaining actionable findings with 64 core tests passing. The final browser review reported zero actionable findings in its exercised scope. It covered normal desktop/mobile flow, cancellation/resume, session refresh, UTC timestamps, eight rapid note-entry trials, delayed finding changes, separate finding histories, original-video playback and independent hashes for every exported file. A setup reviewer also reported zero actionable findings.

The fixes include pre-dispatch request records, uncertain-runtime recovery, bounded sampling work, disabled environment proxies for footage, strict export scope/provenance/reference checks, preserved inference settings, revision-based history ordering and click-time decision context. Reviewers checked the supplied baseline and used synthetic media and isolated test schemas. They did not modify production code.

Ruff lint, Ruff formatting, bytecode compilation, dependency compatibility for 95 installed packages, wheel contents/imports and local documentation links passed. The wheel includes the stylesheet and prompt, with no footage, credentials or model weights.

## Baseline and scope

The starting repository contained documentation and ignore rules, with no application to preserve. `AGENTS.md`, `PROJECT_BRIEF.md` and `PLAN_REQUEST.md` retain their original hashes. Evaluation and risk plans remain unchanged. Three original plans have explicit implementation-status amendments. Existing media/secret ignore rules remain in place.

Authorized additions are the local application, eight-table PostgreSQL schema, migrations, model adapters, tests, CI, dependency snapshot, launcher and current guide. The implementation remains separate from `main`. No PR is required. Predictions are not corrected or retrained automatically. The three independent-evaluation tables remain deferred.

Decode/sampling overhead is linear in source-frame count, with bounded frame buffers and at most 60 saved pairs and 12 Qwen selections. Reference validation uses sets. Uploads and ZIP file bytes are streamed. Loaded per-frame export JSON is capped at 16 MiB. These bounds do not predict model inference time or memory use inside Ollama.

## Limits of this evidence

Passing tests and clean reviews do not prove preservation for every future input. Personal phone footage, physical devices, independent labels, cup-error precision/recall, full human review timing, business savings and prolonged resource use remain unverified. Another machine still needs its own setup validation. The GitHub Actions workflow uses synthetic models and a disposable PostgreSQL service. Its remote result is separate from the local results recorded here.

Start with the short manual test in [V1_GUIDE.md](V1_GUIDE.md#manual-test). Keep footage and credentials private.
