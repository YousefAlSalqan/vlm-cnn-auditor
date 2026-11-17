# Evaluation, testing, and business KPIs

> **Proposed evaluation protocol.** Independent reference labels, the evaluation tables and business KPI calculations described below are not implemented in the app. The current regional default produces R1 suggestions without object counts. Adapt the declared sampling and rule scope before running this protocol. See [verification](V1_VERIFICATION.md) for completed checks and [future work](ROADMAP.md) for next steps.

The experiment asks whether Qwen-assisted review finds verified YOLO errors with less human effort and lower cost. Agreement between two models is not accuracy. Human reference labels are separate from model proposals and Accept/Reject clicks.

All thresholds below are proposed pilot settings. There are no measurements or promised savings in this document.

## 1. Four scenarios with one physical cup

Record four separate 15–30 second clips. Use the same single physical cup. Never introduce a second physical cup to manufacture duplicate detections.

| Scenario | Recording procedure | Reference and rules exercised |
|---|---|---|
| S1: ordinary view | Cup on a table, steady phone, ordinary lighting, small changes in distance and angle | Visible count normally one. Establish correct-detection/no-issue behavior and observe any real misses or duplicates |
| S2: occlusion | Partly cover the same cup, briefly fully cover it, then reveal it | Count one only when the cup is visually identifiable. Fully hidden means visible count zero. Ambiguous partial views are ungradable, not automatically R1 errors |
| S3: empty scene | Remove the same cup from view and later return it. Optionally include a flat cup photograph while the physical cup is outside the view | Count zero while no physical cup is visible. A detected photograph or reflection is not another physical cup. Challenges R2 and R4 |
| S4: low light and blur | Film the same cup in reduced light with a short controlled camera movement, then stabilize | Per-frame visible count is zero, one, or unknown from evidence. Challenges R1/R4 and legitimate abstention. Do not infer visibility only from knowing where the cup was placed |

R3 can occur when YOLO boxes the same cup twice. None of these scenarios guarantees an R3 event or any real detector error. Injected duplicate/false/missing detections belong in software contract tests only. Label them synthetic and exclude them from model-quality and business results.

Retain original clips, recording condition notes, orientation/codec information, and session/model/prompt versions. Keep personal footage outside Git. Use sampled canonical raw JPEGs as the exact annotation surface. One-cup scenarios establish a narrow experiment, not general CV performance.

## 2. Independent reference labels

1. Before exposing Qwen suggestions, an annotator records visible physical-cup count, visibility/ambiguity, and one native-pixel box when a cup is identifiable. Count cup instances, not handles, printed images, or reflections as additional objects. For an occluded cup, draw the visible extent, document that policy, and apply it consistently.
2. Freeze the annotation revision and protocol version. A second annotator adjudicates ambiguous frames and a preselected 20% random subset. If only the owner is available, record the single-annotator limitation. Do not call those labels independent multi-rater truth.
3. Match predictions to human boxes using one-to-one highest-IoU matching at IoU `>=0.50`, with deterministic ties by detection ordinal. With at most one reference cup, this is a single scan of predictions. Preserve detector outputs and annotate disagreements rather than changing the reference to agree with a model.
4. Adjudicate unmatched detections visually: extra boxes on an already matched cup are duplicates, other unmatched predictions are ordinary false positives. An unmatched visible cup is a false negative. Count mismatch is `detector_count != visible_count`. Exclude ungradable frames from quality rates and report their count.
5. Independently adjudicate whether each VLM finding refers to an actual error. Use rule and referenced detection/object to link it to an error. Freeze those links before calculating quality. Accept/Reject actions can support workflow analysis but do not replace this reference.

For the one-cup pilot, one matched prediction is a TP. A duplicate is a subtype of FP, counted once in detector FP totals. A poorly localized or cropped prediction can be both an unmatched FP and leave a reference FN under the IoU protocol. Report that case separately rather than claiming two distinct physical incidents. R4 is a frame-level diagnostic and is not added again to the atomic error total.

Persist frozen boxes/counts in `human_annotations`, independent error labels and finding-to-error links in `reference_adjudications`, and completed timed baseline/assisted observations in `review_trials`. DATA_MODEL defines their strict payloads, revision history, and export projection. T9 collects manual error claims in both conditions and independently verifies them after the timed trial. Acceptance can prefill an assisted claim but never supplies its correctness. Metric inputs are frame bundles, an explicitly selected reference set, adjudicated trial observations, and declared cost inputs. Reports identify all of those revisions.

## 3. Dataset and sampling policy

Begin with the four clips for feasibility and failure discovery. Do not make aggregate business claims from them. A proposed next pilot uses 24 independently recorded clips, six per scenario. Reserve eight clips for prompt/settings development and freeze the remaining sixteen for evaluation, stratified by scenario. Split by complete recording, never adjacent frames. If recordings share the same session/camera setup, treat that dependence as a limitation.

Annotate **all sampled frames** in the held-out sessions, including those Qwen did not select and those where it returned no issue or failed. This exposes missed errors. Report selection coverage and conditional quality on selected frames separately from the whole sampled-frame pipeline. Unsampled video intervals remain outside measured recall. Review an additional random set of unsampled timestamps before claiming video-wide coverage.

Record all exclusions, failures, denominator counts, and per-scenario results. Adjacent frames are correlated. Use clip-level resampling for uncertainty estimates only when there are enough independent clips. Report precision/recall with numerators, denominators, and intervals where defensible. A zero-error or zero-flag pilot cannot estimate the corresponding rate. Expanding the dataset depends on observed error counts and the precision needed for a decision, not an assertion that 24 clips proves generalization.

## 4. Metric definitions

Define an **atomic error** as a human-confirmed missed visible cup or unmatched prediction in a specific frame. Keep a stable evaluation key derived from session/frame, reference object or detection, and error category. Deduplicate repeated model suggestions for the same error. R4 corroboration never increases the error count again. Report temporal episodes separately if consecutive frames show the same issue, rather than presenting all frames as independent incidents.

| Metric | Exact numerator / denominator or formula | Interpretation and limits |
|---|---|---|
| Verified YOLO errors found | Number of unique atomic errors discovered and adjudicated within the review workflow's budget | Report FN, ordinary FP, duplicate, and localization-related cases separately. R4 counts are secondary |
| Detector precision / recall | `TP/(TP+FP)` and `TP/(TP+FN)` against frozen human boxes | Duplicates are FP. Not a measure of VLM quality |
| VLM finding precision | Correct adjudicated findings / all adjudicated findings, after duplicate-proposal rejection | Rule-level and overall. Unadjudicated findings are reported, not silently treated as correct |
| VLM error recall, selected | Unique atomic errors identified by a correct finding / all reference atomic errors on selected, gradable frames | Include failed/abstaining selected frames in the denominator |
| VLM error recall, pipeline | Same numerator / all reference atomic errors on all sampled, gradable frames | Unselected frames count as unflagged. Excludes unsampled intervals |
| R4 precision / recall | Correct flagged count-mismatch frames / flagged gradable frames, and correct flagged count-mismatch frames / all gradable mismatch frames | Separate frame-level metric. Do not add R4 to atomic-error recall |
| Count error | Mean `abs(detector_count - human_visible_count)` on gradable sampled frames | Compare the Qwen estimate only on its available selected frames, with coverage stated |
| Unsafe proposal rate | Incorrect adjudicated proposals / all adjudicated proposals | Measures hallucinated correction risk before a human acts |
| False-correction rate | Incorrect accepted actionable proposals / adjudicated accepted actionable proposals | V1 records acceptance only, so call it accepted-proposal risk. No changes are automatically applied |
| Human review time saved | `(baseline_active_seconds - assisted_active_seconds) / baseline_active_seconds` | Can be negative. Compare matched quality/coverage and report waiting separately |
| Cost per verified error | Fully allocated workflow cost / unique verified errors found under that workflow | Undefined if no verified errors. Never substitute zero |
| Throughput and latency | Sampled frames persisted / end-to-end wall seconds. Report YOLO and VLM p50/p95 wall latency, cold and warm, plus completed pairs/minute | Include processing failures and retries in total time. Report counts with percentiles |
| Resource use | Peak process RSS, CPU utilization, GPU memory/utilization, and disk bytes/session | Machine, runtime, sampling interval, and measurement method accompany results |

Also report image/JSON synchronization success rate, Qwen selection coverage `selected/N`, schema/context validation failures per attempt, successful reviews per selected frame, abstention rate, retry rate, and export hash-verification success. A successful HTTP request or JSON parse is not a verified review.

If an outcome denominator is zero, return null with a reason and numerator/denominator counts. Do not produce NaN/Infinity, “100% accuracy,” or a zero-cost success. Attempt failures retain a distinct status and remain visible in operational metrics.

## 5. Baseline, model comparison, and timing

The baseline is human review of the same uniformly selected frame budget with YOLO overlays and metadata, without VLM findings. The assisted condition adds Qwen's validated suggestions and may prioritize flagged frames. Both conditions have the same candidate frames and a predeclared active review-time budget, initially five minutes per clip. If the baseline finishes earlier, record its actual time. Keep offline ground-truth annotation time separate from timed product review.

Use matched clips and counterbalanced condition order across reviewers. Avoid showing the same reviewer the same clip in both conditions when possible, because memory biases the result. With one reviewer and four clips, report a usability observation, not a controlled time-saving estimate. Record active interaction time, loading/inference wait, review completion, and errors found before the time limit. A finding is valuable only after human verification.

A detector comparison creates a separate session per weight/settings combination but reuses the immutable original and exact source-frame sampling specification. A Qwen comparison freezes those YOLO predictions, image transforms, prompt/rules, selection, generation budget, and references. Change one factor at a time, record tag **and** digest, use fresh attempts with explicit experiment identity, and report paired per-clip results. Do not tune on the held-out clips or treat Qwen's answers as detector labels.

Report cold model load separately from warm inference. Timers use a monotonic clock around the full call, including image preprocessing, request, validation, and persistence, with substage timings. Ollama's reported durations are additional diagnostics, not substitutes for end-to-end measurements. Failed, timed-out, or late-completing requests consume real resources and remain in cost totals.

## 6. Cost accounting and proposed business gates

For each condition, calculate:

```text
labor_cost = active_reviewer_seconds / 3600 * stated_hourly_rate
compute_cost = measured_cpu_hours * cpu_rate + measured_gpu_hours * gpu_rate
energy_cost = measured_wall_kWh * stated_tariff                 # only if measured
workflow_cost = labor_cost + compute_cost + remote_charges + storage_allocation
cost_per_verified_error = workflow_cost / unique_verified_errors_found
```

For an owned local machine, show observed usage separately from the owner's declared amortization/rental assumptions. Include energy only if separately measured and not already covered by a compute rate. Never derive wall power from GPU TDP or utilization alone. Setup/download costs and reference-label collection are separate line items, with any amortization period stated. Do not double-charge CPU/GPU, bundled power, or shared baseline extraction cost. Retain inputs and formulas so changing an hourly rate does not require another inference run.

No cloud/GPU prices are assumed in this plan. A numerical sensitivity example can use declared hypothetical rates, but must never be labeled a measured saving. Downstream operational loss avoided is outside the experiment unless independently measured.

Proposed pilot gates are:

- **Evidence integrity:** 100% of exported complete frames pass identity/revision/checksum checks, and no proposal changes original predictions or media.
- **Review usefulness:** Target finding precision of at least 80%, with rule-level counts and uncertainty. This is a proposed threshold, not achieved quality.
- **Comparable coverage:** Assisted atomic-error discovery must remain within five percentage points of the baseline's measured discovery recall under the same budget. If either denominator is too small, this gate is inconclusive.
- **Business value:** Target at least 20% lower active review time and lower cost per verified error at comparable coverage. Report negative results and waiting time.
- **Usability/resources:** Attempt the under-10-minute warm local target from the implementation plan without OOM, with visible cancellation and recovery. CPU-only latency may fail this target while functional tests still pass.

The owner reviews these thresholds after feasibility. A small or error-free sample produces an inconclusive business result, never a positive ROI claim.

## 7. Verification matrix

| Test level | Cloud / CI with fakes and synthetic media | User's local phone / runtime |
|---|---|---|
| Identity and decoding | CFR/VFR, repeated millisecond values, nonzero PTS, rotation, timestamp gaps, malformed/oversize clips generated at runtime | Actual H.264 MP4/MOV and optional HEVC/HDR, decoder alignment and image orientation |
| Detector | Cup filter, zero boxes, finite/bounded xyxy, incompatible label map, stable IDs, injected duplicate/missing/false boxes | Real weights, cup detection behavior, CPU/CUDA compatibility and memory |
| Reviewer | Exact two-image request order, schema, raw invalid output, unknown/cross-frame IDs, R1–R4, no-issue/abstention, timeout/OOM/cancel, two-attempt ceiling | Qwen actually reads both images and follows schema. Real hallucinations and latency |
| Persistence | Real temporary PostgreSQL, migrations, FKs, append-only histories, rollback, duplicate click, snapshot revision race | Local DB permissions, disk space, restart, filesystem behavior and backup restore |
| UI and export | Fake pipeline end-to-end, stale navigation, accessibility checks, current decision reload, ZIP allowlist/hash/identity/revision verification, crash injection | Side-by-side usability with personal footage, real progress/wait, phone capture and resource measurement |

Stub CI cannot establish model accuracy, genuine YOLO errors, VRAM requirements, actual phone timestamp reliability, energy savings, or business value. No live model endpoint, personal video, or weight download is required for the default CI suite. [DEVELOPMENT_TASKS.md](DEVELOPMENT_TASKS.md) assigns the checks and commands.
