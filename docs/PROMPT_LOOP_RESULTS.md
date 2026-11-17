# Prompt iterations and experiment results

The experiment tested a practical question: can the existing Qwen model produce useful leads when YOLO misses cups? The work changed the question, image views and response contract. It did not train either model. This record preserves the failures and crop-only explanations alongside the useful examples.

## Iteration history

| Stage | Change | What it taught us |
| --- | --- | --- |
| Initial application | [Full-frame review prompt](../auditor/prompts/cup_review_v1.txt), raw/overlay images and a structured R1–R4 response with frame/detection references. | A working evidence pipeline did not guarantee additional detections or reliable model-generated references. |
| Regional audit | [Region-audit prompt](../experiments/prompts/region_audit_v1.txt) explored asking the original review question on smaller views. | Smaller views were worth testing, but the richer review contract still required careful validation. |
| Cup presence V1 | [Two-field regional prompt](../experiments/prompts/cup_presence_v1.txt) asked whether a cup was visible and why. | Clear mugs became review leads, while storage jars and thermoses exposed false-positive descriptions. |
| Cup presence V2 | [Stricter prompt](../experiments/prompts/cup_presence_v2.txt) added visible-shape evidence, exclusions, hanging mugs and an explicit `unclear` answer. | Retained useful mug examples and removed two selected container false positives. An additional glass-cup lead remained ambiguous. |
| Crop controls | Run the same YOLO weights on the identical encoded images given to Qwen. | Hanging-mug and red-handled-mug examples still had zero YOLO cups. The cup-on-saucer example was recovered by cropped YOLO, so it is not evidence of a uniquely VLM benefit. |
| Default integration | Use the exact tested V2 prompt, let the host supply identity, retain raw receipts, and align both models on every fifth source frame. | Structured suggestions now reach the app and verified exports. Full-video accuracy and review-time improvement remain unmeasured. |

The two-field observation and the application's normalized review are different contracts. The host adds trusted identity and validates derivation rather than asking Qwen to invent identifiers or repairing invalid responses. The current [architecture](ARCHITECTURE.md) documents this boundary.

## Experiment scope, 2026-11-17

**App integration update:** The owner subsequently authorized enabling this regional flow. New app sessions now use `auditor/prompts/cup_presence_v2.txt` by default, with YOLO and Qwen on source indices 0, 5, 10 and so on. The experiment record below describes the evidence used to choose that prompt. See [V1_GUIDE.md](V1_GUIDE.md) for current behavior. Model weights and database schema remain unchanged.

The regional inspection experiment produced useful missed-cup review leads on the owner's existing 27-second video. Clear examples include three hanging white/blue-rim mugs, a white cup on a saucer, and a white mug with a red handle. These are assistant visual checks, not independently adjudicated training labels or a general accuracy benchmark.

During the experiment, the installed Qwen3-VL 2B model, its generation options, YOLO weights, saved session and database were unchanged. At that stage the experiment was separate from the app. Subsequent integration changes the default reviewer for new sessions and does not modify existing YOLO predictions.

## What changed

The experiment asks a smaller question: does this raw image region contain a cup, and what does it look like? Qwen returns the separate `CupPresence` diagnostic schema with `cup_present` and `evidence`. It does not generate session UUIDs or detection references.

Fixed horizontal thirds cover each tested frame independently of YOLO's boxes. The experiment retains every result. A positive region becomes an R1 review draft only when no saved YOLO box intersects it. The host adds the original frame identity, rechecks source hashes and crop/image correspondence, and validates the draft against the unchanged `ReviewResponse` contract. The original model response is preserved separately.

These are host-built drafts derived from Qwen observations and geometry. They are not raw Qwen responses that passed the full V1 schema, and they are not accepted labels. Human review is required. Zero overlap is a coarse review heuristic and does not prove precise object localization. Regions touching existing boxes are conservatively skipped, including ones that might contain additional missed cups.

## Observed results

There were 39 Qwen calls: 37 raw-image diagnostic calls and two full-frame V1-schema follow-ups. All 37 diagnostic outputs passed the diagnostic schema. Both full-frame follow-ups failed the original schema because R1 incorrectly contained detection references. Those failures were retained and were not repaired.

| Example | Saved YOLO result in the inspected region | Qwen observation | Assessment |
|---|---|---|---|
| 13 seconds, upper third | No intersecting box | Three white mugs with blue stripes hanging from a shelf | Clear additional review lead. |
| 6 seconds, lower third | No intersecting box | White cup with a dark rim and handle, sitting on a saucer | Clear additional review lead. |
| 8 seconds, lower third | No intersecting box | White mug with a red handle on the left | Clear additional review lead on a frame unused for V2 tuning. |
| 24 seconds, middle/lower thirds | No intersecting box | Cup on a saucer, and a white/red-handled mug | Supported repeat views, likely the same objects seen earlier. Do not count them as new physical discoveries. |

Five additional YOLO control inferences used the identical encoded crop images supplied to Qwen, the same existing YOLO weights and the original detector settings. YOLO returned zero cups for the 13-second hanging-mug crop and zero for the 8-second red-handled-mug crop. Those are the strongest examples of additional Qwen review value. YOLO did recover the cup on a saucer from the 6-second crop, so that example also demonstrates a tiling benefit without needing Qwen. YOLO returned zero on the selected storage-container and dark-wall controls. Original full-frame detections were not replaced.

The stricter V2 prompt kept the known positive cup examples and changed two chosen storage-jar/thermos controls from positive to non-positive. The dark empty-region control remained `unclear`. Some negative answers still relied on blur and should have been more cautious. A `no` answer is not proof that a cup is absent.

V2 was frozen before testing four additional frames at 4, 8, 12 and 24 seconds. Their twelve regions produced four positive zero-overlap candidates. Three were visually supported and one was ambiguous because the specific glass-cup description could not be confirmed securely. Across development and those additional frames, V2 produced seven R1 drafts: six visually supported regional leads and one ambiguous lead. These are proposal counts, not distinct object counts or a precision estimate from independent human labels.

No complete ground-truth annotations were collected. The new frames come from the same video and share objects and backgrounds. Recall, general precision, automatic-label accuracy, tracking improvement and business savings remain unmeasured.

## Reproducing a diagnostic

Use the existing configured environment from the repository root. `run_probe` reads the saved session and does not connect to PostgreSQL. The output folder must be new, inside the repository and outside the source session. Replace the example session path with your saved session folder. Crop coordinates must match that frame's native dimensions.

```python
from pathlib import Path
from experiments.qwen_yolo_loop import make_review_draft, run_probe

session = Path("path/to/session_UUID")
prompt = Path("experiments/prompts/cup_presence_v2.txt").read_text()
probe = run_probe(
    session,
    "frame_000000390",
    prompt,
    Path(".local-runtime/a-new-diagnostic-run"),
    crop_xyxy=(0, 0, 576, 341),
)
draft = make_review_draft(session, probe)
```

The saved record distinguishes `diagnostic_status` from the reused adapter's V1 `status`. A valid diagnostic can have V1 `status="invalid_json"` because the diagnostic intentionally uses a different schema. `response` remains null in that case. Do not count a diagnostic as an accepted application review or silently substitute it into the existing pipeline.

The runner rejects output-directory reuse, preserves raw responses, stops a batch on uncertain requests and uses the existing absolute request deadline. It processes one model request at a time. It adds no dependency, database object, model download or application startup hook.

The experiment and core response/storage checks finished with 27 passing tests. Review found and fixed an image-size verification issue in the experiment helper, with a failing-then-passing regression test for 1920-by-1080 views. It did not affect this video's drafts. Private evidence contains exact prompts, hashes, source transforms, all responses, visual assessments, same-crop YOLO controls and proof images under the ignored `.local-runtime/qwen-yolo-loop` directory. Personal footage and actual session identifiers are not published in this document.
