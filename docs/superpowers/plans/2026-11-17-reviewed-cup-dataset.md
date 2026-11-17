# Reviewed cup dataset and opt-in fine-tuning implementation plan

> **For implementation after plan review:** Work inline in this isolated branch. Do not delegate to subagents in this side conversation. Checkboxes describe future work, not completed features.

**Status:** Proposed plan. Only this document is written now. No application changes, database connections, inference, dataset generation, training, service installation or deployment have occurred.

**Goal:** Build a consolidated cup-annotation draft from saved YOLO/Qwen evidence, let a human correct and approve it, then automatically fine-tune from existing weights when the workflow is enabled.

**Architecture:** A deterministic dataset service consumes verified session exports, keeps model proposals separate from human annotations, and freezes approved YOLO-format releases on local disk. A small review panel in the existing Gradio app provides box editing. One local training job uses the frozen release and an existing checkpoint, producing a separate cup-model candidate.

**Tech stack:** Existing Python, Gradio, Pydantic, OpenCV, HTTPX, FileLock and Ultralytics. Reuse existing model weights and Ollama. No separate annotation service, cloud agent, third model or database migration in the first version.

**Spec:** The confirmed requirements and control behavior below are the request basis. Implementation requires the owner's review of this plan first.

## Confirmed requirements and selected defaults

| Requirement | Decision |
| --- | --- |
| First domain | Cups from the existing test videos. HVAC is outside this branch. |
| Consolidation | Create one reviewable dataset from saved YOLO boxes, Qwen hints and human corrections. |
| Human boundary | Review before training. No model-generated label is automatically approved. |
| Previous learning | Initialize training with existing weights and retain earlier approved cup training examples in later dataset versions. |
| Original object types | The new model targets cups only. Keeping all original COCO classes is not required. |
| Box entry | Support both automatic proposals and manual drawing/correction. |
| Review location | The owner delegated the easiest option. Use the existing app to avoid another annotation tool installation. |

The two controls are:

1. **Enable dataset building and fine-tuning:** defaults OFF on app startup. OFF means no dataset jobs, localization calls or training are started. Normal Save/Start/review/export behavior remains intact.
2. **Automatically propose missing boxes:** defaults ON within the enabled workflow. It is disabled while the first control is OFF. Turning it OFF leaves YOLO draft boxes and Qwen hints available for manual annotation. Manual editing remains available in either proposal mode.

The approval button should say **Approve dataset and fine-tune**. It approves a specific dataset revision and queues training without a second training button. Failed readiness checks explain what is missing. They do not skip human review. Disabling the workflow prevents queued work from starting and requests cancellation of its running job at the next safe boundary. It does not terminate shared model services.

## Current facts that shape the design

- [contracts.py](../../../auditor/contracts.py) and [models.py](../../../auditor/models.py) currently use canonical COCO cup class 41 and native-pixel `xyxy` boxes.
- [regional.py](../../../auditor/regional.py) currently produces regional missed-cup descriptions, not individual object boxes. A region containing three mugs is not a box for one mug.
- [evidence.py](../../../auditor/evidence.py) stores decisions about findings. Accepting a finding does not create a complete, corrected frame annotation.
- [export.py](../../../auditor/export.py) already verifies exported frame identity, hashes and model evidence. Consume those exports as frozen inputs instead of modifying prediction tables.
- A one-class training dataset uses class 0 for `cup`. The current model adapter requires class 41. A new checkpoint must not be automatically substituted into the existing adapter.

## Global constraints

- Keep the main chat's checkout and branch untouched. All planned edits belong to `codex/reviewed-dataset-builder` in this worktree.
- Preserve the existing database schema and original media, YOLO predictions, raw Qwen responses, finding decisions and export contracts. Dataset annotations are a separate file-backed record.
- Do not change the existing every-fifth-frame analysis cadence or default audit prompt. Optional localization calls happen only in the enabled dataset job.
- No new model weights, package installs, cloud storage or managed service are required for the first implementation. Qwen's weights remain unchanged.
- No confidence-based automatic acceptance, negative labels inferred from missing predictions, or training on unreviewed labels.
- No automatic replacement of the running detector. Save each training result as a separate candidate with its dataset and base-model hashes.
- No real database/model/training operations are authorized by this planning document.

## Proposed workflow

1. Select completed sessions or supply their existing ZIP exports. Verify the exports, freeze their identities and build a draft from raw frames.
2. Copy YOLO boxes as unapproved candidates. Attach Qwen evidence to the matching session/frame and highlight possible misses or disagreements.
3. If automatic proposals are enabled, run a separate localization request on the raw views for Qwen-only leads. Validate the returned boxes and their transforms. Otherwise leave those leads for manual boxing.
4. The reviewer adds, removes or corrects boxes, or explicitly excludes a frame. Mark a frame complete only after checking all visible cups. A reviewed empty frame is a deliberate negative example.
5. Freeze a versioned dataset containing the approved annotations and prior approved training examples, validate its split and label format, then queue fine-tuning from the displayed base checkpoint.

The script can check identity, geometry, class mapping and duplication. It can prioritize disagreements. It cannot establish which model is correct without reference evidence. YOLO scores and Qwen wording are not comparable confidence scales.

## Automatic and manual boxes

Use a dataset-specific localization prompt with the existing Qwen model. Require JSON with an array of at most 16 cup boxes in an explicitly declared `0..1000` view-relative coordinate system. The host supplies IDs and maps each box back through the recorded resize/crop transform. Reject malformed, reversed, zero-area or out-of-range boxes. Do not guess units or silently clamp coordinates.

Qwen publishes a [2D-grounding cookbook](https://github.com/QwenLM/Qwen3-VL/blob/main/cookbooks/2d_grounding.ipynb). This supports trying localization, not assuming that the installed 2B model produces accurate boxes. First benchmark it on a small, manually boxed set. Save every localization response and failure separately from the existing audit review. A failed or empty proposal leaves the lead unresolved for manual review.

Reuse a raw image/overlay pair in the new dataset panel. For the smallest editor, select a box, edit its four native coordinates, delete it, or add one by selecting two corners on the image. Confirm coordinate events and resized-display behavior against the pinned Gradio build before using them. Render edits in a separate overlay and never write onto the source JPEG.

Always show proposal origin, whether a box was changed, the Qwen hint and the raw image. High-overlap boxes are a possible-duplicate warning, not proof that two cups are one. Exact duplicate records can be collapsed with provenance retained. A rejected Qwen hint must not delete a YOLO box automatically.

## Dataset storage and approval

Use a dedicated subtree beneath `DATA_ROOT/datasets/<dataset_id>/`:

```text
source_manifest.json      # verified export identities and source hashes
draft.json               # proposals, frame keys, origins and unresolved hints
revisions/               # append-only human edits and frame-completion decisions
releases/<version>/      # frozen approved manifest, raw images, labels and data.yaml
jobs/<job_id>/           # base checkpoint hash, training config, logs and new weights
```

Frame records retain `session_id`, `frame_id`, `timestamp_ms`, source index, source-video hash, raw-image hash, dimensions, original proposal IDs and annotation revision. Human boxes use native-pixel `xyxy` and canonical class 41. The exporter alone converts to training class 0 and normalized YOLO `xywh`. For a 100x100 image and box `[10,20,30,60]`, the training row is `0 0.2 0.4 0.2 0.4`.

The approved release includes only fully reviewed frames. Pending, uncertain or excluded frames cannot become empty-label negatives. Blank labels are allowed only when a human explicitly marks a frame as having no visible cups. Drawn boxes must cover all visible target objects under a documented visible-extent policy.

Approval binds the revision, frame hashes, box set, split assignment, base checkpoint hash and displayed training configuration. Any edit invalidates that draft's approval and creates a new revision. Already frozen releases remain unchanged. The job uses a working copy so training caches do not mutate approved release contents.

Preserve provenance and human review history in the release manifest. YOLO training text files alone do not carry this history. See the official [Ultralytics dataset format](https://docs.ultralytics.com/datasets/detect/).

## Previous data, weights and evaluation

The first run loads compatible pretrained weights from the existing YOLO checkpoint and trains a cup-only model. A changed one-class output head may need initialization rather than direct transfer. This is not training from scratch and not concatenation of old and new weight files. Subsequent runs can start from the latest evaluated cup checkpoint, displayed before approval, while retaining the earlier files.

Later releases combine previous approved cup training data with newly approved data. A source frame's latest reviewed annotation supersedes its earlier annotation, without silently resolving conflicting labels. Keep source groups and exact duplicate images in one split. Group repeated uploads by original-video hash, not just session UUID.

Assign train/validation groups by whole source video and persist those assignments. Test data, if reserved, stays separate and is never recycled into training. Prior validation/test examples must not enter a later training union. A single video is insufficient for independent video-level training and validation. Show a readiness message until another suitable labeled source exists rather than splitting neighboring frames and presenting misleading metrics.

Use a bounded local CPU training run for the initial supported environment: 10 epochs, batch size 2, image size 640 and zero data-loader workers. Display and freeze these values before approval. Do not add a CUDA installation or cloud fallback. Actual runtime and usefulness remain to be measured.

Record base/new checkpoint hashes, class mapping, dataset hash, package versions, training config and evaluation results. Compare cup-only predictions from the base and candidate on the same held-out frames with the same declared matching/threshold policy. Report precision/recall and failure counts, including undefined metrics with reasons. No accuracy gain is promised.

The candidate's internal cup class 0 is different from the app's canonical class 41. Keep that mapping in metadata. Activation and a compatible adapter change are outside this first delivery. Do not silently rewrite old detections or change `YOLO_WEIGHTS`.

## Review focus

| Failure | Required behavior |
| --- | --- |
| Feature OFF, stale callbacks or restart | Zero localization/training jobs. Interrupted jobs never restart merely because the app launches. |
| Qwen identifies several cups but has no valid individual boxes | Keep separate unresolved hints. Never use the entire region as an object annotation. |
| Missing annotations mistaken for empty images | Only explicit complete, reviewed negatives receive empty training labels. |
| Later edits, duplicated sources or repeated approval clicks | Invalidate stale approvals, prevent split leakage and queue at most one job for an identical approved release/configuration. |
| Training interruption, resource contention or new class mapping | Preserve old weights and evidence. Serialize training with analysis using the existing process lock. Never automatically activate an incompatible checkpoint. |

## Proposed code boundary

| File | Responsibility |
| --- | --- |
| `auditor/datasets.py` | Strict draft/revision/release records, verified input ingestion, review persistence, split checks and YOLO export. |
| `auditor/localization.py`, `auditor/prompts/cup_boxes_v1.txt` | Optional bounded Qwen localization requests and host-owned box transforms. Existing prompts remain unchanged. |
| `auditor/training.py` | Immutable job spec, one owned worker, checkpoint initialization, training/evaluation and result artifacts. Lazy-load training libraries. |
| `auditor/dataset_ui.py` | New Gradio controls and dataset-only box editor/review interactions. |
| `auditor/app.py`, `auditor/config.py` | Mount the optional panel and default-OFF settings. Preserve the existing analysis configuration and current flows. |

No changes to `auditor/db.py`, migrations, existing prediction contracts or `auditor/models.py` are required for a training-candidate-only version. Reuse hashing, safe paths, atomic writes and locks from `storage.py`. Do not create a second pipeline or generic plugin framework.

## Implementation tasks after review

### 1. Build the draft and human annotation records

**Interfaces:** `build_draft(source_zips: list[Path], dataset_root: Path) -> DraftDataset`, `save_frame_review(dataset_id, frame_key, expected_revision, boxes, disposition) -> int`, and `freeze_release(dataset_id, expected_revision, training_spec) -> Path`. Define their strict records in `datasets.py`. Disposition is `complete`, `reviewed_negative` or `exclude`.

- [ ] Write failing tests in `tests/test_datasets.py` for source/hash mismatch, class conversion, corrupt input, duplicate frame identity, stale revisions and incomplete negatives.
- [ ] Implement draft ingestion from verified archives and separate annotation revisions. Keep automatic labels unapproved.
- [ ] Add checks that the worked normalization example matches exactly and all original source bytes remain unchanged. Run `pytest tests/test_datasets.py -q` to green.
- [ ] Commit the reviewed dataset core on this branch.

### 2. Add bounded optional box proposals

**Interface:** `propose_boxes(frame_key, raw_jpeg, source_regions, enabled: bool) -> list[BoxProposal]`. Keep raw request receipts, exact native transforms and unresolved statuses. Disabled mode returns without initializing a model client.

- [ ] Write failing tests in `tests/test_localization.py` for disabled zero-call behavior, normalized-to-native mapping, malformed units/boxes, multi-object output, output limits and uncertain timeouts.
- [ ] Implement the separate localization prompt using the installed Qwen model and the existing bounded local transport behavior. At most one attempt per selected region in a job, at most three regions per frame and no automatic retry after uncertainty.
- [ ] Verify unresolved proposals cannot be exported as labels. Run `pytest tests/test_localization.py -q` to green.
- [ ] After implementation is separately authorized, compare candidate boxes with a small manually boxed pilot. Record errors and correction effort rather than claiming grounding accuracy from valid JSON.

### 3. Add the in-app review panel and controls

**Interface:** `build_dataset_panel(settings, source_export_provider)` mounts the dataset workflow without changing existing prediction/finding decisions. Use existing source-export callbacks instead of new SQL tables.

- [ ] Add failing mocked callback/browser checks for the OFF default, child proposal default, manual corner entry, coordinate correction/deletion and frame-completion controls.
- [ ] Implement the panel with an explicit raw-image display, editable proposal list and approval bound to the displayed revision. Reuse Gradio. Do not install CVAT or another editor service.
- [ ] Test rapid frame changes, stale clicks, mobile scaling, turning proposals OFF and restarting with the master control OFF. Existing analysis controls must retain their behavior.
- [ ] Show dataset readiness and the base model/configuration before enabling **Approve dataset and fine-tune**.

### 4. Fine-tune once after approved release

**Interface:** `enqueue_training(release_path: Path, training_spec) -> job_id` and a single worker in `training.py`. A job key is the hash of the approved release, base checkpoint and training configuration. Repeated identical submissions return the existing job.

- [ ] Write failing tests in `tests/test_training.py` for missing/invalid approval, master OFF, source-split leakage, incompatible checkpoints, duplicate enqueue, pending analysis lock and interrupted jobs. Use a fake trainer, not real training in routine tests.
- [ ] Implement local training in a separate owned process or worker with the existing analysis lock, output only beneath the job directory, cancellation and original-checkpoint hash verification.
- [ ] Save a new cup-only checkpoint and held-out evaluation. Do not activate it. A failed/cancelled job leaves the previous model and dataset release usable.
- [ ] Run dataset, localization, training and browser checks, then the repository's isolated test suite. Add one separately authorized small live-training smoke check before claiming the feature works on the target laptop.

## Dependency and complexity review

No additional runtime package is planned. A separate annotation tool is unnecessary for the first two-corner/coordinate editor. CVAT and Label Studio remain alternatives if later work needs team labeling or richer geometry. Their documented [pre-annotations](https://labelstud.io/guide/predictions), [review](https://docs.cvat.ai/docs/qa-analytics/manual-qa/) and [YOLO export](https://docs.cvat.ai/docs/dataset_management/formats/format-yolo-ultralytics/) informed the separation of predictions, human edits and release export.

Index frame/proposal identities and hashes with maps/sets. Ingestion and exact deduplication are expected linear in record count plus streamed bytes. A bounded per-frame overlap comparison can be quadratic in that frame's box count, not across the full dataset. Process/copy images sequentially. Do not retain all image pixels in memory or claim a model speedup from these bounds.

## Completion boundary

The first completed feature must demonstrate default-OFF preservation, automatic/manual cup proposals, exhaustive human frame review, an immutable dataset release, and one fine-tune initialized from existing weights. All original evidence and checkpoints remain preserved. Generalization gains require additional independent footage and measurement.

This proposal is ready for owner review. Application code and runtime behavior remain unchanged at this planning stage.
