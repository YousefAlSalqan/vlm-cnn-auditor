# Data model and synchronized evidence

> **Original design reference.** The proposed tables and contracts below include unbuilt evaluation features. The actual eight-table schema, foreign keys and current JSON/storage flow are documented in [ARCHITECTURE.md](ARCHITECTURE.md). Do not read the proposal as a dump of the deployed database.

> **Regional review update, 2026-11-17:** New app sessions use source-frame stride five and the `cup_regions_v1` raw-response envelope. Existing raw-response, input-transform and runtime-metrics fields retain per-region model strings and request provenance. The normalized `ReviewResponse` and database schema are unchanged. Legacy V1 raw responses remain readable. See [V1_GUIDE.md](V1_GUIDE.md) for current behavior. No migration is required.

> **V1 implementation amendment, 2026-11-16:** The current schema implements `sessions`, `videos`, `frames`, `yolo_detections`, `vlm_reviews`, `vlm_findings`, `finding_detections` and `review_decisions`. Evaluation tables below remain deferred. Alembic migrations and [V1_GUIDE.md](V1_GUIDE.md#actual-database-architecture) describe the actual eight-table schema and recovery fields. The original design remains below for later evaluation work.

This is the proposed V1 persistence contract, not an applied schema. PostgreSQL is canonical for metadata. Media bytes live under one configured local `DATA_ROOT`. JSON is a versioned projection of committed records. [IMPLEMENTATION_PLAN.md](IMPLEMENTATION_PLAN.md) defines the processing flow.

## 1. Identity and invariants

| Field | Definition |
|---|---|
| `session_id` | Application-generated UUID. One uploaded video and one frozen analysis configuration |
| `video_id` | UUID for that original video, unique within its session |
| `source_frame_index` | Zero-based sequential decoded frame position in the selected video stream, before sampling |
| `frame_id` | `frame_` plus the source index padded to exactly nine digits. Unique within a session |
| `timestamp_ms` | Nonnegative elapsed presentation milliseconds from the first presented frame, rounded half up |
| `source_pts`, `time_base_num`, `time_base_den`, `first_pts` | Integer timing evidence retained so milliseconds can be recomputed without floating-point drift |
| Detection/review/finding/decision IDs | UUIDs assigned by the application, never by Qwen |

The frame key is `(session_id, frame_id)`. `timestamp_ms` must agree everywhere, but it is not a unique key. Keep duplicate milliseconds and repeated presentation timestamps in source order. Decreasing timestamps, missing timestamps, or probe/decode count mismatches block processing. Record whether the timestamp is an original PTS or ffprobe's best-effort timestamp. Do not label best-effort evidence as an original camera clock.

For relative ticks `q = source_pts - first_pts` and stream time base `a/b`, calculate `timestamp_ms = floor((2*q*a*1000 + b) / (2*b))` for `q>=0`, `a>0`, `b>0`. Sampling uses the rational presentation time, not the rounded milliseconds. Retain nonzero original start offsets in `first_pts`. For supported rotation metadata, record the single transform from source pixels to upright canonical pixels. All stored boxes refer to the upright canonical JPEG dimensions.

A successful frame has exactly one immutable raw JPEG and one immutable annotated JPEG, plus one current JSON projection. Rerunning an unchanged interrupted stage reuses its IDs and committed outputs. Different model/prompt/settings produce a new session rather than overwriting old predictions.

## 2. PostgreSQL tables

Use UUID, `timestamptz` in UTC, `bigint` for counters/timing ticks, `double precision` for boxes, and `jsonb` for bounded configuration or payloads that are not relational entities. Text statuses use named CHECK constraints for straightforward migrations. Required columns are NOT NULL unless explicitly described as nullable. Every insert uses bound parameters.

### Core evidence

| Table | Columns and constraints |
|---|---|
| `sessions` | `session_id` PK, `created_at`, `updated_at`, `state`, `revision bigint>=1`, frozen `config jsonb`, `config_sha256`, `pipeline_git_sha`, `schema_version=1`, `cancel_requested boolean default false`, nullable `processing_started_at/processing_finished_at`, bounded `stage_metrics jsonb`, nullable `last_error_code` and sanitized `last_error_detail`. State is one of the seven states in the implementation plan |
| `videos` | `video_id` PK, `session_id` UNIQUE FK to sessions, `original_path` UNIQUE within session, `original_sha256`, `byte_count>0`, `container`, `codec`, `stream_index>=0`, `duration_us>0`, `time_base_num>0`, `time_base_den>0`, `first_pts`, `rotation_degrees`, `timestamp_source`, `probe_version`, `decoder_version`, nullable `declared_fps_num/den`, nullable `decoded_frame_count`. Add UNIQUE `(video_id,session_id)` for the scoped frame FK |
| `frames` | Composite PK `(session_id,frame_id)`, `video_id`, `source_frame_index>=0`, `source_pts`, `timestamp_ms>=0`, `width>0`, `height>0`, `raw_path`, `annotated_path`, `json_path`, raw/annotated SHA-256 values, `media_state` (`pending`, `ready`, `failed`, `corrupt`), `vlm_selected boolean`, `vlm_skip_reason` nullable, bounded `timings jsonb`, `revision>=1`, `snapshot_revision>=0 default 0`, nullable `snapshot_sha256`, nullable `error_code`. FK `(video_id,session_id)` to videos. UNIQUE `(session_id,source_frame_index)` and each artifact path within session |
| `yolo_detections` | `detection_id` PK, `session_id`, `frame_id`, `ordinal>=0`, `class_id=41`, `label='cup'`, `score` in `[0,1]`, `x1,y1,x2,y2` finite with `0<=x1<x2` and `0<=y1<y2`. Composite frame FK, UNIQUE `(session_id,frame_id,ordinal)`, UNIQUE `(detection_id,session_id,frame_id)` |

The application checks `frame_id` against `source_frame_index`, timestamp arithmetic against the video record, and boxes against parent width/height inside the insert transaction. SQL CHECKs cannot reference parent rows. Keep parent identity, dimensions, and timing evidence immutable after frame creation. Test both structural database constraints and contextual Pydantic validation. No database claim is made that a syntactically valid box contains a cup.

Raw/annotated paths and hashes may be temporarily null while `media_state=pending` or `failed`. A database CHECK requires all four values when `media_state=ready`. The JSON path is deterministic at insertion. A snapshot is current only if `snapshot_revision == revision` and its hash matches. Snapshot bookkeeping does not increment content revision.

### Model proposals and human records

| Table | Columns and constraints |
|---|---|
| `vlm_reviews` | `review_id` PK, scoped frame FK, `attempt_no` in `{1,2}`, `status`, nullable `outcome`, nullable `needs_review`, nullable `estimated_count` in `[0,100]`, `model_tag`, `model_digest`, `runtime_version`, `prompt_version`, `prompt_sha256`, `rule_version`, `response_schema_version`, `request_sha256`, `input_transform jsonb`, `started_at`, nullable `completed_at`, nullable `wall_ms>=0`, `runtime_metrics jsonb`, nullable `raw_response text`, `raw_response_truncated boolean default false`, nullable sanitized `error_code/detail`. UNIQUE `(session_id,frame_id,attempt_no)` and UNIQUE `(review_id,session_id,frame_id)` |
| `vlm_findings` | `finding_id` PK, `review_id`, `session_id`, `frame_id`, `ordinal>=0`, `rule_id` in `R1..R4`, `evidence` and `explanation` each 1–1,000 characters. Composite FK to the review scope, UNIQUE `(review_id,ordinal)`, UNIQUE `(finding_id,session_id,frame_id)`. Only validated attempts may own findings |
| `finding_detections` | `finding_id`, `detection_id`, `session_id`, `frame_id`. PK `(finding_id,detection_id)`. Scoped FKs to findings and detections prevent references to another frame/session |
| `review_decisions` | `decision_id` PK, `finding_id` FK, `decision_number>=1`, `action` in `accept/reject/defer`, nullable `note` up to 2,000 characters, `reviewer_label`, `created_at`, nullable `supersedes_decision_id` FK. UNIQUE `(finding_id,decision_number)`. The first decision has no predecessor. Later ones reference the latest decision for that same finding |
| `human_annotations` | `annotation_id` PK, scoped frame FK, `annotation_revision>=1`, `annotator_label`, `created_at`, `protocol_version`, `visibility` (`visible`, `partially_occluded`, `not_visible`, `ungradable`), nullable `visible_count` in `{0,1}`, `boxes jsonb`, nullable `ungradable_reason`, nullable `supersedes_annotation_id` FK. UNIQUE `(session_id,frame_id,annotation_revision)` and `(annotation_id,session_id,frame_id)`. Append revisions, preserving earlier labels |
| `reference_adjudications` | `adjudication_id` PK, `session_id`, `frame_id`, `annotation_id`, `adjudication_revision>=1`, `adjudicator_label`, `created_at`, `protocol_version`, strict bounded `error_labels jsonb` and `finding_judgments jsonb`, nullable `supersedes_adjudication_id`. Scoped FK to the frozen annotation and frame. UNIQUE `(annotation_id,adjudication_revision)`. This is independent evaluation judgment, not proposal acceptance |
| `review_trials` | Composite PK `(trial_id UUID,trial_revision>=1)`, `session_id` FK, `condition` (`baseline`, `assisted`), `reviewer_label`, `created_at`, `budget_ms>0`, `active_ms>=0`, `waiting_ms>=0`, strict `reference_set`, `candidate_frame_ids`, `claims`, `timing_provenance`, and `cost_inputs` JSONB, `verification_state` (`recorded`, `adjudicated`, `invalid`), nullable `supersedes_trial_revision`. Append observations and their later adjudication revisions. Closed observations are immutable |

These eleven tables extend the six requested core tables to normalize findings/references, separate independent reference labels/adjudications, and retain reproducible timed trials. Evaluation payloads are small typed records, not a general analytics system. No generic job engine, user table, or model registry is needed. Detector identity belongs to the frozen session configuration. Each Qwen attempt retains its specific runtime/prompt identity.

Review status is `running`, `validated`, `invalid_json`, `invalid_reference`, `timeout`, `transport_error`, `oom`, or `cancelled`. An over-limit response is `invalid_json` with a truncated prefix and explicit limit error. Startup changes orphaned `running` attempts to `transport_error` with `interrupted` detail. A partial UNIQUE index allows only one `validated` review per frame. Outcomes are `findings`, `no_issue`, and `insufficient_evidence`, and are null for failed attempts.

Decision and annotation insertion locks the parent finding/frame row, checks the expected latest revision, and inserts the next revision atomically. An outdated UI submission receives a conflict and reloads. A repeated click with the same application-generated decision UUID is idempotent. Never update or delete historical decisions or labels to change a result. No automatically generated annotation is treated as human reference truth.

`human_annotations.boxes` is a strict, bounded list of at most one `{object_id, bbox_xyxy}` record for this one-cup experiment. A visible cup has one human-drawn native-pixel box. `not_visible` has count zero and no boxes. `ungradable` has null count, no boxes, and a required reason. Human annotation is collected without Qwen proposals, then matched to frozen predictions for evaluation. Ambiguous labels are excluded from quality denominators and counted separately.

### Evaluation record payloads

`reference_adjudications.error_labels` contains at most 101 atomic errors, each `{error_id, category, detection_id, object_id, explanation}`. Category is `false_negative`, `false_positive`, `duplicate`, or `localization_false_positive`. A false negative requires a reference object ID and no detection ID. Other categories require a detection ID from this frame, with object ID when relevant. Distinguish an unmatched poorly localized prediction from a non-cup false positive. Generate stable error UUIDs with UUIDv5 using `session_id` as namespace and `frame_id/category/target_id/annotation_id` as the name. Target IDs are UUIDs or safe object IDs such as `cup_1`. Freeze the annotation revision used to derive them.

`finding_judgments` contains at most eight `{finding_id, correct, linked_error_ids, explanation}` records for the validated review. These are human judgments against the frozen annotation/error labels. All IDs must resolve in the same frame. A correct R1/R2/R3 finding must link the corresponding reference error(s). R4 correctness is checked against the reference count and remains a separate frame-level measure, with optional corroborating error links. Reject cross-frame/unknown references. JSONB reference checks occur in a locked application transaction and are tested separately from SQL FKs. A later annotation does not silently relabel an older adjudication.

Each trial freezes a `reference_set` of `{frame_id, annotation_id, adjudication_id}` and the ordered candidate frame IDs, at most 12. A claim contains `{claim_id, frame_id, rule_id, detection_ids, object_id, finding_id, active_ms_at_claim, verified_error_ids}`. Optional IDs are null. Baseline claims have no finding ID. The assisted UI may prefill a claim from an accepted finding, but independent adjudication assigns `verified_error_ids`. A click alone never verifies an error. Count only verified claims whose active timestamp is within `budget_ms` and deduplicate error IDs.

The timed mode has Start, Pause, Finish, and Mark suspected error controls, with VLM evidence hidden in the baseline condition. Record monotonic active/wait intervals, condition, candidate set, and clock method in `timing_provenance`. Browser blur/inactivity pauses active time and is disclosed. A crash/interruption produces an invalid trial, not an imputed completed observation. Insert the closed raw observation as revision one, then append its independently adjudicated revision. Trial references and verified error links are validated against immutable parent records. `cost_inputs` stores currency, declared rates, allocation method, optional measured energy, and measurement provenance without assuming prices.

Frame `timings` retain decode, YOLO-call, and media-publication milliseconds. Session `stage_metrics` retain total probe/decode/YOLO/review/publication wall time, failures/retries, and available resource observations, with measurement method. Qwen attempts retain their call timing separately. Export timing is returned in `ExportResult` and trial observations, without mutating content revision after an export has been frozen. Unknown measurements remain null, never invented zeroes.

### Indexes and query patterns

- `sessions(state,updated_at)` for resumable/problem sessions.
- `frames(session_id,timestamp_ms,source_frame_index)` for ordered navigation, plus the PK and unique source-index index. Do not make timestamps unique.
- `vlm_reviews(session_id,frame_id,attempt_no)` and the partial validated-review index. `vlm_findings(rule_id,review_id)` supports rule filters. The scoped FKs and `finding_detections(detection_id)` support evidence joins.
- `review_decisions(finding_id,decision_number DESC)`, `human_annotations(session_id,frame_id,annotation_revision DESC)`, `reference_adjudications(annotation_id,adjudication_revision DESC)`, and `review_trials(session_id,condition,trial_id,trial_revision DESC)` support frozen/current references, timed comparisons, and history.
- Add supporting indexes for other FK delete/check paths when migrations reveal a missing leading-key index. Do not add blanket JSONB GIN indexes. Use `EXPLAIN` against actual session queries before adding more.

Keep FK deletes restrictive. There is no automatic session deletion or retention sweep in V1. A future explicit deletion workflow must coordinate DB and disk and preserve the user's chosen retention policy. Application startup never mutates the schema.

## 3. Stable filesystem paths

```text
DATA_ROOT/
  session_<session_id>/
    original.mp4                 # or original.mov, from validated container
    frames/frame_000000120.jpg
    annotated/frame_000000120.jpg
    json/frame_000000120.json
    evaluation.json              # canonical trial-history projection
    manifest.json                # last successfully published manifest
    .staging/                    # incomplete writes, excluded from exports
  exports/<session_id>/<export_id>.zip
```

Database paths are POSIX-style relative paths within the session directory, such as `frames/frame_000000120.jpg`. The session directory name is derived from the UUID. Export archive paths have the same relative layout, with no host paths. Do not use an uploaded filename in a filesystem path.

Default `DATA_ROOT` is `Path.home() / "vlm-cnn-auditor-data"`. On the current Windows machine, an example original path is `C:\Users\Dell\vlm-cnn-auditor-data\session_<UUID>\original.mp4`. It is outside the source checkout and can be overridden by the environment. PostgreSQL stores only its safe relative path, hash, and searchable metadata, never the video bytes. The planning delivery creates no runtime media directory.

Resolve every path under the configured root, reject traversal/absolute paths and symlinks escaping it, and use an explicit export allowlist. Write a temporary file in the destination filesystem, flush and fsync, then atomically rename. Never overwrite a valid original or raw frame with different bytes. If a final immutable path exists, verify its stored digest before reusing it. Mismatches become `corrupt`, requiring repair or a new session.

## 4. Publication protocol and reconciliation

Filesystem rename and PostgreSQL commit are separate operations. Use explicit intermediate states rather than claiming a distributed transaction.

1. **Upload:** Stream to a bounded staging file, validate/probe, hash, and rename to the immutable original path. Commit the session/video rows. A crash before the commit leaves an orphan session directory that reconciliation reports and quarantines, without deleting it automatically.
2. **Frame prepare:** Create the canonical raw JPEG and detector overlay in staging and hash them. Commit frame metadata and detector rows with `media_state=pending`. A successful retry never reinserts detections with new IDs. No pending pair is displayed as complete.
3. **Media publish:** Rename each staged image into its final immutable path, verifying an existing destination. Then commit `media_state=ready`, increment frame/session content revisions, and schedule snapshot publication. A crash between the two renames remains visibly pending.
4. **JSON publish:** Under the session lock, read the complete frame bundle from a consistent database snapshot at revision `r`. Serialize stable field order, detections by ordinal, attempts by attempt number, and histories by revision. Write/rename JSON, then record `snapshot_revision=r` and its hash only if the content revision is still `r`. A changed revision stays pending and is regenerated. Do not include the JSON file's own hash inside its contents.
5. **Further mutations:** In the same transaction that inserts a validated review, decision, annotation, or reference adjudication, increment frame/session revisions. Trial insertion increments session revision. Publish the affected projections afterward. The UI reads canonical DB records and verified media, never trusts a stale JSON file as current.

At startup and before resume/export, reconcile original hashes, media states, staging files, JSON revisions, and checksums. Complete an interrupted rename when recorded staged bytes match. Regenerate JSON from DB. If media is missing, reconstruct only from the original using the frozen decoder/rotation/encoder/model settings and verify expected hashes. Otherwise mark it blocked and retain evidence. Do not rerun a successful inference just to repair JSON.

Test failure injection before/after each file rename, DB commit, snapshot write, and export publication. Interrupted work must either resume to the same IDs/content or remain explicitly blocked. There is no promise of zero-byte-loss across sudden hardware failure on every filesystem. Backups require both PostgreSQL and the session tree from a consistent point.

## 5. JSON contracts

Pydantic models in `auditor/contracts.py` are the source for generated Draft 2020-12 JSON Schemas during implementation. Publish their version with every artifact. Reject extra fields, NaN/Infinity, string-coerced IDs/numbers, and unknown enum values. Schema validation is followed by frame-context validation in the service.

### FrameSnapshotV1 schema

All following top-level properties are required and no extra properties are allowed. Null is allowed only where stated. Every nested record uses the same strict policy.

| Property | Type and contract |
|---|---|
| `schema_version`, `snapshot_revision` | Literal `1`, integer `>=1` |
| `session_id`, `video_id`, `frame_id`, `source_frame_index`, `timestamp_ms` | UUID string, UUID string, regex `^frame_[0-9]{9}$`, integer `>=0`, integer `>=0`. Must match persisted identity |
| `presentation` | Object with integer `source_pts`, `first_pts`, positive integer `time_base_num/den`, `timestamp_source`, and supported `rotation_degrees` |
| `image` | Object with positive integer `width/height` and literal `bbox_format="xyxy_pixels"` |
| `artifacts` | Object containing `raw` and `annotated`, each `{path: safe relative string, sha256: 64 lowercase hex characters}` |
| `yolo` | Object `{model_name, weights_sha256, package_version, settings, detections}`. Settings record image size, threshold, NMS IoU, class, max detections and device. Detections are ordered `{detection_id,ordinal,class_id:41,label:"cup",score,bbox_xyxy:[x1,y1,x2,y2]}` with the bounds above |
| `vlm_selection` | Object `{selected:boolean, reason:string}`. Selected does not mean a successful review |
| `vlm_reviews` | Ordered array of attempt records, carrying review ID, attempt/status, nullable outcome, model/prompt/rule/schema provenance, transform and request hash, timing/usage, raw response/truncation/errors, and validated findings with application-assigned IDs. Failed attempts contain no findings |
| `review_decisions` | Array of the complete decision records defined above, referencing application finding IDs |
| `human_annotations` | Array of independent, versioned human annotation records. Not synthesized from accepted proposals |
| `reference_adjudications` | Array of independently recorded error labels and finding judgments, each pinned to an immutable annotation revision |

The image pair and snapshot exist for every successful sampled YOLO frame, including those Qwen does not select. Pending/corrupt frames remain represented by diagnostic DB/manifest entries, not a misleading “complete” frame snapshot.

### Qwen response JSON Schema

This schema constrains the model response, not the richer persisted snapshot. The application assigns review/finding UUIDs after validation.

```json
{
  "$schema": "https://json-schema.org/draft/2020-12/schema",
  "title": "CupReviewResponseV1",
  "type": "object",
  "additionalProperties": false,
  "required": ["session_id", "frame_id", "timestamp_ms", "outcome", "estimated_count", "needs_review", "findings"],
  "properties": {
    "session_id": {"type": "string", "format": "uuid"},
    "frame_id": {"type": "string", "pattern": "^frame_[0-9]{9}$"},
    "timestamp_ms": {"type": "integer", "minimum": 0},
    "outcome": {"enum": ["findings", "no_issue", "insufficient_evidence"]},
    "estimated_count": {"type": ["integer", "null"], "minimum": 0, "maximum": 100},
    "needs_review": {"type": "boolean"},
    "findings": {
      "type": "array", "maxItems": 8,
      "items": {
        "type": "object", "additionalProperties": false,
        "required": ["rule_id", "detection_ids", "evidence", "explanation"],
        "properties": {
          "rule_id": {"enum": ["R1", "R2", "R3", "R4"]},
          "detection_ids": {"type": "array", "maxItems": 100, "uniqueItems": true, "items": {"type": "string", "format": "uuid"}},
          "evidence": {"type": "string", "minLength": 1, "maxLength": 1000},
          "explanation": {"type": "string", "minLength": 1, "maxLength": 1000}
        }
      }
    }
  },
  "allOf": [
    {"if": {"properties": {"outcome": {"const": "findings"}}}, "then": {"properties": {"findings": {"minItems": 1}, "needs_review": {"const": true}}}},
    {"if": {"properties": {"outcome": {"const": "no_issue"}}}, "then": {"properties": {"findings": {"maxItems": 0}, "needs_review": {"const": false}}}},
    {"if": {"properties": {"outcome": {"const": "insufficient_evidence"}}}, "then": {"properties": {"findings": {"maxItems": 0}, "needs_review": {"const": true}, "estimated_count": {"type": "null"}}}}
  ]
}
```

Context validators require an exact session/frame/timestamp echo. R1 uses no detection references, R2 exactly one, R3 at least two distinct IDs, and R4 zero or more. Every referenced ID must belong to this frame. R4 requires a non-null estimated count different from the detector count. A non-null `no_issue` count must equal the detector count. `insufficient_evidence` contains no findings and requests human review. Reject duplicate identical rule/reference proposals and contradictory outcome fields. Failed parsing or contextual validation stores a diagnostic attempt only. It does not insert trusted model-derived records. Store bounded raw text for debugging even when it cannot be parsed.

Synthetic response example, not a measured model result:

```json
{
  "session_id": "11111111-1111-4111-8111-111111111111",
  "frame_id": "frame_000000120",
  "timestamp_ms": 4000,
  "outcome": "findings",
  "estimated_count": 1,
  "needs_review": true,
  "findings": [{
    "rule_id": "R1",
    "detection_ids": [],
    "evidence": "A cup-shaped object is visible near the center of the raw frame.",
    "explanation": "YOLO returned no cup boxes. A human should verify a possible miss."
  }]
}
```

## 6. Export manifest

Export holds the session lock against decisions/inference, reconciles all included media, and regenerates stale JSON. Read one consistent database view and record `session_revision`. Write `manifest.json` and the ZIP in staging, verify the archive, then publish the ZIP atomically. Release the lock afterward. Do not mix files from before and after a decision.

The archive contains the preserved original, all ready raw/annotated frame pairs, per-frame JSON including review/decision/annotation/adjudication history, `evaluation.json`, and the manifest. The evaluation projection contains `schema_version`, `session_id`, `session_revision`, and all immutable trial revisions, or an empty trial list before experiments exist. A partial-session export is allowed only with an explicit incomplete status and listed failed/unselected frames. The first acceptance export must be complete.

Required manifest properties are `schema_version=1`, UUID `export_id`, UUID `session_id`, `session_revision`, UTC `exported_at`, `session_state`, `complete`, frozen `configuration` and its hash, pipeline/package/model/prompt/rule provenance, source video timing/rotation metadata, `frames`, `diagnostics`, and `files`.

Each frame entry carries the full identity triple, source index, snapshot revision, detection count, Qwen selection/outcome, and its three artifact paths. Each file entry carries `path`, `kind`, `byte_count`, and SHA-256, plus the frame key for frame artifacts. File kinds are `original`, `raw`, `annotated`, `frame_json`, and `evaluation`. Manifest does not hash itself. ZIP checksum is recorded in the export result, outside the ZIP. Exclude staging files, application logs, environment files, host paths, DB dumps, and model weights.

On verification, reject missing/extra archive entries, unsafe paths, duplicate names, hash mismatches, unknown schema versions, inconsistent frame keys, and snapshot/session revision drift. A ZIP being readable is insufficient evidence of a valid session.

## 7. Migrations and compatibility

Use Alembic with an initial schema migration and named constraints/indexes. Keep migrations separate from app launch. Verify upgrade on an empty PostgreSQL 16 database and a database seeded with an earlier supported schema. Migration history must match the application version before writes are permitted.

No migration runs in this planning delivery. Later data migrations back up both database and media, use explicit version conversion, and preserve old prediction/decision history. Do not silently rewrite existing JSON schemas or infer missing timestamps. Old-schema artifacts are readable only through an explicit versioned reader. Destructive downgrade on populated evidence is refused. Restore from a tested backup rather than improvising a lossy downgrade.
