# Structured Add integration contract

`structured` is a generic fixed-Schema memory algorithm for callers that have already decided that an observation is worth retaining. It is independent of any producer, domain, or record type.

## Algorithm binding

- `vanilla` uses `vanilla_add` and Vanilla Search.
- `schema` uses `schema_add` and Schema Search.
- `structured` uses `structured_add` and its independent direct-plus-bounded-graph Structured Search pipeline.

Existing projects continue to default to `vanilla`. Select `structured` only on the task project whose runtime needs low-latency structured writes. Manual global/project maintenance and promotion flows do not need to change algorithms.

## Request contract

Use the normal `POST /v1/memory/add` contract. The caller provides the existing actor/task scope; it does not provide an Episode ID, card identity, merge policy, or Schema subset. An opaque `idempotency_key` is optional and is useful only for transport retries:

```json
{
  "user_id": "user-1",
  "session_id": "task-run-42",
  "messages": [
    {
      "role": "user",
      "content": "Gateway A became unavailable and caused connectivity ticket 42",
      "timestamp": 1785744000000
    }
  ],
  "mode": "sync",
  "idempotency_key": "monitoring:gateway-a:incident-42:event-1",
  "metadata": {
    "source": "monitoring",
    "tags": ["gateway", "incident"]
  }
}
```

When present, the key is treated only as an opaque idempotency value. MindMemOS does not parse its producer-specific segments. Reusing it for a retry converges on the same Add Record and prevents duplicate reinforcement. Omitting it does not disable Episode resolution, semantic consolidation, or historical merging.

Existing optional request evidence such as `score`, `task_id`, and generic metadata is retained without assigning producer-specific meaning to it.

For unordered, non-contiguous sources, replace `messages` with `document_blocks`. The two fields are mutually exclusive. Each block is independently traceable; its position and `document_id` are provenance only, not continuity or identity:

```json
{
  "user_id": "user-1",
  "session_id": "task-run-42",
  "document_blocks": [
    {
      "block_id": "manual-a:p12",
      "document_id": "manual-a",
      "locator": {"page": 12},
      "messages": [{"text": "Gateway A retries twice."}]
    },
    {
      "block_id": "ticket-42:comment-7",
      "document_id": "ticket-42",
      "locator": {"comment": 7},
      "messages": [{"text": "Gateway A waits 5 seconds between retries."}]
    }
  ],
  "mode": "sync",
  "idempotency_key": "import:task-run-42:batch-1"
}
```

Batch blocks are accepted only by `structured`; vanilla/schema projects return `structured.batch_not_supported`. The parent Add Record retains the complete blocks. Result memories retain bounded block IDs, document IDs, locators, hashes, and Add Record/evidence references.

## Runtime behavior

MindMemOS treats the request messages as one document block. Schema eligibility is resolved before complete extraction. A valid caller-provided `metadata.structured_allowed_property_names` list is authoritative and skips the selector; otherwise one bounded selector chooses only fixed-Schema entity/property slots and whether explicit relations need extraction. The selector cannot author facts, titles, Episodes, or merge decisions.

Complete extraction receives only the selected property names and value shapes, not Schema policy prose about whether content is useful, reusable, successful, or worth retaining. It performs a lossless projection: distinct facts, constraints, quantities, formulas, conditions, exceptions, causal relations, certainty, and named subjects are retained instead of summarized into broader claims. Invalid extraction receives at most one shape repair. When coverage validation is enabled, one relation-only audit identifies omitted claims using bounded source-grounded evidence spans; only a valid omission can trigger one directed completion call, and invalid audit/completion output preserves the original valid extraction.

For `document_blocks`, the pipeline independently extracts every block under the same fixed Schema and collects all results before persistence. It then allocates every non-empty block exactly once across existing Episodes or 1..N new batch Episodes. Within each Episode it first resolves same-subject aliases and losslessly consolidates compatible new facts; uncertain or incompatible facts remain separate. Only after that internal collection does typed, Episode-fenced historical recall and create/reinforce/update/supersede merging run. The successful batch builds one mutation plan and calls the database writer once. Any extraction or Episode-allocation failure occurs before mutation.

All entities and properties extracted from that document block are accumulated into one `MemoryDbMutationPlan` and submitted through one database mutation call. The implementation may parallelize Embedding and candidate reads, but it does not issue one independent Add operation per extracted card.

Property values are batch-embedded and same-typed candidates are consolidated before history recall. Historical candidates are scoped by tenant/actor, selected or related Episode, entity type, and property name; generated titles and new entity IDs are not fingerprints or semantic uniqueness keys. Exact-content fingerprints exist only for replay/comparison and storage provenance. Similarity only ranks candidates. Semantic identity and historical consolidation are decided by one contextual merge call over the relevant Schema, current entity facts, bounded historical facts, and Episode context. If the model cannot combine facts without changing meaning or dropping information, it must select `create`. Invalid groups also safely fall back to `create`. The four actions are:

- `create`: write a distinct active memory.
- `reinforce`: keep the memory body and ID, increment support, and append evidence references.
- `update`: write a new active revision and archive the old revision.
- `supersede`: write a new active revision and mark the old revision superseded.

All revisions and Add Records are retained. The Add Record preserves the original request messages, while each memory keeps its source Add Record references and an immutable source-block hash. `structured` search performs direct dense/BM25 property recall and then bounded expansion through canonical entity, shared Episode, and explicit entity relations. It remains independent from Schema Search and returns active property cards. Existing get/list/update/archive, graph visualization, and promotion consumers remain compatible.

## Provider and concurrency behavior

Chat and Embedding clients are resolved inside the existing request-scoped provider context. Dynamic bindings can differ per user/request; static provider configuration remains the fallback when dynamic routing is disabled.

Extraction and ambiguous merge calls share the configured event-loop concurrency bound, including separately constructed worker pipeline instances. LLM and Embedding waits plus the initial candidate recall happen outside process-shared commit stripes. A stripe is keyed by isolated Episode and typed-property scope; it repeats bounded storage recall, finalizes the provisional decisions, and submits the mutation plan. Generated display titles never derive storage IDs. Reinforcement commands use Qdrant conditional payload updates and retry up to `max_write_conflict_retries`, preserving different concurrent evidence events and counting an idempotent event at most once.

## Calibration

`create_below` and same-batch grouping thresholds are starting points, not universal truth. A high historical vector score never reinforces by itself. Calibrate thresholds on real project data while comparing Episode reuse/create accuracy, extraction/merge calls, duplicate rate, merge errors, card completeness, retrieval diversity, optimistic conflicts, and single/five-way concurrent latency. A sustained conflict that exceeds the configured retry limit fails explicitly and can be retried with the same idempotency key when one was supplied.
