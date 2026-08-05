# Structured Add integration contract

`structured` is a generic fixed-Schema memory algorithm for callers that have already decided that an observation is worth retaining. It is independent of any producer, domain, or record type.

## Algorithm binding

- `vanilla` uses `vanilla_add` and Vanilla Search.
- `schema` uses `schema_add` and Schema Search.
- `structured` uses `structured_add` and the structured-only, Episode-aware Schema Search assembly.

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

## Runtime behavior

MindMemOS first recalls a bounded tenant-isolated Episode candidate set. One strict extraction call then decides Episode reuse/create and extracts 0..N fixed-Schema entities, properties, and explicit edges. It does not run the buffered Schema Episode splitter, Schema selector, search-field generator, or higher-order synthesis. Invalid extraction receives at most one repair call.

Property values are batch-embedded and same-typed candidates are consolidated before history recall. Historical candidates are scoped by tenant/actor, selected or related Episode, entity type, and property name; generated titles and new entity IDs are not identity keys. Except for exact normalized content in the selected Episode, similarity only ranks candidates. All ambiguous groups use at most one contextual merge call, with invalid groups safely falling back to `create`. The four actions are:

- `create`: write a distinct active memory.
- `reinforce`: keep the memory body and ID, increment support, and append evidence references.
- `update`: write a new active revision and archive the old revision.
- `supersede`: write a new active revision and mark the old revision superseded.

All revisions and Add Records are retained. Structured fast search fuses direct property recall with relevant-Episode graph expansion and returns standard active memory DTOs with their actual property type and bounded Episode context. Existing get/list/update/archive, graph visualization, and promotion consumers remain compatible.

## Provider and concurrency behavior

Chat and Embedding clients are resolved inside the existing request-scoped provider context. Dynamic bindings can differ per user/request; static provider configuration remains the fallback when dynamic routing is disabled.

Extraction and ambiguous merge calls share the configured event-loop concurrency bound, including separately constructed worker pipeline instances. LLM and Embedding waits plus the initial candidate recall happen outside process-shared commit stripes. A stripe is keyed by isolated Episode and typed-property scope; it repeats bounded storage recall, finalizes the provisional decisions, and submits the mutation plan. Generated display titles never derive storage IDs. Reinforcement commands use Qdrant conditional payload updates and retry up to `max_write_conflict_retries`, preserving different concurrent evidence events and counting an idempotent event at most once.

## Calibration

`create_below` and same-batch grouping thresholds are starting points, not universal truth. A high historical vector score never reinforces by itself. Calibrate thresholds on real project data while comparing Episode reuse/create accuracy, extraction/merge calls, duplicate rate, merge errors, card completeness, retrieval diversity, optimistic conflicts, and single/five-way concurrent latency. A sustained conflict that exceeds the configured retry limit fails explicitly and can be retried with the same idempotency key when one was supplied.
