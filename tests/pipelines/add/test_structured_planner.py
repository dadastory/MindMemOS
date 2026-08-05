import asyncio
import json
from datetime import UTC, datetime

import pytest
from mindmemos.config import StructuredDedupConfig
from mindmemos.llm import ChatResponse, EmbeddingResponse
from mindmemos.pipelines.add.structured.planner import (
    StructuredMergeDecider,
    StructuredMergeRequest,
    StructuredProperty,
    batch_embed,
    classify_similarity,
    consolidate_structured_properties,
    recall_structured_candidates,
)
from mindmemos.typing import MemoryDbSearchHit, MemoryDbSearchResult, MemoryRequestContext, MemoryView


def context():
    return MemoryRequestContext(
        request_id="00000000-0000-0000-0000-000000000001",
        account_id="account-1",
        project_id="project-1",
        api_key_uuid="key-1",
        user_id="user-1",
        session_id="task-1",
        memory_algorithm="structured",
    )


def memory(memory_id: str, content: str, *, score=0.9):
    view = MemoryView(
        memory_id=memory_id,
        project_id="project-1",
        user_id="user-1",
        session_id="task-1",
        content=content,
        mem_type="experience",
        mem_extract_type="structured",
        status="active",
        property_name="strategy",
        entity_id="entity-1",
        entity_type="task_experience",
        created_at=datetime(2026, 8, 1, tzinfo=UTC),
    )
    return MemoryDbSearchHit(memory_id=memory_id, score=score, memory=view, source="dense")


class RecordingEmbed:
    def __init__(self):
        self.batches = []

    async def embed(self, task, text, **kwargs):
        values = [text] if isinstance(text, str) else list(text)
        self.batches.append((task, values))
        return EmbeddingResponse(embeddings=[[float(len(value))] for value in values])


@pytest.mark.asyncio
async def test_batch_embed_chunks_and_preserves_input_order():
    embed = RecordingEmbed()

    result = await batch_embed(embed, ["a", "bb", "ccc", "dddd", "eeeee"], batch_size=2, task="structured")

    assert [len(values) for _, values in embed.batches] == [2, 2, 1]
    assert result == [[1.0], [2.0], [3.0], [4.0], [5.0]]


def test_same_batch_paraphrases_are_grouped_within_one_extracted_entity():
    items = [
        StructuredProperty(
            entity_name="Adaptive mutation",
            entity_type="task_experience",
            property_name="strategy",
            content="Raise mutation after prolonged stagnation.",
            property_time="2026-08-03",
            fingerprint="a",
            memory_id="memory-a",
            entity_id="entity-a",
            vector=[1.0, 0.0],
        ),
        StructuredProperty(
            entity_name="Adaptive mutation",
            entity_type="task_experience",
            property_name="strategy",
            content="Increase mutation rate when generations stop improving.",
            property_time="2026-08-03",
            fingerprint="b",
            memory_id="memory-b",
            entity_id="entity-a",
            vector=[0.999, 0.01],
        ),
        StructuredProperty(
            entity_name="Elitism",
            entity_type="task_experience",
            property_name="strategy",
            content="Keep the best individual in every generation.",
            property_time="2026-08-03",
            fingerprint="c",
            memory_id="memory-c",
            entity_id="entity-c",
            vector=[0.0, 1.0],
        ),
    ]

    groups = consolidate_structured_properties(items, similarity_at_or_above=0.97)

    assert len(groups) == 2
    assert groups[0].batch_contents == [items[0].content, items[1].content]
    assert groups[0].batch_entity_names == ["Adaptive mutation"]
    assert groups[1].batch_contents == [items[2].content]


def test_same_batch_different_property_types_are_never_grouped():
    first = StructuredProperty(
        entity_name="A",
        entity_type="task_experience",
        property_name="strategy",
        content="same wording",
        property_time="2026-08-03",
        fingerprint="a",
        memory_id="memory-a",
        entity_id="entity-a",
        vector=[1.0],
    )
    second = StructuredProperty(
        entity_name="B",
        entity_type="task_experience",
        property_name="error",
        content="same wording",
        property_time="2026-08-03",
        fingerprint="b",
        memory_id="memory-b",
        entity_id="entity-b",
        vector=[1.0],
    )

    assert len(consolidate_structured_properties([first, second], similarity_at_or_above=0.97)) == 2


@pytest.mark.parametrize(
    ("entity_type", "property_name", "value", "left_name", "right_name"),
    [
        ("device", "status", "offline", "gateway-a", "gateway-b"),
        ("user", "preference", "dark mode", "alice", "bob"),
        ("ticket", "state", "open", "ticket-101", "ticket-102"),
        ("alert", "severity", "critical", "disk-alert", "network-alert"),
        ("experiment", "result", "score=0.91", "experiment-a", "experiment-b"),
    ],
)
def test_same_batch_equal_values_remain_separate_across_entities(
    entity_type,
    property_name,
    value,
    left_name,
    right_name,
):
    items = [
        StructuredProperty(
            entity_name=left_name,
            entity_type=entity_type,
            property_name=property_name,
            content=value,
            property_time="2026-08-04",
            fingerprint="same",
            memory_id="memory-a",
            entity_id="entity-a",
            vector=[1.0, 0.0],
        ),
        StructuredProperty(
            entity_name=right_name,
            entity_type=entity_type,
            property_name=property_name,
            content=value,
            property_time="2026-08-04",
            fingerprint="same",
            memory_id="memory-b",
            entity_id="entity-b",
            vector=[1.0, 0.0],
        ),
    ]

    groups = consolidate_structured_properties(items, similarity_at_or_above=0.97)

    assert len(groups) == 2
    assert [group.batch_entity_names for group in groups] == [[left_name], [right_name]]


@pytest.mark.parametrize(
    "score,expected",
    [
        (None, "create"),
        (0.1, "create"),
        (0.819, "create"),
        (0.82, "ambiguous"),
        (0.9, "ambiguous"),
        (0.97, "ambiguous"),
        (1.0, "ambiguous"),
    ],
)
def test_similarity_threshold_classification(score, expected):
    assert classify_similarity(score, StructuredDedupConfig()) == expected


class ConcurrentReader:
    def __init__(self):
        self.active = 0
        self.max_active = 0
        self.requests = []

    async def search_dense(self, ctx, req, *, query_vector):
        self.requests.append((ctx, req, query_vector))
        self.active += 1
        self.max_active = max(self.max_active, self.active)
        await asyncio.sleep(0)
        self.active -= 1
        return MemoryDbSearchResult(query=req.query, hits=[])


@pytest.mark.asyncio
async def test_candidate_recall_is_parallel_typed_and_episode_constrained_without_entity_identity():
    reader = ConcurrentReader()
    items = [
        StructuredProperty(
            entity_name=f"entity-{index}",
            entity_type="task_experience",
            property_name="strategy",
            content=f"strategy-{index}",
            property_time="2026-08-03",
            fingerprint=str(index),
            memory_id=f"memory-{index}",
            entity_id=f"entity-id-{index}",
            vector=[float(index)],
        )
        for index in range(3)
    ]

    results = await recall_structured_candidates(
        reader,
        context(),
        items,
        episode_ids={"episode-current", "episode-related"},
        top_k=5,
    )

    assert len(results) == 3
    assert reader.max_active >= 3
    primary_requests = [
        request
        for request in reader.requests
        if any(item.field == "episode_ids" and item.op == "any" for item in request[1].filters.must)
    ]
    assert len(primary_requests) == 3
    legacy_requests = [
        request
        for request in reader.requests
        if any(item.field == "episode_ids" and item.op == "is_empty" for item in request[1].filters.must)
    ]
    assert len(legacy_requests) == 3
    for _, req, _ in primary_requests:
        conditions = {(item.field, item.value) for item in req.filters.must}
        assert ("status", "active") in conditions
        assert ("user_id", "user-1") in conditions
        assert ("entity_type", "task_experience") in conditions
        assert ("property_name", "strategy") in conditions
        assert all(field != "entity_id" for field, _ in conditions)
        episode_condition = next(item for item in req.filters.must if item.field == "episode_ids")
        assert episode_condition.op == "any"
        assert episode_condition.values == ["episode-current", "episode-related"]
        assert req.top_k == 5


class LegacySchemaReader:
    def __init__(self):
        self.requests = []
        self.legacy = memory("legacy-schema-memory", "Historical strategy", score=0.88)
        self.legacy.memory.mem_extract_type = "schema"

    async def search_dense(self, ctx, req, *, query_vector):
        self.requests.append(req)
        is_legacy_fallback = any(item.field == "episode_ids" and item.op == "is_empty" for item in req.filters.must)
        return MemoryDbSearchResult(
            query=req.query,
            hits=[self.legacy] if is_legacy_fallback else [],
        )


@pytest.mark.asyncio
async def test_candidate_recall_keeps_legacy_schema_memory_in_same_session_eligible():
    reader = LegacySchemaReader()
    item = StructuredProperty(
        entity_name="Renamed title",
        entity_type="task_experience",
        property_name="strategy",
        content="Historical strategy with more evidence",
        property_time="2026-08-03",
        fingerprint="new",
        memory_id="new-memory",
        entity_id="new-entity",
        vector=[1.0],
    )

    results = await recall_structured_candidates(
        reader,
        context(),
        [item],
        episode_ids={"episode-current"},
        top_k=5,
    )

    assert [hit.memory_id for hit in results[0]] == ["legacy-schema-memory"]
    legacy_request = next(
        req
        for req in reader.requests
        if any(condition.field == "episode_ids" and condition.op == "is_empty" for condition in req.filters.must)
    )
    assert any(
        condition.field == "session_id" and condition.value == "task-1" for condition in legacy_request.filters.must
    )


class QueueMergeLLM:
    def __init__(self, response: str | Exception):
        self.response = response
        self.calls = []

    async def chat(self, task, messages, format_parser=None, **kwargs):
        self.calls.append((task, messages, format_parser))
        if isinstance(self.response, Exception):
            raise self.response
        return ChatResponse(finish_reason="stop", content=self.response)


@pytest.mark.asyncio
@pytest.mark.parametrize("action", ["create", "reinforce", "update", "supersede"])
async def test_ambiguous_merge_accepts_four_actions_with_one_call(action):
    payload = {"action": action, "reason": "fixture"}
    if action != "create":
        payload["target_id"] = "memory-1"
    if action in {"update", "supersede"}:
        payload["merged_content"] = "Merged strategy."
    llm = QueueMergeLLM(json.dumps(payload))
    decider = StructuredMergeDecider(llm_client=llm, max_candidates=3)

    decision = await decider.decide("New strategy", [memory(f"memory-{i}", f"old-{i}") for i in range(1, 6)])

    assert decision.action == action
    assert len(llm.calls) == 1
    prompt = llm.calls[0][1][0]["content"]
    assert "memory-3" in prompt
    assert "memory-4" not in prompt
    assert llm.calls[0][2] is None


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "response",
    [
        "not-json",
        '{"action":"unknown"}',
        '{"action":"reinforce","target_id":"not-a-candidate"}',
        '{"action":"update","target_id":"memory-1"}',
        RuntimeError("provider unavailable"),
    ],
)
async def test_invalid_or_failed_merge_falls_back_to_create_without_retry(response):
    llm = QueueMergeLLM(response)
    decider = StructuredMergeDecider(llm_client=llm, max_candidates=3)

    decision = await decider.decide("New strategy", [memory("memory-1", "old")])

    assert decision.action == "create"
    assert len(llm.calls) == 1


@pytest.mark.asyncio
async def test_all_ambiguous_groups_are_decided_in_one_call_with_episode_context():
    first = StructuredProperty(
        entity_name="gateway-a",
        entity_type="device",
        property_name="status",
        content="Gateway is offline.",
        property_time="2026-08-03",
        fingerprint="a",
        memory_id="new-a",
        entity_id="entity-a",
        entity_description="Primary warehouse gateway",
        vector=[1.0],
        batch_contents=["Gateway is offline.", "The gateway cannot be reached."],
    )
    second = StructuredProperty(
        entity_name="ticket-42",
        entity_type="ticket",
        property_name="state",
        content="Ticket is open.",
        property_time="2026-08-03",
        fingerprint="b",
        memory_id="new-b",
        entity_id="entity-b",
        entity_description="Connectivity incident ticket",
        vector=[1.0],
    )
    response = json.dumps(
        {
            "decisions": [
                {
                    "group_index": 0,
                    "action": "reinforce",
                    "target_id": "old-a",
                    "equivalent_ids": [],
                    "reason": "same subject, fact, and context",
                },
                {
                    "group_index": 1,
                    "action": "create",
                    "equivalent_ids": [],
                    "merged_content": "Ticket is open.",
                    "reason": "different subject",
                },
            ]
        }
    )
    llm = QueueMergeLLM(response)
    decider = StructuredMergeDecider(llm_client=llm, max_candidates=3)
    requests = [
        StructuredMergeRequest(
            property=first,
            candidates=[memory("old-a", "Gateway is unreachable.")],
            candidate_entity_contexts={
                "old-a": {
                    "entity_id": "stored-gateway-a",
                    "name": "Warehouse gateway A",
                    "description": "Primary warehouse gateway",
                    "entity_type": "device",
                }
            },
        ),
        StructuredMergeRequest(
            property=second,
            candidates=[memory("old-b", "Ticket is pending.")],
            candidate_entity_contexts={
                "old-b": {
                    "entity_id": "stored-ticket-7",
                    "name": "ticket-7",
                    "description": "A different incident ticket",
                    "entity_type": "ticket",
                }
            },
        ),
    ]

    decisions = await decider.decide_batch(
        requests,
        current_episode={"episode_id": "episode-current", "description": "Current maintenance incident"},
        episode_contexts={"episode-old": {"episode_id": "episode-old", "description": "Earlier maintenance incident"}},
        entity_schema=[
            {
                "entity_type": "device",
                "entity_description": "A physical or virtual device",
                "dynamic_property": {"status": {"desc": "Current operational state"}},
            },
            {
                "entity_type": "ticket",
                "entity_description": "A tracked work item",
                "dynamic_property": {"state": {"desc": "Current workflow state"}},
            },
        ],
    )

    assert [decision.action for decision in decisions] == ["reinforce", "create"]
    assert len(llm.calls) == 1
    prompt = llm.calls[0][1][0]["content"]
    assert "Current maintenance incident" in prompt
    assert "Earlier maintenance incident" in prompt
    assert "The gateway cannot be reached." in prompt
    assert "Primary warehouse gateway" in prompt
    assert "Warehouse gateway A" in prompt
    assert "stored-gateway-a" in prompt
    assert "Current operational state" in prompt
    assert "Compare the new group with every supplied candidate" in prompt
    lowered = prompt.casefold()
    for producer_term in ("reusable mechanism", "intended lesson", "different run", "generation", "task score"):
        assert producer_term not in lowered
    assert "same real-world subject" in lowered
    assert "same fact or claim" in lowered
    assert "compatible contextual validity" in lowered
    assert "one concise canonical statement" in lowered
    assert "language of the new property content" in lowered
    assert "do not concatenate translations" in lowered
    assert "do not repeat paraphrases" in lowered
    assert "allowed_memory_ids" in prompt
    assert "copy target_id and equivalent_ids only" in lowered
    assert "never use an entity_id" in lowered


@pytest.mark.asyncio
async def test_batch_merge_requires_explicit_equivalent_candidate_classification():
    item = StructuredProperty(
        entity_name="Adaptive mutation",
        entity_type="task_experience",
        property_name="strategy",
        content="Increase mutation after stagnation.",
        property_time="2026-08-03",
        fingerprint="new",
        memory_id="new-memory",
        entity_id="new-entity",
        vector=[1.0],
    )
    llm = QueueMergeLLM(
        json.dumps(
            {
                "decisions": [
                    {
                        "group_index": 0,
                        "action": "reinforce",
                        "target_id": "old-a",
                        "reason": "same lesson",
                    }
                ]
            }
        )
    )
    decider = StructuredMergeDecider(llm_client=llm, max_candidates=3)

    decisions = await decider.decide_batch(
        [StructuredMergeRequest(property=item, candidates=[memory("old-a", "old")])],
        current_episode={"episode_id": "episode-current"},
        episode_contexts={},
    )

    assert decisions[0].action == "create"


@pytest.mark.asyncio
async def test_batch_merge_logs_sanitized_validation_reason_counts(monkeypatch):
    item = StructuredProperty(
        entity_name="Gateway A",
        entity_type="device",
        property_name="status",
        content="Gateway A is offline.",
        property_time="2026-08-03",
        fingerprint="new",
        memory_id="new-memory",
        entity_id="new-entity",
        vector=[1.0],
    )
    llm = QueueMergeLLM(
        json.dumps(
            {
                "decisions": [
                    {
                        "group_index": 0,
                        "action": "reinforce",
                        "target_id": "old-a",
                        "reason": "same fact",
                    }
                ]
            }
        )
    )

    class RecordingLogger:
        def __init__(self):
            self.events = []

        def info(self, event, **fields):
            self.events.append((event, fields))

    logger = RecordingLogger()
    monkeypatch.setattr("mindmemos.pipelines.add.structured.planner.logger", logger)
    decider = StructuredMergeDecider(llm_client=llm, max_candidates=3)

    decisions = await decider.decide_batch(
        [StructuredMergeRequest(property=item, candidates=[memory("old-a", "Gateway A is offline.")])],
        current_episode={"episode_id": "episode-current"},
        episode_contexts={},
    )

    assert decisions[0].action == "create"
    assert logger.events == [
        (
            "structured_batch_merge_decisions_validated",
            {
                "supplied_decision_count": 1,
                "resolved_decision_count": 0,
                "invalid_reason_counts": {"missing_equivalent_ids": 1},
            },
        )
    ]


@pytest.mark.asyncio
async def test_contextual_merge_returns_one_canonical_target_and_bounded_equivalent_candidates():
    item = StructuredProperty(
        entity_name="Adaptive mutation",
        entity_type="task_experience",
        property_name="strategy",
        content="Increase mutation after stagnation.",
        property_time="2026-08-03",
        fingerprint="new",
        memory_id="new-memory",
        entity_id="new-entity",
        vector=[1.0],
    )
    llm = QueueMergeLLM(
        json.dumps(
            {
                "decisions": [
                    {
                        "group_index": 0,
                        "action": "reinforce",
                        "target_id": "old-a",
                        "equivalent_ids": ["old-a", "old-b", "old-b"],
                        "reason": "same lesson in the same Episode",
                    }
                ]
            }
        )
    )
    decider = StructuredMergeDecider(llm_client=llm, max_candidates=3)

    decisions = await decider.decide_batch(
        [
            StructuredMergeRequest(
                property=item,
                candidates=[
                    memory("old-a", "Raise mutation when progress stalls."),
                    memory("old-b", "Increase mutation after stagnant generations."),
                    memory("old-c", "Use elitism to preserve the best candidate."),
                    memory("old-d", "This candidate is outside the prompt bound."),
                ],
            )
        ],
        current_episode={"episode_id": "episode-current", "description": "One optimization task"},
        episode_contexts={},
    )

    assert decisions[0].action == "reinforce"
    assert decisions[0].target_id == "old-a"
    assert decisions[0].equivalent_ids == ["old-b"]
    prompt = llm.calls[0][1][0]["content"]
    assert "equivalent_ids" in prompt
    assert "old-d" not in prompt


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "equivalent_ids",
    [
        ["not-a-candidate"],
        ["old-b", "not-a-candidate"],
        "old-b",
    ],
)
async def test_invalid_equivalent_candidate_ids_fall_back_to_create(equivalent_ids):
    item = StructuredProperty(
        entity_name="Adaptive mutation",
        entity_type="task_experience",
        property_name="strategy",
        content="Increase mutation after stagnation.",
        property_time="2026-08-03",
        fingerprint="new",
        memory_id="new-memory",
        entity_id="new-entity",
        vector=[1.0],
    )
    llm = QueueMergeLLM(
        json.dumps(
            {
                "decisions": [
                    {
                        "group_index": 0,
                        "action": "reinforce",
                        "target_id": "old-a",
                        "equivalent_ids": equivalent_ids,
                    }
                ]
            }
        )
    )
    decider = StructuredMergeDecider(llm_client=llm, max_candidates=3)

    decisions = await decider.decide_batch(
        [
            StructuredMergeRequest(
                property=item,
                candidates=[memory("old-a", "old-a"), memory("old-b", "old-b")],
            )
        ],
        current_episode={"episode_id": "episode-current"},
        episode_contexts={},
    )

    assert decisions[0].action == "create"
