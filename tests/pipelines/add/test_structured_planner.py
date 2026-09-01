import asyncio
import json
from datetime import UTC, datetime

import pytest
from mindmemos.config import StructuredDedupConfig
from mindmemos.llm import ChatResponse, EmbeddingResponse
from mindmemos.pipelines.add.structured.planner import (
    StructuredDecision,
    StructuredMergeDecider,
    StructuredMergeRequest,
    StructuredProperty,
    batch_embed,
    classify_similarity,
    consolidate_structured_batch,
    consolidate_structured_properties,
    recall_structured_candidates,
)
from mindmemos.structured_content import merge_structured_contents
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


def typed(description: str, *facts: str):
    return {"description": description, "content": list(facts)}


def typed_memory(memory_id: str, description: str, *facts: str, score=0.9):
    hit = memory(memory_id, "\n".join(facts), score=score)
    hit.memory.metadata["structured_content"] = typed(description, *facts)
    return hit


class RecordingEmbed:
    def __init__(self):
        self.batches = []

    async def embed(self, task, text, **kwargs):
        values = [text] if isinstance(text, str) else list(text)
        self.batches.append((task, values))
        return EmbeddingResponse(embeddings=[[float(len(value))] for value in values])


def test_deterministic_structured_union_keeps_anchor_description_fact_order_and_exact_wording():
    result = merge_structured_contents(
        [
            {
                "description": "Original anchor.",
                "content": ["Fact A with value 5.", "Shared fact."],
            },
            {
                "description": "A model must not replace the anchor.",
                "content": ["Shared fact.", "Fact B under condition C."],
            },
        ]
    )

    assert result == {
        "description": "Original anchor.",
        "content": ["Fact A with value 5.", "Shared fact.", "Fact B under condition C."],
    }


@pytest.mark.asyncio
async def test_batch_embed_chunks_and_preserves_input_order():
    embed = RecordingEmbed()

    result = await batch_embed(embed, ["a", "bb", "ccc", "dddd", "eeeee"], batch_size=2, task="structured")

    assert [len(values) for _, values in embed.batches] == [2, 2, 1]
    assert result == [[1.0], [2.0], [3.0], [4.0], [5.0]]


@pytest.mark.asyncio
async def test_batch_internal_consolidation_rejects_model_authored_content_and_preserves_sources():
    first = StructuredProperty(
        entity_name="Device A",
        entity_type="task_experience",
        property_name="strategy",
        content="Device A retries twice.",
        property_time="2026-08-03",
        fingerprint="a",
        memory_id="memory-a",
        entity_id="entity-a",
        entity_keys=["block-a\x1fDevice A"],
        source_block_ids=["block-a"],
        source_documents=[{"block_id": "block-a", "document_id": "doc-1"}],
        episode_id="episode-1",
    )
    second = StructuredProperty(
        entity_name="Device A",
        entity_type="task_experience",
        property_name="strategy",
        content="Device A waits 5 seconds between retries.",
        property_time="2026-08-03",
        fingerprint="b",
        memory_id="memory-b",
        entity_id="entity-b",
        entity_keys=["block-b\x1fDevice A"],
        source_block_ids=["block-b"],
        source_documents=[{"block_id": "block-b", "document_id": "doc-2"}],
        episode_id="episode-1",
    )
    llm = QueueMergeLLM(
        json.dumps(
            {
                "clusters": [
                    {
                        "member_indexes": [0, 1],
                        "canonical_content": "Device A retries twice and waits 5 seconds between retries.",
                    }
                ]
            }
        )
    )

    result = await consolidate_structured_batch(
        llm,
        [first, second],
        entity_schema=[],
        episode_contexts={"episode-1": {"episode_id": "episode-1"}},
    )

    assert len(result) == 2
    assert [item.content for item in result] == [
        "Device A retries twice.",
        "Device A waits 5 seconds between retries.",
    ]


@pytest.mark.asyncio
async def test_batch_internal_consolidation_accepts_lossless_typed_card_content():
    first = StructuredProperty(
        entity_name="Device A",
        entity_type="task_experience",
        property_name="strategy",
        content={"description": "Retry behavior.", "content": ["Device A retries twice."]},
        property_time="2026-08-03",
        fingerprint="a",
        memory_id="memory-a",
        entity_id="entity-a",
        source_block_ids=["block-a"],
        episode_id="episode-1",
    )
    second = StructuredProperty(
        entity_name="Device A",
        entity_type="task_experience",
        property_name="strategy",
        content={
            "description": "Retry timing.",
            "content": ["Device A waits 5 seconds between retries."],
        },
        property_time="2026-08-03",
        fingerprint="b",
        memory_id="memory-b",
        entity_id="entity-b",
        source_block_ids=["block-b"],
        episode_id="episode-1",
    )
    llm = QueueMergeLLM(
        json.dumps(
            {
                "fact_groups": [
                    {
                        "relation": "complement",
                        "member_indexes": [0, 1],
                    }
                ]
            }
        )
    )

    result = await consolidate_structured_batch(
        llm,
        [first, second],
        entity_schema=[],
        episode_contexts={"episode-1": {"episode_id": "episode-1"}},
    )

    assert [item.content for item in result] == [
        {
            "description": "Retry behavior.",
            "content": [
                "Device A retries twice.",
                "Device A waits 5 seconds between retries.",
            ],
        }
    ]
    assert len(llm.calls) == 1


@pytest.mark.asyncio
async def test_batch_internal_duplicate_group_preserves_both_paraphrased_source_facts():
    first = StructuredProperty(
        entity_name="Shell sort",
        entity_type="knowledge",
        property_name="content",
        content=typed("Increment policy.", "The final increment must be 1."),
        property_time="2026-08-18",
        fingerprint="a",
        memory_id="memory-a",
        entity_id="entity-a",
        source_block_ids=["block-a"],
        episode_id="episode-1",
    )
    second = StructuredProperty(
        entity_name="Shell sort",
        entity_type="knowledge",
        property_name="content",
        content=typed("Gap policy.", "The increment sequence must end with a gap of 1."),
        property_time="2026-08-18",
        fingerprint="b",
        memory_id="memory-b",
        entity_id="entity-b",
        source_block_ids=["block-b"],
        episode_id="episode-1",
    )
    llm = QueueMergeLLM(
        json.dumps(
            {
                "fact_groups": [
                    {"relation": "duplicate", "member_indexes": [0, 1]},
                ]
            }
        )
    )

    result = await consolidate_structured_batch(
        llm,
        [first, second],
        entity_schema=[],
        episode_contexts={"episode-1": {"episode_id": "episode-1"}},
    )

    assert len(result) == 1
    assert result[0].content == typed(
        "Increment policy.",
        "The final increment must be 1.",
        "The increment sequence must end with a gap of 1.",
    )
    assert result[0].source_block_ids == ["block-a", "block-b"]


@pytest.mark.asyncio
async def test_batch_internal_relation_cannot_synthesize_unequal_scalar_values():
    properties = [
        StructuredProperty(
            entity_name="Shell sort",
            entity_type="knowledge",
            property_name="tags",
            content=value,
            property_time="2026-08-18",
            fingerprint=str(index),
            memory_id=f"memory-{index}",
            entity_id=f"entity-{index}",
            episode_id="episode-1",
        )
        for index, value in enumerate(["sorting, gap", "sorting, insertion"])
    ]
    llm = QueueMergeLLM(
        json.dumps({"fact_groups": [{"relation": "complement", "member_indexes": [0, 1]}]})
    )

    result = await consolidate_structured_batch(
        llm,
        properties,
        entity_schema=[],
        episode_contexts={"episode-1": {"episode_id": "episode-1"}},
        max_repair_attempts=0,
    )

    assert [item.content for item in result] == ["sorting, gap", "sorting, insertion"]


@pytest.mark.asyncio
async def test_batch_internal_consolidation_resolves_same_subject_across_different_properties():
    first = StructuredProperty(
        entity_name="Gateway A",
        entity_type="task_experience",
        property_name="strategy",
        content="Gateway A retries twice.",
        property_time="2026-08-03",
        fingerprint="a",
        memory_id="memory-a",
        entity_id="entity-a",
        entity_keys=["block-a\x1fGateway A"],
        source_block_ids=["block-a"],
        episode_id="episode-1",
    )
    second = StructuredProperty(
        entity_name="The gateway",
        entity_type="task_experience",
        property_name="outcome",
        content="Gateway A recovered after 10 seconds.",
        property_time="2026-08-03",
        fingerprint="b",
        memory_id="memory-b",
        entity_id="entity-b",
        entity_keys=["block-b\x1fThe gateway"],
        source_block_ids=["block-b"],
        episode_id="episode-1",
    )
    llm = QueueMergeLLM(
        json.dumps(
            {
                "entity_groups": [
                    {
                        "relation": "same_entity",
                        "member_entity_keys": ["block-a\u001fGateway A", "block-b\u001fThe gateway"],
                    }
                ],
                "fact_groups": [],
            }
        )
    )

    result = await consolidate_structured_batch(
        llm,
        [first, second],
        entity_schema=[],
        episode_contexts={"episode-1": {"episode_id": "episode-1"}},
    )

    assert len(result) == 2
    assert result[0].entity_id == result[1].entity_id == "entity-a"
    assert result[0].entity_name == result[1].entity_name == "Gateway A"


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


@pytest.mark.asyncio
async def test_batch_candidate_recall_uses_each_consolidated_property_episode_fence():
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
            episode_id=f"episode-{index}",
            comparison_episode_ids=[f"episode-{index}"],
        )
        for index in range(2)
    ]

    await recall_structured_candidates(reader, context(), items, episode_ids=None, top_k=3)

    primary = [
        req
        for _, req, _ in reader.requests
        if any(condition.field == "episode_ids" and condition.op == "any" for condition in req.filters.must)
    ]
    assert {
        tuple(next(condition.values for condition in req.filters.must if condition.field == "episode_ids"))
        for req in primary
    } == {("episode-0",), ("episode-1",)}


@pytest.mark.asyncio
async def test_candidate_recall_can_use_session_history_across_episode_boundaries():
    reader = ConcurrentReader()
    item = StructuredProperty(
        entity_name="Repeated task lesson",
        entity_type="llm4ad_memory_card",
        property_name="error_reflection",
        content="Avoid the same underperforming repair strategy.",
        property_time="2026-08-21",
        fingerprint="new",
        memory_id="new-memory",
        entity_id="new-entity",
        vector=[1.0],
        episode_id="episode-new",
        comparison_episode_ids=["episode-new"],
    )

    await recall_structured_candidates(
        reader,
        context(),
        [item],
        episode_ids=None,
        top_k=5,
        history_scope="session",
    )

    assert len(reader.requests) == 1
    request = reader.requests[0][1]
    assert any(
        condition.field == "session_id" and condition.op == "match" and condition.value == "task-1"
        for condition in request.filters.must
    )
    assert all(condition.field != "episode_ids" for condition in request.filters.must)


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
@pytest.mark.parametrize(
    ("relation", "action"),
    [(None, "create"), ("duplicate", "reinforce"), ("complement", "update"), ("conflict", "supersede")],
)
async def test_ambiguous_merge_derives_storage_action_from_relation(relation, action):
    payload = {"relation": relation, "reason": "fixture"}
    if relation is not None:
        payload["target_id"] = "memory-1"
    llm = QueueMergeLLM(json.dumps(payload))
    decider = StructuredMergeDecider(llm_client=llm, max_candidates=3)

    incoming = typed("Incoming strategy.", "New fact.")
    candidates = [typed_memory(f"memory-{i}", f"Historical {i}.", f"Old fact {i}.") for i in range(1, 6)]
    decision = await decider.decide(incoming, candidates)

    assert decision.action == action
    if relation == "complement":
        assert decision.merged_content == typed("Historical 1.", "Old fact 1.", "New fact.")
    elif relation == "conflict":
        assert decision.merged_content == incoming
    else:
        assert decision.merged_content is None
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
        '{"relation":"unknown"}',
        '{"relation":"duplicate","target_id":"not-a-candidate"}',
        '{"relation":"complement","target_id":"memory-1","merged_content":{"description":"bad","content":["bad"]}}',
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
async def test_batch_merge_deterministically_materializes_complementary_typed_content():
    item = StructuredProperty(
        entity_name="gateway-a",
        entity_type="device",
        property_name="status",
        content={"description": "Gateway status.", "content": ["Gateway is offline."]},
        property_time="2026-08-03",
        fingerprint="a",
        memory_id="new-a",
        entity_id="entity-a",
    )
    llm = QueueMergeLLM(
        json.dumps(
            {
                "decisions": [
                    {
                        "group_index": 0,
                        "relation": "complement",
                        "target_id": "memory-1",
                        "reason": "same gateway",
                    }
                ]
            }
        )
    )
    decider = StructuredMergeDecider(llm_client=llm, max_candidates=3)
    historical = memory("memory-1", "Gateway is unreachable.")
    historical.memory.metadata["structured_content"] = {
        "description": "Earlier gateway status.",
        "content": ["Gateway is unreachable."],
    }

    decisions = await decider.decide_batch(
        [
            StructuredMergeRequest(
                property=item,
                candidates=[historical],
            )
        ],
        current_episode={},
        episode_contexts={},
    )

    assert decisions[0].action == "update"
    assert decisions[0].merged_content == {
        "description": "Earlier gateway status.",
        "content": ["Gateway is unreachable.", "Gateway is offline."],
    }


@pytest.mark.asyncio
async def test_batch_merge_rejects_any_model_authored_merged_content():
    item = StructuredProperty(
        entity_name="gateway-a",
        entity_type="device",
        property_name="status",
        content={"description": "Gateway status.", "content": ["Gateway is offline."]},
        property_time="2026-08-03",
        fingerprint="a",
        memory_id="new-a",
        entity_id="entity-a",
    )
    llm = QueueMergeLLM(
        json.dumps(
            {
                "decisions": [
                    {
                        "group_index": 0,
                        "relation": "complement",
                        "target_id": "memory-1",
                        "merged_content": {
                            "description": "Gateway status.",
                            "content": ["Gateway is offline."],
                        },
                        "reason": "lossy",
                    }
                ]
            }
        )
    )
    historical = memory("memory-1", "Gateway is unreachable.")
    historical.memory.metadata["structured_content"] = {
        "description": "Earlier gateway status.",
        "content": ["Gateway is unreachable."],
    }
    decider = StructuredMergeDecider(llm_client=llm, max_candidates=3)

    decisions = await decider.decide_batch(
        [StructuredMergeRequest(property=item, candidates=[historical])],
        current_episode={},
        episode_contexts={},
    )

    assert decisions == [StructuredDecision(action="create", reason="invalid_or_unavailable_batch_merge")]


@pytest.mark.asyncio
async def test_all_ambiguous_groups_are_decided_in_one_call_with_episode_context():
    first = StructuredProperty(
        entity_name="gateway-a",
        entity_type="device",
        property_name="status",
        content=typed("Gateway state.", "Gateway is offline."),
        property_time="2026-08-03",
        fingerprint="a",
        memory_id="new-a",
        entity_id="entity-a",
        entity_description="Primary warehouse gateway",
        vector=[1.0],
        batch_contents=[
            typed("Gateway state.", "Gateway is offline."),
            typed("Gateway reachability.", "The gateway cannot be reached."),
        ],
    )
    second = StructuredProperty(
        entity_name="ticket-42",
        entity_type="ticket",
        property_name="state",
        content=typed("Ticket state.", "Ticket is open."),
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
                    "relation": "duplicate",
                    "target_id": "old-a",
                    "reason": "same subject, fact, and context",
                },
            ]
        }
    )
    llm = QueueMergeLLM(response)
    decider = StructuredMergeDecider(llm_client=llm, max_candidates=3)
    requests = [
        StructuredMergeRequest(
            property=first,
            candidates=[typed_memory("old-a", "Stored gateway state.", "Gateway is unreachable.")],
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
            candidates=[typed_memory("old-b", "Stored ticket state.", "Ticket is pending.")],
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
    assert "relation=duplicate|complement|conflict" in lowered
    assert "do not return an action" in lowered
    assert "merged_content" in lowered
    assert "backend code derives" in lowered
    assert "not fingerprints or uniqueness keys" in lowered
    assert "allowed_memory_ids" in prompt
    assert "copy target_id and duplicate_ids only" in lowered
    assert "never use an entity_id" in lowered


@pytest.mark.asyncio
async def test_one_extracted_entity_cannot_merge_into_two_historical_entities():
    first = StructuredProperty(
        entity_name="Gateway A",
        entity_type="device",
        property_name="status",
        content=typed("Gateway state.", "Gateway A is offline."),
        property_time="2026-08-03",
        fingerprint="first",
        memory_id="new-first",
        entity_id="one-extracted-entity",
        vector=[1.0],
    )
    second = StructuredProperty(
        entity_name="Gateway A",
        entity_type="device",
        property_name="location",
        content=typed("Gateway location.", "Gateway A is in warehouse 2."),
        property_time="2026-08-03",
        fingerprint="second",
        memory_id="new-second",
        entity_id="one-extracted-entity",
        vector=[1.0],
    )
    llm = QueueMergeLLM(
        json.dumps(
            {
                "decisions": [
                    {
                        "group_index": 0,
                        "relation": "duplicate",
                        "target_id": "old-status",
                    },
                    {
                        "group_index": 1,
                        "relation": "duplicate",
                        "target_id": "old-location",
                    },
                ]
            }
        )
    )
    decider = StructuredMergeDecider(llm_client=llm, max_candidates=3)

    decisions = await decider.decide_batch(
        [
            StructuredMergeRequest(
                property=first,
                candidates=[typed_memory("old-status", "Stored gateway state.", "Gateway is offline.")],
                candidate_entity_contexts={"old-status": {"entity_id": "stored-gateway-a"}},
            ),
            StructuredMergeRequest(
                property=second,
                candidates=[typed_memory("old-location", "Stored location.", "Located in warehouse 2.")],
                candidate_entity_contexts={"old-location": {"entity_id": "stored-gateway-b"}},
            ),
        ],
        current_episode={"episode_id": "episode-current"},
        episode_contexts={},
    )

    assert [decision.action for decision in decisions] == ["create", "create"]
    assert {decision.reason for decision in decisions} == {"conflicting_historical_entity_resolution"}


@pytest.mark.asyncio
async def test_batch_merge_rejects_model_authored_storage_action():
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
                    "invalid_reason_counts": {"invalid_relation_decision": 1},
            },
        )
    ]


@pytest.mark.asyncio
async def test_contextual_merge_returns_one_canonical_target_and_bounded_equivalent_candidates():
    item = StructuredProperty(
        entity_name="Adaptive mutation",
        entity_type="task_experience",
        property_name="strategy",
        content=typed("Mutation policy.", "Increase mutation after stagnation."),
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
                        "relation": "duplicate",
                        "target_id": "old-a",
                        "duplicate_ids": ["old-a", "old-b", "old-b"],
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
                    typed_memory("old-a", "Stored mutation policy.", "Raise mutation when progress stalls."),
                    typed_memory("old-b", "Stored mutation policy.", "Increase mutation after stagnant generations."),
                    typed_memory("old-c", "Elitism.", "Use elitism to preserve the best candidate."),
                    typed_memory("old-d", "Outside.", "This candidate is outside the prompt bound."),
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
    assert "duplicate_ids" in prompt
    assert "old-d" not in prompt


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "duplicate_ids",
    [
        ["not-a-candidate"],
        ["old-b", "not-a-candidate"],
        "old-b",
    ],
)
async def test_invalid_duplicate_candidate_ids_fall_back_to_create(duplicate_ids):
    item = StructuredProperty(
        entity_name="Adaptive mutation",
        entity_type="task_experience",
        property_name="strategy",
        content=typed("Mutation policy.", "Increase mutation after stagnation."),
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
                        "relation": "duplicate",
                        "target_id": "old-a",
                        "duplicate_ids": duplicate_ids,
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
                    typed_memory("old-a", "Old A.", "old-a"),
                    typed_memory("old-b", "Old B.", "old-b"),
                ],
            )
        ],
        current_episode={"episode_id": "episode-current"},
        episode_contexts={},
    )

    assert decisions[0].action == "create"
