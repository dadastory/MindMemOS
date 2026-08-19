from __future__ import annotations

from datetime import UTC, datetime

import pytest
from mindmemos.pipelines.add.structured.episode import (
    StructuredBatchBlock,
    StructuredBatchEpisodeAllocator,
    StructuredEpisodeCandidate,
    default_structured_episode_decision,
    recall_structured_episode_candidates,
    structured_batch_episode_id,
)
from mindmemos.typing import EntitySearchHit, EntitySearchResult, EntityView, MemoryRequestContext


def _context() -> MemoryRequestContext:
    return MemoryRequestContext(
        request_id="request-1",
        account_id="account-1",
        project_id="project-1",
        api_key_uuid="key-1",
        user_id="user-1",
        app_id="app-1",
        agent_id="agent-1",
        session_id="session-current",
        memory_algorithm="structured",
    )


class _Embed:
    async def embed(self, *, task, text, **_kwargs):
        assert task == "memory.add.structured_episode_recall"
        assert text == ["candidate evidence"]
        return type("Response", (), {"embeddings": [[0.1, 0.2]]})()


class _Reader:
    def __init__(self):
        self.calls = []

    async def search_entities_dense(self, context, **kwargs):
        self.calls.append((context, kwargs))
        return EntitySearchResult(
            query=kwargs["query"],
            hits=[
                EntitySearchHit(
                    entity_id="episode-related",
                    score=0.96,
                    entity=EntityView(
                        entity_id="episode-related",
                        project_id="project-1",
                        entity_name="Related experiment",
                        entity_type="episodes",
                        description="A related historical experiment",
                        user_id="user-1",
                        session_id="session-old",
                        created_at=datetime(2026, 8, 1, tzinfo=UTC),
                    ),
                ),
                EntitySearchHit(
                    entity_id="episode-current",
                    score=0.91,
                    entity=EntityView(
                        entity_id="episode-current",
                        project_id="project-1",
                        entity_name="Current experiment",
                        entity_type="episodes",
                        description="The current execution process",
                        user_id="user-1",
                        session_id="session-current",
                        created_at=datetime(2026, 8, 2, tzinfo=UTC),
                    ),
                ),
            ],
        )


@pytest.mark.asyncio
async def test_episode_candidates_are_recalled_in_tenant_scope_and_current_session_is_prioritized():
    reader = _Reader()

    result = await recall_structured_episode_candidates(
        reader,
        _Embed(),
        _context(),
        "candidate evidence",
        top_k=5,
    )

    assert [candidate.episode_id for candidate in result] == ["episode-current", "episode-related"]
    assert result[0].same_session is True
    assert len(reader.calls) == 2
    _, request = reader.calls[0]
    conditions = {(item.field, item.value) for item in request["filters"].must}
    assert ("entity_type", "episodes") in conditions
    assert ("user_id", "user-1") in conditions
    assert ("app_id", "app-1") in conditions
    assert ("agent_id", "agent-1") in conditions
    assert request["limit"] == 5
    _, session_request = reader.calls[1]
    assert any(
        item.field == "session_id" and item.value == "session-current" for item in session_request["filters"].must
    )


@pytest.mark.asyncio
async def test_episode_recall_excludes_low_similarity_context_and_requests_active_rows_only():
    class LowScoreReader:
        def __init__(self):
            self.filters = []

        async def search_entities_dense(self, context, **kwargs):
            self.filters.append(kwargs["filters"])
            return EntitySearchResult(
                query=kwargs["query"],
                hits=[
                    EntitySearchHit(
                        entity_id="stale-test-episode",
                        score=0.61,
                        entity=EntityView(
                            entity_id="stale-test-episode",
                            project_id="project-1",
                            entity_name="Old deletion test",
                            entity_type="episodes",
                            description="Unrelated stale context",
                            user_id="user-1",
                            session_id="session-current",
                            created_at=datetime(2026, 8, 1, tzinfo=UTC),
                        ),
                    )
                ],
            )

    reader = LowScoreReader()
    result = await recall_structured_episode_candidates(
        reader,
        _Embed(),
        _context(),
        "candidate evidence",
        top_k=5,
        reuse_at_or_above=0.82,
    )

    assert result == []
    assert all(
        any(condition.field == "status" and condition.value == "active" for condition in filters.must)
        for filters in reader.filters
    )


def test_no_candidate_falls_back_to_new_episode_without_caller_input():
    decision = default_structured_episode_decision(
        content="candidate evidence",
        event_time="2026-08-04T10:00:00+00:00",
        candidates=[],
    )

    assert decision.action == "create"
    assert decision.target_episode_id is None
    assert decision.title == "candidate evidence"
    assert decision.description == "candidate evidence"


def test_default_reuse_only_accepts_a_whitelisted_candidate():
    candidate = StructuredEpisodeCandidate(
        episode_id="episode-1",
        title="Current task",
        description="Current task background",
        score=0.9,
        session_id="session-current",
        same_session=True,
    )

    decision = default_structured_episode_decision(
        content="continuing evidence",
        event_time="2026-08-04T10:00:00+00:00",
        candidates=[candidate],
    )

    assert decision.action == "reuse"
    assert decision.target_episode_id == "episode-1"
    assert decision.related_episode_ids == []


class _Chat:
    def __init__(self, values):
        self.values = list(values)
        self.calls = []

    async def chat(self, **kwargs):
        self.calls.append(kwargs)
        return type("Response", (), {"content": self.values.pop(0)})()


@pytest.mark.asyncio
async def test_batch_episode_allocator_derives_membership_from_closed_assignment_map():
    first_id = "opaque-first"
    second_id = "opaque-second"
    allocator = StructuredBatchEpisodeAllocator(
        llm_client=_Chat(
            [
                '{"groups":[{"group_key":"shell-sort","action":"create",'
                '"title":"Shell sort","description":"One coherent background",'
                '"related_episode_ids":[]}],'
                '"assignments":{"b1":"shell-sort","b2":"shell-sort"}}'
            ]
        )
    )
    blocks = [
        StructuredBatchBlock(block_id=block_id, content=block_id, extracted={"entities": [{"name": block_id}]})
        for block_id in (first_id, second_id)
    ]

    groups = await allocator.allocate(blocks, [])

    assert groups[0].block_ids == [first_id, second_id]


@pytest.mark.asyncio
async def test_batch_episode_allocator_discards_unreferenced_group_definitions():
    allocator = StructuredBatchEpisodeAllocator(
        llm_client=_Chat(
            [
                '{"groups":['
                '{"group_key":"used","action":"create","title":"Used",'
                '"description":"The assigned background","related_episode_ids":[]},'
                '{"group_key":"unused","action":"create","title":"Unused",'
                '"description":"No block selected this background","related_episode_ids":[]}'
                '],"assignments":{"b1":"used","b2":"used"}}'
            ]
        )
    )
    blocks = [
        StructuredBatchBlock(block_id=block_id, content=block_id, extracted={"entities": [{"name": block_id}]})
        for block_id in ("first", "second")
    ]

    groups = await allocator.allocate(blocks, [])

    assert [group.group_key for group in groups] == ["used"]
    assert groups[0].block_ids == ["first", "second"]


@pytest.mark.asyncio
async def test_batch_episode_allocator_covers_unordered_blocks_once_and_can_create_multiple_groups():
    allocator = StructuredBatchEpisodeAllocator(
        llm_client=_Chat(
            [
                '{"groups":['
                '{"group_key":"travel","action":"create",'
                '"title":"Trip","description":"One trip","related_episode_ids":[]},'
                '{"group_key":"device","action":"create",'
                '"title":"Device","description":"One device","related_episode_ids":[]}'
                '],"assignments":{"b1":"travel","b2":"device","b3":"travel"}}'
            ]
        )
    )
    blocks = [
        StructuredBatchBlock(block_id=block_id, content=block_id, extracted={"entities": [{"name": block_id}]})
        for block_id in ["b1", "b2", "b3"]
    ]

    groups = await allocator.allocate(blocks, [])

    assert [group.group_key for group in groups] == ["travel", "device"]
    assert {block_id for group in groups for block_id in group.block_ids} == {"b1", "b2", "b3"}


@pytest.mark.asyncio
async def test_batch_episode_allocator_rejects_unknown_group_assignment_after_one_repair():
    invalid = (
        '{"groups":[{"group_key":"one","action":"create",'
        '"title":"One","description":"One","related_episode_ids":[]}],'
        '"assignments":{"b1":"unknown"}}'
    )
    allocator = StructuredBatchEpisodeAllocator(llm_client=_Chat([invalid, invalid]), max_repair_attempts=1)
    blocks = [StructuredBatchBlock(block_id="b1", content="one", extracted={"entities": [{"name": "one"}]})]

    with pytest.raises(Exception, match="allocation failed validation"):
        await allocator.allocate(blocks, [])


@pytest.mark.asyncio
async def test_batch_episode_allocator_repair_identifies_missing_alias_and_unknown_group():
    invalid = (
        '{"groups":['
        '{"group_key":"one","action":"create",'
        '"title":"One","description":"One","related_episode_ids":[]},'
        '{"group_key":"two","action":"create",'
        '"title":"Two","description":"Two","related_episode_ids":[]}'
        '],"assignments":{"b1":"missing-group","b3":"two"}}'
    )
    repaired = (
        '{"groups":[{"group_key":"all","action":"create",'
        '"title":"All","description":"Complete background","related_episode_ids":[]}],'
        '"assignments":{"b1":"all","b2":"all"}}'
    )
    chat = _Chat([invalid, repaired])
    allocator = StructuredBatchEpisodeAllocator(llm_client=chat, max_repair_attempts=1)
    blocks = [
        StructuredBatchBlock(block_id=block_id, content=block_id, extracted={"entities": [{"name": block_id}]})
        for block_id in ["b1", "b2"]
    ]

    groups = await allocator.allocate(blocks, [])

    assert groups[0].block_ids == ["b1", "b2"]
    repair_prompt = chat.calls[1]["messages"][0]["content"]
    assert 'Allowed assignment aliases: ["b1", "b2"]' in repair_prompt
    assert 'Received assignment aliases: ["b1", "b3"]' in repair_prompt
    assert 'Unknown assignment aliases: ["b3"]' in repair_prompt
    assert 'Missing assignment aliases: ["b2"]' in repair_prompt
    assert 'Unknown assignment group keys: ["missing-group"]' in repair_prompt
    assert 'Unused Episode group keys: ["one", "two"]' in repair_prompt


@pytest.mark.asyncio
async def test_batch_episode_allocator_uses_short_aliases_and_restores_opaque_block_ids():
    first_id = "903d3d9a-0b6e-419d-9feb-4293e9d5a550"
    second_id = "3e55265f-1768-4401-8ea7-9520aa5a7208"
    chat = _Chat(
        [
            '{"groups":[{"group_key":"shell-sort","action":"create",'
            '"title":"Shell sort","description":"One sorting-algorithm background",'
            '"related_episode_ids":[]}],"assignments":{"b1":"shell-sort","b2":"shell-sort"}}'
        ]
    )
    allocator = StructuredBatchEpisodeAllocator(llm_client=chat)
    blocks = [
        StructuredBatchBlock(
            block_id=block_id,
            document_id=block_id,
            content=content,
            locator={
                "parse_run_id": "run-uuid",
                "title": title,
                "sort_order": index,
                "nested": {"document_id": block_id, "section": "algorithm-notes"},
            },
            extracted={
                "entities": [
                    {
                        "name": f"{block_id}\x1f{title}",
                        "_display_name": title,
                        "_source_block_id": block_id,
                        "entity_type": "knowledge",
                        "description": content,
                        "properties": [{"property_name": "content", "value": content}],
                    }
                ],
                "edges": [],
            },
        )
        for index, (block_id, title, content) in enumerate(
            [
                (first_id, "Shell sort algorithm", "Gap-based insertion sorting."),
                (second_id, "Shell sort complexity", "Complexity depends on the gap sequence."),
            ]
        )
    ]

    groups = await allocator.allocate(blocks, [])

    assert groups[0].block_ids == [first_id, second_id]
    prompt = chat.calls[0]["messages"][0]["content"]
    assert '"block_alias": "b1"' in prompt
    assert '"block_alias": "b2"' in prompt
    assert first_id not in prompt
    assert second_id not in prompt
    assert "run-uuid" not in prompt
    assert "\x1f" not in prompt
    assert "Shell sort algorithm" in prompt
    assert "algorithm-notes" in prompt


@pytest.mark.asyncio
async def test_batch_episode_allocator_repair_describes_every_invalid_alias_dimension():
    invalid = (
        '{"groups":[{"group_key":"bad","action":"create",'
        '"title":"Bad","description":"Invalid aliases","related_episode_ids":[]}],'
        '"assignments":{"":"bad","block_1":"bad","b1":"missing-group","b3":"bad"}}'
    )
    repaired = (
        '{"groups":[{"group_key":"all","action":"create",'
        '"title":"All","description":"Complete background","related_episode_ids":[]}],'
        '"assignments":{"b1":"all","b2":"all"}}'
    )
    chat = _Chat([invalid, repaired])
    allocator = StructuredBatchEpisodeAllocator(llm_client=chat, max_repair_attempts=1)
    blocks = [
        StructuredBatchBlock(
            block_id=f"opaque-document-{index}",
            content=f"content-{index}",
            extracted={"entities": [{"name": f"entity-{index}"}]},
        )
        for index in (1, 2)
    ]

    groups = await allocator.allocate(blocks, [])

    assert groups[0].block_ids == ["opaque-document-1", "opaque-document-2"]
    repair_prompt = chat.calls[1]["messages"][0]["content"]
    assert 'Allowed assignment aliases: ["b1", "b2"]' in repair_prompt
    assert 'Received assignment aliases: ["", "block_1", "b1", "b3"]' in repair_prompt
    assert 'Unknown assignment aliases: ["b3", "block_1"]' in repair_prompt
    assert 'Empty assignment aliases: [""]' in repair_prompt
    assert 'Missing assignment aliases: ["b2"]' in repair_prompt
    assert 'Unknown assignment group keys: ["missing-group"]' in repair_prompt
    assert 'Unused Episode group keys: ["bad"]' in repair_prompt
    assert "assignments must be one closed JSON object" in repair_prompt
    assert "assignments must contain exactly 2 keys" in repair_prompt
    assert "put secondary historical Episodes in related_episode_ids" in repair_prompt


@pytest.mark.asyncio
async def test_later_batch_can_reuse_one_whitelisted_episode_and_create_another():
    candidate = StructuredEpisodeCandidate(
        episode_id="episode-old",
        title="Existing background",
        description="Existing context",
        score=0.95,
    )
    allocator = StructuredBatchEpisodeAllocator(
        llm_client=_Chat(
            [
                '{"groups":['
                '{"group_key":"old","action":"reuse",'
                '"target_episode_id":"episode-old","related_episode_ids":[]},'
                '{"group_key":"new","action":"create",'
                '"title":"New background","description":"Independent context","related_episode_ids":[]}'
                '],"assignments":{"b1":"old","b2":"new"}}'
            ]
        )
    )
    blocks = [
        StructuredBatchBlock(block_id=block_id, content=block_id, extracted={"entities": [{"name": block_id}]})
        for block_id in ["b1", "b2"]
    ]

    groups = await allocator.allocate(blocks, [candidate])

    assert [(group.action, group.target_episode_id) for group in groups] == [
        ("reuse", "episode-old"),
        ("create", None),
    ]


def test_batch_episode_id_is_stable_but_independent_of_generated_title():
    first = structured_batch_episode_id(_context(), batch_key="batch-1", group_key="group-a")
    replay = structured_batch_episode_id(_context(), batch_key="batch-1", group_key="group-a")
    separate_group = structured_batch_episode_id(_context(), batch_key="batch-1", group_key="group-b")

    assert first == replay
    assert first != separate_group
