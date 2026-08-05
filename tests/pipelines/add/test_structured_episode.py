from __future__ import annotations

from datetime import UTC, datetime

import pytest
from mindmemos.pipelines.add.structured.episode import (
    StructuredEpisodeCandidate,
    default_structured_episode_decision,
    recall_structured_episode_candidates,
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
