from __future__ import annotations

from datetime import UTC, datetime
from types import SimpleNamespace

import mindmemos.pipelines.search.structured.engine as structured_engine_module
import pytest
from mindmemos.config import TextProcessingConfig
from mindmemos.pipelines.search.structured import StructuredSearchEngine
from mindmemos.typing.llm import EmbeddingResponse
from mindmemos.typing.memory import MemoryRequestContext, MemoryView
from mindmemos.typing.memory_db import MemoryDbSearchHit, MemoryDbSearchResult
from mindmemos.typing.service import SearchPipelineInput


def _context() -> MemoryRequestContext:
    return MemoryRequestContext(
        request_id="req-1",
        account_id="acc-1",
        project_id="project-1",
        api_key_uuid="key-1",
        user_id="user-1",
        app_id="llm4ad",
        session_id="task-1",
        agent_id="task",
    )


def _memory(
    memory_id: str,
    content: str,
    *,
    entity_type: str = "task_experience",
    property_name: str = "strategy",
    entity_id: str | None = None,
    episode_ids: list[str] | None = None,
    status: str = "active",
    user_id: str = "user-1",
) -> MemoryView:
    return MemoryView(
        memory_id=memory_id,
        project_id="project-1",
        content=content,
        mem_type="fact",
        mem_extract_type="structured",
        status=status,
        metadata={"source": "integration", "generation": 7},
        user_id=user_id,
        app_id="llm4ad",
        session_id="task-1",
        agent_id="task",
        entity_id=entity_id or f"entity-{memory_id}",
        entity_type=entity_type,
        property_name=property_name,
        episode_ids=episode_ids or [],
        validate_from=datetime(2026, 8, 17, tzinfo=UTC),
        created_at=datetime(2026, 8, 17, 1, 2, 3, tzinfo=UTC),
    )


class _Embed:
    async def embed(self, *, task: str, text):
        assert task == "search.structured_query"
        return EmbeddingResponse(embeddings=[[1.0, 0.0]])


class _FailingEmbed:
    async def embed(self, *, task: str, text):
        raise RuntimeError("embedding unavailable")


class _Reader:
    def __init__(
        self,
        hits: list[MemoryDbSearchHit],
        *,
        dense_hits: list[MemoryDbSearchHit] | None = None,
        related: list[dict[str, str]] | None = None,
        expanded: list[MemoryView] | None = None,
    ) -> None:
        self.hits = hits
        self.dense_hits = hits if dense_hits is None else dense_hits
        self.related = related or []
        self.expanded = expanded or []
        self.dense_calls: list[SimpleNamespace] = []
        self.hybrid_calls: list[SimpleNamespace] = []
        self.sparse_calls: list[SimpleNamespace] = []
        self.graph_calls: list[SimpleNamespace] = []
        self.filter_calls: list[SimpleNamespace] = []

    async def search_hybrid(
        self,
        context,
        query,
        *,
        dense_vector,
        sparse_vector,
        dense_limit,
        sparse_limit,
    ):
        self.hybrid_calls.append(
            SimpleNamespace(
                context=context,
                query=query,
                dense_vector=dense_vector,
                sparse_vector=sparse_vector,
                dense_limit=dense_limit,
                sparse_limit=sparse_limit,
            )
        )
        return MemoryDbSearchResult(query=query.query, hits=self.hits, total=len(self.hits))

    async def search_dense(self, context, query, *, query_vector, score_threshold=None):
        self.dense_calls.append(
            SimpleNamespace(
                context=context,
                query=query,
                query_vector=query_vector,
                score_threshold=score_threshold,
            )
        )
        hits = [hit for hit in self.dense_hits if score_threshold is None or hit.score >= score_threshold]
        return MemoryDbSearchResult(query=query.query, hits=hits, total=len(hits))

    async def search_sparse(self, context, query, *, indices, values):
        self.sparse_calls.append(SimpleNamespace(context=context, query=query, indices=indices, values=values))
        return MemoryDbSearchResult(query=query.query, hits=self.hits, total=len(self.hits))

    async def list_structured_related_memory_ids(
        self,
        context,
        memory_ids,
        *,
        limit_per_memory,
        max_candidates,
    ):
        self.graph_calls.append(
            SimpleNamespace(
                context=context,
                memory_ids=memory_ids,
                limit_per_memory=limit_per_memory,
                max_candidates=max_candidates,
            )
        )
        return self.related

    async def search_by_filter(self, context, query):
        self.filter_calls.append(SimpleNamespace(context=context, query=query))
        hits = [
            MemoryDbSearchHit(memory_id=memory.memory_id, score=0.0, memory=memory, source="graph")
            for memory in self.expanded
        ]
        return MemoryDbSearchResult(query=query.query, hits=hits, total=len(hits))


def _engine(reader: _Reader, embed) -> StructuredSearchEngine:
    return StructuredSearchEngine(
        db_reader=reader,
        db_writer=SimpleNamespace(),
        embed_client=embed,
        min_relevance_score=0.35,
        text_config=TextProcessingConfig(
            bm25_use_spacy_lemma=False,
            spacy_en_model="missing_en_model",
            spacy_zh_model="missing_zh_model",
            sparse_hash_dim=128,
        ),
    )


def test_structured_engine_constructor_does_not_capture_request_scoped_config(monkeypatch) -> None:
    def fail_if_resolved_during_singleton_construction():
        raise AssertionError("request-scoped config must be resolved inside search_candidates")

    monkeypatch.setattr(structured_engine_module, "get_config", fail_if_resolved_during_singleton_construction)

    StructuredSearchEngine(
        db_reader=SimpleNamespace(),
        db_writer=SimpleNamespace(),
        embed_client=_Embed(),
        text_preprocessor=SimpleNamespace(),
        sparse_encoder=SimpleNamespace(),
    )


@pytest.mark.asyncio
async def test_structured_search_returns_property_rows_without_episode_expansion() -> None:
    property_memory = _memory("memory-1", "Evaluate only the first 16 candidate neighbours.")
    episode_memory = _memory(
        "episode-memory",
        "Internal task background",
        entity_type="episodes",
        property_name="input_messages",
    )
    reader = _Reader(
        [
            MemoryDbSearchHit(memory_id="memory-1", score=0.9, memory=property_memory),
            MemoryDbSearchHit(memory_id="episode-memory", score=0.8, memory=episode_memory),
        ]
    )

    result = await _engine(reader, _Embed()).search_candidates(
        SearchPipelineInput(
            query="reduce local-search work",
            search_pipeline="structured",
            top_k=5,
            filters={"session_id": "task-1"},
        ),
        _context(),
    )

    assert [item.id for item in result] == ["memory-1"]
    assert result[0].memory == property_memory.content
    assert result[0].entity_id == property_memory.entity_id
    assert result[0].entity_type == "task_experience"
    assert result[0].property_name == "strategy"
    assert result[0].metadata == property_memory.metadata
    assert result[0].status == "active"
    assert len(reader.hybrid_calls) == 1
    query_filter = reader.hybrid_calls[0].query.filters
    assert any(item.field == "status" and item.value == "active" for item in query_filter.must)
    assert any(item.field == "entity_type" and item.values == ["episodes"] for item in query_filter.must_not)


@pytest.mark.asyncio
async def test_structured_search_filters_hybrid_results_by_dense_relevance_before_graph_expansion(
    monkeypatch,
) -> None:
    relevant = _memory("memory-relevant", "Shell sort uses diminishing increments.")
    unrelated = _memory("memory-unrelated", "The user prefers a window seat.")
    reader = _Reader(
        [
            MemoryDbSearchHit(memory_id=unrelated.memory_id, score=0.032, memory=unrelated),
            MemoryDbSearchHit(memory_id=relevant.memory_id, score=0.031, memory=relevant),
        ],
        dense_hits=[
            MemoryDbSearchHit(memory_id=relevant.memory_id, score=0.81, memory=relevant),
            MemoryDbSearchHit(memory_id=unrelated.memory_id, score=0.22, memory=unrelated),
        ],
        related=[
            {"seed_memory_id": relevant.memory_id, "memory_id": "memory-neighbor", "source": "shared_entity"},
        ],
        expanded=[_memory("memory-neighbor", "The final increment must be one.")],
    )
    monkeypatch.setattr(
        structured_engine_module,
        "get_config",
        lambda: SimpleNamespace(
            algo_config=SimpleNamespace(search=SimpleNamespace(structured=SimpleNamespace(min_relevance_score=0.35)))
        ),
    )

    result = await _engine(reader, _Embed()).search_candidates(
        SearchPipelineInput(query="Shell sort increments", search_pipeline="structured", top_k=5),
        _context(),
    )

    assert [item.id for item in result] == [relevant.memory_id, "memory-neighbor"]
    assert reader.dense_calls[0].score_threshold == 0.35
    assert reader.graph_calls[0].memory_ids == [relevant.memory_id]


@pytest.mark.asyncio
async def test_structured_search_skips_graph_expansion_when_no_dense_result_meets_threshold(
    monkeypatch,
) -> None:
    unrelated = _memory("memory-unrelated", "The user prefers a window seat.")
    reader = _Reader(
        [MemoryDbSearchHit(memory_id=unrelated.memory_id, score=0.032, memory=unrelated)],
        dense_hits=[MemoryDbSearchHit(memory_id=unrelated.memory_id, score=0.22, memory=unrelated)],
        related=[
            {"seed_memory_id": unrelated.memory_id, "memory_id": "memory-neighbor", "source": "shared_entity"},
        ],
        expanded=[_memory("memory-neighbor", "An unrelated graph neighbor.")],
    )
    monkeypatch.setattr(
        structured_engine_module,
        "get_config",
        lambda: SimpleNamespace(
            algo_config=SimpleNamespace(search=SimpleNamespace(structured=SimpleNamespace(min_relevance_score=0.35)))
        ),
    )

    result = await _engine(reader, _Embed()).search_candidates(
        SearchPipelineInput(query="Shell sort increments", search_pipeline="structured", top_k=5),
        _context(),
    )

    assert result == []
    assert reader.graph_calls == []


@pytest.mark.asyncio
async def test_structured_search_falls_back_to_sparse_without_schema_or_graph_work() -> None:
    reader = _Reader(
        [
            MemoryDbSearchHit(
                memory_id="memory-1",
                score=0.7,
                memory=_memory("memory-1", "Keep the bounded neighbourhood."),
            )
        ]
    )

    result = await _engine(reader, _FailingEmbed()).search_candidates(
        SearchPipelineInput(query="bounded neighbourhood", search_pipeline="structured", top_k=3),
        _context(),
    )

    assert [item.id for item in result] == ["memory-1"]
    assert reader.hybrid_calls == []
    assert len(reader.sparse_calls) == 1


@pytest.mark.asyncio
async def test_structured_search_interleaves_bounded_entity_and_episode_neighbors() -> None:
    direct_one = _memory(
        "memory-1",
        "Shell sort uses diminishing increments.",
        entity_id="shell-sort",
        episode_ids=["episode-shell"],
    )
    direct_two = _memory("memory-2", "Insertion sort shifts records.")
    same_entity = _memory(
        "memory-3",
        "The last Shell-sort increment must be one.",
        entity_id="shell-sort",
        episode_ids=["episode-shell"],
    )
    same_episode = _memory(
        "memory-4",
        "A nearly sorted sequence reduces insertion-sort movement.",
        entity_id="insertion-sort",
        episode_ids=["episode-shell"],
    )
    inactive = _memory("memory-5", "Archived detail.", status="archived")
    other_user = _memory("memory-6", "Another user's detail.", user_id="user-2")
    reader = _Reader(
        [
            MemoryDbSearchHit(memory_id="memory-1", score=0.9, memory=direct_one),
            MemoryDbSearchHit(memory_id="memory-2", score=0.8, memory=direct_two),
        ],
        related=[
            {"seed_memory_id": "memory-1", "memory_id": "memory-3", "source": "shared_entity"},
            {"seed_memory_id": "memory-1", "memory_id": "memory-4", "source": "shared_episode"},
            {"seed_memory_id": "memory-1", "memory_id": "memory-1", "source": "cycle"},
            {"seed_memory_id": "memory-2", "memory_id": "memory-4", "source": "shared_episode"},
            {"seed_memory_id": "memory-2", "memory_id": "memory-5", "source": "shared_episode"},
            {"seed_memory_id": "memory-2", "memory_id": "memory-6", "source": "shared_episode"},
        ],
        expanded=[same_episode, inactive, same_entity, other_user],
    )

    result = await _engine(reader, _Embed()).search_candidates(
        SearchPipelineInput(
            query="why is Shell sort faster",
            search_pipeline="structured",
            top_k=5,
            filters={"session_id": "task-1"},
        ),
        _context(),
    )

    assert [item.id for item in result] == ["memory-1", "memory-3", "memory-2", "memory-4"]
    assert len(reader.graph_calls) == 1
    assert reader.graph_calls[0].memory_ids == ["memory-1", "memory-2"]
    assert len(reader.filter_calls) == 1
    expansion_filter = reader.filter_calls[0].query.filters
    assert any(
        item.field == "memory_id" and set(item.values or []) == {"memory-3", "memory-4", "memory-5", "memory-6"}
        for item in expansion_filter.must
    )


@pytest.mark.asyncio
async def test_structured_search_returns_direct_results_when_graph_expansion_fails() -> None:
    class _GraphFailureReader(_Reader):
        async def list_structured_related_memory_ids(self, *args, **kwargs):
            raise RuntimeError("neo4j unavailable")

    reader = _GraphFailureReader(
        [
            MemoryDbSearchHit(
                memory_id="memory-1",
                score=0.9,
                memory=_memory("memory-1", "Keep the direct result."),
            )
        ]
    )

    result = await _engine(reader, _Embed()).search_candidates(
        SearchPipelineInput(query="direct result", search_pipeline="structured", top_k=5),
        _context(),
    )

    assert [item.id for item in result] == ["memory-1"]
