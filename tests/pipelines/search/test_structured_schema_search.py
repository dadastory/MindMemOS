from __future__ import annotations

from datetime import UTC, datetime
from types import SimpleNamespace

import pytest
from mindmemos.config.algo.search import SearchConfig
from mindmemos.pipelines.search.schema.engine import SchemaSearchEngine
from mindmemos.typing import (
    EntitySearchHit,
    EntitySearchResult,
    EntityView,
    MemoryDbSearchHit,
    MemoryDbSearchResult,
    MemoryRequestContext,
    MemoryView,
    SearchPipelineInput,
)


def _context(algorithm="structured"):
    return MemoryRequestContext(
        request_id="req-1",
        account_id="account-1",
        project_id="project-1",
        api_key_uuid="key-1",
        user_id="user-1",
        session_id="task-1",
        memory_algorithm=algorithm,
    )


def _memory(memory_id, content, entity_id, episode_id):
    return MemoryView(
        memory_id=memory_id,
        project_id="project-1",
        content=content,
        mem_type="experience",
        mem_extract_type="structured",
        status="active",
        user_id="user-1",
        session_id="task-1",
        property_name="good_algorithm",
        entity_id=entity_id,
        entity_type="llm4ad_memory_card",
        episode_ids=[episode_id],
        created_at=datetime(2026, 8, 1, tzinfo=UTC),
    )


class _Reader:
    def __init__(self):
        self.direct_request = None
        self.return_episode_hit = True
        self.episode = EntityView(
            entity_id="episode-1",
            project_id="project-1",
            entity_name="Optimization run",
            entity_type="episodes",
            description="Population stagnated before mutation was increased.",
            user_id="user-1",
            session_id="task-1",
        )
        self.direct = _memory("memory-direct", "Increase mutation after stagnation.", "entity-direct", "episode-1")
        self.graph = _memory("memory-graph", "Preserve elites while increasing mutation.", "entity-graph", "episode-1")

    async def search_hybrid(self, context, req, **_kwargs):
        self.direct_request = req
        return MemoryDbSearchResult(
            query=req.query,
            hits=[MemoryDbSearchHit(memory_id=self.direct.memory_id, score=0.9, memory=self.direct, source="rrf")],
        )

    async def search_entities_hybrid(self, context, **_kwargs):
        return EntitySearchResult(
            query="query",
            hits=(
                [EntitySearchHit(entity_id=self.episode.entity_id, score=0.85, entity=self.episode)]
                if self.return_episode_hit
                else []
            ),
        )

    async def get_entity_neighbors(self, context, entity_id, **_kwargs):
        assert entity_id == self.episode.entity_id
        return [
            EntityView(
                entity_id="entity-graph",
                project_id="project-1",
                entity_name="Graph card",
                entity_type="llm4ad_memory_card",
                user_id="user-1",
                session_id="task-1",
            )
        ]

    async def list_memories(self, context, *, filters, limit):
        return [self.graph], None

    async def get_entity(self, context, entity_id):
        return self.episode if entity_id == self.episode.entity_id else None


class _Embed:
    async def embed(self, *, task, text, **_kwargs):
        values = text if isinstance(text, list) else [text]
        return SimpleNamespace(embeddings=[[1.0, 0.0] for _ in values])


class _Preprocessor:
    def preprocess_query(self, text, **_kwargs):
        return SimpleNamespace(tokens=text.split())


class _Sparse:
    def encode_query(self, _tokens):
        return SimpleNamespace(indices=[1], values=[1.0])


class _EntityManager:
    def get_all_dicts(self):
        return []


@pytest.mark.asyncio
async def test_structured_schema_search_fuses_direct_cards_and_episode_graph_cards():
    reader = _Reader()
    engine = SchemaSearchEngine(
        search_config=SearchConfig(),
        llm_client=SimpleNamespace(),
        embed_client=_Embed(),
        entity_manager=_EntityManager(),
        text_preprocessor=_Preprocessor(),
        sparse_encoder=_Sparse(),
        db_reader=reader,
        db_writer=SimpleNamespace(),
    )

    result = await engine.search_candidates(
        SearchPipelineInput(query="stagnation mutation", search_pipeline="schema", top_k=5),
        _context(),
    )

    assert {candidate.item.id for candidate in result} == {"memory-direct", "memory-graph"}
    assert all(candidate.item.memory_type == "good_algorithm" for candidate in result)
    assert all(candidate.item.metadata["episode_contexts"][0]["episode_id"] == "episode-1" for candidate in result)
    assert any(any(evidence.source == "graph" for evidence in candidate.evidence) for candidate in result)
    extract_filter = next(
        condition
        for group in reader.direct_request.filters.must
        for condition in (group.must if hasattr(group, "must") else [group])
        if condition.field == "mem_extract_type"
    )
    assert extract_filter.op == "any"
    assert extract_filter.values == ["schema", "structured"]


@pytest.mark.asyncio
async def test_structured_direct_result_hydrates_episode_context_not_recalled_as_query_seed():
    reader = _Reader()
    reader.return_episode_hit = False
    engine = SchemaSearchEngine(
        search_config=SearchConfig(),
        llm_client=SimpleNamespace(),
        embed_client=_Embed(),
        entity_manager=_EntityManager(),
        text_preprocessor=_Preprocessor(),
        sparse_encoder=_Sparse(),
        db_reader=reader,
        db_writer=SimpleNamespace(),
    )

    result = await engine.search_candidates(
        SearchPipelineInput(query="mutation", search_pipeline="schema", top_k=5),
        _context(),
    )

    assert result[0].item.metadata["episode_contexts"] == [
        {
            "episode_id": "episode-1",
            "title": "Optimization run",
            "description": "Population stagnated before mutation was increased.",
        }
    ]


@pytest.mark.asyncio
async def test_structured_search_keeps_equal_content_from_different_entities():
    reader = _Reader()
    duplicate = _memory(
        "memory-duplicate",
        "  Increase   mutation after stagnation. ",
        "entity-duplicate",
        "episode-1",
    )
    reader.graph = duplicate
    engine = SchemaSearchEngine(
        search_config=SearchConfig(),
        llm_client=SimpleNamespace(),
        embed_client=_Embed(),
        entity_manager=_EntityManager(),
        text_preprocessor=_Preprocessor(),
        sparse_encoder=_Sparse(),
        db_reader=reader,
        db_writer=SimpleNamespace(),
    )

    result = await engine.search_candidates(
        SearchPipelineInput(query="stagnation mutation", search_pipeline="schema", top_k=5),
        _context(),
    )

    assert {candidate.item.id for candidate in result} == {"memory-direct", "memory-duplicate"}


@pytest.mark.asyncio
async def test_structured_search_collapses_same_canonical_content_within_one_entity():
    reader = _Reader()
    duplicate = _memory(
        "memory-duplicate",
        "  Increase   mutation after stagnation. ",
        reader.direct.entity_id,
        "episode-1",
    )
    reader.graph = duplicate
    engine = SchemaSearchEngine(
        search_config=SearchConfig(),
        llm_client=SimpleNamespace(),
        embed_client=_Embed(),
        entity_manager=_EntityManager(),
        text_preprocessor=_Preprocessor(),
        sparse_encoder=_Sparse(),
        db_reader=reader,
        db_writer=SimpleNamespace(),
    )

    result = await engine.search_candidates(
        SearchPipelineInput(query="stagnation mutation", search_pipeline="schema", top_k=5),
        _context(),
    )

    assert len(result) == 1


@pytest.mark.asyncio
async def test_structured_search_collapses_multiple_active_views_of_one_root_lineage():
    reader = _Reader()
    reader.direct.root_id = ["root-memory"]
    reader.graph.root_id = ["root-memory"]
    reader.graph.content = "A revised rendering of the same rooted card."
    engine = SchemaSearchEngine(
        search_config=SearchConfig(),
        llm_client=SimpleNamespace(),
        embed_client=_Embed(),
        entity_manager=_EntityManager(),
        text_preprocessor=_Preprocessor(),
        sparse_encoder=_Sparse(),
        db_reader=reader,
        db_writer=SimpleNamespace(),
    )

    result = await engine.search_candidates(
        SearchPipelineInput(query="stagnation mutation", search_pipeline="schema", top_k=5),
        _context(),
    )

    assert len(result) == 1
    assert result[0].item.metadata["root_memory_ids"] == ["root-memory"]


@pytest.mark.asyncio
async def test_structured_search_supports_unbounded_final_top_k_with_bounded_recall():
    reader = _Reader()
    engine = SchemaSearchEngine(
        search_config=SearchConfig(),
        llm_client=SimpleNamespace(),
        embed_client=_Embed(),
        entity_manager=_EntityManager(),
        text_preprocessor=_Preprocessor(),
        sparse_encoder=_Sparse(),
        db_reader=reader,
        db_writer=SimpleNamespace(),
    )

    result = await engine.search_candidates(
        SearchPipelineInput(query="mutation", search_pipeline="schema", top_k=None),
        _context(),
    )

    assert result
    assert reader.direct_request.top_k >= 15
