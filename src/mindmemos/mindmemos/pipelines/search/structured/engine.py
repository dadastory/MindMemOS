"""Independent property retrieval with bounded Structured graph expansion."""

from __future__ import annotations

import asyncio
from typing import Any

from ....components.text import SparseVectorEncoder, TextPreprocessor, get_text_preprocessor
from ....config import TextProcessingConfig, get_config
from ....llm import EmbedClient, get_embed_client, require_model_endpoint
from ....logging import get_logger, traced
from ....mappers import parse_search_dsl
from ....typing import (
    FieldCondition,
    MemoryDbSearchHit,
    MemoryDbSearchQuery,
    MemoryRequestContext,
    MemorySearchItem,
    SearchFilter,
    SearchPipelineInput,
)
from ...base import MemoryDbPipelineMixin
from ...utils import format_datetime, format_memory_event_time, format_source_timestamp
from ..base import SearchEngineOptions

logger = get_logger(__name__)

_DEFAULT_RECALL_SIZE = 20
_MAX_RECALL_SIZE = 100
_PREFETCH_FACTOR = 3
_PREFETCH_MIN = 30
_PREFETCH_MAX = 300
_GRAPH_MAX_SEEDS = 10
_GRAPH_LIMIT_PER_SEED = 4
_GRAPH_MAX_CANDIDATES = 100
_CONFIG_UNSET = object()


class StructuredSearchEngine(MemoryDbPipelineMixin):
    """Retrieve active Structured properties and bounded graph neighbors."""

    name = "structured"

    def __init__(
        self,
        *,
        text_config: TextProcessingConfig | None = None,
        text_preprocessor: TextPreprocessor | None = None,
        sparse_encoder: SparseVectorEncoder | None = None,
        embed_client: EmbedClient | None = None,
        min_relevance_score: float | None | object = _CONFIG_UNSET,
        **kwargs: Any,
    ) -> None:
        super().__init__(**kwargs)
        # The pipeline owns this engine for the process lifetime, while provider
        # and algorithm configuration is request scoped. Keep only explicit test
        # overrides here and resolve production text configuration per search.
        self._explicit_text_config = text_config
        self._explicit_text_preprocessor = text_preprocessor
        self._explicit_sparse_encoder = sparse_encoder
        self._explicit_embed = embed_client
        self._explicit_min_relevance_score = min_relevance_score

    @traced("search.structured")
    async def search_candidates(
        self,
        inp: SearchPipelineInput,
        context: MemoryRequestContext,
        *,
        options: SearchEngineOptions | None = None,
    ) -> list[MemorySearchItem]:
        """Run bounded hybrid recall directly against stored memory properties."""

        algo_config = (
            get_config().algo_config
            if self._explicit_text_config is None or self._explicit_min_relevance_score is _CONFIG_UNSET
            else None
        )
        if self._explicit_text_config is None:
            assert algo_config is not None
            text_config = algo_config.text_processing
        else:
            text_config = self._explicit_text_config
        if self._explicit_min_relevance_score is _CONFIG_UNSET:
            assert algo_config is not None
            min_relevance_score = algo_config.search.structured.min_relevance_score
        else:
            min_relevance_score = self._explicit_min_relevance_score
        text_preprocessor = self._explicit_text_preprocessor or get_text_preprocessor(text_config)
        sparse_encoder = self._explicit_sparse_encoder or SparseVectorEncoder(text_config)
        preprocessed = text_preprocessor.preprocess_query(inp.query, include_entities=False)
        if not preprocessed.tokens:
            return []
        sparse = sparse_encoder.encode_query(preprocessed.tokens)
        dense = await self._encode_dense(inp.query)
        recall_size = _recall_size(inp, options)
        query = MemoryDbSearchQuery(
            query=inp.query,
            top_k=recall_size,
            filters=_request_filter(inp),
            mode="rrf" if dense is not None else "bm25",
            ranking="hybrid" if dense is not None else "score",
        )
        if dense is None:
            result = await self.db_reader.search_sparse(
                context,
                query,
                indices=list(sparse.indices),
                values=list(sparse.values),
            )
        else:
            prefetch = min(max(recall_size * _PREFETCH_FACTOR, _PREFETCH_MIN), _PREFETCH_MAX)
            hybrid_search = self.db_reader.search_hybrid(
                context,
                query,
                dense_vector=dense,
                sparse_vector=sparse,
                dense_limit=prefetch,
                sparse_limit=prefetch,
            )
            if min_relevance_score is None:
                result = await hybrid_search
                relevant_memory_ids: set[str] | None = None
            else:
                dense_query = query.model_copy(update={"top_k": prefetch, "mode": "dense", "ranking": "score"})
                result, dense_result = await asyncio.gather(
                    hybrid_search,
                    self.db_reader.search_dense(
                        context,
                        dense_query,
                        query_vector=dense,
                        score_threshold=min_relevance_score,
                    ),
                )
                relevant_memory_ids = {hit.memory_id for hit in dense_result.hits}
        direct = [
            _to_search_item(hit)
            for hit in result.hits
            if hit.memory is not None
            and hit.memory.status == "active"
            and hit.memory.entity_type != "episodes"
            and (dense is None or relevant_memory_ids is None or hit.memory_id in relevant_memory_ids)
        ]
        return await self._expand_graph(inp, context, direct, result.hits)

    async def _expand_graph(
        self,
        inp: SearchPipelineInput,
        context: MemoryRequestContext,
        direct: list[MemorySearchItem],
        direct_hits: list[MemoryDbSearchHit],
    ) -> list[MemorySearchItem]:
        """Add bounded graph neighbors while preserving direct-recall priority."""

        if not direct:
            return direct
        seed_ids = [item.id for item in direct[:_GRAPH_MAX_SEEDS]]
        max_candidates = min(
            _GRAPH_MAX_CANDIDATES,
            max(_DEFAULT_RECALL_SIZE, (inp.top_k or _DEFAULT_RECALL_SIZE) * _GRAPH_LIMIT_PER_SEED),
        )
        try:
            relations = await self.db_reader.list_structured_related_memory_ids(
                context,
                seed_ids,
                limit_per_memory=_GRAPH_LIMIT_PER_SEED,
                max_candidates=max_candidates,
            )
            candidate_ids = list(
                dict.fromkeys(
                    row["memory_id"]
                    for row in relations
                    if row.get("seed_memory_id") in seed_ids and row.get("memory_id") not in seed_ids
                )
            )
            if not candidate_ids:
                return direct
            base_filter = _request_filter(inp)
            expanded = await self.db_reader.search_by_filter(
                context,
                MemoryDbSearchQuery(
                    query=inp.query,
                    top_k=len(candidate_ids),
                    filters=SearchFilter(
                        must=[
                            *base_filter.must,
                            FieldCondition(field="memory_id", op="any", values=candidate_ids),
                        ],
                        should=base_filter.should,
                        must_not=base_filter.must_not,
                    ),
                    mode="graph",
                    ranking="none",
                ),
            )
        except Exception:
            logger.warning("structured_search_graph_expansion_failed", exc_info=True)
            return direct

        direct_by_id = {
            hit.memory_id: hit.memory for hit in direct_hits if hit.memory is not None and hit.memory_id in seed_ids
        }
        expanded_by_id = {
            hit.memory_id: hit
            for hit in expanded.hits
            if hit.memory is not None
            and hit.memory_id in candidate_ids
            and hit.memory.status == "active"
            and hit.memory.entity_type != "episodes"
        }
        neighbors_by_seed: dict[str, list[MemorySearchItem]] = {seed_id: [] for seed_id in seed_ids}
        accepted: set[str] = set(seed_ids)
        for relation in relations:
            seed_id = relation.get("seed_memory_id", "")
            memory_id = relation.get("memory_id", "")
            hit = expanded_by_id.get(memory_id)
            seed = direct_by_id.get(seed_id)
            if (
                seed is None
                or hit is None
                or hit.memory is None
                or memory_id in accepted
                or not _same_actor_scope(seed, hit.memory)
            ):
                continue
            accepted.add(memory_id)
            neighbors_by_seed[seed_id].append(_to_search_item(hit))

        # A graph neighbor must be able to enter a small top-k result even when
        # direct recall already returned its minimum batch of twenty rows.
        interleaved: list[MemorySearchItem] = []
        for item in direct:
            interleaved.append(item)
            neighbors = neighbors_by_seed.get(item.id)
            if neighbors:
                interleaved.append(neighbors.pop(0))
        for seed_id in seed_ids:
            interleaved.extend(neighbors_by_seed[seed_id])
        return interleaved

    async def _encode_dense(self, query: str) -> list[float] | None:
        try:
            if self._explicit_embed is None:
                require_model_endpoint("embedding")
            client = self._explicit_embed or get_embed_client()
            response = await client.embed(task="search.structured_query", text=query)
        except Exception:
            logger.warning("structured_search_dense_embed_failed", exc_info=True)
            return None
        return response.embeddings[0] if response.embeddings else None


def _recall_size(inp: SearchPipelineInput, options: SearchEngineOptions | None) -> int:
    requested = options.recall_top_k if options and options.recall_top_k is not None else inp.top_k
    return min(max(requested or _DEFAULT_RECALL_SIZE, _DEFAULT_RECALL_SIZE), _MAX_RECALL_SIZE)


def _request_filter(inp: SearchPipelineInput) -> SearchFilter:
    base = parse_search_dsl(inp.filters)
    return SearchFilter(
        must=[FieldCondition(field="status", op="match", value="active"), *base.must],
        should=base.should,
        must_not=[
            FieldCondition(field="entity_type", op="any", values=["episodes"]),
            *base.must_not,
        ],
    )


def _same_actor_scope(seed: Any, candidate: Any) -> bool:
    """Prevent graph edges from widening identity scope within one project."""

    if seed.project_id != candidate.project_id:
        return False
    return all(
        getattr(seed, field, None) == getattr(candidate, field, None)
        for field in ("account_id", "user_id", "app_id", "agent_id")
    )


def _to_search_item(hit: MemoryDbSearchHit) -> MemorySearchItem:
    memory = hit.memory
    assert memory is not None
    return MemorySearchItem(
        id=hit.memory_id,
        memory=memory.content,
        memory_type=memory.mem_type,
        last_update_at=format_datetime(memory.update_at or memory.created_at),
        event_time=format_memory_event_time(memory, fallback_to_source_timestamp=True),
        source_timestamp=format_source_timestamp(memory),
        metadata=dict(memory.metadata),
        status=str(memory.status),
        entity_id=memory.entity_id,
        entity_type=memory.entity_type,
        property_name=memory.property_name,
    )
