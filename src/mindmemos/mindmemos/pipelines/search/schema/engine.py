"""Schema-aware single-pass search engine."""

from __future__ import annotations

import asyncio
import re
from typing import Any

from ....components.memory_modeling.schema import EntityManager, get_entity_manager
from ....components.searcher.schema import SchemaSearchExpander, SchemaSearchQueryBuilder
from ....components.searcher.scored_candidate import (
    GraphPathEvidence,
    RetrievalEvidence,
    ScoredSearchCandidate,
    normalize_candidate_scores,
)
from ....components.text import SparseVectorEncoder, TextPreprocessor, detect_prompt_language, get_text_preprocessor
from ....config import get_config
from ....config.algo.search import SearchConfig
from ....llm import (
    EmbedClient,
    LLMClient,
    RerankClient,
    get_embed_client,
    get_llm_client,
    require_model_endpoint,
)
from ....mappers import parse_schema_search_filters
from ....prompts import SearchPromptSet, get_search_prompts
from ....typing import (
    EntityView,
    FieldCondition,
    MemoryDbSearchHit,
    MemoryDbSearchQuery,
    MemoryRequestContext,
    MemorySearchItem,
    MemoryView,
    SearchFilter,
    SearchPipelineInput,
    combine_search_filters,
)
from ...base import MemoryDbPipelineMixin
from ...utils import format_datetime, format_memory_event_time, format_source_timestamp
from ..base import SearchEngineOptions


class SchemaSearchEngine(MemoryDbPipelineMixin):
    """Run one schema-aware entity/property retrieval pass."""

    name = "schema"

    def __init__(
        self,
        *,
        search_config: SearchConfig | None = None,
        expander: SchemaSearchExpander | None = None,
        query_builder: SchemaSearchQueryBuilder | None = None,
        llm_client: LLMClient | None = None,
        embed_client: EmbedClient | None = None,
        rerank_client: RerankClient | None = None,
        entity_manager: EntityManager | None = None,
        text_preprocessor: TextPreprocessor | None = None,
        sparse_encoder: SparseVectorEncoder | None = None,
        prompts: SearchPromptSet | None = None,
        **kwargs: Any,
    ) -> None:
        super().__init__(**kwargs)
        # This engine is held by a process-wide singleton (SearchPipelineImpl._engines),
        # so it MUST stay project-agnostic: all project-scoped deps (LLM/embed/rerank
        # clients, text preprocessor, sparse encoder, prompts, entity manager, schema
        # search config) are resolved per request from the request-scoped ContextVar
        # config (see get_config()). The explicit injections below are overrides for
        # tests only; production leaves them None.
        self._explicit_search_config = search_config
        self._expander = expander
        self._query_builder = query_builder
        self._explicit_llm = llm_client
        self._explicit_embed = embed_client
        self._explicit_rerank = rerank_client
        self._explicit_entity_manager = entity_manager
        self._explicit_text_preprocessor = text_preprocessor
        self._explicit_sparse_encoder = sparse_encoder
        self._explicit_prompts = prompts

    def _get_search_config(self) -> SearchConfig:
        if self._explicit_search_config is not None:
            return self._explicit_search_config
        return get_config().algo_config.search

    def _get_schema_search_config(self):
        return self._get_search_config().schema_search

    async def search_candidates(
        self,
        inp: SearchPipelineInput,
        context: MemoryRequestContext,
        *,
        options: SearchEngineOptions | None = None,
    ) -> list[ScoredSearchCandidate]:
        """Search schema entities and project them to public memory items."""

        schema_cfg = self._get_schema_search_config()

        # ``structured`` persists standard schema entities/memories, but its
        # low-latency search contract is memory-oriented: recall typed properties
        # directly and expand only through their Episode background.  Keep this
        # before schema prompt/query construction so it performs no Chat call.
        if context.memory_algorithm == "structured":
            if self._explicit_embed is None:
                require_model_endpoint("embedding")
            parsed_filters = parse_schema_search_filters(inp.filters, context)
            return await self._search_structured_candidates(
                inp,
                parsed_filters.context,
                parsed_filters.memory_filter,
                parsed_filters.entity_filter,
                options,
            )

        detected_lang = detect_prompt_language(
            inp.query,
            fallback=get_config().algo_config.common.prompt_language,
        )
        request_prompts = self._explicit_prompts or get_search_prompts(detected_lang)

        # Resolve project-scoped deps from the request-scoped config (ContextVar).
        if self._explicit_llm is None:
            require_model_endpoint("chat")
        if self._explicit_embed is None:
            require_model_endpoint("embedding")
        llm = self._explicit_llm or get_llm_client()
        embed = self._explicit_embed or get_embed_client()
        rerank = self._explicit_rerank if self._explicit_rerank is not None else _optional_rerank_client()
        text_preprocessor = self._explicit_text_preprocessor or get_text_preprocessor()
        sparse_encoder = self._explicit_sparse_encoder or SparseVectorEncoder(get_config().algo_config.text_processing)
        project_em = self._explicit_entity_manager or get_entity_manager(project_id=context.project_id)
        project_entity_schema = project_em.get_all_dicts() if project_em else []

        query_builder = self._query_builder or SchemaSearchQueryBuilder(
            llm=llm,
            prompts=request_prompts,
            entity_schema=project_entity_schema,
            current_time_mode=schema_cfg.current_time_mode,
            min_time_window_days=schema_cfg.min_time_window_days,
        )

        parsed_filters = parse_schema_search_filters(inp.filters, context)
        property_filter = query_builder.all_property_filter(entity_schema=project_entity_schema)
        initial_time_window = None
        if not parsed_filters.has_time_filter:
            initial_time_window = await query_builder.extract_time_from_query(inp.query, prompts=request_prompts)

        expander = self._expander or SchemaSearchExpander(
            db_reader=self.db_reader,
            embed_client=embed,
            rerank_client=rerank,
            text_preprocessor=text_preprocessor,
            sparse_encoder=sparse_encoder,
            config=schema_cfg,
        )
        entities = await expander.search(
            ctx=parsed_filters.context,
            query=inp.query,
            entity_types=list(property_filter.keys()) or None,
            property_filter=property_filter,
            time_window=initial_time_window,
            search_filter=parsed_filters.memory_filter,
            entity_search_filter=parsed_filters.entity_filter,
            num_hops=options.num_hops if options and options.num_hops is not None else schema_cfg.multi_hop,
            use_reranker=options.use_reranker if options else None,
            top_k=options.recall_top_k if options else None,
            top_n=options.result_top_n if options and options.result_top_n is not None else inp.top_k,
        )
        if not entities:
            return await self._search_memory_fallback(
                inp, parsed_filters.context, parsed_filters.memory_filter, options
            )

        return normalize_candidate_scores(
            [
                ScoredSearchCandidate(
                    item=MemorySearchItem(
                        id=entity.entity_id,
                        memory=entity.format_entity_prompt(
                            ignore_edge_num=schema_cfg.output_max_edge_num,
                            include_description=False,
                            include_edges=schema_cfg.include_edges,
                        ),
                        memory_type="fact",
                        last_update_at="",
                    ),
                    original_rank=index,
                    rank=index,
                    evidence=[RetrievalEvidence(source="schema", rank=index)],
                )
                for index, entity in enumerate(entities)
            ]
        )

    async def _search_structured_candidates(
        self,
        inp: SearchPipelineInput,
        context: MemoryRequestContext,
        memory_filter: SearchFilter | None,
        entity_filter: SearchFilter | None,
        options: SearchEngineOptions | None,
    ) -> list[ScoredSearchCandidate]:
        """Recall structured memories directly and through relevant Episodes."""

        embed = self._explicit_embed or get_embed_client()
        text_preprocessor = self._explicit_text_preprocessor or get_text_preprocessor()
        sparse_encoder = self._explicit_sparse_encoder or SparseVectorEncoder(get_config().algo_config.text_processing)
        preprocessed = text_preprocessor.preprocess_query(inp.query, include_entities=False)
        if not preprocessed.tokens:
            return []

        embedding = await embed.embed(task="memory.search.structured", text=inp.query)
        if not embedding.embeddings:
            return []
        dense_vector = embedding.embeddings[0]
        sparse_vector = sparse_encoder.encode_query(preprocessed.tokens)
        requested_top_k = _memory_fallback_top_k(inp, options) or self._get_search_config().default.top_k
        recall_top_k = max(requested_top_k, (inp.top_k or requested_top_k) * 3, 15)
        scoped_memory_filter = combine_search_filters(
            _structured_scope_filter(context),
            SearchFilter(
                must=[
                    FieldCondition(field="status", op="match", value="active"),
                    FieldCondition(
                        field="mem_extract_type",
                        op="any",
                        values=["schema", "structured"],
                    ),
                ]
            ),
            memory_filter,
        )
        scoped_episode_filter = combine_search_filters(
            _structured_scope_filter(context),
            SearchFilter(must=[FieldCondition(field="entity_type", op="match", value="episodes")]),
            entity_filter,
        )

        direct_result, episode_result = await asyncio.gather(
            self.db_reader.search_hybrid(
                context,
                MemoryDbSearchQuery(
                    query=inp.query,
                    top_k=recall_top_k,
                    filters=scoped_memory_filter,
                    mode="hybrid",
                    ranking="hybrid",
                ),
                dense_vector=dense_vector,
                sparse_vector=sparse_vector,
            ),
            self.db_reader.search_entities_hybrid(
                context,
                dense_vector=dense_vector,
                sparse_vector=sparse_vector,
                filters=scoped_episode_filter,
                limit=min(recall_top_k, 8),
            ),
        )

        episodes = {hit.entity_id: hit.entity for hit in episode_result.hits if hit.entity is not None}
        missing_episode_ids = list(
            dict.fromkeys(
                episode_id
                for hit in direct_result.hits
                if hit.memory is not None
                for episode_id in hit.memory.episode_ids
                if episode_id not in episodes
            )
        )[:8]
        get_entity = getattr(self.db_reader, "get_entity", None)
        if get_entity is not None and missing_episode_ids:
            loaded = await asyncio.gather(*(get_entity(context, episode_id) for episode_id in missing_episode_ids))
            for episode_id, entity in zip(missing_episode_ids, loaded, strict=True):
                if entity is not None and entity.entity_type == "episodes":
                    episodes[episode_id] = entity
        direct_candidates = [
            _structured_direct_candidate(hit, index=index, episodes=episodes)
            for index, hit in enumerate(direct_result.hits)
            if hit.memory is not None
        ]
        graph_candidates = await self._structured_episode_graph_candidates(
            context,
            episode_result.hits,
            memory_filter=scoped_memory_filter,
            limit=recall_top_k,
        )
        return normalize_candidate_scores(
            _dedupe_structured_candidates([*direct_candidates, *graph_candidates])[:recall_top_k]
        )

    async def _structured_episode_graph_candidates(
        self,
        context: MemoryRequestContext,
        episode_hits: list[Any],
        *,
        memory_filter: SearchFilter | None,
        limit: int,
    ) -> list[ScoredSearchCandidate]:
        async def expand_episode(hit: Any) -> list[ScoredSearchCandidate]:
            episode = hit.entity
            if episode is None:
                return []
            neighbors = await self.db_reader.get_entity_neighbors(
                context,
                episode.entity_id,
                direction="in",
                rel_type="OBSERVED_IN",
                limit=min(limit, 20),
            )
            if not neighbors:
                return []
            by_entity = {neighbor.entity_id: neighbor for neighbor in neighbors}
            graph_filter = combine_search_filters(
                memory_filter,
                SearchFilter(
                    must=[
                        FieldCondition(field="entity_id", op="any", values=list(by_entity)),
                        FieldCondition(field="episode_ids", op="any", values=[episode.entity_id]),
                    ]
                ),
            )
            memories, _ = await self.db_reader.list_memories(
                context,
                filters=graph_filter,
                limit=limit,
            )
            return [
                _structured_graph_candidate(
                    memory,
                    index=index,
                    episode=episode,
                    episode_score=hit.score,
                )
                for index, memory in enumerate(memories)
            ]

        groups = await asyncio.gather(*(expand_episode(hit) for hit in episode_hits))
        return [candidate for group in groups for candidate in group]

    async def _search_memory_fallback(
        self,
        inp: SearchPipelineInput,
        context: MemoryRequestContext,
        filters: SearchFilter | None,
        options: SearchEngineOptions | None,
    ) -> list[ScoredSearchCandidate]:
        """Fallback to direct memory recall when schema entity recall is empty."""

        text_preprocessor = self._explicit_text_preprocessor or get_text_preprocessor()
        sparse_encoder = self._explicit_sparse_encoder or SparseVectorEncoder(get_config().algo_config.text_processing)
        preprocessed = text_preprocessor.preprocess_query(inp.query, include_entities=False)
        if not preprocessed.tokens:
            return []

        sparse = sparse_encoder.encode_query(preprocessed.tokens)
        top_k = _memory_fallback_top_k(inp, options)
        query = MemoryDbSearchQuery(
            query=inp.query,
            top_k=top_k or self._get_search_config().default.top_k,
            filters=filters,
            mode="bm25",
            ranking="score",
        )
        result = await self.db_reader.search_sparse(
            context,
            query,
            indices=list(sparse.indices),
            values=list(sparse.values),
        )
        return normalize_candidate_scores(
            [
                ScoredSearchCandidate(
                    item=_to_memory_search_item(hit),
                    original_rank=hit.rank if hit.rank is not None else index,
                    rank=index,
                    retrieval_score=hit.score,
                    retrieval_score_type="bm25",
                    evidence=[
                        RetrievalEvidence(
                            source="direct",
                            score=hit.score,
                            score_type="bm25",
                            rank=hit.rank if hit.rank is not None else index,
                        )
                    ],
                )
                for index, hit in enumerate(result.hits)
            ]
        )


def _optional_rerank_client() -> RerankClient | None:
    try:
        from ....llm import get_rerank_client

        return get_rerank_client()
    except Exception:
        return None


def _to_memory_search_item(hit: MemoryDbSearchHit) -> MemorySearchItem:
    memory = hit.memory
    return MemorySearchItem(
        id=hit.memory_id,
        memory=memory.content if memory else "",
        memory_type=memory.mem_type if memory else "fact",
        last_update_at=format_datetime((memory.update_at or memory.created_at) if memory else None),
        event_time=format_memory_event_time(memory, fallback_to_source_timestamp=True) if memory else None,
        source_timestamp=format_source_timestamp(memory) if memory else None,
    )


def _memory_fallback_top_k(inp: SearchPipelineInput, options: SearchEngineOptions | None) -> int | None:
    if options and options.recall_top_k is not None:
        return options.recall_top_k
    if options and options.result_top_n is not None:
        return options.result_top_n
    return inp.top_k


def _structured_scope_filter(context: MemoryRequestContext) -> SearchFilter:
    """Build the actor scope used by structured memory and Episode recall."""

    conditions = [FieldCondition(field="account_id", op="match", value=context.account_id)]
    for field in ("user_id", "app_id", "session_id", "agent_id"):
        value = getattr(context, field)
        if value:
            conditions.append(FieldCondition(field=field, op="match", value=value))
    return SearchFilter(must=conditions)


def _structured_direct_candidate(
    hit: MemoryDbSearchHit,
    *,
    index: int,
    episodes: dict[str, EntityView],
) -> ScoredSearchCandidate:
    memory = hit.memory
    assert memory is not None
    return ScoredSearchCandidate(
        item=_structured_memory_search_item(memory, episodes=episodes),
        original_rank=hit.rank if hit.rank is not None else index,
        rank=index,
        retrieval_score=hit.score,
        retrieval_score_type="rrf",
        evidence=[
            RetrievalEvidence(
                source="direct",
                score=hit.score,
                score_type="rrf",
                rank=hit.rank if hit.rank is not None else index,
            )
        ],
    )


def _structured_graph_candidate(
    memory: MemoryView,
    *,
    index: int,
    episode: EntityView,
    episode_score: float,
) -> ScoredSearchCandidate:
    score = max(0.0, min(1.0, float(episode_score))) * 0.85
    return ScoredSearchCandidate(
        item=_structured_memory_search_item(memory, episodes={episode.entity_id: episode}),
        original_rank=index,
        rank=index,
        retrieval_score=score,
        retrieval_score_type="graph_propagation",
        evidence=[
            RetrievalEvidence(
                source="graph",
                score=score,
                score_type="graph_propagation",
                rank=index,
                graph=GraphPathEvidence(
                    seed_memory_id=episode.entity_id,
                    relation="OBSERVED_IN",
                    hops=1,
                    path_score=score,
                    entity_id=memory.entity_id,
                    entity_type=memory.entity_type,
                ),
            )
        ],
    )


def _structured_memory_search_item(
    memory: MemoryView,
    *,
    episodes: dict[str, EntityView],
) -> MemorySearchItem:
    metadata = dict(memory.metadata)
    if memory.root_id:
        metadata["root_memory_ids"] = list(memory.root_id)
    metadata["episode_contexts"] = [
        _episode_context(episode_id, episodes.get(episode_id)) for episode_id in memory.episode_ids
    ]
    return MemorySearchItem(
        id=memory.memory_id,
        memory=memory.content,
        memory_type=memory.property_name or memory.mem_type,
        last_update_at=format_datetime(memory.update_at or memory.created_at),
        event_time=format_memory_event_time(memory, fallback_to_source_timestamp=True),
        source_timestamp=format_source_timestamp(memory),
        metadata=metadata,
        status=memory.status,
        entity_id=memory.entity_id,
        entity_type=memory.entity_type,
        property_name=memory.property_name,
    )


def _episode_context(episode_id: str, episode: EntityView | None) -> dict[str, Any]:
    return {
        "episode_id": episode_id,
        "title": episode.entity_name if episode else "",
        "description": episode.description or "" if episode else "",
    }


def _dedupe_structured_candidates(
    candidates: list[ScoredSearchCandidate],
) -> list[ScoredSearchCandidate]:
    """Collapse duplicate routes/lineage while preserving distinct entity facts."""

    by_identity: dict[tuple[str, ...], ScoredSearchCandidate] = {}
    memory_keys: dict[str, tuple[str, ...]] = {}
    for candidate in candidates:
        item = candidate.item
        canonical = re.sub(r"\s+", " ", item.memory).strip().casefold()
        root_ids = tuple(str(value) for value in item.metadata.get("root_memory_ids", []) if value)
        identity = (
            ("lineage", str(item.entity_type or ""), str(item.property_name or item.memory_type), *root_ids)
            if root_ids
            else (
                "content",
                str(item.entity_id or ""),
                str(item.entity_type or ""),
                str(item.property_name or item.memory_type),
                canonical,
            )
        )
        existing_key = memory_keys.get(item.id)
        key = existing_key or identity
        existing = by_identity.get(key)
        if existing is None:
            by_identity[key] = candidate
            memory_keys[item.id] = key
            continue
        existing.evidence.extend(evidence for evidence in candidate.evidence if evidence not in existing.evidence)
        existing_contexts = list(existing.item.metadata.get("episode_contexts", []))
        seen_episode_ids = {str(value.get("episode_id")) for value in existing_contexts if isinstance(value, dict)}
        for episode_context in candidate.item.metadata.get("episode_contexts", []):
            if not isinstance(episode_context, dict):
                continue
            episode_id = str(episode_context.get("episode_id") or "")
            if episode_id and episode_id not in seen_episode_ids:
                existing_contexts.append(episode_context)
                seen_episode_ids.add(episode_id)
        existing.item.metadata["episode_contexts"] = existing_contexts[:8]
        if (candidate.retrieval_score or 0.0) > (existing.retrieval_score or 0.0):
            existing.retrieval_score = candidate.retrieval_score
            existing.retrieval_score_type = candidate.retrieval_score_type
        memory_keys[item.id] = key
    return sorted(
        by_identity.values(),
        key=lambda item: (-(item.retrieval_score or 0.0), item.original_rank, item.item.id),
    )
