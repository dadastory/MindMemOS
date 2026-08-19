"""Low-latency fixed-Schema Add pipeline with bounded historical merging."""

from __future__ import annotations

import asyncio
import copy
import hashlib
import json
import time
import weakref
from collections import Counter
from collections.abc import Awaitable, Callable
from contextlib import AsyncExitStack
from datetime import UTC, datetime
from typing import Any, Literal

from ....components.extractor.schema._schema_utils import (
    edge_relationships,
    property_relationships,
    schema_memory_type,
)
from ....components.extractor.structured import StructuredExtractor
from ....components.kafka import memory_add_dispatch_key
from ....components.memory_modeling.schema import get_entity_manager
from ....components.text import SparseVectorEncoder, get_text_preprocessor
from ....config import StructuredAddConfig, get_config
from ....errors import ApiError, BadRequestError, ConfigNotInitializedError
from ....llm import get_embed_client, get_llm_client, require_model_endpoint
from ....logging import get_logger, traced
from ....structured_content import (
    STRUCTURED_CONTENT_METADATA_KEY,
    is_structured_content,
    normalize_structured_content,
    structured_content_from_metadata,
    structured_content_semantic_text,
    structured_content_text,
)
from ....typing import (
    AddPipelineAsyncResult,
    AddPipelineInput,
    AddPipelineSyncResult,
    AddStreamCancelled,
    EntityVectorWrite,
    EntityWrite,
    GraphNodeRef,
    GraphRelationship,
    MemoryAddEventItem,
    MemoryDbMemoryUpdateCommand,
    MemoryDbMutationPlan,
    MemoryDbWritePlan,
    MemoryRequestContext,
    MemoryView,
    MemoryWrite,
    VectorWrite,
)
from ...base import MemoryDbPipelineMixin
from ...memory_db import suppress_recording_errors
from ...registry import register
from ..base import AddPipeline
from .episode import (
    StructuredBatchBlock,
    StructuredBatchEpisodeAllocator,
    StructuredEpisodeCandidate,
    StructuredEpisodeDecision,
    recall_structured_episode_candidates,
    structured_batch_episode_id,
    structured_episode_decision_from_result,
    structured_episode_id,
)
from .identity import (
    canonical_content,
    new_structured_entity_id,
    new_structured_memory_id,
    structured_memory_fingerprint,
    structured_revision_memory_id,
)
from .planner import (
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

Consistency = Literal["fast", "strong"]
ProgressReporter = Callable[[str, str, int | None, dict[str, Any] | None], Awaitable[None]]
CancelCheck = Callable[[], Awaitable[bool]]
MEMORY_ADD_TOPIC = "memory.add"
STRUCTURED_VERSION = "structured_add_v1"
logger = get_logger(__name__)
_CLIENT_UNSET = object()
_LOOP_MODEL_LIMITS: weakref.WeakKeyDictionary[Any, dict[int, asyncio.Semaphore]] = weakref.WeakKeyDictionary()
_LOOP_COMMIT_LOCKS: weakref.WeakKeyDictionary[Any, dict[int, list[asyncio.Lock]]] = weakref.WeakKeyDictionary()


def _default_config() -> StructuredAddConfig:
    try:
        return get_config().algo_config.add.structured
    except ConfigNotInitializedError:
        return StructuredAddConfig()


def _default_consistency() -> Consistency:
    try:
        value = get_config().database.default_consistency
    except ConfigNotInitializedError:
        return "fast"
    return value if value in {"fast", "strong"} else "fast"


async def _report(progress: ProgressReporter | None, stage: str, message: str, percent: int) -> None:
    if progress is not None:
        await progress(stage, message, percent, None)


async def _check_cancel(cancel_check: CancelCheck | None, stage: str) -> None:
    if cancel_check is not None and await cancel_check():
        raise AddStreamCancelled(stage, "Add stream cancelled before persistence.")


def _structured_allowed_property_names(metadata: dict[str, Any]) -> set[str] | None:
    """Read an optional caller-owned fixed-Schema property constraint."""

    raw = metadata.get("structured_allowed_property_names")
    if raw is None:
        return None
    if not isinstance(raw, list | tuple | set):
        raise BadRequestError(
            "structured_allowed_property_names must be an array of Schema property names.",
            code="structured.property_constraint_invalid",
        )
    names = {str(value).strip() for value in raw if str(value).strip()}
    if not names or len(names) > 32:
        raise BadRequestError(
            "structured_allowed_property_names must contain between 1 and 32 property names.",
            code="structured.property_constraint_invalid",
        )
    return names


@register(type="add", name="structured_add")
class StructuredAddPipeline(MemoryDbPipelineMixin, AddPipeline):
    """Perform one strict extraction, bounded recall, and a short atomic commit."""

    def __init__(
        self,
        *,
        structured_add_config: StructuredAddConfig | None = None,
        extractor: Any | None = None,
        merge_decider: Any | None = None,
        episode_allocator: Any | None = None,
        entity_manager: Any | None = None,
        llm_client: Any = _CLIENT_UNSET,
        embed_client: Any = _CLIENT_UNSET,
        text_preprocessor: Any | None = None,
        sparse_encoder: Any | None = None,
        consistency: Consistency | None = None,
        **kwargs: Any,
    ) -> None:
        super().__init__(**kwargs)
        self._explicit_config = structured_add_config
        self._explicit_extractor = extractor
        self._explicit_merge_decider = merge_decider
        self._explicit_episode_allocator = episode_allocator
        self._explicit_entity_manager = entity_manager
        self._explicit_llm = llm_client
        self._explicit_embed = embed_client
        self._explicit_preprocessor = text_preprocessor
        self._explicit_sparse = sparse_encoder
        self._explicit_consistency = consistency
        config = structured_add_config or StructuredAddConfig()
        self._commit_locks = [asyncio.Lock() for _ in range(config.concurrency.lock_stripes)]

    def _config(self) -> StructuredAddConfig:
        return self._explicit_config or _default_config()

    def _consistency(self) -> Consistency:
        return self._explicit_consistency or _default_consistency()

    def _model_limit(self, config: StructuredAddConfig) -> asyncio.Semaphore:
        limit = config.concurrency.max_extract_concurrency
        by_limit = _LOOP_MODEL_LIMITS.setdefault(asyncio.get_running_loop(), {})
        return by_limit.setdefault(limit, asyncio.Semaphore(limit))

    def _locks(self, config: StructuredAddConfig) -> list[asyncio.Lock]:
        count = config.concurrency.lock_stripes
        by_count = _LOOP_COMMIT_LOCKS.setdefault(asyncio.get_running_loop(), {})
        locks = by_count.setdefault(count, [asyncio.Lock() for _ in range(count)])
        self._commit_locks = locks
        return locks

    def _clients(self) -> tuple[Any, Any]:
        if self._explicit_llm is _CLIENT_UNSET:
            require_model_endpoint("chat")
            llm = get_llm_client()
        else:
            llm = self._explicit_llm
        if self._explicit_embed is _CLIENT_UNSET:
            require_model_endpoint("embedding")
            embed = get_embed_client()
        else:
            embed = self._explicit_embed
        return llm, embed

    def _text_components(self) -> tuple[Any, Any]:
        preprocessor = self._explicit_preprocessor or get_text_preprocessor()
        sparse = self._explicit_sparse
        if sparse is None:
            sparse = SparseVectorEncoder(get_config().algo_config.text_processing)
        return preprocessor, sparse

    def _runtime(self, context: MemoryRequestContext) -> tuple[Any, Any, Any]:
        config = self._config()
        llm, embed = self._clients()
        if self._explicit_extractor is not None:
            extractor = self._explicit_extractor
        else:
            entity_manager = self._explicit_entity_manager or get_entity_manager(project_id=context.project_id)
            extractor = StructuredExtractor(
                llm_client=llm,
                entity_manager=entity_manager,
                config=config.extraction,
            )
        merge = self._explicit_merge_decider or StructuredMergeDecider(
            llm_client=llm,
            max_candidates=config.dedup.max_merge_candidates,
        )
        return extractor, merge, embed

    async def _prepare_extraction(
        self,
        extracted: dict[str, Any],
        context: MemoryRequestContext,
        embed: Any,
    ) -> tuple[list[StructuredProperty], dict[str, list[float]], dict[str, str]]:
        config = self._config()
        properties: list[StructuredProperty] = []
        entity_texts: dict[str, str] = {}
        entity_ids_by_name: dict[str, str] = {}
        seen: set[tuple[str, str]] = set()
        for entity in extracted.get("entities", []):
            entity_id = new_structured_entity_id()
            entity_key = entity["name"]
            display_name = str(entity.get("_display_name") or entity_key)
            entity_ids_by_name[entity_key] = entity_id
            prepared_properties: list[tuple[dict[str, Any], str]] = []
            for prop in entity.get("properties", []):
                fingerprint = structured_memory_fingerprint(
                    context,
                    entity_type=entity["entity_type"],
                    property_name=prop["property_name"],
                    content=prop["value"],
                )
                prepared_properties.append((prop, fingerprint))
            entity_texts.setdefault(
                entity_id,
                "\n".join(filter(None, [display_name, entity.get("description", ""), entity["entity_type"]])),
            )
            for prop, fingerprint in prepared_properties:
                entity_fingerprint = (entity_id, fingerprint)
                if entity_fingerprint in seen:
                    continue
                seen.add(entity_fingerprint)
                properties.append(
                    StructuredProperty(
                        entity_name=display_name,
                        entity_type=entity["entity_type"],
                        property_name=prop["property_name"],
                        content=prop["value"],
                        property_time=prop.get("time", entity.get("record_time", "")),
                        fingerprint=fingerprint,
                        memory_id=new_structured_memory_id(),
                        entity_id=entity_id,
                        entity_description=str(entity.get("description") or ""),
                        entity_keys=[entity_key],
                        source_block_ids=[str(entity.get("_source_block_id"))]
                        if entity.get("_source_block_id")
                        else [],
                        source_documents=[dict(entity.get("_source_document") or {})]
                        if entity.get("_source_document")
                        else [],
                    )
                )
        texts = [structured_content_semantic_text(item.content) for item in properties] + list(entity_texts.values())
        vectors = (
            await batch_embed(
                embed,
                texts,
                batch_size=config.embedding.batch_size,
                task="memory.add.structured_embed",
            )
            if texts
            else []
        )
        for item, vector in zip(properties, vectors[: len(properties)], strict=True):
            item.vector = vector
        entity_vectors = dict(zip(entity_texts, vectors[len(properties) :], strict=True))
        return properties, entity_vectors, entity_ids_by_name

    async def _prepare_properties(
        self, extracted: dict[str, Any], context: MemoryRequestContext
    ) -> list[StructuredProperty]:
        """Testing/extension helper returning the canonical prepared properties."""

        _, _, embed = self._runtime(context)
        properties, _, _ = await self._prepare_extraction(extracted, context, embed)
        return properties

    async def _decisions(
        self,
        context: MemoryRequestContext,
        properties: list[StructuredProperty],
        candidates: list[list[Any]],
        merge: Any,
        *,
        current_episode: dict[str, Any],
        episode_contexts: dict[str, dict[str, Any]],
        entity_schema: list[dict[str, Any]],
    ) -> list[StructuredDecision]:
        config = self._config().dedup
        decisions: list[StructuredDecision | None] = [None] * len(properties)
        ambiguous: list[tuple[int, StructuredMergeRequest]] = []
        requests = [
            StructuredMergeRequest(
                property=item,
                candidates=list(hits),
                current_episode=episode_contexts.get(item.episode_id or "", current_episode),
            )
            for item, hits in zip(properties, candidates, strict=True)
        ]
        await self._hydrate_candidate_entity_contexts(context, requests)
        for index, (item, hits, request) in enumerate(zip(properties, candidates, requests, strict=True)):
            current_episode_id = item.episode_id or str(current_episode.get("episode_id") or "")
            exact_matches = [
                hit.memory
                for hit in hits
                if hit.memory is not None
                and _is_exact_with_entity_context(
                    hit.memory,
                    item,
                    context,
                    current_episode_id=current_episode_id,
                    candidate_entity=request.candidate_entity_contexts.get(hit.memory_id),
                )
            ]
            if exact_matches:
                decisions[index] = StructuredDecision(
                    action="reinforce",
                    target_id=exact_matches[0].memory_id,
                    equivalent_ids=[memory.memory_id for memory in exact_matches[1:]],
                    reason="exact_content_resolved_entity",
                )
                continue
            best = hits[0] if hits else None
            classification = classify_similarity(best.score if best else None, config)
            if classification == "create" and len(item.batch_contents or [item.content]) == 1:
                decisions[index] = StructuredDecision(action="create", reason="below_similarity_threshold")
                continue
            if config.merge_mode != "llm_on_ambiguous":
                decisions[index] = StructuredDecision(action="create", reason="contextual_merge_disabled")
                continue
            ambiguous.append((index, request))

        if ambiguous:
            async with self._model_limit(self._config()):
                if hasattr(merge, "decide_batch"):
                    batch = await merge.decide_batch(
                        [request for _, request in ambiguous],
                        current_episode=current_episode,
                        episode_contexts=episode_contexts,
                        entity_schema=entity_schema,
                    )
                else:
                    batch = [
                        await merge.decide(request.property.content, request.candidates) for _, request in ambiguous
                    ]
            for (index, _), decision in zip(ambiguous, batch, strict=True):
                decisions[index] = decision
        return [decision or StructuredDecision(action="create", reason="unresolved_group") for decision in decisions]

    async def _hydrate_candidate_entity_contexts(
        self,
        context: MemoryRequestContext,
        requests: list[StructuredMergeRequest],
    ) -> None:
        """Attach bounded stored-entity context without turning names into recall filters."""

        get_entity = getattr(self.db_reader, "get_entity", None)
        if get_entity is None:
            return
        entity_ids = list(
            dict.fromkeys(
                memory.entity_id
                for request in requests
                for candidate in request.candidates
                for memory in [candidate.memory]
                if memory is not None and memory.entity_id
            )
        )
        loaded = await asyncio.gather(*(get_entity(context, entity_id) for entity_id in entity_ids))
        entities = {
            entity_id: entity for entity_id, entity in zip(entity_ids, loaded, strict=True) if entity is not None
        }
        for request in requests:
            for candidate in request.candidates:
                memory = candidate.memory
                if memory is None:
                    continue
                entity = entities.get(memory.entity_id or "")
                request.candidate_entity_contexts[candidate.memory_id] = {
                    "entity_id": memory.entity_id,
                    "name": entity.entity_name if entity is not None else "",
                    "description": entity.description or "" if entity is not None else "",
                    "entity_type": entity.entity_type if entity is not None else memory.entity_type,
                }

    @traced("add.structured_add.sync")
    async def add_sync(
        self,
        inp: AddPipelineInput,
        context: MemoryRequestContext,
        *,
        add_record_id: str | None = None,
    ) -> AddPipelineSyncResult:
        return await self._run(inp, context, add_record_id=add_record_id)

    async def add_sync_stream(
        self,
        inp: AddPipelineInput,
        context: MemoryRequestContext,
        *,
        add_record_id: str | None = None,
        progress: ProgressReporter | None = None,
        cancel_check: CancelCheck | None = None,
    ) -> AddPipelineSyncResult:
        return await self._run(
            inp,
            context,
            add_record_id=add_record_id,
            progress=progress,
            cancel_check=cancel_check,
        )

    async def _run(
        self,
        inp: AddPipelineInput,
        context: MemoryRequestContext,
        *,
        add_record_id: str | None,
        progress: ProgressReporter | None = None,
        cancel_check: CancelCheck | None = None,
    ) -> AddPipelineSyncResult:
        if inp.document_blocks:
            return await self._run_document_batch(
                inp,
                context,
                add_record_id=add_record_id,
                progress=progress,
                cancel_check=cancel_check,
            )
        started_at = time.perf_counter()
        extractor, merge, embed = self._runtime(context)
        content = _input_content(inp)
        config = self._config()
        episode_candidates: list[StructuredEpisodeCandidate] = []
        if hasattr(self.db_reader, "search_entities_dense"):
            try:
                episode_candidates = await recall_structured_episode_candidates(
                    self.db_reader,
                    embed,
                    context,
                    content,
                    top_k=config.episode.candidate_top_k,
                    reuse_at_or_above=config.episode.reuse_at_or_above,
                )
            except Exception:  # noqa: BLE001 - missing Episode history must not block lightweight Add
                logger.warning("structured_episode_recall_failed", request_id=context.request_id, exc_info=True)
        await _report(progress, "llm_extracting", "Extracting structured memory.", 25)
        async with self._model_limit(self._config()):
            extracted = await extractor.extract(
                content=content,
                event_time=inp.event_timestamp_utc.isoformat(),
                prompt_language=inp.prompt_language,
                episode_candidates=episode_candidates,
                allowed_property_names=_structured_allowed_property_names(inp.metadata),
            )
        await _check_cancel(cancel_check, "llm_extracting")
        episode_explicit = isinstance(extracted.get("episode"), dict)
        episode_decision = structured_episode_decision_from_result(
            extracted,
            content=content,
            event_time=inp.event_timestamp_utc.isoformat(),
            candidates=episode_candidates,
        )
        candidate_by_id = {candidate.episode_id: candidate for candidate in episode_candidates}
        selected_episode_id = (
            episode_decision.target_episode_id
            if episode_decision.action == "reuse" and episode_decision.target_episode_id in candidate_by_id
            else structured_episode_id(context, content=content)
        )
        selected_episode = {
            "episode_id": selected_episode_id,
            "title": episode_decision.title,
            "description": episode_decision.description,
            "action": episode_decision.action,
        }
        episode_contexts = {candidate.episode_id: candidate.prompt_value() for candidate in episode_candidates}
        episode_contexts[selected_episode_id] = dict(selected_episode)
        relevant_episode_ids = (
            {selected_episode_id, *episode_decision.related_episode_ids} if episode_explicit else set()
        )
        comparison_episode_ids = (
            {
                *relevant_episode_ids,
                *(candidate.episode_id for candidate in episode_candidates),
            }
            if episode_explicit
            else set()
        )
        if not extracted.get("entities") and not episode_explicit:
            result = AddPipelineSyncResult(status="ok")
            await self._complete_record(context, add_record_id, result)
            await _report(progress, "completed", "Structured memory extraction completed.", 100)
            return result

        await _report(progress, "embedding", "Embedding structured memories.", 50)
        properties, entity_vectors, entity_ids_by_name = await self._prepare_extraction(extracted, context, embed)
        extracted_property_count = len(properties)
        properties = consolidate_structured_properties(
            properties,
            similarity_at_or_above=config.dedup.same_batch_group_at_or_above,
        )
        episode_vector: list[float] | None = None
        if episode_explicit and episode_decision.action == "create":
            episode_vectors = await batch_embed(
                embed,
                ["\n".join(filter(None, [episode_decision.title, episode_decision.description, content]))],
                batch_size=1,
                task="memory.add.structured_episode_embed",
            )
            episode_vector = episode_vectors[0] if episode_vectors else None
        candidates = (
            await recall_structured_candidates(
                self.db_reader,
                context,
                properties,
                episode_ids=comparison_episode_ids or None,
                top_k=config.dedup.candidate_top_k,
            )
            if config.dedup.vector_enabled and properties
            else [[] for _ in properties]
        )
        decisions = await self._decisions(
            context,
            properties,
            candidates,
            merge,
            current_episode=selected_episode,
            episode_contexts=episode_contexts,
            entity_schema=list(getattr(extractor, "schema_context", []) or []),
        )
        revision_indexes = [
            index
            for index, (item, decision) in enumerate(zip(properties, decisions, strict=True))
            if decision.merged_content and decision.merged_content != item.content
        ]
        revision_vectors: dict[int, list[float]] = {}
        if revision_indexes:
            embedded_revisions = await batch_embed(
                embed,
                [
                    structured_content_semantic_text(decisions[index].merged_content or properties[index].content)
                    for index in revision_indexes
                ],
                batch_size=config.embedding.batch_size,
                task="memory.add.structured_embed_revision",
            )
            revision_vectors = dict(zip(revision_indexes, embedded_revisions, strict=True))
        await _check_cancel(cancel_check, "ready_to_persist")
        await _report(progress, "persisting", "Persisting structured memories.", 85)

        commit_locks = self._locks(config)
        scope_key = "\x1f".join(
            [
                context.project_id,
                context.user_id or "",
                context.app_id or "",
                context.session_id or "",
                context.agent_id or "",
            ]
        )
        lock_keys = [f"{scope_key}\x1f{item.entity_type}\x1f{item.property_name}" for item in properties]
        if episode_explicit:
            lock_keys.append(selected_episode_id)
        lock_indexes = sorted({hash(value) % len(commit_locks) for value in lock_keys})
        lock_wait_started_at = time.perf_counter()
        async with AsyncExitStack() as stack:
            for index in lock_indexes:
                await stack.enter_async_context(commit_locks[index])
            lock_acquired_at = time.perf_counter()
            commit_candidates = (
                await recall_structured_candidates(
                    self.db_reader,
                    context,
                    properties,
                    episode_ids=comparison_episode_ids or None,
                    top_k=config.dedup.candidate_top_k,
                )
                if config.dedup.vector_enabled and properties
                else [[] for _ in properties]
            )
            commit_decisions = await self._revalidate_decisions(
                context,
                properties,
                decisions,
                commit_candidates,
                current_episode_id=selected_episode_id if episode_explicit else "",
            )
            plan, events = await self._build_mutation_plan(
                inp,
                context,
                extracted,
                properties,
                entity_vectors,
                entity_ids_by_name,
                commit_decisions,
                revision_vectors,
                schema_version=str(getattr(extractor, "schema_version", "unknown")),
                add_record_id=add_record_id,
                episode_decision=episode_decision if episode_explicit else None,
                selected_episode_id=selected_episode_id if episode_explicit else None,
                episode_vector=episode_vector,
            )
            if plan.has_writes() or plan.has_updates_or_deletes():
                await self.db_writer.apply_mutation_plan(context, plan, consistency=self._consistency())
            lock_released_at = time.perf_counter()

        result = AddPipelineSyncResult(status="ok", memories=events)
        await self._complete_record(context, add_record_id, result)
        action_counts = Counter(
            str(command.memory.metadata.get("merge_action") or "create") for command in plan.memory_writes
        )
        action_counts["reinforce"] += sum(command.reason == "structured_reinforce" for command in plan.memory_updates)
        logger.info(
            "structured_add_completed",
            request_id=context.request_id,
            episode_action=episode_decision.action if episode_explicit else "legacy_unspecified",
            episode_candidate_count=len(episode_candidates),
            comparison_episode_count=len(comparison_episode_ids),
            extracted_property_count=extracted_property_count,
            consolidated_group_count=len(properties),
            property_count=len(properties),
            embedding_input_count=len(properties) + len(entity_vectors) + len(revision_vectors),
            candidate_count=sum(len(items) for items in candidates),
            commit_candidate_count=sum(len(items) for items in commit_candidates),
            actions=dict(action_counts),
            lock_wait_ms=round((lock_acquired_at - lock_wait_started_at) * 1000, 3),
            lock_hold_ms=round((lock_released_at - lock_acquired_at) * 1000, 3),
            elapsed_ms=round((time.perf_counter() - started_at) * 1000, 3),
        )
        await _report(progress, "completed", "Structured memory extraction completed.", 100)
        return result

    async def _run_document_batch(
        self,
        inp: AddPipelineInput,
        context: MemoryRequestContext,
        *,
        add_record_id: str | None,
        progress: ProgressReporter | None,
        cancel_check: CancelCheck | None,
    ) -> AddPipelineSyncResult:
        """Collect an unordered batch before Episode/history merge and one commit."""

        started_at = time.perf_counter()
        config = self._config()
        if len(inp.document_blocks) > config.batch.max_blocks:
            raise BadRequestError(
                f"document block count exceeds max_blocks={config.batch.max_blocks}",
                code="structured.batch_too_large",
            )
        block_contents = {block.block_id: _messages_content(block.messages) for block in inp.document_blocks}
        total_chars = sum(len(content) for content in block_contents.values())
        if total_chars > config.batch.max_total_chars:
            raise BadRequestError(
                f"document block content exceeds max_total_chars={config.batch.max_total_chars}",
                code="structured.batch_too_large",
            )

        extractor, merge, embed = self._runtime(context)
        await _report(progress, "llm_extracting", "Extracting all structured document blocks.", 20)

        async def extract_one(block):
            content = block_contents[block.block_id]
            event_timestamp = block.event_timestamp_ms or inp.event_timestamp
            event_time = datetime.fromtimestamp(event_timestamp / 1000, tz=UTC).isoformat()
            try:
                async with self._model_limit(config):
                    extracted = await extractor.extract(
                        content=content,
                        event_time=event_time,
                        prompt_language=inp.prompt_language,
                        episode_candidates=None,
                        allowed_property_names=_structured_allowed_property_names(
                            block.metadata if "structured_allowed_property_names" in block.metadata else inp.metadata
                        ),
                    )
            except ApiError as exc:
                exc.details = {**(exc.details or {}), "block_id": block.block_id}
                raise
            return block, extracted

        extracted_pairs = list(await asyncio.gather(*(extract_one(block) for block in inp.document_blocks)))
        await _check_cancel(cancel_check, "llm_extracting")

        collected_blocks: list[StructuredBatchBlock] = []
        combined: dict[str, list[dict[str, Any]]] = {"entities": [], "edges": []}
        for block, raw_extracted in extracted_pairs:
            extracted = copy.deepcopy(raw_extracted)
            collected_blocks.append(
                StructuredBatchBlock(
                    block_id=block.block_id,
                    content=block_contents[block.block_id],
                    extracted=extracted,
                    document_id=block.document_id,
                    locator=dict(block.locator),
                )
            )
            name_map: dict[str, str] = {}
            block_hash = hashlib.sha256(block_contents[block.block_id].encode("utf-8")).hexdigest()
            source_document = {
                "block_id": block.block_id,
                "document_id": block.document_id,
                "locator": dict(block.locator),
                "source_block_hash": block_hash,
                "metadata": _bounded_scalar_metadata(block.metadata),
            }
            for entity in extracted.get("entities", []):
                display_name = str(entity["name"])
                entity_key = f"{block.block_id}\x1f{display_name}"
                name_map[display_name] = entity_key
                entity["name"] = entity_key
                entity["_display_name"] = display_name
                entity["_source_block_id"] = block.block_id
                entity["_source_document"] = source_document
                for prop in entity.get("properties", []):
                    card = prop.get("value")
                    if not isinstance(card, dict) or not isinstance(card.get("artifacts"), list):
                        continue
                    card = copy.deepcopy(card)
                    for artifact in card["artifacts"]:
                        if not isinstance(artifact, dict):
                            continue
                        artifact_id = str(artifact.get("artifact_id") or "")
                        artifact["artifact_id"] = f"{block.block_id}:{artifact_id}"
                        artifact["source_block_id"] = block.block_id
                    prop["value"] = normalize_structured_content(card)
                combined["entities"].append(entity)
            for edge in extracted.get("edges", []):
                edge["link_entity1_name"] = name_map[edge["link_entity1_name"]]
                edge["link_entity2_name"] = name_map[edge["link_entity2_name"]]
                combined["edges"].append(edge)

        nonempty_blocks = [block for block in collected_blocks if block.extracted.get("entities")]
        if not nonempty_blocks:
            result = AddPipelineSyncResult(status="ok")
            await self._complete_record(context, add_record_id, result)
            await _report(progress, "completed", "Structured batch contained no durable facts.", 100)
            return result

        async def recall_episodes(block: StructuredBatchBlock):
            if not hasattr(self.db_reader, "search_entities_dense"):
                return []
            try:
                return await recall_structured_episode_candidates(
                    self.db_reader,
                    embed,
                    context,
                    block.content,
                    top_k=config.episode.candidate_top_k,
                    reuse_at_or_above=config.episode.reuse_at_or_above,
                )
            except Exception:  # noqa: BLE001 - absent Episode history is a valid first batch
                logger.warning(
                    "structured_batch_episode_recall_failed",
                    request_id=context.request_id,
                    block_id=block.block_id,
                    exc_info=True,
                )
                return []

        recalled_groups = await asyncio.gather(*(recall_episodes(block) for block in nonempty_blocks))
        candidate_by_id: dict[str, StructuredEpisodeCandidate] = {}
        for candidate in (candidate for group in recalled_groups for candidate in group):
            existing = candidate_by_id.get(candidate.episode_id)
            if existing is None or candidate.score > existing.score:
                candidate_by_id[candidate.episode_id] = candidate
        episode_candidates = sorted(
            candidate_by_id.values(),
            key=lambda item: (not item.same_session, -item.score, item.episode_id),
        )

        allocator = self._explicit_episode_allocator
        if allocator is None:
            llm_client = getattr(merge, "_llm", None)
            if llm_client is None:
                llm_client, _ = self._clients()
            allocator = StructuredBatchEpisodeAllocator(
                llm_client=llm_client,
                max_repair_attempts=config.batch.max_episode_repair_attempts,
            )
        await _report(progress, "episode_allocating", "Collecting blocks into Episode backgrounds.", 35)
        async with self._model_limit(config):
            groups = await allocator.allocate(nonempty_blocks, episode_candidates)

        batch_key = inp.idempotency_key or add_record_id or context.request_id
        block_episode_ids: dict[str, str] = {}
        episode_decisions: dict[str, StructuredEpisodeDecision] = {}
        episode_contexts = {candidate.episode_id: candidate.prompt_value() for candidate in episode_candidates}
        comparison_by_episode: dict[str, list[str]] = {}
        for group in groups:
            episode_id = (
                group.target_episode_id
                if group.action == "reuse"
                else structured_batch_episode_id(context, batch_key=batch_key, group_key=group.group_key)
            )
            if not episode_id:
                raise ValueError("validated Episode group did not resolve an Episode ID")
            decision = StructuredEpisodeDecision(
                action=group.action,
                target_episode_id=group.target_episode_id,
                title=group.title,
                description=group.description,
                related_episode_ids=group.related_episode_ids,
            )
            episode_decisions[episode_id] = decision
            episode_contexts[episode_id] = {
                "episode_id": episode_id,
                "action": group.action,
                "title": group.title,
                "description": group.description,
            }
            comparison_by_episode[episode_id] = list(dict.fromkeys([episode_id, *group.related_episode_ids]))
            for block_id in group.block_ids:
                block_episode_ids[block_id] = episode_id

        await _report(progress, "embedding", "Embedding and consolidating the collected batch.", 50)
        properties, entity_vectors, entity_ids_by_name = await self._prepare_extraction(combined, context, embed)
        for item in properties:
            block_id = item.source_block_ids[0]
            item.episode_id = block_episode_ids[block_id]
            item.comparison_episode_ids = comparison_by_episode[item.episode_id]
            item.batch_contents = [item.content]
            item.batch_entity_names = [item.entity_name]

        llm_client = getattr(merge, "_llm", None)
        if llm_client is not None:
            async with self._model_limit(config):
                properties = await consolidate_structured_batch(
                    llm_client,
                    properties,
                    entity_schema=list(getattr(extractor, "schema_context", []) or []),
                    episode_contexts=episode_contexts,
                    max_repair_attempts=config.batch.max_consolidation_repair_attempts,
                )
        if properties:
            final_vectors = await batch_embed(
                embed,
                [structured_content_semantic_text(item.content) for item in properties],
                batch_size=config.embedding.batch_size,
                task="memory.add.structured_batch_embed_final",
            )
            for item, vector in zip(properties, final_vectors, strict=True):
                item.vector = vector
                item.fingerprint = structured_memory_fingerprint(
                    context,
                    entity_type=item.entity_type,
                    property_name=item.property_name,
                    content=item.content,
                )
                item.memory_id = new_structured_memory_id()
                for entity_key in item.entity_keys:
                    entity_ids_by_name[entity_key] = item.entity_id

        candidates = (
            await recall_structured_candidates(
                self.db_reader,
                context,
                properties,
                episode_ids=None,
                top_k=config.dedup.candidate_top_k,
            )
            if config.dedup.vector_enabled and properties
            else [[] for _ in properties]
        )
        decisions = await self._decisions(
            context,
            properties,
            candidates,
            merge,
            current_episode={},
            episode_contexts=episode_contexts,
            entity_schema=list(getattr(extractor, "schema_context", []) or []),
        )
        revision_indexes = [
            index
            for index, (item, decision) in enumerate(zip(properties, decisions, strict=True))
            if decision.merged_content and decision.merged_content != item.content
        ]
        revision_vectors: dict[int, list[float]] = {}
        if revision_indexes:
            embedded_revisions = await batch_embed(
                embed,
                [
                    structured_content_semantic_text(decisions[index].merged_content or properties[index].content)
                    for index in revision_indexes
                ],
                batch_size=config.embedding.batch_size,
                task="memory.add.structured_embed_revision",
            )
            revision_vectors = dict(zip(revision_indexes, embedded_revisions, strict=True))

        episode_vectors: dict[str, list[float]] = {}
        create_episode_ids = [
            episode_id for episode_id, decision in episode_decisions.items() if decision.action == "create"
        ]
        if create_episode_ids:
            vectors = await batch_embed(
                embed,
                [
                    "\n".join(
                        filter(
                            None,
                            [episode_decisions[episode_id].title, episode_decisions[episode_id].description],
                        )
                    )
                    for episode_id in create_episode_ids
                ],
                batch_size=config.embedding.batch_size,
                task="memory.add.structured_episode_embed",
            )
            episode_vectors = dict(zip(create_episode_ids, vectors, strict=True))

        await _check_cancel(cancel_check, "ready_to_persist")
        await _report(progress, "persisting", "Persisting the structured batch.", 85)
        commit_locks = self._locks(config)
        scope_key = "\x1f".join(
            [
                context.project_id,
                context.user_id or "",
                context.app_id or "",
                context.session_id or "",
                context.agent_id or "",
            ]
        )
        lock_keys = [f"{scope_key}\x1f{item.entity_type}\x1f{item.property_name}" for item in properties]
        lock_keys.extend(episode_decisions)
        lock_indexes = sorted({hash(value) % len(commit_locks) for value in lock_keys})
        async with AsyncExitStack() as stack:
            for index in lock_indexes:
                await stack.enter_async_context(commit_locks[index])
            commit_candidates = (
                await recall_structured_candidates(
                    self.db_reader,
                    context,
                    properties,
                    episode_ids=None,
                    top_k=config.dedup.candidate_top_k,
                )
                if config.dedup.vector_enabled and properties
                else [[] for _ in properties]
            )
            commit_decisions = await self._revalidate_decisions(
                context,
                properties,
                decisions,
                commit_candidates,
                current_episode_id="",
            )
            plan, events = await self._build_mutation_plan(
                inp,
                context,
                combined,
                properties,
                entity_vectors,
                entity_ids_by_name,
                commit_decisions,
                revision_vectors,
                schema_version=str(getattr(extractor, "schema_version", "unknown")),
                add_record_id=add_record_id,
                episode_decision=None,
                selected_episode_id=None,
                episode_vector=None,
                batch_episodes=episode_decisions,
                batch_episode_vectors=episode_vectors,
                batch_block_episode_ids=block_episode_ids,
            )
            if plan.has_writes() or plan.has_updates_or_deletes():
                await self.db_writer.apply_mutation_plan(context, plan, consistency=self._consistency())

        result = AddPipelineSyncResult(status="ok", memories=events)
        await self._complete_record(context, add_record_id, result)
        logger.info(
            "structured_batch_add_completed",
            request_id=context.request_id,
            block_count=len(inp.document_blocks),
            episode_count=len(episode_decisions),
            property_count=len(properties),
            candidate_count=sum(len(items) for items in candidates),
            elapsed_ms=round((time.perf_counter() - started_at) * 1000, 3),
        )
        await _report(progress, "completed", "Structured document batch completed.", 100)
        return result

    async def _revalidate_decisions(
        self,
        context: MemoryRequestContext,
        properties: list[StructuredProperty],
        provisional: list[StructuredDecision],
        candidates: list[list[Any]],
        *,
        current_episode_id: str,
    ) -> list[StructuredDecision]:
        """Finalize decisions from storage state without waiting on a model."""

        resolved: list[StructuredDecision] = []
        requests = [
            StructuredMergeRequest(property=item, candidates=list(hits))
            for item, hits in zip(properties, candidates, strict=True)
        ]
        await self._hydrate_candidate_entity_contexts(context, requests)
        for item, decision, hits, request in zip(properties, provisional, candidates, requests, strict=True):
            resolved_episode_id = item.episode_id or current_episode_id
            exact_matches = [
                hit.memory
                for hit in hits
                if hit.memory is not None
                and _is_exact_with_entity_context(
                    hit.memory,
                    item,
                    context,
                    current_episode_id=resolved_episode_id,
                    candidate_entity=request.candidate_entity_contexts.get(hit.memory_id),
                )
            ]
            if exact_matches:
                resolved.append(
                    StructuredDecision(
                        action="reinforce",
                        target_id=exact_matches[0].memory_id,
                        equivalent_ids=[memory.memory_id for memory in exact_matches[1:]],
                        reason="exact_content_resolved_entity_commit_recheck",
                    )
                )
                continue
            if decision.action == "create":
                resolved.append(decision)
                continue
            target = await self.db_reader.get_memory(context, decision.target_id) if decision.target_id else None
            eligible_ids = {hit.memory_id for hit in hits if hit.memory is not None}
            # Preserve the provisional revision action for an inactive/missing
            # target so the mutation planner can first detect an idempotently
            # committed revision. It will safely fall back to create when no
            # such revision exists.
            if target is None or target.status != "active":
                resolved.append(decision)
                continue
            if target.memory_id not in eligible_ids:
                resolved.append(
                    StructuredDecision(
                        action="create",
                        reason="target_not_eligible_at_commit",
                    )
                )
                continue
            active_equivalent_ids = {
                hit.memory_id
                for hit in hits
                if hit.memory is not None and hit.memory.status == "active" and hit.memory_id != target.memory_id
            }
            resolved.append(
                StructuredDecision(
                    action=decision.action,
                    target_id=decision.target_id,
                    equivalent_ids=[
                        memory_id for memory_id in decision.equivalent_ids if memory_id in active_equivalent_ids
                    ],
                    merged_content=decision.merged_content,
                    reason=decision.reason,
                )
            )
        return resolved

    async def _build_mutation_plan(
        self,
        inp: AddPipelineInput,
        context: MemoryRequestContext,
        extracted: dict[str, Any],
        properties: list[StructuredProperty],
        entity_vectors: dict[str, list[float]],
        entity_ids_by_name: dict[str, str],
        decisions: list[StructuredDecision],
        revision_vectors: dict[int, list[float]],
        schema_version: str,
        *,
        add_record_id: str | None,
        episode_decision: StructuredEpisodeDecision | None,
        selected_episode_id: str | None,
        episode_vector: list[float] | None,
        batch_episodes: dict[str, StructuredEpisodeDecision] | None = None,
        batch_episode_vectors: dict[str, list[float]] | None = None,
        batch_block_episode_ids: dict[str, str] | None = None,
    ) -> tuple[MemoryDbMutationPlan, list[MemoryAddEventItem]]:
        now = datetime.now(UTC)
        preprocessor, sparse_encoder = self._text_components()
        flat = MemoryDbWritePlan()
        updates: list[MemoryDbMemoryUpdateCommand] = []
        events: list[MemoryAddEventItem] = []
        get_entity = getattr(self.db_reader, "get_entity", None)
        episode_specs = dict(batch_episodes or {})
        episode_vectors = dict(batch_episode_vectors or {})
        if episode_decision is not None and selected_episode_id is not None:
            episode_specs[selected_episode_id] = episode_decision
            if episode_vector is not None:
                episode_vectors[selected_episode_id] = episode_vector
        for resolved_episode_id, resolved_decision in episode_specs.items():
            existing_episode = await get_entity(context, resolved_episode_id) if get_entity is not None else None
            if resolved_decision.action != "create" or existing_episode is not None:
                continue
            episode_write = EntityWrite(
                entity_id=resolved_episode_id,
                account_id=context.account_id,
                project_id=context.project_id,
                api_key_uuid=context.api_key_uuid,
                user_id=context.user_id,
                app_id=context.app_id,
                session_id=context.session_id,
                agent_id=context.agent_id,
                request_id=context.request_id,
                entity_name=resolved_decision.title,
                entity_type="episodes",
                description=resolved_decision.description,
                created_at=now,
                schema_version=schema_version,
                metadata={
                    "add_algorithm": STRUCTURED_VERSION,
                    "source_add_record_ids": _bounded_append([], add_record_id, self._config().history.max_source_refs),
                    "source_block_hash": _source_block_hash(inp),
                },
            )
            flat.entities.append(episode_write)
            prepared_episode = preprocessor.preprocess_text(
                "\n".join(filter(None, [episode_write.entity_name, episode_write.description or ""])),
                segment_id=resolved_episode_id,
                include_entities=False,
            )
            episode_sparse = sparse_encoder.encode_document(prepared_episode.tokens)
            flat.entity_vectors.append(
                EntityVectorWrite(
                    entity_id=resolved_episode_id,
                    semantic_vector=episode_vectors.get(resolved_episode_id),
                    bm25_indices=list(episode_sparse.indices),
                    bm25_values=list(episode_sparse.values),
                )
            )
        for resolved_episode_id, resolved_decision in episode_specs.items():
            flat.relationships.extend(
                _related_episode_relationship(context.project_id, resolved_episode_id, related_episode_id)
                for related_episode_id in dict.fromkeys(resolved_decision.related_episode_ids)
                if related_episode_id != resolved_episode_id
            )

        await self._resolve_property_entity_ids(context, properties, decisions)
        entity_by_name = await self._prepare_entity_writes(
            context,
            extracted,
            {
                **entity_ids_by_name,
                **{
                    entity_name: item.entity_id
                    for item in properties
                    for entity_name in (item.batch_entity_names or [item.entity_name])
                },
            },
            entity_vectors,
            flat,
            preprocessor,
            sparse_encoder,
            now,
            schema_version,
        )
        if selected_episode_id:
            property_entity_ids = {item.entity_id for item in properties}
            unique_entities = {entity.entity_id: entity for entity in entity_by_name.values()}
            flat.relationships.extend(
                _episode_relationship(context.project_id, entity_id, selected_episode_id)
                for entity_id in unique_entities
                if entity_id not in property_entity_ids
            )
        elif batch_block_episode_ids:
            property_entity_ids = {item.entity_id for item in properties}
            connected: set[tuple[str, str]] = set()
            for entity_key, entity in entity_by_name.items():
                block_id = entity_key.split("\x1f", 1)[0]
                episode_id = batch_block_episode_ids.get(block_id)
                key = (entity.entity_id, episode_id or "")
                if not episode_id or entity.entity_id in property_entity_ids or key in connected:
                    continue
                connected.add(key)
                flat.relationships.append(_episode_relationship(context.project_id, entity.entity_id, episode_id))

        for item_index, (item, provisional) in enumerate(zip(properties, decisions, strict=True)):
            item_episode_id = item.episode_id or selected_episode_id
            revision_existing: MemoryView | None = None
            if provisional.action in {"update", "supersede"} and provisional.merged_content:
                revision_fingerprint = structured_memory_fingerprint(
                    context,
                    entity_type=item.entity_type,
                    property_name=item.property_name,
                    content=provisional.merged_content,
                )
                revision_event_key = inp.idempotency_key or add_record_id or context.request_id
                revision_existing = await self.db_reader.get_memory(
                    context,
                    structured_revision_memory_id(revision_fingerprint, revision_event_key),
                )
            exact = await self.db_reader.get_memory(context, item.memory_id)
            exact_same_episode = bool(
                exact is not None
                and (
                    item_episode_id is None
                    or item_episode_id in exact.episode_ids
                    or (not exact.episode_ids and exact.session_id == context.session_id)
                )
            )
            decision = provisional
            if revision_existing is not None and revision_existing.status == "active":
                decision = StructuredDecision(
                    action="reinforce",
                    target_id=revision_existing.memory_id,
                    reason="idempotent_revision_recheck",
                )
            elif revision_existing is not None:
                events.append(_idempotent_history_event(revision_existing))
                continue
            elif exact is not None and exact.status == "active" and exact_same_episode:
                decision = StructuredDecision(action="reinforce", target_id=exact.memory_id, reason="exact_recheck")
            target = await self.db_reader.get_memory(context, decision.target_id) if decision.target_id else None
            if decision.action != "create" and (target is None or target.status != "active"):
                decision = StructuredDecision(
                    action="create",
                    reason="target_changed_before_commit",
                )
                target = None
            equivalent_targets: list[MemoryView] = []
            if target is not None:
                for memory_id in decision.equivalent_ids:
                    equivalent = await self.db_reader.get_memory(context, memory_id)
                    if (
                        equivalent is not None
                        and equivalent.status == "active"
                        and equivalent.memory_id != target.memory_id
                        and equivalent.entity_type == item.entity_type
                        and equivalent.property_name == item.property_name
                    ):
                        equivalent_targets.append(equivalent)

            if decision.action == "reinforce" and target is not None:
                updates.append(
                    _reinforce_command(
                        target,
                        inp,
                        add_record_id,
                        context.request_id,
                        now,
                        self._config(),
                        episode_id=item_episode_id,
                        item=item,
                        equivalent_targets=equivalent_targets,
                    )
                )
                for equivalent in equivalent_targets:
                    updates.append(
                        _retire_equivalent_command(
                            equivalent,
                            canonical_memory_id=target.memory_id,
                            action="reinforce",
                            consistency=self._consistency(),
                        )
                    )
                    flat.relationships.append(
                        _memory_derivation_relationship(
                            context.project_id,
                            target.memory_id,
                            equivalent.memory_id,
                            action="reinforce",
                        )
                    )
                if item_episode_id and target.entity_id:
                    flat.relationships.append(
                        _episode_relationship(context.project_id, target.entity_id, item_episode_id)
                    )
                events.append(
                    MemoryAddEventItem(
                        operation="reinforcement",
                        content=target.content,
                        memory_id=target.memory_id,
                        mem_type=target.mem_type,
                        related_memory_ids=[memory.memory_id for memory in equivalent_targets],
                        graph_edge_count=len(equivalent_targets),
                        source_block_ids=item.source_block_ids,
                    )
                )
                continue

            content = decision.merged_content or item.content
            fingerprint = structured_memory_fingerprint(
                context,
                entity_type=item.entity_type,
                property_name=item.property_name,
                content=content,
            )
            if decision.action in {"update", "supersede"} and target is not None:
                event_key = inp.idempotency_key or add_record_id or context.request_id
                memory_id = structured_revision_memory_id(fingerprint, event_key)
            else:
                base_memory_id = item.memory_id
                existing_base = exact if exact_same_episode and base_memory_id == item.memory_id else None
                if exact is not None and exact.status == "active" and not exact_same_episode:
                    event_key = item_episode_id or inp.idempotency_key or add_record_id or context.request_id
                    base_memory_id = structured_revision_memory_id(fingerprint, event_key)
                    existing_base = await self.db_reader.get_memory(context, base_memory_id)
                elif existing_base is None:
                    existing_base = await self.db_reader.get_memory(context, base_memory_id)
                if existing_base is not None and existing_base.status == "active":
                    updates.append(
                        _reinforce_command(
                            existing_base,
                            inp,
                            add_record_id,
                            context.request_id,
                            now,
                            self._config(),
                            episode_id=item_episode_id,
                            item=item,
                        )
                    )
                    events.append(
                        MemoryAddEventItem(
                            operation="reinforcement",
                            content=existing_base.content,
                            memory_id=existing_base.memory_id,
                            mem_type=existing_base.mem_type,
                            source_block_ids=item.source_block_ids,
                        )
                    )
                    continue
                if existing_base is not None:
                    event_key = inp.idempotency_key or add_record_id or context.request_id
                    memory_id = structured_revision_memory_id(fingerprint, event_key)
                    existing_reintroduction = await self.db_reader.get_memory(context, memory_id)
                    if existing_reintroduction is not None and existing_reintroduction.status == "active":
                        updates.append(
                            _reinforce_command(
                                existing_reintroduction,
                                inp,
                                add_record_id,
                                context.request_id,
                                now,
                                self._config(),
                                episode_id=item_episode_id,
                                item=item,
                            )
                        )
                        events.append(
                            MemoryAddEventItem(
                                operation="reinforcement",
                                content=existing_reintroduction.content,
                                memory_id=existing_reintroduction.memory_id,
                                mem_type=existing_reintroduction.mem_type,
                                source_block_ids=item.source_block_ids,
                            )
                        )
                        continue
                    if existing_reintroduction is not None:
                        events.append(_idempotent_history_event(existing_reintroduction))
                        continue
                else:
                    memory_id = base_memory_id

            metadata = _new_metadata(
                inp,
                add_record_id,
                decision,
                target=target,
                equivalent_targets=equivalent_targets,
                max_source_refs=self._config().history.max_source_refs,
                item=item,
            )
            if is_structured_content(content):
                metadata[STRUCTURED_CONTENT_METADATA_KEY] = content
            parent_ids: list[str] = []
            root_ids: list[str] = []
            if target is not None and decision.action in {"update", "supersede"}:
                predecessors = [target, *equivalent_targets]
                parent_ids = list(dict.fromkeys(memory.memory_id for memory in predecessors))
                root_ids = list(
                    dict.fromkeys(
                        root_id for memory in predecessors for root_id in (memory.root_id or [memory.memory_id])
                    )
                )
                metadata["revision_action"] = decision.action
                metadata["previous_memory_id"] = target.memory_id
                if decision.action == "supersede":
                    metadata["supersedes"] = target.memory_id
                    metadata["superseded_memory_ids"] = parent_ids
                for predecessor in predecessors:
                    updates.append(
                        MemoryDbMemoryUpdateCommand(
                            memory_id=predecessor.memory_id,
                            status="archived" if decision.action == "update" else "superseded",
                            reason=f"structured_{decision.action}",
                            consistency=self._consistency(),
                            metadata_patch={
                                "replaced_by": memory_id,
                                "replacement_action": decision.action,
                            },
                        )
                    )

            mem_type = schema_memory_type(item.entity_type, item.property_name)
            episode_ids = _bounded_values(
                [
                    *(target.episode_ids if target is not None else []),
                    *(episode_id for memory in equivalent_targets for episode_id in memory.episode_ids),
                    item_episode_id,
                ],
                self._config().history.max_source_refs,
            )
            memory = MemoryWrite(
                memory_id=memory_id,
                account_id=context.account_id,
                project_id=context.project_id,
                api_key_uuid=context.api_key_uuid,
                user_id=context.user_id,
                app_id=context.app_id,
                session_id=context.session_id,
                agent_id=context.agent_id,
                request_id=context.request_id,
                content_fingerprint=fingerprint,
                idempotency_key=inp.idempotency_key,
                content=structured_content_text(content),
                mem_type=mem_type,
                mem_extract_type="structured",
                mem_extract_version=STRUCTURED_VERSION,
                metadata=metadata,
                validate_from=_parse_time(item.property_time),
                created_at=now,
                last_seen_at=now,
                parent_ids=parent_ids,
                root_id=root_ids,
                property_name=item.property_name,
                entity_id=item.entity_id,
                entity_type=item.entity_type,
                episode_ids=episode_ids,
            )
            prepared = preprocessor.preprocess_text(memory.content, segment_id=memory_id, include_entities=False)
            sparse = sparse_encoder.encode_document(prepared.tokens)
            flat.memories.append(memory)
            flat.vectors.append(
                VectorWrite(
                    memory_id=memory_id,
                    semantic_vector=revision_vectors.get(item_index, item.vector),
                    bm25_indices=list(sparse.indices),
                    bm25_values=list(sparse.values),
                )
            )
            flat.relationships.extend(property_relationships(context.project_id, item.entity_id, memory))
            if item_episode_id:
                flat.relationships.append(_episode_relationship(context.project_id, item.entity_id, item_episode_id))
            if target is not None and decision.action in {"update", "supersede"}:
                flat.relationships.extend(
                    _memory_derivation_relationship(
                        context.project_id,
                        memory_id,
                        predecessor_id,
                        action=decision.action,
                    )
                    for predecessor_id in parent_ids
                )
            events.append(
                MemoryAddEventItem(
                    operation="update" if decision.action in {"update", "supersede"} else "add",
                    content=memory.content,
                    memory_id=memory_id,
                    mem_type=mem_type,
                    related_memory_ids=parent_ids,
                    graph_edge_count=2 + len(parent_ids),
                    source_block_ids=item.source_block_ids,
                )
            )

        flat.relationships.extend(edge_relationships(extracted.get("edges", []), entity_by_name, context.project_id))
        mutation = MemoryDbMutationPlan.from_write_plan(flat)
        mutation.memory_updates.extend(updates)
        return mutation, events

    async def _resolve_property_entity_ids(
        self,
        context: MemoryRequestContext,
        properties: list[StructuredProperty],
        decisions: list[StructuredDecision],
    ) -> None:
        """Resolve one canonical stored entity for every extracted entity batch."""

        grouped: dict[str, list[int]] = {}
        for index, item in enumerate(properties):
            grouped.setdefault(item.entity_id, []).append(index)

        target_ids = list(dict.fromkeys(decision.target_id for decision in decisions if decision.target_id))
        targets = await asyncio.gather(*(self.db_reader.get_memory(context, memory_id) for memory_id in target_ids))
        target_by_id = dict(zip(target_ids, targets, strict=True))
        for indexes in grouped.values():
            canonical_entity_ids = {
                target.entity_id
                for index in indexes
                for target in [target_by_id.get(decisions[index].target_id)]
                if target is not None and target.status == "active" and target.entity_id
            }
            if len(canonical_entity_ids) != 1:
                continue
            canonical_entity_id = next(iter(canonical_entity_ids))
            for index in indexes:
                properties[index].entity_id = canonical_entity_id

    async def _prepare_entity_writes(
        self,
        context: MemoryRequestContext,
        extracted: dict[str, Any],
        entity_ids_by_name: dict[str, str],
        entity_vectors: dict[str, list[float]],
        flat: MemoryDbWritePlan,
        preprocessor: Any,
        sparse_encoder: Any,
        now: datetime,
        schema_version: str,
    ) -> dict[str, EntityWrite]:
        entity_by_name: dict[str, EntityWrite] = {}
        get_entity = getattr(self.db_reader, "get_entity", None)
        raw_by_name = {raw["name"]: raw for raw in extracted.get("entities", [])}
        aliases_by_id: dict[str, list[str]] = {}
        for entity_name, entity_id in entity_ids_by_name.items():
            if entity_name in raw_by_name:
                aliases_by_id.setdefault(entity_id, []).append(entity_name)
        for entity_id, aliases in aliases_by_id.items():
            raw = raw_by_name[aliases[0]]
            existing = await get_entity(context, entity_id) if get_entity is not None else None
            entity = EntityWrite(
                entity_id=entity_id,
                account_id=context.account_id,
                project_id=context.project_id,
                api_key_uuid=context.api_key_uuid,
                user_id=context.user_id,
                app_id=context.app_id,
                session_id=context.session_id,
                agent_id=context.agent_id,
                request_id=context.request_id,
                entity_name=existing.entity_name
                if existing is not None
                else str(raw.get("_display_name") or raw["name"]),
                entity_type=existing.entity_type if existing is not None else raw["entity_type"],
                description=(
                    _lossless_entity_description(
                        existing.description or "" if existing is not None else "",
                        str(raw.get("description") or ""),
                    )
                    if raw.get("_source_block_id")
                    else (existing.description or raw.get("description", ""))
                    if existing is not None
                    else raw.get("description", "")
                ),
                created_at=existing.created_at if existing and existing.created_at else now,
                update_at=now if existing else None,
                schema_version=schema_version,
                metadata={
                    **(dict(existing.metadata) if existing else {}),
                    "add_algorithm": STRUCTURED_VERSION,
                },
            )
            for alias in aliases:
                entity_by_name[alias] = entity
            unchanged = (
                existing is not None
                and existing.entity_name == entity.entity_name
                and existing.entity_type == entity.entity_type
                and (existing.description or "") == (entity.description or "")
            )
            if unchanged:
                continue
            flat.entities.append(entity)
            prepared = preprocessor.preprocess_text(
                entity.entity_name,
                segment_id=entity.entity_id,
                include_entities=False,
            )
            sparse = sparse_encoder.encode_document(prepared.tokens)
            flat.entity_vectors.append(
                EntityVectorWrite(
                    entity_id=entity.entity_id,
                    semantic_vector=entity_vectors.get(entity.entity_id),
                    bm25_indices=list(sparse.indices),
                    bm25_values=list(sparse.values),
                )
            )
        return entity_by_name

    async def _complete_record(
        self, context: MemoryRequestContext, add_record_id: str | None, result: AddPipelineSyncResult
    ) -> None:
        if add_record_id is not None:
            await suppress_recording_errors(
                self.recorder.mark_add_completed(context, add_record_id, result),
                operation="add.structured_add.sync",
            )

    async def add_async(
        self,
        inp: AddPipelineInput,
        context: MemoryRequestContext,
        *,
        add_record_id: str | None = None,
        record_metadata: dict[str, Any] | None = None,
    ) -> AddPipelineAsyncResult:
        from ....infra.kafka import get_producer

        if not get_config().kafka.enabled:
            raise RuntimeError(
                "add_async requires Kafka to be enabled (kafka.enabled=true). Use mode='sync' or enable Kafka."
            )
        message: dict[str, Any] = {"context": context.model_dump(), "input": inp.model_dump(by_alias=True)}
        if add_record_id is not None:
            message["add_record_id"] = add_record_id
        if record_metadata is not None:
            message["record_metadata"] = record_metadata
        await get_producer().send(
            MEMORY_ADD_TOPIC,
            value=message,
            dispatch_key=memory_add_dispatch_key(context),
        )
        return AddPipelineAsyncResult(status="queued")

    async def has_pending(self, context: MemoryRequestContext) -> bool:
        return False


def _input_content(inp: AddPipelineInput) -> str:
    if inp.document_blocks:
        return "\n\n".join(_messages_content(block.messages) for block in inp.document_blocks)
    return _messages_content(inp.messages)


def _messages_content(messages) -> str:
    parts: list[str] = []
    for message in messages:
        if hasattr(message, "content"):
            parts.append(f"{getattr(message, 'role', 'message')}: {message.content}")
        elif hasattr(message, "text"):
            parts.append(message.text)
        elif hasattr(message, "url"):
            parts.append(f"URL: {message.url}")
        elif hasattr(message, "file_path"):
            parts.append(f"File: {message.file_name} ({message.file_path})")
    return "\n".join(parts)


def _lossless_entity_description(existing: str, incoming: str) -> str:
    """Retain old and new context without allowing a fresh extraction to overwrite history."""

    old = str(existing or "").strip()
    new = str(incoming or "").strip()
    if not old:
        return new
    if not new or canonical_content(old) == canonical_content(new):
        return old
    if canonical_content(new) in canonical_content(old):
        return old
    if canonical_content(old) in canonical_content(new):
        return new
    return f"{old}\n{new}"


def _is_exact_with_entity_context(
    memory: MemoryView,
    item: StructuredProperty,
    context: MemoryRequestContext,
    *,
    current_episode_id: str,
    candidate_entity: dict[str, Any] | None,
) -> bool:
    """Use the exact fast path only when subject and contextual identity are resolved."""

    if memory.status != "active":
        return False
    if memory.entity_type != item.entity_type or memory.property_name != item.property_name:
        return False
    historical_content = structured_content_from_metadata(memory.metadata, memory.content)
    if canonical_content(historical_content) != canonical_content(item.content):
        return False
    if not candidate_entity or candidate_entity.get("entity_type") != item.entity_type:
        return False
    if canonical_content(candidate_entity.get("name") or "") != canonical_content(item.entity_name):
        return False
    historical_description = canonical_content(candidate_entity.get("description") or "")
    current_description = canonical_content(item.entity_description)
    if historical_description and current_description and historical_description != current_description:
        return False
    episode_ids = set(memory.episode_ids)
    if current_episode_id:
        return current_episode_id in episode_ids or (not episode_ids and memory.session_id == context.session_id)
    return not episode_ids and memory.session_id == context.session_id


def _new_metadata(
    inp: AddPipelineInput,
    add_record_id: str | None,
    decision: StructuredDecision,
    *,
    target: MemoryView | None = None,
    equivalent_targets: list[MemoryView] | None = None,
    max_source_refs: int = 100,
    item: StructuredProperty | None = None,
) -> dict[str, Any]:
    metadata = dict(inp.metadata)
    predecessors = [memory for memory in [target, *(equivalent_targets or [])] if memory is not None]
    source_add_record_ids = _metadata_list(predecessors, "source_add_record_ids", max_source_refs)
    idempotency_keys = _metadata_list(predecessors, "idempotency_keys", max_source_refs)
    evidence_history = _metadata_evidence(predecessors, max_source_refs)
    metadata.update(
        {
            "add_algorithm": STRUCTURED_VERSION,
            "merge_action": decision.action,
            "merge_reason": decision.reason,
            "source_block_hash": _source_block_hash(inp),
            "source_add_record_ids": _bounded_append(source_add_record_ids, add_record_id, max_source_refs),
            "idempotency_keys": _bounded_append(idempotency_keys, inp.idempotency_key, max_source_refs),
            "evidence_history": _evidence_history(
                evidence_history,
                inp,
                add_record_id,
                max_source_refs,
            ),
        }
    )
    if item is not None:
        metadata["entity_name"] = item.entity_name
        if item.entity_description:
            metadata["entity_description"] = item.entity_description
        metadata["source_block_ids"] = _bounded_values(
            [*_metadata_list(predecessors, "source_block_ids", max_source_refs), *item.source_block_ids],
            max_source_refs,
        )
        metadata["source_documents"] = _bounded_dict_values(
            [
                *(value for memory in predecessors for value in (memory.metadata or {}).get("source_documents", [])),
                *item.source_documents,
            ],
            max_source_refs,
        )
    if predecessors:
        metadata["merged_memory_ids"] = _bounded_values([memory.memory_id for memory in predecessors], max_source_refs)
    if inp.idempotency_key:
        metadata["last_idempotency_key"] = inp.idempotency_key
    return metadata


def _bounded_append(values: Any, value: str | None, limit: int) -> list[str]:
    result = [str(item) for item in values or [] if item]
    if value and value not in result:
        result.append(value)
    return result[-limit:]


def _bounded_values(values: Any, limit: int) -> list[str]:
    result = list(dict.fromkeys(str(item) for item in values or [] if item))
    return result[-limit:] if limit > 0 else result


def _bounded_dict_values(values: Any, limit: int) -> list[dict[str, Any]]:
    result: list[dict[str, Any]] = []
    for value in values or []:
        if isinstance(value, dict) and value not in result:
            result.append(dict(value))
    return result[-limit:] if limit > 0 else result


def _metadata_list(memories: list[MemoryView], key: str, limit: int) -> list[str]:
    return _bounded_values(
        [value for memory in memories for value in (memory.metadata or {}).get(key, [])],
        limit,
    )


def _metadata_evidence(memories: list[MemoryView], limit: int) -> list[dict[str, Any]]:
    result: list[dict[str, Any]] = []
    for memory in memories:
        for raw in (memory.metadata or {}).get("evidence_history", []):
            if not isinstance(raw, dict):
                continue
            item = dict(raw)
            if item not in result:
                result.append(item)
    return result[-limit:] if limit > 0 else result


def _evidence_history(
    values: Any,
    inp: AddPipelineInput,
    add_record_id: str | None,
    limit: int,
) -> list[dict[str, Any]]:
    history = [dict(item) for item in values or [] if isinstance(item, dict)]
    event = {
        "add_record_id": add_record_id,
        "idempotency_key": inp.idempotency_key,
        "source_block_hash": _source_block_hash(inp),
        "source_message_count": len(inp.messages),
        "source_block_ids": [block.block_id for block in inp.document_blocks],
        "source_block_count": len(inp.document_blocks),
        "metadata": _bounded_scalar_metadata(inp.metadata),
    }
    if not any(
        (add_record_id and item.get("add_record_id") == add_record_id)
        or (inp.idempotency_key and item.get("idempotency_key") == inp.idempotency_key)
        for item in history
    ):
        history.append(event)
    return history[-limit:]


def _source_block_hash(inp: AddPipelineInput) -> str:
    """Identify the immutable Add source block for provenance, not uniqueness."""

    if inp.document_blocks:
        payload = [block.model_dump(mode="json", by_alias=True) for block in inp.document_blocks]
        source = json.dumps(payload, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
    else:
        source = _input_content(inp)
    return hashlib.sha256(source.encode("utf-8")).hexdigest()


def _bounded_scalar_metadata(metadata: dict[str, Any]) -> dict[str, Any]:
    """Keep compact generic evidence fields; the Add Record retains the full payload."""

    result: dict[str, Any] = {}
    for raw_key, value in metadata.items():
        key = str(raw_key)[:128]
        if value is None or isinstance(value, bool | int | float):
            result[key] = value
        elif isinstance(value, str):
            result[key] = value[:1024]
        elif isinstance(value, list) and all(
            item is None or isinstance(item, bool | int | float | str) for item in value[:20]
        ):
            result[key] = [item[:256] if isinstance(item, str) else item for item in value[:20]]
    return result


def _reinforce_command(
    target: MemoryView,
    inp: AddPipelineInput,
    add_record_id: str | None,
    request_id: str,
    now: datetime,
    config: StructuredAddConfig,
    *,
    episode_id: str | None = None,
    equivalent_targets: list[MemoryView] | None = None,
    item: StructuredProperty | None = None,
) -> MemoryDbMemoryUpdateCommand:
    event_token = inp.idempotency_key or add_record_id or request_id
    metadata = dict(target.metadata or {})
    equivalents = equivalent_targets or []
    sources = _metadata_list([target, *equivalents], "source_add_record_ids", config.history.max_source_refs)
    idempotency_keys = _metadata_list([target, *equivalents], "idempotency_keys", config.history.max_source_refs)
    evidence = _metadata_evidence([target, *equivalents], config.history.max_source_refs)
    patch = {
        "source_add_record_ids": _bounded_append(sources, add_record_id, config.history.max_source_refs),
        "idempotency_keys": _bounded_append(idempotency_keys, inp.idempotency_key, config.history.max_source_refs),
        "evidence_history": _evidence_history(
            evidence,
            inp,
            add_record_id,
            config.history.max_source_refs,
        ),
        "merged_memory_ids": _bounded_values(
            [*metadata.get("merged_memory_ids", []), *(memory.memory_id for memory in equivalents)],
            config.history.max_source_refs,
        ),
        "last_evidence": _bounded_scalar_metadata(inp.metadata),
        "last_reinforcement_event_id": event_token,
    }
    if item is not None:
        patch["entity_name"] = item.entity_name
        if item.entity_description:
            patch["entity_description"] = item.entity_description
        patch["source_block_ids"] = _bounded_values(
            [
                *_metadata_list([target, *equivalents], "source_block_ids", config.history.max_source_refs),
                *item.source_block_ids,
            ],
            config.history.max_source_refs,
        )
        patch["source_documents"] = _bounded_dict_values(
            [
                *(
                    value
                    for memory in [target, *equivalents]
                    for value in (memory.metadata or {}).get("source_documents", [])
                ),
                *item.source_documents,
            ],
            config.history.max_source_refs,
        )
    if inp.idempotency_key:
        patch["last_idempotency_key"] = inp.idempotency_key
    return MemoryDbMemoryUpdateCommand(
        memory_id=target.memory_id,
        reinforcement_count_delta=1 + sum(1 + max(0, memory.reinforcement_count) for memory in equivalents),
        metadata_patch=patch,
        payload_patch={
            "last_seen_at": now,
            "idempotency_key": inp.idempotency_key,
            "episode_ids": _bounded_values(
                [
                    *target.episode_ids,
                    *(value for memory in equivalents for value in memory.episode_ids),
                    episode_id,
                ],
                config.history.max_source_refs,
            ),
        },
        reason="structured_reinforce",
        consistency="strong",
        dedup_metadata_key="last_reinforcement_event_id",
        optimistic_lock_token=event_token,
        optimistic_lock_retries=config.concurrency.max_write_conflict_retries,
        metadata_list_limit=config.history.max_source_refs,
    )


def _retire_equivalent_command(
    memory: MemoryView,
    *,
    canonical_memory_id: str,
    action: str,
    consistency: Consistency,
) -> MemoryDbMemoryUpdateCommand:
    return MemoryDbMemoryUpdateCommand(
        memory_id=memory.memory_id,
        status="archived",
        reason="structured_equivalent_merge",
        consistency=consistency,
        metadata_patch={
            "merged_into": canonical_memory_id,
            "replacement_action": action,
        },
    )


def _memory_derivation_relationship(
    project_id: str,
    canonical_memory_id: str,
    historical_memory_id: str,
    *,
    action: str,
) -> GraphRelationship:
    return GraphRelationship(
        source=GraphNodeRef(kind="Memory", project_id=project_id, node_id=canonical_memory_id),
        target=GraphNodeRef(kind="Memory", project_id=project_id, node_id=historical_memory_id),
        rel_type="DERIVED_FROM",
        project_id=project_id,
        metadata={"action": action},
    )


def _episode_relationship(project_id: str, entity_id: str, episode_id: str) -> GraphRelationship:
    return GraphRelationship(
        source=GraphNodeRef(kind="Entity", project_id=project_id, node_id=entity_id),
        target=GraphNodeRef(kind="Entity", project_id=project_id, node_id=episode_id),
        rel_type="OBSERVED_IN",
        project_id=project_id,
        metadata={"edge_source": "structured_episode"},
    )


def _related_episode_relationship(
    project_id: str,
    episode_id: str,
    related_episode_id: str,
) -> GraphRelationship:
    return GraphRelationship(
        source=GraphNodeRef(kind="Entity", project_id=project_id, node_id=episode_id),
        target=GraphNodeRef(kind="Entity", project_id=project_id, node_id=related_episode_id),
        rel_type="RELATES_TO",
        project_id=project_id,
        metadata={"edge_source": "structured_episode_context"},
    )


def _idempotent_history_event(memory: MemoryView) -> MemoryAddEventItem:
    """A replay of an already superseded event is acknowledged without resurrection."""

    return MemoryAddEventItem(
        operation="reinforcement",
        content=memory.content,
        memory_id=memory.memory_id,
        mem_type=memory.mem_type,
    )


def _parse_time(value: str) -> datetime | None:
    if not value:
        return None
    try:
        parsed = datetime.fromisoformat(value.replace("Z", "+00:00"))
    except ValueError:
        return None
    return parsed.replace(tzinfo=parsed.tzinfo or UTC).astimezone(UTC)
