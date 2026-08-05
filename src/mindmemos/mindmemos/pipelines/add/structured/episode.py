"""Episode resolution primitives for lightweight structured ingestion."""

from __future__ import annotations

import asyncio
from dataclasses import dataclass, field
from uuid import UUID, uuid5

from ....components.extractor.structured import StructuredEpisodeCandidate
from ....llm import EmbedClient
from ....typing import FieldCondition, MemoryRequestContext, SearchFilter

_EPISODE_NAMESPACE = UUID("c8b604c3-ae12-4f99-a6f1-9f65c8ea0bc4")


@dataclass(slots=True)
class StructuredEpisodeDecision:
    """Validated Episode action emitted alongside structured memories."""

    action: str
    target_episode_id: str | None = None
    title: str = ""
    description: str = ""
    related_episode_ids: list[str] = field(default_factory=list)


async def recall_structured_episode_candidates(
    db_reader,
    embed_client: EmbedClient,
    context: MemoryRequestContext,
    content: str,
    *,
    top_k: int,
) -> list[StructuredEpisodeCandidate]:
    """Recall bounded Episode context without introducing caller parameters."""

    text = " ".join(str(content).split())
    if not text or top_k <= 0:
        return []
    response = await embed_client.embed(task="memory.add.structured_episode_recall", text=[text])
    if not response.embeddings:
        return []

    conditions = [
        FieldCondition(field="status", op="match", value="active"),
        FieldCondition(field="entity_type", op="match", value="episodes"),
    ]
    if context.user_id:
        conditions.append(FieldCondition(field="user_id", op="match", value=context.user_id))
    if context.app_id:
        conditions.append(FieldCondition(field="app_id", op="match", value=context.app_id))
    if context.agent_id:
        conditions.append(FieldCondition(field="agent_id", op="match", value=context.agent_id))

    filters = [SearchFilter(must=conditions)]
    if context.session_id:
        filters.append(
            SearchFilter(
                must=[
                    *conditions,
                    FieldCondition(field="session_id", op="match", value=context.session_id),
                ]
            )
        )
    results = await asyncio.gather(
        *(
            db_reader.search_entities_dense(
                context,
                query=text,
                query_vector=response.embeddings[0],
                filters=search_filter,
                limit=top_k,
            )
            for search_filter in filters
        )
    )
    candidates: list[StructuredEpisodeCandidate] = []
    seen: set[str] = set()
    for hit in (hit for result in results for hit in result.hits):
        entity = hit.entity
        if entity is None or entity.entity_type != "episodes" or entity.entity_id in seen:
            continue
        seen.add(entity.entity_id)
        candidates.append(
            StructuredEpisodeCandidate(
                episode_id=entity.entity_id,
                title=entity.entity_name,
                description=entity.description or "",
                score=hit.score,
                session_id=entity.session_id,
                same_session=bool(context.session_id and entity.session_id == context.session_id),
            )
        )
    candidates.sort(key=lambda item: (not item.same_session, -item.score, item.episode_id))
    return candidates[:top_k]


def default_structured_episode_decision(
    *,
    content: str,
    event_time: str,
    candidates: list[StructuredEpisodeCandidate],
) -> StructuredEpisodeDecision:
    """Return the conservative fallback used by injected/legacy extractors."""

    same_session = [candidate for candidate in candidates if candidate.same_session]
    if len(candidates) == 1 and len(same_session) == 1:
        return StructuredEpisodeDecision(
            action="reuse",
            target_episode_id=candidates[0].episode_id,
            title=candidates[0].title,
            description=candidates[0].description,
        )

    normalized = " ".join(str(content).split())
    title = normalized[:80] or f"Episode {event_time[:10]}"
    return StructuredEpisodeDecision(
        action="create",
        title=title,
        description=normalized or title,
    )


def structured_episode_decision_from_result(
    extracted: dict,
    *,
    content: str,
    event_time: str,
    candidates: list[StructuredEpisodeCandidate],
) -> StructuredEpisodeDecision:
    """Hydrate the validated extractor decision, with a safe legacy fallback."""

    raw = extracted.get("episode")
    if not isinstance(raw, dict):
        return default_structured_episode_decision(
            content=content,
            event_time=event_time,
            candidates=candidates,
        )
    return StructuredEpisodeDecision(
        action=str(raw.get("action") or "create"),
        target_episode_id=str(raw.get("target_episode_id") or "").strip() or None,
        title=str(raw.get("title") or "").strip(),
        description=str(raw.get("description") or "").strip(),
        related_episode_ids=[str(value) for value in raw.get("related_episode_ids", []) if str(value)],
    )


def structured_episode_id(context: MemoryRequestContext, *, content: str) -> str:
    """Converge concurrent first observations without using an LLM-generated title."""

    normalized = " ".join(str(content).split()).casefold()
    key = "\x1f".join(
        [
            context.project_id,
            context.user_id or "",
            context.app_id or "",
            context.agent_id or "",
            context.session_id or "",
            normalized,
        ]
    )
    return str(uuid5(_EPISODE_NAMESPACE, key))


__all__ = [
    "StructuredEpisodeCandidate",
    "StructuredEpisodeDecision",
    "default_structured_episode_decision",
    "recall_structured_episode_candidates",
    "structured_episode_decision_from_result",
    "structured_episode_id",
]
