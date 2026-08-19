"""Episode resolution primitives for lightweight structured ingestion."""

from __future__ import annotations

import asyncio
import json
from dataclasses import dataclass, field
from uuid import UUID, uuid5

from ....components.extractor.schema import parse_json_object
from ....components.extractor.structured import StructuredEpisodeCandidate
from ....errors import ApiError
from ....llm import EmbedClient, LLMClient
from ....typing import FieldCondition, MemoryRequestContext, SearchFilter

_EPISODE_NAMESPACE = UUID("c8b604c3-ae12-4f99-a6f1-9f65c8ea0bc4")
_OMIT_CONTEXT_VALUE = object()


@dataclass(slots=True)
class StructuredEpisodeDecision:
    """Validated Episode action emitted alongside structured memories."""

    action: str
    target_episode_id: str | None = None
    title: str = ""
    description: str = ""
    related_episode_ids: list[str] = field(default_factory=list)


@dataclass(slots=True, frozen=True)
class StructuredBatchBlock:
    """One independently extracted source block awaiting Episode allocation."""

    block_id: str
    content: str
    extracted: dict
    document_id: str | None = None
    locator: dict = field(default_factory=dict)


@dataclass(slots=True)
class StructuredBatchEpisodeGroup:
    """Validated reuse/create group that covers one or more source blocks."""

    group_key: str
    action: str
    block_ids: list[str]
    target_episode_id: str | None = None
    title: str = ""
    description: str = ""
    related_episode_ids: list[str] = field(default_factory=list)


class StructuredEpisodeAllocationError(ApiError):
    """Raised when a batch Episode decision cannot be validated."""

    status_code = 422
    code = "structured.episode_allocation_invalid"


class StructuredBatchEpisodeAllocator:
    """Allocate all non-empty extracted blocks with one bounded model decision."""

    def __init__(self, *, llm_client: LLMClient, max_repair_attempts: int = 1) -> None:
        self._llm = llm_client
        self._max_repair_attempts = max_repair_attempts

    async def allocate(
        self,
        blocks: list[StructuredBatchBlock],
        candidates: list[StructuredEpisodeCandidate],
    ) -> list[StructuredBatchEpisodeGroup]:
        active_blocks = [block for block in blocks if block.extracted.get("entities")]
        if not active_blocks:
            return []
        alias_to_block_id = {f"b{index}": block.block_id for index, block in enumerate(active_blocks, start=1)}
        block_id_to_alias = {block_id: alias for alias, block_id in alias_to_block_id.items()}
        expected_aliases = list(alias_to_block_id)
        prompt = _batch_episode_prompt(active_blocks, candidates, block_id_to_alias=block_id_to_alias)
        last_content = ""
        last_error = "unknown validation error"
        for attempt in range(1 + self._max_repair_attempts):
            current_prompt = prompt
            if attempt:
                current_prompt += (
                    "\nPrevious answer:\n"
                    + last_content
                    + "\nValidation error: "
                    + last_error
                    + "\nCorrection rules: assignments must be one closed JSON object whose keys are exactly the allowed "
                    "block aliases and whose values are group_key values declared in groups. Every allowed alias must be "
                    f"present exactly once, so assignments must contain exactly {len(expected_aliases)} keys. Groups must "
                    "not contain block_ids. When one block relates to more than one historical Episode, choose its single "
                    "primary group and put secondary historical Episodes in related_episode_ids. Merge competing create "
                    "groups when they describe the same background. Return the complete corrected JSON object only."
                )
            response = await self._llm.chat(
                task="memory.add.structured_episode_allocate",
                messages=[{"role": "user", "content": current_prompt}],
            )
            last_content = response.content
            try:
                groups = _validate_batch_episode_groups(
                    parse_json_object(response.content),
                    expected_aliases=expected_aliases,
                    candidates=candidates,
                )
                for group in groups:
                    group.block_ids = [alias_to_block_id[alias] for alias in group.block_ids]
                return groups
            except (TypeError, ValueError, json.JSONDecodeError) as exc:
                last_error = str(exc)
        raise StructuredEpisodeAllocationError(
            f"Structured Episode allocation failed validation: {last_error}",
            details={"block_ids": sorted(alias_to_block_id.values()), "attempts": 1 + self._max_repair_attempts},
        )


async def recall_structured_episode_candidates(
    db_reader,
    embed_client: EmbedClient,
    context: MemoryRequestContext,
    content: str,
    *,
    top_k: int,
    reuse_at_or_above: float = 0.82,
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
        if (
            entity is None
            or entity.entity_type != "episodes"
            or entity.entity_id in seen
            or hit.score < reuse_at_or_above
        ):
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


def structured_batch_episode_id(
    context: MemoryRequestContext,
    *,
    batch_key: str,
    group_key: str,
) -> str:
    """Create a replay-stable Episode ID without title/content identity."""

    key = "\x1f".join(
        [
            context.project_id,
            context.user_id or "",
            context.app_id or "",
            context.agent_id or "",
            context.session_id or "",
            batch_key,
            group_key,
        ]
    )
    return str(uuid5(_EPISODE_NAMESPACE, key))


def _batch_episode_prompt(
    blocks: list[StructuredBatchBlock],
    candidates: list[StructuredEpisodeCandidate],
    *,
    block_id_to_alias: dict[str, str],
) -> str:
    values = []
    for block in blocks:
        if not block.extracted.get("entities"):
            continue
        values.append(
            {
                "block_alias": block_id_to_alias[block.block_id],
                "source_context": _episode_source_context(block),
                "source_content": block.content,
                "extracted": _episode_prompt_extracted(block.extracted),
            }
        )
    return (
        "Allocate every supplied unordered source block exactly once to its semantic Episode background. "
        "Return groups as Episode definitions and assignments as one closed alias-to-group JSON object. Use only the "
        "exact block_alias values supplied below as assignments keys; never invent, rewrite, expand, or derive an alias. "
        "Every alias must be one assignments key and its value must be one declared group_key. Groups must not contain "
        "block_ids. Blocks are independent and array order, adjacency, source context, and generated entity titles do not "
        "prove identity. A group may reuse one supplied historical Episode or create one new Episode. Create as many "
        "groups as necessary; do not force unrelated blocks together. Group non-adjacent blocks only when their complete "
        "facts describe the same coherent background. Preserve all contextual constraints in each new description. "
        "Return JSON {groups:[...],assignments:{block_alias:group_key,...}}. Each group has group_key, "
        "action=reuse|create, target_episode_id, title, description, related_episode_ids. group_key is an opaque unique "
        "request-local key. Reuse may target only a supplied episode_id. Create must have no target and needs a non-empty "
        "title and lossless background description. related_episode_ids may contain only supplied IDs. Never omit an "
        "alias or assign it to an undeclared group.\n"
        f"Existing Episodes: {json.dumps([item.prompt_value() for item in candidates], ensure_ascii=False, sort_keys=True)}\n"
        f"Blocks: {json.dumps(values, ensure_ascii=False, sort_keys=True)}"
    )


def _episode_source_context(block: StructuredBatchBlock) -> dict:
    """Keep human-readable locator context while excluding opaque identity fields."""

    hidden_values = {str(block.block_id), str(block.document_id)}
    sanitized = _sanitize_episode_context(block.locator, hidden_values)
    return sanitized if isinstance(sanitized, dict) else {}


def _sanitize_episode_context(value, hidden_values: set[str]):
    if isinstance(value, dict):
        result = {}
        for key, item in value.items():
            key_text = str(key)
            folded = key_text.casefold()
            if folded.endswith("_id") or folded in {"id", "block_id", "document_id"}:
                continue
            sanitized = _sanitize_episode_context(item, hidden_values)
            if sanitized is not _OMIT_CONTEXT_VALUE:
                result[key_text] = sanitized
        return result
    if isinstance(value, list):
        result = []
        for item in value:
            sanitized = _sanitize_episode_context(item, hidden_values)
            if sanitized is not _OMIT_CONTEXT_VALUE:
                result.append(sanitized)
        return result
    if str(value) in hidden_values:
        return _OMIT_CONTEXT_VALUE
    return value


def _episode_prompt_extracted(extracted: dict) -> dict:
    """Render display-only extracted facts without block-scoped storage keys."""

    internal_to_display: dict[str, str] = {}
    entities: list[dict] = []
    for raw in extracted.get("entities", []):
        internal_name = str(raw.get("name") or "")
        display_name = str(raw.get("_display_name") or internal_name.rsplit("\x1f", 1)[-1]).strip()
        internal_to_display[internal_name] = display_name
        entities.append(
            {
                "name": display_name,
                "entity_type": raw.get("entity_type"),
                "description": raw.get("description"),
                "properties": [
                    {str(key): value for key, value in prop.items() if not str(key).startswith("_")}
                    for prop in raw.get("properties", [])
                    if isinstance(prop, dict)
                ],
            }
        )
    edges: list[dict] = []
    for raw in extracted.get("edges", []):
        if not isinstance(raw, dict):
            continue
        edge = {str(key): value for key, value in raw.items() if not str(key).startswith("_")}
        for field_name in ("link_entity1_name", "link_entity2_name"):
            value = str(edge.get(field_name) or "")
            edge[field_name] = internal_to_display.get(value, value.rsplit("\x1f", 1)[-1])
        edges.append(edge)
    return {"entities": entities, "edges": edges}


def _validate_batch_episode_groups(
    value,
    *,
    expected_aliases: list[str],
    candidates: list[StructuredEpisodeCandidate],
) -> list[StructuredBatchEpisodeGroup]:
    if not isinstance(value, dict) or not isinstance(value.get("groups"), list):
        raise ValueError("response must contain a groups array")
    if not isinstance(value.get("assignments"), dict):
        raise ValueError("response must contain an assignments object")
    expected_alias_set = set(expected_aliases)
    allowed_episode_ids = {candidate.episode_id for candidate in candidates}
    candidate_by_id = {candidate.episode_id: candidate for candidate in candidates}
    group_keys: set[str] = set()
    groups: list[StructuredBatchEpisodeGroup] = []
    for index, raw in enumerate(value["groups"]):
        if not isinstance(raw, dict):
            raise ValueError(f"groups[{index}] must be an object")
        group_key = str(raw.get("group_key") or "").strip()
        if not group_key or group_key in group_keys:
            raise ValueError("group_key must be non-empty and unique")
        group_keys.add(group_key)
        action = str(raw.get("action") or "").strip()
        if action not in {"reuse", "create"}:
            raise ValueError("group action must be reuse or create")
        target = str(raw.get("target_episode_id") or "").strip() or None
        title = str(raw.get("title") or "").strip()
        description = str(raw.get("description") or "").strip()
        if action == "reuse":
            if target not in allowed_episode_ids:
                raise ValueError("reuse target is not a supplied Episode")
            candidate = candidate_by_id[target]
            title = title or candidate.title
            description = description or candidate.description
        elif target is not None or not title or not description:
            raise ValueError("create requires title/description and no target")
        related = raw.get("related_episode_ids", [])
        if not isinstance(related, list):
            raise ValueError("related_episode_ids must be an array")
        related_ids = list(dict.fromkeys(str(item or "").strip() for item in related))
        if any(not item or item not in allowed_episode_ids for item in related_ids):
            raise ValueError("related_episode_ids contains an unsupplied Episode")
        groups.append(
            StructuredBatchEpisodeGroup(
                group_key=group_key,
                action=action,
                block_ids=[],
                target_episode_id=target,
                title=title,
                description=description,
                related_episode_ids=[item for item in related_ids if item != target],
            )
        )
    raw_assignments = value["assignments"]
    assignments = {str(alias).strip(): str(group_key or "").strip() for alias, group_key in raw_assignments.items()}
    received_aliases = list(assignments)
    unknown_aliases = sorted(set(received_aliases) - expected_alias_set - {""})
    missing_aliases = [alias for alias in expected_aliases if alias not in assignments]
    empty_aliases = [alias for alias in received_aliases if not alias]
    unknown_group_keys = sorted(
        {
            assignments[alias]
            for alias in received_aliases
            if alias in expected_alias_set and assignments[alias] not in group_keys
        }
        - {""}
    )
    empty_group_aliases = [alias for alias in expected_aliases if alias in assignments and not assignments[alias]]
    used_group_keys = {
        assignments[alias] for alias in expected_aliases if alias in assignments and assignments[alias] in group_keys
    }
    unused_group_keys = sorted(group_keys - used_group_keys)
    if unknown_aliases or missing_aliases or empty_aliases or unknown_group_keys or empty_group_aliases:
        raise ValueError(
            "block assignment map is invalid. "
            f"Allowed assignment aliases: {json.dumps(expected_aliases)}. "
            f"Received assignment aliases: {json.dumps(received_aliases)}. "
            f"Unknown assignment aliases: {json.dumps(unknown_aliases)}. "
            f"Empty assignment aliases: {json.dumps(empty_aliases)}. "
            f"Missing assignment aliases: {json.dumps(missing_aliases)}. "
            f"Unknown assignment group keys: {json.dumps(unknown_group_keys)}. "
            f"Aliases with empty group keys: {json.dumps(empty_group_aliases)}. "
            f"Unused Episode group keys: {json.dumps(unused_group_keys)}."
        )
    group_by_key = {group.group_key: group for group in groups}
    for alias in expected_aliases:
        group_by_key[assignments[alias]].block_ids.append(alias)
    return [group for group in groups if group.group_key in used_group_keys]


__all__ = [
    "StructuredEpisodeCandidate",
    "StructuredEpisodeDecision",
    "StructuredBatchBlock",
    "StructuredBatchEpisodeAllocator",
    "StructuredBatchEpisodeGroup",
    "StructuredEpisodeAllocationError",
    "default_structured_episode_decision",
    "recall_structured_episode_candidates",
    "structured_episode_decision_from_result",
    "structured_episode_id",
    "structured_batch_episode_id",
]
