"""Planning primitives for lightweight structured Add."""

from __future__ import annotations

import asyncio
import copy
import json
import math
from dataclasses import dataclass, field
from typing import Any, Literal

from ....components.extractor.schema import parse_json_object
from ....config import StructuredDedupConfig
from ....llm import EmbedClient, LLMClient
from ....logging import get_logger
from ....structured_content import (
    is_structured_content,
    merge_structured_contents,
    structured_content_from_metadata,
    structured_content_semantic_text,
    unique_structured_values,
)
from ....typing import (
    FieldCondition,
    MemoryDbSearchHit,
    MemoryDbSearchQuery,
    MemoryRequestContext,
    SearchFilter,
)

logger = get_logger(__name__)
StructuredAction = Literal["create", "reinforce", "update", "supersede"]
StructuredHistoryScope = Literal["episode", "session"]


def _deterministic_relation_content(incoming: Any, historical: Any, relation: str) -> Any | None:
    """Derive a revision body from validated facts, never from model text."""

    if relation == "duplicate":
        if is_structured_content(incoming) and is_structured_content(historical):
            return None
        return None if str(incoming).strip() == str(historical).strip() else _UNSUPPORTED_CONTENT
    if relation == "complement":
        if is_structured_content(incoming) and is_structured_content(historical):
            return merge_structured_contents([historical, incoming])
        return _UNSUPPORTED_CONTENT
    if relation == "conflict":
        if is_structured_content(incoming) and is_structured_content(historical):
            return incoming
        return _UNSUPPORTED_CONTENT
    return _UNSUPPORTED_CONTENT


_UNSUPPORTED_CONTENT = object()


@dataclass(slots=True)
class StructuredProperty:
    entity_name: str
    entity_type: str
    property_name: str
    content: Any
    property_time: str
    fingerprint: str
    memory_id: str
    entity_id: str
    entity_description: str = ""
    vector: list[float] = field(default_factory=list)
    batch_contents: list[Any] = field(default_factory=list)
    batch_entity_names: list[str] = field(default_factory=list)
    entity_keys: list[str] = field(default_factory=list)
    source_block_ids: list[str] = field(default_factory=list)
    source_documents: list[dict[str, Any]] = field(default_factory=list)
    episode_id: str | None = None
    comparison_episode_ids: list[str] = field(default_factory=list)


@dataclass(slots=True)
class StructuredDecision:
    action: StructuredAction = "create"
    target_id: str | None = None
    equivalent_ids: list[str] = field(default_factory=list)
    merged_content: Any | None = None
    reason: str = ""


@dataclass(slots=True)
class StructuredMergeRequest:
    """One consolidated property group and its bounded history candidates."""

    property: StructuredProperty
    candidates: list[MemoryDbSearchHit] = field(default_factory=list)
    candidate_entity_contexts: dict[str, dict[str, Any]] = field(default_factory=dict)
    current_episode: dict[str, Any] = field(default_factory=dict)


async def consolidate_structured_batch(
    llm_client: LLMClient,
    properties: list[StructuredProperty],
    *,
    entity_schema: list[dict[str, Any]],
    episode_contexts: dict[str, dict[str, Any]],
    max_repair_attempts: int = 1,
) -> list[StructuredProperty]:
    """Collect new-new fact relationships before any historical recall.

    Only properties inside the same resolved Episode and Schema slot are offered
    together. The model never authors content; code unions validated source facts.
    Invalid, incomplete, or uncertain output leaves every fact separate.
    """

    buckets: dict[tuple[str, str, str], list[int]] = {}
    for index, item in enumerate(properties):
        buckets.setdefault((item.episode_id or "", item.entity_type, item.property_name), []).append(index)
    groups = [indexes for indexes in buckets.values() if len(indexes) > 1]
    entity_buckets: dict[tuple[str, str], set[str]] = {}
    for item in properties:
        entity_buckets.setdefault((item.episode_id or "", item.entity_type), set()).update(item.entity_keys)
    if not groups and not any(len(keys) > 1 for keys in entity_buckets.values()):
        return properties
    prompt = _internal_consolidation_prompt(properties, groups, entity_schema, episode_contexts)
    last_content = ""
    last_error = "unknown validation error"
    for attempt in range(1 + max_repair_attempts):
        current_prompt = prompt
        if attempt:
            current_prompt += (
                "\nPrevious answer:\n"
                + last_content
                + "\nValidation error: "
                + last_error
                + "\nReturn the complete corrected JSON object only."
            )
        try:
            response = await llm_client.chat(
                task="memory.add.structured_batch_consolidate",
                messages=[{"role": "user", "content": current_prompt}],
            )
            last_content = response.content
            value = parse_json_object(response.content)
            return _apply_internal_clusters(value, properties, groups)
        except Exception as exc:  # noqa: BLE001 - separate facts are the safe fallback
            last_error = str(exc)
    logger.warning("structured_batch_consolidation_fallback", error=last_error)
    return properties


def _internal_consolidation_prompt(
    properties: list[StructuredProperty],
    groups: list[list[int]],
    entity_schema: list[dict[str, Any]],
    episode_contexts: dict[str, dict[str, Any]],
) -> str:
    candidates = []
    for group_index, indexes in enumerate(groups):
        candidates.append(
            {
                "group_index": group_index,
                "allowed_property_indexes": indexes,
                "episode": episode_contexts.get(properties[indexes[0]].episode_id or "", {}),
                "properties": [
                    {
                        "property_index": index,
                        "entity_name": properties[index].entity_name,
                        "entity_description": properties[index].entity_description,
                        "entity_type": properties[index].entity_type,
                        "property_name": properties[index].property_name,
                        "content": properties[index].content,
                        "source_block_ids": properties[index].source_block_ids,
                    }
                    for index in indexes
                ],
            }
        )
    entity_candidates: list[dict[str, Any]] = []
    entity_buckets: dict[tuple[str, str], dict[str, StructuredProperty]] = {}
    for item in properties:
        bucket = entity_buckets.setdefault((item.episode_id or "", item.entity_type), {})
        for entity_key in item.entity_keys:
            bucket.setdefault(entity_key, item)
    for (episode_id, entity_type), values in entity_buckets.items():
        if len(values) < 2:
            continue
        entity_candidates.append(
            {
                "episode": episode_contexts.get(episode_id, {}),
                "entity_type": entity_type,
                "entities": [
                    {
                        "entity_key": entity_key,
                        "name": item.entity_name,
                        "description": item.entity_description,
                        "facts": [
                            prop.content
                            for prop in properties
                            if entity_key in prop.entity_keys and prop.episode_id == episode_id
                        ],
                    }
                    for entity_key, item in values.items()
                ],
            }
        )
    return (
        "Classify relationships only among newly extracted structured facts; no historical memories are present. "
        "Return JSON {entity_groups:[...],fact_groups:[...]}. Each entity_group has relation=same_entity and "
        "member_entity_keys (at least two keys from one allowed Episode/entity_type group). Group entity aliases only "
        "when their complete extracted facts prove they are the same real-world subject; similar display names alone "
        "are insufficient. Each fact_group has relation=duplicate|complement and member_indexes (at least two indexes "
        "from one allowed group). Group facts only when they describe the same real-world subject and the same compatible "
        "fact in the same Episode and Schema property. Generated titles and textual similarity are not identity. "
        "Contradictory, uncertain, or different-subject facts must remain ungrouped. Unmentioned properties remain "
        "separate. A property index may appear at most once. Never reference an index outside its allowed group. "
        "Do not return canonical_name, canonical_description, canonical_content, merged_content, rewritten facts, or "
        "storage actions; backend code preserves and combines the supplied source facts deterministically.\n"
        f"Entity Schema: {json.dumps(entity_schema, ensure_ascii=False, sort_keys=True)}\n"
        f"Candidate entities: {json.dumps(entity_candidates, ensure_ascii=False, sort_keys=True)}\n"
        f"Candidate groups: {json.dumps(candidates, ensure_ascii=False, sort_keys=True)}"
    )


def _apply_internal_clusters(
    value: Any,
    properties: list[StructuredProperty],
    groups: list[list[int]],
) -> list[StructuredProperty]:
    if not isinstance(value, dict) or set(value) - {"entity_groups", "fact_groups"}:
        raise ValueError("batch consolidation may contain only entity_groups and fact_groups")
    if not isinstance(value.get("fact_groups", []), list):
        raise ValueError("fact_groups must be an array")
    working = copy.deepcopy(properties)
    _apply_internal_entity_groups(value.get("entity_groups", []), working)
    allowed_sets = [set(group) for group in groups]
    used: set[int] = set()
    replacements: dict[int, StructuredProperty] = {}
    removed: set[int] = set()
    for raw in value.get("fact_groups", []):
        if not isinstance(raw, dict) or not isinstance(raw.get("member_indexes"), list):
            raise ValueError("fact group must contain member_indexes")
        if set(raw) != {"relation", "member_indexes"}:
            raise ValueError("fact group may contain only relation and member_indexes")
        if raw.get("relation") not in {"duplicate", "complement"}:
            raise ValueError("fact group relation must be duplicate or complement")
        indexes = [int(index) for index in raw["member_indexes"]]
        if len(indexes) < 2 or len(indexes) != len(set(indexes)) or any(index in used for index in indexes):
            raise ValueError("cluster indexes must be unique and used once")
        if not any(set(indexes).issubset(allowed) for allowed in allowed_sets):
            raise ValueError("cluster crosses an Episode or Schema group")
        sources = [working[index].content for index in indexes]
        if all(is_structured_content(source) for source in sources):
            content = merge_structured_contents(sources)
        else:
            normalized = [str(source).strip() for source in sources]
            if len(set(normalized)) != 1:
                raise ValueError("unequal scalar properties cannot be consolidated")
            content = normalized[0]
        used.update(indexes)
        first = working[indexes[0]]
        original_contents = unique_structured_values(
            [value for index in indexes for value in (working[index].batch_contents or [working[index].content])]
        )
        first.content = content
        first.batch_contents = original_contents
        first.batch_entity_names = list(
            dict.fromkeys(
                value
                for index in indexes
                for value in (working[index].batch_entity_names or [working[index].entity_name])
            )
        )
        first.entity_keys = list(
            dict.fromkeys(value for index in indexes for value in (working[index].entity_keys or []))
        )
        first.source_block_ids = list(
            dict.fromkeys(value for index in indexes for value in working[index].source_block_ids)
        )
        first.source_documents = _unique_dicts(
            [value for index in indexes for value in working[index].source_documents]
        )
        replacements[indexes[0]] = first
        removed.update(indexes[1:])
    return [replacements.get(index, item) for index, item in enumerate(working) if index not in removed]


def _apply_internal_entity_groups(raw_groups: Any, properties: list[StructuredProperty]) -> None:
    if not isinstance(raw_groups, list):
        raise ValueError("entity_groups must be an array")
    by_key: dict[str, list[StructuredProperty]] = {}
    identity: dict[str, tuple[str, str]] = {}
    for item in properties:
        for entity_key in item.entity_keys:
            by_key.setdefault(entity_key, []).append(item)
            identity[entity_key] = (item.episode_id or "", item.entity_type)
    used: set[str] = set()
    for raw in raw_groups:
        if not isinstance(raw, dict) or not isinstance(raw.get("member_entity_keys"), list):
            raise ValueError("entity group must contain member_entity_keys")
        if set(raw) != {"relation", "member_entity_keys"} or raw.get("relation") != "same_entity":
            raise ValueError("entity group may only declare same_entity membership")
        keys = [str(value or "").strip() for value in raw["member_entity_keys"]]
        if len(keys) < 2 or len(keys) != len(set(keys)) or any(key in used or key not in by_key for key in keys):
            raise ValueError("entity cluster keys must be known, unique, and used once")
        if len({identity[key] for key in keys}) != 1:
            raise ValueError("entity cluster crosses an Episode or entity type")
        used.update(keys)
        anchor = by_key[keys[0]][0]
        canonical_id = anchor.entity_id
        name = anchor.entity_name
        description = anchor.entity_description
        all_keys = list(dict.fromkeys(key for member in keys for item in by_key[member] for key in item.entity_keys))
        for member in keys:
            for item in by_key[member]:
                item.entity_id = canonical_id
                item.entity_name = name
                if description:
                    item.entity_description = description
                item.entity_keys = all_keys


def _unique_dicts(values: list[dict[str, Any]]) -> list[dict[str, Any]]:
    result: list[dict[str, Any]] = []
    for value in values:
        if value not in result:
            result.append(value)
    return result


def classify_similarity(score: float | None, config: StructuredDedupConfig) -> str:
    """Classify one best candidate using configured deterministic thresholds."""

    if score is None or score < config.create_below:
        return "create"
    return "ambiguous"


async def batch_embed(
    embed_client: EmbedClient,
    texts: list[str],
    *,
    batch_size: int,
    task: str,
) -> list[list[float]]:
    """Embed bounded chunks while preserving input order."""

    vectors: list[list[float]] = []
    for start in range(0, len(texts), batch_size):
        batch = texts[start : start + batch_size]
        response = await embed_client.embed(task=task, text=batch)
        if len(response.embeddings) != len(batch):
            raise ValueError(
                f"structured embedding count mismatch: expected {len(batch)}, got {len(response.embeddings)}"
            )
        vectors.extend(response.embeddings)
    return vectors


def consolidate_structured_properties(
    properties: list[StructuredProperty],
    *,
    similarity_at_or_above: float,
) -> list[StructuredProperty]:
    """Group same-typed near-duplicates from one extraction without using titles."""

    groups: list[StructuredProperty] = []
    for item in properties:
        item.batch_contents = unique_structured_values(item.batch_contents or [item.content])
        item.batch_entity_names = list(dict.fromkeys(item.batch_entity_names or [item.entity_name]))
        matched: StructuredProperty | None = None
        for group in groups:
            if (
                group.entity_id != item.entity_id
                or group.entity_type != item.entity_type
                or group.property_name != item.property_name
            ):
                continue
            if _cosine_similarity(group.vector, item.vector) >= similarity_at_or_above:
                matched = group
                break
        if matched is None:
            groups.append(item)
            continue
        for content in item.batch_contents:
            if content not in matched.batch_contents:
                matched.batch_contents.append(content)
        for entity_name in item.batch_entity_names:
            if entity_name not in matched.batch_entity_names:
                matched.batch_entity_names.append(entity_name)
    return groups


def _cosine_similarity(left: list[float], right: list[float]) -> float:
    if not left or len(left) != len(right):
        return 0.0
    left_norm = math.sqrt(sum(value * value for value in left))
    right_norm = math.sqrt(sum(value * value for value in right))
    if left_norm == 0 or right_norm == 0:
        return 0.0
    return sum(a * b for a, b in zip(left, right, strict=True)) / (left_norm * right_norm)


async def recall_structured_candidates(
    db_reader: Any,
    context: MemoryRequestContext,
    properties: list[StructuredProperty],
    *,
    episode_ids: set[str] | None = None,
    top_k: int,
    history_scope: StructuredHistoryScope = "episode",
) -> list[list[MemoryDbSearchHit]]:
    """Recall typed candidates within the configured history boundary."""

    async def recall(item: StructuredProperty) -> list[MemoryDbSearchHit]:
        base_conditions = [
            FieldCondition(field="status", op="match", value="active"),
            FieldCondition(field="entity_type", op="match", value=item.entity_type),
            FieldCondition(field="property_name", op="match", value=item.property_name),
        ]
        if context.user_id:
            base_conditions.append(FieldCondition(field="user_id", op="match", value=context.user_id))
        if context.app_id:
            base_conditions.append(FieldCondition(field="app_id", op="match", value=context.app_id))
        if context.agent_id:
            base_conditions.append(FieldCondition(field="agent_id", op="match", value=context.agent_id))

        searches = []
        if history_scope == "session":
            # Fixed-schema callers may intentionally group every observation
            # into a new Episode. In that case semantic merge still needs the
            # active typed history from the same task/session.
            if not context.session_id:
                return []
            searches.append(
                _search_structured_property(
                    db_reader,
                    context,
                    item,
                    conditions=[
                        *base_conditions,
                        FieldCondition(field="session_id", op="match", value=context.session_id),
                    ],
                    top_k=top_k,
                )
            )
        else:
            item_episode_ids = set(item.comparison_episode_ids) or set(episode_ids or [])
            if item_episode_ids:
                searches.append(
                    _search_structured_property(
                        db_reader,
                        context,
                        item,
                        conditions=[
                            *base_conditions,
                            FieldCondition(field="episode_ids", op="any", values=sorted(item_episode_ids)),
                        ],
                        top_k=top_k,
                    )
                )
            # Memories written by the pre-structured Schema pipeline do not
            # carry ``episode_ids``. Keep same-session typed memories eligible
            # during migration without weakening the Episode-scoped primary
            # recall path.
            if item_episode_ids and context.session_id:
                searches.append(
                    _search_structured_property(
                        db_reader,
                        context,
                        item,
                        conditions=[
                            *base_conditions,
                            FieldCondition(field="session_id", op="match", value=context.session_id),
                            FieldCondition(field="episode_ids", op="is_empty"),
                        ],
                        top_k=top_k,
                    )
                )
            elif not item_episode_ids:
                searches.append(
                    _search_structured_property(
                        db_reader,
                        context,
                        item,
                        conditions=base_conditions,
                        top_k=top_k,
                    )
                )

        groups = await asyncio.gather(*searches)
        by_id: dict[str, MemoryDbSearchHit] = {}
        for hit in (hit for group in groups for hit in group):
            existing = by_id.get(hit.memory_id)
            if existing is None or hit.score > existing.score:
                by_id[hit.memory_id] = hit
        return sorted(by_id.values(), key=lambda hit: (-hit.score, hit.memory_id))[:top_k]

    return list(await asyncio.gather(*(recall(item) for item in properties)))


async def _search_structured_property(
    db_reader: Any,
    context: MemoryRequestContext,
    item: StructuredProperty,
    *,
    conditions: list[FieldCondition],
    top_k: int,
) -> list[MemoryDbSearchHit]:
    result = await db_reader.search_dense(
        context,
        MemoryDbSearchQuery(
            query=structured_content_semantic_text(item.content),
            top_k=top_k,
            filters=SearchFilter(must=conditions),
            mode="semantic",
            ranking="score",
        ),
        query_vector=item.vector,
    )
    return list(result.hits[:top_k])


class StructuredMergeDecider:
    """Classify history relationships and derive storage actions in code."""

    def __init__(self, *, llm_client: LLMClient, max_candidates: int) -> None:
        self._llm = llm_client
        self._max_candidates = max_candidates

    async def decide(self, content: Any, candidates: list[MemoryDbSearchHit]) -> StructuredDecision:
        supplied = [candidate for candidate in candidates[: self._max_candidates] if candidate.memory is not None]
        prompt = _merge_prompt(content, supplied)
        try:
            response = await self._llm.chat(
                task="memory.add.structured_merge",
                messages=[{"role": "user", "content": prompt}],
            )
            value = parse_json_object(response.content)
            return _relation_decision(value, content=content, candidates=supplied)
        except Exception as exc:  # noqa: BLE001 - uncertainty safely creates a new memory
            logger.info(
                "structured_merge_fallback_create",
                error_type=type(exc).__name__,
                candidate_count=len(supplied),
            )
            return StructuredDecision(action="create", reason="invalid_or_unavailable_merge_decision")

    async def decide_batch(
        self,
        requests: list[StructuredMergeRequest],
        *,
        current_episode: dict[str, Any],
        episode_contexts: dict[str, dict[str, Any]],
        entity_schema: list[dict[str, Any]] | None = None,
    ) -> list[StructuredDecision]:
        """Resolve every ambiguous group through one context-bearing Chat call."""

        if not requests:
            return []
        prompt = _batch_merge_prompt(
            requests,
            current_episode,
            episode_contexts,
            entity_schema or [],
            self._max_candidates,
        )
        defaults = [StructuredDecision(action="create", reason="invalid_or_unavailable_batch_merge") for _ in requests]
        try:
            response = await self._llm.chat(
                task="memory.add.structured_merge_batch",
                messages=[{"role": "user", "content": prompt}],
            )
            value = parse_json_object(response.content)
            raw_decisions = value.get("decisions") if isinstance(value, dict) else None
            if not isinstance(raw_decisions, list):
                raise ValueError("batch merge response requires decisions array")
        except Exception as exc:  # noqa: BLE001 - uncertainty safely creates each group
            logger.info(
                "structured_batch_merge_fallback_create",
                error_type=type(exc).__name__,
                group_count=len(requests),
            )
            return defaults

        resolved = list(defaults)
        seen: set[int] = set()
        invalid_reason_counts: dict[str, int] = {}
        resolved_decision_count = 0

        def invalid(reason: str) -> None:
            invalid_reason_counts[reason] = invalid_reason_counts.get(reason, 0) + 1

        for raw in raw_decisions:
            if not isinstance(raw, dict):
                invalid("decision_not_object")
                continue
            try:
                index = int(raw.get("group_index"))
            except (TypeError, ValueError):
                invalid("invalid_group_index")
                continue
            if index < 0 or index >= len(requests):
                invalid("invalid_group_index")
                continue
            if index in seen:
                invalid("duplicate_group_index")
                continue
            seen.add(index)
            request = requests[index]
            try:
                decision = _relation_decision(
                    raw,
                    content=request.property.content,
                    candidates=[
                        candidate
                        for candidate in request.candidates[: self._max_candidates]
                        if candidate.memory is not None
                    ],
                    require_group_index=True,
                )
            except ValueError:
                invalid("invalid_relation_decision")
                continue
            resolved[index] = decision
            resolved_decision_count += 1
        logger.info(
            "structured_batch_merge_decisions_validated",
            supplied_decision_count=len(raw_decisions),
            resolved_decision_count=resolved_decision_count,
            invalid_reason_counts=invalid_reason_counts,
        )
        return _enforce_single_entity_resolution(resolved, requests)


def _merge_prompt(content: Any, candidates: list[MemoryDbSearchHit]) -> str:
    values = [
        {
            "memory_id": candidate.memory_id,
            "score": candidate.score,
            "content": structured_content_from_metadata(candidate.memory.metadata, candidate.memory.content)
            if candidate.memory
            else "",
        }
        for candidate in candidates
    ]
    return (
        "Classify only whether one new structured fact group has a valid historical relation. Return one JSON "
        "object with relation=duplicate|complement|conflict, target_id, optional duplicate_ids, and reason. Return "
        "relation=null with no target when no candidate has the same subject, Schema fact, and compatible context. "
        "Similarity only retrieves candidates. Do not return an action, merged_content, canonical content, or rewritten "
        "fact. Never reference an ID outside candidates.\n"
        f"New content: {json.dumps(content, ensure_ascii=False, sort_keys=True)}\n"
        f"Candidates: {json.dumps(values, ensure_ascii=False)}"
    )


def _batch_merge_prompt(
    requests: list[StructuredMergeRequest],
    current_episode: dict[str, Any],
    episode_contexts: dict[str, dict[str, Any]],
    entity_schema: list[dict[str, Any]],
    max_candidates: int,
) -> str:
    groups: list[dict[str, Any]] = []
    for index, request in enumerate(requests):
        candidates: list[dict[str, Any]] = []
        for candidate in request.candidates[:max_candidates]:
            memory = candidate.memory
            if memory is None:
                continue
            episode_ids = list(getattr(memory, "episode_ids", []) or memory.metadata.get("episode_ids", []))
            candidates.append(
                {
                    "memory_id": candidate.memory_id,
                    "retrieval_score": candidate.score,
                    "content": structured_content_from_metadata(memory.metadata, memory.content),
                    "entity": request.candidate_entity_contexts.get(
                        candidate.memory_id,
                        {
                            "entity_id": memory.entity_id,
                            "entity_type": memory.entity_type,
                        },
                    ),
                    "episode_ids": episode_ids,
                    "episode_contexts": [episode_contexts[item] for item in episode_ids if item in episode_contexts],
                }
            )
        groups.append(
            {
                "group_index": index,
                "entity_type": request.property.entity_type,
                "property_name": request.property.property_name,
                "new_entity": {
                    "entity_id": request.property.entity_id,
                    "name": request.property.entity_name,
                    "description": request.property.entity_description,
                    "entity_type": request.property.entity_type,
                },
                "new_contents": request.property.batch_contents or [request.property.content],
                "current_episode": request.current_episode or current_episode,
                "source_block_ids": request.property.source_block_ids,
                "allowed_memory_ids": [candidate["memory_id"] for candidate in candidates],
                "candidates": candidates,
            }
        )
    relevant_types = {request.property.entity_type for request in requests}
    relevant_schema = [item for item in entity_schema if str(item.get("entity_type") or "") in relevant_types]
    return (
        "Classify only the historical relationships for every new structured-memory group using typed facts and "
        "Episode background. Text similarity only retrieves candidates and never proves identity. Return one JSON "
        "object with a decisions array. Include a decision only when a supplied candidate has relation="
        "duplicate|complement|conflict. Each decision contains group_index, relation, target_id, optional duplicate_ids "
        "for other supplied candidates that are the same historical memory, and reason. Omit unrelated groups. Compare the new group "
        "with every supplied candidate. Memories are equivalent only when they describe the same real-world subject or "
        "entity, the same fact or claim under the supplied Schema property, and compatible contextual validity. Entity "
        "names are soft evidence rather than mandatory equality keys, so a display-name variation may still refer to the "
        "same subject, while equal property values on different subjects remain distinct. Use duplicate when the new facts "
        "add evidence but no information, complement when they add compatible information, and conflict when they assert an "
        "incompatible state for that same subject/property/context. Copy target_id and duplicate_ids only from that group's "
        "allowed_memory_ids. Never use an entity_id, new-entity ID, or Episode ID as a memory ID. "
        "Groups with the same new_entity.entity_id are properties of one extracted subject. Their non-create targets and "
        "equivalent candidates must belong to one compatible historical entity; if they would map that subject to "
        "different historical entities, choose create for those groups. "
        "Do not return an action, merged_content, canonical content, rewritten facts, summaries, or storage instructions. "
        "Backend code derives reinforcement and revisions from the supplied immutable facts. Generated entity names are "
        "display evidence, not fingerprints or uniqueness keys. Never reference an ID outside that group's candidates.\n"
        f"Relevant Entity Schema: {json.dumps(relevant_schema, ensure_ascii=False, sort_keys=True)}\n"
        f"Current Episode: {json.dumps(current_episode, ensure_ascii=False, sort_keys=True)}\n"
        f"Known Episode contexts: {json.dumps(episode_contexts, ensure_ascii=False, sort_keys=True)}\n"
        f"Groups: {json.dumps(groups, ensure_ascii=False, sort_keys=True)}"
    )


def _relation_decision(
    value: Any,
    *,
    content: Any,
    candidates: list[MemoryDbSearchHit],
    require_group_index: bool = False,
) -> StructuredDecision:
    """Validate one relation-only response and derive a storage decision."""

    if not isinstance(value, dict):
        raise ValueError("relation decision must be an object")
    allowed = {"relation", "target_id", "duplicate_ids", "reason"}
    if require_group_index:
        allowed.add("group_index")
    if set(value) - allowed:
        raise ValueError("relation decision contains model-authored storage fields")
    relation = value.get("relation")
    if relation is None:
        if value.get("target_id") or value.get("duplicate_ids"):
            raise ValueError("unrelated decision cannot select memories")
        return StructuredDecision(action="create", reason=str(value.get("reason") or "no_historical_relation"))
    if relation not in {"duplicate", "complement", "conflict"}:
        raise ValueError("invalid historical relation")
    supplied = {candidate.memory_id: candidate for candidate in candidates if candidate.memory is not None}
    target_id = str(value.get("target_id") or "").strip()
    if target_id not in supplied:
        raise ValueError("relation target is not a supplied candidate")
    duplicate_ids = value.get("duplicate_ids", [])
    if not isinstance(duplicate_ids, list) or any(not isinstance(item, str) or not item for item in duplicate_ids):
        raise ValueError("duplicate_ids must be an array of candidate IDs")
    equivalent_ids = list(dict.fromkeys(item for item in duplicate_ids if item != target_id))
    if any(item not in supplied for item in equivalent_ids):
        raise ValueError("duplicate_ids contains an ID outside supplied candidates")
    target = supplied[target_id].memory
    assert target is not None
    historical = structured_content_from_metadata(target.metadata, target.content)
    derived = _deterministic_relation_content(content, historical, relation)
    if derived is _UNSUPPORTED_CONTENT:
        raise ValueError("relation cannot be materialized deterministically for this property type")
    action = {"duplicate": "reinforce", "complement": "update", "conflict": "supersede"}[relation]
    return StructuredDecision(
        action=action,
        target_id=target_id,
        equivalent_ids=equivalent_ids,
        merged_content=derived,
        reason=str(value.get("reason") or f"historical_{relation}"),
    )


def _enforce_single_entity_resolution(
    decisions: list[StructuredDecision],
    requests: list[StructuredMergeRequest],
) -> list[StructuredDecision]:
    """Keep all properties of one extracted entity on one canonical subject.

    Property decisions are batched, but an extracted entity is one structural
    subject. If the model maps its properties/equivalents to different stored
    entities, preserving the new entity is safer than splitting its facts.
    """

    indexes_by_entity: dict[str, list[int]] = {}
    for index, request in enumerate(requests):
        indexes_by_entity.setdefault(request.property.entity_id, []).append(index)

    resolved = list(decisions)
    for indexes in indexes_by_entity.values():
        historical_entity_ids: set[str] = set()
        for index in indexes:
            decision = resolved[index]
            selected_ids = [decision.target_id, *decision.equivalent_ids]
            contexts = requests[index].candidate_entity_contexts
            for memory_id in selected_ids:
                if not memory_id:
                    continue
                entity_id = str((contexts.get(memory_id) or {}).get("entity_id") or "")
                if entity_id:
                    historical_entity_ids.add(entity_id)
        if len(historical_entity_ids) <= 1:
            continue
        for index in indexes:
            resolved[index] = StructuredDecision(
                action="create",
                reason="conflicting_historical_entity_resolution",
            )
    return resolved
