"""Planning primitives for lightweight structured Add."""

from __future__ import annotations

import asyncio
import json
import math
from dataclasses import dataclass, field
from typing import Any, Literal

from ....components.extractor.schema import parse_json_object
from ....config import StructuredDedupConfig
from ....llm import EmbedClient, LLMClient
from ....logging import get_logger
from ....typing import (
    FieldCondition,
    MemoryDbSearchHit,
    MemoryDbSearchQuery,
    MemoryRequestContext,
    SearchFilter,
)

logger = get_logger(__name__)
StructuredAction = Literal["create", "reinforce", "update", "supersede"]


@dataclass(slots=True)
class StructuredProperty:
    entity_name: str
    entity_type: str
    property_name: str
    content: str
    property_time: str
    fingerprint: str
    memory_id: str
    entity_id: str
    entity_description: str = ""
    vector: list[float] = field(default_factory=list)
    batch_contents: list[str] = field(default_factory=list)
    batch_entity_names: list[str] = field(default_factory=list)


@dataclass(slots=True)
class StructuredDecision:
    action: StructuredAction = "create"
    target_id: str | None = None
    equivalent_ids: list[str] = field(default_factory=list)
    merged_content: str | None = None
    reason: str = ""


@dataclass(slots=True)
class StructuredMergeRequest:
    """One consolidated property group and its bounded history candidates."""

    property: StructuredProperty
    candidates: list[MemoryDbSearchHit] = field(default_factory=list)
    candidate_entity_contexts: dict[str, dict[str, Any]] = field(default_factory=dict)


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
        item.batch_contents = list(dict.fromkeys(item.batch_contents or [item.content]))
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
) -> list[list[MemoryDbSearchHit]]:
    """Recall candidates concurrently by typed property inside relevant Episodes."""

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
        if episode_ids:
            searches.append(
                _search_structured_property(
                    db_reader,
                    context,
                    item,
                    conditions=[
                        *base_conditions,
                        FieldCondition(field="episode_ids", op="any", values=sorted(episode_ids)),
                    ],
                    top_k=top_k,
                )
            )
            # Memories written by the pre-structured Schema pipeline do not
            # carry ``episode_ids``. Keep same-session typed memories eligible
            # during migration without weakening the Episode-scoped primary
            # recall path.
            if context.session_id:
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
        else:
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
            query=item.content,
            top_k=top_k,
            filters=SearchFilter(must=conditions),
            mode="semantic",
            ranking="score",
        ),
        query_vector=item.vector,
    )
    return list(result.hits[:top_k])


class StructuredMergeDecider:
    """Make one best-effort merge decision for an ambiguous property."""

    def __init__(self, *, llm_client: LLMClient, max_candidates: int) -> None:
        self._llm = llm_client
        self._max_candidates = max_candidates

    async def decide(self, content: str, candidates: list[MemoryDbSearchHit]) -> StructuredDecision:
        supplied = [candidate for candidate in candidates[: self._max_candidates] if candidate.memory is not None]
        candidate_ids = {candidate.memory_id for candidate in supplied}
        prompt = _merge_prompt(content, supplied)
        try:
            response = await self._llm.chat(
                task="memory.add.structured_merge",
                messages=[{"role": "user", "content": prompt}],
            )
            value = parse_json_object(response.content)
            if not isinstance(value, dict):
                raise ValueError("merge response must be an object")
            action = value.get("action")
            if action not in {"create", "reinforce", "update", "supersede"}:
                raise ValueError("invalid merge action")
            target_id = str(value.get("target_id") or "") or None
            if action != "create" and target_id not in candidate_ids:
                raise ValueError("merge target is not a supplied candidate")
            equivalent_ids = _validated_equivalent_ids(
                value.get("equivalent_ids", []),
                candidate_ids=candidate_ids,
                target_id=target_id,
                action=action,
            )
            merged_content = str(value.get("merged_content") or "").strip() or None
            if action in {"update", "supersede"} and not merged_content:
                raise ValueError("merged_content is required for update or supersede")
            return StructuredDecision(
                action=action,
                target_id=target_id,
                equivalent_ids=equivalent_ids,
                merged_content=merged_content,
                reason=str(value.get("reason") or ""),
            )
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
            candidate_ids = {
                candidate.memory_id
                for candidate in request.candidates[: self._max_candidates]
                if candidate.memory is not None
            }
            action = raw.get("action")
            target_id = str(raw.get("target_id") or "").strip() or None
            merged_content = str(raw.get("merged_content") or "").strip() or None
            if action not in {"create", "reinforce", "update", "supersede"}:
                invalid("invalid_action")
                continue
            if action != "create" and target_id not in candidate_ids:
                invalid("invalid_target_id")
                continue
            if "equivalent_ids" not in raw:
                invalid("missing_equivalent_ids")
                continue
            try:
                equivalent_ids = _validated_equivalent_ids(
                    raw.get("equivalent_ids", []),
                    candidate_ids=candidate_ids,
                    target_id=target_id,
                    action=action,
                )
            except ValueError:
                invalid("invalid_equivalent_ids")
                continue
            if action in {"update", "supersede"} and not merged_content:
                invalid("missing_merged_content")
                continue
            resolved[index] = StructuredDecision(
                action=action,
                target_id=target_id,
                equivalent_ids=equivalent_ids,
                merged_content=merged_content,
                reason=str(raw.get("reason") or ""),
            )
            resolved_decision_count += 1
        logger.info(
            "structured_batch_merge_decisions_validated",
            supplied_decision_count=len(raw_decisions),
            resolved_decision_count=resolved_decision_count,
            invalid_reason_counts=invalid_reason_counts,
        )
        return resolved


def _merge_prompt(content: str, candidates: list[MemoryDbSearchHit]) -> str:
    values = [
        {
            "memory_id": candidate.memory_id,
            "score": candidate.score,
            "content": candidate.memory.content if candidate.memory else "",
        }
        for candidate in candidates
    ]
    return (
        "Decide how one new structured memory relates to the supplied same-scope candidates. "
        "Return one JSON object only: action=create|reinforce|update|supersede, target_id when action is not "
        "create, equivalent_ids containing every other supplied candidate that is the same memory (or []), "
        "merged_content for update/supersede, and reason. Similarity alone is not equivalence. Never reference "
        "an ID outside candidates.\n"
        f"New content: {content}\n"
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
                    "content": memory.content,
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
                "allowed_memory_ids": [candidate["memory_id"] for candidate in candidates],
                "candidates": candidates,
            }
        )
    relevant_types = {request.property.entity_type for request in requests}
    relevant_schema = [item for item in entity_schema if str(item.get("entity_type") or "") in relevant_types]
    return (
        "Resolve every new structured-memory group using its typed content and Episode background. "
        "Text similarity only retrieves candidates and never proves identity. Return one JSON object with a "
        "decisions array. Each decision must contain group_index, action=create|reinforce|update|supersede, "
        "target_id for non-create actions, equivalent_ids containing every other supplied candidate that represents "
        "the same memory in the Episode context (or []), merged_content for update/supersede (and optionally create), "
        "and reason. The equivalent_ids field is required for every decision, including create. Compare the new group "
        "with every supplied candidate. Memories are equivalent only when they describe the same real-world subject or "
        "entity, the same fact or claim under the supplied Schema property, and compatible contextual validity. Entity "
        "names are soft evidence rather than mandatory equality keys, so a display-name variation may still refer to the "
        "same subject, while equal property values on different subjects remain distinct. Newer compatible information may "
        "update an existing fact; contradictory information may supersede it. Similarity only retrieves candidates and does "
        "not make them equivalent. Copy target_id and equivalent_ids only from that group's allowed_memory_ids. Never use "
        "an entity_id, new-entity ID, or Episode ID as target_id or equivalent_ids. "
        "When merged_content is present, write one concise canonical statement that preserves "
        "each unique compatible fact once. Use the language of the new property content unless the supplied Schema "
        "explicitly requires another language. Do not concatenate translations or equivalent clauses. Do not repeat "
        "paraphrases. Never reference an ID outside that group's candidates.\n"
        f"Relevant Entity Schema: {json.dumps(relevant_schema, ensure_ascii=False, sort_keys=True)}\n"
        f"Current Episode: {json.dumps(current_episode, ensure_ascii=False, sort_keys=True)}\n"
        f"Known Episode contexts: {json.dumps(episode_contexts, ensure_ascii=False, sort_keys=True)}\n"
        f"Groups: {json.dumps(groups, ensure_ascii=False, sort_keys=True)}"
    )


def _validated_equivalent_ids(
    raw: Any,
    *,
    candidate_ids: set[str],
    target_id: str | None,
    action: Any,
) -> list[str]:
    """Validate and normalize the model-selected equivalent candidate set."""

    if not isinstance(raw, list) or any(not isinstance(value, str) or not value for value in raw):
        raise ValueError("equivalent_ids must be an array of non-empty candidate IDs")
    values = list(dict.fromkeys(raw))
    if any(value not in candidate_ids for value in values):
        raise ValueError("equivalent_ids contains an ID outside supplied candidates")
    if action == "create":
        if values:
            raise ValueError("create cannot retire equivalent candidates")
        return []
    return [value for value in values if value != target_id]
