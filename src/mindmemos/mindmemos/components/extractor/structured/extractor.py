"""Single-pass, strict fixed-Schema extraction for lightweight Add requests."""

from __future__ import annotations

import copy
import json
from datetime import UTC, datetime
from typing import Any

from ....config import StructuredExtractionConfig
from ....errors import ApiError
from ....llm import LLMClient
from ....logging import get_logger
from ..schema._schema_utils import parse_json_object, strip_for_generation
from .episode import StructuredEpisodeCandidate

logger = get_logger(__name__)


class StructuredExtractionError(ApiError):
    """Raised when fixed-Schema extraction cannot be validated."""

    status_code = 422
    code = "structured.extraction_invalid"


class StructuredExtractor:
    """Extract zero or more entities using one fixed-Schema model call."""

    def __init__(
        self,
        *,
        llm_client: LLMClient,
        entity_manager: Any,
        config: StructuredExtractionConfig,
    ) -> None:
        self._llm = llm_client
        self._entity_manager = entity_manager
        self._config = config

    @property
    def schema_version(self) -> str:
        """Return the actual project Entity Schema version used for extraction."""

        file_path = getattr(self._entity_manager, "file_path", None)
        name = getattr(file_path, "name", None)
        return str(name or "unknown")

    @property
    def schema_context(self) -> list[dict[str, Any]]:
        """Return the request-scoped generation Schema used by contextual merge."""

        return strip_for_generation(copy.deepcopy(self._entity_manager.get_all_dicts()))

    async def extract(
        self,
        *,
        content: str,
        event_time: str,
        prompt_language: str | None,
        episode_candidates: list[StructuredEpisodeCandidate] | None = None,
    ) -> dict[str, Any]:
        """Return a normalized structured result or fail before persistence."""

        schema = self.schema_context
        prompt = _extraction_prompt(
            schema=schema,
            content=content,
            event_time=event_time,
            prompt_language=prompt_language,
            episode_candidates=episode_candidates,
        )
        last_error = "unknown validation error"
        last_content = ""
        total_calls = 1 + self._config.max_repair_attempts
        for attempt in range(total_calls):
            current_prompt = prompt
            if attempt:
                current_prompt += (
                    "\n\nPrevious answer:\n"
                    + last_content
                    + "\n\nValidation error: "
                    + last_error
                    + "\nReturn the complete corrected JSON object only."
                )
            response = await self._llm.chat(
                task="memory.add.structured_extract",
                messages=[{"role": "user", "content": current_prompt}],
            )
            last_content = response.content
            try:
                parsed = parse_json_object(response.content)
                normalized = self._validate_and_normalize(
                    parsed,
                    event_time=event_time,
                    schema=schema,
                    episode_candidates=episode_candidates,
                    allow_relation_only=attempt > 0,
                )
            except (TypeError, ValueError, json.JSONDecodeError) as exc:
                last_error = str(exc)
                logger.info(
                    "structured_extraction_validation_failed",
                    attempt=attempt + 1,
                    max_attempts=total_calls,
                    error_type=type(exc).__name__,
                )
                continue
            logger.info(
                "structured_extraction_completed",
                attempts=attempt + 1,
                entity_count=len(normalized["entities"]),
                property_count=sum(len(entity["properties"]) for entity in normalized["entities"]),
            )
            return normalized
        raise StructuredExtractionError(
            f"Structured extraction failed Schema validation: {last_error}",
            details={"attempts": total_calls},
        )

    def _validate_and_normalize(
        self,
        value: Any,
        *,
        event_time: str,
        schema: list[dict[str, Any]],
        episode_candidates: list[StructuredEpisodeCandidate] | None,
        allow_relation_only: bool,
    ) -> dict[str, Any]:
        if not isinstance(value, dict):
            raise ValueError("response must be a JSON object")
        entities = value.get("entities", [])
        edges = value.get("edges", [])
        if not isinstance(entities, list) or not isinstance(edges, list):
            raise ValueError("entities and edges must be arrays")
        if len(entities) > self._config.max_entities:
            raise ValueError(f"entity count exceeds max_entities={self._config.max_entities}")

        property_names = _schema_property_names(schema)
        normalized_entities: list[dict[str, Any]] = []
        seen_names: set[str] = set()
        default_date = _event_date(event_time)
        for index, entity in enumerate(entities):
            if not isinstance(entity, dict):
                raise ValueError(f"entity[{index}] must be an object")
            name = str(entity.get("name") or "").strip()
            if not name:
                raise ValueError(f"entity[{index}] requires a non-empty name")
            name_key = name.casefold()
            if name_key in seen_names:
                raise ValueError(f"duplicate entity name: {name}")
            seen_names.add(name_key)

            entity_type = str(entity.get("entity_type") or "").strip()
            if entity_type not in property_names:
                raise ValueError(f"unknown entity_type: {entity_type}")
            properties = entity.get("properties", [])
            if not isinstance(properties, list):
                raise ValueError(f"entity[{index}].properties must be an array")
            if len(properties) > self._config.max_properties_per_entity:
                raise ValueError(
                    f"property count exceeds max_properties_per_entity={self._config.max_properties_per_entity}"
                )

            normalized_properties: list[dict[str, Any]] = []
            for prop_index, prop in enumerate(properties):
                if not isinstance(prop, dict):
                    raise ValueError(f"entity[{index}].properties[{prop_index}] must be an object")
                property_name = str(prop.get("property_name") or "").strip()
                if property_name not in property_names[entity_type]:
                    raise ValueError(f"unknown property_name {property_name!r} for entity_type {entity_type!r}")
                if "value" not in prop or prop.get("value") is None:
                    raise ValueError(f"property {property_name!r} requires a value")
                normalized_value = _normalize_property_value(prop["value"])
                if not normalized_value:
                    raise ValueError(f"property {property_name!r} requires a non-empty value")
                normalized_properties.append(
                    {
                        "property_name": property_name,
                        "value": normalized_value,
                        "time": str(prop.get("time") or default_date),
                        "operation": "set",
                    }
                )
            normalized_entities.append(
                {
                    "name": name,
                    "entity_type": entity_type,
                    "description": str(entity.get("description") or "").strip(),
                    "record_time": str(entity.get("record_time") or default_date),
                    "properties": normalized_properties,
                }
            )

        normalized_edges: list[dict[str, Any]] = []
        valid_names = {entity["name"] for entity in normalized_entities}
        for index, edge in enumerate(edges):
            if not isinstance(edge, dict):
                raise ValueError(f"edge[{index}] must be an object")
            left = str(edge.get("link_entity1_name") or "").strip()
            right = str(edge.get("link_entity2_name") or "").strip()
            if left not in valid_names or right not in valid_names:
                raise ValueError(f"edge[{index}] references unknown entity")
            normalized_edges.append(dict(edge))

        propertyless_names = {entity["name"] for entity in normalized_entities if not entity["properties"]}
        if propertyless_names:
            has_any_property = len(propertyless_names) != len(normalized_entities)
            if not has_any_property and not allow_relation_only:
                raise ValueError(
                    "all extracted entities are propertyless; map every durable fact to a Schema property, "
                    "or confirm a relation-only result with explicit edges"
                )
            referenced_names = {
                name for edge in normalized_edges for name in (edge["link_entity1_name"], edge["link_entity2_name"])
            }
            if not normalized_edges or not propertyless_names.issubset(referenced_names):
                raise ValueError("every propertyless entity must be referenced by an explicit edge")
        result: dict[str, Any] = {"entities": normalized_entities, "edges": normalized_edges}
        if episode_candidates is not None:
            result["episode"] = _validate_episode_decision(value.get("episode"), episode_candidates)
        return result


def _schema_property_names(schema: list[dict[str, Any]]) -> dict[str, set[str]]:
    result: dict[str, set[str]] = {}
    for entity in schema:
        entity_type = str(entity.get("entity_type") or "").strip()
        if not entity_type:
            continue
        names: set[str] = set()
        for field in ("static_property", "dynamic_property"):
            values = entity.get(field, {})
            if isinstance(values, dict):
                names.update(str(name) for name in values)
        result[entity_type] = names
    return result


def _event_date(value: str) -> str:
    text = str(value or "").strip()
    if not text:
        return datetime.now(UTC).date().isoformat()
    try:
        return datetime.fromisoformat(text.replace("Z", "+00:00")).date().isoformat()
    except ValueError:
        return text.split(" ", 1)[0][:10]


def _normalize_property_value(value: Any) -> str:
    if isinstance(value, str):
        return " ".join(value.split())
    if isinstance(value, (dict, list)):
        return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
    return str(value).strip()


def _extraction_prompt(
    *,
    schema: list[dict[str, Any]],
    content: str,
    event_time: str,
    prompt_language: str | None,
    episode_candidates: list[StructuredEpisodeCandidate] | None,
) -> str:
    language = "Chinese when the source is Chinese; otherwise English" if not prompt_language else prompt_language
    episode_instruction = "Do not create episodes."
    if episode_candidates is not None:
        candidates = [candidate.prompt_value() for candidate in episode_candidates]
        episode_instruction = (
            "In the same response, decide whether to reuse one supplied Episode or create a new Episode. "
            "Return an episode object with action=reuse|create, target_episode_id, title, description, and "
            "related_episode_ids. Reuse only an episode_id from the supplied candidates. Use create when this is "
            "an independent process or no candidate exists. related_episode_ids may contain only supplied IDs. "
            f"Supplied Episode candidates: {json.dumps(candidates, ensure_ascii=False, sort_keys=True)}"
        )
    return (
        "You extract durable, objective structured memory from one already-selected candidate. "
        "Do not decide whether the candidate is worth remembering. Return JSON only. "
        "Each entity must have name, entity_type, description, and properties. Include a property whenever any durable "
        "fact maps to a Schema property. Use empty properties only when an entity exists solely as an endpoint of an "
        "explicit relationship stated in the source and no Schema property represents a durable fact about it. "
        "Every entity with empty properties must be referenced by an explicit edge. Each property must have "
        "property_name, value, and optional time. Return zero entities when there is no durable fact. "
        "For every explicit relationship stated in the source between returned entities, add an edge with "
        "link_entity1_name, link_entity2_name, and link_description. Do not infer relationships that the source does "
        "not establish. Use only the provided entity types and first-order properties. Do not create search fields or "
        "higher-order summaries.\n"
        f"Episode instruction: {episode_instruction}\n"
        f"Output language: {language}\n"
        f"Event time: {event_time}\n"
        f"Entity Schema: {json.dumps(schema, ensure_ascii=False, sort_keys=True)}\n"
        f"Candidate content:\n{content}"
    )


def _validate_episode_decision(
    value: Any,
    candidates: list[StructuredEpisodeCandidate],
) -> dict[str, Any]:
    if not isinstance(value, dict):
        raise ValueError("episode must be an object")
    action = str(value.get("action") or "").strip()
    if action not in {"reuse", "create"}:
        raise ValueError("episode action must be reuse or create")
    candidate_by_id = {candidate.episode_id: candidate for candidate in candidates}
    target = str(value.get("target_episode_id") or "").strip() or None
    if action == "reuse":
        if target not in candidate_by_id:
            raise ValueError("target_episode_id is not one of the supplied candidates")
        selected = candidate_by_id[target]
        title = str(value.get("title") or selected.title).strip() or selected.title
        description = str(value.get("description") or selected.description).strip() or selected.description
    else:
        if target is not None:
            raise ValueError("create episode must not have target_episode_id")
        title = str(value.get("title") or "").strip()
        description = str(value.get("description") or "").strip()
        if not title or not description:
            raise ValueError("create episode requires title and description")

    related_value = value.get("related_episode_ids", [])
    if not isinstance(related_value, list):
        raise ValueError("related_episode_ids must be an array")
    related: list[str] = []
    for raw_id in related_value:
        episode_id = str(raw_id or "").strip()
        if episode_id not in candidate_by_id:
            raise ValueError("related_episode_ids contains an episode outside supplied candidates")
        if episode_id != target and episode_id not in related:
            related.append(episode_id)
    return {
        "action": action,
        "target_episode_id": target,
        "title": title,
        "description": description,
        "related_episode_ids": related,
    }
