"""Separated fixed-Schema selection and complete extraction for Structured Add."""

from __future__ import annotations

import ast
import copy
import json
from datetime import UTC, datetime
from typing import Any

from ....config import StructuredExtractionConfig
from ....errors import ApiError
from ....llm import LLMClient
from ....logging import get_logger
from ....structured_content import normalize_structured_artifact, normalize_structured_content
from ..schema._schema_utils import parse_json_object, strip_for_generation
from .episode import StructuredEpisodeCandidate
from .evidence import inventory_source_artifacts

logger = get_logger(__name__)

_EVIDENCE_FORM_INVENTORY = (
    "First inventory only the material evidence forms actually present in Candidate content: ordered procedures or "
    "decisions; quoted source excerpts; code or pseudocode; formulas, parameters, and ranges; performance metrics, "
    "scores, and measurements; experimental or alternative comparisons; errors and observed consequences; "
    "constraints and exceptions; worked examples; and ordinary factual claims. Do not require or invent an evidence "
    "form that is absent from Candidate content. "
)


def _merge_source_artifacts(values: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Normalize exact artifacts while keeping duplicate source bytes only once."""

    result: list[dict[str, Any]] = []
    seen_ids: set[str] = set()
    seen_hashes: set[str] = set()
    for value in values:
        artifact = normalize_structured_artifact(value)
        artifact_id = artifact["artifact_id"]
        if artifact_id in seen_ids:
            raise ValueError(f"duplicate structured source artifact ID: {artifact_id}")
        seen_ids.add(artifact_id)
        if artifact["source_hash"] in seen_hashes:
            continue
        seen_hashes.add(artifact["source_hash"])
        result.append(artifact)
    return result


def _source_artifact_prompt_inventory(values: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Describe attached evidence to the model without copying large immutable bodies."""

    return [
        {
            key: value
            for key, value in {
                "artifact_id": artifact.get("artifact_id"),
                "type": artifact.get("type"),
                "language": artifact.get("language"),
                "source_hash": artifact.get("source_hash"),
                "content_length": len(str(artifact.get("content") or "")),
            }.items()
            if value not in {None, ""}
        }
        for artifact in values
    ]

_EVIDENCE_DETAIL_RULES = (
    "Preserve material detail conditionally for each present form: keep ordering and conditions for procedures; exact "
    "operations, control flow, state changes, termination, and outputs for implementations; symbols, values, units, "
    "ranges, and assumptions for formulas or measurements; subjects, baselines, and outcomes for comparisons; "
    "triggers, errors, and consequences for failures; and setup, action, and result for worked examples. A label, "
    "name, section number, or broad summary does not cover source details that are available. "
)


class StructuredExtractionError(ApiError):
    """Raised when fixed-Schema extraction cannot be validated."""

    status_code = 422
    code = "structured.extraction_invalid"


class StructuredExtractor:
    """Select fixed-Schema slots, then extract complete source-grounded facts."""

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

        schema = strip_for_generation(copy.deepcopy(self._entity_manager.get_all_dicts()))
        for entity in schema:
            dynamic = entity.get("dynamic_property", {})
            if isinstance(dynamic, dict):
                entity["dynamic_property"] = {
                    name: definition
                    for name, definition in dynamic.items()
                    if not isinstance(definition, dict) or definition.get("order", 1) < 2
                }
        return schema

    async def extract(
        self,
        *,
        content: str,
        event_time: str,
        prompt_language: str | None,
        episode_candidates: list[StructuredEpisodeCandidate] | None = None,
        allowed_property_names: set[str] | None = None,
        source_artifacts: list[dict[str, Any]] | None = None,
    ) -> dict[str, Any]:
        """Return a normalized structured result or fail before persistence."""

        source_artifacts = _merge_source_artifacts(
            [*(source_artifacts or []), *inventory_source_artifacts(content)]
        )
        schema = self.schema_context
        selection_is_final = False
        extract_relationships = True
        if allowed_property_names is not None:
            schema = _schema_with_allowed_properties(schema, allowed_property_names)
            selection_is_final = True
        elif self._config.selection_enabled:
            schema, extract_relationships = await self._select_schema(
                schema=schema,
                content=content,
                prompt_language=prompt_language,
            )
            selection_is_final = True
            if not schema:
                return {"entities": [], "edges": []}
        prompt = _extraction_prompt(
            schema=schema,
            content=content,
            event_time=event_time,
            prompt_language=prompt_language,
            episode_candidates=episode_candidates,
            selection_is_final=selection_is_final,
            extract_relationships=extract_relationships,
            source_artifacts=source_artifacts,
        )
        last_error = "unknown validation error"
        last_content = ""
        normalized: dict[str, Any] | None = None
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
                    source_artifacts=source_artifacts,
                )
            except (TypeError, ValueError, json.JSONDecodeError) as exc:
                last_error = str(exc)
                logger.info(
                    "structured_extraction_validation_failed",
                    attempt=attempt + 1,
                    max_attempts=total_calls,
                    error_type=type(exc).__name__,
                    validation_error=str(exc),
                )
                continue
            logger.info(
                "structured_extraction_completed",
                attempts=attempt + 1,
                entity_count=len(normalized["entities"]),
                property_count=sum(len(entity["properties"]) for entity in normalized["entities"]),
            )
            break
        if normalized is None:
            raise StructuredExtractionError(
                f"Structured extraction failed Schema validation: {last_error}",
                details={"attempts": total_calls},
            )
        if not self._config.coverage_validation_enabled or not normalized["entities"]:
            return normalized
        missing_evidence = await self._audit_coverage(
            schema=schema,
            content=content,
            extracted=normalized,
            prompt_language=prompt_language,
            extract_relationships=extract_relationships,
        )
        if not missing_evidence:
            return normalized
        return await self._complete_extraction(
            prompt=prompt,
            schema=schema,
            event_time=event_time,
            episode_candidates=episode_candidates,
            original=normalized,
            missing_evidence=missing_evidence,
            source_artifacts=source_artifacts,
        )

    def validate_prestructured(
        self,
        entities: list[dict[str, Any]],
        *,
        event_time: str,
    ) -> dict[str, Any]:
        """Validate caller-supplied structured entities without invoking an LLM."""

        return self._validate_and_normalize(
            {"entities": entities, "edges": []},
            event_time=event_time,
            schema=self.schema_context,
            episode_candidates=None,
            allow_relation_only=False,
            source_artifacts=None,
        )

    async def _select_schema(
        self,
        *,
        schema: list[dict[str, Any]],
        content: str,
        prompt_language: str | None,
    ) -> tuple[list[dict[str, Any]], bool]:
        response = await self._llm.chat(
            task="memory.add.structured_select",
            messages=[
                {
                    "role": "user",
                    "content": _selection_prompt(
                        schema=schema,
                        content=content,
                        prompt_language=prompt_language,
                    ),
                }
            ],
        )
        try:
            value = parse_json_object(response.content)
            selections, extract_relationships = _validate_schema_selection(value, schema)
            return _schema_with_selected_slots(schema, selections), extract_relationships
        except (TypeError, ValueError, json.JSONDecodeError) as exc:
            raise StructuredExtractionError(
                f"Structured Schema selection failed validation: {exc}",
                details={"phase": "selection"},
            ) from exc

    async def _audit_coverage(
        self,
        *,
        schema: list[dict[str, Any]],
        content: str,
        extracted: dict[str, Any],
        prompt_language: str | None,
        extract_relationships: bool,
    ) -> list[dict[str, str]]:
        try:
            response = await self._llm.chat(
                task="memory.add.structured_coverage",
                messages=[
                    {
                        "role": "user",
                        "content": _coverage_prompt(
                            schema=schema,
                            content=content,
                            extracted=extracted,
                            prompt_language=prompt_language,
                            max_items=self._config.max_coverage_items,
                            extract_relationships=extract_relationships,
                        ),
                    }
                ],
            )
            value = parse_json_object(response.content)
            return _validate_coverage_result(
                value,
                source=content,
                max_items=self._config.max_coverage_items,
            )
        except (TypeError, ValueError, json.JSONDecodeError):
            logger.warning("structured_extraction_coverage_invalid", exc_info=True)
            return []

    async def _complete_extraction(
        self,
        *,
        prompt: str,
        schema: list[dict[str, Any]],
        event_time: str,
        episode_candidates: list[StructuredEpisodeCandidate] | None,
        original: dict[str, Any],
        missing_evidence: list[dict[str, str]],
        source_artifacts: list[dict[str, str]],
    ) -> dict[str, Any]:
        completion_prompt = (
            prompt
            + "\n\nPrevious valid extraction:\n"
            + json.dumps(_dematerialize_artifact_refs(original), ensure_ascii=False, sort_keys=True)
            + "\n\nCoverage audit found these grounded source spans that are not represented:\n"
            + json.dumps(missing_evidence, ensure_ascii=False, sort_keys=True)
            + "\nReturn one complete corrected extraction JSON object. Preserve every already extracted grounded fact "
            "and explicit relationship, add the missing facts without summarizing or rewriting them away, and do not "
            "add anything unsupported by Candidate content."
        )
        try:
            response = await self._llm.chat(
                task="memory.add.structured_complete",
                messages=[{"role": "user", "content": completion_prompt}],
            )
            completed = self._validate_and_normalize(
                parse_json_object(response.content),
                event_time=event_time,
                schema=schema,
                episode_candidates=episode_candidates,
                allow_relation_only=True,
                source_artifacts=source_artifacts,
            )
            if not _preserves_previous_extraction(original, completed):
                raise ValueError("completion removed a previously extracted fact or relationship")
            return completed
        except (TypeError, ValueError, json.JSONDecodeError):
            logger.warning("structured_extraction_completion_invalid", exc_info=True)
            return original

    def _validate_and_normalize(
        self,
        value: Any,
        *,
        event_time: str,
        schema: list[dict[str, Any]],
        episode_candidates: list[StructuredEpisodeCandidate] | None,
        allow_relation_only: bool,
        source_artifacts: list[dict[str, str]] | None = None,
    ) -> dict[str, Any]:
        if not isinstance(value, dict):
            raise ValueError("response must be a JSON object")
        entities = value.get("entities", [])
        edges = value.get("edges", [])
        if not isinstance(entities, list) or not isinstance(edges, list):
            raise ValueError("entities and edges must be arrays")
        if len(entities) > self._config.max_entities:
            raise ValueError(f"entity count exceeds max_entities={self._config.max_entities}")

        property_definitions = _schema_property_definitions(schema)
        property_names = {entity_type: set(definitions) for entity_type, definitions in property_definitions.items()}
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
                expanded_values = _expand_schema_property_value(
                    property_name,
                    prop["value"],
                    property_definitions[entity_type],
                    source_artifacts=source_artifacts or [],
                )
                for expanded_name, expanded_value in expanded_values:
                    if not expanded_value:
                        raise ValueError(f"property {expanded_name!r} requires a non-empty value")
                    normalized_properties.append(
                        {
                            "property_name": expanded_name,
                            "value": expanded_value,
                            "time": str(prop.get("time") or default_date),
                            "operation": "set",
                        }
                    )
            if len(normalized_properties) > self._config.max_properties_per_entity:
                raise ValueError(
                    f"property count exceeds max_properties_per_entity={self._config.max_properties_per_entity}"
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

        if source_artifacts:
            card_values = [
                prop["value"]
                for entity in normalized_entities
                for prop in entity["properties"]
                if isinstance(prop.get("value"), dict) and "artifacts" in prop["value"]
            ]
            if card_values:
                covered = {artifact["artifact_id"] for card in card_values for artifact in card.get("artifacts", [])}
                missing = [
                    artifact["artifact_id"] for artifact in source_artifacts if artifact["artifact_id"] not in covered
                ]
                if missing:
                    raise ValueError(
                        "structured card omitted inventoried source artifact reference(s): " + ", ".join(missing)
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
    return {entity_type: set(definitions) for entity_type, definitions in _schema_property_definitions(schema).items()}


def _schema_with_allowed_properties(
    schema: list[dict[str, Any]],
    allowed_property_names: set[str],
) -> list[dict[str, Any]]:
    """Restrict extraction to caller-selected properties that exist in the fixed Schema."""

    normalized_allowed = {str(name).strip() for name in allowed_property_names if str(name).strip()}
    available = {property_name for names in _schema_property_names(schema).values() for property_name in names}
    unknown = normalized_allowed - available
    if unknown:
        raise StructuredExtractionError(
            "Structured extraction property constraint is not present in the project Schema.",
            details={"unknown_property_names": sorted(unknown)},
        )

    filtered: list[dict[str, Any]] = []
    for entity in schema:
        item = copy.deepcopy(entity)
        for field in ("static_property", "dynamic_property"):
            definitions = item.get(field)
            if isinstance(definitions, dict):
                item[field] = {
                    name: definition for name, definition in definitions.items() if str(name) in normalized_allowed
                }
        if any(isinstance(item.get(field), dict) and item[field] for field in ("static_property", "dynamic_property")):
            filtered.append(item)
    if not filtered:
        raise StructuredExtractionError(
            "Structured extraction property constraint resolved to an empty Schema.",
            details={"allowed_property_names": sorted(normalized_allowed)},
        )
    return filtered


def _validate_schema_selection(
    value: Any,
    schema: list[dict[str, Any]],
) -> tuple[dict[str, set[str]], bool]:
    if not isinstance(value, dict) or set(value) - {"selections", "extract_relationships"}:
        raise ValueError("selection response must contain only selections and extract_relationships")
    raw_selections = value.get("selections", [])
    extract_relationships = value.get("extract_relationships", False)
    if not isinstance(raw_selections, list) or not isinstance(extract_relationships, bool):
        raise ValueError("selections must be an array and extract_relationships must be boolean")
    available = _schema_property_names(schema)
    selections: dict[str, set[str]] = {}
    for index, item in enumerate(raw_selections):
        if not isinstance(item, dict) or set(item) != {"entity_type", "property_names"}:
            raise ValueError(f"selections[{index}] must contain entity_type and property_names")
        entity_type = str(item.get("entity_type") or "").strip()
        property_names = item.get("property_names")
        if entity_type not in available or not isinstance(property_names, list):
            raise ValueError(f"selections[{index}] references an unknown entity type or invalid property_names")
        normalized = {str(name).strip() for name in property_names if str(name).strip()}
        unknown = normalized - available[entity_type]
        if unknown:
            raise ValueError(f"selections[{index}] contains unknown properties: {sorted(unknown)}")
        selections.setdefault(entity_type, set()).update(normalized)
    if extract_relationships and not selections:
        raise ValueError("relationship extraction requires at least one selected entity type")
    return selections, extract_relationships


def _schema_with_selected_slots(
    schema: list[dict[str, Any]],
    selections: dict[str, set[str]],
) -> list[dict[str, Any]]:
    filtered: list[dict[str, Any]] = []
    for entity in schema:
        entity_type = str(entity.get("entity_type") or "")
        if entity_type not in selections:
            continue
        item = copy.deepcopy(entity)
        allowed = selections[entity_type]
        for field in ("static_property", "dynamic_property"):
            definitions = item.get(field)
            if isinstance(definitions, dict):
                item[field] = {name: definition for name, definition in definitions.items() if str(name) in allowed}
        filtered.append(item)
    return filtered


def _schema_shape_contract(schema: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Keep only output keys and value shapes after eligibility is settled."""

    result: list[dict[str, Any]] = []
    for entity in schema:
        properties: dict[str, dict[str, str]] = {}
        for name, definition in (
            _schema_property_definitions([entity]).get(str(entity.get("entity_type") or ""), {}).items()
        ):
            shape: dict[str, str] = {}
            if isinstance(definition, dict):
                for key in ("type", "format"):
                    value = str(definition.get(key) or "").strip()
                    if value:
                        shape[key] = value
            properties[name] = shape
        result.append({"entity_type": entity.get("entity_type"), "properties": properties})
    return result


def _schema_output_contract(
    schema: list[dict[str, Any]],
    *,
    extract_relationships: bool,
    source_artifacts: list[dict[str, str]] | None = None,
) -> dict[str, Any]:
    """Build one valid response example without changing ``properties`` into a map."""

    entities: list[dict[str, Any]] = []
    for entity in schema:
        entity_type = str(entity.get("entity_type") or "").strip()
        properties: list[dict[str, Any]] = []
        for name, definition in _schema_property_definitions([entity]).get(entity_type, {}).items():
            expected_type = str(definition.get("type") or "").strip().casefold() if isinstance(definition, dict) else ""
            value_format = (
                str(definition.get("format") or "").strip().casefold() if isinstance(definition, dict) else ""
            )
            fact_contract = (
                str(definition.get("fact_contract") or "").strip().casefold() if isinstance(definition, dict) else ""
            )
            if expected_type in {"object", "dict", "map"}:
                value: Any = (
                    {
                        "description": "source-grounded context",
                        "content": (
                            [
                                {
                                    "subject": "explicit named subject",
                                    "statement": "complete source-grounded claim about that subject",
                                }
                            ]
                            if fact_contract == "explicit_subject"
                            else ["complete atomic fact"]
                        ),
                        "artifact_refs": [source_artifacts[0]["artifact_id"]]
                        if value_format == "structured_card_content" and source_artifacts
                        else [],
                    }
                    if value_format == "structured_card_content"
                    else {"source_field": "source-grounded value"}
                )
            elif expected_type in {"array", "list"}:
                value = ["source-grounded value"]
            elif expected_type in {"integer", "int", "number", "float"}:
                value = 0
            elif expected_type in {"boolean", "bool"}:
                value = False
            else:
                value = "source-grounded text"
            properties.append({"property_name": name, "value": value})
        entities.append(
            {
                "name": "source-grounded entity name",
                "entity_type": entity_type,
                "description": "source-grounded description",
                "properties": properties,
            }
        )
    edges: list[dict[str, str]] = []
    if extract_relationships:
        edges.append(
            {
                "link_entity1_name": "exact returned entity name",
                "link_entity2_name": "exact returned entity name",
                "link_description": "explicit source-grounded relationship",
            }
        )
    return {"entities": entities, "edges": edges}


def _schema_value_requirements(schema: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Describe value types separately from the valid response shape."""

    return [
        {
            "entity_type": item["entity_type"],
            "property_value_requirements": item["properties"],
        }
        for item in _schema_shape_contract(schema)
    ]


def _schema_property_definitions(schema: list[dict[str, Any]]) -> dict[str, dict[str, Any]]:
    result: dict[str, dict[str, Any]] = {}
    for entity in schema:
        entity_type = str(entity.get("entity_type") or "").strip()
        if not entity_type:
            continue
        definitions: dict[str, Any] = {}
        for field in ("static_property", "dynamic_property"):
            values = entity.get(field, {})
            if isinstance(values, dict):
                definitions.update({str(name): definition for name, definition in values.items()})
        result[entity_type] = definitions
    return result


def _uses_explicit_subject_fact_contract(schema: list[dict[str, Any]]) -> bool:
    return any(
        isinstance(definition, dict)
        and str(definition.get("format") or "").strip().casefold() == "structured_card_content"
        and str(definition.get("fact_contract") or "").strip().casefold() == "explicit_subject"
        for definitions in _schema_property_definitions(schema).values()
        for definition in definitions.values()
    )


def _event_date(value: str) -> str:
    text = str(value or "").strip()
    if not text:
        return datetime.now(UTC).date().isoformat()
    try:
        return datetime.fromisoformat(text.replace("Z", "+00:00")).date().isoformat()
    except ValueError:
        return text.split(" ", 1)[0][:10]


def _expand_schema_property_value(
    property_name: str,
    value: Any,
    definitions: dict[str, Any],
    *,
    source_artifacts: list[dict[str, str]],
) -> list[tuple[str, Any]]:
    """Normalize one value and unwrap accidental first-order Schema envelopes."""

    parsed = _parse_property_mapping(value)
    wrapper: dict[str, Any] | None = None
    if parsed is not None:
        for wrapper_name in ("dynamic_property", "static_property"):
            nested = parsed.get(wrapper_name)
            if isinstance(nested, dict):
                wrapper = nested
                break
        if wrapper is None and any(name in definitions for name in parsed):
            wrapper = parsed
    if wrapper is not None:
        unknown = [str(name) for name in wrapper if str(name) not in definitions]
        if unknown:
            raise ValueError(f"unknown wrapped property_name(s): {', '.join(sorted(unknown))}")
        expanded: list[tuple[str, Any]] = []
        for name, nested_value in wrapper.items():
            normalized_name = str(name)
            if nested_value is None:
                continue
            normalized_value = _normalize_property_value(
                nested_value,
                definition=definitions.get(normalized_name),
                property_name=normalized_name,
                source_artifacts=source_artifacts,
            )
            if normalized_value:
                expanded.append((normalized_name, normalized_value))
        if not expanded:
            raise ValueError(f"property wrapper for {property_name!r} contains no non-empty Schema value")
        return expanded
    return [
        (
            property_name,
            _normalize_property_value(
                value,
                definition=definitions.get(property_name),
                property_name=property_name,
                source_artifacts=source_artifacts,
            ),
        )
    ]


def _parse_property_mapping(value: Any) -> dict[str, Any] | None:
    if isinstance(value, dict):
        return value
    if not isinstance(value, str):
        return None
    text = value.strip()
    if not (text.startswith("{") and text.endswith("}")):
        return None
    try:
        parsed = json.loads(text)
    except (TypeError, ValueError, json.JSONDecodeError):
        try:
            parsed = ast.literal_eval(text)
        except (ValueError, SyntaxError):
            return None
    return parsed if isinstance(parsed, dict) else None


def _normalize_property_value(
    value: Any,
    *,
    definition: Any = None,
    property_name: str = "",
    source_artifacts: list[dict[str, str]],
) -> Any:
    expected_type = str(definition.get("type") or "").strip().casefold() if isinstance(definition, dict) else ""
    value_format = str(definition.get("format") or "").strip().casefold() if isinstance(definition, dict) else ""
    fact_contract = (
        str(definition.get("fact_contract") or "").strip().casefold() if isinstance(definition, dict) else ""
    )
    if expected_type in {"string", "str", "text"} and isinstance(value, (dict, list, tuple, set)):
        raise ValueError(f"property {property_name!r} requires a scalar string value")
    if isinstance(value, str):
        if expected_type in {"string", "str", "text"} and _parse_property_mapping(value) is not None:
            raise ValueError(f"property {property_name!r} requires a scalar string value")
        return " ".join(value.split())
    if expected_type in {"object", "dict", "map"}:
        if not isinstance(value, dict):
            raise ValueError(f"property {property_name!r} requires an object value")
        if value_format == "structured_card_content":
            if source_artifacts and "artifact_refs" not in value:
                raise ValueError(f"property {property_name!r} must include artifact_refs covering its source artifacts")
            if "artifact_refs" in value:
                if set(value) - {"description", "content", "artifact_refs"}:
                    raise ValueError(
                        f"property {property_name!r} structured card may contain only description, content, and artifact_refs"
                    )
                refs = value.get("artifact_refs")
                if not isinstance(refs, list) or any(not isinstance(item, str) or not item for item in refs):
                    raise ValueError(f"property {property_name!r} artifact_refs must be an array of artifact IDs")
                if len(refs) != len(set(refs)):
                    raise ValueError(f"property {property_name!r} artifact_refs must be unique")
                by_id = {artifact["artifact_id"]: artifact for artifact in source_artifacts}
                unknown = [item for item in refs if item not in by_id]
                if unknown:
                    raise ValueError(
                        f"property {property_name!r} references unknown source artifact(s): {', '.join(unknown)}"
                    )
                value = {
                    "description": value.get("description"),
                    "content": value.get("content"),
                    "artifacts": [copy.deepcopy(by_id[item]) for item in refs],
                }
            elif source_artifacts and "artifacts" in value:
                raise ValueError(
                    f"property {property_name!r} must reference source artifacts by artifact_refs instead of rewriting them"
                )
            if fact_contract == "explicit_subject":
                value = _materialize_explicit_subject_facts(value, property_name=property_name)
            normalized = normalize_structured_content(value)
            return normalized
        return value
    if isinstance(value, (dict, list)):
        return value
    return str(value).strip()


def _materialize_explicit_subject_facts(value: dict[str, Any], *, property_name: str) -> dict[str, Any]:
    """Render language-neutral subject/statement facts into the stored string contract."""

    content = value.get("content")
    if not isinstance(content, list):
        return value
    rendered: list[str] = []
    for index, fact in enumerate(content):
        if not isinstance(fact, dict) or set(fact) != {"subject", "statement"}:
            raise ValueError(f"property {property_name!r} fact {index + 1} must contain exactly subject and statement")
        subject = " ".join(str(fact.get("subject") or "").split())
        statement = " ".join(str(fact.get("statement") or "").split())
        if not subject or not statement:
            raise ValueError(f"property {property_name!r} fact {index + 1} requires a non-empty subject and statement")
        rendered.append(statement if subject.casefold() in statement.casefold() else f"{subject}: {statement}")
    result = copy.deepcopy(value)
    result["content"] = rendered
    return result


def _extraction_prompt(
    *,
    schema: list[dict[str, Any]],
    content: str,
    event_time: str,
    prompt_language: str | None,
    episode_candidates: list[StructuredEpisodeCandidate] | None,
    selection_is_final: bool,
    extract_relationships: bool,
    source_artifacts: list[dict[str, str]],
) -> str:
    language = "Chinese when the source is Chinese; otherwise English" if not prompt_language else prompt_language
    allowed_property_names = {
        entity_type: sorted(names) for entity_type, names in sorted(_schema_property_names(schema).items())
    }
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
    selection_instruction = ""
    prompt_schema: Any = schema
    if selection_is_final:
        prompt_schema = _schema_output_contract(
            schema,
            extract_relationships=extract_relationships,
            source_artifacts=source_artifacts,
        )
        selection_instruction = (
            "Schema eligibility and destination properties have already been selected. Do not re-evaluate whether "
            "the content is useful, reusable, successful, important, or worth remembering. "
        )
    relationship_instruction = (
        "For every explicit relationship stated in the source between returned entities, add an edge with "
        "link_entity1_name, link_entity2_name, and link_description. Do not infer relationships that the source does "
        "not establish. "
        if extract_relationships
        else "Relationship extraction was not selected; return edges as an empty array. "
    )
    property_mapping_instruction = (
        "Use only the already-selected entity types and property keys. Eligibility is final; do not drop a grounded "
        "fact because it would not independently satisfy the original retention or success policy. "
        if selection_is_final
        else "Use only the provided entity types and first-order properties. Select a property only when the source "
        "satisfies its Schema definition; a generic fact is not experimental success, failure, or experience without "
        "explicit source evidence. "
    )
    if _uses_explicit_subject_fact_contract(schema):
        structured_fact_instruction = (
            "A structured_card_content object must contain description as concise presentation context and content as "
            "a non-empty array of fact objects. Every fact object must contain exactly subject and statement. Subject "
            "must be the explicit source-grounded name of the person, system, component, method, policy, event, "
            "measurement, or other concrete subject, written in the requested output language. Statement must be a "
            "complete claim about that subject and retain every condition needed to keep it true. This "
            "subject/statement contract applies equally to every source and output language; never use a pronoun, "
            "positional expression, translated local label, or document-local number as the subject. Backend code "
            "deterministically renders subject and statement into stored fact strings. "
        )
    else:
        structured_fact_instruction = (
            "A structured_card_content object must contain description as concise presentation context and content as "
            "a non-empty array of source-grounded fact strings. Put one independently understandable claim in each "
            "content item. "
        )
    return (
        selection_instruction
        + _EVIDENCE_FORM_INVENTORY
        + "Extract only source-grounded entities, atomic facts, and explicit relationships from one already-selected "
        "document block. Do not decide whether the block is worth remembering, which Episode it belongs to, whether "
        "it matches history, or how memories should be merged. Return JSON only. One block may produce zero, one, or "
        "many entities and properties. Preserve every material quantity, time, condition, exception, causal relation, "
        "named subject, and certainty supported by the source. Do not generalize, invent context, or replace several "
        "specific claims with a broader summary. "
        "Each entity must have name, entity_type, description, and properties. Include a property whenever any durable "
        "fact maps to a Schema property. Use empty properties only when an entity exists solely as an endpoint of an "
        "explicit relationship stated in the source and no Schema property represents a durable fact about it. "
        "Every entity with empty properties must be referenced by an explicit edge. Each property must have "
        "property_name, value, and optional time. Return zero entities when there is no durable fact. "
        + _EVIDENCE_DETAIL_RULES
        + "Return each first-order Schema property as its own properties[] item. Never place static_property, "
        "dynamic_property, or a map of several Schema keys inside one value. Match the declared property type. "
        + structured_fact_instruction
        + "The description must not introduce facts absent from content. "
        "Resolve every local cross-reference into the explicit named subject and standalone behavior, relationship, "
        "constraint, decision, measurement, or outcome it denotes; local labels may remain only inside immutable "
        "artifact bodies. Facts must capture durable semantics instead of "
        "transcribing any code, formula, table, quote, example, metric, or other artifact element by element. "
        "When Source artifact inventory is non-empty, the object must also contain "
        "artifact_refs with exact artifact_id values. Use artifact_refs for every inventoried code, formula, table, "
        "example, quote, or metric that supports the card. Never copy, rewrite, summarize, repair, or translate an "
        "artifact body into the JSON response; backend code copies the referenced source bytes exactly. Facts may "
        "explain an artifact but must not replace it. A string property must be plain text rather "
        "than an object or serialized object. "
        + relationship_instruction
        + property_mapping_instruction
        + "Do not create search fields or "
        "higher-order summaries. Property values must remain independently understandable and faithful to the source; "
        "a generated entity name is display text only and must not be treated as identity. Every property_name must be "
        "copied exactly from this key list; never translate a key or replace it with a semantic label: "
        f"{json.dumps(allowed_property_names, ensure_ascii=False, sort_keys=True)}.\n"
        f"Episode instruction: {episode_instruction}\n"
        f"Output language: {language}\n"
        f"Event time: {event_time}\n"
        f"Valid JSON output shape (omit absent entities and properties): "
        f"{json.dumps(prompt_schema, ensure_ascii=False)}\n"
        f"Selected property value requirements: "
        f"{json.dumps(_schema_value_requirements(schema), ensure_ascii=False, sort_keys=True)}\n"
        "Source artifact inventory (immutable bodies are attached separately and must be referenced by ID): "
        f"{json.dumps(_source_artifact_prompt_inventory(source_artifacts), ensure_ascii=False, sort_keys=True)}\n"
        f"Candidate content:\n{content}"
    )


def _selection_prompt(
    *,
    schema: list[dict[str, Any]],
    content: str,
    prompt_language: str | None,
) -> str:
    language = "Chinese when the source is Chinese; otherwise English" if not prompt_language else prompt_language
    return (
        "Decide only which fixed-Schema destinations are eligible for the supplied source block. Return JSON "
        "{selections:[{entity_type,property_names:[...]}],extract_relationships:boolean}. Use only exact entity_type "
        "and property names from the supplied Schema. Include a property when at least one source-grounded fact belongs "
        "there under the Schema policy. Return an empty selections array when nothing qualifies. Set "
        "extract_relationships=true only when the source explicitly relates entities of selected types. Do not extract, "
        "quote, summarize, title, or rewrite source facts. Do not decide Episodes, identity, history, merging, or "
        "storage. "
        + _EVIDENCE_FORM_INVENTORY
        + "Use the evidence forms present only as support for the Schema policy, and select every matching destination "
        "needed to retain their material content.\n"
        f"Output language: {language}\n"
        f"Entity Schema: {json.dumps(schema, ensure_ascii=False, sort_keys=True)}\n"
        f"Candidate content:\n{content}"
    )


def _coverage_prompt(
    *,
    schema: list[dict[str, Any]],
    content: str,
    extracted: dict[str, Any],
    prompt_language: str | None,
    max_items: int,
    extract_relationships: bool,
) -> str:
    language = "Chinese when the source is Chinese; otherwise English" if not prompt_language else prompt_language
    return (
        "Audit source-fact coverage only. Eligibility and Schema destinations are already settled. Compare Candidate "
        "content with Extracted result and find material source claims that fit the selected properties but are absent. "
        + _EVIDENCE_FORM_INVENTORY
        + _EVIDENCE_DETAIL_RULES
        + "Check quantities, formulas, parameter ranges, times, conditions, exceptions, negations, comparisons, causal "
        "claims"
        + (", and explicit relationships" if extract_relationships else "")
        + ". Do not request shorter wording, better titles, tags, inferred advice, or "
        "facts outside the selected slots. Return JSON {complete:boolean,missing_evidence:[{quote,reason}]}. Every quote "
        "must be a verbatim contiguous span from Candidate content and each item must identify a distinct omitted claim. "
        "Set complete=true exactly when missing_evidence is empty, and complete=false exactly when it is non-empty. "
        f"Return at most {max_items} items. Do not rewrite the extraction.\n"
        f"Output language: {language}\n"
        f"Selected shape: {json.dumps(_schema_shape_contract(schema), ensure_ascii=False, sort_keys=True)}\n"
        f"Candidate content: {content}\n"
        f"Extracted result: {json.dumps(extracted, ensure_ascii=False, sort_keys=True)}"
    )


def _validate_coverage_result(value: Any, *, source: str, max_items: int) -> list[dict[str, str]]:
    if not isinstance(value, dict) or set(value) != {"complete", "missing_evidence"}:
        raise ValueError("coverage response must contain complete and missing_evidence")
    complete = value.get("complete")
    evidence = value.get("missing_evidence")
    if not isinstance(complete, bool) or not isinstance(evidence, list):
        raise ValueError("coverage complete must be boolean and missing_evidence must be an array")
    if len(evidence) > max_items:
        raise ValueError("coverage evidence exceeds configured bound")
    if not evidence:
        return []
    if complete:
        raise ValueError("complete coverage cannot contain missing evidence")
    normalized_source = " ".join(source.split())
    result: list[dict[str, str]] = []
    for index, item in enumerate(evidence):
        if not isinstance(item, dict) or set(item) != {"quote", "reason"}:
            raise ValueError(f"missing_evidence[{index}] must contain quote and reason")
        quote = str(item.get("quote") or "").strip()
        reason = str(item.get("reason") or "").strip()
        if not quote or not reason or " ".join(quote.split()) not in normalized_source:
            raise ValueError(f"missing_evidence[{index}] is not grounded in Candidate content")
        result.append({"quote": quote, "reason": reason})
    return result


def _dematerialize_artifact_refs(value: dict[str, Any]) -> dict[str, Any]:
    """Return the model-facing form without exposing mutable artifact bodies."""

    result = copy.deepcopy(value)
    for entity in result.get("entities", []):
        for prop in entity.get("properties", []):
            card = prop.get("value")
            if not isinstance(card, dict) or "artifacts" not in card:
                continue
            card["artifact_refs"] = [
                artifact["artifact_id"]
                for artifact in card.pop("artifacts", [])
                if isinstance(artifact, dict) and artifact.get("artifact_id")
            ]
    return result


def _preserves_previous_extraction(original: dict[str, Any], completed: dict[str, Any]) -> bool:
    def facts(value: dict[str, Any]) -> set[tuple[str, str, str]]:
        result: set[tuple[str, str, str]] = set()
        for entity in value.get("entities", []):
            entity_type = str(entity.get("entity_type") or "")
            for prop in entity.get("properties", []):
                property_name = str(prop.get("property_name") or "")
                prop_value = prop.get("value")
                if isinstance(prop_value, dict) and isinstance(prop_value.get("content"), list):
                    values = prop_value["content"]
                else:
                    values = [prop_value]
                result.update(
                    (entity_type, property_name, json.dumps(item, ensure_ascii=False, sort_keys=True))
                    for item in values
                )
                if isinstance(prop_value, dict):
                    result.update(
                        (entity_type, property_name, f"artifact:{artifact.get('source_hash', '')}")
                        for artifact in prop_value.get("artifacts", [])
                        if isinstance(artifact, dict) and artifact.get("source_hash")
                    )
        return result

    def edges(value: dict[str, Any]) -> set[tuple[str, str, str]]:
        return {
            (
                str(edge.get("link_entity1_name") or ""),
                str(edge.get("link_entity2_name") or ""),
                str(edge.get("link_description") or ""),
            )
            for edge in value.get("edges", [])
        }

    return facts(original).issubset(facts(completed)) and edges(original).issubset(edges(completed))


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
