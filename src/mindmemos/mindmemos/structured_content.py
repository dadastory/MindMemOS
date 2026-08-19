"""Typed card content used by the LLM4AD structured-memory pipeline."""

from __future__ import annotations

import hashlib
import json
from typing import Any

STRUCTURED_CONTENT_METADATA_KEY = "structured_content"
STRUCTURED_ARTIFACT_TYPES = frozenset({"code", "formula", "table", "example", "quote", "metric"})


def _artifact_hash(*, artifact_type: str, language: str, content: str) -> str:
    value = "\x1f".join((artifact_type, language, content)).encode("utf-8")
    return hashlib.sha256(value).hexdigest()


def normalize_structured_artifact(value: Any) -> dict[str, Any]:
    """Validate one immutable source artifact without rewriting its body."""

    if not isinstance(value, dict):
        raise ValueError("structured card artifact must be an object")
    allowed = {"artifact_id", "type", "language", "content", "source_hash", "source_block_id"}
    if set(value) - allowed:
        raise ValueError("structured card artifact contains unsupported fields")
    artifact_id = str(value.get("artifact_id") or "").strip()
    if not artifact_id:
        raise ValueError("structured card artifact requires artifact_id")
    artifact_type = str(value.get("type") or "").strip().casefold()
    if artifact_type not in STRUCTURED_ARTIFACT_TYPES:
        raise ValueError(f"unsupported structured card artifact type: {artifact_type or '<empty>'}")
    content = value.get("content")
    if not isinstance(content, str) or not content.strip():
        raise ValueError("structured card artifact requires non-empty string content")
    language = str(value.get("language") or "").strip()
    expected_hash = _artifact_hash(artifact_type=artifact_type, language=language, content=content)
    supplied_hash = str(value.get("source_hash") or "").strip()
    if supplied_hash and supplied_hash != expected_hash:
        raise ValueError("structured card artifact source_hash does not match its exact content")
    normalized = {
        "artifact_id": artifact_id,
        "type": artifact_type,
        "content": content,
        "source_hash": expected_hash,
    }
    if language:
        normalized["language"] = language
    source_block_id = str(value.get("source_block_id") or "").strip()
    if source_block_id:
        normalized["source_block_id"] = source_block_id
    return normalized


def normalize_structured_content(value: Any) -> dict[str, Any]:
    """Validate one description-plus-facts card with immutable source artifacts."""

    if not isinstance(value, dict):
        raise ValueError("structured card content must be an object")
    if set(value) - {"description", "content", "artifacts"} or not {"description", "content"} <= set(value):
        raise ValueError("structured card content requires description, content, and optional artifacts fields")
    description = value.get("description")
    if not isinstance(description, str) or not description.strip():
        raise ValueError("structured card description must be a non-empty string")
    raw_facts = value.get("content")
    if not isinstance(raw_facts, list):
        raise ValueError("structured card content must be a non-empty string array")
    facts: list[str] = []
    for raw in raw_facts:
        if not isinstance(raw, str):
            raise ValueError("structured card facts must be strings")
        fact = " ".join(raw.split())
        if fact and fact not in facts:
            facts.append(fact)
    if not facts:
        raise ValueError("structured card content must contain at least one fact")
    normalized = {
        "description": " ".join(description.split()),
        "content": facts,
    }
    raw_artifacts = value.get("artifacts", [])
    if not isinstance(raw_artifacts, list):
        raise ValueError("structured card artifacts must be an array")
    artifacts: list[dict[str, Any]] = []
    hashes: set[str] = set()
    for raw_artifact in raw_artifacts:
        artifact = normalize_structured_artifact(raw_artifact)
        if artifact["source_hash"] in hashes:
            continue
        hashes.add(artifact["source_hash"])
        artifacts.append(artifact)
    if artifacts or "artifacts" in value:
        normalized["artifacts"] = artifacts
    return normalized


def is_structured_content(value: Any) -> bool:
    try:
        normalize_structured_content(value)
    except ValueError:
        return False
    return True


def structured_content_text(value: Any) -> str:
    """Return deterministic retrieval/display text without serializing JSON."""

    if not isinstance(value, dict):
        return str(value).strip()
    try:
        normalized = normalize_structured_content(value)
    except ValueError:
        return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
    sections = [
        normalized["description"],
        "\n".join(f"- {fact}" for fact in normalized["content"]),
    ]
    sections.extend(_structured_artifact_text(artifact) for artifact in normalized.get("artifacts", []))
    return "\n\n".join(section for section in sections if section)


def structured_content_semantic_text(value: Any) -> str:
    """Return the concise semantic projection used for dense matching.

    Exact source artifacts remain available in the stored/searchable card, but
    their often large byte bodies must not dominate the vector that decides
    whether two cards describe the same subject and facts.
    """

    if not isinstance(value, dict):
        return str(value).strip()
    try:
        normalized = normalize_structured_content(value)
    except ValueError:
        return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
    return "\n\n".join(
        (
            normalized["description"],
            "\n".join(f"- {fact}" for fact in normalized["content"]),
        )
    )


def _structured_artifact_text(artifact: dict[str, Any]) -> str:
    content = str(artifact["content"])
    if artifact["type"] == "code":
        return f"```{artifact.get('language', '')}\n{content}\n```"
    if artifact["type"] == "formula":
        return f"$$\n{content}\n$$"
    return content


def structured_content_from_metadata(metadata: Any, fallback: Any) -> Any:
    if isinstance(metadata, dict) and STRUCTURED_CONTENT_METADATA_KEY in metadata:
        return normalize_structured_content(metadata[STRUCTURED_CONTENT_METADATA_KEY])
    return fallback


def merge_structured_contents(sources: list[Any]) -> dict[str, Any]:
    """Deterministically union typed facts and exact artifacts without rewrites.

    The first source supplies the stable presentation description. Facts keep
    source order and their original normalized wording; exact repeats are
    retained once. Raw source blocks and revision metadata remain responsible
    for evidence outside this presentation value.
    """

    if not sources:
        raise ValueError("structured content merge requires at least one source")
    normalized_sources = [normalize_structured_content(source) for source in sources]
    merged = {
        "description": normalized_sources[0]["description"],
        "content": list(dict.fromkeys(fact for source in normalized_sources for fact in source["content"])),
    }
    artifacts: list[dict[str, Any]] = []
    hashes: set[str] = set()
    for artifact in (artifact for source in normalized_sources for artifact in source.get("artifacts", [])):
        if artifact["source_hash"] in hashes:
            continue
        hashes.add(artifact["source_hash"])
        artifacts.append(artifact)
    if artifacts:
        merged["artifacts"] = artifacts
    return merged


def unique_structured_values(values: list[Any]) -> list[Any]:
    result: list[Any] = []
    keys: set[str] = set()
    for value in values:
        key = json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
        if key not in keys:
            keys.add(key)
            result.append(value)
    return result
