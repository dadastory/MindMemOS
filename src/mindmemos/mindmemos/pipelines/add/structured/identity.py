"""Canonical identities for convergent structured ingestion."""

from __future__ import annotations

import hashlib
import json
import re
import unicodedata
from typing import Any
from uuid import UUID, uuid4, uuid5

from ....typing import MemoryRequestContext

_ADD_RECORD_NAMESPACE = UUID("e0033a1e-7720-4f42-a65f-0db379afe1e5")
_REVISION_NAMESPACE = UUID("06a7d350-4dc1-45e4-aee2-8840d14e3066")
_WHITESPACE = re.compile(r"\s+")


def canonical_content(value: Any) -> str:
    """Return stable, Unicode-normalized text for strings or JSON values."""

    if isinstance(value, dict):
        normalized = {str(key): _canonical_json_value(item) for key, item in value.items()}
        return json.dumps(normalized, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
    if isinstance(value, list):
        normalized = [_canonical_json_value(item) for item in value]
        return json.dumps(normalized, ensure_ascii=False, separators=(",", ":"))
    return _canonical_text(str(value))


def structured_memory_fingerprint(
    context: MemoryRequestContext,
    *,
    entity_type: str,
    property_name: str,
    content: Any,
) -> str:
    """Hash exact typed content for replay/comparison, never as semantic identity.

    The digest deliberately excludes generated entity names. Storage IDs are
    allocated independently, while semantic equivalence is decided from the
    Schema, Episode, subjects, and full historical content by the merge model.
    """

    parts = [
        _canonical_text(context.account_id),
        _canonical_text(context.project_id),
        _canonical_text(context.user_id or ""),
        _canonical_text(context.app_id or ""),
        _canonical_text(context.session_id or ""),
        _canonical_text(context.agent_id or ""),
        _canonical_text(entity_type),
        _canonical_text(property_name),
        canonical_content(content),
    ]
    encoded = json.dumps(parts, ensure_ascii=False, separators=(",", ":")).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def new_structured_memory_id() -> str:
    """Allocate a storage ID after semantic resolution decides to create."""

    return str(uuid4())


def new_structured_entity_id() -> str:
    """Allocate an entity storage ID independently from its generated display title."""

    return str(uuid4())


def structured_add_record_id(context: MemoryRequestContext, idempotency_key: str) -> str:
    """Derive a stable Add Record ID for one opaque idempotency key."""

    key = "\x1f".join(
        [
            _canonical_text(context.account_id),
            _canonical_text(context.project_id),
            _canonical_text(context.user_id or ""),
            _canonical_text(context.app_id or ""),
            _canonical_text(context.session_id or ""),
            _canonical_text(context.agent_id or ""),
            unicodedata.normalize("NFKC", idempotency_key).strip(),
        ]
    )
    return str(uuid5(_ADD_RECORD_NAMESPACE, key))


def structured_revision_memory_id(fingerprint: str, event_key: str) -> str:
    """Derive a stable ID for an update/supersede revision of a property."""

    return str(uuid5(_REVISION_NAMESPACE, f"{fingerprint}\x1f{event_key}"))


def _canonical_json_value(value: Any) -> Any:
    if isinstance(value, str):
        return _canonical_text(value)
    if isinstance(value, dict):
        return {str(key): _canonical_json_value(item) for key, item in value.items()}
    if isinstance(value, list):
        return [_canonical_json_value(item) for item in value]
    return value


def _canonical_text(value: str) -> str:
    return _WHITESPACE.sub(" ", unicodedata.normalize("NFKC", value).strip()).casefold()
