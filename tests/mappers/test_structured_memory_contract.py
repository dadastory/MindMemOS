from datetime import UTC, datetime

from mindmemos.infra.db.filters import MEMORY_PAYLOAD_INDEX_SCHEMA
from mindmemos.mappers import to_memory_payload
from mindmemos.mappers.result import to_memory_view, to_memory_write
from mindmemos.typing import MemoryWrite


def structured_write(**overrides):
    values = {
        "memory_id": "00000000-0000-0000-0000-000000000010",
        "account_id": "account-1",
        "project_id": "project-1",
        "api_key_uuid": "key-1",
        "user_id": "user-1",
        "session_id": "task-1",
        "content": "Use bounded mutation.",
        "mem_type": "experience",
        "mem_extract_type": "structured",
        "mem_extract_version": "structured_add_v1",
        "content_fingerprint": "f" * 64,
        "idempotency_key": "source:event-1",
        "reinforcement_count": 2,
        "last_seen_at": datetime(2026, 8, 3, tzinfo=UTC),
        "created_at": datetime(2026, 8, 1, tzinfo=UTC),
        "property_name": "strategy",
        "entity_id": "00000000-0000-0000-0000-000000000020",
        "entity_type": "task_experience",
        "episode_ids": ["episode-current", "episode-related"],
    }
    values.update(overrides)
    return MemoryWrite(**values)


def test_structured_fields_round_trip_through_standard_memory_contract():
    write = structured_write()
    payload = to_memory_payload(write)
    view = to_memory_view(payload)
    reconstructed = to_memory_write(payload)

    assert payload["content_fingerprint"] == "f" * 64
    assert view.content_fingerprint == "f" * 64
    assert view.idempotency_key == "source:event-1"
    assert view.reinforcement_count == 2
    assert view.last_seen_at == datetime(2026, 8, 3, tzinfo=UTC)
    assert reconstructed.content_fingerprint == "f" * 64
    assert reconstructed.idempotency_key == "source:event-1"
    assert view.episode_ids == ["episode-current", "episode-related"]
    assert reconstructed.episode_ids == ["episode-current", "episode-related"]


def test_structured_filter_fields_are_indexed_without_entering_public_dsl():
    indexed = {item.field_name for item in MEMORY_PAYLOAD_INDEX_SCHEMA}

    assert {"content_fingerprint", "idempotency_key", "last_seen_at", "episode_ids"} <= indexed


def test_internal_memory_status_accepts_superseded():
    assert structured_write(status="superseded").status == "superseded"


def test_project_scoped_structured_memory_keeps_optional_actor_scope_absent():
    payload = to_memory_payload(structured_write(user_id=None, session_id=None))

    reconstructed = to_memory_write(payload)

    assert reconstructed.user_id is None
    assert reconstructed.session_id is None
