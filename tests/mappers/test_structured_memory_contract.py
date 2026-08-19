from datetime import UTC, datetime

from mindmemos.infra.db.filters import MEMORY_PAYLOAD_INDEX_SCHEMA
from mindmemos.mappers import to_memory_payload
from mindmemos.mappers.result import to_memory_view, to_memory_write
from mindmemos.structured_content import (
    merge_structured_contents,
    normalize_structured_content,
    structured_content_semantic_text,
    structured_content_text,
)
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


def test_structured_card_preserves_code_artifact_bytes_and_renders_fenced_retrieval_text():
    code = "for(i = dk + 1; i <= L.length; ++ i){\n    L.r[0] = L.r[i];\n}"

    value = normalize_structured_content(
        {
            "description": "ShellInsert implementation.",
            "content": ["ShellInsert performs one gapped insertion pass."],
            "artifacts": [
                {
                    "artifact_id": "artifact-1",
                    "type": "code",
                    "language": "cpp",
                    "content": code,
                }
            ],
        }
    )

    assert value["artifacts"][0]["content"] == code
    assert len(value["artifacts"][0]["source_hash"]) == 64
    assert f"```cpp\n{code}\n```" in structured_content_text(value)


def test_structured_card_semantic_text_excludes_verbatim_artifact_body():
    code = "for(i = dk + 1; i <= L.length; ++ i){\n    L.r[0] = L.r[i];\n}"
    value = normalize_structured_content(
        {
            "description": "ShellInsert implementation.",
            "content": ["ShellInsert performs one gapped insertion pass."],
            "artifacts": [
                {
                    "artifact_id": "artifact-1",
                    "type": "code",
                    "language": "cpp",
                    "content": code,
                }
            ],
        }
    )

    semantic_text = structured_content_semantic_text(value)

    assert "ShellInsert implementation." in semantic_text
    assert "ShellInsert performs one gapped insertion pass." in semantic_text
    assert code not in semantic_text
    assert code in structured_content_text(value)


def test_structured_card_merge_unions_facts_and_deduplicates_identical_artifacts():
    code = "ShellInsert(L, dlta[k]);"

    merged = merge_structured_contents(
        [
            {
                "description": "Shell sort implementation.",
                "content": ["ShellSort iterates over the increment sequence."],
                "artifacts": [
                    {
                        "artifact_id": "artifact-a",
                        "type": "code",
                        "language": "cpp",
                        "content": code,
                        "source_block_id": "block-a",
                    }
                ],
            },
            {
                "description": "A later extraction must not replace the anchor.",
                "content": ["The final increment must be 1."],
                "artifacts": [
                    {
                        "artifact_id": "artifact-b",
                        "type": "code",
                        "language": "cpp",
                        "content": code,
                        "source_block_id": "block-b",
                    }
                ],
            },
        ]
    )

    assert merged["description"] == "Shell sort implementation."
    assert merged["content"] == [
        "ShellSort iterates over the increment sequence.",
        "The final increment must be 1.",
    ]
    assert len(merged["artifacts"]) == 1
    assert merged["artifacts"][0]["content"] == code
