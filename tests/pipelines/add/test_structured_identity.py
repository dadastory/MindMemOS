from inspect import signature

from mindmemos.pipelines.add.structured.identity import (
    canonical_content,
    new_structured_entity_id,
    new_structured_memory_id,
    structured_add_record_id,
    structured_memory_fingerprint,
)
from mindmemos.typing import MemoryRequestContext


def context(*, project_id="project-1", user_id="user-1", session_id="task-1"):
    return MemoryRequestContext(
        request_id="00000000-0000-0000-0000-000000000001",
        account_id="account-1",
        project_id=project_id,
        api_key_uuid="key-1",
        user_id=user_id,
        session_id=session_id,
        memory_algorithm="structured",
    )


def test_canonical_content_normalizes_unicode_whitespace_and_json_order():
    assert canonical_content("  Ａ  strategy\nworks ") == "a strategy works"
    assert canonical_content({"b": 2, "a": [" Ａ ", 1]}) == '{"a":["a",1],"b":2}'


def test_structured_fingerprint_is_stable_for_same_scope():
    ctx = context()
    first = structured_memory_fingerprint(
        ctx,
        entity_type="task_experience",
        property_name="strategy",
        content="Use elitism.",
    )
    second = structured_memory_fingerprint(
        ctx,
        entity_type=" TASK_EXPERIENCE ",
        property_name="Strategy",
        content=" use  elitism. ",
    )

    assert first == second
    assert structured_add_record_id(ctx, "source:event-1") == structured_add_record_id(ctx, "source:event-1")


def test_structured_identity_changes_across_tenant_or_task_scope():
    base = structured_memory_fingerprint(
        context(),
        entity_type="task_experience",
        property_name="strategy",
        content="Use elitism.",
    )

    assert base != structured_memory_fingerprint(
        context(project_id="project-2"),
        entity_type="task_experience",
        property_name="strategy",
        content="Use elitism.",
    )
    assert base != structured_memory_fingerprint(
        context(user_id="user-2"),
        entity_type="task_experience",
        property_name="strategy",
        content="Use elitism.",
    )
    assert base != structured_memory_fingerprint(
        context(session_id="task-2"),
        entity_type="task_experience",
        property_name="strategy",
        content="Use elitism.",
    )


def test_generated_title_is_not_part_of_structured_semantic_fingerprint():
    assert "entity_name" not in signature(structured_memory_fingerprint).parameters


def test_new_structured_entities_and_memories_receive_storage_ids_not_title_ids():
    first_entity = new_structured_entity_id()
    second_entity = new_structured_entity_id()
    first_memory = new_structured_memory_id()
    second_memory = new_structured_memory_id()

    assert first_entity != second_entity
    assert first_memory != second_memory
    assert len(first_entity) == len(first_memory) == 36
