import asyncio
from copy import deepcopy
from datetime import UTC, datetime
from types import SimpleNamespace

import mindmemos.pipelines.memory_db.writer as writer_module
import pytest
from mindmemos.config import TextProcessingConfig, get_config, init_config, reset_config
from mindmemos.errors import MemoryUpdateError
from mindmemos.infra.db import reset_database_clients
from mindmemos.pipelines.memory_db.add_record_store import AddRecordStore
from mindmemos.pipelines.memory_db.writer import MemoryDbWriter
from mindmemos.typing import MemoryDbDeleteCommand, MemoryDbMutationPlan, MemoryDbUpdateCommand
from mindmemos.typing.memory import (
    EntityVectorWrite,
    EntityWrite,
    GraphNodeRef,
    GraphRelationship,
    MemoryRequestContext,
    MemoryWrite,
    SourceWrite,
)
from mindmemos.typing.memory_db import MemoryDbWritePlan


@pytest.fixture(autouse=True)
def memory_db_writer_config() -> None:
    init_config(config_path="config/mindmemos/dev.example.yaml")
    reset_database_clients()
    try:
        yield
    finally:
        reset_database_clients()
        reset_config()


def make_text_config() -> TextProcessingConfig:
    return TextProcessingConfig(
        bm25_use_spacy_lemma=False,
        spacy_en_model="missing_en_model",
        spacy_zh_model="missing_zh_model",
        sparse_hash_dim=128,
    )


def make_context() -> MemoryRequestContext:
    return MemoryRequestContext(
        request_id="req-writer-1",
        account_id="acc-1",
        project_id="proj-1",
        api_key_uuid="key-1",
        user_id="user-1",
        session_id="session-1",
    )


def test_writer_resolves_embed_client_per_call_when_provider_binding_enabled(monkeypatch) -> None:
    get_config().provider_binding.enabled = True
    clients: list[object] = []

    def fake_get_embed_client():
        client = object()
        clients.append(client)
        return client

    monkeypatch.setattr(writer_module, "get_embed_client", fake_get_embed_client)
    writer = MemoryDbWriter(clients=SimpleNamespace(qdrant=SimpleNamespace(), neo4j=SimpleNamespace()))

    first = writer._ensure_embed_client()
    second = writer._ensure_embed_client()

    assert first is clients[0]
    assert second is clients[1]
    assert first is not second
    assert writer._embed_client is None


def test_writer_caches_embed_client_when_provider_binding_disabled(monkeypatch) -> None:
    get_config().provider_binding.enabled = False
    clients: list[object] = []

    def fake_get_embed_client():
        client = object()
        clients.append(client)
        return client

    monkeypatch.setattr(writer_module, "get_embed_client", fake_get_embed_client)
    writer = MemoryDbWriter(clients=SimpleNamespace(qdrant=SimpleNamespace(), neo4j=SimpleNamespace()))

    first = writer._ensure_embed_client()
    second = writer._ensure_embed_client()

    assert first is clients[0]
    assert second is clients[0]
    assert len(clients) == 1
    assert writer._embed_client is clients[0]


def make_plan(*, memories: int = 1) -> MemoryDbWritePlan:
    now = datetime.now(UTC)
    memory_writes = [
        MemoryWrite(
            memory_id=f"mem-{index}",
            account_id="acc-1",
            project_id="proj-1",
            api_key_uuid="key-1",
            user_id="user-1",
            session_id="session-1",
            request_id="req-writer-1",
            content=f"test memory {index}",
            mem_type="fact",
            mem_extract_type="vanilla",
            mem_extract_version="test",
            metadata={},
            created_at=now,
        )
        for index in range(memories)
    ]
    return MemoryDbWritePlan(
        memories=memory_writes,
        sources=[
            SourceWrite(
                source_id="src-0",
                account_id="acc-1",
                project_id="proj-1",
                api_key_uuid="key-1",
                user_id="user-1",
                session_id="session-1",
                request_id="req-writer-1",
                source_type="message",
                file_path="",
                file_name="",
                created_at=now,
                persist_payload=False,
            )
        ],
        relationships=[
            GraphRelationship(
                source=GraphNodeRef(kind="Memory", project_id="proj-1", node_id="mem-0"),
                target=GraphNodeRef(kind="Source", project_id="proj-1", node_id="src-0"),
                rel_type="EXTRACTED_FROM",
                project_id="proj-1",
            )
        ]
        if memories
        else [],
    )


class FakeQdrant:
    def __init__(self, record=None, records=None) -> None:
        self.record = record
        self.records = records
        self.calls = []
        self.updated_payloads = []
        self.patches = []
        self.deleted = []
        self.upserted_entities = []
        self.get_memory_calls = []
        self.get_memories_calls = []
        self.record_reads = 0

    async def upsert_memories(self, points):
        self.calls.append(("memories", len(points)))

    async def upsert_entities(self, points):
        self.calls.append(("entities", len(points)))
        self.upserted_entities.extend(points)

    async def upsert_sources(self, points):
        self.calls.append(("sources", len(points)))

    async def get_memory(self, project_id: str, memory_id: str, *, with_vectors: bool = False):
        self.record_reads += 1
        self.get_memory_calls.append((project_id, memory_id, with_vectors))
        return self._record_for(memory_id)

    async def get_memories(self, project_id: str, memory_ids: list[str], *, with_vectors: bool = False):
        self.record_reads += 1
        self.get_memories_calls.append((project_id, list(memory_ids), with_vectors))
        return [record for memory_id in memory_ids if (record := self._record_for(memory_id)) is not None]

    def _record_for(self, memory_id: str):
        record = self.records.get(memory_id) if self.records is not None else self.record
        if record is not None and not hasattr(record, "point_id"):
            record.point_id = memory_id
        return record

    async def patch_memory(
        self,
        project_id: str,
        memory_id: str,
        payload: dict,
        *,
        dense_vector=None,
        sparse_vector=None,
        record=None,
    ):
        self.patches.append(
            {
                "project_id": project_id,
                "memory_id": memory_id,
                "payload": payload,
                "dense_vector": dense_vector,
                "sparse_vector": sparse_vector,
                "record": record,
            }
        )

    async def update_memory_payload(self, project_id: str, memory_id: str, payload: dict):
        self.updated_payloads.append((project_id, memory_id, payload))

    async def delete_memory(self, project_id: str, memory_id: str):
        self.deleted.append((project_id, memory_id))


class ConcurrentCasQdrant(FakeQdrant):
    def __init__(self, *, always_conflict: bool = False) -> None:
        super().__init__(
            records={
                "mem-1": SimpleNamespace(
                    point_id="mem-1",
                    payload={
                        "project_id": "proj-1",
                        "status": "active",
                        "reinforcement_count": 0,
                        "metadata": {},
                    },
                )
            }
        )
        self._prefetch_count = 0
        self._prefetch_ready = asyncio.Event()
        self._cas_lock = asyncio.Lock()
        self.cas_calls = 0
        self.always_conflict = always_conflict

    def snapshot(self):
        record = self.records["mem-1"]
        return SimpleNamespace(point_id=record.point_id, payload=deepcopy(record.payload))

    async def get_memories(self, project_id: str, memory_ids: list[str], *, with_vectors: bool = False):
        self.get_memories_calls.append((project_id, list(memory_ids), with_vectors))
        snapshot = self.snapshot()
        self._prefetch_count += 1
        if self._prefetch_count == 2:
            self._prefetch_ready.set()
        await asyncio.wait_for(self._prefetch_ready.wait(), timeout=1)
        return [snapshot]

    async def get_memory(self, project_id: str, memory_id: str, *, with_vectors: bool = False):
        self.get_memory_calls.append((project_id, memory_id, with_vectors))
        return self.snapshot()

    async def patch_memory(self, project_id, memory_id, payload, **kwargs):
        await asyncio.sleep(0)
        self.records[memory_id].payload.update(deepcopy(payload))
        await super().patch_memory(project_id, memory_id, payload, **kwargs)

    async def compare_and_set_memory_payload(
        self,
        project_id: str,
        memory_id: str,
        payload: dict,
        *,
        expected_payload: dict,
    ):
        self.cas_calls += 1
        async with self._cas_lock:
            current = self.records[memory_id]
            if self.always_conflict or any(
                current.payload.get(key) != value for key, value in expected_payload.items()
            ):
                return self.snapshot()
            current.payload.update(deepcopy(payload))
            return self.snapshot()


class FakeNeo4j:
    def __init__(
        self,
        *,
        fail: bool = False,
        fail_archive: bool = False,
        fail_delete: bool = False,
        fail_content: bool = False,
    ) -> None:
        self.fail = fail
        self.fail_archive = fail_archive
        self.fail_delete = fail_delete
        self.fail_content = fail_content
        self.calls = []
        self.archived = []
        self.deleted = []
        self.content_updates = []

    async def upsert_nodes(self, *, memories=None, entities=None, sources=None):
        if self.fail:
            raise RuntimeError("neo4j connection lost")
        self.calls.append(("nodes", len(memories or []), len(entities or []), len(sources or [])))

    async def upsert_relationships(self, relationships):
        if self.fail:
            raise RuntimeError("neo4j connection lost")
        self.calls.append(("relationships", len(relationships)))

    async def update_memory_content(self, project_id: str, memory_id: str, content: str):
        if self.fail_content:
            raise RuntimeError("graph content update failed")
        self.content_updates.append((project_id, memory_id, content))

    async def archive_memory_node(
        self, project_id: str, memory_id: str, *, reason: str | None = None, status: str = "archived"
    ):
        if self.fail_archive:
            raise RuntimeError("archive failed")
        self.archived.append((project_id, memory_id, reason, status))

    async def delete_memory_node(self, project_id: str, memory_id: str):
        if self.fail_delete:
            raise RuntimeError("delete failed")
        self.deleted.append((project_id, memory_id))


class FakeAddRecordStore(AddRecordStore):
    def __init__(self) -> None:
        self.patches = []

    async def patch(self, project_id: str, add_record_id: str, payload: dict) -> None:
        self.patches.append((project_id, add_record_id, payload))


class FakeEmbedClient:
    def __init__(self, embeddings=None):
        self.embeddings = [[0.1, 0.2, 0.3]] if embeddings is None else embeddings
        self.calls = []

    async def embed(self, task: str, text: str):
        self.calls.append((task, text))
        return SimpleNamespace(embeddings=self.embeddings)


@pytest.mark.asyncio
async def test_write_batches_graph_nodes_and_relationships():
    qdrant = FakeQdrant()
    neo4j = FakeNeo4j()
    writer = MemoryDbWriter(clients=SimpleNamespace(qdrant=qdrant, neo4j=neo4j))

    result = await writer.write(make_context(), make_plan(memories=3), consistency="fast")

    assert result.graph_pending is False
    assert result.memory_ids == ["mem-0", "mem-1", "mem-2"]
    assert neo4j.calls == [("nodes", 3, 0, 1), ("relationships", 1)]


@pytest.mark.asyncio
async def test_write_skips_message_sources_in_qdrant_but_keeps_in_neo4j():
    """message sources (persist_payload=False) are written to Neo4j Source graph
    nodes but skipped for the source_item_v1 Qdrant collection; file/url sources
    go to both stores."""
    qdrant = FakeQdrant()
    neo4j = FakeNeo4j()
    writer = MemoryDbWriter(clients=SimpleNamespace(qdrant=qdrant, neo4j=neo4j))
    now = datetime.now(UTC)

    message_source = SourceWrite(
        source_id="src-msg",
        account_id="acc-1",
        project_id="proj-1",
        api_key_uuid="key-1",
        source_type="message",
        file_path="",
        file_name="",
        created_at=now,
        persist_payload=False,
    )
    file_source = SourceWrite(
        source_id="src-file",
        account_id="acc-1",
        project_id="proj-1",
        api_key_uuid="key-1",
        source_type="file",
        file_path="/tmp/a.txt",
        file_name="a.txt",
        created_at=now,
        persist_payload=True,
    )
    plan = MemoryDbWritePlan(sources=[message_source, file_source])

    await writer.write(make_context(), plan, consistency="fast")

    source_upserts = [call for call in qdrant.calls if call[0] == "sources"]
    assert source_upserts == [("sources", 1)], f"expected only file source in qdrant, got {source_upserts}"
    # both sources still become Neo4j Source nodes; call tuple is ("nodes", memories, entities, sources)
    assert any(call == ("nodes", 0, 0, 2) for call in neo4j.calls), (
        f"expected both sources mirrored to neo4j, got {neo4j.calls}"
    )


@pytest.mark.asyncio
async def test_write_keeps_search_field_entity_points_separate():
    qdrant = FakeQdrant()
    neo4j = FakeNeo4j()
    writer = MemoryDbWriter(clients=SimpleNamespace(qdrant=qdrant, neo4j=neo4j))
    now = datetime.now(UTC)
    entity = EntityWrite(
        entity_id="entity-user",
        account_id="acc-1",
        project_id="proj-1",
        api_key_uuid="key-1",
        user_id="user-1",
        entity_name="User",
        created_at=now,
        metadata={"search_fields": ["User likes Qdrant"]},
    )
    plan = MemoryDbWritePlan(
        entities=[entity],
        entity_vectors=[
            EntityVectorWrite(entity_id="entity-user", semantic_vector=[0.1], bm25_indices=[1], bm25_values=[1.0]),
            EntityVectorWrite(
                entity_id="entity-user#sf0",
                semantic_vector=[0.2],
                bm25_indices=[2],
                bm25_values=[1.0],
            ),
        ],
    )

    await writer.write(make_context(), plan, consistency="fast")

    assert qdrant.calls == [("memories", 0), ("entities", 1), ("sources", 0), ("entities", 1)]
    assert len(qdrant.upserted_entities) == 2
    search_field_point = [
        point for point in qdrant.upserted_entities if point.payload["metadata"].get("is_search_field")
    ]
    assert len(search_field_point) == 1
    assert search_field_point[0].payload["metadata"]["search_field_content"] == "User likes Qdrant"


@pytest.mark.asyncio
async def test_patch_add_record_uses_writer_add_record_store():
    store = FakeAddRecordStore()
    writer = MemoryDbWriter(clients=SimpleNamespace(qdrant=FakeQdrant(), neo4j=FakeNeo4j()), add_record_store=store)

    result = await writer.patch_add_record(make_context(), "add-1", {"feedback_processed": True})

    assert result.changed is True
    assert store.patches == [("proj-1", "add-1", {"feedback_processed": True})]


@pytest.mark.asyncio
async def test_update_entity_writes_entity_through_writer_boundary():
    qdrant = FakeQdrant()
    neo4j = FakeNeo4j()
    writer = MemoryDbWriter(clients=SimpleNamespace(qdrant=qdrant, neo4j=neo4j))
    now = datetime.now(UTC)
    entity = EntityWrite(
        entity_id="entity-user",
        account_id="acc-1",
        project_id="proj-1",
        api_key_uuid="key-1",
        user_id="user-1",
        entity_name="User",
        entity_type="person",
        description="Updated user description",
        created_at=now,
    )
    vector = EntityVectorWrite(entity_id="entity-user", semantic_vector=[0.1, 0.2], bm25_indices=[1], bm25_values=[1.0])

    result = await writer.update_entity(make_context(), entity, entity_vectors=[vector], consistency="fast")

    assert result.entity_ids == ["entity-user"]
    assert len(qdrant.upserted_entities) == 1
    assert qdrant.upserted_entities[0].payload["description"] == "Updated user description"
    assert neo4j.calls == [("nodes", 0, 1, 0)]


@pytest.mark.asyncio
async def test_update_content_refreshes_metadata_and_bm25_index():
    qdrant = FakeQdrant(record=SimpleNamespace(payload={"metadata": {"source": "test"}}))
    neo4j = FakeNeo4j()
    embed = FakeEmbedClient()
    writer = MemoryDbWriter(
        clients=SimpleNamespace(qdrant=qdrant, neo4j=neo4j),
        text_config=make_text_config(),
        embed_client=embed,
    )

    result = await writer.update_memory(
        make_context(),
        MemoryDbUpdateCommand(memory_id="mem-1", content="brand new content"),
    )

    assert result.changed is True
    assert len(qdrant.patches) == 1
    patch = qdrant.patches[0]
    assert patch["payload"]["content"]
    assert patch["payload"]["metadata"]["content_hash"]
    assert patch["payload"]["metadata"]["bm25_text"]
    assert patch["payload"]["metadata"]["source"] == "test"
    assert embed.calls == [("memory.update", patch["payload"]["content"])]
    assert patch["dense_vector"] == [0.1, 0.2, 0.3]
    assert patch["sparse_vector"] is not None
    assert len(patch["sparse_vector"].indices) == len(patch["sparse_vector"].values)
    assert patch["record"] is qdrant.record
    assert qdrant.record_reads == 1
    assert neo4j.content_updates == [("proj-1", "mem-1", patch["payload"]["content"])]


@pytest.mark.asyncio
async def test_update_content_uses_precomputed_vectors_when_provided():
    qdrant = FakeQdrant(record=SimpleNamespace(payload={"metadata": {}}))
    neo4j = FakeNeo4j()
    embed = FakeEmbedClient()
    writer = MemoryDbWriter(
        clients=SimpleNamespace(qdrant=qdrant, neo4j=neo4j),
        text_config=make_text_config(),
        embed_client=embed,
    )

    await writer.update_memory(
        make_context(),
        MemoryDbUpdateCommand(
            memory_id="mem-1",
            content="brand new content",
            embedding=[0.9, 0.8, 0.7],
            bm25_indices=[10, 11],
        ),
    )

    patch = qdrant.patches[0]
    assert embed.calls == []
    assert patch["dense_vector"] == [0.9, 0.8, 0.7]
    assert patch["sparse_vector"].indices == [10, 11]
    assert patch["sparse_vector"].values == [1.0, 1.0]


@pytest.mark.asyncio
async def test_update_content_accepts_legacy_vector_fields():
    qdrant = FakeQdrant(record=SimpleNamespace(payload={"metadata": {}}))
    writer = MemoryDbWriter(
        clients=SimpleNamespace(qdrant=qdrant, neo4j=FakeNeo4j()),
        text_config=make_text_config(),
        embed_client=FakeEmbedClient(),
    )

    await writer.update_memory(
        make_context(),
        MemoryDbUpdateCommand(
            memory_id="mem-1",
            content="new content",
            dense_vector=[0.1, 0.2],
            sparse_vectors={"bm25_indices": [1, 2], "bm25_values": [0.5, 1.5]},
        ),
    )

    patch = qdrant.patches[0]
    assert patch["dense_vector"] == [0.1, 0.2]
    assert patch["sparse_vector"].indices == [1, 2]
    assert patch["sparse_vector"].values == [0.5, 1.5]


@pytest.mark.asyncio
async def test_update_content_raises_when_embedding_provider_returns_empty_vector():
    qdrant = FakeQdrant(record=SimpleNamespace(payload={"metadata": {}}))
    writer = MemoryDbWriter(
        clients=SimpleNamespace(qdrant=qdrant, neo4j=FakeNeo4j()),
        text_config=make_text_config(),
        embed_client=FakeEmbedClient(embeddings=[]),
    )

    with pytest.raises(MemoryUpdateError, match="empty vector"):
        await writer.update_memory(
            make_context(),
            MemoryDbUpdateCommand(memory_id="mem-1", content="brand new content"),
        )

    assert qdrant.patches == []


@pytest.mark.asyncio
async def test_update_without_content_does_not_touch_bm25_index():
    qdrant = FakeQdrant(record=SimpleNamespace(payload={"metadata": {}}))
    neo4j = FakeNeo4j()
    embed = FakeEmbedClient()
    writer = MemoryDbWriter(
        clients=SimpleNamespace(qdrant=qdrant, neo4j=neo4j),
        text_config=make_text_config(),
        embed_client=embed,
    )

    await writer.update_memory(make_context(), MemoryDbUpdateCommand(memory_id="mem-1", status="archived"))

    assert len(qdrant.patches) == 1
    assert embed.calls == []
    assert qdrant.patches[0]["dense_vector"] is None
    assert qdrant.patches[0]["sparse_vector"] is None
    assert neo4j.content_updates == []
    assert neo4j.archived == [("proj-1", "mem-1", "unknown", "archived")]


@pytest.mark.asyncio
async def test_superseded_update_preserves_distinct_graph_lifecycle_status():
    qdrant = FakeQdrant(record=SimpleNamespace(payload={"metadata": {}}))
    neo4j = FakeNeo4j()
    writer = MemoryDbWriter(clients=SimpleNamespace(qdrant=qdrant, neo4j=neo4j))

    await writer.update_memory(
        make_context(),
        MemoryDbUpdateCommand(memory_id="mem-1", status="superseded", reason="new evidence"),
    )

    assert qdrant.patches[0]["payload"]["status"] == "superseded"
    assert neo4j.archived == [("proj-1", "mem-1", "new evidence", "superseded")]


@pytest.mark.asyncio
async def test_update_missing_memory_is_noop():
    qdrant = FakeQdrant(record=None)
    writer = MemoryDbWriter(clients=SimpleNamespace(qdrant=qdrant, neo4j=FakeNeo4j()))

    result = await writer.update_memory(make_context(), MemoryDbUpdateCommand(memory_id="missing", content="new"))

    assert result.changed is False
    assert qdrant.patches == []


@pytest.mark.asyncio
async def test_update_dedup_metadata_key_skips_duplicate_command():
    qdrant = FakeQdrant(record=SimpleNamespace(payload={"metadata": {"last_request_id": "req-1"}}))
    writer = MemoryDbWriter(clients=SimpleNamespace(qdrant=qdrant, neo4j=FakeNeo4j()))

    result = await writer.update_memory(
        make_context(),
        MemoryDbUpdateCommand(
            memory_id="mem-1",
            metadata_patch={"last_request_id": "req-1"},
            dedup_metadata_key="last_request_id",
        ),
    )

    assert result.changed is False
    assert qdrant.patches == []


@pytest.mark.asyncio
async def test_apply_mutation_plan_prefetches_memory_records_once_for_updates():
    qdrant = FakeQdrant(
        records={
            "mem-1": SimpleNamespace(point_id="mem-1", payload={"metadata": {}}),
            "mem-2": SimpleNamespace(point_id="mem-2", payload={"metadata": {}}),
        }
    )
    writer = MemoryDbWriter(clients=SimpleNamespace(qdrant=qdrant, neo4j=FakeNeo4j()))

    result = await writer.apply_mutation_plan(
        make_context(),
        MemoryDbMutationPlan(
            memory_updates=[
                MemoryDbUpdateCommand(memory_id="mem-1", metadata_patch={"a": 1}, consistency="fast"),
                MemoryDbUpdateCommand(memory_id="mem-2", metadata_patch={"b": 2}, consistency="fast"),
            ]
        ),
        consistency="fast",
    )

    assert [mutation.changed for mutation in result.mutations] == [True, True]
    assert qdrant.get_memories_calls == [("proj-1", ["mem-1", "mem-2"], False)]
    assert qdrant.get_memory_calls == []
    assert [patch["memory_id"] for patch in qdrant.patches] == ["mem-1", "mem-2"]


@pytest.mark.asyncio
async def test_repeated_reinforcement_delta_uses_prefetched_record_and_local_patch():
    record = SimpleNamespace(point_id="mem-1", payload={"metadata": {}, "reinforcement_count": 3})
    qdrant = FakeQdrant(records={"mem-1": record})
    writer = MemoryDbWriter(clients=SimpleNamespace(qdrant=qdrant, neo4j=FakeNeo4j()))

    await writer.apply_mutation_plan(
        make_context(),
        MemoryDbMutationPlan(
            memory_updates=[
                MemoryDbUpdateCommand(memory_id="mem-1", reinforcement_count_delta=1, consistency="fast"),
                MemoryDbUpdateCommand(memory_id="mem-1", reinforcement_count_delta=2, consistency="fast"),
            ]
        ),
        consistency="fast",
    )

    assert qdrant.get_memories_calls == [("proj-1", ["mem-1"], False)]
    assert [patch["payload"]["reinforcement_count"] for patch in qdrant.patches] == [4, 6]
    assert record.payload["reinforcement_count"] == 6


def reinforcement_command(
    token: str,
    source: str,
    *,
    episode_id: str | None = None,
    merged_memory_id: str | None = None,
    retries: int = 8,
) -> MemoryDbUpdateCommand:
    return MemoryDbUpdateCommand(
        memory_id="mem-1",
        reinforcement_count_delta=1,
        metadata_patch={
            "last_reinforcement_event_id": token,
            "source_add_record_ids": [source],
            "idempotency_keys": [token],
            "evidence_history": [{"idempotency_key": token, "add_record_id": source}],
            "merged_memory_ids": [merged_memory_id] if merged_memory_id else [],
        },
        payload_patch={"episode_ids": [episode_id]} if episode_id else {},
        dedup_metadata_key="last_reinforcement_event_id",
        optimistic_lock_token=token,
        optimistic_lock_retries=retries,
        consistency="strong",
    )


@pytest.mark.asyncio
async def test_concurrent_reinforcements_use_cas_and_preserve_both_evidence_lists():
    qdrant = ConcurrentCasQdrant()
    first = MemoryDbWriter(clients=SimpleNamespace(qdrant=qdrant, neo4j=FakeNeo4j()))
    second = MemoryDbWriter(clients=SimpleNamespace(qdrant=qdrant, neo4j=FakeNeo4j()))

    results = await asyncio.gather(
        first.apply_mutation_plan(
            make_context(),
            MemoryDbMutationPlan(memory_updates=[reinforcement_command("event-1", "add-1", episode_id="episode-1")]),
            consistency="strong",
        ),
        second.apply_mutation_plan(
            make_context(),
            MemoryDbMutationPlan(memory_updates=[reinforcement_command("event-2", "add-2", episode_id="episode-2")]),
            consistency="strong",
        ),
    )

    payload = qdrant.records["mem-1"].payload
    assert [result.mutations[0].changed for result in results] == [True, True]
    assert payload["reinforcement_count"] == 2
    assert set(payload["metadata"]["source_add_record_ids"]) == {"add-1", "add-2"}
    assert set(payload["metadata"]["idempotency_keys"]) == {"event-1", "event-2"}
    assert {item["idempotency_key"] for item in payload["metadata"]["evidence_history"]} == {
        "event-1",
        "event-2",
    }
    assert set(payload["episode_ids"]) == {"episode-1", "episode-2"}
    assert qdrant.cas_calls >= 3


@pytest.mark.asyncio
async def test_concurrent_reinforcements_preserve_all_merged_memory_ids():
    qdrant = ConcurrentCasQdrant()
    first = MemoryDbWriter(clients=SimpleNamespace(qdrant=qdrant, neo4j=FakeNeo4j()))
    second = MemoryDbWriter(clients=SimpleNamespace(qdrant=qdrant, neo4j=FakeNeo4j()))

    await asyncio.gather(
        first.apply_mutation_plan(
            make_context(),
            MemoryDbMutationPlan(
                memory_updates=[reinforcement_command("event-1", "add-1", merged_memory_id="duplicate-1")]
            ),
            consistency="strong",
        ),
        second.apply_mutation_plan(
            make_context(),
            MemoryDbMutationPlan(
                memory_updates=[reinforcement_command("event-2", "add-2", merged_memory_id="duplicate-2")]
            ),
            consistency="strong",
        ),
    )

    assert set(qdrant.records["mem-1"].payload["metadata"]["merged_memory_ids"]) == {
        "duplicate-1",
        "duplicate-2",
    }


@pytest.mark.asyncio
async def test_concurrent_replay_of_same_reinforcement_token_counts_once():
    qdrant = ConcurrentCasQdrant()
    first = MemoryDbWriter(clients=SimpleNamespace(qdrant=qdrant, neo4j=FakeNeo4j()))
    second = MemoryDbWriter(clients=SimpleNamespace(qdrant=qdrant, neo4j=FakeNeo4j()))
    command = reinforcement_command("event-1", "add-1")

    results = await asyncio.gather(
        first.apply_mutation_plan(make_context(), MemoryDbMutationPlan(memory_updates=[command]), consistency="strong"),
        second.apply_mutation_plan(
            make_context(), MemoryDbMutationPlan(memory_updates=[command]), consistency="strong"
        ),
    )

    payload = qdrant.records["mem-1"].payload
    assert sorted(result.mutations[0].changed for result in results) == [False, True]
    assert payload["reinforcement_count"] == 1
    assert payload["metadata"]["source_add_record_ids"] == ["add-1"]


@pytest.mark.asyncio
async def test_reinforcement_cas_exhaustion_raises_instead_of_losing_increment():
    qdrant = ConcurrentCasQdrant(always_conflict=True)
    qdrant._prefetch_ready.set()
    writer = MemoryDbWriter(clients=SimpleNamespace(qdrant=qdrant, neo4j=FakeNeo4j()))

    with pytest.raises(MemoryUpdateError, match="optimistic update conflict"):
        await writer.apply_mutation_plan(
            make_context(),
            MemoryDbMutationPlan(memory_updates=[reinforcement_command("event-1", "add-1", retries=2)]),
            consistency="strong",
        )

    assert qdrant.records["mem-1"].payload["reinforcement_count"] == 0
    assert qdrant.cas_calls == 3


@pytest.mark.asyncio
async def test_soft_delete_missing_memory_does_not_create_graph_node():
    qdrant = FakeQdrant(record=None)
    neo4j = FakeNeo4j()
    writer = MemoryDbWriter(clients=SimpleNamespace(qdrant=qdrant, neo4j=neo4j))

    result = await writer.delete_memory(make_context(), MemoryDbDeleteCommand(memory_id="missing"))

    assert result.changed is False
    assert qdrant.patches == []
    assert neo4j.archived == []


@pytest.mark.asyncio
async def test_delete_archives_without_physical_deletes():
    qdrant = FakeQdrant(record=SimpleNamespace(payload={"metadata": {}}))
    neo4j = FakeNeo4j()
    writer = MemoryDbWriter(clients=SimpleNamespace(qdrant=qdrant, neo4j=neo4j))

    result = await writer.delete_memory(
        make_context(),
        MemoryDbDeleteCommand(memory_id="mem-1"),
    )

    assert result.changed is True
    assert qdrant.deleted == []
    assert neo4j.deleted == []
    assert qdrant.patches[0]["payload"]["status"] == "archived"
    assert neo4j.archived == [("proj-1", "mem-1", "user_request", "archived")]


@pytest.mark.asyncio
async def test_soft_delete_fast_consistency_tolerates_graph_archive_failure():
    qdrant = FakeQdrant(record=SimpleNamespace(payload={"metadata": {"source": "test"}}))
    neo4j = FakeNeo4j(fail_archive=True)
    writer = MemoryDbWriter(clients=SimpleNamespace(qdrant=qdrant, neo4j=neo4j))

    result = await writer.delete_memory(
        make_context(),
        MemoryDbDeleteCommand(memory_id="mem-1", reason="user_request", consistency="fast"),
    )

    assert result.changed is True
    assert qdrant.patches[0]["payload"]["status"] == "archived"
    assert qdrant.patches[0]["payload"]["metadata"] == {"source": "test", "delete_reason": "user_request"}


@pytest.mark.asyncio
async def test_soft_delete_strong_consistency_raises_graph_archive_failure():
    qdrant = FakeQdrant(record=SimpleNamespace(payload={"metadata": {}}))
    neo4j = FakeNeo4j(fail_archive=True)
    writer = MemoryDbWriter(clients=SimpleNamespace(qdrant=qdrant, neo4j=neo4j))

    with pytest.raises(RuntimeError, match="archive failed"):
        await writer.delete_memory(
            make_context(),
            MemoryDbDeleteCommand(memory_id="mem-1", consistency="strong"),
        )
