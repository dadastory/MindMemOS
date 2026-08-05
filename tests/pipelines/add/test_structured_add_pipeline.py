from __future__ import annotations

import asyncio
from datetime import UTC, datetime

import pytest
from mindmemos.config import StructuredAddConfig
from mindmemos.pipelines.add.structured.pipeline import StructuredAddPipeline
from mindmemos.pipelines.add.structured.planner import StructuredDecision
from mindmemos.typing import (
    AddPipelineInput,
    DialogueMessage,
    EntitySearchHit,
    EntitySearchResult,
    EntityView,
    MemoryDbSearchHit,
    MemoryDbSearchResult,
    MemoryDbWriteResult,
    MemoryRequestContext,
    MemoryView,
)


class _Extractor:
    def __init__(self, value):
        self.value = value
        self.calls = 0
        self.schema_version = "task-memory-schema.json"
        self.schema_context = [
            {
                "entity_type": "AlgorithmExperience",
                "entity_description": "A structured observation",
                "dynamic_property": {"finding": {"desc": "Observed fact"}},
            }
        ]

    async def extract(self, **_kwargs):
        self.calls += 1
        if isinstance(self.value, Exception):
            raise self.value
        return self.value


class _Embed:
    def __init__(self):
        self.calls: list[tuple[str, object]] = []

    async def embed(self, *, task, text, **_kwargs):
        self.calls.append((task, text))
        values = text if isinstance(text, list) else [text]
        return type("Response", (), {"embeddings": [[float(index + 1), 0.5] for index, _ in enumerate(values)]})()


class _Sparse:
    def encode_document(self, _tokens):
        return type("Sparse", (), {"indices": [1], "values": [1.0]})()


class _Preprocessor:
    def preprocess_text(self, text, **_kwargs):
        return type("Prepared", (), {"tokens": text.split()})()


class _Reader:
    def __init__(self, memories=None, hits=None, episode_hits=None, entities=None):
        self.memories = dict(memories or {})
        self.hits = list(hits or [])
        self.episode_hits = list(episode_hits or [])
        self.entities = dict(entities or {})
        self.get_calls: list[str] = []
        self.search_calls = 0

    async def get_memory(self, _ctx, memory_id):
        self.get_calls.append(memory_id)
        return self.memories.get(memory_id)

    async def search_dense(self, _ctx, query, *, query_vector):
        self.search_calls += 1
        return MemoryDbSearchResult(query=query.query, hits=self.hits, total=len(self.hits))

    async def search_entities_dense(self, _ctx, **kwargs):
        return EntitySearchResult(query=kwargs["query"], hits=self.episode_hits, total=len(self.episode_hits))

    async def get_entity(self, _ctx, entity_id):
        return self.entities.get(entity_id)


class _EpisodeFilteredReader(_Reader):
    """Return historical cards only when their Episode is inside the query fence."""

    async def search_dense(self, _ctx, query, *, query_vector):
        self.search_calls += 1
        allowed_episode_ids: set[str] = set()
        for condition in query.filters.must:
            if condition.field == "episode_ids" and condition.op == "any":
                allowed_episode_ids.update(condition.values or [])
        hits = [
            hit
            for hit in self.hits
            if hit.memory is not None and allowed_episode_ids.intersection(hit.memory.episode_ids)
        ]
        return MemoryDbSearchResult(query=query.query, hits=hits, total=len(hits))


class _Writer:
    def __init__(self):
        self.plans = []

    async def apply_mutation_plan(self, _ctx, plan, *, consistency):
        self.plans.append(plan)
        return MemoryDbWriteResult(
            memory_ids=[command.memory.memory_id for command in plan.memory_writes],
            entity_ids=[command.entity.entity_id for command in plan.entity_writes],
        )


class _Recorder:
    def __init__(self):
        self.completed = []

    async def mark_add_completed(self, context, add_record_id, result):
        self.completed.append((context, add_record_id, result))


class _Merge:
    def __init__(self, decision):
        self.decision = decision
        self.calls = 0

    async def decide(self, _content, _candidates):
        self.calls += 1
        return self.decision


class _BatchMerge:
    def __init__(self):
        self.calls = []

    async def decide_batch(
        self,
        requests,
        *,
        current_episode,
        episode_contexts,
        entity_schema=None,
    ):
        self.calls.append((requests, current_episode, episode_contexts, entity_schema))
        return [StructuredDecision(action="create", reason="distinct") for _ in requests]


def _ctx() -> MemoryRequestContext:
    return MemoryRequestContext(
        request_id="request-1",
        account_id="account-1",
        project_id="project-1",
        api_key_uuid="key-1",
        user_id="user-1",
        session_id="task-1",
        memory_algorithm="structured",
    )


def _input(key="event-1") -> AddPipelineInput:
    return AddPipelineInput(
        messages=[DialogueMessage(role="user", content="candidate evidence", timestamp=1_780_000_000_000)],
        metadata={"score": 0.9, "generation": 7, "algorithm_id": "algo-3"},
        idempotency_key=key,
    )


def _extraction(value="Use adaptive mutation"):
    return {
        "entities": [
            {
                "name": "Adaptive mutation strategy",
                "entity_type": "AlgorithmExperience",
                "description": "A reusable strategy",
                "record_time": "2026-08-03",
                "properties": [
                    {
                        "property_name": "finding",
                        "value": value,
                        "time": "2026-08-03",
                        "operation": "set",
                    }
                ],
            }
        ],
        "edges": [],
    }


def _experience_extraction(value="Use adaptive mutation"):
    result = _extraction(value)
    result["entities"][0]["entity_type"] = "task_experience"
    result["entities"][0]["properties"][0]["property_name"] = "strategy"
    return result


def _episode_extraction(*, action="create", target_episode_id=None, value="Use adaptive mutation"):
    result = _experience_extraction(value)
    result["episode"] = {
        "action": action,
        "target_episode_id": target_episode_id,
        "title": "Current experiment",
        "description": "Current experiment background",
        "related_episode_ids": [],
    }
    return result


def _episode_hit(episode_id="episode-existing", *, session_id="task-1"):
    entity = EntityView(
        entity_id=episode_id,
        project_id="project-1",
        entity_name="Existing experiment",
        entity_type="episodes",
        description="Existing experiment background",
        user_id="user-1",
        session_id=session_id,
        created_at=datetime(2026, 1, 1, tzinfo=UTC),
    )
    return EntitySearchHit(entity_id=episode_id, score=0.95, entity=entity)


def _memory(memory_id="old-memory", content="Old guidance") -> MemoryView:
    return MemoryView(
        memory_id=memory_id,
        project_id="project-1",
        content=content,
        mem_type="fact",
        mem_extract_type="structured",
        mem_extract_version="structured_add_v1",
        status="active",
        metadata={
            "source_add_record_ids": ["earlier"],
            "evidence_history": [{"add_record_id": "earlier", "metadata": {"score": 0.5}}],
        },
        account_id="account-1",
        api_key_uuid="key-1",
        user_id="user-1",
        session_id="task-1",
        entity_id="entity-1",
        entity_type="AlgorithmExperience",
        property_name="finding",
        created_at=datetime(2026, 1, 1, tzinfo=UTC),
    )


def _pipeline(*, extraction=None, reader=None, merge=None, config=None):
    writer = _Writer()
    recorder = _Recorder()
    pipeline = StructuredAddPipeline(
        structured_add_config=config or StructuredAddConfig(),
        extractor=_Extractor(_extraction() if extraction is None else extraction),
        merge_decider=merge,
        llm_client=None,
        embed_client=_Embed(),
        text_preprocessor=_Preprocessor(),
        sparse_encoder=_Sparse(),
        db_reader=reader or _Reader(),
        db_writer=writer,
        recorder=recorder,
        consistency="strong",
    )
    return pipeline, writer, recorder


@pytest.mark.asyncio
async def test_prepared_fingerprint_ignores_generated_title_but_storage_ids_are_independent():
    pipeline, _, _ = _pipeline(extraction=_experience_extraction())

    first = await pipeline._prepare_properties(_experience_extraction(), _ctx())
    renamed = _experience_extraction()
    renamed["entities"][0]["name"] = "A different generated title"
    second = await pipeline._prepare_properties(renamed, _ctx())

    assert first[0].fingerprint == second[0].fingerprint
    assert first[0].entity_name != second[0].entity_name
    assert first[0].entity_id != second[0].entity_id
    assert first[0].memory_id != second[0].memory_id


@pytest.mark.asyncio
async def test_preparation_does_not_drop_equal_properties_from_different_entities():
    extraction = _experience_extraction("same status")
    second = dict(extraction["entities"][0])
    second["name"] = "A second entity"
    second["description"] = "A distinct subject with an equal property value"
    second["properties"] = [dict(extraction["entities"][0]["properties"][0])]
    extraction["entities"].append(second)
    pipeline, _, _ = _pipeline(extraction=extraction)

    properties = await pipeline._prepare_properties(extraction, _ctx())

    assert len(properties) == 2
    assert properties[0].fingerprint == properties[1].fingerprint
    assert properties[0].entity_id != properties[1].entity_id


@pytest.mark.asyncio
async def test_contextual_merge_receives_schema_and_new_and_historical_entity_contexts():
    historical = _memory("old-memory", "Historical observation")
    historical.entity_id = "stored-entity"
    stored_entity = EntityView(
        entity_id="stored-entity",
        project_id="project-1",
        entity_name="Historical subject",
        entity_type="AlgorithmExperience",
        description="Stored entity description",
        user_id="user-1",
        session_id="task-1",
        created_at=datetime(2026, 1, 1, tzinfo=UTC),
    )
    reader = _Reader(
        memories={historical.memory_id: historical},
        hits=[MemoryDbSearchHit(memory_id=historical.memory_id, score=0.9, memory=historical)],
        entities={stored_entity.entity_id: stored_entity},
    )
    merge = _BatchMerge()
    pipeline, _, _ = _pipeline(extraction=_extraction("Current observation"), reader=reader, merge=merge)

    await pipeline.add_sync(_input(), _ctx(), add_record_id="add-context")

    assert len(merge.calls) == 1
    requests, _, _, schema = merge.calls[0]
    assert schema == pipeline._explicit_extractor.schema_context
    assert requests[0].property.entity_description == "A reusable strategy"
    assert requests[0].candidate_entity_contexts[historical.memory_id] == {
        "entity_id": "stored-entity",
        "name": "Historical subject",
        "description": "Stored entity description",
        "entity_type": "AlgorithmExperience",
    }


@pytest.mark.asyncio
async def test_create_emits_standard_writes_relationship_event_and_record_output():
    pipeline, writer, recorder = _pipeline(extraction=_experience_extraction())

    result = await pipeline.add_sync(_input(), _ctx(), add_record_id="add-1")

    assert result.status == "ok"
    assert [item.operation for item in result.memories] == ["add"]
    plan = writer.plans[0]
    assert len(plan.memory_writes) == len(plan.entity_writes) == 1
    memory = plan.memory_writes[0].memory
    assert memory.mem_extract_type == "structured"
    assert memory.mem_type == "experience"
    assert memory.content_fingerprint
    assert memory.idempotency_key == "event-1"
    assert memory.metadata["source_add_record_ids"] == ["add-1"]
    assert memory.metadata["score"] == 0.9
    assert plan.memory_writes[0].vector.semantic_vector == [1.0, 0.5]
    assert any(command.relationship.rel_type == "HAS_PROPERTY_MEMORY" for command in plan.relationship_writes)
    assert plan.entity_writes[0].entity.schema_version == "task-memory-schema.json"
    assert result.memories[0].mem_type == "experience"
    assert recorder.completed[0][1] == "add-1"


@pytest.mark.asyncio
async def test_create_persists_lightweight_episode_and_links_card_without_new_request_fields():
    pipeline, writer, _ = _pipeline(extraction=_episode_extraction(), reader=_Reader())

    result = await pipeline.add_sync(_input(key=None), _ctx(), add_record_id="add-episode")

    assert result.status == "ok"
    plan = writer.plans[0]
    episode = next(command.entity for command in plan.entity_writes if command.entity.entity_type == "episodes")
    card = next(command.entity for command in plan.entity_writes if command.entity.entity_type != "episodes")
    memory = plan.memory_writes[0].memory
    assert memory.episode_ids == [episode.entity_id]
    assert any(
        command.relationship.rel_type == "OBSERVED_IN"
        and command.relationship.source.node_id == card.entity_id
        and command.relationship.target.node_id == episode.entity_id
        for command in plan.relationship_writes
    )


@pytest.mark.asyncio
async def test_propertyless_entities_and_explicit_edge_are_persisted_with_episode_provenance():
    extraction = {
        "episode": {
            "action": "create",
            "target_episode_id": None,
            "title": "Service incident",
            "description": "A gateway outage affected a work item",
            "related_episode_ids": [],
        },
        "entities": [
            {
                "name": "gateway-a",
                "entity_type": "device",
                "description": "Primary warehouse gateway",
                "record_time": "2026-08-04",
                "properties": [],
            },
            {
                "name": "ticket-42",
                "entity_type": "ticket",
                "description": "Connectivity incident",
                "record_time": "2026-08-04",
                "properties": [],
            },
        ],
        "edges": [
            {
                "link_entity1_name": "gateway-a",
                "link_entity2_name": "ticket-42",
                "link_description": "caused",
            }
        ],
    }
    pipeline, writer, _ = _pipeline(extraction=extraction)

    result = await pipeline.add_sync(_input(), _ctx(), add_record_id="add-entities")

    assert result.memories == []
    plan = writer.plans[0]
    entity_writes = [command.entity for command in plan.entity_writes]
    domain_entities = [entity for entity in entity_writes if entity.entity_type != "episodes"]
    assert {entity.entity_name for entity in domain_entities} == {"gateway-a", "ticket-42"}
    assert {command.core_vector.entity_id for command in plan.entity_writes if command.core_vector is not None} >= {
        entity.entity_id for entity in domain_entities
    }
    episode_id = next(entity.entity_id for entity in entity_writes if entity.entity_type == "episodes")
    observed_sources = {
        command.relationship.source.node_id
        for command in plan.relationship_writes
        if command.relationship.rel_type == "OBSERVED_IN" and command.relationship.target.node_id == episode_id
    }
    assert observed_sources == {entity.entity_id for entity in domain_entities}
    assert any(
        command.relationship.rel_type == "RELATED_TO"
        and {
            command.relationship.source.node_id,
            command.relationship.target.node_id,
        }
        == {entity.entity_id for entity in domain_entities}
        for command in plan.relationship_writes
    )


@pytest.mark.asyncio
async def test_reuse_does_not_overwrite_episode_and_links_new_card_to_existing_episode():
    hit = _episode_hit()
    reader = _Reader(episode_hits=[hit], entities={hit.entity_id: hit.entity})
    pipeline, writer, _ = _pipeline(
        extraction=_episode_extraction(action="reuse", target_episode_id=hit.entity_id),
        reader=reader,
    )

    await pipeline.add_sync(_input(), _ctx(), add_record_id="add-reuse")

    plan = writer.plans[0]
    assert all(command.entity.entity_id != hit.entity_id for command in plan.entity_writes)
    assert plan.memory_writes[0].memory.episode_ids == [hit.entity_id]
    assert any(
        command.relationship.rel_type == "OBSERVED_IN" and command.relationship.target.node_id == hit.entity_id
        for command in plan.relationship_writes
    )


@pytest.mark.asyncio
async def test_new_episode_persists_whitelisted_related_episode_context():
    related = _episode_hit("episode-related", session_id="previous-task")
    extraction = _episode_extraction(action="create")
    extraction["episode"]["related_episode_ids"] = [related.entity_id]
    reader = _Reader(episode_hits=[related], entities={related.entity_id: related.entity})
    pipeline, writer, _ = _pipeline(extraction=extraction, reader=reader)

    await pipeline.add_sync(_input(), _ctx(), add_record_id="add-related-episode")

    plan = writer.plans[0]
    created_episode = next(command.entity for command in plan.entity_writes if command.entity.entity_type == "episodes")
    assert any(
        command.relationship.rel_type == "RELATES_TO"
        and command.relationship.source.node_id == created_episode.entity_id
        and command.relationship.target.node_id == related.entity_id
        for command in plan.relationship_writes
    )


@pytest.mark.asyncio
async def test_contextual_reinforce_reuses_historical_entity_and_appends_episode_evidence():
    from mindmemos.pipelines.add.structured.planner import StructuredDecision

    current_episode = _episode_hit("episode-current")
    target_entity = EntityView(
        entity_id="entity-existing",
        project_id="project-1",
        entity_name="Canonical historical title",
        entity_type="task_experience",
        description="Canonical historical description",
        user_id="user-1",
        session_id="task-1",
        created_at=datetime(2026, 1, 1, tzinfo=UTC),
    )
    target = _memory("memory-existing", "Increase mutation after stagnation.")
    target.entity_id = target_entity.entity_id
    target.entity_type = "task_experience"
    target.property_name = "strategy"
    target.episode_ids = ["episode-old"]
    card_hit = MemoryDbSearchHit(memory_id=target.memory_id, score=0.99, memory=target, source="dense")
    reader = _Reader(
        memories={target.memory_id: target},
        hits=[card_hit],
        episode_hits=[current_episode],
        entities={target_entity.entity_id: target_entity, current_episode.entity_id: current_episode.entity},
    )
    pipeline, writer, _ = _pipeline(
        extraction=_episode_extraction(
            action="reuse",
            target_episode_id=current_episode.entity_id,
            value="Raise mutation rate when progress stops.",
        ),
        reader=reader,
        merge=_Merge(StructuredDecision(action="reinforce", target_id=target.memory_id)),
    )

    await pipeline.add_sync(_input(), _ctx(), add_record_id="add-cross-episode")

    plan = writer.plans[0]
    assert all(command.entity.entity_id != target_entity.entity_id for command in plan.entity_writes)
    update = next(command for command in plan.memory_updates if command.memory_id == target.memory_id)
    assert update.payload_patch["episode_ids"] == ["episode-old", "episode-current"]
    assert any(
        command.relationship.rel_type == "OBSERVED_IN"
        and command.relationship.source.node_id == target_entity.entity_id
        and command.relationship.target.node_id == "episode-current"
        for command in plan.relationship_writes
    )


@pytest.mark.asyncio
async def test_contextual_reinforce_converges_equivalent_active_candidates_and_preserves_evidence():
    from mindmemos.pipelines.add.structured.planner import StructuredDecision

    canonical = _memory("memory-canonical", "Increase mutation after stagnation.")
    duplicate = _memory("memory-duplicate", "Raise mutation when progress stalls.")
    duplicate.metadata = {
        "source_add_record_ids": ["duplicate-source"],
        "idempotency_keys": ["duplicate-event"],
        "evidence_history": [
            {"add_record_id": "duplicate-source", "idempotency_key": "duplicate-event", "metadata": {"score": 0.8}}
        ],
    }
    duplicate.episode_ids = ["episode-earlier"]
    distinct = _memory("memory-distinct", "Preserve the elite candidate unchanged.")
    hits = [
        MemoryDbSearchHit(memory_id=canonical.memory_id, score=0.97, memory=canonical),
        MemoryDbSearchHit(memory_id=duplicate.memory_id, score=0.96, memory=duplicate),
        MemoryDbSearchHit(memory_id=distinct.memory_id, score=0.91, memory=distinct),
    ]
    reader = _Reader(
        memories={item.memory_id: item for item in [canonical, duplicate, distinct]},
        hits=hits,
    )
    pipeline, writer, _ = _pipeline(
        extraction=_extraction("Increase mutation rate after several stagnant generations."),
        reader=reader,
        merge=_Merge(
            StructuredDecision(
                action="reinforce",
                target_id=canonical.memory_id,
                equivalent_ids=[duplicate.memory_id],
                reason="same strategy and task background",
            )
        ),
    )

    result = await pipeline.add_sync(_input(), _ctx(), add_record_id="add-converge")

    assert result.memories[0].operation == "reinforcement"
    plan = writer.plans[0]
    updates = {command.memory_id: command for command in plan.memory_updates}
    assert updates[canonical.memory_id].reinforcement_count_delta >= 2
    assert updates[canonical.memory_id].metadata_patch["source_add_record_ids"] == [
        "earlier",
        "duplicate-source",
        "add-converge",
    ]
    assert updates[canonical.memory_id].metadata_patch["merged_memory_ids"] == [duplicate.memory_id]
    assert updates[canonical.memory_id].payload_patch["episode_ids"] == ["episode-earlier"]
    assert updates[duplicate.memory_id].status == "archived"
    assert updates[duplicate.memory_id].metadata_patch["merged_into"] == canonical.memory_id
    assert distinct.memory_id not in updates
    assert any(
        command.relationship.rel_type == "DERIVED_FROM"
        and command.relationship.source.node_id == canonical.memory_id
        and command.relationship.target.node_id == duplicate.memory_id
        for command in plan.relationship_writes
    )


@pytest.mark.asyncio
@pytest.mark.parametrize(("action", "old_status"), [("update", "archived"), ("supersede", "superseded")])
async def test_revision_action_converges_all_selected_equivalents_into_one_revision(action, old_status):
    from mindmemos.pipelines.add.structured.planner import StructuredDecision

    canonical = _memory("memory-canonical", "Raise mutation after stagnation.")
    duplicate = _memory("memory-duplicate", "Increase mutation when progress stalls.")
    duplicate.metadata["source_add_record_ids"] = ["duplicate-source"]
    duplicate.metadata["evidence_history"] = [{"add_record_id": "duplicate-source", "metadata": {"score": 0.8}}]
    hits = [
        MemoryDbSearchHit(memory_id=canonical.memory_id, score=0.97, memory=canonical),
        MemoryDbSearchHit(memory_id=duplicate.memory_id, score=0.96, memory=duplicate),
    ]
    reader = _Reader(memories={canonical.memory_id: canonical, duplicate.memory_id: duplicate}, hits=hits)
    pipeline, writer, _ = _pipeline(
        extraction=_extraction("Use adaptive mutation with a capped rate."),
        reader=reader,
        merge=_Merge(
            StructuredDecision(
                action=action,
                target_id=canonical.memory_id,
                equivalent_ids=[duplicate.memory_id],
                merged_content="Use adaptive mutation after stagnation with a capped rate.",
                reason="new evidence consolidates both variants",
            )
        ),
    )

    await pipeline.add_sync(_input(), _ctx(), add_record_id="add-revision-cluster")

    plan = writer.plans[0]
    revision = plan.memory_writes[0].memory
    assert revision.parent_ids == [canonical.memory_id, duplicate.memory_id]
    assert revision.metadata["merged_memory_ids"] == [canonical.memory_id, duplicate.memory_id]
    assert revision.metadata["source_add_record_ids"] == ["earlier", "duplicate-source", "add-revision-cluster"]
    updates = {command.memory_id: command for command in plan.memory_updates}
    assert updates[canonical.memory_id].status == old_status
    assert updates[duplicate.memory_id].status == old_status
    derived_targets = {
        command.relationship.target.node_id
        for command in plan.relationship_writes
        if command.relationship.rel_type == "DERIVED_FROM"
    }
    assert derived_targets == {canonical.memory_id, duplicate.memory_id}


@pytest.mark.asyncio
async def test_new_episode_compares_cards_from_recalled_episode_before_creating_duplicate():
    from mindmemos.pipelines.add.structured.planner import StructuredDecision

    previous_episode = _episode_hit("episode-previous", session_id="task-1")
    historical = _memory("memory-previous", "Raise mutation after five stagnant generations.")
    historical.episode_ids = [previous_episode.entity_id]
    hit = MemoryDbSearchHit(memory_id=historical.memory_id, score=0.94, memory=historical)
    reader = _EpisodeFilteredReader(
        memories={historical.memory_id: historical},
        hits=[hit],
        episode_hits=[previous_episode],
        entities={previous_episode.entity_id: previous_episode.entity},
    )
    merge = _Merge(
        StructuredDecision(
            action="update",
            target_id=historical.memory_id,
            merged_content="Raise mutation after five stagnant generations and reset it after improvement.",
            reason="same reusable mechanism with stronger evidence from a new run",
        )
    )
    pipeline, writer, _ = _pipeline(
        extraction=_episode_extraction(
            action="create",
            value="Increase mutation after five stagnant generations, then reset after improvement.",
        ),
        reader=reader,
        merge=merge,
    )

    result = await pipeline.add_sync(_input(), _ctx(), add_record_id="add-new-run")

    assert merge.calls == 1
    assert result.memories[0].operation == "update"
    plan = writer.plans[0]
    revision = plan.memory_writes[0].memory
    assert revision.parent_ids == [historical.memory_id]
    assert previous_episode.entity_id in revision.episode_ids
    assert (
        next(command for command in plan.memory_updates if command.memory_id == historical.memory_id).status
        == "archived"
    )


@pytest.mark.asyncio
async def test_commit_recheck_reuses_episode_created_during_model_work_without_overwrite():
    from mindmemos.pipelines.add.structured.episode import structured_episode_id

    content = "user: candidate evidence"
    episode_id = structured_episode_id(_ctx(), content=content)
    existing_episode = EntityView(
        entity_id=episode_id,
        project_id="project-1",
        entity_name="Canonical concurrent Episode",
        entity_type="episodes",
        description="Canonical description committed by another request",
        user_id="user-1",
        session_id="task-1",
        created_at=datetime(2026, 1, 1, tzinfo=UTC),
    )
    reader = _Reader(entities={episode_id: existing_episode})
    pipeline, writer, _ = _pipeline(extraction=_episode_extraction(action="create"), reader=reader)

    await pipeline.add_sync(_input(), _ctx(), add_record_id="add-episode-race")

    plan = writer.plans[0]
    assert all(command.entity.entity_id != episode_id for command in plan.entity_writes)
    assert plan.memory_writes[0].memory.episode_ids == [episode_id]
    assert any(
        command.relationship.rel_type == "OBSERVED_IN" and command.relationship.target.node_id == episode_id
        for command in plan.relationship_writes
    )


@pytest.mark.asyncio
async def test_exact_content_requires_contextual_entity_resolution_before_reinforcement():
    reader = _Reader()
    merge = _Merge(StructuredDecision(action="reinforce", target_id="independent-storage-id"))
    pipeline, writer, _ = _pipeline(reader=reader, merge=merge)
    prepared = await pipeline._prepare_properties(_extraction(), _ctx())
    existing = _memory(memory_id="independent-storage-id", content=prepared[0].content)
    existing.reinforcement_count = 2
    reader.memories[existing.memory_id] = existing
    reader.hits = [MemoryDbSearchHit(memory_id=existing.memory_id, score=1.0, memory=existing)]

    result = await pipeline.add_sync(_input(), _ctx(), add_record_id="add-2")

    assert merge.calls == 1
    assert result.memories[0].operation == "reinforcement"
    plan = writer.plans[0]
    assert not plan.memory_writes
    update = plan.memory_updates[0]
    assert update.memory_id == existing.memory_id
    assert update.content is None
    assert update.reinforcement_count_delta == 1
    assert update.payload_patch["last_seen_at"]
    assert update.dedup_metadata_key == "last_reinforcement_event_id"
    assert update.optimistic_lock_token == "event-1"
    assert update.optimistic_lock_retries == 8
    assert update.metadata_patch["evidence_history"][-1]["metadata"] == {
        "score": 0.9,
        "generation": 7,
        "algorithm_id": "algo-3",
    }


@pytest.mark.asyncio
@pytest.mark.parametrize(("action", "old_status"), [("update", "archived"), ("supersede", "superseded")])
async def test_revision_actions_preserve_old_memory_and_lineage(action, old_status):
    existing = _memory()
    hit = MemoryDbSearchHit(memory_id=existing.memory_id, score=0.9, memory=existing)
    reader = _Reader(memories={existing.memory_id: existing}, hits=[hit])
    from mindmemos.pipelines.add.structured.planner import StructuredDecision

    merge = _Merge(
        StructuredDecision(
            action=action,
            target_id=existing.memory_id,
            merged_content="Improved guidance",
            reason="new evidence",
        )
    )
    pipeline, writer, _ = _pipeline(reader=reader, merge=merge)

    result = await pipeline.add_sync(_input(), _ctx(), add_record_id="add-3")

    assert result.memories[0].operation == "update"
    plan = writer.plans[0]
    assert len(plan.memory_writes) == 1
    assert plan.memory_writes[0].memory.memory_id != existing.memory_id
    assert plan.memory_writes[0].memory.parent_ids == [existing.memory_id]
    assert plan.memory_writes[0].vector.semantic_vector
    assert [item["metadata"]["score"] for item in plan.memory_writes[0].memory.metadata["evidence_history"]] == [
        0.5,
        0.9,
    ]
    assert plan.memory_updates[0].status == old_status
    assert plan.relationship_writes[-1].relationship.rel_type == "DERIVED_FROM"
    assert merge.calls == 1


@pytest.mark.asyncio
async def test_empty_extraction_records_success_without_mutation():
    pipeline, writer, recorder = _pipeline(extraction={"entities": [], "edges": []})

    result = await pipeline.add_sync(_input(), _ctx(), add_record_id="add-empty")

    assert result.status == "ok" and result.memories == []
    assert writer.plans == []
    assert recorder.completed[0][1] == "add-empty"


@pytest.mark.asyncio
async def test_revision_replay_reinforces_the_same_revision_instead_of_creating_a_second_id():
    from mindmemos.pipelines.add.structured.identity import (
        structured_memory_fingerprint,
        structured_revision_memory_id,
    )
    from mindmemos.pipelines.add.structured.planner import StructuredDecision

    old = _memory()
    old.status = "archived"
    fingerprint = structured_memory_fingerprint(
        _ctx(),
        entity_type="AlgorithmExperience",
        property_name="finding",
        content="Improved guidance",
    )
    revision_id = structured_revision_memory_id(fingerprint, "event-1")
    revision = _memory(memory_id=revision_id, content="Improved guidance")
    hit = MemoryDbSearchHit(memory_id=old.memory_id, score=0.9, memory=old)
    reader = _Reader(memories={old.memory_id: old, revision_id: revision}, hits=[hit])
    merge = _Merge(
        StructuredDecision(
            action="update",
            target_id=old.memory_id,
            merged_content="Improved guidance",
            reason="new evidence",
        )
    )
    pipeline, writer, _ = _pipeline(reader=reader, merge=merge)

    result = await pipeline.add_sync(_input(), _ctx(), add_record_id="add-replay")

    assert result.memories[0].operation == "reinforcement"
    assert writer.plans[0].memory_writes == []
    assert writer.plans[0].memory_updates[0].memory_id == revision_id


@pytest.mark.asyncio
async def test_reintroduced_archived_content_never_overwrites_the_historical_base_id():
    reader = _Reader()
    pipeline, writer, _ = _pipeline(reader=reader)
    prepared = await pipeline._prepare_properties(_extraction(), _ctx())
    archived = _memory(memory_id=prepared[0].memory_id, content=prepared[0].content)
    archived.status = "archived"
    reader.memories[archived.memory_id] = archived

    await pipeline.add_sync(_input("return-event"), _ctx(), add_record_id="add-return")

    created = writer.plans[0].memory_writes[0].memory
    assert created.memory_id != archived.memory_id
    assert reader.memories[archived.memory_id].status == "archived"


@pytest.mark.asyncio
async def test_extraction_failure_never_mutates_or_marks_completed():
    pipeline, writer, recorder = _pipeline(extraction=RuntimeError("invalid extraction"))

    with pytest.raises(RuntimeError, match="invalid extraction"):
        await pipeline.add_sync(_input(), _ctx(), add_record_id="add-failed")

    assert writer.plans == []
    assert recorder.completed == []


@pytest.mark.asyncio
async def test_stream_reports_progress_and_honors_cancellation_before_embedding_commit():
    pipeline, writer, recorder = _pipeline()
    progress = []

    async def report(stage, message, percent, data):
        progress.append((stage, percent))

    async def cancelled():
        return True

    from mindmemos.typing import AddStreamCancelled

    with pytest.raises(AddStreamCancelled) as exc_info:
        await pipeline.add_sync_stream(
            _input(),
            _ctx(),
            add_record_id="add-cancelled",
            progress=report,
            cancel_check=cancelled,
        )

    assert exc_info.value.stage == "llm_extracting"
    assert progress == [("llm_extracting", 25)]
    assert writer.plans == []
    assert recorder.completed == []


@pytest.mark.asyncio
async def test_stream_success_uses_standard_result_and_completion_stages():
    pipeline, writer, recorder = _pipeline()
    progress = []

    async def report(stage, message, percent, data):
        progress.append((stage, percent))

    result = await pipeline.add_sync_stream(
        _input(),
        _ctx(),
        add_record_id="add-stream",
        progress=report,
    )

    assert result.status == "ok"
    assert [stage for stage, _ in progress] == ["llm_extracting", "embedding", "persisting", "completed"]
    assert writer.plans and recorder.completed[0][1] == "add-stream"


@pytest.mark.asyncio
async def test_model_embedding_and_recall_are_outside_short_commit_lock():
    observations: list[tuple[str, bool]] = []
    pipeline_ref = {}

    class Extractor(_Extractor):
        async def extract(self, **kwargs):
            observations.append(("extract", any(lock.locked() for lock in pipeline_ref["pipeline"]._commit_locks)))
            return await super().extract(**kwargs)

    class Embed(_Embed):
        async def embed(self, **kwargs):
            observations.append(("embed", any(lock.locked() for lock in pipeline_ref["pipeline"]._commit_locks)))
            return await super().embed(**kwargs)

    class Reader(_Reader):
        async def search_dense(self, *args, **kwargs):
            observations.append(("recall", any(lock.locked() for lock in pipeline_ref["pipeline"]._commit_locks)))
            return await super().search_dense(*args, **kwargs)

    class Writer(_Writer):
        async def apply_mutation_plan(self, *args, **kwargs):
            observations.append(("commit", any(lock.locked() for lock in pipeline_ref["pipeline"]._commit_locks)))
            return await super().apply_mutation_plan(*args, **kwargs)

    writer = Writer()
    pipeline = StructuredAddPipeline(
        structured_add_config=StructuredAddConfig(),
        extractor=Extractor(_extraction()),
        llm_client=None,
        embed_client=Embed(),
        text_preprocessor=_Preprocessor(),
        sparse_encoder=_Sparse(),
        db_reader=Reader(),
        db_writer=writer,
        recorder=_Recorder(),
        consistency="strong",
    )
    pipeline_ref["pipeline"] = pipeline

    await pipeline.add_sync(_input(), _ctx(), add_record_id="add-lock")

    assert observations[-1] == ("commit", True)
    assert all(not locked for stage, locked in observations if stage in {"extract", "embed"})
    assert ("recall", False) in observations
    assert ("recall", True) in observations


@pytest.mark.asyncio
async def test_five_concurrent_identical_candidates_overlap_extraction_and_converge_to_one_id():
    class ConcurrentExtractor(_Extractor):
        def __init__(self):
            super().__init__(_extraction())
            self.active = 0
            self.max_active = 0
            self.ready = asyncio.Event()

        async def extract(self, **kwargs):
            self.calls += 1
            self.active += 1
            self.max_active = max(self.max_active, self.active)
            if self.active == 5:
                self.ready.set()
            await asyncio.wait_for(self.ready.wait(), timeout=1)
            self.active -= 1
            return self.value

    extractor = ConcurrentExtractor()
    reader = _Reader()

    class StatefulWriter(_Writer):
        async def apply_mutation_plan(self, ctx, plan, *, consistency):
            result = await super().apply_mutation_plan(ctx, plan, consistency=consistency)
            for command in plan.entity_writes:
                stored_entity = EntityView.model_validate(command.entity.model_dump())
                reader.entities[stored_entity.entity_id] = stored_entity
            for command in plan.memory_writes:
                stored = MemoryView.model_validate(command.memory.model_dump())
                reader.memories[stored.memory_id] = stored
            reader.hits = [
                MemoryDbSearchHit(memory_id=memory.memory_id, score=1.0, memory=memory)
                for memory in reader.memories.values()
                if memory.status == "active"
            ]
            return result

    writer = StatefulWriter()
    pipeline = StructuredAddPipeline(
        structured_add_config=StructuredAddConfig(),
        extractor=extractor,
        llm_client=None,
        embed_client=_Embed(),
        text_preprocessor=_Preprocessor(),
        sparse_encoder=_Sparse(),
        db_reader=reader,
        db_writer=writer,
        recorder=_Recorder(),
        consistency="strong",
    )

    await asyncio.gather(
        *(pipeline.add_sync(_input(f"event-{index}"), _ctx(), add_record_id=f"add-{index}") for index in range(5))
    )

    memory_ids = {command.memory.memory_id for plan in writer.plans for command in plan.memory_writes}
    assert extractor.calls == extractor.max_active == 5
    assert len(memory_ids) == 1


def test_cached_pipeline_resolves_model_clients_per_request_without_cross_user_state(monkeypatch):
    import mindmemos.pipelines.add.structured.pipeline as pipeline_module

    llms = [object(), object()]
    embeds = [_Embed(), _Embed()]
    monkeypatch.setattr(pipeline_module, "require_model_endpoint", lambda _capability: None)
    monkeypatch.setattr(pipeline_module, "get_llm_client", lambda: llms.pop(0))
    monkeypatch.setattr(pipeline_module, "get_embed_client", lambda: embeds.pop(0))
    pipeline = StructuredAddPipeline(
        structured_add_config=StructuredAddConfig(),
        extractor=_Extractor(_extraction()),
        text_preprocessor=_Preprocessor(),
        sparse_encoder=_Sparse(),
        db_reader=_Reader(),
        db_writer=_Writer(),
        recorder=_Recorder(),
    )

    _, first_merge, first_embed = pipeline._runtime(_ctx())
    second_context = _ctx().model_copy(update={"user_id": "user-2", "request_id": "request-2"})
    _, second_merge, second_embed = pipeline._runtime(second_context)

    assert first_merge._llm is not second_merge._llm
    assert first_embed is not second_embed


@pytest.mark.asyncio
async def test_separate_worker_pipeline_instances_share_extract_limit_and_commit_stripe():
    from mindmemos.config import StructuredConcurrencyConfig

    class SharedExtractor(_Extractor):
        def __init__(self):
            super().__init__(_extraction())
            self.active = 0
            self.max_active = 0

        async def extract(self, **kwargs):
            self.active += 1
            self.max_active = max(self.max_active, self.active)
            await asyncio.sleep(0.01)
            self.active -= 1
            return self.value

    class SharedWriter(_Writer):
        def __init__(self):
            super().__init__()
            self.active = 0
            self.max_active = 0

        async def apply_mutation_plan(self, *args, **kwargs):
            self.active += 1
            self.max_active = max(self.max_active, self.active)
            await asyncio.sleep(0.01)
            result = await super().apply_mutation_plan(*args, **kwargs)
            self.active -= 1
            return result

    config = StructuredAddConfig(concurrency=StructuredConcurrencyConfig(max_extract_concurrency=1, lock_stripes=8))
    extractor = SharedExtractor()
    writer = SharedWriter()

    def build():
        return StructuredAddPipeline(
            structured_add_config=config,
            extractor=extractor,
            llm_client=None,
            embed_client=_Embed(),
            text_preprocessor=_Preprocessor(),
            sparse_encoder=_Sparse(),
            db_reader=_Reader(),
            db_writer=writer,
            recorder=_Recorder(),
            consistency="strong",
        )

    first, second = build(), build()
    await asyncio.gather(
        first.add_sync(_input("worker-1"), _ctx(), add_record_id="worker-add-1"),
        second.add_sync(_input("worker-2"), _ctx(), add_record_id="worker-add-2"),
    )

    assert extractor.max_active == 1
    assert writer.max_active == 1
