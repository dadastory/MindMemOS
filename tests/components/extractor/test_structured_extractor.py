import json
from types import SimpleNamespace

import pytest
from mindmemos.components.extractor.structured import StructuredExtractionError, StructuredExtractor
from mindmemos.config import StructuredExtractionConfig
from mindmemos.llm import ChatResponse
from mindmemos.pipelines.add.structured.episode import StructuredEpisodeCandidate


class FakeEntityManager:
    file_path = SimpleNamespace(name="task-schema.json")

    def get_all_dicts(self):
        return [
            {
                "entity_type": "task_experience",
                "entity_description": "Reusable task experience.",
                "static_property": {},
                "dynamic_property": {
                    "strategy": {"desc": "The reusable algorithm or approach.", "order": 1},
                    "error": {"desc": "Observed error evidence.", "order": 1},
                    "summary": {"desc": "Higher order summary.", "order": 2},
                },
            },
            {
                "entity_type": "episodes",
                "entity_description": "Conversation episodes.",
                "static_property": {},
                "dynamic_property": {"input_messages": {"order": 1}},
            },
        ]

    def list_types(self):
        return ["task_experience", "episodes"]


class QueueLLM:
    def __init__(self, *responses: str):
        self.responses = list(responses)
        self.calls = []

    async def chat(self, task, messages, format_parser=None, **kwargs):
        self.calls.append({"task": task, "messages": messages, "format_parser": format_parser})
        content = self.responses.pop(0)
        return ChatResponse(finish_reason="stop", content=content)


def valid_payload(*, entities=None, edges=None, episode=None) -> str:
    payload = {
        "entities": entities
        if entities is not None
        else [
            {
                "name": "Mutation strategy",
                "entity_type": "task_experience",
                "description": "A reusable mutation strategy.",
                "properties": [
                    {
                        "property_name": "strategy",
                        "value": "Prefer bounded mutation with elitism.",
                    }
                ],
            }
        ],
        "edges": edges or [],
    }
    if episode is not None:
        payload["episode"] = episode
    return json.dumps(payload, ensure_ascii=False)


def episode_candidate(episode_id="episode-1"):
    return StructuredEpisodeCandidate(
        episode_id=episode_id,
        title="Existing task",
        description="Existing task background",
        score=0.93,
        session_id="session-1",
        same_session=True,
    )


@pytest.mark.asyncio
async def test_structured_extractor_uses_one_fixed_schema_call_and_normalizes_output():
    llm = QueueLLM(valid_payload())
    extractor = StructuredExtractor(
        llm_client=llm,
        entity_manager=FakeEntityManager(),
        config=StructuredExtractionConfig(),
    )

    result = await extractor.extract(
        content="score=0.91; mutation strategy improved the solution",
        event_time="2026-08-03T08:00:00+00:00",
        prompt_language="EN",
    )

    assert len(llm.calls) == 1
    assert llm.calls[0]["task"] == "memory.add.structured_extract"
    assert llm.calls[0]["format_parser"] is None
    prompt = llm.calls[0]["messages"][0]["content"]
    assert "task_experience" in prompt
    assert '"entity_type": "episodes"' not in prompt
    assert '"summary"' not in prompt
    assert result["entities"][0]["properties"][0]["operation"] == "set"
    assert result["entities"][0]["properties"][0]["time"] == "2026-08-03"


@pytest.mark.asyncio
async def test_structured_extractor_accepts_zero_entities_without_repair():
    llm = QueueLLM(valid_payload(entities=[]))
    extractor = StructuredExtractor(
        llm_client=llm,
        entity_manager=FakeEntityManager(),
        config=StructuredExtractionConfig(),
    )

    result = await extractor.extract(content="nothing durable", event_time="2026-08-03", prompt_language=None)

    assert result == {"entities": [], "edges": []}
    assert len(llm.calls) == 1


@pytest.mark.asyncio
async def test_structured_extractor_prompt_supports_propertyless_entities_and_explicit_edges():
    llm = QueueLLM(valid_payload(entities=[]))
    extractor = StructuredExtractor(
        llm_client=llm,
        entity_manager=FakeEntityManager(),
        config=StructuredExtractionConfig(),
    )

    await extractor.extract(content="device A caused ticket B", event_time="2026-08-04", prompt_language="EN")

    prompt = llm.calls[0]["messages"][0]["content"]
    assert "empty properties only when" in prompt
    assert "endpoint of an explicit relationship" in prompt
    assert "durable fact maps to a Schema property" in prompt
    assert "explicit relationship stated in the source" in prompt
    assert "link_entity1_name" in prompt
    assert "link_entity2_name" in prompt
    assert "link_description" in prompt


@pytest.mark.asyncio
async def test_structured_extractor_repairs_standalone_propertyless_entities_into_typed_properties():
    propertyless = valid_payload(
        entities=[
            {
                "name": "Mutation strategy",
                "entity_type": "task_experience",
                "description": "A reusable mutation strategy.",
                "properties": [],
            }
        ]
    )
    llm = QueueLLM(propertyless, valid_payload())
    extractor = StructuredExtractor(
        llm_client=llm,
        entity_manager=FakeEntityManager(),
        config=StructuredExtractionConfig(max_repair_attempts=1),
    )

    result = await extractor.extract(
        content="A bounded mutation strategy improved the result.",
        event_time="2026-08-04",
        prompt_language="EN",
    )

    assert len(llm.calls) == 2
    assert result["entities"][0]["properties"][0]["property_name"] == "strategy"
    assert "all extracted entities are propertyless" in llm.calls[1]["messages"][0]["content"]


@pytest.mark.asyncio
async def test_structured_extractor_rejects_standalone_propertyless_entities_after_repair():
    propertyless = valid_payload(
        entities=[
            {
                "name": "Mutation strategy",
                "entity_type": "task_experience",
                "description": "A reusable mutation strategy.",
                "properties": [],
            }
        ]
    )
    llm = QueueLLM(propertyless, propertyless)
    extractor = StructuredExtractor(
        llm_client=llm,
        entity_manager=FakeEntityManager(),
        config=StructuredExtractionConfig(max_repair_attempts=1),
    )

    with pytest.raises(StructuredExtractionError, match="propertyless"):
        await extractor.extract(
            content="A bounded mutation strategy improved the result.",
            event_time="2026-08-04",
            prompt_language="EN",
        )

    assert len(llm.calls) == 2


@pytest.mark.asyncio
async def test_structured_extractor_accepts_confirmed_relation_only_result_after_repair():
    entities = [
        {"name": "Device A", "entity_type": "task_experience", "description": "Source device", "properties": []},
        {"name": "Ticket B", "entity_type": "task_experience", "description": "Affected ticket", "properties": []},
    ]
    edges = [
        {
            "link_entity1_name": "Device A",
            "link_entity2_name": "Ticket B",
            "link_description": "Device A caused Ticket B",
        }
    ]
    relation_only = valid_payload(entities=entities, edges=edges)
    llm = QueueLLM(relation_only, relation_only)
    extractor = StructuredExtractor(
        llm_client=llm,
        entity_manager=FakeEntityManager(),
        config=StructuredExtractionConfig(max_repair_attempts=1),
    )

    result = await extractor.extract(
        content="Device A caused Ticket B.",
        event_time="2026-08-04",
        prompt_language="EN",
    )

    assert len(llm.calls) == 2
    assert result["entities"] == [
        {
            "name": "Device A",
            "entity_type": "task_experience",
            "description": "Source device",
            "record_time": "2026-08-04",
            "properties": [],
        },
        {
            "name": "Ticket B",
            "entity_type": "task_experience",
            "description": "Affected ticket",
            "record_time": "2026-08-04",
            "properties": [],
        },
    ]
    assert result["edges"] == edges


@pytest.mark.asyncio
async def test_structured_extractor_rejects_relation_only_result_with_orphan_entity():
    entities = [
        {"name": "Device A", "entity_type": "task_experience", "properties": []},
        {"name": "Ticket B", "entity_type": "task_experience", "properties": []},
        {"name": "Orphan C", "entity_type": "task_experience", "properties": []},
    ]
    edges = [{"link_entity1_name": "Device A", "link_entity2_name": "Ticket B", "link_description": "caused"}]
    relation_only = valid_payload(entities=entities, edges=edges)
    llm = QueueLLM(relation_only, relation_only)
    extractor = StructuredExtractor(
        llm_client=llm,
        entity_manager=FakeEntityManager(),
        config=StructuredExtractionConfig(max_repair_attempts=1),
    )

    with pytest.raises(StructuredExtractionError, match="every propertyless entity must be referenced"):
        await extractor.extract(content="Device A caused Ticket B.", event_time="2026-08-04", prompt_language="EN")

    assert len(llm.calls) == 2


@pytest.mark.asyncio
async def test_structured_extractor_reuses_only_a_whitelisted_episode_in_the_same_call():
    llm = QueueLLM(
        valid_payload(
            episode={
                "action": "reuse",
                "target_episode_id": "episode-1",
                "title": "Existing task",
                "description": "Existing task background plus new evidence",
                "related_episode_ids": [],
            }
        )
    )
    extractor = StructuredExtractor(
        llm_client=llm,
        entity_manager=FakeEntityManager(),
        config=StructuredExtractionConfig(),
    )

    result = await extractor.extract(
        content="continuing evidence",
        event_time="2026-08-03",
        prompt_language="EN",
        episode_candidates=[episode_candidate()],
    )

    assert len(llm.calls) == 1
    assert result["episode"]["action"] == "reuse"
    assert result["episode"]["target_episode_id"] == "episode-1"
    prompt = llm.calls[0]["messages"][0]["content"]
    assert "episode-1" in prompt
    assert "decide whether to reuse one supplied Episode or create a new Episode" in prompt


@pytest.mark.asyncio
async def test_structured_extractor_creates_episode_when_candidate_list_is_empty():
    llm = QueueLLM(
        valid_payload(
            episode={
                "action": "create",
                "target_episode_id": None,
                "title": "New experiment",
                "description": "An independent experiment",
                "related_episode_ids": [],
            }
        )
    )
    extractor = StructuredExtractor(
        llm_client=llm,
        entity_manager=FakeEntityManager(),
        config=StructuredExtractionConfig(),
    )

    result = await extractor.extract(
        content="new experiment evidence",
        event_time="2026-08-03",
        prompt_language=None,
        episode_candidates=[],
    )

    assert result["episode"]["action"] == "create"
    assert result["episode"]["target_episode_id"] is None


@pytest.mark.asyncio
async def test_structured_extractor_repairs_episode_target_outside_whitelist_once():
    invalid = valid_payload(
        episode={
            "action": "reuse",
            "target_episode_id": "episode-not-supplied",
            "title": "Bad target",
            "description": "Bad target",
        }
    )
    repaired = valid_payload(
        episode={
            "action": "reuse",
            "target_episode_id": "episode-1",
            "title": "Existing task",
            "description": "Existing task background",
        }
    )
    llm = QueueLLM(invalid, repaired)
    extractor = StructuredExtractor(
        llm_client=llm,
        entity_manager=FakeEntityManager(),
        config=StructuredExtractionConfig(max_repair_attempts=1),
    )

    result = await extractor.extract(
        content="continuing evidence",
        event_time="2026-08-03",
        prompt_language="EN",
        episode_candidates=[episode_candidate()],
    )

    assert result["episode"]["target_episode_id"] == "episode-1"
    assert len(llm.calls) == 2
    assert "not one of the supplied candidates" in llm.calls[1]["messages"][0]["content"]


@pytest.mark.asyncio
async def test_structured_extractor_repairs_once_without_hidden_parser_retries():
    llm = QueueLLM("not-json", valid_payload())
    extractor = StructuredExtractor(
        llm_client=llm,
        entity_manager=FakeEntityManager(),
        config=StructuredExtractionConfig(max_repair_attempts=1),
    )

    result = await extractor.extract(content="candidate", event_time="2026-08-03", prompt_language="EN")

    assert result["entities"][0]["entity_type"] == "task_experience"
    assert len(llm.calls) == 2
    assert all(call["format_parser"] is None for call in llm.calls)
    assert "Validation error" in llm.calls[1]["messages"][0]["content"]


@pytest.mark.asyncio
async def test_structured_extractor_fails_after_one_invalid_repair():
    llm = QueueLLM("not-json", '{"entities": [{"name": "x", "entity_type": "unknown"}], "edges": []}')
    extractor = StructuredExtractor(
        llm_client=llm,
        entity_manager=FakeEntityManager(),
        config=StructuredExtractionConfig(max_repair_attempts=1),
    )

    with pytest.raises(StructuredExtractionError) as exc_info:
        await extractor.extract(content="candidate", event_time="2026-08-03", prompt_language="EN")

    assert exc_info.value.code == "structured.extraction_invalid"
    assert len(llm.calls) == 2


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "entities,edges,error_fragment",
    [
        ([{"name": "x", "entity_type": "unknown", "properties": []}], [], "unknown entity_type"),
        (
            [
                {
                    "name": "x",
                    "entity_type": "task_experience",
                    "properties": [{"property_name": "unknown", "value": "v"}],
                }
            ],
            [],
            "unknown property_name",
        ),
        (
            [
                {"name": "x", "entity_type": "task_experience", "properties": []},
                {"name": "x", "entity_type": "task_experience", "properties": []},
            ],
            [],
            "duplicate entity name",
        ),
        ([{"name": "", "entity_type": "task_experience", "properties": []}], [], "non-empty name"),
        (
            [
                {
                    "name": "x",
                    "entity_type": "task_experience",
                    "properties": [{"property_name": "strategy"}],
                }
            ],
            [],
            "value",
        ),
        (
            [{"name": "x", "entity_type": "task_experience", "properties": []}],
            [{"link_entity1_name": "x", "link_entity2_name": "missing"}],
            "unknown entity",
        ),
    ],
)
async def test_structured_extractor_strictly_rejects_schema_mismatches(entities, edges, error_fragment):
    llm = QueueLLM(valid_payload(entities=entities, edges=edges))
    extractor = StructuredExtractor(
        llm_client=llm,
        entity_manager=FakeEntityManager(),
        config=StructuredExtractionConfig(max_repair_attempts=0),
    )

    with pytest.raises(StructuredExtractionError, match=error_fragment):
        await extractor.extract(content="candidate", event_time="2026-08-03", prompt_language="EN")
