import json
from types import SimpleNamespace

import pytest
from mindmemos.components.extractor.structured import StructuredExtractionError, StructuredExtractor
from mindmemos.components.extractor.structured.evidence import inventory_source_artifacts
from mindmemos.components.extractor.structured.extractor import _validate_coverage_result
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


class FakeLlm4adEntityManager:
    file_path = SimpleNamespace(name="llm4ad-memory-card.json")

    def get_all_dicts(self):
        return [
            {
                "entity_type": "llm4ad_memory_card",
                "static_property": {"name": "Stable card title"},
                "dynamic_property": {
                    "good_algorithm": {"type": "object", "format": "structured_card_content", "order": 1},
                    "error_reflection": {"type": "object", "format": "structured_card_content", "order": 1},
                    "domain_knowledge": {"type": "object", "format": "structured_card_content", "order": 1},
                    "general_insight": {"type": "object", "format": "structured_card_content", "order": 1},
                    "tags": {"type": "string", "order": 1},
                },
            }
        ]


class FakeExplicitSubjectEntityManager(FakeLlm4adEntityManager):
    def get_all_dicts(self):
        schema = super().get_all_dicts()
        for definition in schema[0]["dynamic_property"].values():
            if isinstance(definition, dict) and definition.get("format") == "structured_card_content":
                definition["fact_contract"] = "explicit_subject"
        return schema


class FakeObjectEntityManager:
    file_path = SimpleNamespace(name="object-schema.json")

    def get_all_dicts(self):
        return [
            {
                "entity_type": "experiment",
                "static_property": {},
                "dynamic_property": {
                    "measurements": {"type": "object", "order": 1},
                },
            }
        ]


class FakePolicyEntityManager:
    file_path = SimpleNamespace(name="policy-schema.json")

    def get_all_dicts(self):
        return [
            {
                "entity_type": "task_experience",
                "entity_description": "Only retain facts useful for future optimization.",
                "entity_instruction": "Reject observations that are not reusable or successful.",
                "static_property": {},
                "dynamic_property": {
                    "strategy": {
                        "type": "object",
                        "format": "structured_card_content",
                        "desc": "Only retain successful reusable strategies.",
                        "order": 1,
                    },
                    "error": {
                        "type": "string",
                        "desc": "Only retain actionable failures.",
                        "order": 1,
                    },
                },
            }
        ]


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


def test_coverage_with_no_missing_evidence_is_deterministically_complete():
    assert (
        _validate_coverage_result(
            {"complete": False, "missing_evidence": []},
            source="The final increment must equal 1.",
            max_items=12,
        )
        == []
    )


def test_source_artifact_inventory_preserves_fenced_code_formula_table_and_quote_exactly():
    source = (
        "Intro.\n\n```cpp\nfor (int i = 0; i < n; ++i) {\n    run(i);\n}\n```\n\n"
        "$$\nT(n) = O(n^2)\n$$\n\n"
        "| gap | score |\n| --- | --- |\n| 1 | 0.91 |\n\n"
        "> Keep the final increment equal to 1.\n"
    )

    artifacts = inventory_source_artifacts(source)

    assert [item["type"] for item in artifacts] == ["code", "formula", "table", "quote"]
    assert artifacts[0]["content"] == "for (int i = 0; i < n; ++i) {\n    run(i);\n}"
    assert artifacts[0]["language"] == "cpp"
    assert artifacts[1]["content"] == "T(n) = O(n^2)"
    assert artifacts[2]["content"] == "| gap | score |\n| --- | --- |\n| 1 | 0.91 |"
    assert artifacts[3]["content"] == "> Keep the final increment equal to 1."


@pytest.mark.asyncio
async def test_structured_extractor_materializes_code_refs_from_source_without_model_rewriting():
    source = (
        "# Shell sort\n\n"
        "```cpp\nVoid Shellinsert (Alist&L, int k) f\n"
        "    for(i = dk + 1; i <= L.length; ++ i){\n"
        "        L.r[0] = L.r[i];\n"
        "    }\n```"
    )
    llm = QueueLLM(
        valid_payload(
            entities=[
                {
                    "name": "ShellInsert implementation",
                    "entity_type": "llm4ad_memory_card",
                    "description": "Source implementation.",
                    "properties": [
                        {
                            "property_name": "good_algorithm",
                            "value": {
                                "description": "ShellInsert gapped insertion implementation.",
                                "content": ["ShellInsert performs one gapped insertion pass."],
                                "artifact_refs": ["artifact-1"],
                            },
                        }
                    ],
                }
            ]
        )
    )
    extractor = StructuredExtractor(
        llm_client=llm,
        entity_manager=FakeLlm4adEntityManager(),
        config=StructuredExtractionConfig(max_repair_attempts=0),
    )

    result = await extractor.extract(content=source, event_time="2026-08-19", prompt_language="EN")

    prompt = llm.calls[0]["messages"][0]["content"]
    assert '"artifact_id": "artifact-1"' in prompt
    assert "artifact_refs" in prompt
    value = result["entities"][0]["properties"][0]["value"]
    assert value["content"] == ["ShellInsert performs one gapped insertion pass."]
    assert value["artifacts"][0]["content"] == (
        "Void Shellinsert (Alist&L, int k) f\n"
        "    for(i = dk + 1; i <= L.length; ++ i){\n"
        "        L.r[0] = L.r[i];\n"
        "    }"
    )


@pytest.mark.asyncio
async def test_structured_extractor_repairs_a_card_that_omits_an_inventoried_source_artifact():
    source = "```cpp\nShellInsert(L, dlta[k]);\n```"
    omitted = valid_payload(
        entities=[
            {
                "name": "ShellSort",
                "entity_type": "llm4ad_memory_card",
                "properties": [
                    {
                        "property_name": "good_algorithm",
                        "value": {
                            "description": "ShellSort implementation.",
                            "content": ["ShellSort invokes ShellInsert for each increment."],
                        },
                    }
                ],
            }
        ]
    )
    repaired = valid_payload(
        entities=[
            {
                "name": "ShellSort",
                "entity_type": "llm4ad_memory_card",
                "properties": [
                    {
                        "property_name": "good_algorithm",
                        "value": {
                            "description": "ShellSort implementation.",
                            "content": ["ShellSort invokes ShellInsert for each increment."],
                            "artifact_refs": ["artifact-1"],
                        },
                    }
                ],
            }
        ]
    )
    llm = QueueLLM(omitted, repaired)
    extractor = StructuredExtractor(
        llm_client=llm,
        entity_manager=FakeLlm4adEntityManager(),
        config=StructuredExtractionConfig(max_repair_attempts=1),
    )

    result = await extractor.extract(content=source, event_time="2026-08-19", prompt_language="EN")

    assert len(llm.calls) == 2
    assert "artifact-1" in llm.calls[1]["messages"][0]["content"]
    assert result["entities"][0]["properties"][0]["value"]["artifacts"][0]["content"] == ("ShellInsert(L, dlta[k]);")


@pytest.mark.asyncio
async def test_structured_extractor_repairs_source_local_references_in_card_facts():
    bad = valid_payload(
        entities=[
            {
                "name": "Equipment calibration report",
                "entity_type": "llm4ad_memory_card",
                "properties": [
                    {
                        "property_name": "good_algorithm",
                        "value": {
                            "description": "Maintenance and performance findings.",
                            "content": [
                                "第3节规定上述设备必须在启动前完成校准。",
                                "Table 2 shows that the latter configuration reduced latency.",
                            ],
                        },
                    }
                ],
            }
        ]
    )
    repaired = valid_payload(
        entities=[
            {
                "name": "Equipment calibration report",
                "entity_type": "llm4ad_memory_card",
                "properties": [
                    {
                        "property_name": "good_algorithm",
                        "value": {
                            "description": "Maintenance and performance findings.",
                            "content": [
                                {
                                    "subject": "Sensor X",
                                    "statement": "Sensor X must be calibrated before startup.",
                                },
                                {
                                    "subject": "Configuration B",
                                    "statement": "Configuration B reduced median latency from 80 ms to 52 ms.",
                                },
                            ],
                        },
                    }
                ],
            }
        ]
    )
    llm = QueueLLM(bad, repaired)
    extractor = StructuredExtractor(
        llm_client=llm,
        entity_manager=FakeExplicitSubjectEntityManager(),
        config=StructuredExtractionConfig(max_repair_attempts=1),
    )

    result = await extractor.extract(
        content=(
            "Sensor X must be calibrated before startup. Configuration B reduced median latency from 80 ms to 52 ms."
        ),
        event_time="2026-08-19",
        prompt_language="ZH",
    )

    assert len(llm.calls) == 2
    assert "must contain exactly subject and statement" in llm.calls[1]["messages"][0]["content"]
    assert result["entities"][0]["properties"][0]["value"]["content"] == [
        "Sensor X must be calibrated before startup.",
        "Configuration B reduced median latency from 80 ms to 52 ms.",
    ]


@pytest.mark.asyncio
async def test_structured_extractor_materializes_explicit_subject_facts_across_languages():
    payload = valid_payload(
        entities=[
            {
                "name": "Deployment guidance",
                "entity_type": "llm4ad_memory_card",
                "properties": [
                    {
                        "property_name": "general_insight",
                        "value": {
                            "description": "Configuration B is selected for deployment.",
                            "content": [
                                {"subject": "配置 B", "statement": "配置 B 因此在最后一个阶段使用。"},
                                {"subject": "设备", "statement": "设备应该在启动前完成校准。"},
                                {"subject": "两个执行阶段", "statement": "两个执行阶段彼此独立。"},
                                {
                                    "subject": "la configuración B",
                                    "statement": "La configuración B reduce la latencia mediana.",
                                },
                            ],
                        },
                    }
                ],
            }
        ]
    )
    llm = QueueLLM(payload)
    extractor = StructuredExtractor(
        llm_client=llm,
        entity_manager=FakeExplicitSubjectEntityManager(),
        config=StructuredExtractionConfig(max_repair_attempts=1),
    )

    result = await extractor.extract(content="candidate", event_time="2026-08-19", prompt_language="ZH")

    assert len(llm.calls) == 1
    prompt = llm.calls[0]["messages"][0]["content"]
    assert "applies equally to every source and output language" in prompt
    assert "Every fact object must contain exactly subject and statement" in prompt
    assert result["entities"][0]["properties"][0]["value"]["content"] == [
        "配置 B 因此在最后一个阶段使用。",
        "设备应该在启动前完成校准。",
        "两个执行阶段彼此独立。",
        "La configuración B reduce la latencia mediana.",
    ]


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
    assert "one already-selected document block" in prompt
    assert "zero, one, or many entities and properties" in prompt
    assert "source-grounded entities, atomic facts" in prompt
    assert "whether it matches history" in prompt
    assert "one independently understandable claim" in prompt
    assert "Do not generalize" in prompt
    assert "generated entity name is display text only" in prompt
    assert result["entities"][0]["properties"][0]["operation"] == "set"
    assert result["entities"][0]["properties"][0]["time"] == "2026-08-03"


@pytest.mark.asyncio
async def test_structured_extractor_unwraps_schema_property_envelope_instead_of_storing_json():
    wrapped_value = {
        "dynamic_property": {
            "good_algorithm": {
                "description": "A reusable Shell-sort increment strategy.",
                "content": ["Use a diminishing increment sequence before the final insertion pass."],
            },
            "tags": "Shell sort, insertion sort",
        }
    }
    llm = QueueLLM(
        valid_payload(
            entities=[
                {
                    "name": "Shell sort increment strategy",
                    "entity_type": "llm4ad_memory_card",
                    "description": "Reusable Shell sort guidance.",
                    "properties": [
                        {
                            "property_name": "good_algorithm",
                            # Reproduce providers that stringify a Python-style
                            # Schema wrapper inside the value field.
                            "value": repr(wrapped_value),
                        }
                    ],
                }
            ]
        )
    )
    extractor = StructuredExtractor(
        llm_client=llm,
        entity_manager=FakeLlm4adEntityManager(),
        config=StructuredExtractionConfig(max_repair_attempts=0),
    )

    result = await extractor.extract(content="candidate", event_time="2026-08-17", prompt_language="EN")

    assert result["entities"][0]["name"] == "Shell sort increment strategy"
    assert result["entities"][0]["properties"] == [
        {
            "property_name": "good_algorithm",
            "value": {
                "description": "A reusable Shell-sort increment strategy.",
                "content": ["Use a diminishing increment sequence before the final insertion pass."],
            },
            "time": "2026-08-17",
            "operation": "set",
        },
        {
            "property_name": "tags",
            "value": "Shell sort, insertion sort",
            "time": "2026-08-17",
            "operation": "set",
        },
    ]


@pytest.mark.asyncio
async def test_structured_extractor_preserves_validated_llm4ad_card_object():
    llm = QueueLLM(
        valid_payload(
            entities=[
                {
                    "name": "Shell sort",
                    "entity_type": "llm4ad_memory_card",
                    "properties": [
                        {
                            "property_name": "domain_knowledge",
                            "value": {
                                "description": " Shell sort complexity. ",
                                "content": [
                                    " The final increment must be 1. ",
                                    "The final increment must be 1.",
                                    "Increment choice controls the observed complexity.",
                                ],
                            },
                        }
                    ],
                }
            ]
        )
    )
    extractor = StructuredExtractor(
        llm_client=llm,
        entity_manager=FakeLlm4adEntityManager(),
        config=StructuredExtractionConfig(max_repair_attempts=0),
    )

    result = await extractor.extract(content="candidate", event_time="2026-08-17", prompt_language="EN")

    assert result["entities"][0]["properties"][0]["value"] == {
        "description": "Shell sort complexity.",
        "content": [
            "The final increment must be 1.",
            "Increment choice controls the observed complexity.",
        ],
    }


@pytest.mark.asyncio
async def test_structured_extractor_enforces_caller_selected_schema_properties():
    wrong_type = valid_payload(
        entities=[
            {
                "name": "Shell sort",
                "entity_type": "llm4ad_memory_card",
                "properties": [
                    {
                        "property_name": "domain_knowledge",
                        "value": {
                            "description": "Shell sort background.",
                            "content": ["The final increment must be 1."],
                        },
                    }
                ],
            }
        ]
    )
    repaired = valid_payload(
        entities=[
            {
                "name": "Shell sort",
                "entity_type": "llm4ad_memory_card",
                "properties": [
                    {
                        "property_name": "good_algorithm",
                        "value": {
                            "description": "A reusable Shell-sort strategy.",
                            "content": ["The final increment must be 1."],
                        },
                    },
                    {"property_name": "tags", "value": "Shell sort"},
                ],
            }
        ]
    )
    llm = QueueLLM(wrong_type, repaired)
    extractor = StructuredExtractor(
        llm_client=llm,
        entity_manager=FakeLlm4adEntityManager(),
        config=StructuredExtractionConfig(max_repair_attempts=1),
    )

    result = await extractor.extract(
        content="candidate",
        event_time="2026-08-17",
        prompt_language="EN",
        allowed_property_names={"good_algorithm", "tags"},
    )

    assert len(llm.calls) == 2
    assert "good_algorithm" in llm.calls[0]["messages"][0]["content"]
    assert "domain_knowledge" not in {
        prop["property_name"] for entity in result["entities"] for prop in entity["properties"]
    }
    assert [prop["property_name"] for prop in result["entities"][0]["properties"]] == [
        "good_algorithm",
        "tags",
    ]


@pytest.mark.asyncio
async def test_structured_extractor_repairs_malformed_llm4ad_card_object():
    invalid = valid_payload(
        entities=[
            {
                "name": "Shell sort",
                "entity_type": "llm4ad_memory_card",
                "properties": [
                    {
                        "property_name": "domain_knowledge",
                        "value": {"description": "Shell sort", "content": "not-an-array"},
                    }
                ],
            }
        ]
    )
    repaired = valid_payload(
        entities=[
            {
                "name": "Shell sort",
                "entity_type": "llm4ad_memory_card",
                "properties": [
                    {
                        "property_name": "domain_knowledge",
                        "value": {
                            "description": "Shell sort uses diminishing increments.",
                            "content": ["Its complexity depends on the selected increment sequence."],
                        },
                    }
                ],
            }
        ]
    )
    llm = QueueLLM(invalid, repaired)
    extractor = StructuredExtractor(
        llm_client=llm,
        entity_manager=FakeLlm4adEntityManager(),
        config=StructuredExtractionConfig(max_repair_attempts=1),
    )

    result = await extractor.extract(content="candidate", event_time="2026-08-17", prompt_language="EN")

    assert len(llm.calls) == 2
    assert "content" in llm.calls[1]["messages"][0]["content"]
    assert result["entities"][0]["properties"][0]["value"]["description"].startswith("Shell sort uses")


@pytest.mark.asyncio
async def test_structured_extractor_preserves_object_value_when_schema_declares_object():
    llm = QueueLLM(
        valid_payload(
            entities=[
                {
                    "name": "Experiment 7",
                    "entity_type": "experiment",
                    "properties": [
                        {
                            "property_name": "measurements",
                            "value": {"score": 0.91, "constraints": ["stable", "bounded"]},
                        }
                    ],
                }
            ]
        )
    )
    extractor = StructuredExtractor(
        llm_client=llm,
        entity_manager=FakeObjectEntityManager(),
        config=StructuredExtractionConfig(max_repair_attempts=0),
    )

    result = await extractor.extract(content="candidate", event_time="2026-08-17", prompt_language="EN")

    assert result["entities"][0]["properties"][0]["value"] == {
        "score": 0.91,
        "constraints": ["stable", "bounded"],
    }


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
async def test_structured_extractor_prompt_lists_exact_schema_keys_for_model_output():
    llm = QueueLLM(valid_payload(entities=[]))
    extractor = StructuredExtractor(
        llm_client=llm,
        entity_manager=FakeEntityManager(),
        config=StructuredExtractionConfig(),
    )

    await extractor.extract(content="candidate", event_time="2026-08-04", prompt_language="EN")

    prompt = llm.calls[0]["messages"][0]["content"]
    assert '"task_experience": ["error", "strategy"]' in prompt
    assert "property_name must be copied exactly from this key list" in prompt


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


@pytest.mark.asyncio
async def test_caller_selection_skips_eligibility_call_and_extraction_does_not_receive_policy_prose():
    llm = QueueLLM(
        valid_payload(
            entities=[
                {
                    "name": "Shell sort",
                    "entity_type": "task_experience",
                    "properties": [
                        {
                            "property_name": "strategy",
                            "value": {
                                "description": "Shell sort increment behavior.",
                                "content": ["The final increment must equal 1."],
                            },
                        }
                    ],
                }
            ]
        )
    )
    extractor = StructuredExtractor(
        llm_client=llm,
        entity_manager=FakePolicyEntityManager(),
        config=StructuredExtractionConfig(),
    )

    await extractor.extract(
        content="The final increment must equal 1.",
        event_time="2026-08-18",
        prompt_language="EN",
        allowed_property_names={"strategy"},
    )

    assert [call["task"] for call in llm.calls] == ["memory.add.structured_extract"]
    prompt = llm.calls[0]["messages"][0]["content"]
    assert "Schema eligibility and destination properties have already been selected" in prompt
    assert "Reject observations that are not reusable or successful" not in prompt
    assert "Only retain successful reusable strategies" not in prompt
    assert "a generic fact is not experimental success" not in prompt
    assert '"format": "structured_card_content"' in prompt


@pytest.mark.asyncio
async def test_final_selection_extraction_prompt_uses_one_unambiguous_array_output_contract():
    llm = QueueLLM(valid_payload(entities=[]))
    extractor = StructuredExtractor(
        llm_client=llm,
        entity_manager=FakeLlm4adEntityManager(),
        config=StructuredExtractionConfig(),
    )

    await extractor.extract(
        content="The final increment must equal 1.",
        event_time="2026-08-18",
        prompt_language="EN",
        allowed_property_names={"good_algorithm", "tags"},
    )

    prompt = llm.calls[0]["messages"][0]["content"]
    assert '"properties": [{"property_name": "good_algorithm"' in prompt
    assert (
        '"value": {"description": "source-grounded context", "content": ["complete atomic fact"], '
        '"artifact_refs": []}' in prompt
    )
    assert '"properties": {"good_algorithm"' not in prompt
    assert '"type": "object"' in prompt


@pytest.mark.asyncio
async def test_structured_prompts_inventory_only_source_evidence_forms_without_requiring_code():
    selection = json.dumps(
        {
            "selections": [
                {
                    "entity_type": "llm4ad_memory_card",
                    "property_names": ["good_algorithm", "tags"],
                }
            ],
            "extract_relationships": False,
        }
    )
    extracted = valid_payload(
        entities=[
            {
                "name": "Mutation benchmark",
                "entity_type": "llm4ad_memory_card",
                "description": "Measured mutation result.",
                "properties": [
                    {
                        "property_name": "good_algorithm",
                        "value": {
                            "description": "Bounded mutation improved the measured score.",
                            "content": [
                                "With mutation rate 0.1, the score improved from 0.72 to 0.81.",
                                "For example, candidate A retained feasibility after mutation.",
                            ],
                        },
                    },
                    {"property_name": "tags", "value": "mutation, benchmark"},
                ],
            }
        ]
    )
    coverage = json.dumps({"complete": True, "missing_evidence": []})
    llm = QueueLLM(selection, extracted, coverage)
    extractor = StructuredExtractor(
        llm_client=llm,
        entity_manager=FakeLlm4adEntityManager(),
        config=StructuredExtractionConfig(
            selection_enabled=True,
            coverage_validation_enabled=True,
        ),
    )

    result = await extractor.extract(
        content=(
            "With mutation rate 0.1, the score improved from 0.72 to 0.81. "
            "For example, candidate A retained feasibility after mutation."
        ),
        event_time="2026-08-18",
        prompt_language="EN",
    )

    assert result["entities"][0]["properties"][0]["value"]["content"][0].endswith("0.81.")
    assert [call["task"] for call in llm.calls] == [
        "memory.add.structured_select",
        "memory.add.structured_extract",
        "memory.add.structured_coverage",
    ]
    prompts = [call["messages"][0]["content"] for call in llm.calls]
    for prompt in prompts:
        assert "evidence forms actually present" in prompt
        assert "Do not require or invent an evidence form that is absent" in prompt
        assert "performance metrics" in prompt
        assert "worked examples" in prompt
        assert "code or pseudocode" in prompt
    assert "A label, name, section number, or broad summary does not cover" in prompts[1]
    assert "A label, name, section number, or broad summary does not cover" in prompts[2]


@pytest.mark.asyncio
async def test_generic_block_selects_schema_slots_before_complete_extraction():
    selection = json.dumps(
        {
            "selections": [{"entity_type": "task_experience", "property_names": ["strategy"]}],
            "extract_relationships": False,
        }
    )
    extracted = valid_payload(
        entities=[
            {
                "name": "Shell sort",
                "entity_type": "task_experience",
                "properties": [
                    {
                        "property_name": "strategy",
                        "value": {
                            "description": "Shell sort increment behavior.",
                            "content": ["The final increment must equal 1."],
                        },
                    }
                ],
            }
        ]
    )
    llm = QueueLLM(selection, extracted)
    extractor = StructuredExtractor(
        llm_client=llm,
        entity_manager=FakePolicyEntityManager(),
        config=StructuredExtractionConfig(selection_enabled=True),
    )

    result = await extractor.extract(
        content="The final increment must equal 1.",
        event_time="2026-08-18",
        prompt_language="EN",
    )

    assert [call["task"] for call in llm.calls] == [
        "memory.add.structured_select",
        "memory.add.structured_extract",
    ]
    selection_prompt = llm.calls[0]["messages"][0]["content"]
    extraction_prompt = llm.calls[1]["messages"][0]["content"]
    assert "Reject observations that are not reusable or successful" in selection_prompt
    assert "Do not extract, quote, summarize, title, or rewrite source facts" in selection_prompt
    assert "Reject observations that are not reusable or successful" not in extraction_prompt
    assert '"error"' not in extraction_prompt
    assert result["entities"][0]["properties"][0]["property_name"] == "strategy"


@pytest.mark.asyncio
async def test_grounded_coverage_omission_triggers_one_directed_complete_extraction():
    source = "Shell sort has O(n^2) worst-case complexity, but reaches O(n) when the input is already sorted."
    initial = valid_payload(
        entities=[
            {
                "name": "Shell sort",
                "entity_type": "task_experience",
                "properties": [
                    {
                        "property_name": "strategy",
                        "value": {
                            "description": "Shell sort complexity.",
                            "content": ["Shell sort has O(n^2) worst-case complexity."],
                        },
                    }
                ],
            }
        ]
    )
    audit = json.dumps(
        {
            "complete": False,
            "missing_evidence": [
                {
                    "quote": "reaches O(n) when the input is already sorted",
                    "reason": "The best-case condition is absent.",
                }
            ],
        }
    )
    completed = valid_payload(
        entities=[
            {
                "name": "Shell sort",
                "entity_type": "task_experience",
                "properties": [
                    {
                        "property_name": "strategy",
                        "value": {
                            "description": "Shell sort complexity.",
                            "content": [
                                "Shell sort has O(n^2) worst-case complexity.",
                                "Shell sort reaches O(n) when the input is already sorted.",
                            ],
                        },
                    }
                ],
            }
        ]
    )
    llm = QueueLLM(initial, audit, completed)
    extractor = StructuredExtractor(
        llm_client=llm,
        entity_manager=FakePolicyEntityManager(),
        config=StructuredExtractionConfig(selection_enabled=True, coverage_validation_enabled=True),
    )

    result = await extractor.extract(
        content=source,
        event_time="2026-08-18",
        prompt_language="EN",
        allowed_property_names={"strategy"},
    )

    assert [call["task"] for call in llm.calls] == [
        "memory.add.structured_extract",
        "memory.add.structured_coverage",
        "memory.add.structured_complete",
    ]
    completion_prompt = llm.calls[2]["messages"][0]["content"]
    assert "reaches O(n) when the input is already sorted" in completion_prompt
    assert "Preserve every already extracted grounded fact" in completion_prompt
    assert result["entities"][0]["properties"][0]["value"]["content"] == [
        "Shell sort has O(n^2) worst-case complexity.",
        "Shell sort reaches O(n) when the input is already sorted.",
    ]


@pytest.mark.asyncio
async def test_ungrounded_coverage_evidence_keeps_original_valid_extraction():
    source = "The final Shell-sort increment must equal 1."
    initial = valid_payload(
        entities=[
            {
                "name": "Shell sort",
                "entity_type": "task_experience",
                "properties": [
                    {
                        "property_name": "strategy",
                        "value": {
                            "description": "Shell sort increments.",
                            "content": [source],
                        },
                    }
                ],
            }
        ]
    )
    audit = json.dumps(
        {
            "complete": False,
            "missing_evidence": [{"quote": "Invented score=0.99", "reason": "Missing score"}],
        }
    )
    llm = QueueLLM(initial, audit)
    extractor = StructuredExtractor(
        llm_client=llm,
        entity_manager=FakePolicyEntityManager(),
        config=StructuredExtractionConfig(selection_enabled=True, coverage_validation_enabled=True),
    )

    result = await extractor.extract(
        content=source,
        event_time="2026-08-18",
        prompt_language="EN",
        allowed_property_names={"strategy"},
    )

    assert [call["task"] for call in llm.calls] == [
        "memory.add.structured_extract",
        "memory.add.structured_coverage",
    ]
    assert result["entities"][0]["properties"][0]["value"]["content"] == [source]
