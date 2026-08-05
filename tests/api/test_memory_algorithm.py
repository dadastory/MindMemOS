import pytest
from mindmemos.api.algorithm import binding_for_memory_algorithm, resolve_memory_algorithm
from mindmemos.config import init_config, reset_config
from mindmemos.errors import AuthenticationError
from mindmemos.pipelines import create_pipeline


def test_structured_algorithm_uses_lightweight_add_and_schema_search() -> None:
    assert binding_for_memory_algorithm("structured").add_pipeline == "structured_add"
    assert binding_for_memory_algorithm("structured").search_pipeline == "schema"


def test_existing_algorithm_bindings_remain_unchanged() -> None:
    assert binding_for_memory_algorithm("vanilla").add_pipeline == "vanilla_add"
    assert binding_for_memory_algorithm("vanilla").search_pipeline == "vanilla"
    assert binding_for_memory_algorithm("schema").add_pipeline == "schema_add"
    assert binding_for_memory_algorithm("schema").search_pipeline == "schema"


def test_unknown_algorithm_is_still_rejected() -> None:
    with pytest.raises(AuthenticationError):
        resolve_memory_algorithm("unknown")


def test_builtin_registry_constructs_structured_pipeline() -> None:
    init_config(config_path="config/mindmemos/dev.example.yaml")
    try:
        pipeline = create_pipeline(type="add", name="structured_add")
        assert type(pipeline).__name__ == "StructuredAddPipeline"
        assert type(pipeline).__module__.endswith("pipelines.add.structured.pipeline")
    finally:
        reset_config()
