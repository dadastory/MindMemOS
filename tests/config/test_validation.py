import pytest
from mindmemos.config import init_config, reset_config
from mindmemos.errors import InvalidConfigError, MissingConfigValueError


def test_gateway_jwt_requires_secret_during_config_init(tmp_path, monkeypatch) -> None:
    monkeypatch.setenv("MINDMEMOS_GATEWAY_JWT_SECRET", "env-secret")
    monkeypatch.setenv("MINDMEMOS_GATEWAY_INTERNAL_SECRET", "legacy-env-secret")
    config_path = tmp_path / "dev.yaml"
    config_path.write_text(
        """
auth:
  mode: gateway_jwt
""",
        encoding="utf-8",
    )

    try:
        with pytest.raises(MissingConfigValueError, match="auth.gateway_jwt_secret"):
            init_config(config_path=config_path)
    finally:
        reset_config()


def test_telemetry_requires_endpoint_during_config_init(tmp_path, monkeypatch) -> None:
    monkeypatch.setenv("MINDMEMOS_TELEMETRY_ENDPOINT", "")
    config_path = tmp_path / "dev.yaml"
    config_path.write_text(
        """
telemetry:
  enabled: true
  telemetry_endpoint:
""",
        encoding="utf-8",
    )

    try:
        with pytest.raises(MissingConfigValueError, match="telemetry.telemetry_endpoint"):
            init_config(config_path=config_path)
    finally:
        reset_config()


def test_embedding_dimensions_must_match_qdrant_vector_size(tmp_path) -> None:
    config_path = tmp_path / "dev.yaml"
    config_path.write_text(
        """
embed_model_router:
  endpoints:
    - model: text-embedding-test
      api_key: sk-test
      api_base: http://example.test/v1
      dimensions: 128
database:
  qdrant:
    vector_size: 256
""",
        encoding="utf-8",
    )

    try:
        with pytest.raises(InvalidConfigError, match="dimensions"):
            init_config(config_path=config_path)
    finally:
        reset_config()


def test_vanilla_add_chunk_budget_must_leave_extractable_space(tmp_path) -> None:
    config_path = tmp_path / "dev.yaml"
    config_path.write_text(
        """
algo_config:
  add:
    vanilla:
      chunk_soft_token_budget: 8000
      chunk_hard_token_budget: 10000
""",
        encoding="utf-8",
    )

    try:
        with pytest.raises(InvalidConfigError, match="chunk_hard_token_budget"):
            init_config(config_path=config_path)
    finally:
        reset_config()


@pytest.mark.parametrize(
    "body",
    [
        "extraction:\n          max_repair_attempts: 2",
        "dedup:\n          candidate_top_k: 2\n          max_merge_candidates: 3",
        "dedup:\n          create_below: 0.98\n          same_batch_group_at_or_above: 0.97",
        "embedding:\n          batch_size: 0",
        "concurrency:\n          max_extract_concurrency: 0",
        "concurrency:\n          max_write_conflict_retries: 0",
    ],
)
def test_structured_add_rejects_unbounded_or_inconsistent_settings(tmp_path, body: str) -> None:
    config_path = tmp_path / "dev.yaml"
    config_path.write_text(
        f"""
algo_config:
  add:
    structured:
      {body}
""",
        encoding="utf-8",
    )

    try:
        with pytest.raises(InvalidConfigError, match="algo_config.add.structured"):
            init_config(config_path=config_path)
    finally:
        reset_config()


@pytest.mark.parametrize(
    ("field", "value"),
    [
        ("dedup_threshold", 0),
        ("recall_size", 101),
        ("hybrid_prefetch_factor", 11),
        ("hybrid_prefetch_min", 301),
        ("hybrid_prefetch_max", 301),
        ("dedup_max_candidates", 129),
    ],
)
def test_vanilla_search_rejects_unsafe_limits(tmp_path, field: str, value: int) -> None:
    config_path = tmp_path / "dev.yaml"
    config_path.write_text(
        f"""
algo_config:
  search:
    vanilla:
      {field}: {value}
""",
        encoding="utf-8",
    )

    try:
        with pytest.raises(InvalidConfigError, match=field):
            init_config(config_path=config_path)
    finally:
        reset_config()


def test_vanilla_search_prefetch_min_must_not_exceed_max(tmp_path) -> None:
    config_path = tmp_path / "dev.yaml"
    config_path.write_text(
        """
algo_config:
  search:
    vanilla:
      hybrid_prefetch_min: 101
      hybrid_prefetch_max: 100
""",
        encoding="utf-8",
    )

    try:
        with pytest.raises(InvalidConfigError, match="hybrid_prefetch_min"):
            init_config(config_path=config_path)
    finally:
        reset_config()


@pytest.mark.parametrize(
    ("field", "value"),
    [
        ("max_token_budget", 0),
        ("max_candidates", 0),
        ("relevance_weight", -0.1),
        ("query_overlap_weight", -0.1),
        ("recency_weight", -0.1),
        ("cost_weight", -0.1),
        ("recency_half_life_days", 0),
        ("missing_recency_score", 1.1),
        ("graph_provenance_limit", 0),
    ],
)
def test_memory_retention_rejects_invalid_limits(tmp_path, field: str, value: float) -> None:
    config_path = tmp_path / "dev.yaml"
    config_path.write_text(
        f"""
algo_config:
  search:
    retention:
      {field}: {value}
""",
        encoding="utf-8",
    )

    try:
        with pytest.raises(InvalidConfigError, match=field):
            init_config(config_path=config_path)
    finally:
        reset_config()


def test_memory_retention_rejects_non_finite_weight(tmp_path) -> None:
    config_path = tmp_path / "dev.yaml"
    config_path.write_text(
        """
algo_config:
  search:
    retention:
      relevance_weight: .nan
""",
        encoding="utf-8",
    )

    try:
        with pytest.raises(InvalidConfigError, match="relevance_weight"):
            init_config(config_path=config_path)
    finally:
        reset_config()


def test_memory_retention_minimum_budget_must_not_exceed_maximum(tmp_path) -> None:
    config_path = tmp_path / "dev.yaml"
    config_path.write_text(
        """
algo_config:
  search:
    retention:
      min_token_budget: 100
      max_token_budget: 10
""",
        encoding="utf-8",
    )

    try:
        with pytest.raises(InvalidConfigError, match="min_token_budget"):
            init_config(config_path=config_path)
    finally:
        reset_config()
