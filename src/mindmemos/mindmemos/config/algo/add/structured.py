"""Configuration for the lightweight structured Add algorithm."""

from __future__ import annotations

from dataclasses import dataclass, field


@dataclass
class StructuredExtractionConfig:
    """Limits for one fixed-Schema extraction and its optional repair."""

    max_repair_attempts: int = 1
    max_entities: int = 20
    max_properties_per_entity: int = 15
    selection_enabled: bool = False
    coverage_validation_enabled: bool = False
    max_coverage_items: int = 12


@dataclass
class StructuredDedupConfig:
    """Exact and semantic history consolidation policy."""

    vector_enabled: bool = True
    candidate_top_k: int = 5
    create_below: float = 0.82
    same_batch_group_at_or_above: float = 0.97
    merge_mode: str = "llm_on_ambiguous"
    max_merge_candidates: int = 3


@dataclass
class StructuredEpisodeConfig:
    """Bounded Episode recall performed by the shared extraction call."""

    mode: str = "single_pass"
    candidate_top_k: int = 5
    reuse_at_or_above: float = 0.82


@dataclass
class StructuredGraphConfig:
    """Graph shape emitted by lightweight structured writes."""

    mode: str = "entity_property_episode"


@dataclass
class StructuredEmbeddingConfig:
    """Embedding batching policy."""

    batch_size: int = 64


@dataclass
class StructuredConcurrencyConfig:
    """Provider and short commit-section concurrency controls."""

    max_extract_concurrency: int = 5
    lock_stripes: int = 64
    max_write_conflict_retries: int = 8


@dataclass
class StructuredHistoryConfig:
    """Revision and source-reference retention policy."""

    max_source_refs: int = 100


@dataclass
class StructuredBatchConfig:
    """Bounds for unordered multi-block structured ingestion."""

    max_blocks: int = 128
    max_total_chars: int = 200_000
    max_episode_repair_attempts: int = 1
    max_consolidation_repair_attempts: int = 1


@dataclass
class StructuredAddConfig:
    """Root configuration for ``structured_add``."""

    extraction: StructuredExtractionConfig = field(default_factory=StructuredExtractionConfig)
    dedup: StructuredDedupConfig = field(default_factory=StructuredDedupConfig)
    episode: StructuredEpisodeConfig = field(default_factory=StructuredEpisodeConfig)
    graph: StructuredGraphConfig = field(default_factory=StructuredGraphConfig)
    embedding: StructuredEmbeddingConfig = field(default_factory=StructuredEmbeddingConfig)
    concurrency: StructuredConcurrencyConfig = field(default_factory=StructuredConcurrencyConfig)
    history: StructuredHistoryConfig = field(default_factory=StructuredHistoryConfig)
    batch: StructuredBatchConfig = field(default_factory=StructuredBatchConfig)


__all__ = [
    "StructuredAddConfig",
    "StructuredBatchConfig",
    "StructuredConcurrencyConfig",
    "StructuredDedupConfig",
    "StructuredEmbeddingConfig",
    "StructuredEpisodeConfig",
    "StructuredExtractionConfig",
    "StructuredGraphConfig",
    "StructuredHistoryConfig",
]
