"""Add-operation algorithm configuration."""

from __future__ import annotations

from dataclasses import dataclass, field

from .schema import (
    DrainConfig,
    EpisodesChunkerConfig,
    SchemaAddConfig,
    SchemaAddEpisodeEdgeConfig,
    SchemaAddExtractionConfig,
    SchemaAddHigherOrderConfig,
    SchemaAddMergeConfig,
)
from .structured import (
    StructuredAddConfig,
    StructuredBatchConfig,
    StructuredConcurrencyConfig,
    StructuredDedupConfig,
    StructuredEmbeddingConfig,
    StructuredEpisodeConfig,
    StructuredExtractionConfig,
    StructuredGraphConfig,
    StructuredHistoryConfig,
)
from .vanilla import VanillaAddConfig


@dataclass
class AddAlgoConfig:
    """Configuration for add-operation algorithms."""

    schema: SchemaAddConfig = field(default_factory=SchemaAddConfig)
    structured: StructuredAddConfig = field(default_factory=StructuredAddConfig)
    vanilla: VanillaAddConfig = field(default_factory=VanillaAddConfig)


__all__ = [
    "AddAlgoConfig",
    "DrainConfig",
    "EpisodesChunkerConfig",
    "SchemaAddConfig",
    "SchemaAddEpisodeEdgeConfig",
    "SchemaAddExtractionConfig",
    "SchemaAddHigherOrderConfig",
    "SchemaAddMergeConfig",
    "StructuredAddConfig",
    "StructuredBatchConfig",
    "StructuredConcurrencyConfig",
    "StructuredDedupConfig",
    "StructuredEmbeddingConfig",
    "StructuredEpisodeConfig",
    "StructuredExtractionConfig",
    "StructuredGraphConfig",
    "StructuredHistoryConfig",
    "VanillaAddConfig",
]
