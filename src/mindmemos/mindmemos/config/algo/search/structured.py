"""Configuration for lightweight Structured retrieval."""

from __future__ import annotations

from dataclasses import dataclass, field


@dataclass
class StructuredSearchConfig:
    """Quality controls for Structured hybrid and graph retrieval."""

    min_relevance_score: float | None = field(default=0.35)
    """Minimum dense cosine score required before hybrid ranking and graph expansion."""
