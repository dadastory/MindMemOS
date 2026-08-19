"""Episode candidate contract shared by structured extraction and ingestion."""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(slots=True, frozen=True)
class StructuredEpisodeCandidate:
    """One whitelisted Episode supplied to the structured extraction call."""

    episode_id: str
    title: str
    description: str
    score: float
    session_id: str | None = None
    same_session: bool = False

    def prompt_value(self) -> dict[str, object]:
        return {
            "episode_id": self.episode_id,
            "title": self.title,
            "description": self.description,
            "same_session": self.same_session,
        }


__all__ = ["StructuredEpisodeCandidate"]
