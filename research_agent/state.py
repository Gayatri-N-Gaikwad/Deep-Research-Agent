"""Shared state schema for the research agent pipeline."""

from typing import Literal, Optional, TypedDict

QueryIntent = Literal["factual", "comparative", "exploratory", "procedural"]
Difficulty = Literal["shallow", "direct", "deep"]


class EvidenceItem(TypedDict):
    """Raw evidence item retrieved from a search track.

    Attributes:
        sub_query: The specific sub-query that produced this search result.
        source_url: The URL of the source page.
        title: The title of the search result.
        content: The text snippet or summary from the search result.
        track_id: The index of the sub-query track (0-indexed).
    """

    sub_query: str
    source_url: str
    title: str
    content: str
    track_id: int


class ResearchState(TypedDict, total=False):
    """Shared state schema across all nodes in the research pipeline.

    Attributes:
        raw_query: The user's original query string.
        language: Language code for research, hardcoded to 'en' in Stage 1.
        intent: Classification of the query intent ('factual', 'comparative', 'exploratory', 'procedural').
        entities: Extracted key entities, subjects, and topics.
        constraints: Explicit constraints extracted from query (time range, length/format, scope).
        difficulty: Research difficulty tier ('shallow', 'direct', 'deep') set by router node (Stage 2).
        plan: A short natural-language research plan for deep queries (Stage 3).
        sub_queries: Concrete decomposed sub-queries for parallel research (Stage 3).
        evidence: Raw evidence items collected from search tracks (Stage 4).
        iteration_count: Multi-hop reasoning iteration counter.
    """

    raw_query: str
    language: str
    intent: str
    entities: list[str]
    constraints: list[str]
    difficulty: Optional[Difficulty]
    plan: Optional[str]
    sub_queries: list[str]
    evidence: list[EvidenceItem]
    iteration_count: int



def create_initial_state(raw_query: str) -> ResearchState:
    """Create a default initialized ResearchState given a raw query string."""
    return {
        "raw_query": raw_query,
        "language": "en",
        "intent": "",
        "entities": [],
        "constraints": [],
        "difficulty": None,
        "plan": None,
        "sub_queries": [],
        "evidence": [],
        "iteration_count": 0,
    }

