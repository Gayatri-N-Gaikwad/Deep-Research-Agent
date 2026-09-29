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


class RankedEvidenceItem(EvidenceItem):
    """Ranked and scored evidence item produced by hybrid retrieval (Stage 5 deep path).

    Inherits all fields from EvidenceItem:
        sub_query: str
        source_url: str
        title: str
        content: str
        track_id: int

    Additional Hybrid Retrieval Attributes:
        bm25_score: float      # Raw BM25Okapi lexical score
        bm25_rank: int         # 1-indexed lexical rank
        semantic_score: float  # Cosine similarity score [-1.0, 1.0]
        semantic_rank: int     # 1-indexed semantic rank
        rrf_score: float       # Reciprocal Rank Fusion score (k=60)
        rank: int              # Final 1-indexed fused rank (1 is top rank)
    """

    bm25_score: float
    bm25_rank: int
    semantic_score: float
    semantic_rank: int
    rrf_score: float
    rank: int


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
        ranked_evidence: Deduplicated, dual-signal ranked evidence list (Stage 5 Hybrid Retrieval).
        iteration_count: Multi-hop reasoning iteration counter.
        answer: Final or direct-path answer text. Populated by the direct path (Stage 5); deep and shallow paths leave this None until the Synthesis stage.
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
    ranked_evidence: list[RankedEvidenceItem]
    iteration_count: int
    answer: Optional[str]


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
        "ranked_evidence": [],
        "iteration_count": 0,
        "answer": None,
    }
