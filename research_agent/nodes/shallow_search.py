"""Shallow search node for the research agent pipeline."""

from typing import Optional

from research_agent.nodes.search_tracks import search_single_track
from research_agent.state import ResearchState


class ShallowSearchError(RuntimeError):
    """Raised when shallow search fails or produces zero results."""
    pass


def shallow_search(state: ResearchState, client=None) -> ResearchState:
    """Execute a shallow single-track web search for the raw query.

    Takes ResearchState containing 'raw_query', executes a single search track
    using Tavily, and populates state['evidence'] with all retrieved results tagged
    with track_id=0.

    Args:
        state: The current ResearchState dictionary.
        client: Optional custom or mocked TavilyClient instance for testing.

    Returns:
        Updated ResearchState dictionary with evidence populated.

    Raises:
        ValueError: If 'raw_query' is missing or empty.
        ShallowSearchError: If search fails or returns zero results.
    """
    raw_query = state.get("raw_query")
    if not raw_query or not str(raw_query).strip():
        raise ValueError("ResearchState 'raw_query' must be a non-empty string.")

    try:
        evidence_items = search_single_track(
            sub_query=raw_query,
            track_id=0,
            client=client,
        )
    except Exception as exc:
        raise ShallowSearchError(
            f"Shallow search failed for query '{raw_query}'. Error: {exc}"
        ) from exc

    if not evidence_items:
        raise ShallowSearchError(
            f"Shallow search produced zero results for query '{raw_query}'."
        )

    return {
        **state,
        "evidence": evidence_items,
    }
