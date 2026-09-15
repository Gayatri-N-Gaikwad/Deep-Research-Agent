"""Parallel Search Tracks node for deep research query execution via Tavily."""

import concurrent.futures
import logging
import os
from typing import Optional

from dotenv import load_dotenv
from tavily import TavilyClient

from research_agent.state import EvidenceItem, ResearchState

load_dotenv()

logger = logging.getLogger(__name__)


class SearchTracksError(RuntimeError):
    """Raised when all search tracks fail or search execution cannot proceed."""
    pass


def get_tavily_client(client=None):
    """Instantiate a TavilyClient using TAVILY_API_KEY from environment or return injected client.

    Args:
        client: Optional existing or mocked client instance.

    Returns:
        TavilyClient or injected client instance.

    Raises:
        ValueError: If TAVILY_API_KEY environment variable is not set and no client is provided.
    """
    if client is not None:
        return client

    api_key = os.getenv("TAVILY_API_KEY")
    if not api_key:
        raise ValueError(
            "TAVILY_API_KEY environment variable is not set. "
            "Please set it in your environment or in a .env file."
        )

    return TavilyClient(api_key=api_key)


def search_single_track(
    sub_query: str,
    track_id: int,
    client=None,
) -> list[EvidenceItem]:
    """Execute a single Tavily search for a sub-query and return formatted evidence items.

    Args:
        sub_query: The specific sub-query to search for.
        track_id: The index identifying this search track (0-indexed).
        client: Optional TavilyClient or mock client.

    Returns:
        List of EvidenceItem dicts tagged with the track_id and sub_query.

    Raises:
        ValueError: If sub_query is empty or missing.
        Exception: Any network or API exception raised by TavilyClient.search.
    """
    if not sub_query or not str(sub_query).strip():
        raise ValueError("Sub-query must be a non-empty string.")

    tavily_client = get_tavily_client(client=client)
    response = tavily_client.search(query=sub_query, max_results=5)

    results = response.get("results", []) if isinstance(response, dict) else []
    evidence_items: list[EvidenceItem] = []

    for item in results:
        evidence_items.append({
            "sub_query": sub_query,
            "source_url": item.get("url", ""),
            "title": item.get("title", ""),
            "content": item.get("content", ""),
            "track_id": track_id,
        })

    return evidence_items


def search_tracks(state: ResearchState, client=None) -> ResearchState:
    """Concurrently search across all sub-queries and store merged results in state['evidence'].

    Uses ThreadPoolExecutor to fire searches across all sub-queries concurrently.
    If a single track fails, logs the error and continues with results from the
    other tracks. If all tracks fail, raises SearchTracksError.

    Args:
        state: Current ResearchState containing 'sub_queries'.
        client: Optional TavilyClient or mock client.

    Returns:
        Updated ResearchState with 'evidence' populated.

    Raises:
        SearchTracksError: If all search tracks fail.
    """
    sub_queries = state.get("sub_queries", [])
    if not sub_queries:
        return {
            **state,
            "evidence": [],
        }

    tavily_client = get_tavily_client(client=client)

    max_workers = min(len(sub_queries), 5)
    track_results_map: dict[int, list[EvidenceItem]] = {}
    errors: list[tuple[int, str, Exception]] = []

    with concurrent.futures.ThreadPoolExecutor(max_workers=max_workers) as executor:
        future_to_track = {
            executor.submit(
                search_single_track, sq, idx, client=tavily_client
            ): (idx, sq)
            for idx, sq in enumerate(sub_queries)
        }

        for future in concurrent.futures.as_completed(future_to_track):
            track_id, sq = future_to_track[future]
            try:
                items = future.result()
                track_results_map[track_id] = items
            except Exception as exc:
                logger.warning("Search track %d failed for query '%s': %s", track_id, sq, exc)
                errors.append((track_id, sq, exc))

    # Total failure: if every track failed
    if len(errors) == len(sub_queries):
        error_details = "; ".join(f"Track {tid} ('{q}'): {err}" for tid, q, err in errors)
        raise SearchTracksError(
            f"All {len(sub_queries)} search tracks failed. Details: {error_details}"
        )

    # Flatten and order results deterministically by track_id
    merged_evidence: list[EvidenceItem] = []
    for track_id in sorted(track_results_map.keys()):
        merged_evidence.extend(track_results_map[track_id])

    return {
        **state,
        "evidence": merged_evidence,
    }
