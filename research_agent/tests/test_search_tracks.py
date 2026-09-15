"""Unit tests for search_tracks node using mocked Tavily responses."""

import time
import pytest

from research_agent.nodes.search_tracks import (
    SearchTracksError,
    get_tavily_client,
    search_single_track,
    search_tracks,
)
from research_agent.state import ResearchState


class MockTavilyClient:
    """Mock TavilyClient returning configurable search results or errors."""

    def __init__(self, failure_queries=None, delay=0.0):
        self.failure_queries = set(failure_queries or [])
        self.delay = delay
        self.call_count = 0

    def search(self, query: str, max_results: int = 5):
        self.call_count += 1
        if self.delay > 0:
            time.sleep(self.delay)

        if query in self.failure_queries:
            raise RuntimeError(f"Simulated network error for query: {query}")

        return {
            "results": [
                {
                    "url": f"https://example.com/{abs(hash(query))}/doc1",
                    "title": f"Result 1 for {query}",
                    "content": f"Detailed snippet content 1 for {query}",
                },
                {
                    "url": f"https://example.com/{abs(hash(query))}/doc2",
                    "title": f"Result 2 for {query}",
                    "content": f"Detailed snippet content 2 for {query}",
                },
            ]
        }


def test_search_tracks_all_succeed_mocked():
    """Verify all 3 parallel search tracks succeed, merge results, and tag track_ids."""
    mock_client = MockTavilyClient()
    sub_queries = [
        "US macroeconomic policy",
        "Japan economic strategy",
        "Comparative debt and GDP outcomes",
    ]
    state: ResearchState = {
        "raw_query": "Compare US and Japan economies",
        "sub_queries": sub_queries,
        "difficulty": "deep",
    }

    result = search_tracks(state, client=mock_client)

    evidence = result.get("evidence", [])
    # 3 tracks * 2 items per track = 6 items
    assert len(evidence) == 6

    # Verify track tags and sub-query mapping
    for item in evidence:
        assert "sub_query" in item
        assert "source_url" in item
        assert "title" in item
        assert "content" in item
        assert "track_id" in item
        assert item["sub_query"] == sub_queries[item["track_id"]]

    # Verify all 3 tracks are represented
    track_ids = {item["track_id"] for item in evidence}
    assert track_ids == {0, 1, 2}


def test_search_tracks_partial_failure_mocked():
    """Verify node continues gracefully when 1 track fails, preserving remaining tracks."""
    sub_queries = [
        "Track 0 query",
        "Track 1 query (failing)",
        "Track 2 query",
    ]
    mock_client = MockTavilyClient(failure_queries={"Track 1 query (failing)"})
    state: ResearchState = {
        "raw_query": "Test partial failure",
        "sub_queries": sub_queries,
        "difficulty": "deep",
    }

    result = search_tracks(state, client=mock_client)

    evidence = result.get("evidence", [])
    # 2 successful tracks * 2 items = 4 items
    assert len(evidence) == 4

    track_ids = {item["track_id"] for item in evidence}
    assert track_ids == {0, 2}
    assert 1 not in track_ids


def test_search_tracks_all_failed_raises_search_tracks_error():
    """Verify SearchTracksError is raised if all search tracks fail."""
    sub_queries = ["Track A", "Track B", "Track C"]
    mock_client = MockTavilyClient(failure_queries=sub_queries)
    state: ResearchState = {
        "raw_query": "Test total failure",
        "sub_queries": sub_queries,
        "difficulty": "deep",
    }

    with pytest.raises(SearchTracksError) as exc_info:
        search_tracks(state, client=mock_client)

    assert "All 3 search tracks failed" in str(exc_info.value)


def test_search_tracks_runs_concurrently():
    """Verify that search tracks execute concurrently rather than sequentially."""
    delay = 0.15
    mock_client = MockTavilyClient(delay=delay)
    sub_queries = ["Query 1", "Query 2", "Query 3"]
    state: ResearchState = {
        "raw_query": "Test concurrency",
        "sub_queries": sub_queries,
        "difficulty": "deep",
    }

    start_time = time.perf_counter()
    result = search_tracks(state, client=mock_client)
    elapsed = time.perf_counter() - start_time

    assert len(result.get("evidence", [])) == 6
    # Sequential run would take >= 0.45s (3 * 0.15s).
    # Concurrent run with 3 workers takes close to 0.15s.
    assert elapsed < 0.35, f"Expected concurrent execution (<0.35s), but took {elapsed:.2f}s"


def test_search_single_track_missing_api_key_raises(monkeypatch):
    """Verify ValueError is raised if TAVILY_API_KEY is not set in environment."""
    monkeypatch.delenv("TAVILY_API_KEY", raising=False)
    with pytest.raises(ValueError) as exc_info:
        get_tavily_client(client=None)

    assert "TAVILY_API_KEY environment variable is not set" in str(exc_info.value)


def test_search_single_track_empty_query_raises():
    """Verify ValueError is raised if sub_query is empty."""
    with pytest.raises(ValueError):
        search_single_track("", 0, client=MockTavilyClient())

    with pytest.raises(ValueError):
        search_single_track("   ", 0, client=MockTavilyClient())


def test_search_tracks_empty_sub_queries():
    """Verify empty sub_queries returns empty evidence immediately."""
    state: ResearchState = {
        "raw_query": "No sub queries",
        "sub_queries": [],
        "difficulty": "deep",
    }
    result = search_tracks(state, client=None)
    assert result["evidence"] == []
