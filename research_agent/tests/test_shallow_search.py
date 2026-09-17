"""Fast unit tests for shallow_search node using mocked Tavily responses."""

import pytest

from research_agent.nodes.shallow_search import ShallowSearchError, shallow_search
from research_agent.state import ResearchState


class MockTavilyClient:
    """Mock Tavily client returning fixed search results."""

    def __init__(self, results=None, raise_exc=False):
        self.results = results if results is not None else [
            {"url": "https://example.com/1", "title": "Result 1", "content": "Content 1"},
            {"url": "https://example.com/2", "title": "Result 2", "content": "Content 2"},
        ]
        self.raise_exc = raise_exc
        self.last_query = None

    def search(self, query: str, max_results: int = 5):
        self.last_query = query
        if self.raise_exc:
            raise RuntimeError("Tavily API error")
        return {"results": self.results}


def test_shallow_search_happy_path():
    """Verify shallow search populates evidence list with track_id=0 and retains all 5 keys."""
    mock_client = MockTavilyClient()
    state: ResearchState = {"raw_query": "What is the capital of France?"}

    result = shallow_search(state, client=mock_client)

    assert "evidence" in result
    assert len(result["evidence"]) == 2
    assert mock_client.last_query == "What is the capital of France?"

    expected_keys = {"sub_query", "source_url", "title", "content", "track_id"}
    for item in result["evidence"]:
        assert set(item.keys()) == expected_keys
        assert item["track_id"] == 0
        assert item["sub_query"] == "What is the capital of France?"

    assert result.get("answer") is None


def test_shallow_search_empty_query_raises_value_error():
    """Verify ValueError is raised if raw_query is empty or whitespace-only."""
    with pytest.raises(ValueError):
        shallow_search({"raw_query": ""})

    with pytest.raises(ValueError):
        shallow_search({"raw_query": "   "})


def test_shallow_search_raises_shallow_search_error_on_api_failure():
    """Verify ShallowSearchError is raised when Tavily search fails."""
    mock_client = MockTavilyClient(raise_exc=True)
    state: ResearchState = {"raw_query": "Test search failure"}

    with pytest.raises(ShallowSearchError) as exc_info:
        shallow_search(state, client=mock_client)

    assert "Shallow search failed" in str(exc_info.value)


def test_shallow_search_raises_shallow_search_error_on_zero_results():
    """Verify ShallowSearchError is raised when Tavily search returns empty results list."""
    mock_client = MockTavilyClient(results=[])
    state: ResearchState = {"raw_query": "Obscure non-existent query"}

    with pytest.raises(ShallowSearchError) as exc_info:
        shallow_search(state, client=mock_client)

    assert "zero results" in str(exc_info.value)
