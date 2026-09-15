"""Fast unit tests for query_understanding node using mocked LLM responses."""

import pytest

from research_agent.nodes.query_understanding import (
    QueryAnalysis,
    QueryUnderstandingError,
    query_understanding,
)
from research_agent.state import ResearchState


class MockLLMWrapper:
    """Mock LLM returning a pre-defined QueryAnalysis instance for fast offline unit tests."""

    def __init__(self, response: QueryAnalysis):
        self.response = response

    def with_structured_output(self, schema):
        class MockRunnable:
            def __init__(self, response):
                self.response = response

            def invoke(self, messages):
                return self.response

        return MockRunnable(self.response)


def test_factual_query_mocked():
    """Mocked unit test: verifies node behavior, state update, and schema handling for factual queries."""
    mock_response = QueryAnalysis(
        intent="factual",
        entities=["water", "boiling point", "sea level"],
        constraints=[],
    )
    llm = MockLLMWrapper(mock_response)
    state: ResearchState = {
        "raw_query": "What is the boiling point of water at sea level?"
    }
    result = query_understanding(state, llm=llm)

    assert result["intent"] == "factual"
    assert result["constraints"] == []
    assert result["language"] == "en"
    assert "water" in result["entities"]


def test_comparative_query_mocked():
    """Mocked unit test: verifies node behavior, state update, and schema handling for comparative queries."""
    mock_response = QueryAnalysis(
        intent="comparative",
        entities=["United States", "Japan", "economic policies"],
        constraints=[],
    )
    llm = MockLLMWrapper(mock_response)
    state: ResearchState = {
        "raw_query": "Compare the economic policies of the United States and Japan"
    }
    result = query_understanding(state, llm=llm)

    assert result["intent"] == "comparative"
    assert result["language"] == "en"
    assert "United States" in result["entities"]
    assert "Japan" in result["entities"]


def test_query_with_explicit_constraints_mocked():
    """Mocked unit test: verifies node behavior, state update, and schema handling for constrained queries."""
    mock_response = QueryAnalysis(
        intent="exploratory",
        entities=["quantum computing"],
        constraints=["since 2023", "under 200 words"],
    )
    llm = MockLLMWrapper(mock_response)
    state: ResearchState = {
        "raw_query": "Summarize developments in quantum computing since 2023, in under 200 words"
    }
    result = query_understanding(state, llm=llm)

    assert len(result["constraints"]) == 2
    assert "since 2023" in result["constraints"]
    assert "under 200 words" in result["constraints"]
    assert result["language"] == "en"


def test_ambiguous_short_query_mocked():
    """Mocked unit test: verifies node behavior, state update, and schema handling for short queries."""
    mock_response = QueryAnalysis(
        intent="exploratory",
        entities=["AI"],
        constraints=[],
    )
    llm = MockLLMWrapper(mock_response)
    state: ResearchState = {"raw_query": "AI"}
    result = query_understanding(state, llm=llm)

    assert result["intent"] == "exploratory"
    assert result["language"] == "en"
    assert "AI" in result["entities"]


def test_empty_query_raises_value_error():
    """Verify ValueError is raised if raw_query is empty or missing."""
    with pytest.raises(ValueError):
        query_understanding({"raw_query": ""})

    with pytest.raises(ValueError):
        query_understanding({"raw_query": "   "})


def test_retry_once_and_raise_query_understanding_error():
    """Verify that node retries once before raising QueryUnderstandingError on repeated failure."""
    class FailingRunnable:
        def __init__(self):
            self.call_count = 0

        def invoke(self, messages):
            self.call_count += 1
            raise RuntimeError(f"Simulated connection failure on attempt {self.call_count}")

    runnable = FailingRunnable()

    class FailingLLM:
        def with_structured_output(self, schema):
            return runnable

    failing_llm = FailingLLM()
    state: ResearchState = {"raw_query": "Test error handling"}

    with pytest.raises(QueryUnderstandingError) as exc_info:
        query_understanding(state, llm=failing_llm)

    assert runnable.call_count == 2
    assert "Query understanding failed after 1 retry" in str(exc_info.value)
