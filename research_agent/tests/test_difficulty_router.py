"""Fast unit tests for difficulty_router node using mocked LLM responses."""

import pytest

from research_agent.nodes.difficulty_router import (
    DifficultyClassification,
    DifficultyRouterError,
    difficulty_router,
)
from research_agent.state import ResearchState


class MockLLMWrapper:
    """Mock LLM returning a pre-defined DifficultyClassification instance for offline unit tests."""

    def __init__(self, response: DifficultyClassification):
        self.response = response

    def with_structured_output(self, schema):
        class MockRunnable:
            def __init__(self, response):
                self.response = response

            def invoke(self, messages):
                return self.response

        return MockRunnable(self.response)


def test_shallow_query_mocked():
    """Mocked unit test: single narrow factual lookup -> shallow difficulty."""
    mock_response = DifficultyClassification(
        difficulty="shallow",
        reasoning="Single narrow factual lookup answerable with a single search.",
    )
    llm = MockLLMWrapper(mock_response)
    state: ResearchState = {
        "raw_query": "What is the capital of France?",
        "intent": "factual",
        "entities": ["capital", "France"],
        "constraints": [],
    }
    result = difficulty_router(state, llm=llm)

    assert result["difficulty"] == "shallow"
    assert result["raw_query"] == "What is the capital of France?"


def test_direct_query_mocked():
    """Mocked unit test: arithmetic / common fact -> direct difficulty."""
    mock_response = DifficultyClassification(
        difficulty="direct",
        reasoning="Basic arithmetic answerable from general knowledge with no search needed.",
    )
    llm = MockLLMWrapper(mock_response)
    state: ResearchState = {
        "raw_query": "What is 15 times 12?",
        "intent": "factual",
        "entities": ["15", "12"],
        "constraints": [],
    }
    result = difficulty_router(state, llm=llm)

    assert result["difficulty"] == "direct"
    assert result["raw_query"] == "What is 15 times 12?"


def test_deep_query_mocked():
    """Mocked unit test: multi-entity comparison -> deep difficulty."""
    mock_response = DifficultyClassification(
        difficulty="deep",
        reasoning="Requires multi-entity comparison and synthesis across diverse sources.",
    )
    llm = MockLLMWrapper(mock_response)
    state: ResearchState = {
        "raw_query": "Compare the economic policies of the United States and Japan",
        "intent": "comparative",
        "entities": ["United States", "Japan", "economic policies"],
        "constraints": [],
    }
    result = difficulty_router(state, llm=llm)

    assert result["difficulty"] == "deep"
    assert result["raw_query"] == "Compare the economic policies of the United States and Japan"


def test_empty_query_raises_value_error():
    """Verify ValueError is raised if raw_query is empty or missing."""
    with pytest.raises(ValueError):
        difficulty_router({"raw_query": ""})

    with pytest.raises(ValueError):
        difficulty_router({"raw_query": "   "})


def test_retry_once_and_raise_difficulty_router_error():
    """Verify that node retries once before raising DifficultyRouterError on repeated failure."""
    class FailingRunnable:
        def __init__(self):
            self.call_count = 0

        def invoke(self, messages):
            self.call_count += 1
            raise RuntimeError(f"Simulated network error on attempt {self.call_count}")

    runnable = FailingRunnable()

    class FailingLLM:
        def with_structured_output(self, schema):
            return runnable

    failing_llm = FailingLLM()
    state: ResearchState = {
        "raw_query": "Test error handling",
        "intent": "factual",
        "entities": [],
        "constraints": [],
    }

    with pytest.raises(DifficultyRouterError) as exc_info:
        difficulty_router(state, llm=failing_llm)

    assert runnable.call_count == 2
    assert "Difficulty router failed after 1 retry" in str(exc_info.value)
