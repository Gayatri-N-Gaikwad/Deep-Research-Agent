"""Fast unit tests for direct_answer node using mocked LLM responses."""

import pytest

from research_agent.nodes.direct_answer import (
    DirectAnswer,
    DirectAnswerError,
    direct_answer,
)
from research_agent.state import ResearchState


class MockLLMWrapper:
    """Mock LLM returning a pre-defined DirectAnswer instance for fast offline unit tests."""

    def __init__(self, response: DirectAnswer):
        self.response = response

    def with_structured_output(self, schema):
        class MockRunnable:
            def __init__(self, response):
                self.response = response

            def invoke(self, messages):
                return self.response

        return MockRunnable(self.response)


def test_direct_answer_happy_path():
    """Verify direct answer node populates answer in state while leaving evidence empty."""
    mock_response = DirectAnswer(
        answer="Water boils at 100 degrees Celsius at standard atmospheric pressure.",
        confidence="high",
    )
    llm = MockLLMWrapper(mock_response)
    state: ResearchState = {
        "raw_query": "What temperature does water boil at?",
        "evidence": [],
    }

    result = direct_answer(state, llm=llm)

    assert result["answer"] == "Water boils at 100 degrees Celsius at standard atmospheric pressure."
    assert result["evidence"] == []


def test_direct_answer_empty_query_raises_value_error():
    """Verify ValueError is raised if raw_query is empty or whitespace-only."""
    with pytest.raises(ValueError):
        direct_answer({"raw_query": ""})

    with pytest.raises(ValueError):
        direct_answer({"raw_query": "   "})


def test_retry_once_and_raise_direct_answer_error():
    """Verify node retries once before raising DirectAnswerError on repeated failure."""
    class FailingRunnable:
        def __init__(self):
            self.call_count = 0

        def invoke(self, messages):
            self.call_count += 1
            raise RuntimeError(f"Simulated LLM error on attempt {self.call_count}")

    runnable = FailingRunnable()

    class FailingLLM:
        def with_structured_output(self, schema):
            return runnable

    failing_llm = FailingLLM()
    state: ResearchState = {"raw_query": "What is 2+2?"}

    with pytest.raises(DirectAnswerError) as exc_info:
        direct_answer(state, llm=failing_llm)

    assert runnable.call_count == 2
    assert "Direct answer generation failed after 1 retry" in str(exc_info.value)
