"""Fast unit tests for planner node using mocked LLM responses."""

import pytest

from research_agent.nodes.planner import (
    PlannerError,
    ResearchPlan,
    planner,
)
from research_agent.state import ResearchState


class MockLLMWrapper:
    """Mock LLM returning a pre-defined ResearchPlan instance for offline unit tests."""

    def __init__(self, response: ResearchPlan):
        self.response = response

    def with_structured_output(self, schema):
        class MockRunnable:
            def __init__(self, response):
                self.response = response

            def invoke(self, messages):
                return self.response

        return MockRunnable(self.response)


def test_comparative_query_decomposition_mocked():
    """Mocked unit test: comparative query decomposes into plan and distinct sub-queries."""
    mock_plan = ResearchPlan(
        plan="Analyze the fiscal, monetary, and regulatory frameworks of the US and Japan, evaluating structural differences and macroeconomic outcomes.",
        sub_queries=[
            "What are the core fiscal and monetary policies of the United States, including interest rate policy and government spending?",
            "What are the key features of Japan's economic policies, including Abenomics and Bank of Japan monetary interventions?",
            "How do the economic outcomes and debt dynamics of the United States and Japan directly compare?",
        ],
    )
    llm = MockLLMWrapper(mock_plan)
    state: ResearchState = {
        "raw_query": "Compare the economic policies of the United States and Japan",
        "intent": "comparative",
        "entities": ["United States", "Japan", "economic policies"],
        "constraints": [],
        "difficulty": "deep",
    }
    result = planner(state, llm=llm)

    # Assert plan is non-empty
    assert result["plan"]
    assert len(result["plan"].strip()) > 0

    # Assert 2-3 sub-queries returned
    assert 2 <= len(result["sub_queries"]) <= 3

    # Assert decomposition addresses both entities specifically
    all_sub_queries_text = " ".join(result["sub_queries"])
    assert "United States" in all_sub_queries_text
    assert "Japan" in all_sub_queries_text

    # Assert original state fields are preserved
    assert result["raw_query"] == state["raw_query"]
    assert result["difficulty"] == "deep"


def test_multi_aspect_query_decomposition_mocked():
    """Mocked unit test: multi-aspect query decomposes into distinct sub-facets."""
    mock_plan = ResearchPlan(
        plan="Examine the origins, peak systemic events, and long-term regulatory and economic aftermath of the 2008 financial collapse.",
        sub_queries=[
            "What were the primary root causes and subprime mortgage market triggers of the 2008 financial crisis?",
            "What were the pivotal events during the height of the 2008 crisis, including the Lehman Brothers bankruptcy and emergency bailouts?",
            "What were the major long-term global economic consequences and regulatory reforms following the 2008 crisis?",
        ],
    )
    llm = MockLLMWrapper(mock_plan)
    state: ResearchState = {
        "raw_query": "What were the causes, key events, and consequences of the 2008 financial crisis?",
        "intent": "exploratory",
        "entities": ["2008 financial crisis", "causes", "key events", "consequences"],
        "constraints": [],
        "difficulty": "deep",
    }
    result = planner(state, llm=llm)

    # Assert plan is non-empty
    assert result["plan"]

    # Assert 2-3 distinct sub-queries
    assert 2 <= len(result["sub_queries"]) <= 3

    # Assert each aspect is uniquely represented (not identical restatements)
    sub_q = result["sub_queries"]
    assert any("causes" in sq.lower() or "triggers" in sq.lower() for sq in sub_q)
    assert any("events" in sq.lower() or "lehman" in sq.lower() for sq in sub_q)
    assert any("consequences" in sq.lower() or "reforms" in sq.lower() for sq in sub_q)
    assert len(set(sub_q)) == len(sub_q)  # All sub-queries are distinct


def test_empty_query_raises_value_error():
    """Verify ValueError is raised if raw_query is empty or whitespace."""
    with pytest.raises(ValueError):
        planner({"raw_query": ""})

    with pytest.raises(ValueError):
        planner({"raw_query": "   "})


def test_retry_once_and_raise_planner_error():
    """Verify that node retries once before raising PlannerError on repeated failure."""
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
    state: ResearchState = {
        "raw_query": "Test planner error handling",
        "intent": "comparative",
        "entities": [],
        "constraints": [],
        "difficulty": "deep",
    }

    with pytest.raises(PlannerError) as exc_info:
        planner(state, llm=failing_llm)

    assert runnable.call_count == 2
    assert "Planner failed after 1 retry" in str(exc_info.value)
