"""Golden boundary test cases for Difficulty Router: shallow vs. direct classification.

Validates the timeless vs. time-variable classification boundary:
- 'direct': mathematically or physically fixed, timeless constants and settled facts
            ("could this answer ever be different in the future?" -> NO)
- 'shallow': institutional, geopolitical, demographic, or statistical facts that could plausibly change over time
             ("could this answer ever be different in the future?" -> YES)

Provides both:
1. Fast offline unit tests (mocked LLM) verifying state propagation and schema contracts.
2. Live integration tests (marked with @pytest.mark.integration) calling the real router LLM.
"""

from dataclasses import dataclass
import os
from typing import Literal

from dotenv import load_dotenv
import pytest

from research_agent.nodes.difficulty_router import (
    DifficultyClassification,
    difficulty_router,
)
from research_agent.state import ResearchState

load_dotenv()


@dataclass(frozen=True)
class BoundaryCase:
    """Specification of a golden difficulty-router boundary benchmark case."""

    id: str
    query: str
    expected_difficulty: Literal["direct", "shallow", "deep"]
    category: str
    rationale: str


GOLDEN_BOUNDARY_CASES = [
    BoundaryCase(
        id="freezing_point_water",
        query="What is the freezing point of water in Celsius?",
        expected_difficulty="direct",
        category="physical_constant",
        rationale="Timeless physical constant under standard conditions; answer cannot change in the future.",
    ),
    BoundaryCase(
        id="value_of_pi",
        query="What is the value of pi?",
        expected_difficulty="direct",
        category="mathematical_constant",
        rationale="Timeless mathematical constant; fixed forever.",
    ),
    BoundaryCase(
        id="speed_of_light",
        query="What is the speed of light in a vacuum?",
        expected_difficulty="direct",
        category="physical_constant",
        rationale="Universal physical constant; cannot change in the future.",
    ),
    BoundaryCase(
        id="boiling_point_water",
        query="What is the boiling point of water at sea level?",
        expected_difficulty="direct",
        category="physical_constant",
        rationale="Settled physical property under standard atmospheric conditions; cannot change in the future.",
    ),
    BoundaryCase(
        id="capital_australia",
        query="What is the capital of Australia?",
        expected_difficulty="shallow",
        category="institutional_geopolitical",
        rationale="Geopolitical/institutional fact that could theoretically change over time.",
    ),
    BoundaryCase(
        id="population_japan",
        query="What is the current population of Japan?",
        expected_difficulty="shallow",
        category="demographic_statistical",
        rationale="Actively fluctuating statistical metric requiring up-to-date retrieval.",
    ),
    BoundaryCase(
        id="un_sec_gen",
        query="Who is the current Secretary-General of the United Nations?",
        expected_difficulty="shallow",
        category="current_officeholder",
        rationale="Institutional officeholder whose tenure expires and changes periodically.",
    ),
]


class MockLLMWrapper:
    """Mock LLM returning pre-configured DifficultyClassification for offline unit tests."""

    def __init__(self, response: DifficultyClassification):
        self.response = response

    def with_structured_output(self, schema):
        class MockRunnable:
            def __init__(self, response):
                self.response = response

            def invoke(self, messages):
                return self.response

        return MockRunnable(self.response)


# ==============================================================================
# 1. Offline Mocked Unit Tests (Zero API calls, runs with pytest -m "not integration")
# ==============================================================================


@pytest.mark.parametrize("case", GOLDEN_BOUNDARY_CASES, ids=lambda c: f"mocked_{c.id}")
def test_mocked_boundary_case_contract(case: BoundaryCase):
    """Verify state propagation and schema contract for boundary cases."""
    mock_response = DifficultyClassification(
        difficulty=case.expected_difficulty,
        reasoning=case.rationale,
    )
    llm = MockLLMWrapper(mock_response)
    state: ResearchState = {
        "raw_query": case.query,
        "intent": "factual",
        "entities": [],
        "constraints": [],
    }
    result = difficulty_router(state, llm=llm)

    assert result["difficulty"] == case.expected_difficulty
    assert result["raw_query"] == case.query


# ==============================================================================
# 2. Live Integration Tests (Real LLM calls, runs with pytest -m integration -k boundary)
# ==============================================================================


@pytest.mark.integration
@pytest.mark.skipif(
    not os.getenv("NVIDIA_API_KEY"),
    reason="Requires NVIDIA_API_KEY set in environment or .env file",
)
@pytest.mark.parametrize("case", GOLDEN_BOUNDARY_CASES, ids=lambda c: f"boundary_{c.id}")
def test_live_difficulty_router_boundary_case(case: BoundaryCase):
    """Live boundary test: executes the actual router node with NVIDIA NIM on boundary cases."""
    state: ResearchState = {
        "raw_query": case.query,
        "intent": "factual",
        "entities": [],
        "constraints": [],
    }
    result = difficulty_router(state)
    actual_difficulty = result.get("difficulty")

    assert actual_difficulty == case.expected_difficulty, (
        f"\nBoundary query misclassified by router:"
        f"\n  Query:    '{case.query}'"
        f"\n  Category: {case.category}"
        f"\n  Expected: '{case.expected_difficulty}'"
        f"\n  Actual:   '{actual_difficulty}'"
        f"\n  Rationale: {case.rationale}"
    )
