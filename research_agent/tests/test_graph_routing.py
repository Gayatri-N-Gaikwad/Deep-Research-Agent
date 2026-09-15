"""Tests for LangGraph conditional routing based on difficulty."""

from unittest.mock import patch
import pytest

from research_agent.graph import build_graph, route_by_difficulty
from research_agent.nodes.difficulty_router import DifficultyClassification
from research_agent.nodes.planner import ResearchPlan
from research_agent.nodes.query_understanding import QueryAnalysis
from research_agent.state import ResearchState, create_initial_state


class MockRunnable:
    """Mock Runnable returning pre-defined structured output."""

    def __init__(self, response):
        self.response = response

    def invoke(self, *args, **kwargs):
        return self.response


class MockTavilySearchClient:
    """Mock Tavily client for graph routing test."""

    def search(self, query: str, max_results: int = 5):
        return {
            "results": [
                {
                    "url": f"https://example.com/{abs(hash(query))}/item",
                    "title": f"Title for {query[:20]}",
                    "content": f"Content summary for {query[:20]}",
                }
            ]
        }


def test_route_by_difficulty_raises_value_error_on_invalid():
    """Verify route_by_difficulty raises ValueError if difficulty is missing or invalid."""
    with pytest.raises(ValueError) as exc_info:
        route_by_difficulty({"raw_query": "test"})
    assert "Invalid or missing difficulty" in str(exc_info.value)

    with pytest.raises(ValueError) as exc_info:
        route_by_difficulty({"raw_query": "test", "difficulty": None})
    assert "Invalid or missing difficulty" in str(exc_info.value)

    with pytest.raises(ValueError) as exc_info:
        route_by_difficulty({"raw_query": "test", "difficulty": "unknown_tier"})
    assert "Expected one of 'shallow', 'direct', or 'deep'" in str(exc_info.value)


@pytest.mark.parametrize(
    "target_difficulty, expected_branch",
    [
        ("shallow", "shallow_placeholder"),
        ("direct", "direct_placeholder"),
    ],
)
def test_graph_conditional_routing_placeholders(target_difficulty, expected_branch, capsys):
    """Verify that build_graph routes to the correct placeholder node for shallow and direct tiers."""
    mock_query_output = QueryAnalysis(
        intent="factual",
        entities=["test entity"],
        constraints=[],
    )
    mock_router_output = DifficultyClassification(
        difficulty=target_difficulty,
        reasoning=f"Classified as {target_difficulty}",
    )

    with patch(
        "research_agent.nodes.query_understanding.get_structured_llm",
        return_value=MockRunnable(mock_query_output),
    ), patch(
        "research_agent.nodes.difficulty_router.get_structured_llm",
        return_value=MockRunnable(mock_router_output),
    ):
        app = build_graph()
        initial_state = create_initial_state(f"Test query for {target_difficulty}")
        final_state = app.invoke(initial_state)

        # Assert final state contains correct classification
        assert final_state["difficulty"] == target_difficulty

        # Assert that the correct placeholder node was reached via captured stdout
        captured = capsys.readouterr()
        assert f"Reached branch: {expected_branch}" in captured.out

        # Verify other placeholder branch was not reached
        other_branch = "direct_placeholder" if target_difficulty == "shallow" else "shallow_placeholder"
        assert f"Reached branch: {other_branch}" not in captured.out
        assert "Reached branch: deep_placeholder" not in captured.out


def test_graph_conditional_routing_deep_reaches_search_tracks(capsys):
    """Verify build_graph routes 'deep' queries to planner then search_tracks and populates evidence."""
    mock_query_output = QueryAnalysis(
        intent="comparative",
        entities=["United States", "Japan"],
        constraints=[],
    )
    mock_router_output = DifficultyClassification(
        difficulty="deep",
        reasoning="Multi-entity comparative study requiring decomposition.",
    )
    expected_plan = "Analyze US and Japanese fiscal/monetary frameworks, then compare direct outcomes."
    expected_sub_queries = [
        "What are the foundational macroeconomic and monetary policies of the United States?",
        "What are the core economic and monetary policies of Japan?",
        "How do the economic outcomes and debt dynamics of the US and Japan directly compare?",
    ]
    mock_planner_output = ResearchPlan(
        plan=expected_plan,
        sub_queries=expected_sub_queries,
    )
    mock_tavily = MockTavilySearchClient()

    with patch(
        "research_agent.nodes.query_understanding.get_structured_llm",
        return_value=MockRunnable(mock_query_output),
    ), patch(
        "research_agent.nodes.difficulty_router.get_structured_llm",
        return_value=MockRunnable(mock_router_output),
    ), patch(
        "research_agent.nodes.planner.get_structured_llm",
        return_value=MockRunnable(mock_planner_output),
    ), patch(
        "research_agent.nodes.search_tracks.get_tavily_client",
        return_value=mock_tavily,
    ):
        app = build_graph()
        initial_state = create_initial_state("Compare the economic policies of the United States and Japan")
        final_state = app.invoke(initial_state)

        # Assert final state contains correct classification and planner output
        assert final_state["difficulty"] == "deep"
        assert final_state["plan"] == expected_plan
        assert final_state["sub_queries"] == expected_sub_queries
        assert len(final_state["sub_queries"]) == 3

        # Assert search_tracks node ran and populated evidence
        assert "evidence" in final_state
        assert len(final_state["evidence"]) == 3
        track_ids = {item["track_id"] for item in final_state["evidence"]}
        assert track_ids == {0, 1, 2}

        # Assert placeholder branches were NOT reached
        captured = capsys.readouterr()
        assert "Reached branch: shallow_placeholder" not in captured.out
        assert "Reached branch: direct_placeholder" not in captured.out
        assert "Reached branch: deep_placeholder" not in captured.out
