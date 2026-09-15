"""LangGraph construction and compilation for the research agent pipeline."""

from langgraph.graph import END, START, StateGraph
from langgraph.graph.state import CompiledStateGraph

from research_agent.nodes.difficulty_router import difficulty_router
from research_agent.nodes.placeholders import (
    direct_placeholder,
    shallow_placeholder,
)
from research_agent.nodes.planner import planner
from research_agent.nodes.query_understanding import query_understanding
from research_agent.nodes.search_tracks import search_tracks
from research_agent.state import ResearchState


def route_by_difficulty(state: ResearchState) -> str:
    """Route graph execution based on classified difficulty.

    Args:
        state: The current ResearchState.

    Returns:
        The difficulty string matching the branch name ('shallow', 'direct', or 'deep').

    Raises:
        ValueError: If difficulty is missing or invalid.
    """
    difficulty = state.get("difficulty")
    if difficulty not in ("shallow", "direct", "deep"):
        raise ValueError(
            f"Invalid or missing difficulty in state: '{difficulty}'. "
            "Expected one of 'shallow', 'direct', or 'deep'."
        )
    return difficulty


def build_graph() -> CompiledStateGraph:
    """Build and compile the Stage 4 research pipeline StateGraph.

    Constructs a StateGraph parameterized by ResearchState with:
    1. START -> query_understanding
    2. query_understanding -> difficulty_router
    3. difficulty_router -> conditional routing (route_by_difficulty)
       - 'shallow' -> shallow_placeholder -> END
       - 'direct'  -> direct_placeholder -> END
       - 'deep'    -> planner -> search_tracks -> END

    Returns:
        CompiledStateGraph: The compiled runnable LangGraph instance.
    """
    workflow = StateGraph(ResearchState)

    # Add pipeline nodes
    workflow.add_node("query_understanding", query_understanding)
    workflow.add_node("difficulty_router", difficulty_router)
    workflow.add_node("shallow_placeholder", shallow_placeholder)
    workflow.add_node("direct_placeholder", direct_placeholder)
    workflow.add_node("planner", planner)
    workflow.add_node("search_tracks", search_tracks)

    # Wire entry sequence
    workflow.add_edge(START, "query_understanding")
    workflow.add_edge("query_understanding", "difficulty_router")

    # Conditional branching based on difficulty
    workflow.add_conditional_edges(
        "difficulty_router",
        route_by_difficulty,
        {
            "shallow": "shallow_placeholder",
            "direct": "direct_placeholder",
            "deep": "planner",
        },
    )

    # Route branches to END
    workflow.add_edge("shallow_placeholder", END)
    workflow.add_edge("direct_placeholder", END)
    workflow.add_edge("planner", "search_tracks")
    workflow.add_edge("search_tracks", END)

    return workflow.compile()


