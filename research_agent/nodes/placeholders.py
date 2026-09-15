"""Placeholder branch nodes for Stage 2 conditional routing."""

from research_agent.state import ResearchState


def shallow_placeholder(state: ResearchState) -> ResearchState:
    """Pass-through placeholder node for shallow research branch.

    Args:
        state: Current ResearchState.

    Returns:
        ResearchState unchanged.
    """
    print("  -> Reached branch: shallow_placeholder")
    return state


def direct_placeholder(state: ResearchState) -> ResearchState:
    """Pass-through placeholder node for direct answer branch.

    Args:
        state: Current ResearchState.

    Returns:
        ResearchState unchanged.
    """
    print("  -> Reached branch: direct_placeholder")
    return state


# TODO: Unused since Stage 3 (replaced by planner node). Remove in final cleanup pass before project submission.
def deep_placeholder(state: ResearchState) -> ResearchState:
    """Pass-through placeholder node for deep research branch.

    Args:
        state: Current ResearchState.

    Returns:
        ResearchState unchanged.
    """
    print("  -> Reached branch: deep_placeholder")
    return state
