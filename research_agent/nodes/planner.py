"""Planner and Query Decomposition node for the deep-research path."""

import os
from typing import Optional

from dotenv import load_dotenv
from pydantic import BaseModel, Field

from research_agent.nodes.llm_providers import get_structured_llm_for_provider
from research_agent.state import ResearchState

load_dotenv()


class PlannerError(RuntimeError):
    """Raised when the planner node fails to generate a research plan after retrying."""
    pass


class ResearchPlan(BaseModel):
    """Pydantic schema for structured output from the Planner LLM.

    Contains an overarching research plan and 2-3 decomposed sub-queries.
    """

    plan: str = Field(
        description="1-2 sentence high-level description of the research strategy."
    )
    sub_queries: list[str] = Field(
        description=(
            "2 to 3 focused, independently-searchable sub-questions that together "
            "thoroughly address the original query from distinct angles."
        )
    )


PLANNER_SYSTEM_PROMPT = """You are an expert research planner in a multi-stage research pipeline.
Your task is to analyze a complex research query and its previously extracted metadata (intent, entities, constraints) to construct a structured research plan and decompose the query into 2 to 3 focused, independently-searchable sub-queries.

DECOMPOSITION RULES:
1. Genuinely decompose the topic into distinct angles, sub-topics, entities, or aspects.
2. Sub-queries must NOT be mere paraphrases or restatements of the original query.
3. For comparative queries (e.g., "Compare X and Y"):
   - Formulate sub-queries addressing Entity X specifically, Entity Y specifically, and/or a direct comparative analysis between them.
4. For multi-aspect queries (e.g., questions involving causes, events, consequences, or multiple perspectives):
   - Formulate sub-queries addressing the distinct aspects (e.g., one sub-query on root causes/origins, one on core events/mechanisms, one on long-term consequences/outcomes).
5. Ensure exactly 2 or 3 sub-queries are generated. Each sub-query must be standalone and directly searchable by a downstream retrieval engine.
6. Formulate a concise 1-2 sentence overall research plan summarizing the strategy.

CRITICAL RULES:
- Base the plan on the query, intent, entities, and constraints provided.
- DO NOT answer or synthesize the query itself. Only plan and decompose.
- Return structured output conforming strictly to the requested schema.
"""


def get_structured_llm(llm=None):
    """Instantiate structured output LLM for research planning.

    Uses Gemini (gemini-3.6-flash) by default, or wraps the provided llm if injected.

    Args:
        llm: Optional underlying chat model. If None, instantiates Gemini model via provider registry.

    Returns:
        A Runnable returning validated ResearchPlan instances.
    """
    if llm is not None:
        return llm.with_structured_output(ResearchPlan)

    return get_structured_llm_for_provider("gemini", ResearchPlan)


def planner(state: ResearchState, llm=None) -> ResearchState:
    """Decompose a deep research query into a research plan and 2-3 sub-queries.

    Reads raw_query, intent, entities, and constraints from state, invokes
    gemini-3.6-flash with structured output, and returns updated ResearchState
    with 'plan' and 'sub_queries' populated.

    Retries once on LLM failure or malformed output before raising PlannerError.

    Args:
        state: The current ResearchState dictionary.
        llm: Optional custom or mocked LLM instance for testing.

    Returns:
        Updated ResearchState dictionary with 'plan' and 'sub_queries' set.

    Raises:
        ValueError: If 'raw_query' is missing or empty.
        PlannerError: If plan generation fails after 1 retry.
    """
    raw_query = state.get("raw_query")
    if not raw_query or not str(raw_query).strip():
        raise ValueError("ResearchState 'raw_query' must be a non-empty string.")

    intent = state.get("intent", "")
    entities = state.get("entities", [])
    constraints = state.get("constraints", [])

    context_str = (
        f"Raw Query: {raw_query}\n"
        f"Extracted Intent: {intent}\n"
        f"Extracted Entities: {entities}\n"
        f"Extracted Constraints: {constraints}"
    )

    messages = [
        {"role": "system", "content": PLANNER_SYSTEM_PROMPT},
        {"role": "user", "content": f"Construct a research plan and decomposed sub-queries for the following query data:\n{context_str}"},
    ]

    structured_llm = get_structured_llm(llm=llm)

    last_error: Optional[Exception] = None
    max_attempts = 2  # 1 initial attempt + 1 retry

    for attempt in range(max_attempts):
        try:
            result = structured_llm.invoke(messages)
            if isinstance(result, dict):
                result = ResearchPlan(**result)
            elif not isinstance(result, ResearchPlan):
                raise ValueError(f"Unexpected response format from structured LLM: {type(result)}")

            return {
                **state,
                "plan": result.plan,
                "sub_queries": result.sub_queries,
            }
        except Exception as exc:
            last_error = exc

    raise PlannerError(
        f"Planner failed after 1 retry for query '{raw_query}'. Error: {last_error}"
    ) from last_error
