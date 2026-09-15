"""Difficulty Router node for the research agent pipeline."""

import os
from typing import Literal, Optional

from dotenv import load_dotenv
from pydantic import BaseModel, Field

from research_agent.nodes.llm_providers import get_structured_llm_for_provider
from research_agent.state import Difficulty, ResearchState

load_dotenv()


class DifficultyRouterError(RuntimeError):
    """Raised when difficulty router analysis fails after retrying."""
    pass


class DifficultyClassification(BaseModel):
    """Pydantic schema for structured output from the router LLM.

    Classifies query difficulty into shallow, direct, or deep, along with a brief reasoning.
    """

    difficulty: Literal["shallow", "direct", "deep"] = Field(
        description=(
            "The research difficulty tier for the query: "
            "'shallow' (single narrow factual lookup answerable with one search), "
            "'direct' (answerable from general knowledge/arithmetic with no search needed), "
            "or 'deep' (requires multi-hop reasoning, decomposition, multiple entities, or synthesis)."
        )
    )
    reasoning: str = Field(
        description="A short explanation of why this difficulty tier was assigned."
    )


DIFFICULTY_ROUTER_SYSTEM_PROMPT = """You are an expert research routing engine in a multi-stage research pipeline.
Your task is to analyze the user's research query and its previously extracted metadata (intent, entities, constraints) to classify the research difficulty into exactly one of three tiers:

1. 'shallow':
   - A single, narrow factual lookup answerable with one targeted search.
   - Examples: "What is the capital of France?", "Who is the CEO of Apple?", "When was the Eiffel Tower built?"

2. 'direct':
   - Answerable directly from general knowledge or simple deduction/arithmetic with no web search needed at all.
   - Examples: "What is 15 times 12?", "Define photosynthesis", "What is the boiling point of water at sea level?", "How many hours are in a day?"

3. 'deep':
   - Requires decomposition into multiple sub-questions, comparison across multiple entities, multi-hop reasoning, or synthesis across several sources.
   - Any query with multiple explicit constraints stacked together (e.g., time range + format limits + scope limits) or comparing multiple countries, policies, or complex systems.
   - Examples: "Compare the economic policies of the United States and Japan", "Summarize developments in quantum computing since 2023, in under 200 words", "Analyze the geopolitical impact of renewable energy transition in Europe".

CRITICAL RULES:
- Base your classification on the query and extracted intent, entities, and constraints.
- DO NOT answer or solve the query itself.
- Return structured output conforming strictly to the requested schema.
"""


def get_structured_llm(llm=None):
    """Instantiate structured output LLM for difficulty routing.

    Uses NVIDIA NIM (Nemotron 3 Super with reasoning disabled) by default, or wraps the provided llm if injected.

    Args:
        llm: Optional underlying chat model. If None, instantiates NVIDIA NIM model via provider registry.

    Returns:
        A Runnable returning validated DifficultyClassification instances.
    """
    if llm is not None:
        return llm.with_structured_output(DifficultyClassification)

    return get_structured_llm_for_provider("nvidia_nim", DifficultyClassification)


def difficulty_router(state: ResearchState, llm=None) -> ResearchState:
    """Classify the research difficulty of the query into shallow, direct, or deep.

    Reads raw_query, intent, entities, and constraints from state, invokes
    gemini-3.6-flash with structured output to classify difficulty, and returns
    updated ResearchState with 'difficulty' set.

    Retries once on LLM failure or malformed output before raising DifficultyRouterError.

    Args:
        state: The current ResearchState dictionary.
        llm: Optional custom or mocked LLM instance for testing.

    Returns:
        Updated ResearchState dictionary with 'difficulty' set.

    Raises:
        ValueError: If 'raw_query' is missing or empty.
        DifficultyRouterError: If classification fails after 1 retry.
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
        {"role": "system", "content": DIFFICULTY_ROUTER_SYSTEM_PROMPT},
        {"role": "user", "content": f"Analyze query difficulty based on the following query data:\n{context_str}"},
    ]

    structured_llm = get_structured_llm(llm=llm)

    last_error: Optional[Exception] = None
    max_attempts = 2  # 1 initial attempt + 1 retry

    for attempt in range(max_attempts):
        try:
            result = structured_llm.invoke(messages)
            if isinstance(result, dict):
                result = DifficultyClassification(**result)
            elif not isinstance(result, DifficultyClassification):
                raise ValueError(f"Unexpected response format from structured LLM: {type(result)}")

            return {
                **state,
                "difficulty": result.difficulty,
            }
        except Exception as exc:
            last_error = exc

    raise DifficultyRouterError(
        f"Difficulty router failed after 1 retry for query '{raw_query}'. Error: {last_error}"
    ) from last_error
