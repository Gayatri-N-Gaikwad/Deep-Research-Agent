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
            "'direct' (timeless, mathematically or physically fixed facts, arithmetic, physical/chemical constants, or settled definitions that cannot change in the future), "
            "'shallow' (single lookup of institutional, political, demographic, or statistical facts that could plausibly change over time), "
            "or 'deep' (requires multi-hop reasoning, decomposition, comparison across multiple entities, or synthesis)."
        )
    )
    reasoning: str = Field(
        description="A short explanation of why this difficulty tier was assigned, applying the timeless vs. time-variable test."
    )


DIFFICULTY_ROUTER_SYSTEM_PROMPT = """You are an expert research routing engine in a multi-stage research pipeline.
Your task is to analyze the user's research query and its previously extracted metadata (intent, entities, constraints) to classify the research difficulty into exactly one of three tiers: 'direct', 'shallow', or 'deep'.

CLASSIFICATION PRINCIPLES & RULES:

1. 'direct' (Timeless Facts & Direct Computation):
   - The answer is timeless: it is mathematically, physically, or scientifically fixed and cannot change regardless of when someone asks.
   - Core Decision Test: "Could this answer ever be different in the future?" If NO, it is 'direct'. There is zero verification value in performing a web search.
   - Includes:
     * Arithmetic, unit conversions, and calculations (e.g. "What is 15 times 12?", "Convert 50 miles to kilometers").
     * Mathematical constants and properties (e.g. "What is the value of pi?", "What is a prime number?").
     * Physical and chemical constants or standard properties (e.g. "What is the freezing point of water in Celsius?", "What is the speed of light in a vacuum?", "What is the boiling point of water at sea level?").
     * Settled scientific definitions and concepts (e.g. "Define photosynthesis", "What is an electron?").
     * Historical facts about completed past events (e.g. "When was the Eiffel Tower built?", "When did Apollo 11 land on the moon?").

2. 'shallow' (Time-Variable or Institutional Single Lookup):
   - The answer is currently stable but could plausibly change over time — it depends on external institutional, geopolitical, demographic, or market state.
   - Core Decision Test: "Could this answer theoretically change over time (even if slowly or rarely)?" If YES, it is 'shallow'. It requires a single targeted web search to confirm the latest current status.
   - Includes:
     * Institutional or political facts (e.g. "What is the capital of France?", "What is the capital of Australia?", "Who is the CEO of Apple?", "Who is the current Secretary-General of the United Nations?").
     * Demographic, economic, or statistical figures (e.g. "What is the current population of Japan?", "What is the GDP of Germany?").
     * Dynamic current data or queries phrased with "current", "latest", "now", or "today" (e.g. "What is the current price of gold?").

3. 'deep' (Multi-Hop, Decomposition & Synthesis):
   - Requires multi-step reasoning, query decomposition, comparison across multiple entities, multi-part conjunctions, or multi-faceted synthesis across multiple sources.
   - Any query comparing multiple countries, policies, entities, or complex systems.
   - Any query with multiple explicit operational constraints stacked together (e.g. time range + format limits + scope limits).
   - Examples:
     * "Compare the economic policies of the United States and Japan"
     * "Summarize developments in quantum computing since 2023, in under 200 words"
     * "Analyze the geopolitical and economic impact of the renewable energy transition in Europe"

CRITICAL RULES:
- Base your classification strictly on the user's query and extracted intent, entities, and constraints.
- Apply the timeless vs. time-variable test to distinguish between 'direct' and 'shallow'.
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
