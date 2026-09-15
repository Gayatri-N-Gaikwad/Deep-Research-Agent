"""Query Understanding node for the research agent pipeline."""

import os
from typing import Literal, Optional

from dotenv import load_dotenv
from pydantic import BaseModel, Field

from research_agent.nodes.llm_providers import get_structured_llm_for_provider
from research_agent.state import ResearchState

load_dotenv()


class QueryUnderstandingError(RuntimeError):
    """Raised when query understanding analysis fails after retrying."""
    pass


class QueryAnalysis(BaseModel):
    """Pydantic schema for structured output from the LLM.

    Extracts query intent, entities, and explicit constraints without answering the query.
    Note: Language is intentionally omitted here to prevent discrepancies with the
    hardcoded 'en' pipeline state.
    """

    intent: Literal["factual", "comparative", "exploratory", "procedural"] = Field(
        description=(
            "Classification of query intent: "
            "'factual' (direct facts, definitions, specific single data points), "
            "'comparative' (comparing two or more entities, policies, or concepts), "
            "'exploratory' (broad, open-ended, ambiguous, or survey topics), "
            "or 'procedural' (step-by-step instructions, how-to guides, workflows)."
        )
    )
    entities: list[str] = Field(
        default_factory=list,
        description="Key entities, subjects, proper nouns, organizations, or concepts extracted from the query.",
    )
    constraints: list[str] = Field(
        default_factory=list,
        description="Explicit constraints in the query (e.g. time range, format/length limits, scope limits). Empty list if none.",
    )


QUERY_UNDERSTANDING_SYSTEM_PROMPT = """You are an expert query understanding engine in a multi-stage research pipeline.
Your sole job is to analyze the user's research query and extract metadata according to the structured schema.

Guidelines:
1. Intent:
   - 'factual': queries seeking a specific fact, definition, scientific value, date, or direct answer.
   - 'comparative': queries asking to compare, contrast, or evaluate differences/similarities between two or more entities, policies, or topics.
   - 'exploratory': broad, ambiguous, open-ended, or overview queries requiring wide research.
   - 'procedural': queries asking how to do something or requiring step-by-step instructions.
2. Entities:
   - Extract key topics, organizations, countries, individuals, technologies, or subjects explicitly mentioned (including scientific context like 'sea level', 'water').
3. Constraints:
   - Extract explicit operational or output constraints: time boundaries (e.g. 'since 2023', 'in the 21st century'), length/format restrictions (e.g. 'under 200 words', 'in bullet points'), or scope boundaries on the research task itself.
   - Do NOT treat scientific conditions or measurement context (e.g., 'at sea level', 'at room temperature', 'in Celsius') as constraints; treat them as entities/context.
   - If no explicit constraints exist, return an empty list [].

CRITICAL RULES:
- Analyze ONLY the query provided.
- Under NO circumstances should you answer, solve, summarize, or elaborate on the query itself.
- Only extract the requested metadata fields.
"""


def get_structured_llm(llm=None):
    """Instantiate structured output LLM for query understanding.

    Uses Groq (openai/gpt-oss-120b) by default, or wraps the provided llm if injected.

    Args:
        llm: Optional underlying chat model. If None, instantiates Groq model via provider registry.

    Returns:
        A Runnable returning validated QueryAnalysis instances.
    """
    if llm is not None:
        return llm.with_structured_output(QueryAnalysis)

    return get_structured_llm_for_provider("groq", QueryAnalysis)


def query_understanding(state: ResearchState, llm=None) -> ResearchState:
    """Analyze the user's raw query and extract intent, entities, and constraints.

    Takes ResearchState containing 'raw_query', invokes gemini-3.6-flash with
    structured output to extract query intent, entities, and constraints, hardcodes
    language to 'en', and returns the updated ResearchState.

    Retries once on LLM failure or malformed output before raising QueryUnderstandingError.

    Args:
        state: The current ResearchState dictionary.
        llm: Optional custom or mocked LLM instance for testing.

    Returns:
        Updated ResearchState dictionary with intent, entities, constraints, and language.

    Raises:
        ValueError: If 'raw_query' is missing or empty.
        QueryUnderstandingError: If extraction fails after 1 retry.
    """
    raw_query = state.get("raw_query")
    if not raw_query or not str(raw_query).strip():
        raise ValueError("ResearchState 'raw_query' must be a non-empty string.")

    structured_llm = get_structured_llm(llm=llm)

    messages = [
        {"role": "system", "content": QUERY_UNDERSTANDING_SYSTEM_PROMPT},
        {"role": "user", "content": f"Query: {raw_query}"},
    ]

    last_error: Optional[Exception] = None
    max_attempts = 2  # 1 initial attempt + 1 retry

    for attempt in range(max_attempts):
        try:
            result = structured_llm.invoke(messages)
            if isinstance(result, dict):
                result = QueryAnalysis(**result)
            elif not isinstance(result, QueryAnalysis):
                raise ValueError(f"Unexpected response format from structured LLM: {type(result)}")

            return {
                **state,
                "raw_query": raw_query,
                "language": "en",
                "intent": result.intent,
                "entities": result.entities,
                "constraints": result.constraints,
            }
        except Exception as exc:
            last_error = exc

    raise QueryUnderstandingError(
        f"Query understanding failed after 1 retry for query '{raw_query}'. Error: {last_error}"
    ) from last_error
