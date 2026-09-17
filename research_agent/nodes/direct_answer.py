"""Direct Answer node for the research agent pipeline."""

from typing import Optional

from dotenv import load_dotenv
from pydantic import BaseModel, Field

from research_agent.nodes.llm_providers import get_structured_llm_for_provider
from research_agent.state import ResearchState

load_dotenv()


class DirectAnswerError(RuntimeError):
    """Raised when direct answer generation fails after retrying."""
    pass


class DirectAnswer(BaseModel):
    """Pydantic schema for structured direct answer output."""

    answer: str = Field(
        description="Concise, direct answer generated solely from model knowledge without searching external sources."
    )
    confidence: str = Field(
        description="Assessment of confidence in the provided answer (e.g. 'high', 'medium', 'low')."
    )


DIRECT_ANSWER_SYSTEM_PROMPT = """You are a direct query answering engine in a multi-stage research pipeline.
Your job is to answer the user's query directly and concisely from your pre-trained model knowledge.

Guidelines:
1. Answer directly, concisely, and accurately based ONLY on your established internal knowledge.
2. State plainly when you are not confident or lack sufficient knowledge to answer, rather than guessing.
3. Under NO circumstances should you fabricate sources, URLs, citations, or external references.
"""


def get_structured_llm(llm=None):
    """Instantiate structured output LLM for direct answer generation.

    Uses Groq (openai/gpt-oss-120b) by default, or wraps the provided llm if injected.

    Args:
        llm: Optional underlying chat model. If None, instantiates Groq model via provider registry.

    Returns:
        A Runnable returning validated DirectAnswer instances.
    """
    if llm is not None:
        return llm.with_structured_output(DirectAnswer)

    return get_structured_llm_for_provider("groq", DirectAnswer)


def direct_answer(state: ResearchState, llm=None) -> ResearchState:
    """Generate a direct answer to the query using model knowledge without external search.

    Takes ResearchState containing 'raw_query', invokes Groq with structured output
    to produce a direct answer, and returns updated state with 'answer' set.

    Retries once on LLM failure before raising DirectAnswerError.

    Args:
        state: The current ResearchState dictionary.
        llm: Optional custom or mocked LLM instance for testing.

    Returns:
        Updated ResearchState dictionary with 'answer' populated.

    Raises:
        ValueError: If 'raw_query' is missing or empty.
        DirectAnswerError: If answer generation fails after 1 retry.
    """
    raw_query = state.get("raw_query")
    if not raw_query or not str(raw_query).strip():
        raise ValueError("ResearchState 'raw_query' must be a non-empty string.")

    structured_llm = get_structured_llm(llm=llm)

    messages = [
        {"role": "system", "content": DIRECT_ANSWER_SYSTEM_PROMPT},
        {"role": "user", "content": f"Query: {raw_query}"},
    ]

    last_error: Optional[Exception] = None
    max_attempts = 2

    for attempt in range(max_attempts):
        try:
            result = structured_llm.invoke(messages)
            if isinstance(result, dict):
                result = DirectAnswer(**result)
            elif not isinstance(result, DirectAnswer):
                raise ValueError(f"Unexpected response format from structured LLM: {type(result)}")

            return {
                **state,
                "answer": result.answer,
            }
        except Exception as exc:
            last_error = exc

    raise DirectAnswerError(
        f"Direct answer generation failed after 1 retry for query '{raw_query}'. Error: {last_error}"
    ) from last_error
