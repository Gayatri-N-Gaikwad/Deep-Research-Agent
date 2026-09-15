"""Live integration tests calling external LLM providers across pipeline nodes.

Tests exercise actual node functions against their assigned providers:
- query_understanding -> Groq (requires GROQ_API_KEY)
- difficulty_router -> NVIDIA NIM (requires NVIDIA_API_KEY)
- planner -> Gemini (requires GOOGLE_API_KEY)

Run explicitly with: pytest -m integration
"""

import os
import pytest
from dotenv import load_dotenv

from research_agent.nodes.difficulty_router import difficulty_router
from research_agent.nodes.planner import planner
from research_agent.nodes.query_understanding import query_understanding
from research_agent.state import ResearchState

load_dotenv()

pytestmark = [pytest.mark.integration]


# ==============================================================================
# Query Understanding Live Tests (Groq via openai/gpt-oss-120b)
# ==============================================================================

@pytest.mark.skipif(
    not os.getenv("GROQ_API_KEY"),
    reason="Requires GROQ_API_KEY set in environment or .env file",
)
def test_live_factual_query():
    """Live API test: Clear factual query using Groq."""
    state: ResearchState = {
        "raw_query": "What is the boiling point of water at sea level?"
    }
    result = query_understanding(state)

    assert result["intent"] == "factual"
    assert isinstance(result["constraints"], list)
    assert len(result["constraints"]) == 0
    assert result["language"] == "en"
    assert isinstance(result["entities"], list)
    assert len(result["entities"]) > 0
    entities_lower = [e.lower() for e in result["entities"]]
    assert any("water" in e or "boiling point" in e for e in entities_lower)


@pytest.mark.skipif(
    not os.getenv("GROQ_API_KEY"),
    reason="Requires GROQ_API_KEY set in environment or .env file",
)
def test_live_comparative_query():
    """Live API test: Comparative query using Groq."""
    state: ResearchState = {
        "raw_query": "Compare the economic policies of the United States and Japan"
    }
    result = query_understanding(state)

    assert result["intent"] == "comparative"
    assert result["language"] == "en"
    assert isinstance(result["entities"], list)
    entities_lower = [e.lower() for e in result["entities"]]
    assert any("united states" in e or "usa" in e or "us" in e for e in entities_lower)
    assert any("japan" in e for e in entities_lower)


@pytest.mark.skipif(
    not os.getenv("GROQ_API_KEY"),
    reason="Requires GROQ_API_KEY set in environment or .env file",
)
def test_live_query_with_explicit_constraints():
    """Live API test: Query with explicit time and format limits using Groq."""
    state: ResearchState = {
        "raw_query": "Summarize developments in quantum computing since 2023, in under 200 words"
    }
    result = query_understanding(state)

    assert isinstance(result["constraints"], list)
    assert len(result["constraints"]) > 0
    constraints_text = " ".join(result["constraints"]).lower()
    assert "2023" in constraints_text or "200" in constraints_text
    assert result["language"] == "en"


@pytest.mark.skipif(
    not os.getenv("GROQ_API_KEY"),
    reason="Requires GROQ_API_KEY set in environment or .env file",
)
def test_live_ambiguous_short_query():
    """Live API test: Ambiguous/short query using Groq."""
    state: ResearchState = {
        "raw_query": "AI"
    }
    result = query_understanding(state)

    assert result is not None
    assert result["intent"] in ["exploratory", "factual"]
    assert result["language"] == "en"
    assert isinstance(result["entities"], list)
    entities_lower = [e.lower() for e in result["entities"]]
    assert any("ai" in e or "artificial intelligence" in e for e in entities_lower)


@pytest.mark.skipif(
    not os.getenv("GROQ_API_KEY"),
    reason="Requires GROQ_API_KEY set in environment or .env file",
)
def test_live_query_understanding_via_groq():
    """Live API test: Explicit test exercising query_understanding node via default Groq provider."""
    state: ResearchState = {
        "raw_query": "Compare the economic policies of the United States and Japan"
    }
    result = query_understanding(state)

    assert result["intent"] == "comparative"
    assert result["language"] == "en"
    assert isinstance(result["entities"], list)
    assert len(result["entities"]) >= 2
    assert isinstance(result["constraints"], list)


# ==============================================================================
# Difficulty Router Live Tests (NVIDIA NIM via Nemotron 3 Super)
# ==============================================================================

@pytest.mark.skipif(
    not os.getenv("NVIDIA_API_KEY"),
    reason="Requires NVIDIA_API_KEY set in environment or .env file",
)
def test_live_difficulty_shallow():
    """Live API test: Single narrow lookup -> shallow difficulty."""
    state: ResearchState = {
        "raw_query": "What is the capital of France?",
        "intent": "factual",
        "entities": ["capital", "France"],
        "constraints": [],
    }
    result = difficulty_router(state)

    assert result["difficulty"] == "shallow"


@pytest.mark.skipif(
    not os.getenv("NVIDIA_API_KEY"),
    reason="Requires NVIDIA_API_KEY set in environment or .env file",
)
def test_live_difficulty_direct():
    """Live API test: Arithmetic / common knowledge -> direct difficulty."""
    state: ResearchState = {
        "raw_query": "What is 15 times 12?",
        "intent": "factual",
        "entities": ["15", "12"],
        "constraints": [],
    }
    result = difficulty_router(state)

    assert result["difficulty"] == "direct"


@pytest.mark.skipif(
    not os.getenv("NVIDIA_API_KEY"),
    reason="Requires NVIDIA_API_KEY set in environment or .env file",
)
def test_live_difficulty_deep():
    """Live API test: Multi-entity comparison -> deep difficulty."""
    state: ResearchState = {
        "raw_query": "Compare the economic policies of the United States and Japan",
        "intent": "comparative",
        "entities": ["United States", "Japan", "economic policies"],
        "constraints": [],
    }
    result = difficulty_router(state)

    assert result["difficulty"] == "deep"


@pytest.mark.skipif(
    not os.getenv("NVIDIA_API_KEY"),
    reason="Requires NVIDIA_API_KEY set in environment or .env file",
)
def test_live_difficulty_router_via_nvidia_nim():
    """Live API test: Explicit test exercising difficulty_router node via default NVIDIA NIM provider."""
    state: ResearchState = {
        "raw_query": "Compare the economic policies of the United States and Japan",
        "intent": "comparative",
        "entities": ["United States", "Japan", "economic policies"],
        "constraints": [],
    }
    result = difficulty_router(state)

    assert result["difficulty"] in ["shallow", "direct", "deep"]
    assert result["difficulty"] == "deep"


# ==============================================================================
# Planner Live Tests (Gemini via gemini-3.6-flash)
# ==============================================================================

@pytest.mark.skipif(
    not os.getenv("GOOGLE_API_KEY"),
    reason="Requires GOOGLE_API_KEY set in environment or .env file",
)
def test_live_planner_via_gemini():
    """Live API test: Explicit test exercising planner node via default Gemini provider."""
    state: ResearchState = {
        "raw_query": "Compare the economic policies of the United States and Japan",
        "intent": "comparative",
        "entities": ["United States", "Japan", "economic policies"],
        "constraints": [],
        "difficulty": "deep",
    }
    result = planner(state)

    assert result.get("plan")
    assert isinstance(result["plan"], str)
    assert len(result["plan"].strip()) > 0
    assert isinstance(result.get("sub_queries"), list)
    assert 2 <= len(result["sub_queries"]) <= 3
    assert all(isinstance(sq, str) and len(sq.strip()) > 0 for sq in result["sub_queries"])

