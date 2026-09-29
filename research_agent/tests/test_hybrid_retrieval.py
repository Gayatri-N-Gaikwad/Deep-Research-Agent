"""Unit tests for Stage 5 Hybrid Retrieval node using mocked rankers."""

import pytest
import numpy as np

from research_agent.nodes.hybrid_retrieval import (
    DEFAULT_EMBEDDING_MODEL,
    RRF_K,
    compute_cosine_similarity,
    deduplicate_evidence_by_url,
    hybrid_retrieval,
    tokenize_text,
)
from research_agent.state import EvidenceItem, ResearchState


class MockEmbeddingModel:
    """Mock SentenceTransformer model returning deterministic vectors based on text content."""

    def __init__(self, vector_map=None, default_dim=768):
        self.vector_map = vector_map or {}
        self.default_dim = default_dim
        self.encode_call_count = 0

    def encode(self, texts, show_progress_bar=False):
        self.encode_call_count += 1
        if isinstance(texts, str):
            return self._get_vector(texts)
        return [self._get_vector(t) for t in texts]

    def _get_vector(self, text: str) -> np.ndarray:
        if text in self.vector_map:
            return np.array(self.vector_map[text], dtype=np.float32)

        # Deterministic pseudo-embedding based on hash of text
        np.random.seed(abs(hash(text)) % (2**32))
        vec = np.random.randn(self.default_dim).astype(np.float32)
        norm = np.linalg.norm(vec)
        return vec / (norm + 1e-9)


def test_deduplicate_evidence_by_url():
    """Verify exact URL deduplication retains the first occurrence and preserves track info."""
    raw_evidence: list[EvidenceItem] = [
        {
            "sub_query": "query 1",
            "source_url": "https://example.com/page-a",
            "title": "Page A (Track 0)",
            "content": "Content A",
            "track_id": 0,
        },
        {
            "sub_query": "query 2",
            "source_url": "https://example.com/page-b",
            "title": "Page B",
            "content": "Content B",
            "track_id": 1,
        },
        {
            "sub_query": "query 3",
            "source_url": "https://example.com/page-a",  # Duplicate URL
            "title": "Page A Duplicate (Track 2)",
            "content": "Content A duplicate snippet",
            "track_id": 2,
        },
        {
            "sub_query": "query 1",
            "source_url": "",  # Missing URL
            "title": "No URL Doc 1",
            "content": "No URL Content 1",
            "track_id": 0,
        },
    ]

    deduped = deduplicate_evidence_by_url(raw_evidence)

    assert len(deduped) == 3
    urls = [d["source_url"] for d in deduped]
    assert urls == ["https://example.com/page-a", "https://example.com/page-b", ""]
    # Confirm first occurrence was kept
    assert deduped[0]["title"] == "Page A (Track 0)"
    assert deduped[0]["track_id"] == 0


def test_tokenize_text():
    """Verify tokenizer creates clean lowercase alphanumeric tokens."""
    tokens = tokenize_text("Macroeconomic Policies: U.S. & Japan (2026)!")
    assert "macroeconomic" in tokens
    assert "policies" in tokens
    assert "u" in tokens
    assert "s" in tokens
    assert "japan" in tokens
    assert "2026" in tokens


def test_compute_cosine_similarity():
    """Verify cosine similarity calculation between identical, orthogonal, and opposing vectors."""
    v1 = [1.0, 0.0, 0.0]
    v2 = [1.0, 0.0, 0.0]
    v3 = [0.0, 1.0, 0.0]
    v4 = [-1.0, 0.0, 0.0]

    assert pytest.approx(compute_cosine_similarity(v1, v2), 0.001) == 1.0
    assert pytest.approx(compute_cosine_similarity(v1, v3), 0.001) == 0.0
    assert pytest.approx(compute_cosine_similarity(v1, v4), 0.001) == -1.0


def test_hybrid_retrieval_rrf_scoring_math():
    """Verify Reciprocal Rank Fusion formula score = 1/(60+r_bm25) + 1/(60+r_sem)."""
    query = "quantum computing cryptography"

    # Pre-configure embeddings to produce explicit semantic ranks
    # Doc 0: high semantic, high bm25
    # Doc 1: low semantic, high bm25
    # Doc 2: high semantic, low bm25
    query_vec = [1.0, 0.0, 0.0]
    vec_doc0 = [0.99, 0.1, 0.0]   # high similarity to query
    vec_doc1 = [0.1, 0.9, 0.0]    # low similarity
    vec_doc2 = [0.95, 0.1, 0.0]   # high similarity

    doc0_text = "Quantum computing and post quantum cryptography breakthroughs."
    doc1_text = "Quantum computing hardware standards and qubit counts."
    doc2_text = "Modern cryptography and information security fundamentals."

    mock_model = MockEmbeddingModel(
        vector_map={
            query: query_vec,
            f"Doc 0 {doc0_text}": vec_doc0,
            f"Doc 1 {doc1_text}": vec_doc1,
            f"Doc 2 {doc2_text}": vec_doc2,
        },
        default_dim=3,
    )

    state: ResearchState = {
        "raw_query": query,
        "difficulty": "deep",
        "evidence": [
            {
                "sub_query": "quantum cryptography",
                "source_url": "https://example.com/doc0",
                "title": "Doc 0",
                "content": doc0_text,
                "track_id": 0,
            },
            {
                "sub_query": "quantum hardware",
                "source_url": "https://example.com/doc1",
                "title": "Doc 1",
                "content": doc1_text,
                "track_id": 1,
            },
            {
                "sub_query": "cryptography overview",
                "source_url": "https://example.com/doc2",
                "title": "Doc 2",
                "content": doc2_text,
                "track_id": 2,
            },
        ],
    }

    result = hybrid_retrieval(state, embedding_model=mock_model)

    ranked = result["ranked_evidence"]
    assert len(ranked) == 3

    # Check that rank fields exist and conform to schema
    for item in ranked:
        assert "bm25_score" in item
        assert "bm25_rank" in item
        assert "semantic_score" in item
        assert "semantic_rank" in item
        assert "rrf_score" in item
        assert "rank" in item

        # Verify mathematical RRF definition: 1 / (60 + r_bm25) + 1 / (60 + r_sem)
        expected_rrf = (1.0 / (RRF_K + item["bm25_rank"])) + (1.0 / (RRF_K + item["semantic_rank"]))
        assert pytest.approx(item["rrf_score"], abs=1e-6) == expected_rrf

    # Verify descending ordering by final rank
    assert ranked[0]["rank"] == 1
    assert ranked[1]["rank"] == 2
    assert ranked[2]["rank"] == 3
    assert ranked[0]["rrf_score"] >= ranked[1]["rrf_score"] >= ranked[2]["rrf_score"]


def test_hybrid_retrieval_empty_evidence():
    """Verify empty evidence list from Stage 4 returns empty ranked_evidence without error."""
    state: ResearchState = {
        "raw_query": "Test query with no evidence",
        "difficulty": "deep",
        "evidence": [],
    }

    result = hybrid_retrieval(state, embedding_model=MockEmbeddingModel())

    assert "ranked_evidence" in result
    assert result["ranked_evidence"] == []
    assert result["raw_query"] == "Test query with no evidence"


def test_hybrid_retrieval_single_item_evidence():
    """Verify single-item evidence list degrades gracefully without division-by-zero."""
    single_doc: EvidenceItem = {
        "sub_query": "only query",
        "source_url": "https://example.com/only",
        "title": "Only Title",
        "content": "Only content snippet available.",
        "track_id": 0,
    }
    state: ResearchState = {
        "raw_query": "Only query",
        "difficulty": "deep",
        "evidence": [single_doc],
    }

    result = hybrid_retrieval(state, embedding_model=MockEmbeddingModel())

    ranked = result.get("ranked_evidence", [])
    assert len(ranked) == 1
    assert ranked[0]["rank"] == 1
    assert ranked[0]["bm25_rank"] == 1
    assert ranked[0]["semantic_rank"] == 1
    assert pytest.approx(ranked[0]["rrf_score"], abs=1e-5) == 2.0 / (RRF_K + 1)
    assert ranked[0]["source_url"] == "https://example.com/only"


def test_hybrid_retrieval_all_bm25_zero_matches():
    """Verify semantic ranker differentiates documents when all BM25 scores are 0 (no keyword overlap)."""
    query = "abcdefg nonmatching query"
    query_vec = [1.0, 0.0]
    vec_close = [0.9, 0.1]
    vec_far = [0.1, 0.9]

    mock_model = MockEmbeddingModel(
        vector_map={
            query: query_vec,
            "Doc 1 Solar energy": vec_close,
            "Doc 2 Quantum mechanics": vec_far,
        },
        default_dim=2,
    )

    state: ResearchState = {
        "raw_query": query,
        "difficulty": "deep",
        "evidence": [
            {
                "sub_query": "solar",
                "source_url": "https://example.com/solar",
                "title": "Doc 1",
                "content": "Solar energy",
                "track_id": 0,
            },
            {
                "sub_query": "quantum",
                "source_url": "https://example.com/quantum",
                "title": "Doc 2",
                "content": "Quantum mechanics",
                "track_id": 1,
            },
        ],
    }

    result = hybrid_retrieval(state, embedding_model=mock_model)

    ranked = result["ranked_evidence"]
    assert len(ranked) == 2
    # All BM25 scores should be 0.0
    assert ranked[0]["bm25_score"] == 0.0
    assert ranked[1]["bm25_score"] == 0.0

    # Closer semantic document should be rank 1
    assert ranked[0]["source_url"] == "https://example.com/solar"
    assert ranked[0]["rank"] == 1
    assert ranked[1]["rank"] == 2


def test_hybrid_retrieval_preserves_state_immutability():
    """Verify node does not mutate existing fields and adds ranked_evidence additively."""
    state: ResearchState = {
        "raw_query": "Macroeconomics in US and Japan",
        "language": "en",
        "intent": "comparative",
        "entities": ["US", "Japan"],
        "constraints": ["2026"],
        "difficulty": "deep",
        "plan": "Analyze fiscal policies.",
        "sub_queries": ["US policy", "Japan policy"],
        "evidence": [
            {
                "sub_query": "US policy",
                "source_url": "https://example.com/us",
                "title": "US Economics",
                "content": "Federal reserve interest rates.",
                "track_id": 0,
            }
        ],
        "iteration_count": 1,
        "answer": None,
    }

    result = hybrid_retrieval(state, embedding_model=MockEmbeddingModel())

    # Assert all original keys and values are preserved
    assert result["raw_query"] == "Macroeconomics in US and Japan"
    assert result["language"] == "en"
    assert result["intent"] == "comparative"
    assert result["entities"] == ["US", "Japan"]
    assert result["constraints"] == ["2026"]
    assert result["difficulty"] == "deep"
    assert result["plan"] == "Analyze fiscal policies."
    assert result["sub_queries"] == ["US policy", "Japan policy"]
    assert result["iteration_count"] == 1
    assert result["answer"] is None

    # Assert new additive field is present
    assert "ranked_evidence" in result
    assert len(result["ranked_evidence"]) == 1


def test_get_embedding_model_singleton_caching():
    """Verify get_embedding_model caches and reuses the exact same instance in memory."""
    import sys
    from unittest.mock import MagicMock, patch
    from research_agent.nodes.hybrid_retrieval import get_embedding_model

    hr_mod = sys.modules["research_agent.nodes.hybrid_retrieval"]

    # Reset cache state for clean isolation
    original_model = hr_mod._CACHED_EMBEDDING_MODEL
    original_name = hr_mod._CACHED_MODEL_NAME
    hr_mod._CACHED_EMBEDDING_MODEL = None
    hr_mod._CACHED_MODEL_NAME = None

    try:
        mock_instance_a = MagicMock(name="InstanceA")
        mock_class = MagicMock(return_value=mock_instance_a)

        with patch("sentence_transformers.SentenceTransformer", mock_class):
            model1 = get_embedding_model("test-model")
            model2 = get_embedding_model("test-model")

            # Assert exact object identity (same instance in memory)
            assert model1 is model2
            assert model1 is mock_instance_a
            # Assert constructor was called only once
            assert mock_class.call_count == 1

            # Assert switching model name instantiates the new model
            mock_instance_b = MagicMock(name="InstanceB")
            mock_class.return_value = mock_instance_b
            model3 = get_embedding_model("different-model")

            assert model3 is not model1
            assert model3 is mock_instance_b
            assert mock_class.call_count == 2
    finally:
        # Restore cache state
        hr_mod._CACHED_EMBEDDING_MODEL = original_model
        hr_mod._CACHED_MODEL_NAME = original_name
