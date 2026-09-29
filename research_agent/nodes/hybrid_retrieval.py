"""Stage 5 Hybrid Retrieval node for LangGraph deep research pipeline.

Fuses lexical (BM25Okapi) and semantic (IndicSBERT cosine similarity) signals
over deduplicated multi-track evidence using Reciprocal Rank Fusion (RRF, k=60).
"""

import logging
import os
import re
from typing import Any, List, Optional, Sequence

import numpy as np

from research_agent.state import EvidenceItem, RankedEvidenceItem, ResearchState

logger = logging.getLogger(__name__)

# Verified default embedding model on HuggingFace Hub
DEFAULT_EMBEDDING_MODEL: str = os.getenv(
    "EMBEDDING_MODEL_NAME",
    "l3cube-pune/indic-sentence-bert-nli",
)

# Standard Reciprocal Rank Fusion constant
RRF_K: int = 60

# Module-level cache for lazy singleton model loading
_CACHED_EMBEDDING_MODEL = None
_CACHED_MODEL_NAME: Optional[str] = None


class HybridRetrievalError(RuntimeError):
    """Raised when hybrid retrieval ranking encounters an unrecoverable failure."""
    pass


def get_embedding_model(model_name: Optional[str] = None):
    """Retrieve or lazily initialize the SentenceTransformer model instance as a process-level singleton.

    Caches the instantiated model in memory so it is only loaded once per Python process,
    preventing multi-second disk/HF-hub reloads on subsequent pipeline invocations.

    Args:
        model_name: Optional custom model name or path. Defaults to DEFAULT_EMBEDDING_MODEL.

    Returns:
        SentenceTransformer instance.
    """
    global _CACHED_EMBEDDING_MODEL, _CACHED_MODEL_NAME
    target_model = model_name or DEFAULT_EMBEDDING_MODEL

    if _CACHED_EMBEDDING_MODEL is None or _CACHED_MODEL_NAME != target_model:
        from sentence_transformers import SentenceTransformer

        logger.info("Initializing SentenceTransformer with model: '%s'", target_model)
        _CACHED_EMBEDDING_MODEL = SentenceTransformer(target_model)
        _CACHED_MODEL_NAME = target_model

    return _CACHED_EMBEDDING_MODEL


def tokenize_text(text: str) -> list[str]:
    """Tokenize text into lowercase alphanumeric tokens for BM25.

    Args:
        text: Input string.

    Returns:
        List of lowercase tokens.
    """
    if not text:
        return []
    return re.findall(r"\w+", text.lower())


def compute_cosine_similarity(vec1: Sequence[float], vec2: Sequence[float]) -> float:
    """Compute cosine similarity between two 1D numeric vectors.

    Args:
        vec1: First 1D vector (query).
        vec2: Second 1D vector (document).

    Returns:
        Cosine similarity float in [-1.0, 1.0].
    """
    v1 = np.asarray(vec1, dtype=np.float32)
    v2 = np.asarray(vec2, dtype=np.float32)
    norm1 = float(np.linalg.norm(v1))
    norm2 = float(np.linalg.norm(v2))
    if norm1 == 0.0 or norm2 == 0.0:
        return 0.0
    return float(np.dot(v1, v2) / (norm1 * norm2 + 1e-9))


def deduplicate_evidence_by_url(evidence: list[EvidenceItem]) -> list[EvidenceItem]:
    """Deduplicate evidence items by exact source_url match, preserving first occurrence.

    Items with empty or missing URLs are retained without deduplication.

    Args:
        evidence: Raw list of EvidenceItem dictionaries.

    Returns:
        Deduplicated list of EvidenceItem dictionaries.
    """
    seen_urls = set()
    deduped: list[EvidenceItem] = []

    for item in evidence:
        url = item.get("source_url", "").strip()
        if url:
            if url in seen_urls:
                continue
            seen_urls.add(url)
            deduped.append(item)
        else:
            deduped.append(item)

    return deduped


def hybrid_retrieval(
    state: ResearchState,
    embedding_model=None,
    bm25_cls=None,
    k: int = RRF_K,
) -> ResearchState:
    """Consolidate, deduplicate, and rank evidence across tracks via BM25 and IndicSBERT RRF.

    Executes immediately after search_tracks on the 'deep' branch:
    1. Deduplicates evidence items by exact source_url match.
    2. Computes BM25Okapi lexical scores and ranks.
    3. Computes IndicSBERT semantic cosine similarity scores and ranks against raw_query.
    4. Combines rankings with Reciprocal Rank Fusion (RRF, k=60).
    5. Populates state['ranked_evidence'] with RankedEvidenceItem objects.

    Args:
        state: Current ResearchState containing 'raw_query' and 'evidence'.
        embedding_model: Optional injected SentenceTransformer or mock embedding model.
        bm25_cls: Optional injected BM25Okapi class or mock.
        k: RRF smoothing constant (default: 60).

    Returns:
        Updated ResearchState with 'ranked_evidence' populated.
    """
    raw_query = state.get("raw_query", "") or ""
    raw_evidence = state.get("evidence", []) or []

    # Edge Case 1: Empty evidence list from Stage 4 (degrades gracefully)
    if not raw_evidence:
        logger.info("Hybrid retrieval received empty evidence list; returning empty ranked_evidence.")
        return {
            **state,
            "ranked_evidence": [],
        }

    # Step 1: URL Deduplication
    deduped_evidence = deduplicate_evidence_by_url(raw_evidence)
    num_docs = len(deduped_evidence)

    # Edge Case 2: Very small evidence set (single document)
    if num_docs == 1:
        single_doc = deduped_evidence[0]
        single_rrf_score = float(2.0 / (k + 1))
        ranked_single: RankedEvidenceItem = {
            **single_doc,
            "bm25_score": 1.0,
            "bm25_rank": 1,
            "semantic_score": 1.0,
            "semantic_rank": 1,
            "rrf_score": single_rrf_score,
            "rank": 1,
        }
        return {
            **state,
            "ranked_evidence": [ranked_single],
        }

    # Prepare document text representations (Title + Content)
    doc_texts = [
        f"{doc.get('title', '')} {doc.get('content', '')}".strip()
        for doc in deduped_evidence
    ]

    # Step 2: Lexical Ranking (BM25Okapi)
    if bm25_cls is None:
        from rank_bm25 import BM25Okapi
        bm25_factory = BM25Okapi
    else:
        bm25_factory = bm25_cls

    tokenized_corpus = [tokenize_text(doc_text) for doc_text in doc_texts]
    query_tokens = tokenize_text(raw_query)

    try:
        bm25 = bm25_factory(tokenized_corpus)
        bm25_scores = [float(s) for s in bm25.get_scores(query_tokens)]
    except Exception as exc:
        logger.warning("BM25 scoring encountered error (%s); falling back to zero scores.", exc)
        bm25_scores = [0.0] * num_docs

    # Sort docs by BM25 score descending (stable sort preserving index)
    bm25_sorted_indices = sorted(
        range(num_docs),
        key=lambda i: bm25_scores[i],
        reverse=True,
    )
    # Map index -> 1-indexed bm25 rank
    bm25_ranks = {
        doc_idx: rank for rank, doc_idx in enumerate(bm25_sorted_indices, start=1)
    }

    # Step 3: Semantic Ranking (Cosine Similarity with SentenceTransformer)
    model = embedding_model if embedding_model is not None else get_embedding_model()

    try:
        if hasattr(model, "encode"):
            # SentenceTransformer or standard embedding mock
            query_emb = model.encode(raw_query, show_progress_bar=False)
            doc_embs = model.encode(doc_texts, show_progress_bar=False)
        else:
            raise AttributeError("Embedding model does not provide an 'encode' method.")

        semantic_scores = [
            compute_cosine_similarity(query_emb, doc_emb)
            for doc_emb in doc_embs
        ]
    except Exception as exc:
        logger.warning("Semantic embedding calculation failed (%s); falling back to zero scores.", exc)
        semantic_scores = [0.0] * num_docs

    # Sort docs by semantic score descending
    semantic_sorted_indices = sorted(
        range(num_docs),
        key=lambda i: semantic_scores[i],
        reverse=True,
    )
    # Map index -> 1-indexed semantic rank
    semantic_ranks = {
        doc_idx: rank for rank, doc_idx in enumerate(semantic_sorted_indices, start=1)
    }

    # Step 4: Reciprocal Rank Fusion (RRF)
    # score(doc) = 1 / (k + rank_bm25) + 1 / (k + rank_semantic)
    rrf_scores: dict[int, float] = {}
    for doc_idx in range(num_docs):
        r_bm25 = bm25_ranks[doc_idx]
        r_sem = semantic_ranks[doc_idx]
        rrf_scores[doc_idx] = (1.0 / (k + r_bm25)) + (1.0 / (k + r_sem))

    # Sort final documents by RRF score descending (tie-breaker: semantic, then bm25)
    final_sorted_indices = sorted(
        range(num_docs),
        key=lambda i: (rrf_scores[i], semantic_scores[i], bm25_scores[i]),
        reverse=True,
    )

    ranked_evidence: list[RankedEvidenceItem] = []
    for final_rank, doc_idx in enumerate(final_sorted_indices, start=1):
        doc = deduped_evidence[doc_idx]
        ranked_item: RankedEvidenceItem = {
            **doc,
            "bm25_score": round(bm25_scores[doc_idx], 6),
            "bm25_rank": bm25_ranks[doc_idx],
            "semantic_score": round(semantic_scores[doc_idx], 6),
            "semantic_rank": semantic_ranks[doc_idx],
            "rrf_score": round(rrf_scores[doc_idx], 8),
            "rank": final_rank,
        }
        ranked_evidence.append(ranked_item)

    return {
        **state,
        "ranked_evidence": ranked_evidence,
    }
