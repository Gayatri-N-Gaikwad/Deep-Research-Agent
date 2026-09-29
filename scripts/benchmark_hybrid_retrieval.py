"""Benchmark latency for Stage 5 Hybrid Retrieval node.

Measures:
1. Cold invocation (process-level model initialization)
2. Warm per-query latency across multiple runs (dedup + BM25 + SBERT encode + RRF)
"""

from pathlib import Path
import sys
import time

# Ensure repo root on path
REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from research_agent.nodes.hybrid_retrieval import hybrid_retrieval
from research_agent.state import ResearchState

# Sample realistic state with 7 multi-track evidence items (with 1 duplicate URL to test dedup)
mock_evidence = [
    {
        "sub_query": "US macro policy",
        "source_url": "https://fed.gov/policy",
        "title": "Fed Monetary Framework",
        "content": "The Federal Reserve targets inflation and employment using interest rates and quantitative easing.",
        "track_id": 0,
    },
    {
        "sub_query": "US macro policy",
        "source_url": "https://treasury.gov/debt",
        "title": "US Treasury Debt",
        "content": "US federal debt levels and fiscal deficit projections for 2026.",
        "track_id": 0,
    },
    {
        "sub_query": "Japan economic strategy",
        "source_url": "https://boj.or.jp/policy",
        "title": "Bank of Japan Yield Curve Control",
        "content": "Bank of Japan monetary policy, negative interest rates, and yield curve management.",
        "track_id": 1,
    },
    {
        "sub_query": "Japan economic strategy",
        "source_url": "https://mof.go.jp/debt",
        "title": "Japan Ministry of Finance",
        "content": "Japanese sovereign debt ratio relative to GDP and deflation management.",
        "track_id": 1,
    },
    {
        "sub_query": "Comparative debt dynamics",
        "source_url": "https://imf.org/report",
        "title": "IMF Global Economic Outlook",
        "content": "Comparative assessment of fiscal trajectories in the US and Japan.",
        "track_id": 2,
    },
    {
        "sub_query": "Comparative debt dynamics",
        "source_url": "https://fed.gov/policy",  # Duplicate URL with Track 0
        "title": "Fed Policy Duplicate",
        "content": "Duplicate item from track 2.",
        "track_id": 2,
    },
    {
        "sub_query": "Comparative debt dynamics",
        "source_url": "https://oecd.org/japan-us",
        "title": "OECD Comparative Data",
        "content": "Structural reforms, productivity metrics, and aging demographics in Japan vs US.",
        "track_id": 2,
    },
]

state: ResearchState = {
    "raw_query": "Compare the economic policies of the United States and Japan",
    "sub_queries": ["US macro policy", "Japan economic strategy", "Comparative debt dynamics"],
    "evidence": mock_evidence,
    "difficulty": "deep",
}

print("=" * 70)
print("Benchmarking Stage 5 Hybrid Retrieval Node Latency")
print("=" * 70)

# 1. Cold invocation (triggers lazy module-level singleton initialization)
t_cold_start = time.perf_counter()
res_cold = hybrid_retrieval(state)
t_cold = time.perf_counter() - t_cold_start
print(f"Cold Invocation (first call, initializing model): {t_cold:.2f}s")

# 2. Warm invocations (measuring actual runtime cost of hybrid retrieval)
runs = 5
warm_timings_ms = []
for i in range(runs):
    t_start = time.perf_counter()
    res = hybrid_retrieval(state)
    elapsed_ms = (time.perf_counter() - t_start) * 1000
    warm_timings_ms.append(elapsed_ms)

avg_ms = sum(warm_timings_ms) / len(warm_timings_ms)
min_ms = min(warm_timings_ms)
max_ms = max(warm_timings_ms)

print(f"\nWarm Invocations ({runs} runs):")
print(f"  Average per-query latency: {avg_ms:.1f} ms")
print(f"  Min per-query latency:     {min_ms:.1f} ms")
print(f"  Max per-query latency:     {max_ms:.1f} ms")

print(f"\nDeduplication Result:")
print(f"  Raw evidence items: {len(mock_evidence)}")
print(f"  Deduplicated & ranked items: {len(res['ranked_evidence'])}")

print("\nFinal Ranked Evidence Output:")
for item in res["ranked_evidence"]:
    print(
        f"  Rank {item['rank']}: RRF={item['rrf_score']:.6f} | "
        f"BM25={item['bm25_score']:.3f} (rk {item['bm25_rank']}) | "
        f"Sem={item['semantic_score']:.3f} (rk {item['semantic_rank']}) | "
        f"{item['title']}"
    )
print("=" * 70)
