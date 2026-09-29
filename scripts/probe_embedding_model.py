"""Indic-Capable Sentence Embedding Model Compatibility Probe.

Evaluates candidate HuggingFace repositories for native sentence-transformers
compatibility, measured embedding dimensions, model footprint on disk,
cold-load time, and English semantic encoding sanity.

Usage:
    python scripts/probe_embedding_model.py
    python scripts/probe_embedding_model.py --candidates l3cube-pune/indic-sentence-bert-nli ai4bharat/indic-bert
"""

import argparse
from dataclasses import dataclass
import json
import os
from pathlib import Path
import sys
import time
from typing import Any, Dict, List, Optional
import urllib.error
import urllib.request

# Ensure repository root is on sys.path
REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

# Canonical benchmark query used across stages
BENCHMARK_QUERY = "Compare the economic policies of the United States and Japan"
ENGLISH_DOC_MATCH = "Macroeconomic policy frameworks and fiscal deficits of the United States and Japan."
ENGLISH_DOC_UNRELATED = "The biological life cycle of the Monarch butterfly in North America."

# Default candidates covering Indic and multilingual SBERT models
DEFAULT_CANDIDATES = [
    "l3cube-pune/indic-sentence-bert-nli",
    "ai4bharat/indic-sentence-bert",
    "ai4bharat/indic-bert",
    "sentence-transformers/paraphrase-multilingual-mpnet-base-v2",
]


@dataclass
class HubMetadata:
    repo_id: str
    exists: bool
    status_code: int
    has_sentence_transformers_tag: bool
    has_modules_json: bool
    weights_files: List[str]
    error_msg: Optional[str] = None


@dataclass
class ProbeResult:
    repo_id: str
    hub_accessible: bool
    direct_load_success: bool
    load_time_sec: float
    measured_dimension: Optional[int]
    disk_size_mb: Optional[float]
    test_encoding_shape: Optional[tuple]
    cosine_sim_relevant: Optional[float]
    cosine_sim_unrelated: Optional[float]
    error_detail: Optional[str] = None


def check_hf_hub_repo(repo_id: str) -> HubMetadata:
    """Query HuggingFace Model Hub API for repository existence and files."""
    api_url = f"https://huggingface.co/api/models/{repo_id}"
    req = urllib.request.Request(api_url, headers={"User-Agent": "DeepResearchAgent-Probe/1.0"})

    try:
        with urllib.request.urlopen(req, timeout=10) as resp:
            status_code = resp.status
            data = json.loads(resp.read().decode("utf-8"))
            tags = data.get("tags", [])
            siblings = [s.get("rfilename", "") for s in data.get("siblings", [])]

            has_sbert_tag = "sentence-transformers" in tags
            has_modules = "modules.json" in siblings
            weights = [s for s in siblings if s.endswith((".bin", ".safetensors"))]

            return HubMetadata(
                repo_id=repo_id,
                exists=True,
                status_code=status_code,
                has_sentence_transformers_tag=has_sbert_tag,
                has_modules_json=has_modules,
                weights_files=weights,
            )
    except urllib.error.HTTPError as exc:
        return HubMetadata(
            repo_id=repo_id,
            exists=False,
            status_code=exc.code,
            has_sentence_transformers_tag=False,
            has_modules_json=False,
            weights_files=[],
            error_msg=f"HTTP {exc.code}: {exc.reason}",
        )
    except Exception as exc:
        return HubMetadata(
            repo_id=repo_id,
            exists=False,
            status_code=0,
            has_sentence_transformers_tag=False,
            has_modules_json=False,
            weights_files=[],
            error_msg=str(exc),
        )


def compute_cosine_similarity(vec1, vec2) -> float:
    """Compute cosine similarity between two 1D vectors."""
    import numpy as np

    v1 = np.asarray(vec1, dtype=float)
    v2 = np.asarray(vec2, dtype=float)
    dot = np.dot(v1, v2)
    norm1 = np.linalg.norm(v1)
    norm2 = np.linalg.norm(v2)
    if norm1 == 0 or norm2 == 0:
        return 0.0
    return float(dot / (norm1 * norm2))


def measure_cache_size(repo_id: str) -> Optional[float]:
    """Measure the size on disk (MB) of the downloaded HF repo snapshot."""
    # Standard HF cache path
    cache_dir = Path.home() / ".cache" / "huggingface" / "hub"
    repo_folder_name = "models--" + repo_id.replace("/", "--")
    target_dir = cache_dir / repo_folder_name

    if not target_dir.exists():
        return None

    total_bytes = sum(f.stat().st_size for f in target_dir.rglob("*") if f.is_file())
    return round(total_bytes / (1024 * 1024), 2)


def probe_candidate(repo_id: str, load_model: bool = True) -> ProbeResult:
    """Run full probe on a candidate model repo."""
    print(f"\n[{repo_id}]")
    print(f"  Step 1: Checking HuggingFace Hub metadata...")
    hub_meta = check_hf_hub_repo(repo_id)

    if not hub_meta.exists:
        print(f"  [FAIL] Not found or inaccessible on HF Hub: {hub_meta.error_msg}")
        return ProbeResult(
            repo_id=repo_id,
            hub_accessible=False,
            direct_load_success=False,
            load_time_sec=0.0,
            measured_dimension=None,
            disk_size_mb=None,
            test_encoding_shape=None,
            cosine_sim_relevant=None,
            cosine_sim_unrelated=None,
            error_detail=hub_meta.error_msg,
        )

    print(f"  [INFO] Hub Status: HTTP {hub_meta.status_code}")
    print(f"  [INFO] Has sentence-transformers tag: {hub_meta.has_sentence_transformers_tag}")
    print(f"  [INFO] Has modules.json (direct SBERT compatible): {hub_meta.has_modules_json}")
    print(f"  [INFO] Weights files: {', '.join(hub_meta.weights_files) if hub_meta.weights_files else 'None'}")

    if not load_model:
        return ProbeResult(
            repo_id=repo_id,
            hub_accessible=True,
            direct_load_success=False,
            load_time_sec=0.0,
            measured_dimension=None,
            disk_size_mb=None,
            test_encoding_shape=None,
            cosine_sim_relevant=None,
            cosine_sim_unrelated=None,
            error_detail="Model loading skipped (--no-load)",
        )

    print(f"  Step 2: Attempting direct load via sentence_transformers.SentenceTransformer('{repo_id}')...")
    start_load = time.perf_counter()
    try:
        from sentence_transformers import SentenceTransformer

        model = SentenceTransformer(repo_id)
        load_time = round(time.perf_counter() - start_load, 2)
        print(f"  [PASS] Model loaded successfully in {load_time:.2f}s")
    except Exception as exc:
        load_time = round(time.perf_counter() - start_load, 2)
        print(f"  [FAIL] Direct load failed after {load_time:.2f}s: {exc}")
        return ProbeResult(
            repo_id=repo_id,
            hub_accessible=True,
            direct_load_success=False,
            load_time_sec=load_time,
            measured_dimension=None,
            disk_size_mb=None,
            test_encoding_shape=None,
            cosine_sim_relevant=None,
            cosine_sim_unrelated=None,
            error_detail=f"SentenceTransformer load failed: {exc}",
        )

    # Step 3: Measured dimension and disk footprint
    print(f"  Step 3: Measuring dimensions and disk footprint...")
    measured_dim = model.get_sentence_embedding_dimension()
    disk_size_mb = measure_cache_size(repo_id)
    print(f"  [PASS] Measured Embedding Dimension: {measured_dim}")
    print(f"  [PASS] Cache Size on Disk: {disk_size_mb} MB" if disk_size_mb else "  [INFO] Cache Size on Disk: N/A")

    # Step 4: Encode English benchmark query & calculate similarity
    print(f"  Step 4: Testing English text encoding and semantic similarity separation...")
    test_texts = [BENCHMARK_QUERY, ENGLISH_DOC_MATCH, ENGLISH_DOC_UNRELATED]
    t0 = time.perf_counter()
    embeddings = model.encode(test_texts, show_progress_bar=False)
    encode_time = round((time.perf_counter() - t0) * 1000, 1)

    query_emb = embeddings[0]
    match_emb = embeddings[1]
    unrelated_emb = embeddings[2]

    sim_relevant = round(compute_cosine_similarity(query_emb, match_emb), 4)
    sim_unrelated = round(compute_cosine_similarity(query_emb, unrelated_emb), 4)

    print(f"  [PASS] Encoded {len(test_texts)} sentences in {encode_time}ms")
    print(f"  [PASS] Query Vector Shape: {query_emb.shape}, Norm: {round(float(query_emb.dot(query_emb)**0.5), 4)}")
    print(f"  [INFO] Semantic Sim (Query vs Relevant Doc):   {sim_relevant}")
    print(f"  [INFO] Semantic Sim (Query vs Unrelated Doc):  {sim_unrelated}")
    print(f"  [INFO] Separation Margin:                     {round(sim_relevant - sim_unrelated, 4)}")

    return ProbeResult(
        repo_id=repo_id,
        hub_accessible=True,
        direct_load_success=True,
        load_time_sec=load_time,
        measured_dimension=measured_dim,
        disk_size_mb=disk_size_mb,
        test_encoding_shape=tuple(query_emb.shape),
        cosine_sim_relevant=sim_relevant,
        cosine_sim_unrelated=sim_unrelated,
    )


def print_summary_table(results: List[ProbeResult]):
    """Print markdown-formatted summary of probe results."""
    print("\n" + "=" * 90)
    print("PROBE RESULTS SUMMARY TABLE")
    print("=" * 90)
    headers = [
        "Candidate Repo ID",
        "Hub Accessible",
        "Direct SBERT Load",
        "Dimension",
        "Disk Size",
        "Load Time",
        "Sim (Rel / Unrel)",
    ]
    print(f"| {' | '.join(headers)} |")
    print(f"|{'|'.join(['---' for _ in headers])}|")

    for r in results:
        hub_str = "YES" if r.hub_accessible else "NO"
        load_str = "SUCCESS" if r.direct_load_success else "FAILED"
        dim_str = str(r.measured_dimension) if r.measured_dimension else "-"
        size_str = f"{r.disk_size_mb} MB" if r.disk_size_mb else "-"
        time_str = f"{r.load_time_sec:.2f}s" if r.load_time_sec else "-"
        sim_str = f"{r.cosine_sim_relevant} / {r.cosine_sim_unrelated}" if r.cosine_sim_relevant is not None else "-"

        print(
            f"| `{r.repo_id}` | {hub_str} | {load_str} | {dim_str} | {size_str} | {time_str} | {sim_str} |"
        )
    print("=" * 90 + "\n")


def main():
    parser = argparse.ArgumentParser(description="Probe Indic and Multilingual Sentence-Transformers Models.")
    parser.add_argument(
        "--candidates",
        nargs="+",
        default=DEFAULT_CANDIDATES,
        help="List of HuggingFace repository IDs to probe.",
    )
    parser.add_argument(
        "--no-load",
        action="store_true",
        help="Only check HuggingFace Hub metadata without downloading weights.",
    )
    args = parser.parse_args()

    print("=" * 90)
    print("Stage 5 Embedding Model Compatibility Probe")
    print(f"Benchmark Query: '{BENCHMARK_QUERY}'")
    print(f"Candidates to test: {args.candidates}")
    print("=" * 90)

    results = []
    for cand in args.candidates:
        res = probe_candidate(cand, load_model=not args.no_load)
        results.append(res)

    print_summary_table(results)


if __name__ == "__main__":
    main()
