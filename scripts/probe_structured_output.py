"""Structured-Output Compatibility Probe for Candidate LLM Providers.

Tests each candidate provider (Groq, Mistral, NVIDIA NIM, OpenRouter, and Gemini baseline)
against the 3 core Pydantic schemas in the codebase:
- QueryAnalysis (query_understanding)
- DifficultyClassification (difficulty_router)
- ResearchPlan (planner)

Usage:
    python scripts/probe_structured_output.py
    python scripts/probe_structured_output.py --providers groq mistral
    python scripts/probe_structured_output.py --show-payloads
"""

import argparse
from dataclasses import dataclass
import os
from pathlib import Path
import sys
import time
from typing import Any, Dict, List, Optional, Type

from dotenv import load_dotenv
from pydantic import BaseModel

# Ensure repository root is on sys.path
REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

load_dotenv(REPO_ROOT / ".env")

from research_agent.nodes.difficulty_router import (
    DIFFICULTY_ROUTER_SYSTEM_PROMPT,
    DifficultyClassification,
)
from research_agent.nodes.llm_providers import PROVIDER_REGISTRY, get_llm_for_provider
from research_agent.nodes.planner import PLANNER_SYSTEM_PROMPT, ResearchPlan
from research_agent.nodes.query_understanding import (
    QUERY_UNDERSTANDING_SYSTEM_PROMPT,
    QueryAnalysis,
)

# Canonical benchmark query used across stages
BENCHMARK_QUERY = "Compare the economic policies of the United States and Japan"


@dataclass
class SchemaTestCase:
    name: str
    schema: Type[BaseModel]
    messages: List[Dict[str, str]]
    validator: Any


def build_test_cases() -> Dict[str, SchemaTestCase]:
    """Construct the standardized test input messages and validators for each schema."""
    return {
        "QueryAnalysis": SchemaTestCase(
            name="QueryAnalysis",
            schema=QueryAnalysis,
            messages=[
                {"role": "system", "content": QUERY_UNDERSTANDING_SYSTEM_PROMPT},
                {"role": "user", "content": f"Query: {BENCHMARK_QUERY}"},
            ],
            validator=lambda obj: (
                isinstance(obj, QueryAnalysis)
                and obj.intent in ["factual", "comparative", "exploratory", "procedural"]
                and isinstance(obj.entities, list)
                and len(obj.entities) > 0
            ),
        ),
        "DifficultyClassification": SchemaTestCase(
            name="DifficultyClassification",
            schema=DifficultyClassification,
            messages=[
                {"role": "system", "content": DIFFICULTY_ROUTER_SYSTEM_PROMPT},
                {
                    "role": "user",
                    "content": (
                        f"Analyze query difficulty based on the following query data:\n"
                        f"Raw Query: {BENCHMARK_QUERY}\n"
                        f"Extracted Intent: comparative\n"
                        f"Extracted Entities: ['United States', 'Japan', 'economic policies']\n"
                        f"Extracted Constraints: []"
                    ),
                },
            ],
            validator=lambda obj: (
                isinstance(obj, DifficultyClassification)
                and obj.difficulty in ["shallow", "direct", "deep"]
                and bool(obj.reasoning)
            ),
        ),
        "ResearchPlan": SchemaTestCase(
            name="ResearchPlan",
            schema=ResearchPlan,
            messages=[
                {"role": "system", "content": PLANNER_SYSTEM_PROMPT},
                {
                    "role": "user",
                    "content": (
                        f"Construct a research plan and decomposed sub-queries for the following query data:\n"
                        f"Raw Query: {BENCHMARK_QUERY}\n"
                        f"Extracted Intent: comparative\n"
                        f"Extracted Entities: ['United States', 'Japan', 'economic policies']\n"
                        f"Extracted Constraints: []"
                    ),
                },
            ],
            validator=lambda obj: (
                isinstance(obj, ResearchPlan)
                and bool(obj.plan)
                and isinstance(obj.sub_queries, list)
                and len(obj.sub_queries) in [2, 3]
            ),
        ),
    }


@dataclass
class ProbeResult:
    provider: str
    schema_name: str
    status: str  # "PASS", "FAIL", "ERROR", "SKIPPED"
    latency_sec: float = 0.0
    error_message: Optional[str] = None
    output_payload: Optional[Any] = None


def run_probe_for_target(
    provider_name: str,
    test_case: SchemaTestCase,
) -> ProbeResult:
    """Execute a single live probe call against a (provider, schema) pair."""
    config = PROVIDER_REGISTRY[provider_name]
    api_key = os.getenv(config.env_var)

    if not api_key:
        return ProbeResult(
            provider=provider_name,
            schema_name=test_case.name,
            status="SKIPPED",
            error_message=f"Missing env var: {config.env_var}",
        )

    t0 = time.perf_counter()
    try:
        base_llm = get_llm_for_provider(provider_name=provider_name, temperature=0.0)
        structured_llm = base_llm.with_structured_output(test_case.schema)

        raw_output = structured_llm.invoke(test_case.messages)
        latency = time.perf_counter() - t0

        # Normalization: ensure it is coerced to the Pydantic model
        if isinstance(raw_output, dict):
            validated_obj = test_case.schema.model_validate(raw_output)
        elif isinstance(raw_output, test_case.schema):
            validated_obj = raw_output
        else:
            return ProbeResult(
                provider=provider_name,
                schema_name=test_case.name,
                status="FAIL",
                latency_sec=latency,
                error_message=f"Returned unexpected object type: {type(raw_output)}",
                output_payload=raw_output,
            )

        # Semantic validator check
        if test_case.validator(validated_obj):
            return ProbeResult(
                provider=provider_name,
                schema_name=test_case.name,
                status="PASS",
                latency_sec=latency,
                output_payload=validated_obj.model_dump(),
            )
        else:
            return ProbeResult(
                provider=provider_name,
                schema_name=test_case.name,
                status="FAIL",
                latency_sec=latency,
                error_message="Object failed semantic schema assertions",
                output_payload=validated_obj.model_dump(),
            )

    except Exception as exc:
        latency = time.perf_counter() - t0
        return ProbeResult(
            provider=provider_name,
            schema_name=test_case.name,
            status="ERROR",
            latency_sec=latency,
            error_message=f"{type(exc).__name__}: {str(exc)[:180]}",
        )


def format_cell(res: ProbeResult) -> str:
    """Format single result cell for summary table."""
    if res.status == "PASS":
        return f"PASS ({res.latency_sec:.2f}s)"
    elif res.status == "SKIPPED":
        return "SKIP (no key)"
    elif res.status == "FAIL":
        return f"FAIL ({res.latency_sec:.2f}s)"
    else:  # ERROR
        return f"ERROR ({res.latency_sec:.2f}s)"


def main():
    parser = argparse.ArgumentParser(description="Probe LLM providers for structured output compatibility.")
    parser.add_argument(
        "--providers",
        nargs="+",
        default=["gemini", "groq", "nvidia_nim", "nvidia_nim_ultra", "openrouter"],
        help="List of providers to test (default: all candidate providers + gemini)",
    )
    parser.add_argument(
        "--schemas",
        nargs="+",
        default=None,
        help="List of schemas to test (e.g. DifficultyClassification). Default: all schemas.",
    )
    parser.add_argument(
        "--show-payloads",
        action="store_true",
        help="Print the full validated payload or error details for each successful call.",
    )
    args = parser.parse_args()

    test_cases = build_test_cases()
    if args.schemas:
        test_cases = {k: v for k, v in test_cases.items() if k in args.schemas}
        if not test_cases:
            print(f"No matching schemas found for: {args.schemas}")
            return

    schema_names = list(test_cases.keys())

    print("=" * 80)
    print("STRUCTURED-OUTPUT COMPATIBILITY PROBE")
    print(f"Benchmark Query: \"{BENCHMARK_QUERY}\"")
    print(f"Candidate Providers: {', '.join(args.providers)}")
    print(f"Schemas Probed: {', '.join(schema_names)}")
    print("=" * 80)

    results: Dict[str, Dict[str, ProbeResult]] = {p: {} for p in args.providers}

    for provider in args.providers:
        if provider not in PROVIDER_REGISTRY:
            print(f"\n[!] Skipping unknown provider '{provider}'")
            continue

        config = PROVIDER_REGISTRY[provider]
        key_status = "[KEY FOUND]" if os.getenv(config.env_var) else "[KEY MISSING]"
        print(f"\nProbing Provider: {provider:<12} (model: {config.default_model}) {key_status}")

        for schema_name, test_case in test_cases.items():
            sys.stdout.write(f"  -> Testing {schema_name:<26} ... ")
            sys.stdout.flush()

            res = run_probe_for_target(provider, test_case)
            results[provider][schema_name] = res

            if res.status == "PASS":
                print(f"[OK] PASS in {res.latency_sec:.2f}s")
            elif res.status == "SKIPPED":
                print(f"[SKIP] {res.error_message}")
            elif res.status == "FAIL":
                print(f"[FAIL] {res.error_message}")
            else:
                print(f"[ERROR] {res.error_message}")

    # Render Summary Table
    print("\n" + "=" * 80)
    print("PROBE COMPATIBILITY MATRIX")
    print("=" * 80)

    header = f"{'Provider':<14} | {'Model':<34} | " + " | ".join(f"{s:<15}" for s in schema_names)
    print(header)
    print("-" * len(header))

    for provider, schema_dict in results.items():
        if not schema_dict:
            continue
        model_str = PROVIDER_REGISTRY[provider].default_model
        if len(model_str) > 32:
            model_str = model_str[:29] + "..."

        row = [f"{provider:<14}", f"{model_str:<34}"]
        for s in schema_names:
            cell = format_cell(schema_dict[s]) if s in schema_dict else "N/A"
            row.append(f"{cell:<15}")
        print(" | ".join(row))

    print("-" * len(header))

    # Diagnostics / Errors section
    has_errors = any(
        res.status in ("FAIL", "ERROR")
        for p_res in results.values()
        for res in p_res.values()
    )
    if has_errors:
        print("\n" + "!" * 80)
        print("FAILURE & ERROR DIAGNOSTICS")
        print("!" * 80)
        for provider, p_res in results.items():
            for schema_name, res in p_res.items():
                if res.status in ("FAIL", "ERROR"):
                    print(f"[{provider.upper()}] - {schema_name}:")
                    print(f"  Status:  {res.status}")
                    print(f"  Latency: {res.latency_sec:.2f}s")
                    print(f"  Details: {res.error_message}")
                    if res.output_payload:
                        print(f"  Payload: {res.output_payload}")
                    print()

    # Optional payload inspection
    if args.show_payloads:
        print("\n" + "=" * 80)
        print("OUTPUT PAYLOADS")
        print("=" * 80)
        for provider, p_res in results.items():
            for schema_name, res in p_res.items():
                if res.status == "PASS":
                    print(f"[{provider.upper()}] - {schema_name} (in {res.latency_sec:.2f}s):")
                    print(f"  {res.output_payload}\n")


if __name__ == "__main__":
    main()
