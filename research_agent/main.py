"""Command Line Interface for running the LangGraph Research Pipeline."""

import os
import sys

from dotenv import load_dotenv

from research_agent.graph import build_graph
from research_agent.state import create_initial_state


def pretty_print_state(result: dict) -> None:
    """Pretty-print the extracted pipeline state to the terminal."""
    difficulty = result.get("difficulty", "N/A")
    if difficulty == "deep":
        branch = "search_tracks"
    elif difficulty == "shallow":
        branch = "shallow_search"
    elif difficulty == "direct":
        branch = "direct_answer"
    else:
        branch = "N/A"

    print("\n" + "=" * 60)
    print(" RESEARCH PIPELINE - STAGE 5 (SHALLOW & DIRECT PATHS)")
    print("=" * 60)
    print(f" Raw Query:      {result.get('raw_query', '')}")
    print(f" Language:       {result.get('language', 'en')}")
    print(f" Intent:         {result.get('intent', 'N/A')}")
    print(f" Difficulty:     {difficulty}")
    print(f" Branch Reached: {branch}")
    print("-" * 60)
    print(" Providers Used:")
    print("   - Query Understanding: groq (openai/gpt-oss-120b)")
    print("   - Difficulty Router:   nvidia_nim (nvidia/nemotron-3-super-120b-a12b)")
    if result.get("plan"):
        print("   - Planner:             gemini (gemini-3.6-flash)")
    if result.get("answer"):
        print("   - Direct Answer:       groq (openai/gpt-oss-120b)")
    print("-" * 60)

    entities = result.get("entities", [])
    print(f" Entities ({len(entities)}):")
    if entities:
        for ent in entities:
            print(f"   - {ent}")
    else:
        print("   (None identified)")

    constraints = result.get("constraints", [])
    print(f" Constraints ({len(constraints)}):")
    if constraints:
        for c in constraints:
            print(f"   - {c}")
    else:
        print("   (None)")

    plan = result.get("plan")
    if plan:
        print("-" * 60)
        print(f" Research Plan:\n   {plan}")

    answer = result.get("answer")
    if answer:
        print("-" * 60)
        print(f" Direct Answer:\n   {answer}")

    sub_queries = result.get("sub_queries", [])
    if sub_queries:
        print(f" Sub-queries ({len(sub_queries)}):")
        for sq in sub_queries:
            print(f"   - {sq}")

    evidence = result.get("evidence", [])
    if evidence:
        print("-" * 60)
        print(f" Evidence Collected ({len(evidence)} total items):")
        from collections import Counter
        track_counts = Counter(item["track_id"] for item in evidence)
        for track_id in sorted(track_counts.keys()):
            print(f"   - Track {track_id}: {track_counts[track_id]} items")

    print("=" * 60 + "\n")


def main() -> None:
    """Main CLI entry point."""
    load_dotenv()

    required_keys = ["GROQ_API_KEY", "NVIDIA_API_KEY", "GOOGLE_API_KEY", "TAVILY_API_KEY"]
    missing = [k for k in required_keys if not os.getenv(k)]
    if missing:
        print(
            f"ERROR: Missing environment variable(s): {', '.join(missing)}\n"
            "Please ensure these keys are set in your environment or in a .env file.",
            file=sys.stderr,
        )
        sys.exit(1)

    print("=== LangGraph Research Pipeline CLI (Stage 5) ===")
    try:
        user_query = input("Enter your research query: ").strip()
    except (EOFError, KeyboardInterrupt):
        print("\nAborted.")
        sys.exit(0)

    if not user_query:
        print("Query cannot be empty. Exiting.")
        sys.exit(1)

    print(f"\nProcessing query: \"{user_query}\"...")

    app = build_graph()
    initial_state = create_initial_state(user_query)

    try:
        final_state = app.invoke(initial_state)
        pretty_print_state(final_state)
    except Exception as exc:
        print(f"\nPipeline Error: {exc}", file=sys.stderr)
        sys.exit(1)


if __name__ == "__main__":
    main()
