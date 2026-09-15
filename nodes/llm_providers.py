"""Shim forwarding to research_agent.nodes.llm_providers."""

from research_agent.nodes.llm_providers import (  # noqa: F401
    PROVIDER_REGISTRY,
    ProviderConfig,
    get_llm_for_provider,
    get_structured_llm_for_provider,
)

__all__ = [
    "ProviderConfig",
    "PROVIDER_REGISTRY",
    "get_llm_for_provider",
    "get_structured_llm_for_provider",
]
