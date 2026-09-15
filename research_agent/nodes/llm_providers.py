"""Multi-provider LLM abstraction layer for the Deep Research Pipeline.

Provides a unified configuration registry and factory function returning LangChain
chat models across Gemini (native SDK) and OpenAI-compatible providers
(Groq, Mistral, NVIDIA NIM, OpenRouter).
"""

from dataclasses import dataclass
import os
from typing import Any, Dict, Optional, Type

from dotenv import load_dotenv
from langchain_core.language_models.chat_models import BaseChatModel
from pydantic import BaseModel

load_dotenv()


@dataclass(frozen=True)
class ProviderConfig:
    """Configuration specification for an LLM provider."""

    provider_name: str
    env_var: str
    default_model: str
    base_url: Optional[str] = None
    is_openai_compatible: bool = True
    default_headers: Optional[Dict[str, str]] = None
    default_timeout: float = 45.0
    extra_body: Optional[Dict[str, Any]] = None


# Registry mapping provider identifier to its connectivity and model configuration
PROVIDER_REGISTRY: Dict[str, ProviderConfig] = {
    "gemini": ProviderConfig(
        provider_name="gemini",
        env_var="GOOGLE_API_KEY",
        default_model="gemini-3.6-flash",
        base_url=None,
        is_openai_compatible=False,
        default_timeout=45.0,
    ),
    "groq": ProviderConfig(
        provider_name="groq",
        env_var="GROQ_API_KEY",
        default_model="openai/gpt-oss-120b",
        base_url="https://api.groq.com/openai/v1",
        is_openai_compatible=True,
        default_timeout=45.0,
    ),
    # Disabled: Free tier rate limits too restrictive for iterative testing, revisit later if needed
    # "mistral": ProviderConfig(
    #     provider_name="mistral",
    #     env_var="MISTRAL_API_KEY",
    #     default_model="mistral-small-latest",
    #     base_url="https://api.mistral.ai/v1",
    #     is_openai_compatible=True,
    #     default_timeout=45.0,
    # ),
    "nvidia_nim": ProviderConfig(
        provider_name="nvidia_nim",
        env_var="NVIDIA_API_KEY",
        default_model="nvidia/nemotron-3-super-120b-a12b",
        base_url="https://integrate.api.nvidia.com/v1",
        is_openai_compatible=True,
        default_timeout=45.0,
        extra_body={
            "chat_template_kwargs": {"enable_thinking": False}
        },
    ),
    "nvidia_nim_ultra": ProviderConfig(
        provider_name="nvidia_nim_ultra",
        env_var="NVIDIA_API_KEY",
        default_model="nvidia/nemotron-3-ultra-550b-a55b",
        base_url="https://integrate.api.nvidia.com/v1",
        is_openai_compatible=True,
        default_timeout=60.0,
    ),
    "openrouter": ProviderConfig(
        provider_name="openrouter",
        env_var="OPENROUTER_API_KEY",
        default_model="nex-agi/nex-n2.5-mini:free",
        base_url="https://openrouter.ai/api/v1",
        is_openai_compatible=True,
        default_headers={
            "HTTP-Referer": "https://github.com/Deep-Research-Pipeline",
            "X-Title": "Deep Research Pipeline",
        },
        default_timeout=45.0,
    ),
}


def get_llm_for_provider(
    provider_name: str,
    model: Optional[str] = None,
    temperature: float = 0.0,
    timeout: Optional[float] = None,
    extra_body: Optional[Dict[str, Any]] = None,
    **kwargs: Any,
) -> BaseChatModel:
    """Instantiate a LangChain chat model instance for the specified provider.

    Args:
        provider_name: One of 'gemini', 'groq', 'nvidia_nim', 'nvidia_nim_ultra', 'openrouter'.
        model: Optional model override. If None, uses provider's default_model.
        temperature: Sampling temperature (default 0.0 for deterministic extraction).
        timeout: Request timeout in seconds. If None, uses provider's default_timeout (45s).
        extra_body: Additional request body parameters (e.g. chat_template_kwargs).
        **kwargs: Additional keyword arguments forwarded to the chat model constructor.

    Returns:
        BaseChatModel instance ready to be wrapped with `.with_structured_output(Schema)`.

    Raises:
        ValueError: If provider is unknown, disabled, or if the provider's API key is unset.
    """
    key = provider_name.lower().strip()
    if key not in PROVIDER_REGISTRY:
        raise ValueError(
            f"Unsupported provider '{provider_name}'. "
            f"Supported providers: {list(PROVIDER_REGISTRY.keys())}"
        )

    config = PROVIDER_REGISTRY[key]
    api_key = os.getenv(config.env_var)
    if not api_key:
        raise ValueError(
            f"Environment variable '{config.env_var}' for provider '{provider_name}' "
            f"is not set. Please set it in your environment or .env file."
        )

    target_model = model or config.default_model
    effective_timeout = timeout if timeout is not None else config.default_timeout

    if not config.is_openai_compatible:
        # Native Gemini client
        from langchain_google_genai import ChatGoogleGenerativeAI

        return ChatGoogleGenerativeAI(
            model=target_model,
            api_key=api_key,
            temperature=temperature,
            timeout=effective_timeout,
            **kwargs,
        )

    # Standard OpenAI-compatible client
    from langchain_openai import ChatOpenAI

    merged_extra_body = {}
    if config.extra_body:
        merged_extra_body.update(config.extra_body)
    if extra_body:
        merged_extra_body.update(extra_body)

    return ChatOpenAI(
        model=target_model,
        api_key=api_key,
        base_url=config.base_url,
        temperature=temperature,
        timeout=effective_timeout,
        request_timeout=effective_timeout,
        default_headers=config.default_headers,
        extra_body=merged_extra_body or None,
        **kwargs,
    )


def get_structured_llm_for_provider(
    provider_name: str,
    schema: Type[BaseModel],
    model: Optional[str] = None,
    temperature: float = 0.0,
    timeout: Optional[float] = None,
    extra_body: Optional[Dict[str, Any]] = None,
    **kwargs: Any,
):
    """Convenience helper returning an LLM Runnable wrapped with structured output.

    Args:
        provider_name: Name of the provider.
        schema: Target Pydantic BaseModel class.
        model: Optional model override.
        temperature: Sampling temperature.
        timeout: Request timeout in seconds (default 45s).
        extra_body: Additional request body parameters (e.g. chat_template_kwargs).
        **kwargs: Forwarded to get_llm_for_provider.

    Returns:
        A Runnable returning validated schema instances or raw dicts.
    """
    llm = get_llm_for_provider(
        provider_name=provider_name,
        model=model,
        temperature=temperature,
        timeout=timeout,
        extra_body=extra_body,
        **kwargs,
    )
    return llm.with_structured_output(schema)


