from __future__ import annotations

from typing import Callable, Dict
import urllib.parse

from .types import LLMAdapter

ProviderFactory = Callable[[urllib.parse.ParseResult, Dict[str, str]], LLMAdapter]

_PROVIDER_REGISTRY: Dict[str, ProviderFactory] = {}


def register_llm_provider(name: str, factory: ProviderFactory) -> None:
    _PROVIDER_REGISTRY[name] = factory


def from_url(dsn: str) -> LLMAdapter:
    """
    DSN: llm+<provider>://<host>/<model>?k=v&...
    Example:
      llm+openai:///gpt-4.1-mini?api_key_env=OPENAI_API_KEY
      llm+openai://127.0.0.1:8000/Qwen2.5?api_key=none
    """
    u = urllib.parse.urlparse(dsn)
    if not u.scheme.startswith("llm+"):
        raise ValueError("DSN must start with scheme: llm+<provider>://...")

    provider = u.scheme.split("+", 1)[1].strip()
    if not provider:
        raise ValueError("Missing provider in DSN: llm+<provider>://...")

    qs = {k: v[-1] for k, v in urllib.parse.parse_qs(u.query).items()}

    factory = _PROVIDER_REGISTRY.get(provider)
    if factory is None:
        raise ValueError(f"Unknown LLM provider '{provider}'. Registered: {sorted(_PROVIDER_REGISTRY)}")

    return factory(u, qs)
