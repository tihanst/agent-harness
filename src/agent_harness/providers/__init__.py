"""LLM provider registry and factory."""

from agent_harness.providers.base import LLMProvider

_PROVIDERS: dict[str, type[LLMProvider]] = {}


def register_provider(name: str, cls: type[LLMProvider]) -> None:
    _PROVIDERS[name] = cls


def get_provider(name: str) -> type[LLMProvider]:
    if name not in _PROVIDERS:
        raise ValueError(f"Unknown provider: {name!r}. Available: {list(_PROVIDERS)}")
    return _PROVIDERS[name]


def _register_builtins() -> None:
    from agent_harness.providers.anthropic import AnthropicProvider
    from agent_harness.providers.openai import OpenAIProvider

    register_provider("anthropic", AnthropicProvider)
    register_provider("openai", OpenAIProvider)
    register_provider("together", OpenAIProvider)
    register_provider("groq", OpenAIProvider)
    register_provider("ollama", OpenAIProvider)


_register_builtins()
