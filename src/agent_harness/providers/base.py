"""Abstract base types for LLM providers."""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any, AsyncIterator


@dataclass
class ToolDefinition:
    """Provider-agnostic tool definition."""

    name: str
    description: str
    input_schema: dict[str, Any]


@dataclass
class ToolCall:
    """A tool invocation requested by the LLM."""

    id: str
    name: str
    arguments: dict[str, Any]


@dataclass
class ToolResult:
    """Result of executing a tool call."""

    call_id: str
    content: str
    is_error: bool = False


@dataclass
class AssistantMessage:
    """Response from the LLM, possibly containing tool calls."""

    text: str = ""
    tool_calls: list[ToolCall] = field(default_factory=list)


class LLMProvider(ABC):
    """Abstract base class for LLM providers."""

    @abstractmethod
    def configure(
        self,
        model: str | None = None,
        api_key: str | None = None,
        system_prompt: str = "",
        base_url: str | None = None,
    ) -> None:
        """Configure the provider with model, API key, system prompt, and optional base URL."""

    @abstractmethod
    async def send(
        self,
        messages: list[dict[str, Any]],
        tools: list[ToolDefinition] | None = None,
    ) -> AssistantMessage:
        """Send messages and return a complete response."""

    @abstractmethod
    async def send_streaming(
        self,
        messages: list[dict[str, Any]],
        tools: list[ToolDefinition] | None = None,
    ) -> AsyncIterator[str | AssistantMessage]:
        """Stream response text chunks, yielding the final AssistantMessage last."""
        # yield is needed to make this a valid async generator stub
        yield  # type: ignore[misc]

    @abstractmethod
    def format_tool_results(
        self,
        assistant_msg: AssistantMessage,
        results: list[ToolResult],
    ) -> list[dict[str, Any]]:
        """Format an assistant message + tool results as message dicts for the next send() call."""
