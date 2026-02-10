"""NativeTool ABC and registry for extensible tool definitions."""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any

from agent_harness.providers.base import ToolDefinition, ToolResult


class NativeTool(ABC):
    """Abstract base class for native tools provided to the agent."""

    @abstractmethod
    def get_tools(self) -> list[ToolDefinition]:
        """Return the tool definitions this native tool provides."""

    @abstractmethod
    async def call(self, tool_name: str, arguments: dict[str, Any]) -> ToolResult:
        """Execute a tool call and return the result."""


class NativeToolRegistry:
    """Maps tool names to native tool instances."""

    def __init__(self) -> None:
        self._native_tools: list[NativeTool] = []
        self._tool_map: dict[str, NativeTool] = {}

    def register(self, native_tool: NativeTool) -> None:
        """Register a native tool and index its tools."""
        self._native_tools.append(native_tool)
        for tool in native_tool.get_tools():
            self._tool_map[tool.name] = native_tool

    def get_tools(self) -> list[ToolDefinition]:
        """Return all tool definitions from all registered native tools."""
        tools: list[ToolDefinition] = []
        for native_tool in self._native_tools:
            tools.extend(native_tool.get_tools())
        return tools

    def has_tool(self, name: str) -> bool:
        return name in self._tool_map

    async def call_tool(self, name: str, arguments: dict[str, Any]) -> ToolResult:
        native_tool = self._tool_map.get(name)
        if native_tool is None:
            return ToolResult(call_id="", content=f"Unknown native tool: {name!r}", is_error=True)
        return await native_tool.call(name, arguments)
