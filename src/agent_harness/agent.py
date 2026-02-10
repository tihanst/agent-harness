"""Core agentic tool-use loop."""

from __future__ import annotations

import sys
from typing import Any

from agent_harness.mcp.manager import MCPManager
from agent_harness.providers.base import (
    AssistantMessage,
    LLMProvider,
    ToolDefinition,
    ToolResult,
)
from agent_harness.native_tools.registry import NativeToolRegistry

MAX_ITERATIONS = 20


class Agent:
    """Orchestrates the LLM ↔ tool-use loop."""

    def __init__(
        self,
        provider: LLMProvider,
        mcp_manager: MCPManager,
        native_tool_registry: NativeToolRegistry,
    ) -> None:
        self._provider = provider
        self._mcp = mcp_manager
        self._native_tools = native_tool_registry
        self._messages: list[dict[str, Any]] = []

    def reset(self) -> None:
        """Clear conversation history."""
        self._messages.clear()

    def _get_all_tools(self) -> list[ToolDefinition]:
        """Aggregate tools from native tools and MCP servers."""
        tools: list[ToolDefinition] = []
        tools.extend(self._native_tools.get_tools())
        tools.extend(self._mcp.get_tools())
        return tools

    async def _execute_tool(self, name: str, arguments: dict[str, Any]) -> ToolResult:
        """Route tool call: native tools first, then MCP."""
        if self._native_tools.has_tool(name):
            return await self._native_tools.call_tool(name, arguments)
        if self._mcp.has_tool(name):
            return await self._mcp.call_tool(name, arguments)
        return ToolResult(call_id="", content=f"Unknown tool: {name!r}", is_error=True)

    async def run(self, user_message: str) -> str:
        """Run the full agent loop for a user message. Returns final text response."""
        self._messages.append({"role": "user", "content": user_message})
        tools = self._get_all_tools()

        for _ in range(MAX_ITERATIONS):
            response = await self._run_streaming(tools)

            if not response.tool_calls:
                return response.text

            # Execute tool calls
            results: list[ToolResult] = []
            for tc in response.tool_calls:
                result = await self._execute_tool(tc.name, tc.arguments)
                result.call_id = tc.id
                results.append(result)
                status = "error" if result.is_error else "ok"
                print(f"  [{tc.name}] → {status}", file=sys.stderr)

            # Format and append to history
            formatted = self._provider.format_tool_results(response, results)
            self._messages.extend(formatted)

        return "[Agent stopped: maximum iterations reached]"

    async def _run_streaming(self, tools: list[ToolDefinition]) -> AssistantMessage:
        """Send with streaming, printing text as it arrives."""
        final_msg: AssistantMessage | None = None
        printed_any_text = False
        async for chunk in self._provider.send_streaming(self._messages, tools or None):
            if isinstance(chunk, str):
                print(chunk, end="", flush=True)
                printed_any_text = True
            elif isinstance(chunk, AssistantMessage):
                final_msg = chunk
        if printed_any_text:
            print()  # newline after streamed text
        assert final_msg is not None
        return final_msg
