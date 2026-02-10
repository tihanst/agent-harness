"""Multi-server MCP connection manager with tool aggregation and call routing."""

from __future__ import annotations

import json
import sys
from contextlib import AsyncExitStack
from typing import Any

from mcp import ClientSession, StdioServerParameters
from mcp.client.stdio import stdio_client

from agent_harness.config import MCPServerConfig
from agent_harness.providers.base import ToolDefinition, ToolResult


class MCPManager:
    """Manages connections to multiple MCP servers via stdio transport."""

    def __init__(self) -> None:
        self._exit_stack = AsyncExitStack()
        self._sessions: dict[str, ClientSession] = {}
        # tool_name -> server_name
        self._tool_routes: dict[str, str] = {}
        self._tools: list[ToolDefinition] = []

    async def connect_all(self, servers: dict[str, MCPServerConfig]) -> None:
        """Connect to all configured MCP servers and discover tools."""
        for name, config in servers.items():
            try:
                await self._connect_server(name, config)
            except Exception as e:
                print(f"Warning: Failed to connect to MCP server {name!r}: {e}", file=sys.stderr)

    async def _connect_server(self, name: str, config: MCPServerConfig) -> None:
        params = StdioServerParameters(
            command=config.command,
            args=config.args,
            env=config.env if config.env else None,
        )
        stdio_transport = await self._exit_stack.enter_async_context(
            stdio_client(params)
        )
        read_stream, write_stream = stdio_transport
        session = await self._exit_stack.enter_async_context(
            ClientSession(read_stream, write_stream)
        )
        await session.initialize()
        self._sessions[name] = session

        # Discover tools
        result = await session.list_tools()
        for tool in result.tools:
            # Handle name collisions by prefixing
            tool_name = tool.name
            if tool_name in self._tool_routes:
                tool_name = f"{name}__{tool.name}"
            self._tool_routes[tool_name] = name
            self._tools.append(
                ToolDefinition(
                    name=tool_name,
                    description=tool.description or "",
                    input_schema=tool.inputSchema,
                )
            )

    def get_tools(self) -> list[ToolDefinition]:
        """Return all discovered tools across all servers."""
        return list(self._tools)

    def has_tool(self, name: str) -> bool:
        return name in self._tool_routes

    async def call_tool(self, name: str, arguments: dict[str, Any]) -> ToolResult:
        """Route a tool call to the appropriate server."""
        server_name = self._tool_routes.get(name)
        if server_name is None:
            return ToolResult(
                call_id="",
                content=f"Unknown MCP tool: {name!r}",
                is_error=True,
            )
        session = self._sessions[server_name]
        # If name was prefixed, strip it to get the original tool name
        original_name = name
        if name.startswith(f"{server_name}__"):
            original_name = name[len(f"{server_name}__"):]
        try:
            result = await session.call_tool(original_name, arguments)
            # Extract text content from result
            text_parts = []
            for content in result.content:
                if hasattr(content, "text"):
                    text_parts.append(content.text)
                else:
                    text_parts.append(json.dumps(content.model_dump()))
            return ToolResult(
                call_id="",  # caller fills in the call_id
                content="\n".join(text_parts),
                is_error=result.isError if hasattr(result, "isError") else False,
            )
        except Exception as e:
            return ToolResult(call_id="", content=str(e), is_error=True)

    async def close(self) -> None:
        """Close all server connections."""
        await self._exit_stack.aclose()
