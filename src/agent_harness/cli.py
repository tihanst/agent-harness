"""CLI entry point with argparse and REPL loop."""

from __future__ import annotations

import argparse
import asyncio
import sys

from agent_harness.agent import Agent
from agent_harness.config import Config
from agent_harness.mcp.manager import MCPManager
from agent_harness.providers import get_provider
from agent_harness.native_tools.builtin.datetime_tool import DateTimeTool
from agent_harness.native_tools.registry import NativeToolRegistry


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="agent-harness",
        description="CLI agent harness with MCP support and pluggable LLM backends",
    )
    parser.add_argument(
        "query",
        nargs="*",
        help="One-shot query. If omitted, starts interactive REPL.",
    )
    parser.add_argument(
        "-c", "--config",
        help="Path to config JSON file",
    )
    parser.add_argument(
        "-p", "--provider",
        help="LLM provider name (anthropic, openai)",
    )
    parser.add_argument(
        "-m", "--model",
        help="Model name/ID to use",
    )
    parser.add_argument(
        "-k", "--api-key",
        help="API key (overrides config and env var)",
    )
    parser.add_argument(
        "-s", "--system-prompt",
        help="System prompt",
    )
    parser.add_argument(
        "-b", "--base-url",
        help="Base URL for the LLM API (for OpenAI-compatible providers like Together.ai)",
    )
    return parser


async def run_async(args: argparse.Namespace) -> None:
    # Load config
    config = Config.load(args.config)
    config.override(
        provider=args.provider,
        model=args.model,
        api_key=args.api_key,
        base_url=args.base_url,
        system_prompt=args.system_prompt,
    )

    # Set up provider
    provider_cls = get_provider(config.provider)
    provider = provider_cls()
    provider.configure(
        model=config.model,
        api_key=config.api_key,
        system_prompt=config.system_prompt,
        base_url=config.base_url,
    )

    # Set up native tools
    native_tool_registry = NativeToolRegistry()
    native_tool_registry.register(DateTimeTool())

    # Set up MCP
    mcp_manager = MCPManager()
    if config.mcp_servers:
        print("Connecting to MCP servers...", file=sys.stderr)
        await mcp_manager.connect_all(config.mcp_servers)
        mcp_tools = mcp_manager.get_tools()
        if mcp_tools:
            print(f"  {len(mcp_tools)} tool(s) available from MCP servers", file=sys.stderr)

    agent = Agent(provider, mcp_manager, native_tool_registry)

    try:
        query = " ".join(args.query) if args.query else None
        if query:
            # One-shot mode
            await agent.run(query)
        else:
            # REPL mode
            await repl(agent)
    finally:
        await mcp_manager.close()


async def repl(agent: Agent) -> None:
    """Interactive read-eval-print loop."""
    print("agent-harness REPL (type /quit to exit, /reset to clear history)")
    while True:
        try:
            user_input = input("\n> ").strip()
        except (EOFError, KeyboardInterrupt):
            print()
            break

        if not user_input:
            continue
        if user_input == "/quit":
            break
        if user_input == "/reset":
            agent.reset()
            print("Conversation history cleared.")
            continue

        await agent.run(user_input)


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()
    asyncio.run(run_async(args))


if __name__ == "__main__":
    main()
