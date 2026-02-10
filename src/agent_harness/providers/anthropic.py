"""Anthropic Claude provider with streaming and tool use."""

from __future__ import annotations

import os
from typing import Any, AsyncIterator

import anthropic

from agent_harness.providers.base import (
    AssistantMessage,
    LLMProvider,
    ToolCall,
    ToolDefinition,
    ToolResult,
)

DEFAULT_MODEL = "claude-sonnet-4-5-20250929"


class AnthropicProvider(LLMProvider):
    def __init__(self) -> None:
        self._client: anthropic.AsyncAnthropic | None = None
        self._model: str = DEFAULT_MODEL
        self._system_prompt: str = ""

    def configure(
        self,
        model: str | None = None,
        api_key: str | None = None,
        system_prompt: str = "",
        base_url: str | None = None,
    ) -> None:
        self._model = model or DEFAULT_MODEL
        self._system_prompt = system_prompt
        key = api_key or os.environ.get("ANTHROPIC_API_KEY")
        kwargs: dict[str, Any] = {"api_key": key}
        if base_url:
            kwargs["base_url"] = base_url
        self._client = anthropic.AsyncAnthropic(**kwargs)

    def _format_tools(self, tools: list[ToolDefinition] | None) -> list[dict[str, Any]]:
        if not tools:
            return []
        return [
            {
                "name": t.name,
                "description": t.description,
                "input_schema": t.input_schema,
            }
            for t in tools
        ]

    def _parse_response(self, response: anthropic.types.Message) -> AssistantMessage:
        text_parts: list[str] = []
        tool_calls: list[ToolCall] = []
        for block in response.content:
            if block.type == "text":
                text_parts.append(block.text)
            elif block.type == "tool_use":
                tool_calls.append(
                    ToolCall(id=block.id, name=block.name, arguments=block.input)
                )
        return AssistantMessage(text="\n".join(text_parts), tool_calls=tool_calls)

    async def send(
        self,
        messages: list[dict[str, Any]],
        tools: list[ToolDefinition] | None = None,
    ) -> AssistantMessage:
        assert self._client is not None, "Provider not configured. Call configure() first."
        kwargs: dict[str, Any] = {
            "model": self._model,
            "max_tokens": 8192,
            "messages": messages,
        }
        if self._system_prompt:
            kwargs["system"] = self._system_prompt
        formatted_tools = self._format_tools(tools)
        if formatted_tools:
            kwargs["tools"] = formatted_tools
        response = await self._client.messages.create(**kwargs)
        return self._parse_response(response)

    async def send_streaming(
        self,
        messages: list[dict[str, Any]],
        tools: list[ToolDefinition] | None = None,
    ) -> AsyncIterator[str | AssistantMessage]:
        assert self._client is not None, "Provider not configured. Call configure() first."
        kwargs: dict[str, Any] = {
            "model": self._model,
            "max_tokens": 8192,
            "messages": messages,
        }
        if self._system_prompt:
            kwargs["system"] = self._system_prompt
        formatted_tools = self._format_tools(tools)
        if formatted_tools:
            kwargs["tools"] = formatted_tools

        text_parts: list[str] = []
        tool_calls: list[ToolCall] = []
        current_tool_id: str = ""
        current_tool_name: str = ""
        current_tool_json: str = ""

        async with self._client.messages.stream(**kwargs) as stream:
            async for event in stream:
                if event.type == "content_block_start":
                    if event.content_block.type == "tool_use":
                        current_tool_id = event.content_block.id
                        current_tool_name = event.content_block.name
                        current_tool_json = ""
                elif event.type == "content_block_delta":
                    if event.delta.type == "text_delta":
                        text_parts.append(event.delta.text)
                        yield event.delta.text
                    elif event.delta.type == "input_json_delta":
                        current_tool_json += event.delta.partial_json
                elif event.type == "content_block_stop":
                    if current_tool_id:
                        import json
                        args = json.loads(current_tool_json) if current_tool_json else {}
                        tool_calls.append(
                            ToolCall(id=current_tool_id, name=current_tool_name, arguments=args)
                        )
                        current_tool_id = ""
                        current_tool_name = ""
                        current_tool_json = ""

        yield AssistantMessage(text="".join(text_parts), tool_calls=tool_calls)

    def format_tool_results(
        self,
        assistant_msg: AssistantMessage,
        results: list[ToolResult],
    ) -> list[dict[str, Any]]:
        # Build the assistant message content blocks
        content: list[dict[str, Any]] = []
        if assistant_msg.text:
            content.append({"type": "text", "text": assistant_msg.text})
        for tc in assistant_msg.tool_calls:
            content.append({
                "type": "tool_use",
                "id": tc.id,
                "name": tc.name,
                "input": tc.arguments,
            })

        # Build tool result messages
        result_map = {r.call_id: r for r in results}
        tool_result_content: list[dict[str, Any]] = []
        for tc in assistant_msg.tool_calls:
            r = result_map[tc.id]
            tool_result_content.append({
                "type": "tool_result",
                "tool_use_id": tc.id,
                "content": r.content,
                **({"is_error": True} if r.is_error else {}),
            })

        return [
            {"role": "assistant", "content": content},
            {"role": "user", "content": tool_result_content},
        ]
