"""OpenAI-compatible provider with streaming and tool use."""

from __future__ import annotations

import json
import os
from typing import Any, AsyncIterator

import openai

from agent_harness.providers.base import (
    AssistantMessage,
    LLMProvider,
    ToolCall,
    ToolDefinition,
    ToolResult,
)

DEFAULT_MODEL = "gpt-4o"


class OpenAIProvider(LLMProvider):
    def __init__(self) -> None:
        self._client: openai.AsyncOpenAI | None = None
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
        key = api_key or os.environ.get("OPENAI_API_KEY") or "dummy"
        kwargs: dict[str, Any] = {"api_key": key}
        if base_url:
            kwargs["base_url"] = base_url
        self._client = openai.AsyncOpenAI(**kwargs)

    def _format_tools(self, tools: list[ToolDefinition] | None) -> list[dict[str, Any]]:
        if not tools:
            return []
        return [
            {
                "type": "function",
                "function": {
                    "name": t.name,
                    "description": t.description,
                    "parameters": t.input_schema,
                },
            }
            for t in tools
        ]

    async def send(
        self,
        messages: list[dict[str, Any]],
        tools: list[ToolDefinition] | None = None,
    ) -> AssistantMessage:
        assert self._client is not None, "Provider not configured. Call configure() first."
        full_messages = self._build_messages(messages)
        kwargs: dict[str, Any] = {
            "model": self._model,
            "messages": full_messages,
        }
        formatted_tools = self._format_tools(tools)
        if formatted_tools:
            kwargs["tools"] = formatted_tools
        response = await self._client.chat.completions.create(**kwargs)
        choice = response.choices[0]
        text = choice.message.content or ""
        tool_calls: list[ToolCall] = []
        if choice.message.tool_calls:
            for tc in choice.message.tool_calls:
                tool_calls.append(
                    ToolCall(
                        id=tc.id,
                        name=tc.function.name,
                        arguments=json.loads(tc.function.arguments),
                    )
                )
        return AssistantMessage(text=text, tool_calls=tool_calls)

    async def send_streaming(
        self,
        messages: list[dict[str, Any]],
        tools: list[ToolDefinition] | None = None,
    ) -> AsyncIterator[str | AssistantMessage]:
        assert self._client is not None, "Provider not configured. Call configure() first."
        full_messages = self._build_messages(messages)
        kwargs: dict[str, Any] = {
            "model": self._model,
            "messages": full_messages,
            "stream": True,
        }
        formatted_tools = self._format_tools(tools)
        if formatted_tools:
            kwargs["tools"] = formatted_tools

        text_parts: list[str] = []
        # tool_call index -> accumulated data
        tc_map: dict[int, dict[str, str]] = {}

        stream = await self._client.chat.completions.create(**kwargs)
        async for chunk in stream:
            delta = chunk.choices[0].delta if chunk.choices else None
            if delta is None:
                continue
            if delta.content:
                text_parts.append(delta.content)
                yield delta.content
            if delta.tool_calls:
                for tc_delta in delta.tool_calls:
                    idx = tc_delta.index
                    if idx not in tc_map:
                        tc_map[idx] = {"id": "", "name": "", "arguments": ""}
                    if tc_delta.id:
                        tc_map[idx]["id"] = tc_delta.id
                    if tc_delta.function:
                        if tc_delta.function.name:
                            tc_map[idx]["name"] = tc_delta.function.name
                        if tc_delta.function.arguments:
                            tc_map[idx]["arguments"] += tc_delta.function.arguments

        tool_calls: list[ToolCall] = []
        for idx in sorted(tc_map):
            data = tc_map[idx]
            tool_calls.append(
                ToolCall(
                    id=data["id"],
                    name=data["name"],
                    arguments=json.loads(data["arguments"]) if data["arguments"] else {},
                )
            )

        yield AssistantMessage(text="".join(text_parts), tool_calls=tool_calls)

    def _build_messages(self, messages: list[dict[str, Any]]) -> list[dict[str, Any]]:
        result: list[dict[str, Any]] = []
        if self._system_prompt:
            result.append({"role": "system", "content": self._system_prompt})
        result.extend(messages)
        return result

    def format_tool_results(
        self,
        assistant_msg: AssistantMessage,
        results: list[ToolResult],
    ) -> list[dict[str, Any]]:
        # Build the assistant message with tool_calls
        assistant_dict: dict[str, Any] = {"role": "assistant"}
        if assistant_msg.text:
            assistant_dict["content"] = assistant_msg.text
        else:
            assistant_dict["content"] = None
        if assistant_msg.tool_calls:
            assistant_dict["tool_calls"] = [
                {
                    "id": tc.id,
                    "type": "function",
                    "function": {
                        "name": tc.name,
                        "arguments": json.dumps(tc.arguments),
                    },
                }
                for tc in assistant_msg.tool_calls
            ]

        out: list[dict[str, Any]] = [assistant_dict]
        for r in results:
            out.append({
                "role": "tool",
                "tool_call_id": r.call_id,
                "content": r.content,
            })
        return out
