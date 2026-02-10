"""Built-in native tool: get the current date and time."""

from __future__ import annotations

from datetime import datetime, timezone
from typing import Any

from agent_harness.providers.base import ToolDefinition, ToolResult
from agent_harness.native_tools.registry import NativeTool


class DateTimeTool(NativeTool):
    def get_tools(self) -> list[ToolDefinition]:
        return [
            ToolDefinition(
                name="get_current_datetime",
                description="Get the current date and time in ISO 8601 format. Optionally specify a timezone offset like '+02:00' or '-05:00'.",
                input_schema={
                    "type": "object",
                    "properties": {
                        "timezone_offset": {
                            "type": "string",
                            "description": "UTC offset in format '+HH:MM' or '-HH:MM'. Defaults to UTC.",
                        }
                    },
                    "required": [],
                },
            )
        ]

    async def call(self, tool_name: str, arguments: dict[str, Any]) -> ToolResult:
        offset_str = arguments.get("timezone_offset")
        if offset_str:
            try:
                # Parse ±HH:MM
                sign = 1 if offset_str.startswith("+") else -1
                parts = offset_str.lstrip("+-").split(":")
                hours = int(parts[0])
                minutes = int(parts[1]) if len(parts) > 1 else 0
                from datetime import timedelta

                tz = timezone(timedelta(hours=sign * hours, minutes=sign * minutes))
            except (ValueError, IndexError):
                return ToolResult(
                    call_id="",
                    content=f"Invalid timezone offset: {offset_str!r}. Use format '+HH:MM' or '-HH:MM'.",
                    is_error=True,
                )
        else:
            tz = timezone.utc

        now = datetime.now(tz)
        return ToolResult(call_id="", content=now.isoformat())
