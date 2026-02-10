"""Configuration loading from JSON files with CLI overrides."""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any


@dataclass
class MCPServerConfig:
    """Configuration for a single MCP server."""

    command: str
    args: list[str] = field(default_factory=list)
    env: dict[str, str] = field(default_factory=dict)


@dataclass
class Config:
    """Top-level agent configuration."""

    provider: str = "anthropic"
    model: str | None = None
    api_key: str | None = None
    base_url: str | None = None
    system_prompt: str = "You are a helpful assistant."
    mcp_servers: dict[str, MCPServerConfig] = field(default_factory=dict)

    @classmethod
    def load(cls, path: str | Path | None = None) -> Config:
        """Load config from a JSON file. Returns defaults if no path given."""
        if path is None:
            return cls()
        data = json.loads(Path(path).read_text())
        return cls._from_dict(data)

    @classmethod
    def _from_dict(cls, data: dict[str, Any]) -> Config:
        servers: dict[str, MCPServerConfig] = {}
        for name, srv in data.get("mcpServers", {}).items():
            servers[name] = MCPServerConfig(
                command=srv["command"],
                args=srv.get("args", []),
                env=srv.get("env", {}),
            )
        return cls(
            provider=data.get("provider", "anthropic"),
            model=data.get("model"),
            api_key=data.get("apiKey"),
            base_url=data.get("baseUrl"),
            system_prompt=data.get("systemPrompt", "You are a helpful assistant."),
            mcp_servers=servers,
        )

    def override(self, **kwargs: Any) -> None:
        """Apply CLI overrides. None values are skipped."""
        for key, value in kwargs.items():
            if value is not None and hasattr(self, key):
                setattr(self, key, value)
