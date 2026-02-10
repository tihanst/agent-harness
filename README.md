# Agent Harness

A CLI agent harness with MCP server support, built-in native tools, and pluggable LLM backends. It connects any supported LLM to external tools via the [Model Context Protocol](https://modelcontextprotocol.io/) and in-process native tools, managing the agentic tool-use loop so the LLM can call tools, process results, and iterate toward an answer.

The harness itself contains no model logic — it orchestrates the connection between LLMs and tools.

## Requirements

- Python 3.11+
- [uv](https://docs.astral.sh/uv/) package manager
- Node.js/npm (only if using MCP servers that run via `npx`)

## Quick Start

```bash
# One-shot query using Anthropic (default provider)
export ANTHROPIC_API_KEY="your-key"
uv run agent-harness "What time is it?"

# Interactive REPL
uv run agent-harness

# With a config file
uv run agent-harness -c config.json

# Override provider and model
uv run agent-harness -p openai -m gpt-4o "Summarize this project"
```

## Configuration

The agent can be configured with a JSON file, CLI flags, or both. CLI flags override config values.

| JSON key       | CLI flag             | Default                              |
|----------------|----------------------|--------------------------------------|
| `provider`     | `-p` / `--provider`  | `anthropic`                          |
| `model`        | `-m` / `--model`     | Provider default (hard-coded claude-sonnet-4-5-20250929 (Anthropic), gpt-4o (OpenAI) |
| `apiKey`       | `-k` / `--api-key`   | `ANTHROPIC_API_KEY` or `OPENAI_API_KEY` env var |
| `baseUrl`      | `-b` / `--base-url`  | Provider default                     |
| `systemPrompt` | `-s` / `--system-prompt` | `You are a helpful assistant.`   |
| `mcpServers`   | Config only          | None                                 |

### Example Config

```json
{
  "provider": "anthropic",
  "model": "claude-sonnet-4-5-20250929",
  "systemPrompt": "You are a helpful assistant.",
  "mcpServers": {
    "filesystem": {
      "command": "npx",
      "args": ["-y", "@modelcontextprotocol/server-filesystem", "/tmp"],
      "env": {}
    }
  }
}
```

## Providers

The harness supports multiple LLM providers through a pluggable backend system. Anthropic and OpenAI have dedicated implementations because their APIs differ in wire format. Any OpenAI-compatible API (Together.ai, Groq, Ollama, etc.) works through the OpenAI provider with a custom `base_url`.

| Provider    | Example usage                                                        |
|-------------|----------------------------------------------------------------------|
| Anthropic   | `uv run agent-harness -p anthropic -m claude-sonnet-4-5-20250929 "query"` |
| OpenAI      | `uv run agent-harness -p openai -m gpt-4o "query"`                   |
| Together.ai | `uv run agent-harness -p together -b https://api.together.xyz/v1 -m meta-llama/Llama-3-70B "query"` |
| Groq        | `uv run agent-harness -p groq -b https://api.groq.com/openai/v1 -m llama3-70b-8192 "query"` |
| Ollama      | `uv run agent-harness -p ollama -b http://localhost:11434/v1 -m llama3 "query"` |

`together`, `groq`, and `ollama` are aliases for the OpenAI provider — they use the same code, just pointed at a different URL.

## MCP Servers

MCP servers are external processes that expose tools to the agent over stdio. The agent spawns them as subprocesses, discovers their tools at startup, and routes tool calls to the appropriate server during the agent loop.

Any MCP server that supports stdio transport works. Configure them in the `mcpServers` section of your config file:

```json
{
  "mcpServers": {
    "postgres": {
      "command": "npx",
      "args": ["-y", "@modelcontextprotocol/server-postgres", "postgresql://localhost/mydb"]
    },
    "huggingface": {
      "command": "npx",
      "args": ["-y", "@llmindset/hf-mcp-server"],
      "env": {
        "HF_TOKEN": "hf_..."
      }
    }
  }
}
```

If two servers expose tools with the same name, the second tool is prefixed with the server name (e.g. `huggingface__search`).

## Built-in Native Tools

Native tools run in-process without an external server. They are checked before MCP tools when routing a tool call.

| Tool                   | Description                            |
|------------------------|----------------------------------------|
| `get_current_datetime` | Returns current date/time in ISO 8601  |

## REPL Commands

| Command  | Description              |
|----------|--------------------------|
| `/quit`  | Exit the REPL            |
| `/reset` | Clear conversation history |

## How It Works

1. The CLI parses config and flags, selects a provider class, and configures it
2. MCP servers are spawned and their tools are discovered
3. The user's message is sent to the LLM along with all available tools
4. If the LLM returns tool calls, the agent executes them (native tools first, then MCP) and sends results back to the LLM
5. Steps 3-4 repeat until the LLM returns a text response (or 20 iterations are reached)

## Project Structure

```
src/agent_harness/
├── cli.py              # Entry point, argument parsing, REPL loop
├── config.py           # Config dataclass and JSON loading
├── agent.py            # Agentic tool-use loop
├── providers/
│   ├── __init__.py     # Provider registry and factory
│   ├── base.py         # LLMProvider ABC and shared data types
│   ├── anthropic.py    # Anthropic Claude implementation
│   └── openai.py       # OpenAI-compatible implementation
├── mcp/
│   └── manager.py      # Multi-server MCP connections and tool routing
└── native_tools/
    ├── registry.py     # NativeTool ABC and registry
    └── builtin/
        └── datetime_tool.py
```

## Adding Providers and Tools 

### Adding a Provider

1. Subclass `LLMProvider` in `providers/`
2. Implement `configure()`, `send()`, `send_streaming()`, and `format_tool_results()`
3. Register it in `providers/__init__.py`: `register_provider("name", MyProvider)`

### Adding a Native Tool

1. Subclass `NativeTool` in `native_tools/builtin/`
2. Implement `get_tools()` and `call()`
3. Register it in `cli.py`: `native_tool_registry.register(MyTool())`
