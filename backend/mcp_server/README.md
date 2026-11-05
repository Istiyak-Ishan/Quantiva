# Quantiva MCP Server

Exposes Quantiva's document intelligence as [Model Context Protocol](https://modelcontextprotocol.io) tools, making Quantiva usable from any MCP-compatible agent host.

## Tools

| Tool | Description |
|------|-------------|
| `search_documents` | Hybrid semantic + keyword search over user documents |
| `summarize_document` | AI-generated document summary (concise or detailed) |
| `ask_question` | Full multi-step reasoning with citations and confidence |

## Setup

```bash
pip install fastmcp httpx
```

## Add to Claude Desktop

Edit `~/.claude/config.json` (macOS) or `%APPDATA%\Claude\config.json` (Windows):

```json
{
  "mcpServers": {
    "quantiva": {
      "command": "python",
      "args": ["-m", "mcp_server.server"],
      "cwd": "/path/to/quantiva/backend",
      "env": {
        "QUANTIVA_API_URL": "http://localhost:8000/api/v1",
        "QUANTIVA_TOKEN": "your-jwt-token"
      }
    }
  }
}
```

## Add to Antigravity IDE

Open Settings → MCP Servers → Add:

```json
{
  "name": "quantiva",
  "command": "python -m mcp_server.server",
  "cwd": "I:\\AI Engineer Projects\\Quantiva\\backend",
  "env": {
    "QUANTIVA_API_URL": "http://localhost:8000/api/v1",
    "QUANTIVA_TOKEN": "your-jwt-token"
  }
}
```

## Why MCP Matters

Without MCP, every agent that wants to use Quantiva needs a custom integration.
With MCP, any host that speaks the protocol gets all three tools instantly — zero
extra code on the host side.
