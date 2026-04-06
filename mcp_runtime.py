from __future__ import annotations

import os
import sys
from pathlib import Path
from typing import Any

from langchain_mcp_adapters.client import MultiServerMCPClient


PROJECT_ROOT = Path(__file__).resolve().parent
ENV_PATH = PROJECT_ROOT / ".env"


def load_project_env() -> None:
    """Load simple KEY=VALUE pairs from the local .env file if it exists."""

    if not ENV_PATH.exists():
        return

    for raw_line in ENV_PATH.read_text(encoding="utf-8").splitlines():
        line = raw_line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue

        key, value = line.split("=", 1)
        key = key.strip()
        value = value.strip().strip('"').strip("'")
        if key:
            os.environ[key] = value


load_project_env()


MCP_CONFIG = {
    "math": {
        "command": sys.executable,
        "args": [str(PROJECT_ROOT / "Tools" / "math_server.py")],
        "transport": "stdio",
    },
    "search": {
        "command": sys.executable,
        "args": [str(PROJECT_ROOT / "Tools" / "search_server.py")],
        "transport": "stdio",
    },
    "weather": {
        "url": os.getenv("WEATHER_MCP_URL", "http://localhost:8000/mcp"),
        "transport": "streamable_http",
    },
}


mcp_client = MultiServerMCPClient(MCP_CONFIG)


async def get_mcp_tools(server_names: list[str]) -> tuple[list[Any], dict[str, Any]]:
    """Return MCP tools and a name -> tool lookup map."""

    tools: list[Any] = []
    for server_name in server_names:
        server_tools = await mcp_client.get_tools(server_name=server_name)
        tools.extend(server_tools)

    tool_map = {tool.name: tool for tool in tools}
    return tools, tool_map