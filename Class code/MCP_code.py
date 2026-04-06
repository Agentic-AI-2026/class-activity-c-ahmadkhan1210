from __future__ import annotations

import sys
from pathlib import Path

import nest_asyncio

nest_asyncio.apply()

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from mcp_runtime import get_mcp_tools  # noqa: E402


print("MCP client helper ready")

tools, tools_map = await get_mcp_tools(["search", "math"])
