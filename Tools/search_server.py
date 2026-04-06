from __future__ import annotations

import os
from pathlib import Path

from mcp.server.fastmcp import FastMCP
from tavily import TavilyClient


PROJECT_ROOT = Path(__file__).resolve().parent.parent
ENV_PATH = PROJECT_ROOT / ".env"


def load_project_env() -> None:
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


mcp = FastMCP("search")
TAVILY_API_KEY = os.getenv("TAVILY_API_KEY", "").strip()
tavily = TavilyClient(api_key=TAVILY_API_KEY) if TAVILY_API_KEY else None

@mcp.tool()
def search_web(query: str) -> str:
    """Search the web for real-time information.
    Use this for factual questions, historical data, or general lookups."""
    if tavily is None:
        return "Search server error: TAVILY_API_KEY is not set. Add it to .env or your environment."

    try:
        # depth="basic" is faster and costs 1 credit
        response = tavily.search(query=query, search_depth="basic", max_results=3)
        results = response.get('results', [])
        
        if not results:
            return f"No results found for: '{query}'"
            
        return "\n\n".join([
            f"[{i+1}] {r['title']}\n    {r['content']}"
            for i, r in enumerate(results)
        ])
    except Exception as e:
        return f"Search error: {e}"

@mcp.tool()
def search_news(query: str) -> str:
    """Search for the latest news articles on a topic.
    Use this for recent events, announcements, or developments within the last month."""
    if tavily is None:
        return "Search server error: TAVILY_API_KEY is not set. Add it to .env or your environment."

    try:
        # topic="news" triggers Tavily's news-specific crawler
        response = tavily.search(query=query, topic="news", search_depth="basic", max_results=3)
        results = response.get('results', [])
        
        if not results:
            return f"No news found for: '{query}'"
            
        return "\n\n".join([
            f"[{i+1}] {r['title']}\n"
            f"    Date: {r.get('published_date', 'Recent')}\n"
            f"    Content: {r['content']}\n"
            f"    Source: {r.get('url', 'Unknown')}"
            for i, r in enumerate(results)
        ])
    except Exception as e:
        return f"News search error: {e}"

if __name__ == "__main__":
    mcp.run(transport="stdio")
