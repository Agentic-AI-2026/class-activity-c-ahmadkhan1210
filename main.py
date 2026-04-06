from __future__ import annotations

import argparse
import asyncio
import os
from typing import Any

from langchain_anthropic import ChatAnthropic
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_groq import ChatGroq
from langchain_ollama import ChatOllama

from graph import AgentState, build_react_graph
from mcp_runtime import get_mcp_tools, load_project_env


DEFAULT_QUERY = (
	"What is the weather in Lahore and who is the current Prime Minister of Pakistan? "
	"Now get the age of PM and tell us will this weather suits PM health."
)


def create_llm() -> Any:
	provider = os.getenv("LLM_PROVIDER", "ollama").strip().lower()
	temperature = float(os.getenv("LLM_TEMPERATURE", "0"))

	if provider == "anthropic":
		return ChatAnthropic(
			model=os.getenv("ANTHROPIC_MODEL", "claude-3-5-sonnet-latest"),
			temperature=temperature,
		)

	if provider == "google":
		return ChatGoogleGenerativeAI(
			model=os.getenv("GOOGLE_MODEL", "gemini-2.0-flash"),
			temperature=temperature,
		)

	if provider == "ollama":
		return ChatOllama(
			model=os.getenv("OLLAMA_MODEL", "llama3.1"),
			temperature=temperature,
		)

	if provider == "groq":
		return ChatGroq(
			model=os.getenv("GROQ_MODEL", "llama-3.3-70b-versatile"),
			temperature=temperature,
		)

	raise ValueError(f"Unsupported LLM_PROVIDER: {provider}")


def parse_args() -> argparse.Namespace:
	parser = argparse.ArgumentParser(description="Run the LangGraph ReAct agent.")
	parser.add_argument(
		"--query",
		default=os.getenv("TEST_QUERY", DEFAULT_QUERY),
		help="User query to run through the agent.",
	)
	return parser.parse_args()


async def run_agent(user_input: str) -> dict[str, Any]:
	load_project_env()
	llm = create_llm()
	tools, tools_map = await get_mcp_tools(["search", "math", "weather"])
	graph = build_react_graph(llm, tools_map, max_steps=int(os.getenv("MAX_STEPS", "8")))

	initial_state: AgentState = {
		"input": user_input,
		"agent_scratchpad": "",
		"final_answer": "",
		"steps": [],
		"pending_tool_calls": [],
		"iteration_count": 0,
		"last_model_output": "",
	}

	result = await graph.ainvoke(initial_state)
	result["loaded_tools"] = [tool.name for tool in tools]
	return result


async def main() -> None:
	args = parse_args()
	result = await run_agent(args.query)

	print("Final Answer:\n")
	print(result.get("final_answer", ""))

	if os.getenv("SHOW_TRACE", "0") == "1":
		print("\nScratchpad:\n")
		print(result.get("agent_scratchpad", ""))


if __name__ == "__main__":
	asyncio.run(main())
