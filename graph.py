from __future__ import annotations

import json
import re
from typing import Any, TypedDict

from langchain_core.messages import HumanMessage, SystemMessage
from langgraph.graph import END, START, StateGraph


REACT_SYSTEM = """You are a ReAct agent.
Strictly follow this loop:
Thought -> Action (tool call) -> Observation -> Thought -> ...

Rules:
1. Always use tools for factual information.
2. For multi-part questions, call tools one fact at a time.
3. Always use the calculator or math tools for arithmetic.
4. Only provide a final answer after all required tool calls are complete.
5. If you need a fact, do not guess.
6. When you decide to use a tool, use the native tool-calling interface only.
7. Never output manual function markup such as <function=...>, XML tags, or JSON-like tool snippets in plain text.
"""


class AgentState(TypedDict, total=False):
	input: str
	agent_scratchpad: str
	final_answer: str
	steps: list[dict[str, Any]]
	pending_tool_calls: list[dict[str, Any]]
	iteration_count: int
	last_model_output: str
	text_action_mode: bool


def build_react_graph(llm: Any, tools_map: dict[str, Any], max_steps: int = 8):
	"""Build a compiled LangGraph ReAct workflow."""

	tool_names = ", ".join(sorted(tools_map))
	bound_llm = llm.bind_tools(list(tools_map.values()))

	def parse_text_tool_calls(text: str) -> list[dict[str, Any]]:
		"""Parse text-style action outputs into tool calls as a fallback.

		Supports formats like:
		- <function=search_web{"query":"..."}></function>
		- Action: search_web({"query":"..."})
		"""
		parsed: list[dict[str, Any]] = []
		raw = (text or "").strip()
		if not raw:
			return parsed

		fn_matches = re.findall(r"<function=([a-zA-Z0-9_]+)(\{.*?\})></function>", raw, flags=re.DOTALL)
		for name, args_blob in fn_matches:
			if name not in tools_map:
				continue
			try:
				args = json.loads(args_blob)
			except Exception:  # noqa: BLE001
				args = {}
			parsed.append({"id": f"parsed-{name}", "name": name, "args": args})

		if parsed:
			return parsed

		action_matches = re.findall(r"Action:\s*([a-zA-Z0-9_]+)\((\{.*?\})\)", raw, flags=re.DOTALL)
		for name, args_blob in action_matches:
			if name not in tools_map:
				continue
			try:
				args = json.loads(args_blob)
			except Exception:  # noqa: BLE001
				args = {}
			parsed.append({"id": f"parsed-{name}", "name": name, "args": args})

		return parsed

	async def react_node(state: AgentState) -> dict[str, Any]:
		iteration_count = int(state.get("iteration_count", 0)) + 1
		scratchpad = state.get("agent_scratchpad", "").strip()
		steps = list(state.get("steps", []))
		text_action_mode = bool(state.get("text_action_mode", False))

		if iteration_count > max_steps:
			final_answer = "Max steps reached before the agent converged."
			steps.append({"kind": "final", "content": final_answer, "reason": "step_limit"})
			next_scratchpad = scratchpad
			if next_scratchpad:
				next_scratchpad += "\n"
			next_scratchpad += f"Final Answer: {final_answer}"
			return {
				"iteration_count": iteration_count,
				"agent_scratchpad": next_scratchpad,
				"final_answer": final_answer,
				"pending_tool_calls": [],
				"last_model_output": final_answer,
				"steps": steps,
			}

		prompt = [
			SystemMessage(content=f"{REACT_SYSTEM}\n\nAvailable tools: {tool_names}"),
			HumanMessage(
				content=(
					f"Question: {state['input']}\n\n"
					f"Scratchpad:\n{scratchpad if scratchpad else 'None'}\n\n"
					"Continue the ReAct loop."
				)
			),
		]

		response = None
		if text_action_mode:
			response = await llm.ainvoke(prompt)
		else:
			try:
				response = await bound_llm.ainvoke(prompt)
			except Exception as exc:  # noqa: BLE001
				error_text = str(exc)
				if "tool_use_failed" in error_text or "Failed to call a function" in error_text:
					# Some Groq model responses fail native tool-calling validation.
					# Fall back to text action mode and parse actions ourselves.
					text_action_mode = True
					response = await llm.ainvoke(prompt)
				else:
					raise

		thought = (response.content or "").strip()

		structured_tool_calls = list(getattr(response, "tool_calls", None) or [])
		pending_from_text = parse_text_tool_calls(thought)
		tool_calls = structured_tool_calls or pending_from_text

		if tool_calls:
			pending_tool_calls: list[dict[str, Any]] = []
			action_lines: list[str] = []

			for tool_call in tool_calls:
				args = tool_call.get("args") or {}
				pending_tool_calls.append(
					{
						"id": tool_call.get("id"),
						"name": tool_call["name"],
						"args": args,
					}
				)
				action_lines.append(
					f"Action: {tool_call['name']}({json.dumps(args, ensure_ascii=False, sort_keys=True)})"
				)

			next_scratchpad_parts = [part for part in [scratchpad, f"Thought: {thought}" if thought else ""] if part]
			next_scratchpad_parts.extend(action_lines)
			steps.append(
				{
					"kind": "action",
					"iteration": iteration_count,
					"content": thought,
					"tool_calls": pending_tool_calls,
				}
			)

			return {
				"iteration_count": iteration_count,
				"agent_scratchpad": "\n".join(next_scratchpad_parts),
				"pending_tool_calls": pending_tool_calls,
				"final_answer": "",
				"last_model_output": thought,
				"steps": steps,
				"text_action_mode": text_action_mode,
			}

		final_answer = thought or "The model returned no tool calls and no final answer."
		next_scratchpad_parts = [part for part in [scratchpad, f"Final Answer: {final_answer}"] if part]
		steps.append({"kind": "final", "iteration": iteration_count, "content": final_answer})

		return {
			"iteration_count": iteration_count,
			"agent_scratchpad": "\n".join(next_scratchpad_parts),
			"final_answer": final_answer,
			"pending_tool_calls": [],
			"last_model_output": thought,
			"steps": steps,
			"text_action_mode": text_action_mode,
		}

	async def tool_node(state: AgentState) -> dict[str, Any]:
		scratchpad = state.get("agent_scratchpad", "").strip()
		steps = list(state.get("steps", []))
		observations: list[str] = []

		for tool_call in state.get("pending_tool_calls", []):
			tool_name = tool_call["name"]
			tool_args = tool_call.get("args") or {}
			tool = tools_map.get(tool_name)

			if tool is None:
				result = f"Error: tool '{tool_name}' is not registered."
			else:
				try:
					result = await tool.ainvoke(tool_args)
				except Exception as exc:  # noqa: BLE001
					result = f"Tool error from {tool_name}: {exc}"

			observation = str(result)
			observations.append(f"Observation: {observation}")
			steps.append(
				{
					"kind": "observation",
					"tool": tool_name,
					"args": tool_args,
					"content": observation,
				}
			)

		next_scratchpad = scratchpad
		if observations:
			if next_scratchpad:
				next_scratchpad += "\n"
			next_scratchpad += "\n".join(observations)

		return {
			"agent_scratchpad": next_scratchpad,
			"pending_tool_calls": [],
			"steps": steps,
		}

	def router(state: AgentState) -> str:
		if state.get("final_answer"):
			return "__end__"
		if state.get("pending_tool_calls"):
			return "tools"
		return "__end__"

	workflow = StateGraph(AgentState)
	workflow.add_node("react", react_node)
	workflow.add_node("tools", tool_node)
	workflow.add_edge(START, "react")
	workflow.add_conditional_edges(
		"react",
		router,
		{
			"tools": "tools",
			"__end__": END,
		},
	)
	workflow.add_edge("tools", "react")
	return workflow.compile()
