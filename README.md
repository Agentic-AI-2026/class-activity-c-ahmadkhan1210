[![Review Assignment Due Date](https://classroom.github.com/assets/deadline-readme-button-22041afd0340ce965d47ae6ef1cefeee28c7c493a6346c4f15d667ab976d596c.svg)](https://classroom.github.com/a/LWzSlHLS)

# Quiz: Convert ReAct Agent to LangGraph 🦜🕸️

## Objective

Convert a standard working **ReAct agent** (implemented in LangChain) into a **LangGraph workflow**. Your implementation must preserve the iterative reasoning and tool-usage behavior inherent to the ReAct framework.

---

## 🛠 Provided Resources

- **Existing ReAct agent code** (LangChain-based).
- **Tool implementations** (functional and ready for use).

---

## 📋 Requirements

### 1. Define State

Create a state structure (TypedDict or Pydantic) to represent the workflow. Your state must include:

- `input`: The original user query.
- `agent_scratchpad`: Stores intermediate reasoning (Thoughts, Actions, Observations).
- `final_answer`: The final response delivered to the user.
- `steps`: (Optional) A list to track history of actions and observations.

### 2. ReAct Node (Reasoning + Action)

Implement a node that:

1. Takes the current state.
2. Calls the LLM using **ReAct-style prompting**.
3. Produces either an **Action** (tool name + arguments) or a **Final Answer**.
4. Updates the state accordingly.

### 3. Tool Execution Node

Implement a node that:

1. Executes the tool selected by the ReAct node.
2. Passes the correct arguments to the tool.
3. Stores the **Observation** (result) back in the state.
4. Updates the scratchpad to prepare for the next reasoning step.

### 4. Graph Flow

Construct a LangGraph workflow that follows this logic:

> **START** -> `react_node` -> **Conditional Edge**
>
> - If **Action** -> `tool_node` -> `react_node`
> - If **Final Answer** -> **END**

**The graph must:**

- Support iterative reasoning loops.
- Continue execution until a terminal state (Final Answer) is reached.

### 5. Conditional Routing

Implement the router logic to determine the next step based on the model's output:

- `is_action` -> Route to `tool_node`.
- `is_final` -> Route to `END`.

---

## 🧪 Test Case

Your implementation should successfully process complex, multi-step queries such as:

> _"What is the weather in Lahore and who is the current Prime Minister of Pakistan? Now get the age of PM and tell us will this weather suits PM health."_

---

## ⚠️ Constraints

- **No Hardcoding:** Do not hardcode outputs; the logic must be dynamic.
- **Reasoning Integrity:** Maintain the "Thought -> Action -> Observation" flow.
- **Scalability:** The agent must be capable of calling tools multiple times in a single run.
- **State Management:** Ensure proper state updates to prevent infinite loops or data loss between iterations.

---

## 🚀 Submission

Push your solution to this repository using the following structure:

```text
.
├── main.py          # Entry point for execution
├── graph.py         # LangGraph definition (optional)
└── README.md        # Project documentation
```

## Setup

### Prerequisites

- Python 3.10 or newer.
- A running Ollama instance if you use `LLM_PROVIDER=ollama`.
- A valid `TAVILY_API_KEY` for the search MCP server.
- Optional: `ANTHROPIC_API_KEY`, `GOOGLE_API_KEY`, or `GROQ_API_KEY` if you switch providers.

### Environment Variables

Create a `.env` file in the project root using `.env.example` as a template.

Useful variables:

- `LLM_PROVIDER` - `ollama`, `anthropic`, `google`, or `groq`.
- `OLLAMA_MODEL` - model name when using Ollama.
- `ANTHROPIC_API_KEY` - only needed for Anthropic.
- `GOOGLE_API_KEY` - only needed for Google Gemini.
- `GROQ_API_KEY` - only needed for Groq.
- `GROQ_MODEL` - Groq model name, for example `llama-3.3-70b-versatile`.
- `TAVILY_API_KEY` - required by `Tools/search_server.py`.
- `WEATHER_MCP_URL` - HTTP URL for the weather MCP server.
- `MAX_STEPS` - safety limit for the ReAct loop.
- `TEST_QUERY` - optional default query for `main.py`.

### Install

Install the project dependencies before running the agent:

```bash
pip install langgraph langchain langchain-core langchain-mcp-adapters langchain-ollama langchain-anthropic langchain-google-genai langchain-groq mcp tavily-python requests
```

### Run

1. Start the weather MCP server in a separate terminal:
   `python Tools/weather_server.py`
2. Make sure `TAVILY_API_KEY` is configured for `Tools/search_server.py`.
3. Run the agent:
   `python main.py`
4. Optionally pass a custom query:
   `python main.py --query "your question here"`

### Graph Flow

The implemented workflow follows the assignment requirement:

`START -> react_node -> conditional router -> tool_node -> react_node -> ... -> END`

The state tracks the original input, the running scratchpad, the final answer, and a step history so iterative reasoning stays visible across tool calls.

### Notes

- `Class code/MCP_code.py` now mirrors the runtime helper and is kept as a compatibility example.
- `Tools/search_server.py` reads its Tavily key from the environment instead of hardcoding an empty string.
- The repo includes a `.gitignore` so local environments, caches, and `.env` files stay untracked.
