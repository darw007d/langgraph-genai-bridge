# langgraph-genai-bridge

**Native Google GenAI SDK integration for LangGraph** with context caching, tool auto-conversion, and bidirectional message translation.

> By **Pierre Samson** ([@darw007d](https://github.com/darw007d)) and **Claude Opus** (Anthropic)

## Why?

LangGraph is the best orchestration framework for AI agents. Google's native GenAI SDK has features (context caching, native structured output) that LangChain's wrapper doesn't expose. This bridge gives you both.

| Feature | LangChain Wrapper | This Bridge |
|---------|------------------|-------------|
| Context Caching | Not supported | Built-in (5x cost reduction) |
| Structured Output | Via wrapper (buggy) | Native `response_schema` |
| Tool Calling | Wrapped | Native FunctionDeclaration |
| Latency | Higher (abstraction layer) | Lower (direct SDK) |
| LangGraph Compatible | Yes | Yes |

## Install

```bash
pip install langgraph-genai-bridge
```

## Quick Start

```python
from langgraph_genai_bridge import GenAIBridge

# Initialize
bridge = GenAIBridge(api_key="your-google-api-key", model="gemini-2.5-flash")

# Register your LangChain tools
bridge.set_tools(my_langchain_tools)

# Enable context caching (saves ~80% on input tokens)
bridge.enable_caching(ttl_seconds=3600)

# Use inside a LangGraph node — returns LangChain AIMessage
def orchestrator_node(state):
    response = bridge.invoke(
        state["messages"],
        system_prompt="You are a helpful trading agent."
    )
    return {"messages": [response]}
```

## Features

### Context Caching

Google's context caching lets you pay for your system prompt **once per hour** instead of every API call. For an agent running 12 cycles/hour with a 2000-token system prompt, that's 24,000 tokens/hour saved.

```python
bridge.enable_caching(ttl_seconds=3600)  # Cache for 1 hour

# First call: creates cache (normal cost)
# Subsequent calls: uses cache (near-free input tokens)
response = bridge.invoke(messages, system_prompt=my_long_prompt)
```

### Tool Auto-Conversion

Automatically converts LangChain `@tool` decorated functions to Google GenAI `FunctionDeclaration` format. No manual schema writing needed.

```python
from langchain_core.tools import tool

@tool
def get_stock_price(ticker: str) -> str:
    """Get the current price for a stock ticker."""
    return f"{ticker}: $150.00"

bridge.set_tools([get_stock_price])  # Auto-converts
```

### Bidirectional Message Translation

Seamlessly converts between LangChain message types and Google GenAI Content objects:

| LangChain | Direction | Google GenAI |
|-----------|-----------|-------------|
| `SystemMessage` | -> | Context Cache / system_instruction |
| `HumanMessage` | -> | `Content(role="user")` |
| `AIMessage` (with tool_calls) | -> | `Content(role="model")` with `FunctionCall` |
| `ToolMessage` | -> | `Content` with `FunctionResponse` |
| | <- | `AIMessage(content=..., tool_calls=[...])` |

### Graceful Fallback

If the native SDK fails, automatically falls back to your LangChain wrapper:

```python
from langchain_google_genai import ChatGoogleGenerativeAI

# Set up fallback
langchain_llm = ChatGoogleGenerativeAI(model="gemini-2.5-flash")
langchain_with_tools = langchain_llm.bind_tools(my_tools)
bridge.set_langchain_fallback(langchain_with_tools)

# If native SDK fails -> seamlessly falls back to LangChain
response = bridge.invoke(messages)
```

## Full LangGraph Example

```python
from langgraph.graph import StateGraph, START, END
from langgraph.prebuilt import tools_condition
from langchain_core.tools import tool
from langgraph_genai_bridge import GenAIBridge

# Define tools
@tool
def search_web(query: str) -> str:
    """Search the web for information."""
    return f"Results for: {query}"

# Initialize bridge
bridge = GenAIBridge(api_key="...", model="gemini-2.5-flash")
bridge.set_tools([search_web])
bridge.enable_caching(ttl_seconds=3600)

# LangGraph nodes
def agent(state):
    return {"messages": [bridge.invoke(state["messages"], system_prompt="You are helpful.")]}

def tool_node(state):
    # Your existing tool execution logic
    ...

# Build graph (standard LangGraph pattern)
workflow = StateGraph(...)
workflow.add_node("agent", agent)
workflow.add_node("tools", tool_node)
workflow.add_edge(START, "agent")
workflow.add_conditional_edges("agent", tools_condition)
workflow.add_edge("tools", "agent")
app = workflow.compile()
```

## Cost Savings Benchmark

Measured on a trading agent with 35+ tools, 2000-token system prompt, 12 cycles/hour:

| Metric | LangChain Wrapper | GenAI Bridge |
|--------|------------------|-------------|
| Input tokens/hour | ~120,000 | ~25,000 |
| Cost/day (Gemini Flash) | ~5 EUR | ~1 EUR |
| Latency per call | ~800ms | ~500ms |

## API Reference

### `GenAIBridge(api_key, model, temperature, max_output_tokens)`

Main bridge class.

### `bridge.set_tools(langchain_tools)`

Register LangChain @tool functions for native function calling.

### `bridge.enable_caching(ttl_seconds=3600)`

Enable context caching for system prompts.

### `bridge.invoke(messages, system_prompt=None, max_tool_output=3000)`

Call Gemini and return a LangChain AIMessage. Compatible with `tools_condition`.

### `bridge.invoke_structured(messages, response_schema, system_prompt=None, return_raw=False)` *(v0.2+)*

Structured output via Gemini's native `response_schema` — **preserves caching and Flex tier**, unlike LangChain's `with_structured_output()` wrapper which forces a separate ChatGoogleGenerativeAI client and bypasses Google context caching.

```python
from pydantic import BaseModel
from langgraph_genai_bridge import GenAIBridge

class TradeVerdict(BaseModel):
    action: str          # "BUY" / "SELL" / "SKIP"
    confidence: float    # 0.0–1.0

bridge = GenAIBridge(api_key="...", model="gemini-2.5-pro")
bridge.enable_caching(ttl_seconds=3600)

# First call: caches the playbook (one-time cost)
# Subsequent calls: cache_read at ~10% of input rate AND parsed structured output
verdict = bridge.invoke_structured(
    messages=[HumanMessage("Score AAPL")],
    response_schema=TradeVerdict,
    system_prompt=long_playbook_text,
)
print(verdict.action, verdict.confidence)
```

Tools and `response_schema` are mutually exclusive in Gemini's API — when a schema is provided, registered tools are not attached. Use one or the other per call.

Pass `return_raw=True` to get `{"parsed": <instance>, "raw": <google_response>}` for usage metadata access.

### `bridge.set_langchain_fallback(langchain_llm)`

Set a LangChain ChatModel as fallback. Used by both `invoke()` (returns AIMessage) and `invoke_structured()` (calls `.with_structured_output(schema).invoke()`).

### `bridge.invalidate_cache()`

Force cache invalidation.

## Changelog

- **v0.2.0** — `invoke_structured()` for native `response_schema` output that preserves Google context caching (Pro tier). Closes the structured-output ↔ caching trade-off that v0.1 had to fall back to LangChain for.
- **v0.1.5** — Friendly `flex=True` boolean alongside `service_tier="flex"`.
- **v0.1.4** — `usage_metadata` propagated on returned AIMessage (token tracking).
- **v0.1.3** — `service_tier` parameter (Flex 50% rebate).
- **v0.1.2** — Packaging fix (v0.1.0/0.1.1 sdist was empty due to flat-layout misconfig).

## License

MIT License. Co-authored by [Pierre Samson](https://github.com/darw007d) and Claude Opus (Anthropic).

Sister to the Phase 19 PyPI library family — same "small, tested, publishable" ethos:
[phawkes](https://pypi.org/project/phawkes/) (Hawkes processes) ·
[fisherrao](https://pypi.org/project/fisherrao/) (information geometry) ·
[tailcor](https://pypi.org/project/tailcor/) (tail-contagion decomposition) ·
[diebold-yilmaz](https://pypi.org/project/diebold-yilmaz/) (spillover index) ·
[hodgex](https://pypi.org/project/hodgex/) (Hodge Laplacians).
