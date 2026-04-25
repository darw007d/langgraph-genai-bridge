"""
langgraph-genai-bridge: Native Google GenAI SDK integration for LangGraph.

Provides context caching, bidirectional message translation, and tool
declaration auto-conversion — enabling LangGraph agents to use Google's
native SDK features (context caching, structured output) while keeping
LangGraph's orchestration capabilities.

Usage:
    from langgraph_genai_bridge import GenAIBridge

    bridge = GenAIBridge(api_key="...", model="gemini-2.5-flash")
    bridge.set_tools(langchain_tools)

    # Inside a LangGraph node:
    def my_node(state):
        response = bridge.invoke(state["messages"], system_prompt="...")
        return {"messages": [response]}
"""

from langgraph_genai_bridge.bridge import GenAIBridge
from langgraph_genai_bridge.cache import ContextCacheManager
from langgraph_genai_bridge.tools import convert_langchain_tools

__version__ = "0.1.5"
__author__ = "Pierre Samson, Claude Opus (Anthropic)"
__all__ = ["GenAIBridge", "ContextCacheManager", "convert_langchain_tools"]
