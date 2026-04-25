"""
GenAIBridge: Drop-in replacement for ChatGoogleGenerativeAI in LangGraph nodes.

Provides Google GenAI native SDK access with context caching, while returning
LangChain-compatible AIMessage objects that LangGraph's tools_condition understands.

Usage:
    from langgraph_genai_bridge import GenAIBridge

    bridge = GenAIBridge(api_key="...", model="gemini-2.5-flash")
    bridge.set_tools(my_langchain_tools)

    # Inside a LangGraph node:
    def orchestrator_node(state):
        messages = state["messages"]
        response = bridge.invoke(messages, system_prompt="You are a helpful agent.")
        return {"messages": [response]}

    # With caching (saves ~80% on input tokens):
    bridge.enable_caching(ttl_seconds=3600)
"""

import logging
from typing import List, Optional

from langgraph_genai_bridge.cache import ContextCacheManager
from langgraph_genai_bridge.messages import langchain_to_genai, genai_to_langchain
from langgraph_genai_bridge.tools import convert_langchain_tools

logger = logging.getLogger("langgraph-genai-bridge")


class GenAIBridge:
    """
    Bridge between LangGraph (LangChain messages) and Google GenAI native SDK.

    Enables:
    - Context caching for system prompts (5x cost reduction)
    - Native tool calling (function declarations)
    - Bidirectional message translation
    - Graceful fallback to LangChain wrapper on failure
    """

    def __init__(
        self,
        api_key: str,
        model: str = "gemini-2.5-flash",
        temperature: float = 0.1,
        max_output_tokens: int = 8192,
        flex: bool = False,
        service_tier: Optional[str] = None,
    ):
        """
        Args:
            api_key: Google API key
            model: Gemini model name (default: gemini-2.5-flash)
            temperature: Sampling temperature (default: 0.1 for deterministic)
            max_output_tokens: Maximum output length
            flex: Boolean toggle for Google's "flex" pricing tier — opts in to
                the ~50% latency-tolerant discount on supported models. The
                friendly form of `service_tier="flex"`. Default False = standard
                tier. Mutually exclusive with `service_tier`; pass one or the
                other, not both.
            service_tier: Advanced override — pass an explicit tier string
                directly to GenerateContentConfig.service_tier. Use this if
                Google adds new tiers beyond "flex" that you want before this
                library ships a friendly boolean for them. None = no tier.
        """
        if flex and service_tier:
            raise ValueError(
                "GenAIBridge: pass `flex=True` OR `service_tier='...'`, not both"
            )
        resolved_tier = service_tier if service_tier else ("flex" if flex else None)

        from google import genai

        self.client = genai.Client(api_key=api_key)
        self.model = model
        self.temperature = temperature
        self.max_output_tokens = max_output_tokens
        self.service_tier = resolved_tier

        self._tool_declarations = None
        self._cache_manager: Optional[ContextCacheManager] = None
        self._langchain_fallback = None

        tier_tag = f", tier={resolved_tier}" if resolved_tier else ""
        logger.info(f"GenAIBridge initialized: model={model}, temp={temperature}{tier_tag}")

    def set_tools(self, langchain_tools: list):
        """
        Register LangChain tools for function calling.
        Automatically converts @tool definitions to GenAI FunctionDeclarations.

        Args:
            langchain_tools: List of LangChain @tool decorated functions
        """
        self._tool_declarations = convert_langchain_tools(langchain_tools)
        logger.info(f"Registered {len(self._tool_declarations)} tools.")

    def enable_caching(self, ttl_seconds: int = 3600):
        """
        Enable context caching for system prompts.
        The system prompt will be cached for ttl_seconds, reducing input
        token costs by ~80% for repeated calls with the same prompt.

        Args:
            ttl_seconds: Cache time-to-live (default: 1 hour)
        """
        self._cache_manager = ContextCacheManager(
            client=self.client,
            model=self.model,
            ttl_seconds=ttl_seconds,
        )
        logger.info(f"Context caching enabled (TTL: {ttl_seconds}s)")

    def set_langchain_fallback(self, langchain_llm):
        """
        Set a LangChain ChatModel as fallback when native SDK fails.

        Args:
            langchain_llm: A ChatGoogleGenerativeAI instance with tools bound
        """
        self._langchain_fallback = langchain_llm

    def invoke(
        self,
        messages: list,
        system_prompt: Optional[str] = None,
        max_tool_output: int = 3000,
    ) -> "AIMessage":
        """
        Invoke Gemini via native SDK with context caching support.
        Returns a LangChain AIMessage compatible with LangGraph's tools_condition.

        Args:
            messages: List of LangChain messages (from LangGraph state)
            system_prompt: System prompt text (will be cached if caching enabled)
            max_tool_output: Maximum chars per tool response in context

        Returns:
            LangChain AIMessage with content and tool_calls
        """
        from google.genai import types as genai_types
        from langchain_core.messages import SystemMessage

        try:
            # Extract system prompt from first message if not provided
            if not system_prompt and messages and isinstance(messages[0], SystemMessage):
                system_prompt = messages[0].content

            # Convert messages to GenAI format
            contents = langchain_to_genai(messages, skip_system=True)

            if not contents:
                from langchain_core.messages import AIMessage
                return AIMessage(content="No input messages to process.")

            # Truncate tool responses in context
            for content in contents:
                for part in content.parts:
                    if hasattr(part, 'function_response') and part.function_response:
                        result = part.function_response.response.get("result", "")
                        if len(result) > max_tool_output:
                            part.function_response.response["result"] = (
                                result[:max_tool_output] + f"\n[TRUNCATED]"
                            )

            # Build config
            config_kwargs = {
                "temperature": self.temperature,
                "max_output_tokens": self.max_output_tokens,
            }
            if self.service_tier:
                config_kwargs["service_tier"] = self.service_tier
            config = genai_types.GenerateContentConfig(**config_kwargs)

            if self._tool_declarations:
                config.tools = [genai_types.Tool(
                    function_declarations=self._tool_declarations
                )]

            # Try context cache
            model_to_use = self.model
            if self._cache_manager and system_prompt:
                cache_name = self._cache_manager.get_or_create(system_prompt)
                if cache_name:
                    model_to_use = cache_name
                else:
                    config.system_instruction = system_prompt
            elif system_prompt:
                config.system_instruction = system_prompt

            # Call Gemini native
            response = self.client.models.generate_content(
                model=model_to_use,
                contents=contents,
                config=config,
            )

            # Convert to LangChain AIMessage
            return genai_to_langchain(response)

        except Exception as e:
            logger.warning(f"Native SDK failed: {e}")

            # Fallback to LangChain wrapper
            if self._langchain_fallback:
                logger.info("Falling back to LangChain wrapper.")
                return self._langchain_fallback.invoke(messages)

            # No fallback available — return error message
            from langchain_core.messages import AIMessage
            return AIMessage(content=f"GenAI Bridge error: {e}")

    def invalidate_cache(self):
        """Force cache invalidation (use when system prompt changes)."""
        if self._cache_manager:
            self._cache_manager.invalidate()
