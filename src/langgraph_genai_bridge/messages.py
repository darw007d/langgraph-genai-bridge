"""
Bidirectional Message Translation: LangChain <-> Google GenAI.

Converts between LangChain message types (HumanMessage, AIMessage, ToolMessage,
SystemMessage) and Google GenAI Content objects. This is the core translation
layer that allows LangGraph's state machine to work with Google's native SDK.
"""

import time
import logging
from typing import List, Optional

logger = logging.getLogger("langgraph-genai-bridge")


def langchain_to_genai(messages: list, skip_system: bool = True) -> list:
    """
    Convert LangChain messages to Google GenAI Content objects.

    Args:
        messages: List of LangChain message objects
        skip_system: If True, skip SystemMessage (assumed to be in context cache)

    Returns:
        List of google.genai.types.Content objects
    """
    from google.genai import types as genai_types
    from langchain_core.messages import SystemMessage, HumanMessage

    contents = []

    for msg in messages:
        try:
            if isinstance(msg, SystemMessage):
                if skip_system:
                    continue
                contents.append(genai_types.Content(
                    role="user",
                    parts=[genai_types.Part(text=f"[System] {msg.content}")]
                ))

            elif isinstance(msg, HumanMessage):
                contents.append(genai_types.Content(
                    role="user",
                    parts=[genai_types.Part(text=msg.content)]
                ))

            elif hasattr(msg, "tool_calls") and msg.tool_calls:
                # AIMessage with tool calls
                parts = []
                if msg.content:
                    text = msg.content if isinstance(msg.content, str) else str(msg.content)
                    if text:
                        parts.append(genai_types.Part(text=text))
                for tc in msg.tool_calls:
                    parts.append(genai_types.Part(
                        function_call=genai_types.FunctionCall(
                            name=tc["name"],
                            args=tc["args"]
                        )
                    ))
                contents.append(genai_types.Content(role="model", parts=parts))

            elif hasattr(msg, "tool_call_id"):
                # ToolMessage -> function response
                tool_name = msg.name if hasattr(msg, "name") and msg.name else "tool"
                content = msg.content if hasattr(msg, "content") else ""
                contents.append(genai_types.Content(
                    role="user",
                    parts=[genai_types.Part(
                        function_response=genai_types.FunctionResponse(
                            name=tool_name,
                            response={"result": content},
                        )
                    )]
                ))

            elif hasattr(msg, "content"):
                # Generic message — determine role from type
                role = "model" if "AI" in str(type(msg)) else "user"
                content = msg.content if isinstance(msg.content, str) else str(msg.content)
                if content:
                    contents.append(genai_types.Content(
                        role=role,
                        parts=[genai_types.Part(text=content)]
                    ))

        except Exception as e:
            logger.debug(f"Message conversion skipped: {type(msg).__name__}: {e}")

    return contents


def _extract_usage_metadata(response) -> dict | None:
    """Pull the GenAI response's usage_metadata into LangChain's standard shape.

    LangChain v0.3+ AIMessage carries usage_metadata as
    `{"input_tokens": int, "output_tokens": int, "total_tokens": int,
      "input_token_details": {"cache_read": int}}`.

    Google's GenerateContentResponse exposes usage_metadata with fields
    like `prompt_token_count`, `candidates_token_count`,
    `cached_content_token_count`, `total_token_count`, `thoughts_token_count`.
    Mapping below preserves all useful counts including thinking tokens
    (separate field, charged at output rate but useful to track).
    """
    um = getattr(response, "usage_metadata", None)
    if um is None:
        return None
    prompt = getattr(um, "prompt_token_count", 0) or 0
    candidates = getattr(um, "candidates_token_count", 0) or 0
    cached = getattr(um, "cached_content_token_count", 0) or 0
    thoughts = getattr(um, "thoughts_token_count", 0) or 0
    total = getattr(um, "total_token_count", None)
    if total is None:
        total = prompt + candidates + thoughts
    out: dict = {
        "input_tokens": int(prompt),
        "output_tokens": int(candidates + thoughts),
        "total_tokens": int(total),
    }
    if cached:
        out["input_token_details"] = {"cache_read": int(cached)}
    if thoughts:
        out.setdefault("output_token_details", {})["reasoning"] = int(thoughts)
    return out


def genai_to_langchain(response) -> "AIMessage":
    """
    Convert a Google GenAI response to a LangChain AIMessage.

    Args:
        response: google.genai GenerateContentResponse

    Returns:
        LangChain AIMessage with content, tool_calls (if any), and
        usage_metadata in LangChain v0.3+ standard shape (so downstream
        token-tracking interceptors keep working transparently).
    """
    from langchain_core.messages import AIMessage

    if not response.candidates:
        return AIMessage(content="No response from model.")

    text_parts = []
    tool_calls = []

    for part in response.candidates[0].content.parts:
        if part.text:
            text_parts.append(part.text)
        elif part.function_call:
            fc = part.function_call
            tool_calls.append({
                "name": fc.name,
                "args": dict(fc.args) if fc.args else {},
                "id": f"call_{fc.name}_{int(time.time() * 1000)}",
            })

    usage = _extract_usage_metadata(response)
    msg_kwargs: dict = {
        "content": "".join(text_parts) if text_parts else "",
    }
    if tool_calls:
        msg_kwargs["tool_calls"] = tool_calls
    if usage is not None:
        msg_kwargs["usage_metadata"] = usage
    return AIMessage(**msg_kwargs)
