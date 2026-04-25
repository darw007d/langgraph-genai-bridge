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


def genai_to_langchain(response) -> "AIMessage":
    """
    Convert a Google GenAI response to a LangChain AIMessage.

    Args:
        response: google.genai GenerateContentResponse

    Returns:
        LangChain AIMessage with content and tool_calls (if any)
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

    return AIMessage(
        content="".join(text_parts) if text_parts else "",
        tool_calls=tool_calls if tool_calls else None,
    )
