"""
Tool Declaration Converter: LangChain @tool -> Google GenAI FunctionDeclaration.

Automatically converts LangChain tool definitions to the format required
by Google's native GenAI SDK for function calling.
"""

import logging
from typing import List

logger = logging.getLogger("langgraph-genai-bridge")

# JSON Schema type -> Google GenAI type mapping
_TYPE_MAP = {
    "string": "STRING",
    "number": "NUMBER",
    "integer": "INTEGER",
    "boolean": "BOOLEAN",
    "array": "ARRAY",
    "object": "OBJECT",
}


def convert_langchain_tools(tools: list) -> list:
    """
    Convert a list of LangChain @tool functions to Google GenAI FunctionDeclarations.

    Args:
        tools: List of LangChain tool objects (decorated with @tool)

    Returns:
        List of google.genai.types.FunctionDeclaration objects
    """
    from google.genai import types as genai_types

    declarations = []
    for t in tools:
        try:
            schema = t.args_schema.schema() if hasattr(t, 'args_schema') and t.args_schema else {}
            properties = {}
            required = []

            for param_name, param_info in schema.get("properties", {}).items():
                param_type = param_info.get("type", "string")
                genai_type = _TYPE_MAP.get(param_type, "STRING")

                properties[param_name] = genai_types.Schema(
                    type=genai_type,
                    description=param_info.get("description", ""),
                )

                if param_name in schema.get("required", []):
                    required.append(param_name)

            declaration = genai_types.FunctionDeclaration(
                name=t.name,
                description=(t.description or "")[:500],
                parameters=genai_types.Schema(
                    type="OBJECT",
                    properties=properties,
                    required=required if required else None,
                ) if properties else None,
            )
            declarations.append(declaration)

        except Exception as e:
            logger.warning(f"Failed to convert tool '{getattr(t, 'name', '?')}': {e}")

    logger.info(f"Converted {len(declarations)}/{len(tools)} LangChain tools to GenAI declarations.")
    return declarations
