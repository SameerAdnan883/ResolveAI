import re
import json
import logging
from typing import Dict, Any

logger = logging.getLogger("ResolveAI.Agents")


def extract_json_from_llm(text: str) -> Dict[str, Any]:
    """Safely extracts JSON payload from LLM responses even if wrapped in markdown codeblocks."""
    text = text.strip()
    match = re.search(r"```(?:json)?\s*([\s\S]*?)\s*```", text)
    if match:
        text = match.group(1).strip()
    try:
        return json.loads(text)
    except Exception as e:
        logger.warning(f"Failed to parse json directly: {e}. Attempting regex extraction.")
        json_obj = re.search(r"(\{.*\})", text, re.DOTALL)
        if json_obj:
            return json.loads(json_obj.group(1))
        raise ValueError(f"Could not parse valid JSON from response: {text}")
