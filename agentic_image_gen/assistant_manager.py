from __future__ import annotations

import asyncio
import json
import os
from pathlib import Path
from typing import Optional

from openai import OpenAI

CONFIG_PATH = Path("assistant_config.json")
ASSISTANT_NAME = "Image-Gen Loop Prompter"
ASSISTANT_INSTRUCTIONS = (
    "You are the prompter half of a two-agent image-generation loop.  "
    "At each turn you refine the prompt so the generator can better satisfy "
    "the evaluator's feedback.  Preserve subject and style constraints."
)
MODEL_NAME = "gpt-4o"


def get_client() -> OpenAI:
    """Create an OpenAI client using the API key from the environment."""
    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        raise ValueError("OPENAI_API_KEY environment variable is required")
    return OpenAI(api_key=api_key)


async def create_assistant() -> str:
    """Create the assistant and persist its ID.

    Returns:
        str: The created assistant ID.
    """
    client = get_client()
    assistant = await asyncio.to_thread(
        client.beta.assistants.create,
        name=ASSISTANT_NAME,
        instructions=ASSISTANT_INSTRUCTIONS,
        model=MODEL_NAME,
        tools=[],
    )
    CONFIG_PATH.write_text(json.dumps({"assistant_id": assistant.id}))
    return assistant.id


def load_assistant_id() -> Optional[str]:
    """Load the assistant ID from the config file if present."""
    if CONFIG_PATH.exists():
        data = json.loads(CONFIG_PATH.read_text())
        return data.get("assistant_id")
    return None
