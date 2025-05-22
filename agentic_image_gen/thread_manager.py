from __future__ import annotations

import asyncio
import json
from pathlib import Path
from typing import Optional

from openai import OpenAI

from .assistant_manager import get_client

CONFIG_PATH = Path("thread_config.json")


async def create_thread() -> str:
    """Create a conversation thread and store its ID.

    Returns:
        str: The created thread ID.
    """
    client: OpenAI = get_client()
    thread = await asyncio.to_thread(client.beta.threads.create)
    CONFIG_PATH.write_text(json.dumps({"thread_id": thread.id}))
    return thread.id


def load_thread_id() -> Optional[str]:
    """Load the thread ID from config if available."""
    if CONFIG_PATH.exists():
        data = json.loads(CONFIG_PATH.read_text())
        return data.get("thread_id")
    return None
