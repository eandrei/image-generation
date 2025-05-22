from __future__ import annotations

import uuid
from typing import Dict, List

_THREADS: Dict[str, List[str]] = {}


async def create_thread(reference_images: List[str] | None = None) -> str:
    """Create a new thread and store reference images."""
    thread_id = uuid.uuid4().hex
    _THREADS[thread_id] = reference_images or []
    return thread_id


async def add_message(thread_id: str, message: str) -> None:
    """Append a message to the thread history."""
    if thread_id not in _THREADS:
        raise ValueError("Invalid thread_id")
    _THREADS[thread_id].append(message)


async def get_history(thread_id: str) -> List[str]:
    """Retrieve the message history for a thread."""
    if thread_id not in _THREADS:
        raise ValueError("Invalid thread_id")
    return _THREADS[thread_id]
