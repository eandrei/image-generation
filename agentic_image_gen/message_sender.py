from __future__ import annotations

import asyncio
from typing import List, cast

from openai.types.beta.threads import message_create_params

from .assistant_manager import get_client


async def send_message(thread_id: str, prompt: str, images: List[str]) -> str:
    """Send a user message to a thread using the OpenAI client.

    Args:
        thread_id: The ID of the thread to post the message to.
        prompt: The text content of the message.
        images: A list of image file IDs to attach to the message.

    Returns:
        The ID of the created message.
    """
    client = get_client()
    attachments = [{"file_id": file_id, "tools": [{"type": "file_search"}]} for file_id in images]
    typed_attachments = cast("list[message_create_params.Attachment]", attachments)
    message = await asyncio.to_thread(
        client.beta.threads.messages.create,
        thread_id=thread_id,
        role="user",
        content=prompt,
        attachments=typed_attachments,
    )
    return message.id
