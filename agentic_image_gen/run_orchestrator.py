from __future__ import annotations

import os

from openai import AsyncOpenAI
from openai.lib.streaming import AsyncAssistantEventHandler


def get_async_client() -> AsyncOpenAI:
    """Create an asynchronous OpenAI client using the API key from the environment."""
    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        raise ValueError("OPENAI_API_KEY environment variable is required")
    return AsyncOpenAI(api_key=api_key)


async def run_and_stream(thread_id: str, assistant_id: str) -> list[dict]:
    """Run the assistant on the given thread and capture tool call information.

    Args:
        thread_id: The thread identifier.
        assistant_id: The assistant identifier.

    Returns:
        A list of dictionaries describing tool calls encountered during the run.
    """
    client = get_async_client()
    handler = AsyncAssistantEventHandler()

    async with client.beta.threads.runs.stream(
        thread_id=thread_id,
        assistant_id=assistant_id,
        event_handler=handler,
    ) as event_handler:
        await event_handler.until_done()

    steps = await handler.get_final_run_steps()
    tool_calls: list[dict] = []
    for step in steps:
        if step.step_details.type != "tool_calls":
            continue
        for call in step.step_details.tool_calls:
            info = {"id": call.id, "type": call.type}
            function = getattr(call, "function", None)
            if function is not None:
                info.update({"name": function.name, "arguments": function.arguments})
            tool_calls.append(info)

    return tool_calls
