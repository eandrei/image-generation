from __future__ import annotations

import os
from typing import Final, cast

from openai import AsyncOpenAI

MODEL_NAME: Final = "gpt-4o"


def _get_client() -> AsyncOpenAI:
    """Return an AsyncOpenAI client using the environment API key."""
    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        raise ValueError("OPENAI_API_KEY environment variable is required")
    return AsyncOpenAI(api_key=api_key)


async def generate_prompt(previous_prompt: str, feedback: str) -> str:
    """Generate a refined prompt based on previous prompt and evaluator feedback.

    Args:
        previous_prompt: The prompt from the previous iteration.
        feedback: Feedback from the evaluator to apply.

    Returns:
        The refined prompt text.
    """
    client = _get_client()
    response = await client.chat.completions.create(
        model=MODEL_NAME,
        messages=[
            {
                "role": "system",
                "content": (
                    "Refine the following image generation prompt using the provided "
                    "feedback. Return only the improved prompt."
                ),
            },
            {
                "role": "user",
                "content": f"Prompt: {previous_prompt}\nFeedback: {feedback}",
            },
        ],
        temperature=0.2,
    )
    content = cast(str, response.choices[0].message.content)
    return content.strip()
