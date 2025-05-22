from __future__ import annotations

import os
from typing import Final, List, cast

from openai import AsyncOpenAI

MODEL_NAME: Final = "dall-e-3"


def _get_client() -> AsyncOpenAI:
    """Return an AsyncOpenAI client using the environment API key."""
    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        raise ValueError("OPENAI_API_KEY environment variable is required")
    return AsyncOpenAI(api_key=api_key)


async def generate_image(prompt: str) -> str:
    """Generate an image URL from the provided prompt.

    Args:
        prompt: Prompt describing the desired image.

    Returns:
        URL of the generated image.
    """
    client = _get_client()
    response = await client.images.generate(
        model=MODEL_NAME,
        prompt=prompt,
        n=1,
        size="1024x1024",
    )
    data = cast(List, response.data)
    return cast(str, data[0].url)
