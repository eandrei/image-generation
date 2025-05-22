from __future__ import annotations

import os

from openai import AsyncOpenAI


async def generate_image(prompt: str) -> str:
    """Generate an image using the provided prompt.

    Args:
        prompt: The textual description of the desired image.

    Returns:
        A URL or file path pointing to the generated image.
    """
    client = AsyncOpenAI(api_key=os.getenv("OPENAI_API_KEY"))
    response = await client.images.generate(model="dall-e-3", prompt=prompt, n=1, size="1024x1024")
    data = response.data or []
    if data:
        return data[0].url or ""
    return ""
