from __future__ import annotations

import asyncio
import base64
import json
import os
from pathlib import Path

from openai import AsyncOpenAI

SYSTEM_PROMPT = (
    'You evaluate generated images and respond with JSON: {"score": <0-100>, "feedback": <string>}'
)


async def evaluate_image(image_path: str, prompt: str) -> dict:
    """Evaluate an image against a prompt using OpenAI's vision model.

    Args:
        image_path: Path or URL to the image to evaluate.
        prompt: The prompt used to generate the image.

    Returns:
        Dict containing `score` and `feedback` keys.
    """
    client = AsyncOpenAI(api_key=os.getenv("OPENAI_API_KEY"))

    if Path(image_path).exists():
        file_bytes = await asyncio.to_thread(Path(image_path).read_bytes)
        b64_data = base64.b64encode(file_bytes).decode()
        image_url = f"data:image/png;base64,{b64_data}"
    else:
        image_url = image_path

    response = await client.chat.completions.create(
        model="gpt-4o",
        messages=[
            {"role": "system", "content": SYSTEM_PROMPT},
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": prompt},
                    {"type": "image_url", "image_url": {"url": image_url}},
                ],
            },
        ],
        temperature=0,
    )
    content = response.choices[0].message.content or ""
    return json.loads(content.strip())
