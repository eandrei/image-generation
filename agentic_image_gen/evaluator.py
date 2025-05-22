from __future__ import annotations

import base64
import json
import os
from pathlib import Path
from typing import Final, cast

from openai import AsyncOpenAI

MODEL_NAME: Final = "gpt-4o"


def _get_client() -> AsyncOpenAI:
    """Return an AsyncOpenAI client using the environment API key."""
    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        raise ValueError("OPENAI_API_KEY environment variable is required")
    return AsyncOpenAI(api_key=api_key)


async def evaluate_image(image_path: str, prompt: str) -> dict:
    """Evaluate an image against a prompt and return score and feedback.

    Args:
        image_path: Path to the image file to evaluate.
        prompt: Prompt used to generate the image.

    Returns:
        Dictionary containing ``score`` and ``feedback`` keys.
    """
    client = _get_client()
    image_bytes = Path(image_path).read_bytes()
    encoded = base64.b64encode(image_bytes).decode("utf-8")
    response = await client.chat.completions.create(
        model=MODEL_NAME,
        messages=[
            {
                "role": "system",
                "content": (
                    "Evaluate the provided image for alignment with the prompt. "
                    'Respond with JSON {"score": <0-100>, "feedback": <text>}'
                ),
            },
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": prompt},
                    {
                        "type": "image_url",
                        "image_url": {
                            "url": f"data:image/png;base64,{encoded}",
                        },
                    },
                ],
            },
        ],
        temperature=0.2,
    )
    content = cast(str, response.choices[0].message.content)
    return json.loads(content)
