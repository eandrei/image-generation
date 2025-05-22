from __future__ import annotations

import asyncio
import base64
import json
import os
import re
from pathlib import Path
from typing import Any

from openai import AsyncOpenAI

SYSTEM_PROMPT = (
    'You evaluate generated images and respond with JSON: {"score": <0-100>, "feedback": <string>}'
)


def _parse_json_response(content: str) -> dict[str, Any]:
    """Extract JSON object from LLM response."""
    text = content.strip()
    if text.startswith("```"):
        text = re.sub(r"^```(?:json)?\n", "", text)
        text = re.sub(r"\n```$", "", text)
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        match = re.search(r"\{.*\}", text, re.DOTALL)
        if match:
            return json.loads(match.group(0))
        raise


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
        response_format={"type": "json_object"},
    )
    content = response.choices[0].message.content or ""
    return _parse_json_response(content)
