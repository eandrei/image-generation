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
    'You are an expert jewelry photography critic. ' 
    'Evaluate generated images of jewelry based on the provided prompt. ' 
    'Respond ONLY with a JSON object adhering to this schema: {"score": <0-100>, "feedback": <string>}.'
    'Your evaluation criteria should focus on:'
    '1. Clarity and Focus: Is the main jewelry piece sharp, in focus, and well-defined?'
    '2. Material Rendition: Are metals (gold, silver, platinum, etc.) and gemstones (diamonds, rubies, sapphires, emeralds, pearls, etc.) depicted accurately, with appropriate luster, sparkle, and color?'
    '3. Lighting: Is the lighting suitable for high-quality product photography (e.g., soft, even, studio lighting)? Does it enhance the texture, form, and brilliance of the jewelry? Avoid harsh glares or overly dark shadows that obscure details.'
    '4. Background: Is the background clean, non-distracting, and complementary to the jewelry? (e.g., plain white, neutral gradient, or as specified in the prompt). Transparent backgrounds, if requested, should be flawless.'
    '5. Composition: Is the jewelry piece well-composed within the frame? Is it presented from an appealing angle?'
    '6. Detail & Craftsmanship: Are fine details of the jewelry (engravings, settings, small stones) visible and well-rendered?'
    '7. Commercial Appeal: Does the image look professional and appealing for use on an e-commerce website, social media, or marketing materials?'
    '8. Artifacts: Are there any unnatural AI-generated artifacts, distortions, or misinterpretations of the prompt?'
    'Base your score on how well the image meets these criteria in relation to the user\'s prompt. ' 
    'Provide concise, actionable feedback, highlighting strengths and areas for improvement.'
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
