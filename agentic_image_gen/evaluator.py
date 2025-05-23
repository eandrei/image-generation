from __future__ import annotations

import asyncio
import base64
import json
import os
import re
import sys
from pathlib import Path
from typing import Any

from openai import AsyncOpenAI

SYSTEM_PROMPT = (
    'You are an expert jewelry photography critic. Your primary task is to evaluate how faithfully a generated image reproduces a jewelry product based on the user\'s prompt. ' 
    'If the prompt implies the recreation of a specific item (e.g., \'the ring from reference image X\', or a very detailed textual description of a unique piece), then the accuracy of that product\'s depiction is paramount.'
    'Respond ONLY with a JSON object adhering to this schema: {"score": <0-100>, "feedback": <string>}.'
    'Evaluation Priorities (score heavily on these in order):'
    '1. Product Detail Fidelity & Accuracy: weight 80%'
    '   - Core Requirement: How closely does the main jewelry piece in the generated image match the specific characteristics described or implied by the prompt? This includes its unique design, shape, settings, number and type of gemstones, and any distinctive features.'
    '   - Material Rendition: Are metals (gold, silver, platinum, etc.) and gemstones (diamonds, sapphires, etc.) depicted with convincing color, luster, texture, and sparkle true to the described/implied item?'
    '   - Detail & Craftsmanship Clarity: Are fine details like engravings, prongs, milgrain, stone faceting, and small accent stones rendered clearly, sharply, and accurately? The image should withstand scrutiny as if it were a real product photo.'
    '   - Focus: Is the jewelry piece itself impeccably sharp and in focus?'
    '2. Natural Integration & Context (Supporting the Product): weight 10%'
    '   - Believability in Scene: If a lifestyle image, does the product look naturally integrated into the scene (e.g., correctly worn, interacting with surfaces, casting appropriate shadows/reflections)? It must not look like a simple cutout or poorly composited.'
    '   - Lighting on Product: Does the lighting on the jewelry itself appear consistent with the surrounding scene while also optimally showcasing the product\'s features, brilliance, and form? Avoid harsh glares or shadows that obscure product details.'
    '   - Composition & Angle: Is the jewelry presented from an angle that is flattering and informative for the product, as guided by the prompt?'
    '   - Background & Scene Elements: Do the background and other scene elements (if any) complement the product and align with the prompt, without distracting from the jewelry itself?'
    '3. Overall Image Quality & Commercial Viability: weight 10%'
    '   - Artifacts & Distortions: Are there any AI-generated artifacts, unnatural textures, or distortions, particularly on or near the jewelry piece?'
    '   - Commercial Appeal: Considering the above, does the image effectively showcase the jewelry for its intended commercial purpose (e.g., e-commerce, social media)?'
    'Your feedback should be specific, actionable, and clearly distinguish between successes/failures in product representation versus contextual elements. If product accuracy is low, the overall score should reflect this significantly, even if the background scene is well-rendered.'
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

    try:
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
        
        if not content.strip():
            print(f"Warning: Received empty response from OpenAI API for evaluation. Response: {response}", file=sys.stderr)
            # Return a default response instead of crashing
            return {
                "score": 0,
                "feedback": "Error: Unable to evaluate image due to empty API response."
            }
        
        return _parse_json_response(content)
        
    except Exception as e:
        print(f"Error during image evaluation: {e}", file=sys.stderr)
        print(f"Response details: {getattr(response, 'model_dump', lambda: 'No response details available')()}", file=sys.stderr)
        # Return a default response instead of crashing
        return {
            "score": 0,
            "feedback": f"Error: Unable to evaluate image due to API error: {str(e)}"
        }
