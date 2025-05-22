from __future__ import annotations

import os

from openai import AsyncOpenAI

SYSTEM_PROMPT = "You refine image generation prompts based on evaluator feedback while keeping the original intent."


async def generate_prompt(previous_prompt: str, feedback: str) -> str:
    """Generate a refined prompt using OpenAI's API.

    Args:
        previous_prompt: The prior prompt that was used for image generation.
        feedback: Feedback from the evaluator describing how to improve the prompt.

    Returns:
        The refined prompt suggested by the language model.
    """
    client = AsyncOpenAI(api_key=os.getenv("OPENAI_API_KEY"))
    response = await client.chat.completions.create(
        model="gpt-4o",
        messages=[
            {"role": "system", "content": SYSTEM_PROMPT},
            {
                "role": "user",
                "content": f"Prompt: {previous_prompt}\nFeedback: {feedback}",
            },
        ],
        temperature=0.2,
    )
    content = response.choices[0].message.content or ""
    return content.strip()
