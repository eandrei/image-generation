from __future__ import annotations

from typing import Optional


async def generate_prompt(previous_prompt: str, feedback: Optional[list[str]] = None) -> str:
    """Generate a new prompt given previous prompt and feedback.

    Args:
        previous_prompt: The prompt from the previous iteration.
        feedback: Optional list of feedback strings from the evaluator.

    Returns:
        str: The refined prompt.
    """
    raise NotImplementedError
