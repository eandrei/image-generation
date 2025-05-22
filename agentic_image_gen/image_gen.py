from __future__ import annotations

from typing import Optional


async def generate_image(prompt: str, reference_images: Optional[list[str]] = None) -> str:
    """Generate an image from the given prompt.

    Args:
        prompt: The prompt to use for image generation.
        reference_images: Optional list of reference images.

    Returns:
        str: Path or URL to the generated image.
    """
    raise NotImplementedError
