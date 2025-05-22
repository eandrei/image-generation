from __future__ import annotations

import asyncio
import base64
import mimetypes
import os
import tempfile
import uuid
from pathlib import Path
from typing import Any # Added for type hinting

import aiohttp # Added for downloading images from URLs
from openai import AsyncOpenAI


async def _fetch_and_encode_image(session: aiohttp.ClientSession, image_source: str) -> str | None:
    """Fetch image from URL or load from local path, then encode to base64 data URL.

    Args:
        session: The aiohttp ClientSession for making HTTP requests.
        image_source: The URL or local path to the image.

    Returns:
        A base64 data URL string if successful, or None if processing fails.
    """
    try:
        if image_source.startswith(("http://", "https://")):
            async with session.get(image_source) as response:
                if response.status != 200:
                    print(f"Warning: Failed to download image from URL {image_source}. Status: {response.status}")
                    return None
                binary_data = await response.read()
                mime_type = response.content_type
                if not mime_type or not mime_type.startswith("image/"):
                    # Fallback to guessing if content_type is not specific enough
                    guessed_mime_type, _ = mimetypes.guess_type(image_source, strict=False)
                    if guessed_mime_type and guessed_mime_type.startswith("image/"):
                        mime_type = guessed_mime_type
                    else:
                        print(f"Warning: Could not determine a valid image MIME type for URL {image_source}. Content-Type: {mime_type}")
                        # Default to image/png if unable to determine, or handle error more strictly
                        mime_type = "image/png" 
        else:
            image_path = Path(image_source)
            if not image_path.is_file():
                print(f"Warning: Reference image path not found or not a file: {image_source}")
                return None

            mime_type, _ = mimetypes.guess_type(image_path)
            if not mime_type or not mime_type.startswith("image/"):
                print(f"Warning: Could not determine a valid image MIME type for {image_source}")
                return None

            with image_path.open("rb") as f:
                binary_data = f.read()
        
        base64_encoded_data = base64.b64encode(binary_data).decode("utf-8")
        return f"data:{mime_type};base64,{base64_encoded_data}"
    except Exception as e:
        print(f"Error processing image source {image_source}: {e}")
        return None


async def generate_image(
    prompt: str,
    reference_images: list[str] | None = None,
    previous_response_id: str | None = None,
) -> dict[str, str | None]:
    """Generate an image using OpenAI's Responses API.
    Supports initial generation with text and reference images,
    and follow-up generation using a previous_response_id.

    Args:
        prompt: The textual description or follow-up instruction.
        reference_images: Optional list of paths/URLs to reference images (for initial generation).
        previous_response_id: Optional ID of the previous response for multi-turn generation.

    Returns:
        A dictionary containing:
            "image_path": Path to the generated image (or empty string on failure).
            "response_id": The ID of the OpenAI API response (or None on failure).
    """
    client = AsyncOpenAI(api_key=os.getenv("OPENAI_API_KEY"))
    generated_image_path = ""
    current_api_response_id: str | None = None

    try:
        call_params: dict[str, Any] = {
            "model": "gpt-4o",
            "tools": [{"type": "image_generation"}]
        }

        if previous_response_id:
            call_params["input"] = prompt  # Simple string prompt for follow-up
            call_params["previous_response_id"] = previous_response_id
        else:
            # Initial generation: construct detailed input
            input_user_content_list: list[dict] = [{"type": "input_text", "text": prompt}]
            if reference_images:
                async with aiohttp.ClientSession() as session:
                    tasks = [_fetch_and_encode_image(session, ref_img_src) for ref_img_src in reference_images]
                    encoded_images = await asyncio.gather(*tasks)
                    for ref_img_src, base64_data_url in zip(reference_images, encoded_images):
                        if base64_data_url:
                            input_user_content_list.append({"type": "input_image", "image_url": base64_data_url})
                        else:
                            print(f"Skipping reference image due to processing error: {ref_img_src}")
            
            # Avoid API call if only an empty text prompt and no reference images for initial call
            is_prompt_empty = not prompt.strip()
            has_only_empty_text_prompt = len(input_user_content_list) == 1 and input_user_content_list[0]["type"] == "input_text" and is_prompt_empty
            
            if has_only_empty_text_prompt:
                print("Warning: No valid prompt or reference images provided for initial image generation.")
                return {"image_path": "", "response_id": None}

            call_params["input"] = [{"role": "user", "content": input_user_content_list}]

        response = await client.responses.create(**call_params)
        current_api_response_id = response.id
        
        image_base64_data = None
        if response.output:
            for output_item in response.output:
                if hasattr(output_item, 'type') and output_item.type == "image_generation_call":
                    if hasattr(output_item, 'result') and output_item.result:
                        image_base64_data = output_item.result
                        break
        
        if image_base64_data:
            if ',' in image_base64_data:
                image_base64_data = image_base64_data.split(',', 1)[1]
            image_bytes = base64.b64decode(image_base64_data)
            temp_dir = tempfile.gettempdir()
            file_name = f"generated_image_{uuid.uuid4()}.png"
            generated_image_path = str(Path(temp_dir) / file_name)
            with open(generated_image_path, "wb") as f:
                f.write(image_bytes)
        else:
            print("No image data found in the API response.")
            if response.output and hasattr(response.output[0], 'content') :
                print(f"API response output content: {response.output[0].content}")

    except Exception as e:
        print(f"Error during image generation with Responses API: {e}")
    
    return {"image_path": generated_image_path, "response_id": current_api_response_id}
