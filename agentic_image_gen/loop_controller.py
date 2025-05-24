from __future__ import annotations

import sys
from typing import List

from . import (
    assistant_manager,
    evaluator,
    image_gen,
    prompter,
    run_orchestrator,
    thread_manager,
)

MAX_ITERATIONS = 3
SCORE_THRESHOLD = 95


async def run_image_generation_loop(
    prompt: str, 
    reference_images: list[str] | None,
    quality: str,
    size: str,
    background: str,
    output_format: str
) -> dict:
    """Run the iterative prompt→image→evaluate loop.

    Args:
        prompt: The initial prompt from the user.
        reference_images: Optional list of reference image URLs or paths.
        quality: Quality of the generated image (low, medium, high, auto).
        size: Dimensions of the generated image (e.g., 1024x1024).
        background: Background of the generated image (opaque, transparent, auto).
        output_format: Output format (png, jpeg, webp).

    Returns:
        A dictionary containing the best image, final score and full history.
    """
    thread_id = await thread_manager.create_thread()
    assistant_id = assistant_manager.load_assistant_id()
    if assistant_id is None:
        assistant_id = await assistant_manager.create_assistant()

    full_history: List[dict] = []

    best_score = -1
    best_image_url = ""
    current_prompt = prompt
    current_openai_response_id: str | None = None

    for i in range(MAX_ITERATIONS):
        
        iteration_prompt = current_prompt

        gen_result = await image_gen.generate_image(
            prompt=current_prompt, 
            reference_images=reference_images,
            previous_response_id=current_openai_response_id,
            quality=quality,
            size=size,
            background=background,
            output_format=output_format
        )
        image_url = gen_result["image_path"] or ""
        current_openai_response_id = gen_result["response_id"]

        if not image_url:
            print("Failed to generate image in this iteration. Skipping evaluation and prompting.", file=sys.stderr)
            full_history.append({
                "prompter_query": iteration_prompt,
                "result_image": None,
                "evaluator_query": None,
                "score": None,
            })
            if not current_openai_response_id:
                print("Critical failure in initial image generation. Aborting loop.", file=sys.stderr)
                break
            continue

     
        evaluation = await evaluator.evaluate_image(image_url, iteration_prompt)
       
        
        iteration_feedback = evaluation["feedback"]
        score = evaluation["score"]

        full_history.append({
            "prompter_query": iteration_prompt,
            "result_image": image_url,
            "evaluator_query": iteration_feedback,
            "score": score,
        })

        if score > best_score:
            best_score = score
            best_image_url = image_url

        if score >= SCORE_THRESHOLD:
            break

        current_prompt = await prompter.generate_prompt(iteration_prompt, iteration_feedback)

        await run_orchestrator.run_and_stream(thread_id, assistant_id)

    return {
        "best_image_url": best_image_url,
        "final_score": best_score,
        "full_history": full_history,
        "thread_id": thread_id,
    }
