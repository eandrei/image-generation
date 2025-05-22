from __future__ import annotations

from typing import List

from . import (
    assistant_manager,
    evaluator,
    image_gen,
    prompter,
    run_orchestrator,
    thread_manager,
)

MAX_ITERATIONS = 2
SCORE_THRESHOLD = 90


async def run_image_generation_loop(prompt: str, reference_images: list[str] | None) -> dict:
    """Run the iterative prompt→image→evaluate loop.

    Args:
        prompt: The initial prompt from the user.
        reference_images: Optional list of reference image URLs or paths.

    Returns:
        A dictionary containing the best image, final score and full history.
    """
    thread_id = await thread_manager.create_thread()
    assistant_id = assistant_manager.load_assistant_id()
    if assistant_id is None:
        assistant_id = await assistant_manager.create_assistant()

    prompt_history: List[str] = [prompt]
    image_history: List[str] = []
    feedback_history: List[str] = []

    best_score = -1
    best_image_url = ""
    current_prompt = prompt
    current_openai_response_id: str | None = None

    for i in range(MAX_ITERATIONS):
        print(f"\nIteration {i+1}/{MAX_ITERATIONS}")
        
        gen_result = await image_gen.generate_image(
            current_prompt, 
            reference_images if not current_openai_response_id else None,
            current_openai_response_id
        )
        image_url = gen_result["image_path"] or ""
        current_openai_response_id = gen_result["response_id"]

        if not image_url:
            print("Failed to generate image in this iteration. Skipping evaluation and prompting.")
            if not current_openai_response_id:
                print("Critical failure in initial image generation. Aborting loop.")
                break
            continue

        image_history.append(image_url)

        print(f"Generated image: {image_url}")
        print(f"Evaluating image with prompt: {current_prompt}")
        evaluation = await evaluator.evaluate_image(image_url, current_prompt)
        feedback_history.append(evaluation["feedback"])
        score = evaluation["score"]

        if score > best_score:
            best_score = score
            best_image_url = image_url

        if score >= SCORE_THRESHOLD:
            break

        current_prompt = await prompter.generate_prompt(current_prompt, evaluation["feedback"])
        prompt_history.append(current_prompt)

        await run_orchestrator.run_and_stream(thread_id, assistant_id)

    return {
        "best_image_url": best_image_url,
        "final_score": best_score,
        "prompt_history": prompt_history,
        "image_history": image_history,
        "feedback_history": feedback_history,
        "thread_id": thread_id,
    }
