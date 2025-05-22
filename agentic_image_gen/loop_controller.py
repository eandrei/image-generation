from __future__ import annotations

from typing import List, Optional, TypedDict

from . import evaluator, image_gen, prompter, thread_manager

MAX_ITERATIONS = 10
SCORE_THRESHOLD = 90


class LoopResult(TypedDict):
    best_image_url: Optional[str]
    final_score: int
    prompt_history: List[str]
    image_history: List[str]
    feedback_history: List[str]


async def run_image_generation_loop(
    prompt: str, reference_images: Optional[list[str]] = None
) -> LoopResult:
    """Run the prompt→image→evaluation loop.

    Args:
        prompt: The initial user prompt.
        reference_images: Optional list of reference images.

    Returns:
        LoopResult: Final image information and iteration histories.
    """
    prompt_history: List[str] = [prompt]
    image_history: List[str] = []
    feedback_history: List[str] = []

    best_image: Optional[str] = None
    best_score = -1

    thread_id = await thread_manager.create_thread(reference_images)

    current_prompt = prompt
    for _ in range(MAX_ITERATIONS):
        image_path = await image_gen.generate_image(current_prompt, reference_images)
        image_history.append(image_path)

        evaluation = await evaluator.evaluate_image(image_path, current_prompt)
        score = int(evaluation["score"])
        feedback = str(evaluation["feedback"])
        feedback_history.append(feedback)

        if score > best_score:
            best_score = score
            best_image = image_path

        await thread_manager.add_message(thread_id, current_prompt)
        await thread_manager.add_message(thread_id, feedback)

        if score >= SCORE_THRESHOLD:
            break

        current_prompt = await prompter.generate_prompt(current_prompt, [feedback])
        prompt_history.append(current_prompt)

    return {
        "best_image_url": best_image,
        "final_score": best_score,
        "prompt_history": prompt_history,
        "image_history": image_history,
        "feedback_history": feedback_history,
    }
