import asyncio

from agentic_image_gen.loop_controller import SCORE_THRESHOLD, run_image_generation_loop


def test_early_exit_on_high_score(monkeypatch):
    images = []

    async def fake_generate_image(prompt, reference_images=None):
        images.append(prompt)
        return "image0.png"

    async def fake_evaluate_image(image_path, prompt):
        return {"score": SCORE_THRESHOLD + 1, "feedback": "good"}

    async def fake_generate_prompt(prev_prompt, feedback):
        return prev_prompt + " refined"

    async def fake_create_thread(ref_imgs=None):
        return "thread1"

    async def fake_add_message(thread_id, message):
        pass

    monkeypatch.setattr("agentic_image_gen.image_gen.generate_image", fake_generate_image)
    monkeypatch.setattr("agentic_image_gen.evaluator.evaluate_image", fake_evaluate_image)
    monkeypatch.setattr("agentic_image_gen.prompter.generate_prompt", fake_generate_prompt)
    monkeypatch.setattr("agentic_image_gen.thread_manager.create_thread", fake_create_thread)
    monkeypatch.setattr("agentic_image_gen.thread_manager.add_message", fake_add_message)

    result = asyncio.run(run_image_generation_loop("start", None))

    assert result["final_score"] >= SCORE_THRESHOLD
    assert len(result["image_history"]) == 1
    assert len(result["prompt_history"]) == 1
    assert len(result["feedback_history"]) == 1
    assert result["best_image_url"] == "image0.png"


def test_history_tracking(monkeypatch):
    calls = {"generate_image": 0, "evaluate_image": 0, "generate_prompt": 0}

    async def fake_generate_image(prompt, reference_images=None):
        calls["generate_image"] += 1
        return f"image{calls['generate_image']}.png"

    scores = [10, 50, SCORE_THRESHOLD + 5]
    feedbacks = ["bad", "better", "great"]

    async def fake_evaluate_image(image_path, prompt):
        idx = calls["evaluate_image"]
        calls["evaluate_image"] += 1
        return {"score": scores[idx], "feedback": feedbacks[idx]}

    async def fake_generate_prompt(prev_prompt, feedback):
        calls["generate_prompt"] += 1
        return f"{prev_prompt}-{calls['generate_prompt']}"

    async def fake_create_thread(ref_imgs=None):
        return "thread2"

    async def fake_add_message(thread_id, message):
        pass

    monkeypatch.setattr("agentic_image_gen.image_gen.generate_image", fake_generate_image)
    monkeypatch.setattr("agentic_image_gen.evaluator.evaluate_image", fake_evaluate_image)
    monkeypatch.setattr("agentic_image_gen.prompter.generate_prompt", fake_generate_prompt)
    monkeypatch.setattr("agentic_image_gen.thread_manager.create_thread", fake_create_thread)
    monkeypatch.setattr("agentic_image_gen.thread_manager.add_message", fake_add_message)

    result = asyncio.run(run_image_generation_loop("start", None))

    assert result["final_score"] == scores[-1]
    assert result["best_image_url"] == "image3.png"
    assert result["prompt_history"] == ["start", "start-1", "start-1-2"]
    assert len(result["image_history"]) == 3
    assert len(result["feedback_history"]) == 3
