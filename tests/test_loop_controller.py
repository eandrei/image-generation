from unittest.mock import AsyncMock

import pytest

from agentic_image_gen import loop_controller


@pytest.mark.asyncio
async def test_early_exit_on_high_score(monkeypatch):
    monkeypatch.setattr(
        loop_controller.thread_manager, "create_thread", AsyncMock(return_value="t1")
    )
    monkeypatch.setattr(loop_controller.assistant_manager, "load_assistant_id", lambda: "a1")
    monkeypatch.setattr(loop_controller.run_orchestrator, "run_and_stream", AsyncMock())
    monkeypatch.setattr(loop_controller.image_gen, "generate_image", AsyncMock(return_value="img1"))
    monkeypatch.setattr(
        loop_controller.evaluator,
        "evaluate_image",
        AsyncMock(return_value={"score": 95, "feedback": "good"}),
    )
    prompter_mock = AsyncMock(return_value="improved")
    monkeypatch.setattr(loop_controller.prompter, "generate_prompt", prompter_mock)

    result = await loop_controller.run_image_generation_loop("start", None)

    assert result["best_image_url"] == "img1"
    assert result["final_score"] == 95
    assert result["prompt_history"] == ["start"]
    assert result["image_history"] == ["img1"]
    assert result["feedback_history"] == ["good"]
    prompter_mock.assert_not_awaited()


@pytest.mark.asyncio
async def test_history_tracking(monkeypatch):
    monkeypatch.setattr(
        loop_controller.thread_manager, "create_thread", AsyncMock(return_value="t1")
    )
    monkeypatch.setattr(loop_controller.assistant_manager, "load_assistant_id", lambda: "a1")
    run_mock = AsyncMock()
    monkeypatch.setattr(loop_controller.run_orchestrator, "run_and_stream", run_mock)
    monkeypatch.setattr(
        loop_controller.image_gen,
        "generate_image",
        AsyncMock(side_effect=["img1", "img2", "img3"]),
    )
    monkeypatch.setattr(
        loop_controller.evaluator,
        "evaluate_image",
        AsyncMock(
            side_effect=[
                {"score": 30, "feedback": "fb1"},
                {"score": 50, "feedback": "fb2"},
                {"score": 92, "feedback": "fb3"},
            ]
        ),
    )
    prompter_mock = AsyncMock(side_effect=["p2", "p3"])
    monkeypatch.setattr(loop_controller.prompter, "generate_prompt", prompter_mock)

    result = await loop_controller.run_image_generation_loop("start", None)

    assert result["best_image_url"] == "img3"
    assert result["final_score"] == 92
    assert result["prompt_history"] == ["start", "p2", "p3"]
    assert result["image_history"] == ["img1", "img2", "img3"]
    assert result["feedback_history"] == ["fb1", "fb2", "fb3"]
    assert prompter_mock.await_count == 2
    assert run_mock.await_count == 2
