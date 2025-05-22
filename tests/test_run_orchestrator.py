from unittest.mock import AsyncMock, MagicMock

import pytest
from openai._models import construct_type
from openai.types.beta.threads.runs.run_step import RunStep

from agentic_image_gen import run_orchestrator as ro


@pytest.mark.asyncio
async def test_run_and_stream_collects_tool_calls(monkeypatch):
    run_step = construct_type(
        value={
            "id": "step1",
            "assistant_id": "asst_123",
            "cancelled_at": None,
            "completed_at": 1,
            "created_at": 1,
            "expired_at": None,
            "failed_at": None,
            "last_error": None,
            "metadata": None,
            "object": "thread.run.step",
            "run_id": "run_123",
            "status": "completed",
            "step_details": {
                "type": "tool_calls",
                "tool_calls": [
                    {
                        "id": "call1",
                        "type": "function",
                        "function": {"name": "foo", "arguments": "{}"},
                    }
                ],
            },
            "thread_id": "thread_123",
            "type": "tool_calls",
            "usage": None,
        },
        type_=RunStep,
    )

    handler = MagicMock()
    handler.until_done = AsyncMock()
    handler.get_final_run_steps = AsyncMock(return_value=[run_step])

    class FakeStream:
        async def __aenter__(self):
            return handler

        async def __aexit__(self, exc_type, exc, tb):
            pass

    fake_runs = MagicMock()
    fake_runs.stream.return_value = FakeStream()

    fake_client = MagicMock()
    fake_client.beta.threads.runs = fake_runs

    monkeypatch.setattr(ro, "get_async_client", lambda: fake_client)
    monkeypatch.setattr(ro, "AsyncAssistantEventHandler", lambda: handler)

    result = await ro.run_and_stream("thread_123", "asst_123")

    fake_runs.stream.assert_called_once_with(
        thread_id="thread_123",
        assistant_id="asst_123",
        event_handler=handler,
    )
    handler.until_done.assert_awaited()
    assert result == [{"id": "call1", "type": "function", "name": "foo", "arguments": "{}"}]
