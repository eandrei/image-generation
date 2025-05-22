import json
from unittest.mock import AsyncMock, MagicMock

import pytest

from agentic_image_gen import evaluator


@pytest.mark.anyio("asyncio")
async def test_evaluate_image_file(monkeypatch, tmp_path):
    img_file = tmp_path / "img.png"
    img_file.write_bytes(b"data")

    mock_client = MagicMock()
    mock_response = MagicMock()
    mock_response.choices = [
        MagicMock(message=MagicMock(content=json.dumps({"score": 95, "feedback": "good"})))
    ]
    mock_client.chat.completions.create = AsyncMock(return_value=mock_response)
    monkeypatch.setattr(evaluator, "AsyncOpenAI", MagicMock(return_value=mock_client))

    result = await evaluator.evaluate_image(str(img_file), "prompt")

    assert result == {"score": 95, "feedback": "good"}
