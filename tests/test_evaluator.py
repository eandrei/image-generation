import asyncio
import json
import sys
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock, patch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from agentic_image_gen import evaluator


def test_evaluate_image(tmp_path, monkeypatch):
    monkeypatch.setenv("OPENAI_API_KEY", "sk-test")
    image_file = tmp_path / "img.png"
    image_file.write_bytes(b"data")

    mock_client = AsyncMock()
    mock_resp = MagicMock()
    mock_resp.choices = [
        MagicMock(message=MagicMock(content=json.dumps({"score": 91, "feedback": "good"})))
    ]
    mock_client.chat.completions.create.return_value = mock_resp
    with patch("agentic_image_gen.evaluator.AsyncOpenAI", return_value=mock_client):
        result = asyncio.run(evaluator.evaluate_image(str(image_file), "prompt"))

    assert result == {"score": 91, "feedback": "good"}
    mock_client.chat.completions.create.assert_called_once()
