import asyncio
import sys
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock, patch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from agentic_image_gen import prompter


def test_generate_prompt(monkeypatch):
    monkeypatch.setenv("OPENAI_API_KEY", "sk-test")
    mock_client = AsyncMock()
    mock_resp = MagicMock()
    mock_resp.choices = [MagicMock(message=MagicMock(content="new prompt"))]
    mock_client.chat.completions.create.return_value = mock_resp
    with patch("agentic_image_gen.prompter.AsyncOpenAI", return_value=mock_client):
        result = asyncio.run(prompter.generate_prompt("old", "fix lighting"))

    assert result == "new prompt"
    mock_client.chat.completions.create.assert_called_once()
