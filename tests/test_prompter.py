from unittest.mock import AsyncMock, MagicMock

import pytest

from agentic_image_gen import prompter


@pytest.mark.anyio("asyncio")
async def test_generate_prompt(monkeypatch):
    mock_client = MagicMock()
    mock_response = MagicMock()
    mock_response.choices = [MagicMock(message=MagicMock(content="new prompt"))]
    mock_client.chat.completions.create = AsyncMock(return_value=mock_response)
    monkeypatch.setattr(prompter, "AsyncOpenAI", MagicMock(return_value=mock_client))

    result = await prompter.generate_prompt("old", "feedback")

    assert result == "new prompt"
