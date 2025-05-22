from unittest.mock import AsyncMock, MagicMock

import pytest

from agentic_image_gen import image_gen


@pytest.mark.anyio("asyncio")
async def test_generate_image(monkeypatch):
    mock_client = MagicMock()
    mock_response = MagicMock()
    mock_response.data = [MagicMock(url="img.png")]
    mock_client.images.generate = AsyncMock(return_value=mock_response)
    monkeypatch.setattr(image_gen, "AsyncOpenAI", MagicMock(return_value=mock_client))

    result = await image_gen.generate_image("prompt")

    assert result == "img.png"
