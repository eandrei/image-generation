import asyncio
import sys
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock, patch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from agentic_image_gen import image_gen


def test_generate_image(monkeypatch):
    monkeypatch.setenv("OPENAI_API_KEY", "sk-test")
    mock_client = AsyncMock()
    mock_resp = MagicMock()
    mock_resp.data = [MagicMock(url="http://img.png")]
    mock_client.images.generate.return_value = mock_resp
    with patch("agentic_image_gen.image_gen.AsyncOpenAI", return_value=mock_client):
        url = asyncio.run(image_gen.generate_image("prompt"))

    assert url == "http://img.png"
    mock_client.images.generate.assert_called_once()
