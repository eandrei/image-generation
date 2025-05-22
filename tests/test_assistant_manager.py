import json
from unittest.mock import MagicMock, patch

import pytest

from agentic_image_gen import assistant_manager as am


@pytest.mark.asyncio
async def test_create_assistant_writes_config(tmp_path, monkeypatch):
    config_file = tmp_path / "assistant_config.json"
    monkeypatch.setattr(am, "CONFIG_PATH", config_file)

    fake_assistant = MagicMock(id="asst_123")
    fake_client = MagicMock()
    fake_client.beta.assistants.create.return_value = fake_assistant

    with patch("agentic_image_gen.assistant_manager.get_client", return_value=fake_client):
        assistant_id = await am.create_assistant()

    assert assistant_id == "asst_123"
    assert json.loads(config_file.read_text())["assistant_id"] == "asst_123"
    assert am.load_assistant_id() == "asst_123"
