import json
from unittest.mock import MagicMock, patch

import pytest

from agentic_image_gen import thread_manager as tm


@pytest.mark.asyncio
async def test_create_thread_writes_config(tmp_path, monkeypatch):
    config_file = tmp_path / "thread_config.json"
    monkeypatch.setattr(tm, "CONFIG_PATH", config_file)

    fake_thread = MagicMock(id="thread_123")
    fake_client = MagicMock()
    fake_client.beta.threads.create.return_value = fake_thread

    with patch("agentic_image_gen.thread_manager.get_client", return_value=fake_client):
        thread_id = await tm.create_thread()

    assert thread_id == "thread_123"
    assert json.loads(config_file.read_text())["thread_id"] == "thread_123"
    assert tm.load_thread_id() == "thread_123"
