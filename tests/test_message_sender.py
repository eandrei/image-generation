import asyncio
from unittest.mock import MagicMock

from agentic_image_gen import message_sender as ms


def test_send_message(monkeypatch):
    fake_message = MagicMock(id="msg_123")
    fake_client = MagicMock()
    fake_client.beta.threads.messages.create.return_value = fake_message

    monkeypatch.setattr(ms, "get_client", lambda: fake_client)

    message_id = asyncio.run(ms.send_message("thread_1", "hello", ["img1", "img2"]))

    assert message_id == "msg_123"
    fake_client.beta.threads.messages.create.assert_called_once_with(
        thread_id="thread_1",
        role="user",
        content="hello",
        attachments=[
            {"file_id": "img1", "tools": [{"type": "file_search"}]},
            {"file_id": "img2", "tools": [{"type": "file_search"}]},
        ],
    )
