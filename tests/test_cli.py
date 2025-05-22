from types import SimpleNamespace
from unittest.mock import AsyncMock

import pytest

from agentic_image_gen import cli


@pytest.mark.asyncio
async def test_cli_main_runs_loop(monkeypatch, capsys):
    mock_loop = AsyncMock(return_value={"best_image_url": "img.png"})
    monkeypatch.setattr(cli, "run_image_generation_loop", mock_loop)

    monkeypatch.setattr(
        cli.argparse.ArgumentParser,
        "parse_args",
        lambda self: SimpleNamespace(prompt="hello", refs=None),
    )

    await cli.main()

    mock_loop.assert_awaited_with("hello", None)
    captured = capsys.readouterr().out
    assert "img.png" in captured
