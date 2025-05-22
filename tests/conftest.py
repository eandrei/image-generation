import asyncio
import sys
from pathlib import Path

import pytest

@pytest.fixture
def anyio_backend():
    return "asyncio"

# Ensure package root is on sys.path for imports
ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))


def pytest_configure(config):
    config.addinivalue_line("markers", "asyncio: run test in event loop")


@pytest.hookimpl(tryfirst=True)
def pytest_pyfunc_call(pyfuncitem):
    if pyfuncitem.get_closest_marker("asyncio"):
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
        arg_names = pyfuncitem.function.__code__.co_varnames[
            : pyfuncitem.function.__code__.co_argcount
        ]
        kwargs = {name: pyfuncitem.funcargs[name] for name in arg_names}
        loop.run_until_complete(pyfuncitem.obj(**kwargs))
        loop.close()
        return True
    return None

