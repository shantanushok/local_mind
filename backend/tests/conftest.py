import logging
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))


@pytest.fixture(autouse=True, scope="session")
def suppress_console_logs():
    """Removes console StreamHandlers from root logger during test runs.

    Prevents individual request logging (e.g. Incoming Request, HTTP Request)
    from cluttering the test runner and benchmark tables, while keeping caplog
    and test assertions fully functional.
    """
    root = logging.getLogger()
    for handler in list(root.handlers):
        if isinstance(handler, logging.StreamHandler):
            root.removeHandler(handler)
    logging.getLogger("httpx").setLevel(logging.WARNING)
    logging.getLogger("httpcore").setLevel(logging.WARNING)
    logging.getLogger("uvicorn").setLevel(logging.WARNING)
    logging.getLogger("app").setLevel(logging.WARNING)
    logging.getLogger("middleware.csrf").setLevel(logging.WARNING)
