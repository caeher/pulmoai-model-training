"""Block outbound network connections throughout test collection and execution."""

from __future__ import annotations

import socket
from typing import Any

_original_connect = socket.socket.connect
_original_create_connection = socket.create_connection


def pytest_unconfigure(config):
    socket.socket.connect = _original_connect
    socket.create_connection = _original_create_connection


def _deny_network(*args: Any, **kwargs: Any) -> None:
    del args, kwargs
    raise RuntimeError("Network access is disabled in the test suite.")


socket.create_connection = _deny_network
socket.socket.connect = _deny_network


from pathlib import Path  # noqa: E402

import pytest  # noqa: E402


@pytest.fixture
def config_dir() -> Path:
    return Path(__file__).resolve().parents[1] / "configs"
