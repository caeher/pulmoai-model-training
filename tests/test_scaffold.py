"""Synthetic checks for package imports and safety defaults."""

from __future__ import annotations

import importlib
import os

import pytest

from respiratory_xray import __version__
from respiratory_xray.safety import NetworkAccessDenied, require_network_access


def test_package_submodules_import_without_ml_dependencies() -> None:
    """All scaffold namespaces are importable without initializing workloads."""
    for module_name in (
        "data",
        "models",
        "training",
        "evaluation",
        "inference",
    ):
        importlib.import_module(f"respiratory_xray.{module_name}")

    assert __version__ == "0.1.0"


def test_network_is_denied_by_default(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("CI", raising=False)
    monkeypatch.delenv("RESPIRATORY_XRAY_ALLOW_NETWORK", raising=False)

    with pytest.raises(NetworkAccessDenied, match="disabled by default"):
        require_network_access("synthetic acquisition")


@pytest.mark.parametrize("ci_value", ["true", "1", "yes"])
def test_network_is_denied_in_ci_even_if_opted_in(
    monkeypatch: pytest.MonkeyPatch, ci_value: str
) -> None:
    monkeypatch.setenv("CI", ci_value)
    monkeypatch.setenv("RESPIRATORY_XRAY_ALLOW_NETWORK", "1")

    with pytest.raises(NetworkAccessDenied, match="disabled in CI"):
        require_network_access("synthetic acquisition")


def test_local_opt_in_allows_future_guarded_operation(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("CI", raising=False)
    monkeypatch.setenv("RESPIRATORY_XRAY_ALLOW_NETWORK", "1")

    assert require_network_access("synthetic operation") is None
    assert os.environ["RESPIRATORY_XRAY_ALLOW_NETWORK"] == "1"
