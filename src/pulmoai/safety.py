"""Explicit opt-in guard for future network operations.

This module does not read credential files or perform network operations. Code
that may access the network must call :func:`require_network_access` first.
CI always denies access, even if the opt-in variable is set.
"""

from __future__ import annotations

import os

NETWORK_OPT_IN = "PULMOAI_ALLOW_NETWORK"


class NetworkAccessDenied(RuntimeError):
    """Raised when a guarded network operation has not been explicitly enabled."""


def require_network_access(operation: str) -> None:
    """Raise unless network use is explicitly enabled outside CI.

    Args:
        operation: Short description included in the error for the caller.
    """
    if os.environ.get("CI", "").strip().lower() in {"1", "true", "yes"}:
        raise NetworkAccessDenied(f"{operation} is disabled in CI.")

    if os.environ.get(NETWORK_OPT_IN, "0").strip() != "1":
        raise NetworkAccessDenied(
            f"{operation} is disabled by default. Set {NETWORK_OPT_IN}=1 to opt in locally."
        )
