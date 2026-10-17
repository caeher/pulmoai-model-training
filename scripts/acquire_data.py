"""Nonfunctional acquisition placeholder; this script never downloads data."""

from __future__ import annotations

import sys

from pulmoai.safety import NetworkAccessDenied, require_network_access


def main() -> int:
    """Fail clearly until a separately reviewed acquisition tool is implemented."""
    try:
        require_network_access("Dataset acquisition")
    except NetworkAccessDenied as error:
        print(error, file=sys.stderr)
        return 2

    print(
        "No dataset downloader is implemented. PulmoAI never downloads datasets automatically.",
        file=sys.stderr,
    )
    return 2


if __name__ == "__main__":
    raise SystemExit(main())
