"""Compatibility entry point for explicit local LUNA16 preparation."""

import sys

from pulmoai.cli import main

if __name__ == "__main__":
    raise SystemExit(main(["prepare", *sys.argv[1:]]))
