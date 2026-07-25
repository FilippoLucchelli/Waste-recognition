"""Backward-compatible evaluation entry point."""

from waste_recognition.cli.evaluate import main

if __name__ == "__main__":
    raise SystemExit(main())
