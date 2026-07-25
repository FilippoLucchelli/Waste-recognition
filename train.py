"""Backward-compatible training entry point."""

from waste_recognition.cli.train import main

if __name__ == "__main__":
    raise SystemExit(main())
