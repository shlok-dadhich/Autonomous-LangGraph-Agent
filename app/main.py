"""Entrypoint — replaces root main.py. Delegates to app.cli."""

from app.cli import main

if __name__ == "__main__":
    main()
