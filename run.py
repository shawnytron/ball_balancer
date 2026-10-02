"""Run from this checkout without requiring an editable package installation."""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent / "src"))

from beam_balancer.cli import entrypoint

if __name__ == "__main__":
    entrypoint()
