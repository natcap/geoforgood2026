"""Discover and load tools a Tool Smith agent has forged into
`natcap_agents/tools/generated/`.

`load_forged()` returns the smolagents Tool objects found there, so a crew
picks up freshly-built tools on the next `build_crew()` — no manual wiring.
`list_forged()` returns a compact summary for display.

Note: loading runs the generated modules' import-time code. These are tools
your own crew wrote, but treat `generated/` as trusted for that reason.
"""
from __future__ import annotations

from pathlib import Path

from smolagents.tools import Tool

from .tools.registry import load_tools

_GENERATED_DIR = Path(__file__).resolve().parent / "tools" / "generated"
_PKG = "natcap_agents.tools.generated"


def load_forged(reload: bool = True) -> list[Tool]:
    return load_tools(_GENERATED_DIR, _PKG, reload=reload)


def list_forged() -> list[dict]:
    """Return [{name, description, inputs}] for each forged tool (for display)."""
    return [
        {
            "name": t.name,
            "description": (t.description or "").strip().split("\n")[0],
            "inputs": ", ".join(t.inputs.keys()),
        }
        for t in load_forged()
    ]
