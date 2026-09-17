"""Loads the hand-written geospatial model tools from `tools/models/`.

Add your own model as a `@tool`-decorated function in its own file under
`tools/models/` and it's picked up automatically on the next `build_crew()` —
no manual wiring. See `tools/models/example_forest_loss.py` for the expected
shape: verify EE assets, compute server-side, then publish the result to the
shared `results` board (`results.current().add_layer(...)` /
`.add_stat(...)`) so the app's map and stats table show it.
"""
from __future__ import annotations

from pathlib import Path

from smolagents.tools import Tool

from .tools.registry import load_tools

_MODELS_DIR = Path(__file__).resolve().parent / "tools" / "models"
_PKG = "natcap_agents.tools.models"


def load_project_tools(reload: bool = True) -> list[Tool]:
    return load_tools(_MODELS_DIR, _PKG, reload=reload)
