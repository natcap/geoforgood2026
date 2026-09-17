"""Generic tool-directory loader, shared by forge.py (tool_smith-generated
tools) and project_tools.py (hand-written model tools).

Import every .py module in a directory and collect the smolagents Tool
objects defined in it, so a crew can pick up new tools on the next
`build_crew()` with no manual wiring.
"""
from __future__ import annotations

import importlib
import sys
from pathlib import Path

from smolagents.tools import Tool


def load_tools(directory: Path, package: str, reload: bool = True) -> list[Tool]:
    """Import every .py module in `directory` (as `package.<stem>`) and return
    the Tool objects found in them.

    Args:
        directory: Folder to scan for .py files (non-recursive).
        package: Dotted import path corresponding to `directory`.
        reload: re-import modules already loaded this session, so edits made
            mid-session are picked up.
    """
    found: dict[str, Tool] = {}
    if not directory.exists():
        return []
    for py in sorted(directory.glob("*.py")):
        if py.stem == "__init__":
            continue
        modname = f"{package}.{py.stem}"
        try:
            if reload and modname in sys.modules:
                mod = importlib.reload(sys.modules[modname])
            else:
                mod = importlib.import_module(modname)
        except Exception as e:  # a half-written tool shouldn't break the crew
            print(f"[registry] skipped {py.stem}.py: {type(e).__name__}: {e}")
            continue
        for obj in vars(mod).values():
            if isinstance(obj, Tool):
                found[obj.name] = obj  # last definition of a name wins
    return list(found.values())
