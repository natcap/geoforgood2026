"""Guardrails for the code-writing agents.

The orchestrator is a smolagents `CodeAgent`: it executes the Python it writes
in smolagents' local executor, which only permits imports on an allowlist. Keep
that allowlist tight and centralized here. Swap the agent's executor_type to
'e2b' or 'docker' for full process isolation (see agents.py).
"""
from __future__ import annotations

# Modules the code agent may import. Earth Engine does the heavy compute
# server-side, so agents rarely need anything beyond ee + helpers.
AUTHORIZED_IMPORTS: list[str] = [
    "ee",
    "geemap",
    "math",
    "json",
    "statistics",
    "datetime",
    "re",
]

# Safety ceiling on agent reasoning loops (overridable per agent).
DEFAULT_MAX_STEPS = 8


def ensure_ee() -> None:
    """Idempotently initialize Earth Engine. Tools call this so they work even
    if invoked before bootstrap() ran."""
    from .auth import init_earth_engine
    init_earth_engine()
