"""Central configuration, loaded once from the environment / .env file.

Everything the sandbox needs to talk to GCP lives here so the rest of the
codebase never reads os.environ directly.
"""
from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path

from dotenv import load_dotenv

# Load .env from the repo root if present. Real env vars always win over .env.
_REPO_ROOT = Path(__file__).resolve().parent.parent
load_dotenv(_REPO_ROOT / ".env")


@dataclass(frozen=True)
class Settings:
    # --- GCP / Vertex ---
    project_id: str
    vertex_location: str
    # --- Earth Engine service account ---
    ee_service_account: str | None
    credentials_path: str | None
    # --- Models (bare Gemini ids, no provider prefix) ---
    orchestrator_model: str
    worker_model: str
    # --- LLM backend: 'vertex' (default) or 'gemini' (AI Studio) ---
    llm_backend: str
    vertex_api_key: str | None   # API-key auth for Vertex (no service account needed)
    gemini_api_key: str | None   # API key for the AI Studio / Gemini API path

    @property
    def credentials_abspath(self) -> str | None:
        if not self.credentials_path:
            return None
        p = Path(self.credentials_path)
        return str(p if p.is_absolute() else (_REPO_ROOT / p))


def _get(name: str, default: str | None = None) -> str | None:
    val = os.environ.get(name, default)
    return val.strip() if isinstance(val, str) else val


def load_settings() -> Settings:
    project_id = _get("GCP_PROJECT_ID")
    if not project_id:
        raise RuntimeError(
            "GCP_PROJECT_ID is not set. Copy .env.example to .env and fill it in."
        )
    return Settings(
        project_id=project_id,
        vertex_location=_get("VERTEX_LOCATION", "us-central1"),
        ee_service_account=_get("EE_SERVICE_ACCOUNT"),
        credentials_path=_get("GOOGLE_APPLICATION_CREDENTIALS"),
        orchestrator_model=_get("ORCHESTRATOR_MODEL", "gemini-2.5-pro"),
        worker_model=_get("WORKER_MODEL", "gemini-2.5-flash"),
        llm_backend=(_get("LLM_BACKEND", "vertex") or "vertex").lower(),
        vertex_api_key=_get("VERTEX_API_KEY"),
        gemini_api_key=_get("GEMINI_API_KEY"),
    )


# Import-time singleton is intentionally avoided so that importing the package
# never fails just because .env isn't filled in yet. Call load_settings().
