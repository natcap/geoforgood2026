"""Single place that turns one GCP service-account key into working
Earth Engine *and* Vertex Gemini access.

Design goal: one credential, zero interactive prompts. The same key file
authorizes both services, so a single bootstrap() call gets everything working.
"""
from __future__ import annotations

import os

import ee

from .config import Settings, load_settings
from .models import VertexAIServerModel

_EE_INITIALIZED = False


def _export_service_account_creds(settings: Settings) -> None:
    """Force GOOGLE_APPLICATION_CREDENTIALS to an ABSOLUTE path.

    python-dotenv loads whatever is in .env verbatim (often a relative
    './secrets/sa-key.json'), and google-auth resolves that against the process
    working directory. We overwrite it with the absolute path so the key is
    found from anywhere.
    """
    p = settings.credentials_abspath
    if p:
        os.environ["GOOGLE_APPLICATION_CREDENTIALS"] = p  # hard set, not setdefault


def init_earth_engine(settings: Settings | None = None) -> None:
    """Initialize the Earth Engine client with the service account.

    Falls back to Application Default Credentials (ADC) if no explicit
    service-account email is configured but GOOGLE_APPLICATION_CREDENTIALS
    points at a key file.
    """
    global _EE_INITIALIZED
    if _EE_INITIALIZED:
        return

    settings = settings or load_settings()
    _export_service_account_creds(settings)
    key_path = settings.credentials_abspath

    if settings.ee_service_account and key_path:
        if not os.path.exists(key_path):
            raise FileNotFoundError(
                f"Service-account key not found at {key_path}. "
                "Check GOOGLE_APPLICATION_CREDENTIALS in your .env."
            )
        creds = ee.ServiceAccountCredentials(settings.ee_service_account, key_path)
        ee.Initialize(creds, project=settings.project_id)
    else:
        # ADC path: works if GOOGLE_APPLICATION_CREDENTIALS is set or the
        # environment is already GCP-authenticated (e.g. a VM).
        ee.Initialize(project=settings.project_id)

    _EE_INITIALIZED = True


def make_model(model_id: str, settings: Settings | None = None, **kwargs) -> VertexAIServerModel:
    """Build a Gemini model via Google's google-genai SDK (not LiteLLM).

    Auth, selected by LLM_BACKEND and any API key in .env:
      - 'vertex' (default) + an API key (VERTEX_API_KEY): Vertex **Express mode** —
        authenticates with the API key, no service account needed.
      - 'vertex' + no API key: Vertex via the **service account** (ADC /
        GOOGLE_APPLICATION_CREDENTIALS, forced absolute) + project/location.
      - 'gemini' + an API key: the **Gemini Developer API** with the key.
    """
    settings = settings or load_settings()
    api_key = settings.vertex_api_key or settings.gemini_api_key

    if settings.llm_backend == "gemini":
        if not api_key:
            raise RuntimeError(
                "LLM_BACKEND=gemini but no API key set. Put your key in "
                "GEMINI_API_KEY (or VERTEX_API_KEY) in .env."
            )
        return VertexAIServerModel(model_id, api_key=api_key, use_vertex=False, **kwargs)

    # Vertex AI.
    if api_key:
        # Express mode: API key straight to Vertex.
        return VertexAIServerModel(model_id, api_key=api_key, use_vertex=True, **kwargs)

    # Service-account / ADC path.
    _export_service_account_creds(settings)  # absolute path so ADC finds the key
    return VertexAIServerModel(
        model_id, project=settings.project_id, location=settings.vertex_location,
        use_vertex=True, **kwargs,
    )


def orchestrator_model(settings: Settings | None = None, **kwargs) -> VertexAIServerModel:
    settings = settings or load_settings()
    return make_model(settings.orchestrator_model, settings, **kwargs)


def worker_model(settings: Settings | None = None, **kwargs) -> VertexAIServerModel:
    settings = settings or load_settings()
    return make_model(settings.worker_model, settings, **kwargs)


def bootstrap() -> Settings:
    """Convenience: load settings, init EE, return settings. Call once at start."""
    settings = load_settings()
    init_earth_engine(settings)
    return settings
