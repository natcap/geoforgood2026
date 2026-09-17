"""Single place that turns one GCP credential into working Earth Engine
*and* Vertex Gemini access.

Design goal: one credential, zero interactive prompts. Two credentials work
here, and the same one authorizes both services:

  * a **service-account key** file (GOOGLE_APPLICATION_CREDENTIALS), or
  * **Application Default Credentials** — `gcloud auth application-default
    login` on a laptop, or the attached service account on GCE / Cloud Run /
    Colab, where there is no key file to ship.

Whichever is present, a single bootstrap() call gets everything working.
"""
from __future__ import annotations

import os
from pathlib import Path

import ee

from .config import Settings, load_settings
from .models import VertexAIServerModel

_EE_INITIALIZED = False

# Where `gcloud auth application-default login` writes user credentials.
_ADC_WELL_KNOWN = Path.home() / ".config" / "gcloud" / "application_default_credentials.json"


def _key_file(settings: Settings) -> str | None:
    """The service-account key's absolute path, or None if there isn't one on disk."""
    p = settings.credentials_abspath
    return p if p and os.path.exists(p) else None


def _apply_gcp_credentials(settings: Settings) -> str:
    """Point google-auth at whichever credential this machine actually has.

    Returns "service_account" or "adc". Two environment quirks are handled:

    * python-dotenv loads GOOGLE_APPLICATION_CREDENTIALS verbatim (often a
      relative './secrets/sa-key.json') and google-auth resolves that against
      the process working directory, so we rewrite it absolute.
    * If .env still carries a key path but no key was ever dropped in, we
      *remove* the variable. Left set, it makes google.auth.default() raise
      "file was not found" instead of falling back to ADC.
    """
    key_path = _key_file(settings)
    if key_path:
        os.environ["GOOGLE_APPLICATION_CREDENTIALS"] = key_path  # hard set, not setdefault
        return "service_account"

    os.environ.pop("GOOGLE_APPLICATION_CREDENTIALS", None)
    # User ADC carries no project of its own, and some Google APIs reject a
    # call with no quota project attached. Ours is the one to bill.
    os.environ.setdefault("GOOGLE_CLOUD_QUOTA_PROJECT", settings.project_id)
    return "adc"


def _adc_hint(settings: Settings, error: Exception) -> str:
    """Explain an ADC failure in terms of the commands that fix it."""
    lines = [
        "Earth Engine could not initialize with Application Default Credentials "
        f"for project '{settings.project_id}': {type(error).__name__}: {error}",
    ]
    configured = settings.credentials_abspath
    if configured:
        lines.append(
            f"GOOGLE_APPLICATION_CREDENTIALS points at {configured}, which does "
            "not exist, so ADC was used instead. Either drop the key file there "
            "or clear that line in .env."
        )
    if not _ADC_WELL_KNOWN.exists():
        lines.append(
            "No ADC file found. Run once, in a terminal:\n"
            f"  gcloud auth application-default login --project {settings.project_id}"
        )
    lines.append(
        "The signed-in account also needs the project registered for Earth "
        "Engine: https://code.earthengine.google.com/register"
    )
    return "\n".join(lines)


def init_earth_engine(settings: Settings | None = None) -> None:
    """Initialize the Earth Engine client.

    Uses the service-account key when one is configured and present on disk,
    and Application Default Credentials otherwise. Never calls
    ee.Authenticate(): it opens a browser and blocks, which would hang any
    non-interactive caller.
    """
    global _EE_INITIALIZED
    if _EE_INITIALIZED:
        return

    settings = settings or load_settings()
    mode = _apply_gcp_credentials(settings)

    if mode == "service_account" and settings.ee_service_account:
        creds = ee.ServiceAccountCredentials(
            settings.ee_service_account, _key_file(settings)
        )
        ee.Initialize(creds, project=settings.project_id)
        _EE_INITIALIZED = True
        return

    # ADC, or a key file with no EE_SERVICE_ACCOUNT email set (google-auth
    # reads the email straight out of the JSON in that case).
    try:
        ee.Initialize(project=settings.project_id)
    except Exception as error:  # noqa: BLE001 - re-raised with the fix attached
        raise RuntimeError(_adc_hint(settings, error)) from error

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
    _apply_gcp_credentials(settings)  # key file if present, else ADC
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
