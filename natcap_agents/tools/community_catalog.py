"""Search the GEE Community Catalog (https://gee-community-catalog.org/about/)
— datasets published as ordinary Earth Engine assets (mostly under
`projects/sat-io/...`) that are NOT in Google's own EE Data Catalog.

Sourced from the catalog's own data file (awesome-gee-community-datasets),
cached on disk for a day so repeated searches don't re-download ~3.5 MB.
"""
from __future__ import annotations

import json
import time
import urllib.request
from pathlib import Path

from smolagents import tool

_SOURCE_URL = (
    "https://raw.githubusercontent.com/samapriya/awesome-gee-community-datasets"
    "/master/community_datasets.json"
)
_CACHE_PATH = Path(__file__).resolve().parent.parent.parent / ".cache" / "gee_community_catalog.json"
_CACHE_TTL_S = 24 * 3600

_catalog: list[dict] | None = None


def _load_catalog() -> list[dict]:
    global _catalog
    if _catalog is not None:
        return _catalog

    if _CACHE_PATH.exists() and (time.time() - _CACHE_PATH.stat().st_mtime) < _CACHE_TTL_S:
        _catalog = json.loads(_CACHE_PATH.read_text())
        return _catalog

    with urllib.request.urlopen(_SOURCE_URL, timeout=20) as resp:  # noqa: S310
        data = json.loads(resp.read())

    _CACHE_PATH.parent.mkdir(parents=True, exist_ok=True)
    _CACHE_PATH.write_text(json.dumps(data))
    _catalog = data
    return _catalog


@tool
def search_community_catalog(query: str, limit: int = 8) -> str:
    """Search the GEE Community Catalog (gee-community-catalog.org) for a
    dataset NOT in Google's own EE Data Catalog. Use this when
    search_full_catalog finds nothing — it covers a lot of ecology/conservation
    data (e.g. shorelines, mangroves, soil, canopy height) contributed by the
    community as ordinary `projects/sat-io/...` EE assets.

    Args:
        query: Free-text keywords, e.g. 'mangrove' or 'canopy height'.
        limit: Max number of results to show (most relevant first).
    """
    try:
        catalog = _load_catalog()
    except Exception as e:  # noqa: BLE001
        return f"Could not load the GEE community catalog (check network access): {e}"

    q = query.lower()
    hits = [
        d for d in catalog
        if q in d.get("title", "").lower()
        or q in d.get("tags", "").lower()
        or q in (d.get("thematic_group") or "").lower()
        or q in d.get("provider", "").lower()
    ]
    if not hits:
        return f"No matches in the GEE community catalog for '{query}'. Try broader or different terms."

    out = []
    for d in hits[:limit]:
        out.append(
            f"- {d.get('id', '?')}  [{d.get('type', '?')}]\n"
            f"    title: {d.get('title', '')}\n"
            f"    provider: {d.get('provider', '')}\n"
            f"    tags: {d.get('tags', '')}\n"
            f"    docs: {d.get('docs', '')}"
        )
    more = f"\n(+{len(hits) - limit} more matches not shown; narrow your query)" if len(hits) > limit else ""
    return "\n".join(out) + more


COMMUNITY_CATALOG_TOOLS = [search_community_catalog]
