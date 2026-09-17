"""Dataset discovery/verification tools.

Generic, no domain curation — any model tool (or the orchestrator writing ad
hoc `ee` code) that needs an Earth Engine asset ID should verify it here first,
since a model that invents an ID or a stale version silently produces wrong
results.
"""
from __future__ import annotations

from smolagents import tool

from ..safety import ensure_ee


@tool
def search_full_catalog(query: str, limit: int = 8) -> str:
    """Search the full live Earth Engine public data catalog (~1000+ datasets).

    Args:
        query: Free-text keywords, e.g. 'soil moisture' or 'nighttime lights'.
        limit: Max number of results to show (most relevant first).
    """
    try:
        import geemap
    except Exception as e:  # noqa: BLE001
        return f"geemap not available: {e}"

    try:
        hits = geemap.search_ee_data(query)
    except Exception as e:  # noqa: BLE001
        return f"Catalog search failed (check network access): {e}"

    if not hits:
        return f"No matches in the EE catalog for '{query}'. Try broader or different terms."

    out = []
    for d in hits[:limit]:
        out.append(
            f"- {d.get('id', '?')}  [{d.get('type', '?')}]\n"
            f"    title: {d.get('title', '')}\n"
            f"    provider: {d.get('provider', '')}\n"
            f"    dates: {d.get('dates', '')}"
        )
    more = f"\n(+{len(hits) - limit} more matches not shown; narrow your query)" if len(hits) > limit else ""
    return "\n".join(out) + more


@tool
def verify_asset(asset_id: str) -> str:
    """Confirm an Earth Engine asset EXISTS and return its real type and band/property names.

    Always call this before finalizing an asset ID picked from memory or search
    results — versions/ids drift over time.

    Args:
        asset_id: Exact Earth Engine asset id to verify against live EE.
    """
    ensure_ee()
    import re
    import warnings

    import ee

    successor = None
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        try:
            meta = ee.data.getAsset(asset_id)
        except Exception as e:  # noqa: BLE001
            return f"NOT FOUND / not accessible: {asset_id}\n  ({type(e).__name__}: {e})"

        atype = meta.get("type", "UNKNOWN")
        lines = [f"EXISTS: {asset_id}", f"  type: {atype}"]
        try:
            if atype in ("IMAGE_COLLECTION", "ImageCollection"):
                col = ee.ImageCollection(asset_id)
                first = ee.Image(col.first())
                lines.append(f"  bands: {first.bandNames().getInfo()}")
            elif atype in ("IMAGE", "Image"):
                lines.append(f"  bands: {ee.Image(asset_id).bandNames().getInfo()}")
            elif atype in ("TABLE", "FeatureCollection"):
                fc = ee.FeatureCollection(asset_id)
                props = ee.Feature(fc.first()).propertyNames().getInfo()
                lines.append(f"  properties: {props}")
        except Exception as e:  # noqa: BLE001
            lines.append(f"  (exists but band/property probe failed: {e})")

        for w in caught:
            text = str(w.message)
            if "deprecat" in text.lower() or "superseded" in text.lower():
                m = re.search(r"superseded by\s+(\S+)", text)
                successor = m.group(1).rstrip(".") if m else None
                break

    if successor:
        lines.insert(1, f"  DEPRECATED: use the successor '{successor}' instead of this id.")
    return "\n".join(lines)


CATALOG_TOOLS = [search_full_catalog, verify_asset]
