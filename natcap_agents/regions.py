"""Server-side region registry.

Turns a place name into a small, reusable handle instead of ever handing a
full GeoJSON polygon to a model. A country/park boundary can be thousands of
vertices — round-tripping that through a tool call argument bloats every later
turn in the crew's context (and can blow past a function-call argument size
limit). `resolve_region()` looks a name up, keeps the actual ee.Geometry
server-side, and returns a short region_id (plus a compact bbox for context).
Pass that region_id — or a plain bounding box — to any tool that takes
`region`; `region_geometry()` is what those tools call to resolve it.
"""
from __future__ import annotations

import itertools
import json

from smolagents import tool

from .safety import ensure_ee

_COUNTER = itertools.count(1)
_REGISTRY: dict[str, object] = {}  # region_id -> ee.Geometry

# (name property, EE asset id, human label), searched in this order.
_SOURCES = [
    ("ADM2_NAME", "FAO/GAUL/2015/level2", "district/county"),
    ("ADM1_NAME", "FAO/GAUL/2015/level1", "state/province"),
    ("ADM0_NAME", "FAO/GAUL/2015/level0", "country"),
    ("NAME", "WCMC/WDPA/current/polygons", "protected area"),
]

_MAX_INLINE_REGION_CHARS = 2000  # a plain rectangle easily fits; a real polygon won't


def _bbox_deg(geometry) -> list[float] | None:
    try:
        coords = geometry.bounds(1).coordinates().getInfo()[0]
    except Exception:  # noqa: BLE001
        return None
    lons = [c[0] for c in coords]
    lats = [c[1] for c in coords]
    return [round(min(lons), 3), round(min(lats), 3), round(max(lons), 3), round(max(lats), 3)]


@tool
def resolve_region(query: str, max_results: int = 3) -> str:
    """Look up a place name and register its boundary server-side, WITHOUT
    returning the polygon itself (a country/park boundary can be thousands of
    vertices). Returns one short region_id per match plus a compact
    description (name, kind, bounding box in degrees) — pass that region_id to
    any tool that takes `region`.

    Searches, in order: districts/counties, states/provinces, countries (FAO
    GAUL), and protected areas (WDPA).

    Args:
        query: Place name to search for, e.g. 'Yosemite' or 'Kenya' or 'Amazonas'.
        max_results: Max number of matches to return (default 3).
    """
    ensure_ee()
    import ee

    hits: list[str] = []
    for name_field, asset_id, kind in _SOURCES:
        if len(hits) >= max_results:
            break
        fc = ee.FeatureCollection(asset_id)
        matches = fc.filter(ee.Filter.stringContains(name_field, query))
        try:
            n = matches.limit(max_results - len(hits)).size().getInfo()
        except Exception:  # noqa: BLE001
            continue
        if not n:
            continue
        feats = matches.limit(max_results - len(hits)).toList(max_results - len(hits)).getInfo()
        for f in feats:
            name = (f.get("properties") or {}).get(name_field, query)
            geom = ee.Feature(f).geometry()
            region_id = f"R{next(_COUNTER)}"
            _REGISTRY[region_id] = geom
            bbox = _bbox_deg(geom)
            hits.append(f"- region_id={region_id}  name={name}  kind={kind}  bbox_deg={bbox}")

    if not hits:
        return (
            f"No match for '{query}' in districts/states/countries/protected areas. "
            "Try a different spelling or a broader/narrower name, or pass a bounding box "
            "'west,south,east,north' (degrees) directly as `region` instead."
        )
    return "\n".join(hits)


def region_geometry(region: str):
    """Resolve a `region` argument to a live ee.Geometry.

    Accepts, in this order: a region_id returned by resolve_region(); a plain
    'west,south,east,north' bounding box in degrees; or (as a last resort) a
    small GeoJSON geometry/Feature/FeatureCollection string. Raises ValueError
    on anything implausibly large or unrecognized, so a bulky polygon never
    silently round-trips through a tool call.
    """
    import ee

    region = region.strip()

    if region in _REGISTRY:
        return _REGISTRY[region]

    parts = region.split(",")
    if len(parts) == 4:
        try:
            w, s, e, n = (float(p) for p in parts)
            return ee.Geometry.Rectangle([w, s, e, n])
        except ValueError:
            pass  # not a bbox; fall through

    if len(region) > _MAX_INLINE_REGION_CHARS:
        raise ValueError(
            f"`region` is {len(region)} chars — too large to pass directly. Call "
            "resolve_region(query) instead and pass its region_id, or use a bounding "
            "box 'west,south,east,north'."
        )

    try:
        geo = json.loads(region)
    except Exception as e:  # noqa: BLE001
        raise ValueError(
            f"Unrecognized `region` '{region[:60]}': not a known region_id, a "
            "'west,south,east,north' bbox, or valid GeoJSON."
        ) from e

    if geo.get("type") == "FeatureCollection":
        return ee.FeatureCollection(geo).geometry()
    if geo.get("type") == "Feature":
        return ee.Feature(geo).geometry()
    return ee.Geometry(geo)


REGION_TOOLS = [resolve_region]
