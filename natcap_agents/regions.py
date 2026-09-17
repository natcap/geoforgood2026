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

# (name property, EE asset id, human label, [context fields checked out to
# country/state]), searched in this order. Context fields let a query like
# 'San Francisco, USA' disambiguate a common place name (there are dozens of
# San Franciscos across Latin America and the Philippines) by also requiring
# a country/state match, not just the place name.
_SOURCES = [
    ("ADM2_NAME", "FAO/GAUL/2015/level2", "district/county", ["ADM1_NAME", "ADM0_NAME"]),
    ("ADM1_NAME", "FAO/GAUL/2015/level1", "state/province", ["ADM0_NAME"]),
    ("ADM0_NAME", "FAO/GAUL/2015/level0", "country", []),
    ("NAME", "WCMC/WDPA/current/polygons", "protected area", []),
]


def _split_query(query: str) -> tuple[str, str | None]:
    """Split 'Place, Context' (e.g. 'San Francisco, California') into (place,
    context); returns (query, None) if there's no comma."""
    if "," in query:
        place, _, context = query.partition(",")
        return place.strip(), context.strip() or None
    return query.strip(), None


def _case_variants(text: str) -> list[str]:
    """A handful of plausible casings to try. ee.Filter.stringContains is
    case-sensitive and admin-boundary datasets aren't consistently cased, but
    normalizing case by transforming the whole collection (.map() over every
    feature before filtering) is what made lookups slow — WDPA alone has
    ~280,000 features, and a per-feature map() blocks Earth Engine's filter
    pushdown/indexing. ORing a few literal-cased filters together stays cheap
    and lets EE optimize normally.
    """
    candidates = [text, text.title(), text.upper(), text.lower()]
    return list(dict.fromkeys(c for c in candidates if c))


# Common abbreviations that never literally appear inside these datasets' full
# country names (GAUL spells it 'United States of America', so a context of
# 'USA' would otherwise never match via substring) — expanded into an extra
# search term alongside the literal casings above.
_COUNTRY_ALIASES = {
    "usa": "United States of America",
    "us": "United States of America",
    "u.s.": "United States of America",
    "u.s.a.": "United States of America",
    "uk": "United Kingdom",
    "uae": "United Arab Emirates",
    "drc": "Democratic Republic of the Congo",
    "car": "Central African Republic",
}


def _context_terms(context: str) -> list[str]:
    terms = _case_variants(context)
    alias = _COUNTRY_ALIASES.get(context.strip().lower())
    if alias:
        terms += _case_variants(alias)
    return list(dict.fromkeys(terms))


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
    description (name, country/state, kind, bounding box in degrees) — pass
    that region_id to any tool that takes `region`.

    Matching tries a few common casings (Title Case, UPPER, lower) rather than
    requiring exact case. Many place names repeat worldwide (there are dozens
    of towns named 'San Francisco' across Latin America and the Philippines,
    for instance) — if a plain name returns the wrong country, or might,
    disambiguate with 'Place, Context', e.g. 'San Francisco, USA' or
    'San Francisco, California'; the context is matched against the state/
    country the place is in (common abbreviations like 'USA'/'UK' are
    recognized). Always check the country/state on a hit before trusting it
    for a well-known place with a common name.

    Searches, in order: districts/counties, states/provinces, countries (FAO
    GAUL), and protected areas (WDPA).

    Args:
        query: Place name to search for, optionally 'Place, Context' to
            disambiguate, e.g. 'Yosemite', 'Kenya', or 'San Francisco, USA'.
        max_results: Max number of matches to return (default 3).
    """
    ensure_ee()
    import ee

    place, context = _split_query(query)
    place_terms = _case_variants(place)

    hits: list[str] = []
    for name_field, asset_id, kind, context_fields in _SOURCES:
        if len(hits) >= max_results:
            break

        row_filter = ee.Filter.Or(*[ee.Filter.stringContains(name_field, t) for t in place_terms])
        if context and context_fields:
            context_terms = _context_terms(context)
            row_filter = ee.Filter.And(row_filter, ee.Filter.Or(*[
                ee.Filter.stringContains(cf, t) for cf in context_fields for t in context_terms
            ]))

        matches = ee.FeatureCollection(asset_id).filter(row_filter)
        limit = max_results - len(hits)
        try:
            n = matches.limit(limit).size().getInfo()
        except Exception:  # noqa: BLE001
            continue
        if not n:
            continue
        feats = matches.limit(limit).toList(limit).getInfo()
        for f in feats:
            props = f.get("properties") or {}
            name = props.get(name_field, place)
            where = " / ".join(props[cf] for cf in reversed(context_fields) if props.get(cf))
            geom = ee.Feature(f).geometry()
            region_id = f"R{next(_COUNTER)}"
            _REGISTRY[region_id] = geom
            bbox = _bbox_deg(geom)
            label = f"{name}, {where}" if where else name
            hits.append(f"- region_id={region_id}  name={label}  kind={kind}  bbox_deg={bbox}")

    if not hits:
        msg = f"No match for '{query}' in districts/states/countries/protected areas."
        if context:
            msg += f" (searched for '{place}' with context '{context}')"
        msg += (
            " Try a different spelling or a broader/narrower name, add a country/state for "
            "disambiguation ('Place, Context'), or pass a bounding box 'west,south,east,north' "
            "(degrees) directly as `region` instead."
        )
        return msg
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
