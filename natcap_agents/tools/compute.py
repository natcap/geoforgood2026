"""Generic Earth Engine helper tools.

Encapsulate the two operations that are (a) common and (b) easy to get subtly
wrong: pulling a scalar/statistic back to the client, and producing a viewable
layer. Both guard against a huge accidental `.getInfo()`, and both publish
their result to the shared `results` board — the app's map and stats table
pick it up automatically, with no extra wiring from the agent.

`region` is always a small handle — a region_id from resolve_region(), or a
'west,south,east,north' bbox — never a raw polygon (see ../regions.py for why).
"""
from __future__ import annotations

import json

from smolagents import tool

from .. import results
from ..regions import region_geometry
from ..safety import ensure_ee


def _resolve_image(asset_id: str):
    import ee
    asset = ee.data.getAsset(asset_id)
    atype = asset.get("type", "")
    return ee.ImageCollection(asset_id).median() if "COLLECTION" in atype.upper() else ee.Image(asset_id)


@tool
def compute_region_stats(
    asset_id: str,
    band: str,
    region: str,
    reducer: str = "mean",
    scale: int = 100,
    label: str = "",
) -> str:
    """Reduce one band of an image over a region, return the scalar result, and
    record it as a row in the app's stats table.

    Args:
        asset_id: Image or ImageCollection id. Collections are median-composited.
        band: Band name to reduce (must exist on the asset).
        region: A region_id from resolve_region(), or a 'west,south,east,north'
            bounding box in degrees. Never pass a raw polygon/GeoJSON here.
        reducer: One of 'mean', 'sum', 'min', 'max', 'median', 'count'.
        scale: Pixel scale in metres for the reduction (default 100).
        label: Short label for this row (e.g. the region's name); defaults to
            the asset id.
    """
    ensure_ee()
    import ee

    reducers = {
        "mean": ee.Reducer.mean(), "sum": ee.Reducer.sum(),
        "min": ee.Reducer.min(), "max": ee.Reducer.max(),
        "median": ee.Reducer.median(), "count": ee.Reducer.count(),
    }
    if reducer not in reducers:
        return f"Unknown reducer '{reducer}'. Options: {', '.join(reducers)}."

    try:
        geom = region_geometry(region)
    except ValueError as e:
        return str(e)

    try:
        img = _resolve_image(asset_id).select(band)
        val = img.reduceRegion(
            reducer=reducers[reducer], geometry=geom, scale=scale, maxPixels=1e10, bestEffort=True,
        ).get(band).getInfo()
    except Exception as e:  # noqa: BLE001
        return f"Computation failed: {type(e).__name__}: {e}"

    results.current().add_stat(
        label=label or asset_id, asset=asset_id, band=band, reducer=reducer, scale_m=scale, value=val,
    )
    return f"{reducer}({asset_id}:{band}) over region at {scale} m = {val}"


@tool
def get_map_tiles(asset_id: str, vis_params_json: str, name: str = "") -> str:
    """Generate an XYZ tile URL for an Earth Engine layer and add it to the app's map.

    Args:
        asset_id: Image or ImageCollection id (collections are median-composited).
        vis_params_json: Visualization params as a JSON string, e.g.
            '{"bands":["B4","B3","B2"],"min":0,"max":3000}'.
        name: Layer name shown in the map's legend; defaults to asset_id.
    """
    ensure_ee()

    try:
        vis = json.loads(vis_params_json)
    except Exception as e:  # noqa: BLE001
        return f"Bad vis_params_json: {e}"

    try:
        img = _resolve_image(asset_id)
        m = img.getMapId(vis)
        tile_url = m["tile_fetcher"].url_format
    except Exception as e:  # noqa: BLE001
        return f"Tile generation failed: {type(e).__name__}: {e}"

    layer_name = name or asset_id
    results.current().add_layer(name=layer_name, tile_url=tile_url)
    return f"Added layer '{layer_name}' to the map. tile_url_template: {tile_url}"


COMPUTE_TOOLS = [compute_region_stats, get_map_tiles]
