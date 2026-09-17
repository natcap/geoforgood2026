"""Shared helpers: data-driven color stretch + one-call map-layer publishing.

A quantity with no natural bound (e.g. green-space supply in m^2/person) can't
use a fixed guessed min/max — most of a region clips to one flat color at
either end (a solid-green map is the symptom). `percentile_range` computes a
real stretch from the data instead; `publish_layer` gets the tile URL for a
band and publishes it to the results board together with legend metadata, so
the app can draw a colorbar that actually matches what's on the map.
"""
from __future__ import annotations

from typing import Any


def percentile_range(
    image, band: str, aoi, scale: float, low: float = 2, high: float = 98,
    fallback: tuple[float, float] = (0.0, 1.0),
) -> tuple[float, float]:
    """Return (lo, hi) as the low/high percentile of `band` over `aoi`.

    Falls back to `fallback` if the region has no valid pixels for this band
    (e.g. it's entirely masked out there).
    """
    import ee

    stats = (
        image.select(band)
        .reduceRegion(
            reducer=ee.Reducer.percentile([low, high]),
            geometry=aoi, scale=scale, maxPixels=1e9, bestEffort=True,
        )
        .getInfo()
    )
    lo = stats.get(f"{band}_p{int(low)}")
    hi = stats.get(f"{band}_p{int(high)}")
    if lo is None or hi is None:
        return fallback
    if lo >= hi:  # degenerate (near-)constant region; widen so the legend isn't a single tick
        hi = lo + 1.0
    return float(lo), float(hi)


def publish_layer(
    board, name: str, image, band: str, palette: list[str], vis_min: float, vis_max: float,
    label: str | None = None, unit: str = "",
) -> str:
    """Get an EE tile URL for `band` and publish it to the results board with
    legend metadata, in one call. Returns the tile URL.
    """
    vis: dict[str, Any] = {"bands": [band], "min": vis_min, "max": vis_max, "palette": palette}
    tile_url = image.select(band).getMapId(vis)["tile_fetcher"].url_format
    board.add_layer(
        name=name, tile_url=tile_url,
        legend={"label": label or name, "min": vis_min, "max": vis_max, "palette": palette, "unit": unit},
    )
    return tile_url
