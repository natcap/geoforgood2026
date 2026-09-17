"""Shared helper: composite Dynamic World land cover for an exact date range.

Used by any model tool that needs land cover matching a specific analysis
window — a fixed-year snapshot (like ESA WorldCover) can't tell two different
periods' land cover apart, which matters whenever a model is run once per
period and compared (e.g. summer 2022 vs. summer 2023): the comparison should
reflect both the climate/population difference AND how the landscape itself
changed between them.
"""
from __future__ import annotations

ASSET_ID = "GOOGLE/DYNAMICWORLD/V1"
NATIVE_SCALE = 10  # metres

# Dynamic World's 'label' band: the argmax class per pixel, 0-8.
#   0 water  1 trees  2 grass  3 flooded_vegetation  4 crops
#   5 shrub_and_scrub  6 built  7 bare  8 snow_and_ice
URBAN_CODE = 6  # built
RURAL_CODES = [0, 1, 2, 3, 4, 5, 7, 8]  # everything but built


def composite_lulc(aoi, start_date: str, end_date: str):
    """Composite Dynamic World's discrete land-cover class ('label' band) over
    [start_date, end_date] for `aoi`, as the per-pixel majority (mode) class
    across that window's Sentinel-2 scenes.

    Returns the composite ee.Image, or None if there are no Dynamic World
    scenes covering this region/window (e.g. a very short or very old date
    range).
    """
    import ee

    col = (
        ee.ImageCollection(ASSET_ID)
        .filterBounds(aoi)
        .filterDate(start_date, end_date)
        .select("label")
    )
    if col.size().getInfo() == 0:
        return None
    return col.mode().rename("label")
