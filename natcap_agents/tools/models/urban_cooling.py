"""Wraps scripts/urban_cooling.py's InVEST Urban Cooling model as a crew tool.

The model itself (InvestUrbanCoolingModel) lives in scripts/urban_cooling.py —
this file only adapts it to the tool contract: resolve `region` via
regions.region_geometry (never a raw polygon), supply real Earth Engine input
layers for the region — Dynamic World land cover composited for the EXACT
analysis window (see _dynamic_world_lulc below) plus TerraClimate reference
ET, and MODIS LST for the rural reference temperature / UHI magnitude (see
_reference_climate) — run the model, and publish a map layer + stats rows to
the results board.

Dynamic World (not a static land-cover snapshot like ESA WorldCover) is used
deliberately: comparing two different years means comparing two different
land-cover states too, so each run composites Dynamic World from only that
run's own date range.
"""
from __future__ import annotations

import sys
from pathlib import Path

from smolagents import tool

from ... import results
from ...regions import region_geometry
from ...safety import ensure_ee

# scripts/ lives at the repo root, one level up from natcap_agents/ — add it to
# sys.path so the model's own module (owned/edited independently under
# scripts/) is the single source of truth; we don't fork a copy of it here.
_REPO_ROOT = Path(__file__).resolve().parents[3]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from scripts.urban_cooling import InvestUrbanCoolingModel  # noqa: E402

# Dynamic World's 'label' band: the argmax class per pixel, 0-8.
#   0 water  1 trees  2 grass  3 flooded_vegetation  4 crops
#   5 shrub_and_scrub  6 built  7 bare  8 snow_and_ice
# InVEST biophysical properties per class, following the same reasoning as the
# ESA WorldCover mapping this replaces (kc/shade/albedo per land-cover type;
# 'built' is the only class with building_intensity / green_area=0).
_DYNAMIC_WORLD_BIOPHYSICAL_TABLE = {
    0: {"kc": 1.0, "green_area": 1, "shade": 0.0, "albedo": 0.08, "building_intensity": 0.0},  # water
    1: {"kc": 1.0, "green_area": 1, "shade": 0.9, "albedo": 0.18, "building_intensity": 0.0},  # trees
    2: {"kc": 0.7, "green_area": 1, "shade": 0.1, "albedo": 0.22, "building_intensity": 0.0},  # grass
    3: {"kc": 0.9, "green_area": 1, "shade": 0.3, "albedo": 0.15, "building_intensity": 0.0},  # flooded_vegetation
    4: {"kc": 0.6, "green_area": 1, "shade": 0.1, "albedo": 0.20, "building_intensity": 0.0},  # crops
    5: {"kc": 0.8, "green_area": 1, "shade": 0.4, "albedo": 0.20, "building_intensity": 0.0},  # shrub_and_scrub
    6: {"kc": 0.1, "green_area": 0, "shade": 0.05, "albedo": 0.15, "building_intensity": 0.75},  # built
    7: {"kc": 0.1, "green_area": 0, "shade": 0.0, "albedo": 0.30, "building_intensity": 0.0},  # bare
    8: {"kc": 0.1, "green_area": 0, "shade": 0.0, "albedo": 0.80, "building_intensity": 0.0},  # snow_and_ice
}
_DYNAMIC_WORLD_URBAN_CODE = 6  # built
_DYNAMIC_WORLD_RURAL_CODES = [0, 1, 2, 3, 4, 5, 7, 8]  # everything but built

_DYNAMIC_WORLD_ASSET = "GOOGLE/DYNAMICWORLD/V1"
_TERRACLIMATE_ASSET = "IDAHO_EPSCOR/TERRACLIMATE"
_MODIS_LST_ASSET = "MODIS/061/MOD11A1"
_MODIS_LST_SCALE = 1000  # native resolution (m) of MOD11A1
_DYNAMIC_WORLD_SCALE = 10  # native resolution (m) of Dynamic World


def _dynamic_world_lulc(aoi, start_date: str, end_date: str):
    """Composite Dynamic World's discrete land-cover class ('label' band) over
    [start_date, end_date] for `aoi`, as the per-pixel majority (mode) class
    across that window's Sentinel-2 scenes.

    Returns the composite ee.Image, or None if there are no Dynamic World
    scenes covering this region/window (e.g. a very short or very old date
    range).
    """
    import ee

    col = (
        ee.ImageCollection(_DYNAMIC_WORLD_ASSET)
        .filterBounds(aoi)
        .filterDate(start_date, end_date)
        .select("label")
    )
    if col.size().getInfo() == 0:
        return None
    return col.mode().rename("label")


def _reference_climate(aoi, lulc, start_date: str, end_date: str, rural_buffer_m: float):
    """Derive (t_ref, uhi_max) from MODIS land-surface temperature — no web
    search, no user-supplied guess. t_ref is the mean LST over vegetated/rural
    Dynamic World pixels in a ring buffered outward from `aoi` (the
    surrounding countryside, not the urban area itself); uhi_max is how much
    hotter the hottest built-up pixel inside `aoi` is than that rural
    reference.

    Returns (t_ref, uhi_max) as plain floats, or (None, None) if the window/
    region has no usable MODIS LST or land-cover pixels (e.g. a cloudy period
    or a region with no rural surroundings to reference against).
    """
    import ee

    lst_c = (
        ee.ImageCollection(_MODIS_LST_ASSET)
        .filterDate(start_date, end_date)
        .select("LST_Day_1km")
        .mean()
        .multiply(0.02)
        .subtract(273.15)
        .rename("lst_c")
    )

    rural_mask = lulc.remap(_DYNAMIC_WORLD_RURAL_CODES, [1] * len(_DYNAMIC_WORLD_RURAL_CODES), 0)
    urban_mask = lulc.eq(_DYNAMIC_WORLD_URBAN_CODE)

    rural_ring = aoi.buffer(rural_buffer_m).difference(aoi, ee.ErrorMargin(1))

    scalars = ee.Dictionary({
        "t_ref": lst_c.updateMask(rural_mask).reduceRegion(
            reducer=ee.Reducer.mean(), geometry=rural_ring, scale=_MODIS_LST_SCALE,
            maxPixels=1e9, bestEffort=True,
        ).get("lst_c"),
        "t_urban_max": lst_c.updateMask(urban_mask).reduceRegion(
            reducer=ee.Reducer.max(), geometry=aoi, scale=_MODIS_LST_SCALE,
            maxPixels=1e9, bestEffort=True,
        ).get("lst_c"),
    }).getInfo()

    t_ref = scalars.get("t_ref")
    t_urban_max = scalars.get("t_urban_max")
    if t_ref is None or t_urban_max is None:
        return None, None
    return t_ref, t_urban_max - t_ref


@tool
def urban_cooling(
    region: str,
    region_label: str = "region",
    eto_start_date: str = "2023-06-01",
    eto_end_date: str = "2023-08-31",
    rural_buffer_km: float = 20.0,
    scale: int = 30,
) -> str:
    """InVEST Urban Cooling model: heat mitigation and air temperature over a
    region, from Dynamic World land cover and TerraClimate reference ET.

    Land cover is Dynamic World composited from ONLY this call's own date
    range (not a fixed-year snapshot), so runs for different periods reflect
    that period's actual land cover — comparing two years compares both the
    climate AND how the landscape itself changed between them.

    The rural reference temperature and urban heat island magnitude the model
    needs are derived directly from MODIS land-surface temperature for the
    same date range (rural = vegetated pixels in a ring around the region;
    urban = the hottest built-up pixel inside it) — not looked up or guessed,
    so this is reproducible for any date range without an external search.

    Adds a Heat Mitigation Index (HMI) map layer and records mean HMI / mean
    air temperature as stats-table rows.

    Args:
        region: A region_id from resolve_region(), or a 'west,south,east,north'
            bounding box in degrees. Never pass a raw polygon/GeoJSON here.
        region_label: Short label for the stats-table row and map layer name.
        eto_start_date: Start date (YYYY-MM-DD) of the analysis window (drives
            the Dynamic World land-cover composite, the reference-ET average,
            and the rural/urban LST comparison — all for this exact window).
        eto_end_date: End date (YYYY-MM-DD) of the analysis window.
        rural_buffer_km: How far outside the region to look for a rural LST
            reference (default 20 km). Widen this for a small/dense region
            with little vegetated land nearby.
        scale: Analysis resolution in metres (default 30).
    """
    ensure_ee()
    import ee

    try:
        aoi = region_geometry(region)
    except ValueError as e:
        return str(e)

    lulc = _dynamic_world_lulc(aoi, eto_start_date, eto_end_date)
    if lulc is None:
        return (
            f"No Dynamic World land-cover scenes found for {region_label} in "
            f"{eto_start_date}..{eto_end_date} — try a wider date range."
        )

    t_ref, uhi_max = _reference_climate(aoi, lulc, eto_start_date, eto_end_date, rural_buffer_km * 1000)
    if t_ref is None:
        return (
            f"Could not derive a rural reference temperature for {region_label} in "
            f"{eto_start_date}..{eto_end_date} (no usable MODIS LST / land-cover pixels — "
            "try a wider rural_buffer_km, a different date range, or check the region)."
        )

    ref_eto = (
        ee.ImageCollection(_TERRACLIMATE_ASSET)
        .filterDate(eto_start_date, eto_end_date)
        .select("pet")
        .mean()
        .multiply(0.1)  # TerraClimate PET scale factor -> mm/month
        .divide(30.0)   # -> mm/day
        .rename("eto")
    )

    model = InvestUrbanCoolingModel(
        lulc_image=lulc,
        biophysical_dict=_DYNAMIC_WORLD_BIOPHYSICAL_TABLE,
        ref_eto_image=ref_eto,
        t_ref=t_ref,
        uhi_max=uhi_max,
        scale=float(scale),
        calc_wbgt=False,  # skip work-productivity bands; this tool only reports HMI/t_air
    )

    try:
        out = model.compute(aoi=aoi)
        stats = out.select(["hmi", "t_air"]).reduceRegion(
            reducer=ee.Reducer.mean(), geometry=aoi, scale=scale, maxPixels=1e9, bestEffort=True,
        ).getInfo()
    except Exception as e:  # noqa: BLE001
        return f"Computation failed: {type(e).__name__}: {e}"

    mean_hmi = stats.get("hmi")
    mean_t_air = stats.get("t_air")
    if mean_hmi is None or mean_t_air is None:
        return f"Computation returned no data for {region_label} — region may not overlap the input layers."

    vis = {"bands": ["hmi"], "min": 0, "max": 1, "palette": ["blue", "yellow", "red"]}
    tile_url = out.select("hmi").getMapId(vis)["tile_fetcher"].url_format

    board = results.current()
    board.add_layer(name=f"{region_label}: heat mitigation index", tile_url=tile_url)
    board.add_stat(
        label=region_label, model="urban_cooling", metric="mean_heat_mitigation_index",
        value=round(mean_hmi, 3),
    )
    board.add_stat(
        label=region_label, model="urban_cooling", metric="mean_air_temperature_c",
        value=round(mean_t_air, 2),
    )

    return (
        f"Urban cooling in {region_label}: mean heat mitigation index = {mean_hmi:.3f}, "
        f"mean air temperature = {mean_t_air:.2f} degC "
        f"(t_ref={t_ref:.1f} degC, uhi_max={uhi_max:.1f} degC from MODIS LST; "
        f"InVEST Urban Cooling, Dynamic World + TerraClimate {eto_start_date}..{eto_end_date})."
    )
