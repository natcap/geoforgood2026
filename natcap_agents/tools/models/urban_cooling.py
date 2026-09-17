"""Wraps scripts/urban_cooling.py's InVEST Urban Cooling model as a crew tool.

The model itself (InvestUrbanCoolingModel) lives in scripts/urban_cooling.py —
this file only adapts it to the tool contract: resolve `region` via
regions.region_geometry (never a raw polygon), supply real Earth Engine input
layers for the region — Dynamic World land cover composited for the EXACT
analysis window (see ../dynamic_world.py) plus TerraClimate reference ET, and
MODIS LST for the rural reference temperature / UHI magnitude (see
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
from ..dynamic_world import URBAN_CODE as _DW_URBAN_CODE
from ..dynamic_world import composite_lulc as _dynamic_world_lulc
from ..mapping import local_utm_crs, publish_layer

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

_TERRACLIMATE_ASSET = "IDAHO_EPSCOR/TERRACLIMATE"
_MODIS_LST_ASSET = "MODIS/061/MOD11A1"
_MODIS_LST_SCALE = 1000  # native resolution (m) of MOD11A1

# MODIS daytime LST is masked out over open water (and unreliable over
# flooded_vegetation/snow), so "rural = everything but built" — fine for the
# general urban/rural split elsewhere — silently starves the reference
# temperature of valid pixels in a water-heavy region (a coastal or bay-side
# AOI). Use only classes MODIS LST actually retrieves over.
_LST_LAND_CODES = [1, 2, 4, 5, 7]  # trees, grass, crops, shrub_and_scrub, bare

# If the first rural ring (radius = rural_buffer_km) has no valid land pixels
# — plausible when the surrounding area is also mostly water — widen it
# before giving up.
_RURAL_BUFFER_RETRY_MULTIPLIERS = (1, 2, 4)


def _reference_climate(aoi, lulc, start_date: str, end_date: str, rural_buffer_m: float):
    """Derive (t_ref, uhi_max) from MODIS land-surface temperature — no web
    search, no user-supplied guess. t_ref is the mean LST over well-observed
    land pixels (trees/grass/crops/shrub/bare — not water, which MODIS LST
    doesn't retrieve over) in a ring buffered outward from `aoi` (the
    surrounding countryside); uhi_max is how much hotter the hottest built-up
    pixel inside `aoi` is than that rural reference (or, if `aoi` has no
    built-up pixels at all, the hottest well-observed land pixel in `aoi`).

    Returns (t_ref, uhi_max, notes) as (float, float, list[str]), or
    (None, None, notes) if no usable MODIS LST / land-cover pixels were found
    even after widening the search ring. `notes` records any fallback taken,
    for the caller to surface to the user.
    """
    import ee

    notes: list[str] = []

    lst_c = (
        ee.ImageCollection(_MODIS_LST_ASSET)
        .filterDate(start_date, end_date)
        .select("LST_Day_1km")
        .mean()
        .multiply(0.02)
        .subtract(273.15)
        .rename("lst_c")
    )

    land_mask = lulc.remap(_LST_LAND_CODES, [1] * len(_LST_LAND_CODES), 0)
    built_mask = lulc.eq(_DW_URBAN_CODE)
    lst_land = lst_c.updateMask(land_mask)

    t_ref = None
    for multiplier in _RURAL_BUFFER_RETRY_MULTIPLIERS:
        rural_ring = aoi.buffer(rural_buffer_m * multiplier).difference(aoi, ee.ErrorMargin(1))
        t_ref = lst_land.reduceRegion(
            reducer=ee.Reducer.mean(), geometry=rural_ring, scale=_MODIS_LST_SCALE,
            maxPixels=1e9, bestEffort=True,
        ).get("lst_c").getInfo()
        if t_ref is not None:
            if multiplier != 1:
                notes.append(
                    f"widened the rural search ring to {multiplier}x rural_buffer_km to find "
                    "enough land (this area's surroundings are water-heavy)"
                )
            break

    if t_ref is None:
        return None, None, notes

    t_urban_max = lst_c.updateMask(built_mask).reduceRegion(
        reducer=ee.Reducer.max(), geometry=aoi, scale=_MODIS_LST_SCALE,
        maxPixels=1e9, bestEffort=True,
    ).get("lst_c").getInfo()

    if t_urban_max is None:
        # No built-up pixels in the AOI (e.g. it's mostly water with little or
        # no urban footprint) — fall back to the hottest well-observed land
        # pixel there instead of failing outright.
        t_urban_max = lst_land.reduceRegion(
            reducer=ee.Reducer.max(), geometry=aoi, scale=_MODIS_LST_SCALE,
            maxPixels=1e9, bestEffort=True,
        ).get("lst_c").getInfo()
        if t_urban_max is not None:
            notes.append(
                "no built-up pixels found in the region; used its hottest land pixel instead "
                "of a true urban/built reference for uhi_max"
            )

    if t_urban_max is None:
        return None, None, notes

    return t_ref, t_urban_max - t_ref, notes


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

    # `crs` fixes the composite to a local metric grid (aoi's own UTM zone) at
    # exactly `scale` metres/pixel — required before the InVEST model builds
    # and convolves its own decay kernels over it. Without this, a reducer
    # composite has no defined native projection, so the kernels' real-world
    # size would depend on whatever resolution a given request implies (e.g.
    # the current map zoom level for interactive tile rendering) instead of
    # staying fixed. See mapping.local_utm_crs.
    lulc = _dynamic_world_lulc(aoi, eto_start_date, eto_end_date, scale=scale, crs=local_utm_crs(aoi))
    if lulc is None:
        return (
            f"No Dynamic World land-cover scenes found for {region_label} in "
            f"{eto_start_date}..{eto_end_date} — try a wider date range."
        )

    t_ref, uhi_max, climate_notes = _reference_climate(
        aoi, lulc, eto_start_date, eto_end_date, rural_buffer_km * 1000,
    )
    if t_ref is None:
        return (
            f"Could not derive a rural reference temperature for {region_label} in "
            f"{eto_start_date}..{eto_end_date} — no usable MODIS land-surface-temperature "
            "pixels even after widening the rural search ring (common for a region that's "
            "mostly water). Try a much larger rural_buffer_km, a different date range, or "
            "check the region."
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

    board = results.current()
    publish_layer(
        board, name=f"{region_label}: heat mitigation index", image=out, band="hmi",
        palette=["blue", "yellow", "red"], vis_min=0, vis_max=1,
        label="Heat mitigation index",
    )
    board.add_stat(
        label=region_label, model="urban_cooling", metric="mean_heat_mitigation_index",
        value=round(mean_hmi, 3),
    )
    board.add_stat(
        label=region_label, model="urban_cooling", metric="mean_air_temperature_c",
        value=round(mean_t_air, 2),
    )

    notes_suffix = f" NOTE: {'; '.join(climate_notes)}." if climate_notes else ""
    return (
        f"Urban cooling in {region_label}: mean heat mitigation index = {mean_hmi:.3f}, "
        f"mean air temperature = {mean_t_air:.2f} degC "
        f"(t_ref={t_ref:.1f} degC, uhi_max={uhi_max:.1f} degC from MODIS LST; "
        f"InVEST Urban Cooling, Dynamic World + TerraClimate {eto_start_date}..{eto_end_date})."
        f"{notes_suffix}"
    )
