"""Wraps scripts/urban_flood_risk_mitigation.py's InVEST model as a crew tool.

The model itself (InvestUrbanFloodRiskMitigationModel) lives in
scripts/urban_flood_risk_mitigation.py — this file only adapts it to the tool
contract: resolve `region` via regions.region_geometry (never a raw polygon),
supply real Earth Engine input layers for the region (ESA WorldCover LULC +
a soil hydrologic group raster from the GEE Community Catalog), run the
model over the whole region as one area (no per-watershed breakdown, no
building-damage valuation — see the tool's docstring), and publish a map
layer + stats rows to the results board.
"""
from __future__ import annotations

import sys
from pathlib import Path

from smolagents import tool

from ... import results
from ...regions import region_geometry
from ...safety import ensure_ee
from ..mapping import publish_layer

# scripts/ lives at the repo root, one level up from natcap_agents/ — add it to
# sys.path so the model's own module (owned/edited independently under
# scripts/) is the single source of truth; we don't fork a copy of it here.
_REPO_ROOT = Path(__file__).resolve().parents[3]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from scripts.urban_flood_risk_mitigation import InvestUrbanFloodRiskMitigationModel  # noqa: E402

_WORLDCOVER_ASSET = "ESA/WorldCover/v200"

# Soil hydrologic groups (250 m), from the GEE Community Catalog. Dual-group
# codes (A/D, B/D, C/D) are collapsed to D — the conservative/undrained choice
# for flood work — since InVEST only accepts single groups 1-4.
_SOIL_GROUP_ASSET = "projects/sat-io/open-datasets/HiHydroSoilv2_0/Hydrologic_Soil_Group_250m"
_SOIL_GROUP_REMAP_FROM = [1, 2, 3, 4, 14, 24, 34]
_SOIL_GROUP_REMAP_TO = [1, 2, 3, 4, 4, 4, 4]

# Curve numbers per ESA WorldCover class per soil group (NRCS TR-55, ARC-II,
# approximated) — same table as the script's own run_model() demo. The user
# guide recommends replacing these with study-area-specific values.
_CURVE_NUMBER_TABLE = {
    10: {"cn_a": 30, "cn_b": 55, "cn_c": 70, "cn_d": 77},   # Tree cover
    20: {"cn_a": 30, "cn_b": 48, "cn_c": 65, "cn_d": 73},   # Shrubland
    30: {"cn_a": 39, "cn_b": 61, "cn_c": 74, "cn_d": 80},   # Grassland
    40: {"cn_a": 67, "cn_b": 78, "cn_c": 85, "cn_d": 89},   # Cropland
    50: {"cn_a": 77, "cn_b": 85, "cn_c": 90, "cn_d": 92},   # Built-up (residential, 65% impervious)
    60: {"cn_a": 77, "cn_b": 86, "cn_c": 91, "cn_d": 94},   # Bare/sparse
    70: {"cn_a": 98, "cn_b": 98, "cn_c": 98, "cn_d": 98},   # Snow and ice
    80: {"cn_a": 99, "cn_b": 99, "cn_c": 99, "cn_d": 99},   # Permanent water bodies
    90: {"cn_a": 99, "cn_b": 99, "cn_c": 99, "cn_d": 99},   # Herbaceous wetland
    95: {"cn_a": 99, "cn_b": 99, "cn_c": 99, "cn_d": 99},   # Mangroves
    100: {"cn_a": 49, "cn_b": 69, "cn_c": 79, "cn_d": 84},  # Moss and lichen
}


@tool
def urban_flood_risk_mitigation(
    region: str,
    rainfall_depth_mm: float,
    region_label: str = "region",
    scale: int = 30,
) -> str:
    """InVEST Urban Flood Risk Mitigation model: runoff retention and flood
    volume for a design storm, from ESA WorldCover land cover and a global
    soil hydrologic group raster (Curve Number method).

    Adds a runoff-retention-index map layer (green = retains more of the
    storm, red = floods more) and records the mean retention index plus total
    retention/flood volumes as stats-table rows.

    NOT included in this pass: per-watershed breakdown (InVEST normally
    aggregates over watershed/sewershed polygons; this reports one result for
    the whole region) and building-damage valuation (needs a building
    footprint layer and a currency damage-per-type table this tool doesn't
    have a principled default for — ask if you need that added).

    Args:
        region: A region_id from resolve_region(), or a 'west,south,east,north'
            bounding box in degrees. Never pass a raw polygon/GeoJSON here.
        rainfall_depth_mm: Design storm depth in mm — a deliberate input
            choice (e.g. a 10-year/24-hour storm for the area), not something
            to guess; ask the user or a rainfall-frequency reference if unsure.
        region_label: Short label for the stats-table row and map layer name.
        scale: Analysis resolution in metres (default 30, matching WorldCover).
    """
    ensure_ee()
    import ee

    try:
        aoi = region_geometry(region)
    except ValueError as e:
        return str(e)

    lulc = ee.ImageCollection(_WORLDCOVER_ASSET).first().select("Map")
    soil_group = (
        ee.Image(_SOIL_GROUP_ASSET)
        .remap(_SOIL_GROUP_REMAP_FROM, _SOIL_GROUP_REMAP_TO)
        .rename("soil_group")
    )

    model = InvestUrbanFloodRiskMitigationModel(
        lulc_image=lulc,
        biophysical_dict=_CURVE_NUMBER_TABLE,
        soil_group_image=soil_group,
        rainfall_depth=rainfall_depth_mm,
        scale=float(scale),
    )

    try:
        missing = model.missing_lucodes(aoi)
    except Exception as e:  # noqa: BLE001
        missing = None  # non-fatal — the check itself failing shouldn't block the run
        missing_note = f" (missing-lucode check failed: {type(e).__name__}: {e})"
    else:
        missing_note = (
            f" NOTE: land-cover codes {missing} in this region have no curve-number entry and "
            "were excluded from the result, which may undercount it." if missing else ""
        )

    try:
        out = model.compute(aoi=aoi)
        stats = out.select(["runoff_retention_index", "runoff_retention_m3", "q_m3"]).reduceRegion(
            reducer=ee.Reducer.mean().combine(reducer2=ee.Reducer.sum(), sharedInputs=True),
            geometry=aoi, scale=scale, maxPixels=1e9, bestEffort=True,
        ).getInfo()
    except Exception as e:  # noqa: BLE001
        return f"Computation failed: {type(e).__name__}: {e}"

    retention_index = stats.get("runoff_retention_index_mean")
    retention_m3 = stats.get("runoff_retention_m3_sum")
    flood_m3 = stats.get("q_m3_sum")
    if retention_index is None or retention_m3 is None or flood_m3 is None:
        return f"Computation returned no data for {region_label} — region may not overlap the input layers."

    board = results.current()
    publish_layer(
        board, name=f"{region_label}: runoff retention index", image=out, band="runoff_retention_index",
        palette=["red", "yellow", "green"], vis_min=0, vis_max=1,
        label="Runoff retention index",
    )
    board.add_stat(
        label=region_label, model="urban_flood_risk_mitigation", metric="mean_runoff_retention_index",
        value=round(retention_index, 3),
    )
    board.add_stat(
        label=region_label, model="urban_flood_risk_mitigation", metric="total_retention_volume_m3",
        value=round(retention_m3, 1),
    )
    board.add_stat(
        label=region_label, model="urban_flood_risk_mitigation", metric="total_flood_volume_m3",
        value=round(flood_m3, 1),
    )

    return (
        f"Urban flood risk in {region_label} for a {rainfall_depth_mm:.0f} mm design storm: "
        f"mean runoff retention index = {retention_index:.3f}, "
        f"total retention volume = {retention_m3:,.0f} m^3, "
        f"total flood (runoff) volume = {flood_m3:,.0f} m^3 "
        f"(InVEST Urban Flood Risk Mitigation, ESA WorldCover + HiHydroSoil)."
        f"{missing_note}"
    )
