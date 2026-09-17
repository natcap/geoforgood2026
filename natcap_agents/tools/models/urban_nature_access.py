"""Wraps scripts/urban_nature_access.py's InVEST model as a crew tool.

The model itself (UrbanNatureAccessGEE, a 2SFCA green-space supply/demand
model) lives in scripts/urban_nature_access.py — this file only adapts it to
the tool contract: resolve `region` via regions.region_geometry (never a raw
polygon), supply real Earth Engine input layers for the region — Dynamic
World land cover composited for the EXACT analysis window (see
../dynamic_world.py) plus the nearest available WorldPop population year —
run the model, and publish a map layer + stats rows to the results board.

Dynamic World (not a static land-cover snapshot like ESA WorldCover) is used
deliberately: comparing two different years means comparing two different
land-cover states too, so each run composites Dynamic World from only that
run's own date range. A single `start_date`/`end_date` pair drives both the
land-cover composite AND the population-year lookup — there's no separate
`population_year` to keep in sync by hand.
"""
from __future__ import annotations

import sys
from pathlib import Path

from smolagents import tool

from ... import results
from ...regions import region_geometry
from ...safety import ensure_ee
from ..dynamic_world import composite_lulc
from ..mapping import local_utm_crs, percentile_range, publish_layer

# scripts/ lives at the repo root, one level up from natcap_agents/ — add it to
# sys.path so the model's own module (owned/edited independently under
# scripts/) is the single source of truth; we don't fork a copy of it here.
_REPO_ROOT = Path(__file__).resolve().parents[3]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from scripts.urban_nature_access import UrbanNatureAccessGEE  # noqa: E402

_WORLDPOP_ASSET = "WorldPop/GP/100m/pop"  # annual global mosaics, ~2000-2020

# Dynamic World class -> fraction counted as usable "nature" for the 2SFCA
# supply calculation. Same reasoning as the script's own ESA WorldCover
# weights (trees/shrub/grass/wetland score highest, built lowest), recoded to
# Dynamic World's 9 classes (0 water 1 trees 2 grass 3 flooded_vegetation
# 4 crops 5 shrub_and_scrub 6 built 7 bare 8 snow_and_ice).
_NATURE_WEIGHTS = {
    0: 0.5,   # water
    1: 1.0,   # trees
    2: 0.8,   # grass
    3: 1.0,   # flooded_vegetation (wetland-like)
    4: 0.2,   # crops
    5: 0.9,   # shrub_and_scrub
    6: 0.05,  # built
    7: 0.0,   # bare
    8: 0.0,   # snow_and_ice
}


def _population_image_for_year(aoi, requested_year: int):
    """Pick the WorldPop image for `requested_year`, clamped to whatever years
    this collection actually has for `aoi` (it stops around 2020, so a 2023
    request lands on the latest available year instead of finding nothing).

    Returns (image, used_year), or (None, None) if the collection has no
    coverage for this region at all.
    """
    import ee

    col = ee.ImageCollection(_WORLDPOP_ASSET).filterBounds(aoi)
    years = ee.List(col.aggregate_array("system:time_start")).map(
        lambda t: ee.Date(t).get("year")
    )
    bounds = ee.Dictionary({
        "min": years.reduce(ee.Reducer.min()),
        "max": years.reduce(ee.Reducer.max()),
    }).getInfo()
    min_year, max_year = bounds.get("min"), bounds.get("max")
    if min_year is None or max_year is None:
        return None, None

    use_year = max(int(min_year), min(requested_year, int(max_year)))
    # This collection is ONE IMAGE PER COUNTRY PER YEAR (its own STAC schema
    # lists 'country' as a per-image property) — not a single global mosaic.
    # `.first()` grabs an arbitrary one of however many country-tiles overlap
    # `aoi`, which silently produces a partly- or entirely-masked image for
    # any region straddling (or near) a tile boundary — e.g. Singapore, which
    # sits right against Malaysia's/Indonesia's tiles, or a large country like
    # the US that may itself be split across multiple tiles. `.mosaic()`
    # combines every matching tile into one seamless image instead.
    image = col.filter(ee.Filter.calendarRange(use_year, use_year, "year")).mosaic()
    return image, use_year


def _mask_coverage(image, band: str, aoi, scale: float) -> float | None:
    """Fraction (0-1) of `aoi` where `band` has a valid (non-masked) pixel.
    `.mask()` is itself always fully defined (0 or 1, never masked), so its
    mean over the region is exactly the valid-data fraction — a cheap way to
    tell whether a "no data"/patchy result is a real gap in the source data
    (e.g. persistent cloud/fog leaving Dynamic World unclassified for the
    whole date window) rather than a bug in this tool.
    """
    import ee

    frac = image.select(band).mask().reduceRegion(
        reducer=ee.Reducer.mean(), geometry=aoi, scale=scale, maxPixels=1e9, bestEffort=True,
    ).get(band).getInfo()
    return frac


@tool
def urban_nature_access(
    region: str,
    region_label: str = "region",
    start_date: str = "2023-06-01",
    end_date: str = "2023-08-31",
    search_radius_m: float = 500.0,
    decay_type: str = "exponential",
    demand_per_capita_m2: float = 30.0,
    scale: int = 30,
) -> str:
    """InVEST Urban Nature Access model (2SFCA): per-capita green-space supply
    and demand balance for a region, from Dynamic World land cover and
    WorldPop population.

    Land cover is Dynamic World composited from ONLY this call's own date
    range (not a fixed-year snapshot), so runs for different periods reflect
    that period's actual land cover. Population uses the WorldPop year closest
    to `start_date` (the dataset stops around 2020, so a more recent date
    falls back to the latest year available, noted in the result).

    Adds a per-capita nature-supply map layer (red = undersupplied, green =
    meets or exceeds demand_per_capita_m2) and records total population, total
    nature area, mean supply per capita, and the percent of population
    undersupplied as stats-table rows.

    Args:
        region: A region_id from resolve_region(), or a 'west,south,east,north'
            bounding box in degrees. Never pass a raw polygon/GeoJSON here.
        region_label: Short label for the stats-table row and map layer name.
        start_date: Start date (YYYY-MM-DD) of the analysis window (drives the
            Dynamic World land-cover composite and the WorldPop year lookup).
        end_date: End date (YYYY-MM-DD) of the analysis window.
        search_radius_m: Pedestrian catchment radius in metres (default 500).
        decay_type: Spatial impedance function: 'dichotomy', 'exponential',
            'gaussian', or 'density' (default 'exponential').
        demand_per_capita_m2: Required green-space area per person, in square
            metres (default 30).
        scale: Analysis resolution in metres (default 30). The model convolves
            a kernel sized search_radius_m / scale pixels across a raster of
            this resolution — a city-sized region at a much finer scale can be
            very slow; widen `scale` for a large area.
    """
    ensure_ee()

    try:
        aoi = region_geometry(region)
    except ValueError as e:
        return str(e)

    # `crs` fixes both inputs to the SAME local metric grid (aoi's own UTM
    # zone) at exactly `scale` metres/pixel — required before the model
    # convolves its search-radius kernel over them. Without this, a reducer
    # composite like `lulc` has no defined native projection, so the kernel's
    # real-world size would depend on whatever resolution a given request
    # implies (e.g. the current map zoom level for interactive tile
    # rendering) instead of staying fixed. See mapping.local_utm_crs.
    crs = local_utm_crs(aoi)

    lulc = composite_lulc(aoi, start_date, end_date, scale=scale, crs=crs)
    if lulc is None:
        return (
            f"No Dynamic World land-cover scenes found for {region_label} in "
            f"{start_date}..{end_date} — try a wider date range."
        )

    requested_year = int(start_date[:4])
    worldpop, used_year = _population_image_for_year(aoi, requested_year)
    if worldpop is None:
        return f"No WorldPop coverage at all for {region_label}."
    year_note = f" (WorldPop has no {requested_year} data; used nearest available year {used_year})" \
        if used_year != requested_year else ""

    # Deliberately NOT clipped to `aoi` here: the model convolves this with a
    # search_radius_m kernel, and clipping before a neighbourhood operation
    # starves it of real neighbouring population data near the region's own
    # boundary (there's nothing there to sum/average once you've cut it off,
    # so pixels within roughly one kernel radius of the edge can come out
    # wrong or masked). The model itself already clips the final combined
    # result to `aoi` at the end, which is the right place for it.
    pop_resampled = (
        worldpop.select("population")
        .resample("bilinear")
        .reproject(crs=crs, scale=scale)
        .divide((100.0 / scale) ** 2)
    )

    model = UrbanNatureAccessGEE(
        search_radius=search_radius_m,
        decay_type=decay_type,
        demand_per_capita=demand_per_capita_m2,
        scale=float(scale),
    )

    try:
        out = model.run(
            lulc_image=lulc,
            population_image=pop_resampled,
            lulc_nature_dict=_NATURE_WEIGHTS,
            aoi=aoi,
        )
        summary = model.to_summary_dict(out, aoi)
    except Exception as e:  # noqa: BLE001
        return f"Computation failed: {type(e).__name__}: {e}"

    population = summary.get("population")
    nature_area = summary.get("nature_area_m2")
    supply_per_capita = summary.get("supply_per_capita")
    pct_undersupplied = summary.get("pct_population_undersupplied")
    if population is None or supply_per_capita is None:
        return f"Computation returned no data for {region_label} — region may not overlap the input layers."

    # If the map shows big gaps within the region, find out WHERE they come
    # from rather than leaving it a mystery. Three checks, in order of how far
    # through the pipeline they are: Dynamic World's own classification
    # (input), the model's own `population` band — which is pop.unmask(0), so
    # it SHOULD be ~100% regardless of anything else — and the final
    # `supply_per_capita` band the map actually shows. If population is fine
    # but supply_per_capita isn't, the gap isn't missing input data at all —
    # it's introduced by the model's own convolution math (scripts/
    # urban_nature_access.py convolves a population image that was clipped to
    # the AOI *before* convolving it, so the kernel runs out of real
    # neighbours near the region's own edge — a classic clip-before-
    # neighbourhood-op ordering issue).
    coverage_note = ""
    try:
        supply_coverage = _mask_coverage(out, "supply_per_capita", aoi, scale)
        if supply_coverage is not None and supply_coverage < 0.9:
            lulc_coverage = _mask_coverage(lulc, "label", aoi, scale)
            pop_coverage = _mask_coverage(out, "population", aoi, scale)
            lulc_pct = f"{lulc_coverage * 100:.0f}%" if lulc_coverage is not None else "unknown"
            pop_pct = f"{pop_coverage * 100:.0f}%" if pop_coverage is not None else "unknown"
            if pop_coverage is not None and pop_coverage > 0.95 and (lulc_coverage is None or lulc_coverage > 0.95):
                cause = (
                    "both inputs have near-full coverage there, so the gap is coming from the "
                    "model's own convolution, not missing data — most likely the population "
                    "image being clipped to the region before it's convolved, starving the "
                    "kernel of real neighbours near the region's edge"
                )
            else:
                cause = "usually means persistent cloud/fog left those input pixels unclassified for this date range"
            coverage_note = (
                f" NOTE: only {supply_coverage * 100:.0f}% of {region_label} has valid data "
                f"(input coverage there — Dynamic World: {lulc_pct}, population: {pop_pct}) — {cause}."
            )
    except Exception:  # noqa: BLE001
        pass  # diagnostic only — never let this block the main result

    # supply_per_capita has no natural upper bound (it can run far above the
    # demand target in leafy areas), so a fixed max clips most of the region
    # to one flat color — stretch it from the region's own 2nd-98th percentile
    # instead, anchored at 0 (no supply) so the scale stays interpretable.
    _, p98 = percentile_range(
        out, "supply_per_capita", aoi, scale, low=2, high=98,
        fallback=(0.0, demand_per_capita_m2 * 2),
    )

    board = results.current()
    publish_layer(
        board, name=f"{region_label}: nature supply per capita", image=out, band="supply_per_capita",
        palette=["red", "yellow", "green"], vis_min=0, vis_max=p98,
        label="Nature supply per capita", unit="m^2/person",
    )
    board.add_stat(label=region_label, model="urban_nature_access", metric="population", value=round(population))
    board.add_stat(
        label=region_label, model="urban_nature_access", metric="total_nature_area_m2",
        value=round(nature_area, 1) if nature_area is not None else None,
    )
    board.add_stat(
        label=region_label, model="urban_nature_access", metric="mean_supply_per_capita_m2",
        value=round(supply_per_capita, 2),
    )
    board.add_stat(
        label=region_label, model="urban_nature_access", metric="pct_population_undersupplied",
        value=round(pct_undersupplied, 1) if pct_undersupplied is not None else None,
    )

    return (
        f"Urban nature access in {region_label} ({start_date}..{end_date} land cover, "
        f"{used_year} population{year_note}): {population:,.0f} people, "
        f"mean supply = {supply_per_capita:.1f} m^2/person (target {demand_per_capita_m2:.0f} m^2/person), "
        f"{pct_undersupplied:.1f}% of population undersupplied "
        f"(InVEST Urban Nature Access / 2SFCA, Dynamic World + WorldPop)."
        f"{coverage_note}"
    )
