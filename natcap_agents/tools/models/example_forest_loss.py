"""EXAMPLE model tool — a template for the project's own geospatial models.

Delete or replace this once your models are in place; it exists to (a) prove
the prompt -> tool -> map/stats -> UI pipeline works end to end, and (b) show
the two things a model tool must do to show up correctly in the app:

  1. Call `ensure_ee()` before touching `ee` (Earth Engine may not be
     initialized yet in a fresh process).
  2. Publish a map layer via `results.current().add_layer(...)` and/or a stats
     row via `results.current().add_stat(...)` — the app reads the board, not
     your return string. The return string is just what the agent sees.

This one computes tree cover loss (Hansen Global Forest Change) inside a
region and adds both a loss-year layer and a summary stat row.
"""
from __future__ import annotations

from smolagents import tool

from ... import results
from ...regions import region_geometry
from ...safety import ensure_ee
from ..mapping import publish_layer

_HANSEN_ASSET = "UMD/hansen/global_forest_change_2023_v1_11"


@tool
def example_forest_loss(region: str, region_label: str = "region", min_tree_cover: int = 30) -> str:
    """EXAMPLE model — tree cover loss area (Hansen Global Forest Change) inside a region.

    Template only; replace with a real project model. Kept as a working
    end-to-end example of the prompt -> tool -> map/stats pipeline.

    Args:
        region: A region_id from resolve_region(), or a 'west,south,east,north'
            bounding box in degrees. Never pass a raw polygon/GeoJSON here.
        region_label: Short label used for the stats-table row and map layer name.
        min_tree_cover: Year-2000 canopy-cover threshold (%) counted as forest.
    """
    ensure_ee()
    import ee

    try:
        geom = region_geometry(region)
    except ValueError as e:
        return str(e)

    gfc = ee.Image(_HANSEN_ASSET)
    is_forest = gfc.select("treecover2000").gte(min_tree_cover)
    loss = gfc.select("lossyear").gt(0).And(is_forest).rename("forest_loss")

    try:
        area_m2 = loss.multiply(ee.Image.pixelArea()).reduceRegion(
            reducer=ee.Reducer.sum(), geometry=geom, scale=30, maxPixels=1e10, bestEffort=True,
        ).get("forest_loss")
        loss_ha = ee.Number(area_m2).divide(10_000).getInfo()
    except Exception as e:  # noqa: BLE001
        return f"Computation failed: {type(e).__name__}: {e}"

    # `loss` is a 0/1 mask (did this pixel lose forest, yes/no) — a single
    # highlight color, not a min..max gradient (there's nothing in between).
    board = results.current()
    publish_layer(
        board, name=f"{region_label}: forest loss", image=loss.selfMask(), band="forest_loss",
        palette=["orangered"], vis_min=0, vis_max=1,
        label="Forest loss (Hansen GFC)",
    )
    board.add_stat(
        label=region_label, model="example_forest_loss", metric="tree_cover_loss_ha",
        value=round(loss_ha, 1),
    )

    return (
        f"Tree cover loss in {region_label}: {loss_ha:.1f} ha "
        f"(Hansen GFC, >= {min_tree_cover}% year-2000 cover)."
    )
