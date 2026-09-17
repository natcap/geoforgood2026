"""
InVEST Urban Flood Risk Mitigation Model - Google Earth Engine (Python API)
==========================================================================

References:
- InVEST User Guide: https://storage.googleapis.com/releases.naturalcapitalproject.org/invest-userguide/latest/en/urban_flood_mitigation.html
- InVEST source: https://github.com/natcap/invest/tree/main/src/natcap/invest/urban_flood_risk_mitigation
- NRCS-USDA (2004), Part 630 National Engineering Handbook, Ch. 10 (Curve Number method).
- NRCS TR-55 (1999), Urban Hydrology for Small Watersheds (general CN tables).


The Model
---------
For each pixel i, defined by a land use type and soil characteristics, runoff
Q (mm) is estimated with the Curve Number method (eq. 127):

    Q_p,i = (P - lambda * S_max,i)^2 / (P + (1 - lambda) * S_max,i)   if P > lambda * S_max,i
    Q_p,i = 0                                                         otherwise

where P is the design storm depth (mm), S_max,i is the potential retention
(mm), and lambda * S_max is the initial abstraction (lambda = 0.2, hard-coded
in InVEST).  S_max is a function of the curve number CN (eq. 128):

    S_max,i = 25400 / CN_i - 254

The fraction of runoff retention per pixel (eq. 129), the runoff retention
volume (eq. 130), and the runoff ("flood") volume (eq. 131) are then:

    R_i      = 1 - Q_p,i / P
    R_m3_i   = R_i * P * pixel.area * 1e-3
    Q_m3_i   = Q_p,i * pixel.area * 1e-3

Optionally, potential damage to built infrastructure is summed per watershed W
(eq. 132) and combined with retention volume into a relative service indicator
(eq. 133):

    Affected.build_W = sum over buildings b of a(b, W) * d(b)
    Service.built_W  = Affected.build_W * sum over pixels i in W of R_m3_i

where a(b, W) is the area (m^2) of building footprint b intersecting watershed
W, and d(b) is the damage value (currency/m^2) for that building's type.


Data Needs
----------
workspace directory (workspace directory, required): The folder where all the model's output files will be written. If this folder does not exist, it will be created. If data already exists in the folder, it will be overwritten.

file suffix (text, optional): Suffix that will be appended to all output file names. Useful to differentiate between model runs.

area of interest (vector, polygon/multipolygon, required): A map of areas over which to aggregate and summarize the final results. These may be watershed or sewershed boundaries.

rainfall depth (number, units: mm, required): Depth of rainfall for the design storm of interest. This is 𝑃 in equation (127).

land use/land cover (raster, required): Map of LULC. All values in this raster must have corresponding entries in the Biophysical Table.  All outputs will be produced at the resolution of this raster.

soil hydrologic group (raster, required): Map of soil hydrologic groups. Pixels may have values 1, 2, 3, or 4, corresponding to soil hydrologic groups A, B, C, or D, respectively.  Note that values other than 1, 2, 3 and 4 (such as 13 and 14) are not acceptable.

biophysical table (CSV, required): Table of curve number data for each LULC class. All LULC codes in the LULC raster must have corresponding entries in this table for each soil group.

Columns:
  * lucode (integer, required): LULC codes from the LULC raster. Each code must be a unique integer.
  * cn_a (number, units: unitless, required): The curve number value for this LULC type in the soil group code A. Curve numbers must be greater than 0 and less than or equal to 100.
  * cn_b (number, units: unitless, required): The curve number value for this LULC type in the soil group code B. Curve numbers must be greater than 0 and less than or equal to 100.
  * cn_c (number, units: unitless, required): The curve number value for this LULC type in the soil group code C. Curve numbers must be greater than 0 and less than or equal to 100.
  * cn_d (number, units: unitless, required): The curve number value for this LULC type in the soil group code D. Curve numbers must be greater than 0 and less than or equal to 100.

built infrastructure (vector, polygon/multipolygon, optional): Map of building footprints.
Fields needed in the built infrastructure vector:
  * type (integer, required): Code indicating the building type. These codes must match those in the Damage Loss Table.

damage loss table (CSV, conditionally required): Table of potential damage loss data for each building type. All values in the Built Infrastructure vector 'type' field must have corresponding entries in this table. Required if the Built Infrastructure vector is provided.
Columns in the damage loss table:
  * type (integer, required): Building type code.
  * damage (number, units: currency units/m², required): Potential damage loss for this building type. Any currency may be used.


Notes on this Earth Engine port
-------------------------------
- InVEST raises an error when the biophysical table is missing an LULC code, or
  when the soil group raster contains values other than 1-4.  Earth Engine is
  lazily evaluated and cannot raise mid-computation, so those pixels are masked
  out instead and therefore excluded from the zonal statistics.  Use
  ``missing_lucodes()`` to check the table against the LULC raster before
  trusting the aggregated numbers.
- InVEST aligns the LULC and soil rasters with mode resampling at the LULC pixel
  size.  Here the ``scale`` argument plays that role, and Earth Engine resamples
  the categorical inputs with nearest neighbour by default.
- InVEST uses the nominal pixel size product for pixel area; this port uses
  ``ee.Image.pixelArea()``, which accounts for the true per-pixel area.

Disclaimer
---------- 
Created by AI (Claude Code used) so be sure to interrogate results for accuracy.
"""

import ee

# Initial abstraction ratio; hard-coded in the InVEST design doc.
LAMBDA = 0.2

# Soil hydrologic group raster value -> biophysical table column.
SOIL_GROUP_COLUMNS = {1: "cn_a", 2: "cn_b", 3: "cn_c", 4: "cn_d"}


class InvestUrbanFloodRiskMitigationModel:
    """
    Implements the InVEST Urban Flood Risk Mitigation model in Google Earth Engine.
    """

    def __init__(
        self,
        lulc_image: ee.Image,
        biophysical_dict: dict,
        soil_group_image: ee.Image,
        rainfall_depth,
        damage_dict: dict = None,
        building_type_property: str = "type",
        scale: float = 30.0,
    ):
        """
        Parameters:
            lulc_image: ee.Image containing discrete integer land use/cover codes.
            biophysical_dict: Dict structured as:
                {
                    lucode: {
                        'cn_a': float (0 < CN <= 100),
                        'cn_b': float,
                        'cn_c': float,
                        'cn_d': float
                    }, ...
                }
            soil_group_image: ee.Image of soil hydrologic groups, with values
                1, 2, 3, 4 corresponding to groups A, B, C, D.  Any other value
                is masked out.  Dual groups (e.g. A/D) must be collapsed to a
                single group before they are passed in.
            rainfall_depth: Design storm depth in mm.  A number (as in InVEST) or,
                as an Earth Engine extension, an ee.Image of per-pixel depths.
            damage_dict: Optional dict mapping building type code to potential
                damage loss in currency units per m^2, e.g. {1: 250.0}.  Required
                to compute 'aff_bld' and 'serv_blt'.
            building_type_property: Name of the building type field on the built
                infrastructure features.
            scale: Analysis resolution in meters (InVEST uses the LULC resolution).
        """
        self.lulc = lulc_image
        self.table = biophysical_dict
        self.soil_group = soil_group_image
        self.rainfall = (
            rainfall_depth
            if isinstance(rainfall_depth, ee.Image)
            else ee.Image.constant(float(rainfall_depth))
        )
        self.damage_table = damage_dict or {}
        self.building_type_property = building_type_property
        self.scale = float(scale)

    def _curve_number(self) -> ee.Image:
        """
        Maps the (LULC, soil group) combination to a curve number, mirroring
        InVEST's ``_lu_to_cn_op``.

        Soil group values outside 1-4 are masked before the lookup so that
        dual-group codes such as 14 cannot collide with a valid combination.
        """
        soil = self.soil_group.updateMask(
            self.soil_group.gte(1).And(self.soil_group.lte(4))
        )

        # Encode (lucode, soil group) as a single integer: lucode * 100 + group.
        from_values = []
        to_values = []
        for key in self.table.keys():
            lucode = int(key)
            row = self.table[key]
            for soil_code, column in SOIL_GROUP_COLUMNS.items():
                from_values.append(lucode * 100 + soil_code)
                to_values.append(float(row[column]))

        combined = self.lulc.multiply(100).add(soil)
        # No default value: unmatched (lucode, group) pairs are masked out.
        return combined.remap(from_values, to_values).rename("cn")

    def missing_lucodes(self, aoi: ee.Geometry) -> list:
        """
        Returns the LULC codes present within the AOI that have no row in the
        biophysical table.  InVEST fails outright on these; here they are masked,
        so this check is worth running before interpreting the results.
        """
        histogram = self.lulc.reduceRegion(
            reducer=ee.Reducer.frequencyHistogram(),
            geometry=aoi,
            scale=self.scale,
            maxPixels=1e9,
        ).getInfo()

        present = next(iter(histogram.values()), None) or {}
        tabulated = {int(key) for key in self.table.keys()}
        return sorted(
            int(float(code)) for code in present if int(float(code)) not in tabulated
        )

    def compute(self, aoi: ee.Geometry = None) -> ee.Image:
        """
        Executes the per-pixel InVEST Urban Flood Risk Mitigation pipeline and
        returns a multi-band ee.Image.

        Bands:
            cn                     - curve number (unitless)
            s_max                  - potential retention (mm), eq. 128
            q_mm                   - runoff (mm), eq. 127
            runoff_retention_index - retention fraction (0-1), eq. 129
            runoff_retention_m3    - retention volume (m^3/pixel), eq. 130
            q_m3                   - runoff volume (m^3/pixel), eq. 131
        """
        # 1. Curve number from LULC and soil hydrologic group
        cn = self._curve_number()

        # 2. Potential retention S_max = 25400 / CN - 254
        # A CN of 0 means infinite retention, so S_max is set higher than any
        # possible storm depth (the largest recorded storm depth is 6,433 mm).
        s_max = (
            ee.Image.constant(25400.0)
            .divide(cn)
            .subtract(254.0)
            .where(cn.eq(0), 100000.0)
            .rename("s_max")
        )

        # 3. Runoff Q_p (mm) via the Curve Number method
        precipitation = self.rainfall
        initial_abstraction = s_max.multiply(LAMBDA)
        q_mm = (
            precipitation.subtract(initial_abstraction)
            .pow(2)
            .divide(precipitation.add(s_max.multiply(1.0 - LAMBDA)))
            .where(precipitation.lte(initial_abstraction), 0.0)
            .rename("q_mm")
        )

        # 4. Runoff retention index R = 1 - Q_p / P
        runoff_retention_index = (
            ee.Image.constant(1.0)
            .subtract(q_mm.divide(precipitation))
            .rename("runoff_retention_index")
        )

        # 5. Volumes (m^3 per pixel); 1e-3 converts mm of depth to meters
        pixel_area = ee.Image.pixelArea()
        runoff_retention_m3 = (
            runoff_retention_index.multiply(precipitation)
            .multiply(pixel_area)
            .multiply(1e-3)
            .rename("runoff_retention_m3")
        )
        q_m3 = (
            q_mm.multiply(pixel_area).multiply(1e-3).rename("q_m3")
        )

        output_stack = ee.Image.cat([
            cn,
            s_max,
            q_mm,
            runoff_retention_index,
            runoff_retention_m3,
            q_m3,
        ])

        if aoi is not None:
            output_stack = output_stack.clip(aoi)

        return output_stack

    def _damage_per_aoi(
        self,
        aoi_collection: ee.FeatureCollection,
        buildings: ee.FeatureCollection,
    ) -> ee.FeatureCollection:
        """
        Sets 'aff_bld' on each AOI feature: the sum over intersecting buildings
        of (footprint area within the AOI) * (damage per m^2 for its type),
        mirroring InVEST's ``_calculate_damage_to_infrastructure_in_aoi``.
        """
        damage_rates = ee.Dictionary(
            {str(int(key)): float(value) for key, value in self.damage_table.items()}
        )
        type_property = self.building_type_property
        matches_key = "_buildings"

        # outer=True keeps AOI features that contain no buildings at all.
        joined = ee.Join.saveAll(matchesKey=matches_key, outer=True).apply(
            primary=aoi_collection,
            secondary=buildings,
            condition=ee.Filter.intersects(
                leftField=".geo", rightField=".geo", maxError=ee.ErrorMargin(1)
            ),
        )

        def aggregate_damage(aoi_feature):
            aoi_geometry = aoi_feature.geometry()
            matches = ee.List(
                ee.Algorithms.If(
                    aoi_feature.get(matches_key),
                    aoi_feature.get(matches_key),
                    ee.List([]),
                )
            )

            def building_damage(building):
                building = ee.Feature(building)
                key = building.getNumber(type_property).format("%d")
                rate = ee.Number(
                    ee.Algorithms.If(
                        damage_rates.contains(key), damage_rates.get(key), 0
                    )
                )
                intersecting_area = (
                    building.geometry()
                    .intersection(aoi_geometry, ee.ErrorMargin(1))
                    .area(1)
                )
                return building.set("_damage", intersecting_area.multiply(rate))

            total_damage = ee.Number(
                ee.Algorithms.If(
                    matches.size().gt(0),
                    ee.FeatureCollection(matches.map(building_damage)).aggregate_sum(
                        "_damage"
                    ),
                    0,
                )
            )
            # Rebuild the feature without the matched-building list, which would
            # otherwise carry every footprint geometry into the results.
            properties = aoi_feature.toDictionary().remove([matches_key], True)
            return ee.Feature(aoi_geometry, properties).set(
                "aff_bld", total_damage
            )

        return joined.map(aggregate_damage)

    def aggregate(
        self,
        aoi_collection: ee.FeatureCollection,
        model_outputs: ee.Image = None,
        buildings: ee.FeatureCollection = None,
    ) -> ee.FeatureCollection:
        """
        Aggregates the per-pixel outputs over the AOI (watershed/sewershed)
        features, producing the fields InVEST writes to flood_risk_service.shp.

        Fields set on each feature:
            rnf_rt_idx - average runoff retention index
            rnf_rt_m3  - sum of runoff retention volumes (m^3)
            flood_vol  - total flood (runoff) volume (m^3)
            aff_bld    - potential damage to built infrastructure (currency),
                         only when buildings and a damage table are provided
            serv_blt   - Service.built indicator (currency * m^3), same condition

        The raw reducer outputs (e.g. 'runoff_retention_m3_sum') and the input
        AOI's own properties are retained alongside these fields.
        """
        outputs = self.compute() if model_outputs is None else model_outputs

        # InVEST takes the mean of the retention index and the sum of both
        # volume rasters per AOI feature.
        statistics = outputs.select(
            ["runoff_retention_index", "runoff_retention_m3", "q_m3"]
        ).reduceRegions(
            collection=aoi_collection,
            reducer=ee.Reducer.mean().combine(
                reducer2=ee.Reducer.sum(), sharedInputs=True
            ),
            scale=self.scale,
        )

        def set_invest_fields(feature):
            retention_volume = ee.Number(
                ee.Algorithms.If(
                    feature.get("runoff_retention_m3_sum"),
                    feature.get("runoff_retention_m3_sum"),
                    0,
                )
            )
            return feature.set({
                "rnf_rt_idx": feature.get("runoff_retention_index_mean"),
                "rnf_rt_m3": retention_volume,
                "flood_vol": feature.get("q_m3_sum"),
            })

        statistics = statistics.map(set_invest_fields)

        if buildings is None or not self.damage_table:
            return statistics

        with_damage = self._damage_per_aoi(statistics, buildings)

        # Service.built = Affected.build * sum(R_m3) over the watershed.
        return with_damage.map(
            lambda feature: feature.set(
                "serv_blt",
                ee.Number(feature.get("aff_bld")).multiply(
                    ee.Number(feature.get("rnf_rt_m3"))
                ),
            )
        )

    def to_summary_list(
        self,
        aggregated: ee.FeatureCollection,
        id_properties: list = None,
    ) -> list:
        """
        Reduces the aggregated collection to a list of plain Python dicts, one
        per AOI feature, holding only the InVEST summary fields (plus any
        identifying properties named in ``id_properties``).
        """
        fields = ["rnf_rt_idx", "rnf_rt_m3", "flood_vol"]
        if self.damage_table:
            fields += ["aff_bld", "serv_blt"]
        properties = list(id_properties or []) + fields

        features = aggregated.getInfo()["features"]
        return [
            {
                name: feature["properties"].get(name)
                for name in properties
                if name in feature["properties"]
            }
            for feature in features
        ]


def run_model(
    bbox=[36.75, -1.35, 36.90, -1.22],
    rainfall_depth=40.0,
    scale=30.0,
    use_open_buildings=True,
    gcp_project='sdss-natcap-geoforgood26',
):
    """Runs the model over a bounding box using global Earth Engine datasets.

    The default bounding box covers central Nairobi, Kenya.  Google Open
    Buildings only covers the Global South, so for a study area outside that
    footprint (e.g. the central Paris box used by urban_cooling.py,
    [2.25, 48.81, 2.42, 48.90]) pass ``use_open_buildings=False``; the built
    infrastructure input is optional in InVEST and 'aff_bld'/'serv_blt' are
    simply not produced.

    Parameters:
        bbox (list): Bounding box coordinates [west, south, east, north].
        rainfall_depth (float): Design storm depth in mm.
        scale (float): Execution resolution in meters.
        use_open_buildings (bool): Include the built infrastructure valuation.
        gcp_project (str): Cloud project used to initialize Earth Engine.

    Returns:
        tuple: (per-watershed summary list, mean per-pixel statistics dict)
    """
    # Initialize Earth Engine
    ee.Authenticate()
    ee.Initialize(project=gcp_project)

    # 1. Define the study area
    study_area = ee.Geometry.BBox(*bbox)

    # 2. Load ESA WorldCover 10m (2021) as the LULC input
    worldcover = ee.ImageCollection("ESA/WorldCover/v200").first().select("Map")

    # 3. Soil hydrologic groups: HiHydroSoil v2.0 (250 m), from the
    # awesome-gee-community-catalog.  Class values 14/24/34 are the dual groups
    # A/D, B/D and C/D; InVEST only accepts 1-4, so they are collapsed to D
    # (the undrained condition, i.e. the conservative choice for flood work).
    soil_group = (
        ee.Image("projects/sat-io/open-datasets/HiHydroSoilv2_0/Hydrologic_Soil_Group_250m")
        .remap([1, 2, 3, 4, 14, 24, 34], [1, 2, 3, 4, 4, 4, 4])
        .rename("soil_group")
    )

    # 4. Biophysical table: curve numbers per ESA WorldCover class per soil
    # group, approximated from NRCS TR-55 cover types (ARC-II).  The user guide
    # recommends replacing these with values specific to the study area, and
    # converting to ARC-III conditions when the focus is on flood effects.
    # Water and wetlands connected to the stream network get CN = 99.
    biophysical_table = {
        10: {"cn_a": 30, "cn_b": 55, "cn_c": 70, "cn_d": 77},   # Tree cover (woods, good)
        20: {"cn_a": 30, "cn_b": 48, "cn_c": 65, "cn_d": 73},   # Shrubland (brush, good)
        30: {"cn_a": 39, "cn_b": 61, "cn_c": 74, "cn_d": 80},   # Grassland (pasture, good)
        40: {"cn_a": 67, "cn_b": 78, "cn_c": 85, "cn_d": 89},   # Cropland (row crops, good)
        50: {"cn_a": 77, "cn_b": 85, "cn_c": 90, "cn_d": 92},   # Built-up (residential, 65% impervious)
        60: {"cn_a": 77, "cn_b": 86, "cn_c": 91, "cn_d": 94},   # Bare/sparse (fallow, bare soil)
        70: {"cn_a": 98, "cn_b": 98, "cn_c": 98, "cn_d": 98},   # Snow and ice
        80: {"cn_a": 99, "cn_b": 99, "cn_c": 99, "cn_d": 99},   # Permanent water bodies
        90: {"cn_a": 99, "cn_b": 99, "cn_c": 99, "cn_d": 99},   # Herbaceous wetland
        95: {"cn_a": 99, "cn_b": 99, "cn_c": 99, "cn_d": 99},   # Mangroves
        100: {"cn_a": 49, "cn_b": 69, "cn_c": 79, "cn_d": 84},  # Moss and lichen (pasture, fair)
    }

    # 5. Area of interest: HydroSHEDS level-12 basins, clipped to the study box
    # so the example stays inexpensive.  InVEST aggregates over the AOI polygons
    # exactly as supplied, so pass whole basins (or sewersheds) for a real run.
    watersheds = (
        ee.FeatureCollection("WWF/HydroSHEDS/v1/Basins/hybas_12")
        .filterBounds(study_area)
        .map(lambda feature: feature.intersection(study_area, ee.ErrorMargin(1)))
    )

    # 6. Optional built infrastructure and damage loss table.  Open Buildings
    # carries no building type, so every footprint is assigned type 1.  The
    # damage value below is illustrative only.
    buildings = None
    damage_table = None
    if use_open_buildings:
        buildings = (
            ee.FeatureCollection("GOOGLE/Research/open-buildings/v3/polygons")
            .filterBounds(study_area)
            .map(lambda feature: feature.set("type", 1))
        )
        damage_table = {1: 250.0}  # currency units per m^2

    # 7. Instantiate and compute
    model = InvestUrbanFloodRiskMitigationModel(
        lulc_image=worldcover,
        biophysical_dict=biophysical_table,
        soil_group_image=soil_group,
        rainfall_depth=rainfall_depth,
        damage_dict=damage_table,
        scale=scale,
    )

    results = model.compute(aoi=study_area)

    # 8. Aggregate per watershed and summarize per-pixel results
    aggregated = model.aggregate(
        aoi_collection=watersheds,
        model_outputs=results,
        buildings=buildings,
    )
    summary = model.to_summary_list(aggregated, id_properties=["HYBAS_ID"])

    pixel_stats = results.reduceRegion(
        reducer=ee.Reducer.mean(),
        geometry=study_area,
        scale=scale,
        maxPixels=1e9,
    ).getInfo()

    return summary, pixel_stats
