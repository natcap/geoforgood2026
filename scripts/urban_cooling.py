"""
InVEST Urban Cooling Model - Google Earth Engine (Python API)
============================================================

References:
- InVEST User Guide: https://storage.googleapis.com/releases.naturalcapitalproject.org/invest-userguide/latest/en/urban_cooling_model.html
- Bosch et al. (2021) / Hamel et al. (2024), GMD.


Data Needs
----------
workspace directory (workspace directory, required): The folder where all the model’s output files will be written. If this folder does not exist, it will be created. If data already exists in the folder, it will be overwritten.

file suffix (text, optional): Suffix that will be appended to all output file names. Useful to differentiate between model runs.

land use/land cover (raster, units: unitless, required): Map of LULC for the area of interest. All values in this raster must have corresponding entries in the Biophysical Table.  The model will use the resolution and projection of this layer to resample and reproject all outputs. The resolution should be small enough to capture the effect of green spaces in the landscape, although LULC categories can comprise a mix of vegetated and non-vegetated covers (e.g. “residential”, which may have 30% canopy cover).

biophysical table (CSV, required): A table mapping each LULC code to biophysical data for that LULC class. All values in the LULC raster must have corresponding entries in this table.

Columns:
  * lucode (integer, required): LULC codes from the LULC raster. Each code must be a unique integer.
  * kc (number, units: unitless, required): Crop coefficient for this LULC class.
  * green_area (true/false): Enter 1 to indicate that the LULC is considered a green area. Enter 0 to indicate that the LULC is not considered a green area.  Green areas larger than 2 hectares have an additional cooling effect.
  * shade (ratio, conditionally required): The proportion of area in this LULC class that is covered by tree canopy at least 2 meters high. Required if the ‘factors’ option is selected for the Cooling Capacity Calculation Method.
  * albedo (ratio, conditionally required): The proportion of solar radiation that is directly reflected by this LULC class. Required if the ‘factors’ option is selected for the Cooling Capacity Calculation Method.
  * building_intensity (ratio, conditionally required): The ratio of building floor area to footprint area, with all values in this column normalized between 0 and 1. Required if the ‘intensity’ option is selected for the Cooling Capacity Calculation Method.

reference evapotranspiration (raster, units: mm, required): Map of reference evapotranspiration values.  These values can be for a specific date or monthly values can be used as a proxy.

area of interest (vector, polygon/multipolygon, required): A map of areas over which to aggregate and summarize the final results.  The AOI(s) will typically be city or neighborhood boundaries.

maximum cooling distance (number, units: m, required): Distance over which green areas larger than 2 hectares have a cooling effect.  This is 𝑑𝑐⁢𝑜⁢𝑜⁢𝑙 in equation (118). Recommended value: 450 m.

reference air temperature (number, units: °C, required): Air temperature in a rural reference area where the urban heat island effect is not observed. This is 𝑇𝑎⁢𝑖⁢𝑟,𝑟⁢𝑒⁢𝑓 in equation (120). This could be nighttime or daytime temperature, for a specific date or an average over several days. The results will be given for the same period of interest.

UHI effect (number, units: °C, required): The magnitude of the urban heat island effect, i.e., the difference between the rural reference temperature and the maximum temperature observed in the city. This model is designed for cases where UHI is positive, meaning the urban air temperature is greater than the rural reference temperature. This is 𝑈⁢𝐻⁡𝐼𝑚𝑎𝑥 in equation (120).

air blending distance (number, units: m, required): Radius over which to average air temperatures to account for air mixing. Recommended value range for initial run: 500 m to 600 m; see Schatz et al. 2014 and Lonsdorf et al. 2021.

cooling capacity calculation method (option, required): The air temperature predictor method to use. Values must be one of the following text strings:
  * ”factors”: Use the weighted shade, albedo, and ETI factors as a temperature predictor (for daytime temperatures).
  * ”intensity”: Use building intensity as a temperature predictor (for nighttime temperatures).

buildings (vector, polygon/multipolygon, conditionally required): A map of built infrastructure footprints. Required if Run Energy Savings Valuation is selected.
Fields needed in the buildings vector:
  * type (integer, required): Code indicating the building type. These codes must match those in the Energy Consumption Table.

run energy savings valuation (true/false): Run the energy savings valuation model.

run work productivity valuation (true/false): Run the work productivity valuation model.

energy consumption table (CSV, conditionally required): A table of energy consumption data for each building type. Required if Run Energy Savings Valuation is selected.
Columns in the energy conservation table:
  * type (integer, required): Building type codes matching those in the Buildings vector.
  * consumption (number, units: kWh/(m² · °C), required): Energy consumption by footprint area for this building type.  Note: The consumption value is per unit of footprint area, not floor area. This value must be adjusted for the average number of stories for structures of this type.
  * cost (number, units: currency units/kWh, optional): The cost of electricity for this building type. If this column is provided, the energy savings outputs will be in the this currency unit rather than kWh.  The values in this column are very likely to be the same for all building types.

average relative humidity (percent, conditionally required): The average relative humidity over the time period of interest. Required if Run Work Productivity Valuation is selected.

shade weight (ratio, optional): The relative weight to apply to shade when calculating the cooling capacity index. If not provided, defaults to 0.6.

albedo weight (ratio, optional): The relative weight to apply to albedo when calculating the cooling capacity index. If not provided, defaults to 0.2.

evapotranspiration weight (ratio, optional): The relative weight to apply to ETI when calculating the cooling capacity index. If not provided, defaults to 0.2.

"""

import math
import ee

class InvestUrbanCoolingModel:
    """
    Implements the InVEST Urban Cooling model in Google Earth Engine.
    """

    def __init__(
        self,
        lulc_image: ee.Image,
        biophysical_dict: dict,
        ref_eto_image: ee.Image,
        cc_method: str = "factors",
        w_shade: float = 0.6,
        w_albedo: float = 0.2,
        w_eti: float = 0.2,
        d_cool: float = 450.0,
        green_area_threshold_ha: float = 2.0,
        t_ref: float = 25.0,
        uhi_max: float = 3.5,
        air_blending_distance: float = 500.0,
        calc_wbgt: bool = True,
        avg_rel_humidity: float = 50.0,
        scale: float = 30.0,
    ):
        """
        Parameters:
            lulc_image: ee.Image containing discrete integer land use/cover codes.
            biophysical_dict: Dict structured as:
                {
                    lucode: {
                        'kc': float,
                        'green_area': int (0 or 1),
                        'shade': float (0-1),
                        'albedo': float (0-1),
                        'building_intensity': float (0-1)
                    }, ...
                }
            ref_eto_image: ee.Image of reference evapotranspiration (mm).
            cc_method: 'factors' (daytime) or 'intensity' (nighttime).
            w_shade, w_albedo, w_eti: Factor weights for cooling capacity (must sum to ~1.0).
            d_cool: Distance (m) over which green areas have an external cooling effect (default: 450 m).
            green_area_threshold_ha: Minimum green area (ha) within d_cool needed to provide park cooling (default: 2.0 ha).
            t_ref: Rural reference air temperature in degrees Celsius.
            uhi_max: Maximum magnitude of the Urban Heat Island effect in degrees Celsius.
            air_blending_distance: Radius (m) over which to average air temperatures for atmospheric mixing (default: 500 m).
            calc_wbgt: Whether to compute Wet Bulb Globe Temperature and work productivity loss.
            avg_rel_humidity: Average relative humidity (%) during heat event.
            scale: Analysis resolution in meters (used to build discrete kernel weights).
        """
        self.lulc = lulc_image
        self.table = biophysical_dict
        self.ref_eto = ref_eto_image
        self.cc_method = cc_method.lower()
        self.w_shade = w_shade
        self.w_albedo = w_albedo
        self.w_eti = w_eti
        self.d_cool = float(d_cool)
        self.green_threshold_ha = float(green_area_threshold_ha)
        self.t_ref = float(t_ref)
        self.uhi_max = float(uhi_max)
        self.air_blending_distance = float(air_blending_distance)
        self.calc_wbgt = calc_wbgt
        self.rh = float(avg_rel_humidity)
        self.scale = float(scale)

    def _build_decay_kernel(self) -> ee.Kernel:
        """
        Builds a normalized 2D exponential distance-decay convolution kernel
        matching InVEST: w(d) = exp(-d / d_cool) for d <= d_cool, 0 elsewhere.
        """
        radius_pixels = int(math.ceil(self.d_cool / self.scale))
        size = 2 * radius_pixels + 1
        weights = []

        for y in range(-radius_pixels, radius_pixels + 1):
            row = []
            for x in range(-radius_pixels, radius_pixels + 1):
                dist = math.hypot(x, y) * self.scale
                if dist <= self.d_cool:
                    row.append(math.exp(-dist / self.d_cool))
                else:
                    row.append(0.0)
            weights.append(row)

        return ee.Kernel.fixed(
            width=size,
            height=size,
            weights=weights,
            x=radius_pixels,
            y=radius_pixels,
            normalize=True,
        )

    def _remap_biophysical(self, attribute: str, default_val: float = 0.0) -> ee.Image:
        """Remaps LULC codes to a specific biophysical table property."""
        lucodes = [int(k) for k in self.table.keys()]
        values = [float(self.table[k].get(attribute, default_val)) for k in self.table.keys()]
        return self.lulc.remap(lucodes, values, default_val).rename(attribute)

    def compute(self, aoi: ee.Geometry = None, et_max_val: float = None) -> ee.Image:
        """
        Executes the InVEST Urban Cooling pipeline and returns a multi-band ee.Image.
        """
        # 1. Remap biophysical properties from LULC
        kc = self._remap_biophysical("kc", 0.0)
        green_area = self._remap_biophysical("green_area", 0.0)
        shade = self._remap_biophysical("shade", 0.0)
        albedo = self._remap_biophysical("albedo", 0.15)
        building_intensity = self._remap_biophysical("building_intensity", 0.0)

        # 2. Determine ET_max (from raster region or passed scalar)
        if et_max_val is not None:
            et_max = ee.Number(et_max_val)
        elif aoi is not None:
            et_max = ee.Number(
                self.ref_eto.reduceRegion(
                    reducer=ee.Reducer.max(),
                    geometry=aoi,
                    scale=self.scale,
                    maxPixels=1e9,
                ).values().get(0)
            )
        else:
            # Fallback global default if AOI is omitted
            et_max = ee.Number(10.0)

        # 3. Evapotranspiration Index (ETI)
        # ETI = (Kc * ET0) / ET_max
        eti = (
            kc.multiply(self.ref_eto)
            .divide(et_max)
            .clamp(0.0, 1.0)
            .rename("eti")
        )

        # 4. Local Cooling Capacity (CC)
        if self.cc_method == "factors":
            # Daytime: CC = w_shade * shade + w_albedo * albedo + w_eti * ETI
            cc = (
                shade.multiply(self.w_shade)
                .add(albedo.multiply(self.w_albedo))
                .add(eti.multiply(self.w_eti))
                .clamp(0.0, 1.0)
                .rename("cc")
            )
        else:
            # Nighttime: CC = 1.0 - building_intensity
            cc = (
                ee.Image.constant(1.0)
                .subtract(building_intensity)
                .clamp(0.0, 1.0)
                .rename("cc")
            )

        # 5. Green Area (GA) in hectares within d_cool
        # Circular flat kernel over radius d_cool
        circle_kernel = ee.Kernel.circle(radius=self.d_cool, units="meters", normalize=False)
        green_pixel_count = green_area.convolve(circle_kernel)

        # Pixel area in hectares: m^2 / 10,000
        pixel_area_ha = ee.Image.pixelArea().divide(10000.0)
        ga_ha = green_pixel_count.multiply(pixel_area_ha).rename("green_area_ha")

        # 6. Large Green Space Cooling Effect (CC_park)
        # Exponential distance decay kernel
        decay_kernel = self._build_decay_kernel()
        cc_park = (
            green_area.multiply(cc)
            .convolve(decay_kernel)
            .rename("cc_park")
        )

        # 7. Heat Mitigation Index (HMI)
        # HMI = CC if (CC >= CC_park) OR (GA < 2.0 ha), else CC_park
        condition_use_cc = cc.gte(cc_park).Or(ga_ha.lt(self.green_threshold_ha))
        hmi = (
            cc_park.where(condition_use_cc, cc)
            .clamp(0.0, 1.0)
            .rename("hmi")
        )

        # 8. Air Temperature without atmospheric mixing
        # T_air_nomix = T_ref + (1 - HMI) * UHI_max
        t_air_nomix = (
            ee.Image.constant(self.t_ref)
            .add(ee.Image.constant(1.0).subtract(hmi).multiply(self.uhi_max))
            .rename("t_air_nomix")
        )

        # 9. Air Temperature with atmospheric mixing (Gaussian blur)
        # InVEST uses a Gaussian kernel with user-specified air blending distance
        gaussian_kernel = ee.Kernel.gaussian(
            radius=self.air_blending_distance,
            units="meters",
            normalize=True,
        )
        t_air = t_air_nomix.convolve(gaussian_kernel).rename("t_air")

        # Combine baseline outputs
        output_stack = ee.Image.cat([
            cc,
            ga_ha,
            cc_park,
            hmi,
            t_air_nomix,
            t_air,
        ])

        # 10. Optional: Wet Bulb Globe Temperature (WBGT) & Work Productivity Loss
        if self.calc_wbgt:
            # Vapor pressure e_i (hPa)
            vapor_pressure = t_air.expression(
                "(rh / 100.0) * 6.105 * exp((17.27 * T) / (237.7 + T))",
                {"rh": ee.Number(self.rh), "T": t_air},
            ).rename("vapor_pressure")

            # WBGT = 0.567 * T_air + 0.393 * e_i + 3.94
            wbgt = t_air.expression(
                "0.567 * T + 0.393 * e + 3.94",
                {"T": t_air, "e": vapor_pressure},
            ).rename("wbgt")

            # Light work productivity loss (%)
            light_work_loss = (
                ee.Image(0.0)
                .where(wbgt.gte(31.5).And(wbgt.lt(32.0)), 25.0)
                .where(wbgt.gte(32.0).And(wbgt.lt(32.5)), 50.0)
                .where(wbgt.gte(32.5), 75.0)
                .rename("light_work_loss")
            )

            # Heavy work productivity loss (%)
            heavy_work_loss = (
                ee.Image(0.0)
                .where(wbgt.gte(27.5).And(wbgt.lt(29.5)), 25.0)
                .where(wbgt.gte(29.5).And(wbgt.lt(31.5)), 50.0)
                .where(wbgt.gte(31.5), 75.0)
                .rename("heavy_work_loss")
            )

            output_stack = output_stack.addBands([
                wbgt,
                light_work_loss,
                heavy_work_loss,
            ])

        if aoi is not None:
            output_stack = output_stack.clip(aoi)

        return output_stack

def run_model(bbox=[2.25, 48.81, 2.42, 48.90], gcp_project='sdss-natcap-geoforgood26'):
    # Initialize Earth Engine
    ee.Authenticate()
    ee.Initialize(project=gcp_project)


    # 1. Define Area of Interest (Central Paris)
    aoi = ee.Geometry.BBox(*bbox)

    # 2. Load ESA WorldCover 10m (2021)
    worldcover = ee.ImageCollection("ESA/WorldCover/v200").first().select("Map")

    # 3. Biophysical Table mapping ESA WorldCover classes
    # Classes: 10: Trees, 20: Shrub, 30: Grass, 40: Cropland, 50: Built-up, 60: Bare, 80: Water
    biophysical_table = {
        10: {"kc": 1.0, "green_area": 1, "shade": 0.9, "albedo": 0.18, "building_intensity": 0.0},
        20: {"kc": 0.8, "green_area": 1, "shade": 0.4, "albedo": 0.20, "building_intensity": 0.0},
        30: {"kc": 0.7, "green_area": 1, "shade": 0.1, "albedo": 0.22, "building_intensity": 0.0},
        40: {"kc": 0.6, "green_area": 1, "shade": 0.1, "albedo": 0.20, "building_intensity": 0.0},
        50: {"kc": 0.1, "green_area": 0, "shade": 0.05, "albedo": 0.15, "building_intensity": 0.75},
        60: {"kc": 0.1, "green_area": 0, "shade": 0.0, "albedo": 0.30, "building_intensity": 0.0},
        80: {"kc": 1.0, "green_area": 1, "shade": 0.0, "albedo": 0.08, "building_intensity": 0.0},
    }

    # 4. Load Reference ET (TerraClimate summer monthly average rescaled to daily mm)
    terraclimate = (
        ee.ImageCollection("IDAHO_EPSCOR/TERRACLIMATE")
        .filterDate("2023-06-01", "2023-08-31")
        .select("pet")
        .mean()
        .multiply(0.1)  # Scale factor for mm/month
        .divide(30.0)   # Convert to mm/day
        .rename("eto")
    )

    # 5. Instantiate and compute
    model = InvestUrbanCoolingModel(
        lulc_image=worldcover,
        biophysical_dict=biophysical_table,
        ref_eto_image=terraclimate,
        cc_method="factors",           # Daytime cooling capacity
        w_shade=0.6,
        w_albedo=0.2,
        w_eti=0.2,
        d_cool=450.0,                  # Park cooling radius: 450 m
        green_area_threshold_ha=2.0,   # InVEST threshold: 2 ha
        t_ref=22.0,                    # Rural reference temp (°C)
        uhi_max=4.5,                   # UHI intensity (°C)
        air_blending_distance=600.0,   # Air mixing radius (m)
        calc_wbgt=True,
        avg_rel_humidity=45.0,
        scale=30.0,
    )

    results = model.compute(aoi=aoi, et_max_val=8.0)

    # 6. Export or sample statistics
    mean_stats = results.reduceRegion(
        reducer=ee.Reducer.mean(),
        geometry=aoi,
        scale=30,
        maxPixels=1e8,
    )
    return mean_stats.getInfo()
