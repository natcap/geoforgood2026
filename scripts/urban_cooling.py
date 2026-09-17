"""
InVEST Urban Cooling Model - Google Earth Engine (Python API)
============================================================
Translates the Natural Capital Project's InVEST Urban Cooling Model into
Google Earth Engine for scalable cloud-based urban microclimate modeling.

References:
- InVEST User Guide: https://storage.googleapis.com/releases.naturalcapitalproject.org/invest-userguide/latest/en/urban_cooling_model.html
- Bosch et al. (2021) / Hamel et al. (2024), GMD.
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
