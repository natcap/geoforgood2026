"""
InVEST Urban Nature Access (UNA) Model - Python Google Earth Engine (GEE) Implementation
======================================================================================

MODEL DESCRIPTION (For AI Agents & Researchers)
------------------------------------------------
The InVEST Urban Nature Access (UNA) model evaluates the spatial distribution of 
urban nature supply relative to human population demand. It calculates urban nature 
availability across a continuous landscape using a modified Two-Step Floating 
Catchment Area (2SFCA) methodology. 

By modeling walking accessibility rather than simple straight-line buffers, it accounts 
for competition among local residents for shared green spaces. This tool is critical 
for evaluating environmental justice, green infrastructure equity, and urban planning policies.

Methodological Steps:
1. Nature Area Mapping (S_j): Identifies the urban nature supply value for each pixel 
   by multiplying its area by a biophysical "nature proportion" mapped from LULC types.
2. Step 1 (Supply-to-Population Ratio, R_j): Calculates population demand within a 
   floating catchment (defined by `search_radius` d_0) around each nature pixel. 
   Population count is weighted using a spatial distance-decay function (f(d)) to 
   simulate decreasing travel probability. The raw nature area is divided by this 
   weighted service-load to produce a localized provider-to-population ratio.
3. Step 2 (Per-Capita Supply, A_i): Sums the distance-weighted R_j values of all 
   accessible nature pixels within the catchment of each population pixel. This represents 
   the true available green space (m² per person) allocated to a resident at that cell.
4. Balance Analysis (Bal_i): Computes the difference between local per-capita supply (A_i) 
   and a specified policy demand target (g_cap) to determine surplus or deficit.

INPUTS DESCRIPTION
------------------
To run this model, the following inputs are consumed or generated:

*   `aoi_bbox` (list of float):
    Bounding box defined as [West, South, East, North] in decimal degrees. 
    Defines the spatial boundary of the analysis.

*   `search_radius` (float, meters):
    The maximum catchment/walking distance (d_0). Represents the physical threshold 
    residents are willing to travel to access nature (typically 300m - 1000m).

*   `decay_type` (string):
    The spatial impedance function modeling travel behavior over distance. Choices:
    - 'dichotomy': Flat circular buffer (all pixels within d_0 contribute equally; weight = 1).
    - 'exponential': Exponentially decreasing weight: weight = exp(-d / d_0).
    - 'gaussian': Normal distribution with sigma = d_0 / 3: weight = exp(-0.5 * (d / sigma)^2).
    - 'density': Epanechnikov-like kernel: weight = 0.75 * (1 - (d / d_0)^2).

*   `demand_per_capita` (float, m² per person):
    The target urban nature policy standard (g_cap). Represents the desired minimum 
    green space allocated per resident (e.g., WHO guidelines recommend 9 - 30 m²/capita).

*   `scale` (float, meters):
    The output spatial resolution. Usually matches the finest input dataset (typically 
    10m for ESA WorldCover or 30m for Landsat-based LULC).

*   `lulc_nature_weights` (dict):
    A biophysical dictionary mapping integer categorical land use/land cover codes 
    to an urban nature fraction float [0.0 - 1.0]. A value of 1.0 represents entirely 
    natural surface (e.g. dense forest) while 0.0 represents zero nature benefit (e.g. concrete).

*   `population_image` (ee.Image):
    A raster representing human population count per pixel. Typically sourced from 
    WorldPop or GHSL, and resampled to match the LULC analysis scale.

Provides a single function call interface to execute the model and retrieve
results as Python dictionaries.
"""

import math
import ee
import numpy as np

class UrbanNatureAccessGEE:

  def __init__(
      self,
      search_radius=500.0,
      decay_type="exponential",
      demand_per_capita=30.0,
      scale=10.0,
  ):
    self.search_radius = float(search_radius)
    self.decay_type = decay_type.lower()
    self.demand_per_capita = float(demand_per_capita)
    self.scale = float(scale)
    self.kernel = self._build_decay_kernel()

  def _build_decay_kernel(self):
    """Builds the spatial decay convolution kernel."""
    if self.decay_type == "dichotomy":
      return ee.Kernel.circle(
          radius=self.search_radius, units="meters", normalize=False
      )

    radius_px = int(math.ceil(self.search_radius / self.scale))
    y, x = np.ogrid[-radius_px : radius_px + 1, -radius_px : radius_px + 1]
    dist_m = np.sqrt(x**2 + y**2) * self.scale
    mask = dist_m <= self.search_radius

    if self.decay_type == "exponential":
      weights = np.where(mask, np.exp(-dist_m / self.search_radius), 0.0)
    elif self.decay_type == "gaussian":
      sigma = self.search_radius / 3.0
      weights = np.where(mask, np.exp(-0.5 * (dist_m / sigma) ** 2), 0.0)
    elif self.decay_type == "density":
      weights = np.where(
          mask, 0.75 * (1.0 - (dist_m / self.search_radius) ** 2), 0.0
      )
    else:
      raise ValueError(f"Unknown decay type: {self.decay_type}")

    kernel_matrix = weights.tolist()
    k_size = len(kernel_matrix)
    return ee.Kernel.fixed(
        width=k_size,
        height=k_size,
        weights=kernel_matrix,
        x=radius_px,
        y=radius_px,
        normalize=False,
    )

  def compute_nature_area(self, lulc_image, lulc_nature_dict):
    from_codes = list(lulc_nature_dict.keys())
    to_props = [float(lulc_nature_dict[k]) for k in from_codes]

    nature_fraction = lulc_image.remap(
        from_codes, to_props, defaultValue=0.0
    ).rename("nature_fraction")
    pixel_area_m2 = ee.Image.pixelArea()
    return (
        nature_fraction.multiply(pixel_area_m2)
        .rename("nature_area")
        .updateMask(nature_fraction.gt(0))
    )

  def run(self, lulc_image, population_image, lulc_nature_dict, aoi=None):
    """Runs the 2SFCA algorithm and returns a multi-band ee.Image."""
    nature_area = self.compute_nature_area(
        lulc_image, lulc_nature_dict
    ).unmask(0)
    pop = population_image.unmask(0)

    # Step 1: Distance-weighted population within catchment of nature pixels
    pop_weighted = pop.convolve(self.kernel).rename("pop_weighted")

    # Supply ratio R_j = S_j / Pop_weighted
    ratio_r = (
        nature_area.divide(pop_weighted)
        .where(pop_weighted.lte(0).Or(nature_area.lte(0)), 0)
        .rename("supply_ratio")
    )

    # Step 2: Per-capita supply (A_i) by convolving R_j back over population catchment
    supply_per_capita = ratio_r.convolve(self.kernel).rename(
        "supply_per_capita"
    )
    accessible_nature = nature_area.convolve(self.kernel).rename(
        "accessible_nature"
    )

    # Demand and Balance
    demand_total = pop.multiply(self.demand_per_capita).rename("demand_total")
    balance_per_capita = (
        supply_per_capita.subtract(self.demand_per_capita)
        .where(pop.lte(0), 0)
        .rename("balance_per_capita")
    )
    balance_total = balance_per_capita.multiply(pop).rename("balance_total")

    undersupplied_pop = pop.where(balance_per_capita.gte(0), 0).rename(
        "undersupplied_pop"
    )
    oversupplied_pop = pop.where(balance_per_capita.lt(0), 0).rename(
        "oversupplied_pop"
    )

    result = ee.Image.cat([
        nature_area.rename("nature_area_m2"),
        pop.rename("population"),
        accessible_nature,
        supply_per_capita,
        demand_total,
        balance_per_capita,
        balance_total,
        undersupplied_pop,
        oversupplied_pop,
    ])

    if aoi:
      result = result.clip(aoi)
    return result

  def to_summary_dict(self, model_outputs, aoi):
    """Reduces the spatial raster to a summary statistics dictionary."""
    sum_bands = [
        "nature_area_m2",
        "population",
        "demand_total",
        "balance_total",
        "undersupplied_pop",
        "oversupplied_pop",
    ]
    sums = model_outputs.select(sum_bands).reduceRegion(
        reducer=ee.Reducer.sum(), geometry=aoi, scale=self.scale, maxPixels=1e9
    )

    pop_mask = model_outputs.select("population").gt(0)
    means = (
        model_outputs.select(["supply_per_capita", "balance_per_capita"])
        .updateMask(pop_mask)
        .reduceRegion(
            reducer=ee.Reducer.mean(),
            geometry=aoi,
            scale=self.scale,
            maxPixels=1e9,
        )
    )

    combined = sums.combine(means)

    pop_val = ee.Number(combined.get("population"))
    under_pop = ee.Number(combined.get("undersupplied_pop"))
    pct_undersupplied = ee.Algorithms.If(
        pop_val.gt(0), under_pop.divide(pop_val).multiply(100.0), 0
    )

    final_dict = combined.set("pct_population_undersupplied", pct_undersupplied)
    return final_dict.getInfo()

  def to_pixel_dict(self, model_outputs, aoi, bands=None):
    """Extracts raw pixel values as a dictionary of 2D NumPy arrays."""
    image_to_fetch = model_outputs.select(bands) if bands else model_outputs
    raw_data = image_to_fetch.sampleRectangle(
        region=aoi, defaultValue=0
    ).getInfo()
    properties = raw_data.get("properties", {})
    return {band: np.array(matrix) for band, matrix in properties.items()}


# =============================================================================
# Reusable Function Call Entry Point
# =============================================================================
def run_urban_nature_access(
    aoi_bbox,
    search_radius=500.0,
    decay_type="exponential",
    demand_per_capita=30.0,
    scale=10.0,
    lulc_nature_weights=None,
    pixel_bands_to_extract=None,
    gcp_project='sdss-natcap-geoforgood26'
):
  """Executes the InVEST Urban Nature Access model over a specified bounding box

  and returns the results as Python dictionaries.

  Parameters:
      aoi_bbox (list): Bounding box coordinates [west, south, east, north].
      search_radius (float): Pedestrian catchment travel threshold in meters.
      decay_type (str): Spatial impedance function ('dichotomy', 'exponential',
        'gaussian', 'density').
      demand_per_capita (float): Required greenspace target per capita in m2.
      scale (float): Execution resolution in meters.
      lulc_nature_weights (dict, optional): Custom weights mapping for LULC
        classes.
      pixel_bands_to_extract (list, optional): List of specific raster bands to
        retrieve as numpy arrays.

  Returns:
      tuple: (summary_metrics_dict, pixel_arrays_dict)
  """
  # Initialize Earth Engine
  ee.Authenticate()
  ee.Initialize(project=gcp_project)

  aoi = ee.Geometry.BBox(*aoi_bbox)

  # Default to ESA WorldCover classes if no weight map is supplied
  if lulc_nature_weights is None:
    lulc_nature_weights = {
        10: 1.0,  # Trees
        20: 0.9,  # Shrubland
        30: 0.8,  # Grassland
        40: 0.2,  # Cropland
        50: 0.05,  # Built-up
        60: 0.0,  # Bare
        80: 0.5,  # Water
        90: 1.0,  # Wetland
    }

  # Load LULC Dataset (ESA WorldCover 10m)
  worldcover = (
      ee.ImageCollection("ESA/WorldCover/v200").first().select("Map").clip(aoi)
  )

  # Load & Resample Population Dataset (WorldPop 100m -> 10m)
  worldpop = (
      ee.ImageCollection("WorldPop/GP/100m/pop")
      .filter(ee.Filter.date("2020-01-01", "2021-01-01"))
      .first()
      .select("population")
      .clip(aoi)
  )
  pop_resampled = (
      worldpop.resample("bilinear")
      .reproject(crs=worldcover.projection(), scale=scale)
      .divide((100.0 / scale) ** 2)
  )

  # Initialize and Run Model
  model = UrbanNatureAccessGEE(
      search_radius=search_radius,
      decay_type=decay_type,
      demand_per_capita=demand_per_capita,
      scale=scale,
  )

  model_outputs = model.run(
      lulc_image=worldcover,
      population_image=pop_resampled,
      lulc_nature_dict=lulc_nature_weights,
      aoi=aoi,
  )

  # Generate Output Dictionaries
  summary_dict = model.to_summary_dict(model_outputs, aoi)
  pixel_dict = model.to_pixel_dict(
      model_outputs, aoi, bands=pixel_bands_to_extract
  )

  return summary_dict, pixel_dict
