import xarray as xr
from geospatial_grid.gsgrid import GSGrid
from geospatial_grid.reprojections import reproject_using_grid
from pyproj import CRS
from rasterio.enums import Resampling

METEOFRANCE_NEW_CLASSES = {
    "snow_cover": range(1, 201),
    "no_snow": (0,),
    "clouds": (255,),
    "water": (220,),
    "nodata": (230,),
    "fill": (254,),
}


def reprojection_mf_fsc_l3_to_grid(meteofrance_snow_cover: xr.DataArray, output_grid: GSGrid) -> xr.DataArray:
    # Validity "zombie mask": wherever there is at least one non valid pixel, the output grid pixel is set as invalid (<-> cloud)
    # nasa_dataset = nasa_dataset.where(nasa_dataset <= NASA_CLASSES["snow_cover"][-1], NASA_CLASSES["fill"][0])

    resampled_max = reproject_using_grid(
        meteofrance_snow_cover,
        output_grid=output_grid,
        resampling_method=Resampling.max,
        nodata=METEOFRANCE_NEW_CLASSES["fill"][0],
    )

    # Tricky forest with snow when resampling using average
    # Whenever a resampled pixel includes forest with snow mask, a quantitative estimation connot be performed unless we choose a FSC value for forest with snow
    # The solution would be to resample forest with snow using max, but this is problematic when forest with snow is next to no snow because it increases the snow detections
    # Therefore we set it to 50% FSC (which means 100 in meteofrance encoding).
    # The contingency analysis will not be biased. The quantitative analysis will be more uncertain and perhaps biaised. The recommendation is to use a forest mask resampled with max for quantitative analysis
    resampled_bilinear = reproject_using_grid(
        meteofrance_snow_cover.where(meteofrance_snow_cover <= METEOFRANCE_NEW_CLASSES["snow_cover"][-1], 0).astype("f4"),
        output_grid=output_grid,
        resampling_method=Resampling.bilinear,
    )

    resampled_nearest = reproject_using_grid(
        meteofrance_snow_cover,
        output_grid=output_grid,
        resampling_method=Resampling.nearest,
    )

    water_mask = resampled_nearest == METEOFRANCE_NEW_CLASSES["water"][0]

    cloud_mask = resampled_max == METEOFRANCE_NEW_CLASSES["clouds"][0]
    nodata_mask = resampled_max == METEOFRANCE_NEW_CLASSES["nodata"][0]

    invalid_mask = cloud_mask | nodata_mask

    # We exclude these values from the next resampling operations
    valid_qualitative_mask = water_mask
    out_snow_cover = resampled_bilinear.where(valid_qualitative_mask == False, resampled_nearest)
    out_snow_cover = out_snow_cover.where(invalid_mask == False, resampled_max)
    out_snow_cover = out_snow_cover.rio.write_nodata(METEOFRANCE_NEW_CLASSES["nodata"][0])
    return out_snow_cover.astype("u1")


class EdelweissGrandesRoussesGrid(GSGrid):
    """This grid bound correspond to a bounding box including all mountaineous areas over metropolitan France in WGS84 geographic coordinates."""

    def __init__(self):
        super().__init__(
            x0=937750,
            y0=6.46425e06,
            resolution=250,
            width=143,
            height=101,
            crs=CRS.from_epsg(2154),
            name="Edelweiss_GrandesRousses",
        )
