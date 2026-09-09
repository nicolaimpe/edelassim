import xarray as xr
from geospatial_grid.gsgrid import GSGrid
from geospatial_grid.reprojections import reproject_using_grid
from ndsi_fsc_calibration.snow_cover_products import S2_CLASSES
from pyproj import CRS
from rasterio.enums import Resampling

# We define Météo-France class encoding via a dictio
METEOFRANCE_CLASSES = {
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
        nodata=METEOFRANCE_CLASSES["fill"][0],
    )

    # Tricky forest with snow when resampling using average
    # Whenever a resampled pixel includes forest with snow mask, a quantitative estimation connot be performed unless we choose a FSC value for forest with snow
    # The solution would be to resample forest with snow using max, but this is problematic when forest with snow is next to no snow because it increases the snow detections
    # Therefore we set it to 50% FSC (which means 100 in meteofrance encoding).
    # The contingency analysis will not be biased. The quantitative analysis will be more uncertain and perhaps biaised. The recommendation is to use a forest mask resampled with max for quantitative analysis
    resampled_bilinear = reproject_using_grid(
        meteofrance_snow_cover.where(meteofrance_snow_cover <= METEOFRANCE_CLASSES["snow_cover"][-1], 0).astype("f4"),
        output_grid=output_grid,
        resampling_method=Resampling.bilinear,
    )

    resampled_nearest = reproject_using_grid(
        meteofrance_snow_cover,
        output_grid=output_grid,
        resampling_method=Resampling.nearest,
    )

    water_mask = resampled_nearest == METEOFRANCE_CLASSES["water"][0]

    cloud_mask = resampled_max == METEOFRANCE_CLASSES["clouds"][0]
    nodata_mask = resampled_max == METEOFRANCE_CLASSES["nodata"][0]

    invalid_mask = cloud_mask | nodata_mask

    # We exclude these values from the next resampling operations
    valid_qualitative_mask = water_mask
    out_snow_cover = resampled_bilinear.where(valid_qualitative_mask == False, resampled_nearest)
    out_snow_cover = out_snow_cover.where(invalid_mask == False, resampled_max)
    out_snow_cover = out_snow_cover.rio.write_nodata(METEOFRANCE_CLASSES["nodata"][0])
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


def scale_virrs(image_sat: xr.DataArray) -> xr.DataArray:
    return image_sat.where(image_sat <= 200) / 200


def find_clear_dates_s2(snow_cover_s2: xr.DataArray):
    # Determine which days are clear for observations
    n_pixels_area = snow_cover_s2.sizes["x"] * snow_cover_s2.sizes["y"]
    n_data_pixel = snow_cover_s2.count(dim=("x", "y"))
    cloud_mask_s2 = snow_cover_s2 >= S2_CLASSES["clouds"][0]
    # Quick criterium to determine whether a Sentinel-2 image is exploitable
    cloud_flag_s2 = (cloud_mask_s2.sum(dim=("x", "y")) / n_data_pixel > 0.8) | (n_data_pixel / n_pixels_area < 0.3)
    good_dates_s2 = cloud_flag_s2.where(cloud_flag_s2 == 0, drop=True).time.values
    good_dates_s2 = [date.astype("M8[ms]").astype("O") for date in good_dates_s2]
    return good_dates_s2


def find_clear_dates_viirs(snow_cover_viirs: xr.DataArray):
    n_data_pixel = snow_cover_viirs.count(dim=("x", "y"))

    # cloud_mask_viirs = np.isnan(snow_cover_viirs)
    cloud_flag_viirs = (n_data_pixel / (snow_cover_viirs.sizes["x"] * snow_cover_viirs.sizes["y"])) < 0.50
    good_dates_viirs = cloud_flag_viirs.where(cloud_flag_viirs == 0, drop=True).time.values
    good_dates_viirs = [date.astype("M8[ms]").astype("O") for date in good_dates_viirs]

    return good_dates_viirs


def valid_snow_cover_fraction_viirs_mf(viirs_mf_data: xr.DataArray) -> xr.DataArray:
    # satellite_fsc_data = viirs_mf_data.where(viirs_mf_data != METEOFRANCE_CLASSES["water"], 0)
    valid_data_mask = viirs_mf_data < METEOFRANCE_CLASSES["water"]
    valid_snow_cover_fraction = viirs_mf_data.where(valid_data_mask) / METEOFRANCE_CLASSES["snow_cover"][-1]
    return valid_snow_cover_fraction


def valid_snow_cover_fraction_s2(sentinel2_fsc_data: xr.DataArray):
    valid_data_mask = sentinel2_fsc_data < S2_CLASSES["clouds"]
    valid_snow_cover_fraction = sentinel2_fsc_data.where(valid_data_mask) / S2_CLASSES["snow_cover"][-1]
    return valid_snow_cover_fraction
