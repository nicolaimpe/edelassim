import xarray as xr
from geospatial_grid.georeferencing import georef_netcdf_rioxarray
from pyproj import CRS

from edelassim.bdclim import find_station_locations
from edelassim.observations import valid_snow_cover_fraction_s2, valid_snow_cover_fraction_viirs_mf

# Snowline plots
LABELS = {"s2": "Sentinel-2", "viirs": "VIIRS", "edel_ol": "Edelweiss OL", "edel_an": "Edelweiss assim"}
COLORS = {"s2": "black", "viirs": "red", "edel_ol": "blue", "edel_an": "magenta"}


xpid = "xpid_placeholder"
working_folder = "/home/imperatoren/work/edelweiss_assimilation/"
simulation_folder = f"{working_folder}/simulations/postprocess"
edelweiss_openloop_folder = f"{simulation_folder}/grandesrousses250m/open_loop/"
edelweiss_analysis_folder = f"{simulation_folder}/reanalysis/{xpid}"
observation_folder = f"{working_folder}/observations/grandesrousses250m/"
topography_data_folder = f"{working_folder}/data/grandesrousses250m/auxiliary/topography/"
landcover_folder = f"{working_folder}/data/grandesrousses250m/auxiliary/"
forest_mask_path = f"{landcover_folder}/forest_mask/forest_mask_corine_grandesrousses_max.nc"

edelweiss_openloop_file = f"{edelweiss_openloop_folder}/spatial.nc"
edelweiss_analysis_file = f"{edelweiss_analysis_folder}/spatial.nc"
viirs_file = f"{observation_folder}/meteofrance/spatial.nc"
s2_path = f"{observation_folder}/s2/spatial.nc"
bdclim_filepath = f"{observation_folder}/bdclim/bdclim.nc"
pleiades_path = f"{observation_folder}/pleiades_grandesrousses_all.nc"

s2_snowline_path = f"{observation_folder}/s2/snowline_parametrization.nc"
edelweiss_openloop_snowline_file = f"{edelweiss_openloop_folder}/snowline_parametrization.nc"
edelweiss_analysis_snowline_file = f"{edelweiss_analysis_folder}/snowline_parametrization.nc"
viirs_snowline_file = f"{observation_folder}/meteofrance/snowline_parametrization.nc"

dem_file = f"{topography_data_folder}/250m/DEM_GR_L93_250m.tif"
glacier_mask_path = f"{landcover_folder}/glacier_mask/glacier_mask_glims_2022_grandesrousses.nc"

glacier_mask = xr.open_dataset(glacier_mask_path).data_vars["__xarray_dataarray_variable__"]
forest_mask = xr.open_dataset(forest_mask_path).sel(band=1).data_vars["__xarray_dataarray_variable__"]
mask = glacier_mask + forest_mask


def get_spatial_datasets(actual_xpid: str) -> tuple[xr.Dataset, ...]:
    actual_edelweiss_analysis_path = edelweiss_analysis_file.replace("xpid_placeholder", actual_xpid)
    edelweiss_ol = xr.open_dataset(edelweiss_openloop_file).where(1 - mask)
    edelweiss_an = xr.open_dataset(actual_edelweiss_analysis_path).where(1 - mask)
    viirs = valid_snow_cover_fraction_viirs_mf(xr.open_dataset(viirs_file).where(1 - mask))
    # dem = xr.open_dataarray(dem_file)

    s2 = xr.open_dataset(s2_path).data_vars["snow_cover_fraction"]
    mask_20m = georef_netcdf_rioxarray(mask, crs=CRS.from_epsg(2154)).rio.reproject_match(s2)
    s2 = valid_snow_cover_fraction_s2(s2.where(1 - mask_20m))
    pleiades = xr.open_dataset(pleiades_path)
    bdclim = xr.open_dataset(bdclim_filepath)
    return edelweiss_ol, edelweiss_an, viirs, s2, pleiades, bdclim


def get_snowlines(actual_xpid: str) -> tuple[xr.Dataset, ...]:
    actual_edelweiss_analysis_path = edelweiss_analysis_snowline_file.replace("xpid_placeholder", actual_xpid)
    snowline_openloop_edel_ds = xr.open_dataset(edelweiss_openloop_snowline_file).sel(slope="8 - 30")
    snowline_analysis_edel_ds = xr.open_dataset(actual_edelweiss_analysis_path).sel(slope="8 - 30")
    snowline_viirs_ds = xr.open_dataset(viirs_snowline_file).sel(slope="8 - 30")
    snowline_s2_ds = xr.open_dataset(s2_snowline_path).sel(slope="8 - 30")
    return snowline_openloop_edel_ds, snowline_analysis_edel_ds, snowline_viirs_ds, snowline_s2_ds
