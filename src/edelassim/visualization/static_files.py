from glob import glob

import xarray as xr
from geospatial_grid.georeferencing import georef_netcdf_rioxarray
from pyproj import CRS, Transformer

from edelassim.bdclim import (
    extract_bdclim_locations_on_s2,
    extract_bdclim_locations_on_spatial_dataset,
    find_station_locations,
)
from edelassim.observations import (
    find_clear_dates_s2,
    find_clear_dates_viirs,
    valid_snow_cover_fraction_s2,
    valid_snow_cover_fraction_viirs_mf,
)
from edelassim.postprocess_surfex.soda import duplicated_particles_multiple_assimilation

# Snowline plots
LABELS = {"s2": "Sentinel-2", "viirs": "VIIRS", "edel_ol": "OL", "edel_an": "Assim", "bdclim": "In situ"}
COLORS = {"s2": "black", "viirs": "red", "edel_ol": "blue", "edel_an": "magenta", "bdclim": "black"}


xpid = "xpid_placeholder"
working_folder = "/home/imperatoren/work/edelweiss_assimilation/"
simulation_folder = f"{working_folder}/simulations/postprocess"
edelweiss_openloop_folder = f"{simulation_folder}/reanalysis/open_loop/"
edelweiss_analysis_folder = f"{simulation_folder}/reanalysis/{xpid}"
observation_folder = f"{working_folder}/observations/grandesrousses250m/"
topography_data_folder = f"{working_folder}/data/grandesrousses250m/auxiliary/topography/"
landcover_folder = f"{working_folder}/data/grandesrousses250m/auxiliary/"
forest_mask_path = f"{landcover_folder}/forest_mask/forest_mask_corine_grandesrousses_max.nc"

edelweiss_openloop_file = f"{edelweiss_openloop_folder}/spatial.nc"
edelweiss_analysis_file = f"{edelweiss_analysis_folder}/spatial.nc"
viirs_file = f"{observation_folder}/meteofrance/spatial.nc"
s2_path = f"{observation_folder}/s2/spatial.nc"
bdclim_filepath = f"{observation_folder}/bdclim/snow_depth_daily_avg.nc"
pleiades_path = f"{observation_folder}/pleiades_grandesrousses_all.nc"

s2_snowline_path = f"{observation_folder}/s2/snowline_parametrization.nc"
edelweiss_openloop_snowline_file = f"{edelweiss_openloop_folder}/snowline_parametrization.nc"
edelweiss_analysis_snowline_file = f"{edelweiss_analysis_folder}/snowline_parametrization.nc"
viirs_snowline_file = f"{observation_folder}/meteofrance/snowline_parametrization.nc"

dem_file = f"{topography_data_folder}/250m/DEM_GR_L93_250m.tif"
glacier_mask_path = f"{landcover_folder}/glacier_mask/glacier_mask_glims_2022_grandesrousses.nc"
slope_20m_file = f"{topography_data_folder}/250m/SLP_GR_L93_250m.tif"

glacier_mask = xr.open_dataset(glacier_mask_path).data_vars["__xarray_dataarray_variable__"]
forest_mask = xr.open_dataset(forest_mask_path).sel(band=1).data_vars["__xarray_dataarray_variable__"]
mask = glacier_mask + forest_mask
mask_20m = georef_netcdf_rioxarray(mask, crs=CRS.from_epsg(2154)).rio.reproject_match(xr.open_dataset(s2_path))

slope_20m = xr.open_dataarray(slope_20m_file)


def list_satellite_clear_dates(period: slice = slice("2021-11", "2022-06")):
    good_dates_viirs = find_clear_dates_viirs(
        xr.open_dataset(viirs_file).sel(time=period).data_vars["snow_cover_fraction"].where(1 - mask)
    )
    good_dates_s2 = find_clear_dates_s2(
        xr.open_dataset(s2_path).sel(time=period).data_vars["snow_cover_fraction"].where(1 - mask_20m)
    )
    return good_dates_viirs, good_dates_s2


def get_duplicated_particles_assim_dates(acutal_xpid: str) -> tuple[xr.DataArray, xr.DataArray]:
    part_files = glob(f"{working_folder}/simulations/edelweiss/reanalysis/{acutal_xpid}/soda/*PART*")
    duplicated_particles = duplicated_particles_multiple_assimilation(part_files=part_files)
    assim_dates = duplicated_particles.coords["time"]
    return duplicated_particles, assim_dates


def get_spatial_datasets(period: slice = slice("2021-11", "2022-06")) -> tuple[xr.Dataset, ...]:
    edelweiss_ol = xr.open_dataset(edelweiss_openloop_file).where(1 - mask).sel(time=period)
    viirs = valid_snow_cover_fraction_viirs_mf(xr.open_dataset(viirs_file).where(1 - mask)).sel(time=period)
    s2 = xr.open_dataset(s2_path)
    s2 = valid_snow_cover_fraction_s2(s2.where(1 - mask_20m)).sel(time=period)
    pleiades = xr.open_dataset(pleiades_path).sel(time=period)
    bdclim = xr.open_dataset(bdclim_filepath).sel(time=period)
    mask_stations = extract_bdclim_locations_on_spatial_dataset(bdclim, spatial_dataset=mask)
    mask_stations = mask_stations.assign_coords({"x": bdclim.coords["x"], "y": bdclim.coords["y"]})
    bdclim = bdclim.where(1 - mask_stations, drop=True).data_vars["snow_depth"]
    return edelweiss_ol, viirs, s2, pleiades, bdclim


def get_data_assimilation_simulation(actual_xpid: str, period: slice = slice("2021-11", "2022-06")):
    return xr.open_dataset(edelweiss_analysis_file.replace("xpid_placeholder", actual_xpid)).where(1 - mask).sel(time=period)


def get_data_assimilation_snowline(actual_xpid: str, period: slice = slice("2021-11", "2022-06")):
    return xr.open_dataset(edelweiss_analysis_snowline_file.replace("xpid_placeholder", actual_xpid)).sel(slope="8 - 30")


def get_snowlines() -> tuple[xr.Dataset, ...]:
    snowline_openloop_edel_ds = xr.open_dataset(edelweiss_openloop_snowline_file).sel(slope="8 - 30")
    snowline_viirs_ds = xr.open_dataset(viirs_snowline_file).sel(slope="8 - 30")
    snowline_s2_ds = xr.open_dataset(s2_snowline_path).sel(slope="8 - 30")
    return snowline_openloop_edel_ds, snowline_viirs_ds, snowline_s2_ds


def get_snowline_simulation(actual_xpid: str, period: slice = slice("2021-11", "2022-06")):
    return (
        xr.open_dataset(edelweiss_analysis_snowline_file.replace("xpid_placeholder", actual_xpid))
        .sel(slope="8 - 30")
        .sel(time=period)
    )


def get_station_location_data(
    bdclim: xr.Dataset, edelweiss_openloop: xr.Dataset, viirs: xr.Dataset, s2: xr.Dataset
) -> tuple[xr.Dataset, ...]:

    sd_edel_station_openloop = extract_bdclim_locations_on_spatial_dataset(bdclim, spatial_dataset=edelweiss_openloop)
    viirs_station = extract_bdclim_locations_on_spatial_dataset(bdclim, spatial_dataset=viirs)
    s2_station = extract_bdclim_locations_on_s2(bdclim=bdclim, s2=s2)
    return sd_edel_station_openloop, viirs_station, s2_station
