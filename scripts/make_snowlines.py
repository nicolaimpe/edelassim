import logging
from datetime import datetime

import matplotlib.pyplot as plt

# import IPython
import pandas as pd
import xarray as xr
from geospatial_grid.georeferencing import georef_netcdf_rioxarray
from ipywidgets import Dropdown, interact
from matplotlib import pyplot as plt
from matplotlib.widgets import Slider
from mountain_data_binner.mountain_binner import MountainBinnerConfig
from pyproj import CRS

from edelassim.observation_operators import dickinson
from edelassim.snowlines import SnowCoverFractionToSnowline, valid_snow_cover_fraction_s2, valid_snow_cover_fraction_viirs_mf

# Module configuration
logger = logging.getLogger("logger")
logging.basicConfig(level=logging.INFO)


def edel_to_snowline(snow_depth_data: xr.Dataset, obs_operator_param: float):
    edelweiss_scf_a = xr.Dataset(
        {"snow_cover_fraction": dickinson(sd=snow_depth_data.data_vars["DSN_T_ISBA"], a=obs_operator_param, b=0.11)}
    )
    snowline_calculator = SnowCoverFractionToSnowline(
        fsc_image=edelweiss_scf_a,
        mnt_data_paths=MountainBinnerConfig(
            slope_map_path=slope_filepath, aspect_map_path=aspect_filepath, dem_path=dem_filepath
        ),
    )
    edelweiss_snowline = snowline_calculator.transform()
    edelweiss_snowline = edelweiss_snowline.assign_coords({"a": ("a", [obs_operator_param])})
    return edelweiss_snowline


if __name__ == "__main__":
    ################################ User inputs #############################################
    s2_folder = "/home/imperatoren/work/edelweiss_assimilation/observations/grandesrousses250m/s2"
    edelweiss_folder = (
        "/home/imperatoren/work/edelweiss_assimilation/simulations/postprocess/grandesrousses250m/open_loop/all_members"
    )
    viirs_folder = "/home/imperatoren/work/edelweiss_assimilation/observations/grandesrousses250m/meteofrance/"

    topography_data_folder = "/home/imperatoren/work/edelweiss_assimilation/data/grandesrousses250m/auxiliary/topography/"

    ############################### Snowline penalization function calculation ###################################
    dem_filepath = f"{topography_data_folder}/20m/DEM_GR_UTM_20m.tif"
    slope_filepath = f"{topography_data_folder}/20m/SLP_GR_UTM_20m.tif"
    aspect_filepath = f"{topography_data_folder}/20m/ASP_GR_UTM_20m.tif"

    # Sentinel-2 snowline
    logger.info("Sentinel-2 snowline calculation")
    sentinel2_image = xr.open_dataset(f"{s2_folder}/spatial.nc").drop_vars("spatial_ref")
    snowline_calculator = SnowCoverFractionToSnowline(
        fsc_image=valid_snow_cover_fraction_s2(sentinel2_image),
        mnt_data_paths=MountainBinnerConfig(
            slope_map_path=slope_filepath, aspect_map_path=aspect_filepath, dem_path=dem_filepath
        ),
    )
    snowline_calculator.transform(export_path=f"{s2_folder}/snowline_paremetrization.nc")
    # EDELWEISS snowline
    logger.info("Edelweiss preprocesisng")

    # Corresponding for 0.7, 0.5, 0.3, 0.1 m of snow height for 100% snow cover and b=0.11
    obs_oper_param_list = [1.157, 1.22, 1.367, 2.1]
    edelweiss_ol = xr.open_dataset(f"{edelweiss_folder}/pro/PRO_GrandesRousses250m_2021080206_2022080106.nc")
    dem_filepath = f"{topography_data_folder}/250m/DEM_GR_L93_250m.tif"
    slope_filepath = f"{topography_data_folder}/250m/SLP_GR_L93_250m.tif"
    aspect_filepath = f"{topography_data_folder}/250m/ASP_GR_L93_250m.tif"

    edelweiss_snowline_list = []

    logger.info("Edelweiss snowline calculation")
    for param_a in obs_oper_param_list:
        logger.info(f"a = {param_a}")
        a_snowline = edel_to_snowline(snow_depth_data=edelweiss_ol, obs_operator_param=param_a)
        edelweiss_snowline_list.append(a_snowline)

    edelweiss_snowline = xr.concat(objs=edelweiss_snowline_list, dim="a")
    edelweiss_snowline.to_netcdf(f"{edelweiss_folder}/snowline_paremetrization.nc")
    # # VIIRS snowline
    logger.info("VIIRS snowline calculation")
    viirs = xr.open_dataset(f"{viirs_folder}/spatial.nc")

    topography_data_folder = "/home/imperatoren/work/edelweiss_assimilation/data/grandesrousses250m/auxiliary/topography"
    dem_filepath = f"{topography_data_folder}/250m/DEM_GR_L93_250m.tif"
    slope_filepath = f"{topography_data_folder}/250m/SLP_GR_L93_250m.tif"
    aspect_filepath = f"{topography_data_folder}/250m/ASP_GR_L93_250m.tif"
    snowline_calculator = SnowCoverFractionToSnowline(
        fsc_image=valid_snow_cover_fraction_viirs_mf(viirs),
        mnt_data_paths=MountainBinnerConfig(
            slope_map_path=slope_filepath, aspect_map_path=aspect_filepath, dem_path=dem_filepath
        ),
    )
    # snowline_calculator.transform(export_path=f"{viirs_folder}/snowline_paremetrization.nc")
