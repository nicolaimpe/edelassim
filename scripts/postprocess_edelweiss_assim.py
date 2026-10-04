"""Process SURFEX simulation output for further analysis"""

import datetime
import glob
import logging
import os

import xarray as xr
import yaml
from mountain_data_binner.mountain_binner import MountainBinnerConfig

from edelassim.postprocess_surfex.forcing import reorder_forcing_by_duplicated_particles
from edelassim.postprocess_surfex.pro import (
    append_average_member_value,
    append_median_member_value,
    edel_to_snowline,
    postprocess_pro,
)
from edelassim.postprocess_surfex.soda import duplicated_particles_multiple_assimilation
from edelassim.snowlines import compute_forcing_snowline_parametrization

# Module configuration
logger = logging.getLogger("logger")
logging.basicConfig(level=logging.INFO)


# def get_pro_assim_simulation(vortex_config_file: str, date_begin: str, date_end: str, assim_dates: str, xpid: str):
#     with open(vortex_config_file, "r") as file:
#         # Charger le contenu du fichier en tant que dictionnaire Python
#         config = yaml.safe_load(file)
#     config["experiment"] = xpid
#     config["datebegin"] = [date_begin, *assim_dates[:-1]]
#     config["dateend"]["datebegin"] = {k: v for k, v in zip(config["datebegin"], [*assim_dates, date_end])}
#     return config


# def get_part_file():
#     return
# lftp hendrix
# mirror --exclude-glob mb00*/prep/*/* --exclude-glob mb00*/soda/*/*


# vortex_config_folder = "../config/vortex_configs"
# get_pro_assim_simulation(vortex_config_file=f"{vortex_config_folder}/config_pro_assim.yaml")

# Use wget to pull results
# wget -r -l 0 ftp://imperatoren@hendrix/vortex/edelweiss/reanalysis/assim_viirs_cloudcover07_wy2122_d93_50cm/ -A "*PRO*"
# wget -r -l 0 ftp://imperatoren@hendrix/vortex/edelweiss/reanalysis/assim_viirs_cloudcover07_wy2122_d93_50cm/ -A "*PART*"
if __name__ == "__main__":
    xpid = "assim_viirs_cloudcover07_wy2122_d93_50cm"
    vconf = "reanalysis"

    simulation_folder = f"/home/imperatoren/work/edelweiss_assimilation/simulations/edelweiss/{vconf}/{xpid}"
    topography_data_folder = "/home/imperatoren/work/edelweiss_assimilation/data/grandesrousses250m/auxiliary/topography/"
    output_folder = f"/home/imperatoren/work/edelweiss_assimilation/simulations/postprocess/{vconf}/{xpid}"
    forcing_folder = "/home/imperatoren/work/edelweiss_assimilation/forcing/"

    # ######### EDELWEISS Regrid and compute mean member
    logger.info("Edelweiss postprocessing - regridding")
    snow_depth_edelweiss = postprocess_pro(simulation_folder=simulation_folder, output_file=None)
    snow_depth_edelweiss = append_median_member_value(snow_depth_edelweiss)
    snow_depth_edelweiss = append_average_member_value(snow_depth_edelweiss)
    snow_depth_edelweiss.to_netcdf(f"{output_folder}/spatial.nc")

    
    # ############ EDELWEISS snowline
    logger.info("Edelweiss postprocessing - snowline calculation")
    # Corresponding for inf, 0.7, 0.5, 0.3, 0.1 m of snow height for 100% snow cover and b=0.11
    # obs_oper_param_list = [1, 1.157, 1.22, 1.367, 2.1]
    obs_operator_param = 1.22
    edelweiss = xr.open_dataset(f"{output_folder}/spatial.nc")
    dem_filepath = f"{topography_data_folder}/250m/DEM_GR_L93_250m.tif"
    slope_filepath = f"{topography_data_folder}/250m/SLP_GR_L93_250m.tif"
    aspect_filepath = f"{topography_data_folder}/250m/ASP_GR_L93_250m.tif"

    edelweiss_snowline_list = []

    # # logger.info("Edelweiss snowline calculation")
    topography_paths = MountainBinnerConfig(
        slope_map_path=slope_filepath, aspect_map_path=aspect_filepath, dem_path=dem_filepath
    )
    # for param_b in obs_oper_param_list:
    # logger.info(f"b = {param_b}")
    # b_snowline = edel_to_snowline(snow_depth_data=edelweiss, obs_operator_param=obs_operator_param, paths=topography_paths)
    # edelweiss_snowline_list.append(b_snowline)

    # edelweiss_snowline = xr.concat(objs=edelweiss_snowline_list, dim="b")
    # edelweiss_snowline.to_netcdf(f"{output_folder}/snowline_paremetrization.nc")

    edel_to_snowline(snow_depth_data=edelweiss, obs_operator_param=obs_operator_param, paths=topography_paths).to_netcdf(
        f"{output_folder}/snowline_parametrization.nc"
    )
    ############ Focing analysis
    # Swap forcing memmbers in assimilation window using PART file
    logger.info(
        "Use PART files to calculate an 'analysis' forcing, i.e. reindex forcing according to PART for each assimilation window"
    )

    part_files = glob.glob(f"{simulation_folder}/soda/*PART*")
    duplicated_particles = duplicated_particles_multiple_assimilation(part_files=part_files)

    # This file has at -1 the average member
    forcing = xr.open_dataset(f"{forcing_folder}/grandesrousses250m/open_loop/spatial.nc").sel(member=slice(0, None))
    forcing_reordered = reorder_forcing_by_duplicated_particles(
        forcing=forcing, duplicated_particles=duplicated_particles, start_assim_date="2021-11-01"
    )
    # forcing_reordered = forcing.copy(deep=True)
    # Reverse calculation of total snowfall and rainfall to be able toc ompute an average member again
    total_snowfall = forcing_reordered.data_vars["precip_total"] * (1 + forcing_reordered.data_vars["phase"]) / 2
    total_rainfall = forcing_reordered.data_vars["precip_total"] * (1 - forcing_reordered.data_vars["phase"]) / 2
    total_snowfall = append_median_member_value(total_snowfall)
    total_rainfall = append_median_member_value(total_rainfall)
    total_snowfall = append_average_member_value(total_snowfall)
    total_rainfall = append_average_member_value(total_rainfall)

    total_precip = total_snowfall + total_rainfall
    phase = xr.Dataset({"phase": (total_snowfall - total_rainfall) / total_precip})

    forcing_processed_spatial_dataset = xr.Dataset(
        {
            "precip_total": total_precip,
            "phase": phase.data_vars["phase"],
        }
    )
    forcing_processed_spatial_dataset.to_netcdf(f"{forcing_folder}/postprocess/{vconf}/{xpid}/spatial.nc")

    logger.info("Computing analysis forcing snowline parametrization")
    forcing_snowline = compute_forcing_snowline_parametrization(
        forcing_distributed=forcing_processed_spatial_dataset, mountain_binner_config=topography_paths
    )
    out_filepath = f"{forcing_folder}/postprocess/{vconf}/{xpid}/snowline_parametrization.nc"
    if os.path.exists(out_filepath):
        os.remove(out_filepath)
    forcing_snowline.to_netcdf(out_filepath)
