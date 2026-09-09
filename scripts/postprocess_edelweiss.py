"""Process SURFEX simulation output for further analysis"""

import datetime
import glob
import logging
import os

import xarray as xr
from mountain_data_binner.mountain_binner import MountainBinnerConfig

from edelassim.postprocess_surfex.forcing import reorder_forcing_by_duplicated_particles
from edelassim.postprocess_surfex.pro import append_average_member_value, edel_to_snowline, postprocess_pro
from edelassim.postprocess_surfex.soda import duplicated_particles_multiple_assimilation
from edelassim.snowlines import compute_forcing_snowline_parametrization

# Module configuration
logger = logging.getLogger("logger")
logging.basicConfig(level=logging.INFO)


if __name__ == "__main__":
    xpid = "assim_viirs_all_clear_dates_november_2021"
    vconf = "reanalysis"

    simulation_folder = f"/home/imperatoren/work/edelweiss_assimilation/simulations/edelweiss/{vconf}/{xpid}"
    topography_data_folder = "/home/imperatoren/work/edelweiss_assimilation/data/grandesrousses250m/auxiliary/topography/"
    output_folder = f"/home/imperatoren/work/edelweiss_assimilation/simulations/postprocess/{vconf}/{xpid}"
    forcing_folder = "/home/imperatoren/work/edelweiss_assimilation/forcing/"

    # # Regrid and compute mean member
    # snow_depth_edelweiss = postprocess_pro(simulation_folder=simulation_folder, output_file=None)
    # snow_depth_edelweiss = append_average_member_value(snow_depth_edelweiss)
    # snow_depth_edelweiss.to_netcdf(f"{output_folder}/spatial.nc")

    # # EDELWEISS snowline
    # logger.info("Edelweiss preprocessing")
    # # Corresponding for 0.7, 0.5, 0.3, 0.1 m of snow height for 100% snow cover and b=0.11
    # obs_oper_param_list = [1.157, 1.22, 1.367, 2.1]
    # edelweiss = xr.open_dataset(f"{output_folder}/spatial.nc")
    dem_filepath = f"{topography_data_folder}/250m/DEM_GR_L93_250m.tif"
    slope_filepath = f"{topography_data_folder}/250m/SLP_GR_L93_250m.tif"
    aspect_filepath = f"{topography_data_folder}/250m/ASP_GR_L93_250m.tif"

    # edelweiss_snowline_list = []

    # logger.info("Edelweiss snowline calculation")
    topography_paths = MountainBinnerConfig(
        slope_map_path=slope_filepath, aspect_map_path=aspect_filepath, dem_path=dem_filepath
    )
    # for param_a in obs_oper_param_list:
    #     logger.info(f"a = {param_a}")
    #     a_snowline = edel_to_snowline(snow_depth_data=edelweiss, obs_operator_param=param_a, paths=topography_paths)
    #     edelweiss_snowline_list.append(a_snowline)

    # edelweiss_snowline = xr.concat(objs=edelweiss_snowline_list, dim="a")
    # edelweiss_snowline.to_netcdf(f"{output_folder}/snowline_paremetrization.nc")

    # Focing analysis
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
    # Reverse calculation of total snowfall and rainfall to be able toc ompute an average member again
    total_snowfall = forcing_reordered.data_vars["precip_total"] * (1 + forcing_reordered.data_vars["phase"]) / 2
    total_rainfall = forcing_reordered.data_vars["precip_total"] * (1 - forcing_reordered.data_vars["phase"]) / 2
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
    # forcing.close()
    forcing_processed_spatial_dataset.to_netcdf(f"{forcing_folder}/{vconf}/{xpid}/spatial.nc")

    logger.info("Computing analysis forcing snowline parametrization")
    forcing_snowline = compute_forcing_snowline_parametrization(
        forcing_distributed=forcing_processed_spatial_dataset, mountain_binner_config=topography_paths
    )
    out_filepath = f"{forcing_folder}/{vconf}/{xpid}/snowline_parametrization.nc"
    # transformed.to_netcdf(f"{forcing_folder}/snowline_parametrization_onsaitjamais.nc")
    if os.path.exists(out_filepath):
        os.remove(out_filepath)
    forcing_snowline.to_netcdf(out_filepath)
