"""Process SURFEX simulation output for further analysis"""

import logging

import xarray as xr
from mountain_data_binner.mountain_binner import MountainBinnerConfig

from edelassim.postprocess_surfex.pro import append_average_member_value, edel_to_snowline, postprocess_pro

# Module configuration
logger = logging.getLogger("logger")
logging.basicConfig(level=logging.INFO)


if __name__ == "__main__":
    xpid = "assim_viirs_all_clear_dates_november_2021"

    simulation_folder = f"/home/imperatoren/work/edelweiss_assimilation/simulations/edelweiss/reanalysis/{xpid}"
    topography_data_folder = "/home/imperatoren/work/edelweiss_assimilation/data/grandesrousses250m/auxiliary/topography/"
    output_folder = f"/home/imperatoren/work/edelweiss_assimilation/simulations/postprocess/reanalysis/{xpid}"

    # Regrid and compute mean member
    snow_depth_edelweiss = postprocess_pro(simulation_folder=simulation_folder, output_file=None)
    snow_depth_edelweiss = append_average_member_value(snow_depth_edelweiss)
    snow_depth_edelweiss.to_netcdf(f"{output_folder}/spatial.nc")

    # EDELWEISS snowline
    logger.info("Edelweiss preprocessing")
    # Corresponding for 0.7, 0.5, 0.3, 0.1 m of snow height for 100% snow cover and b=0.11
    obs_oper_param_list = [1.157, 1.22, 1.367, 2.1]
    edelweiss = xr.open_dataset(f"{output_folder}/spatial.nc")
    dem_filepath = f"{topography_data_folder}/250m/DEM_GR_L93_250m.tif"
    slope_filepath = f"{topography_data_folder}/250m/SLP_GR_L93_250m.tif"
    aspect_filepath = f"{topography_data_folder}/250m/ASP_GR_L93_250m.tif"

    edelweiss_snowline_list = []

    logger.info("Edelweiss snowline calculation")
    topography_paths = MountainBinnerConfig(
        slope_map_path=slope_filepath, aspect_map_path=aspect_filepath, dem_path=dem_filepath
    )
    for param_a in obs_oper_param_list:
        logger.info(f"a = {param_a}")
        a_snowline = edel_to_snowline(snow_depth_data=edelweiss, obs_operator_param=param_a, paths=topography_paths)
        edelweiss_snowline_list.append(a_snowline)

    edelweiss_snowline = xr.concat(objs=edelweiss_snowline_list, dim="a")
    edelweiss_snowline.to_netcdf(f"{output_folder}/snowline_paremetrization.nc")
