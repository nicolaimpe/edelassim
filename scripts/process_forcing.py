import glob
import logging
import os

import numpy as np
import xarray as xr
from mountain_data_binner.mountain_binner import MountainBinner, MountainBinnerConfig

from edelassim.postprocess_surfex.pro import append_average_member_value
from edelassim.snowlines import compute_forcing_snowline_parametrization, create_semidistributed_bins

logger = logging.getLogger("logger")
logging.basicConfig(level=logging.INFO)


if __name__ == "__main__":
    forcing_files = glob.glob(
        "/home/imperatoren/work/edelweiss_assimilation/data/grandesrousses250m/forcing/ALPAGA/mb00*/meteo/FORCING_2021-08-01T06:00:00Z_2022-08-01T06:00:00Z.nc"
    )
    topography_data_folder = "/home/imperatoren/work/edelweiss_assimilation/data/grandesrousses250m/auxiliary/topography"
    dem_filepath = f"{topography_data_folder}/250m/DEM_GR_L93_250m.tif"
    slope_filepath = f"{topography_data_folder}/250m/SLP_GR_L93_250m.tif"
    aspect_filepath = f"{topography_data_folder}/250m/ASP_GR_L93_250m.tif"
    output_folder = "/home/imperatoren/work/edelweiss_assimilation/forcing/grandesrousses250m/open_loop"

    logger.info("Loading data")
    forcing = xr.open_mfdataset(sorted(forcing_files), engine="snowtools", combine="nested", concat_dim="member")
    forcing = forcing.rename({"xx": "x", "yy": "y"})
    forcing = forcing.assign_coords({"member": np.arange(0, 17)})

    total_snowfall = forcing.data_vars["Snowf"].resample(time="D").sum()
    total_rainfall = forcing.data_vars["Rainf"].resample(time="D").sum()
    total_snowfall = append_average_member_value(total_snowfall)
    total_rainfall = append_average_member_value(total_rainfall)
    total_precip = total_snowfall + total_rainfall
    phase = xr.Dataset({"phase": (total_snowfall - total_rainfall) / total_precip})

    logger.info("Computing grid daily information")
    forcing_processed_spatial_dataset = xr.Dataset(
        {
            "precip_total": total_precip * 3600,
            "phase": phase.data_vars["phase"],
        }
    )
    forcing.close()
    forcing_processed_spatial_dataset.to_netcdf(f"{output_folder}/spatial.nc")

    logger.info("Computing snowline")
    forcing_daily = xr.open_dataset(f"{output_folder}/spatial.nc")
    topography_filepaths = MountainBinnerConfig(
        slope_map_path=slope_filepath, aspect_map_path=aspect_filepath, dem_path=dem_filepath
    )
    out_filepath = f"{output_folder}/snowline_parametrization.nc"
    transformed = compute_forcing_snowline_parametrization(
        forcing_distributed=forcing_daily, mountain_binner_config=topography_filepaths
    )
    if os.path.exists(out_filepath):
        os.remove(out_filepath)
    transformed.to_netcdf(out_filepath)
