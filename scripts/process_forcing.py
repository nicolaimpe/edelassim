import glob
import logging
import os

import numpy as np
import xarray as xr
from mountain_data_binner.mountain_binner import MountainBinner, MountainBinnerConfig

from edelassim.snowlines import create_semidistributed_bins

logger = logging.getLogger("logger")
logging.basicConfig(level=logging.INFO)


def daily_min(data: xr.DataArray) -> xr.DataArray:
    return data.resample(time="D").min()


def daily_max(data: xr.DataArray) -> xr.DataArray:
    return data.resample(time="D").max()


def daily_mean(data: xr.DataArray) -> xr.DataArray:
    # print(data.time)
    # return data.resample(time="D").mean()
    return data.resample(time="D").mean()


if __name__ == "__main__":
    forcing_files = glob.glob(
        "/home/imperatoren/work/edelweiss_assimilation/data/grandesrousses250m/forcing/ALPAGA/mb00*/meteo/FORCING_2021-08-01T06:00:00Z_2022-08-01T06:00:00Z.nc"
    )
    topography_data_folder = "/home/imperatoren/work/edelweiss_assimilation/data/grandesrousses250m/auxiliary/topography"
    dem_filepath = f"{topography_data_folder}/250m/DEM_GR_L93_250m.tif"
    slope_filepath = f"{topography_data_folder}/250m/SLP_GR_L93_250m.tif"
    aspect_filepath = f"{topography_data_folder}/250m/ASP_GR_L93_250m.tif"
    output_folder = "/home/imperatoren/work/edelweiss_assimilation/forcing/grandesrousses250m/daily_forcing"

    logging.info("Loading data")
    forcing = xr.open_mfdataset(sorted(forcing_files), engine="snowtools", combine="nested", concat_dim="member")
    forcing = forcing.rename({"xx": "x", "yy": "y"})
    forcing = forcing.assign_coords({"member": np.arange(0, 17)})

    total_snowfall = forcing.data_vars["Snowf"].resample(time="D").sum()
    total_rainfall = forcing.data_vars["Rainf"].resample(time="D").sum()
    total_precip = total_snowfall + total_rainfall
    phase = xr.Dataset({"phase": (total_snowfall - total_rainfall) / total_precip})

    logging.info("Computing grid daily information")
    forcing_processed_spatial_dataset = xr.Dataset(
        {
            "precip_total": total_precip * 3600,
            # "phase_min": phase_min_daily,
            # "phase_max": phase_max_daily,
            "phase": phase.data_vars["phase"],
            # "total_rain": forcing.data_vars["Rainf"].resample(time="D").sum(),
            # "total_snow": forcing.data_vars["Snowf"].resample(time="D").sum(),
        }
    )
    forcing.close()
    forcing_processed_spatial_dataset.to_netcdf(f"{output_folder}/spatial.nc")

    logging.info("Computing snowline")
    forcing_daily = xr.open_dataset(f"{output_folder}/spatial.nc")
    mountain_binner = MountainBinner(
        MountainBinnerConfig(slope_map_path=slope_filepath, aspect_map_path=aspect_filepath, dem_path=dem_filepath)
    )

    bin_dictionary = create_semidistributed_bins(
        mountain_binner=mountain_binner, dem=xr.open_dataarray(dem_filepath), alt_step=100
    )

    def mean_with_log(data: xr.DataArray):
        logging.info(data.altitude.values[0])
        return data.unstack().mean(dim=("x", "y"))

    transformed = mountain_binner.prepare(distributed_data=forcing_daily, bin_dict=bin_dictionary).mean()

    transformed = transformed.drop_vars(("slope", "aspect", "altitude"))
    transformed = mountain_binner.rename_coords(transformed)
    out_filepath = f"{output_folder}/snowline_parametrization.nc"
    transformed.to_netcdf(f"{output_folder}/snowline_parametrization_onsaitjamais.nc")
    if os.path.exists(out_filepath):
        os.remove(out_filepath)
    transformed.to_netcdf(out_filepath)
