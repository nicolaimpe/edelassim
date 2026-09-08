# Preprocess simulation output (assimilation of snow depth in this case)

import glob
import logging
from datetime import timedelta

import numpy as np
import pandas as pd
import xarray as xr
from mountain_data_binner.mountain_binner import MountainBinnerConfig
from pandas import date_range
from pyproj import CRS

from edelassim.observation_operators import dickinson
from edelassim.snowlines import SnowCoverFractionToSnowline


def postprocess_pro(simulation_folder: str, output_file: str | None = None) -> xr.Dataset:

    member_folders = sorted(glob.glob(f"{simulation_folder}/mb*"))
    member_simulations = []
    member_numbers = []
    for member_folder in member_folders:
        member_all_period = xr.open_mfdataset(
            sorted(glob.glob(f"{member_folder}/offline/pro/*.nc")), concat_dim="time", combine="nested"
        )

        member_all_period = member_all_period.resample(time="1d").nearest()
        member_simulations.append(member_all_period)
        member_numbers.append(int(member_folder.split("/")[-1][2:]))
    all_edel = xr.concat(member_simulations, dim=pd.Index(member_numbers, name="member"), coords="all")
    all_edel = all_edel.drop_vars("Projection_Type")
    all_edel = all_edel.rename({"xx": "x", "yy": "y"})
    all_edel = all_edel.rio.write_crs(CRS.from_epsg(2154)).rio.write_coordinate_system()
    if output_file is not None:
        all_edel.to_netcdf(output_file)
    return all_edel


def append_average_member_value(snow_depth_edel: xr.DataArray) -> xr.DataArray:

    snow_depth_edel_average = snow_depth_edel.mean(dim="member")
    snow_depth_edel_average = snow_depth_edel_average.assign_coords(member=[-1])
    out = xr.concat([snow_depth_edel_average, snow_depth_edel], dim="member")
    return out


def edel_to_snowline(snow_depth_data: xr.Dataset, obs_operator_param: float, paths: MountainBinnerConfig):
    edelweiss_scf_a = xr.Dataset(
        {"snow_cover_fraction": dickinson(sd=snow_depth_data.data_vars["DSN_T_ISBA"], a=obs_operator_param, b=0.11)}
    )
    snowline_calculator = SnowCoverFractionToSnowline(
        fsc_image=edelweiss_scf_a,
        mnt_data_paths=MountainBinnerConfig(
            slope_map_path=paths.slope_map_path, aspect_map_path=paths.aspect_map_path, dem_path=paths.dem_path
        ),
    )
    edelweiss_snowline = snowline_calculator.transform()
    edelweiss_snowline = edelweiss_snowline.assign_coords({"a": ("a", [obs_operator_param])})
    return edelweiss_snowline
