# from edelassim.postprocess_surfex.pro import postprocess_pro
# Preprocess simulation output (assimilation of snow depth in this case)
import glob
from datetime import timedelta

import numpy as np
import pandas as pd
import xarray as xr
from pandas import date_range
from pyproj import CRS

simulation_folder = "/home/imperatoren/work/edelweiss_assimilation/simulations/edelweiss/grandesrousses250m/open_loop"
output_folder = "/home/imperatoren/work/edelweiss_assimilation/simulations/postprocess/grandesrousses250m/open_loop/"
filename = "all_members/pro/PRO_GrandesRousses250m_2021080206_2022080106.nc.nc"
output_file = f"{output_folder}/{filename}"


def postprocess_pro(simulation_folder: str, output_file: str | None = None) -> xr.Dataset:

    member_folders = sorted(glob.glob(f"{simulation_folder}/mb*"))
    member_simulations = []
    member_numbers = []
    for member_folder in member_folders:
        member_all_period = xr.open_mfdataset(
            sorted(glob.glob(f"{member_folder}/pro/*.nc")), concat_dim="time", combine="nested"
        )
        member_all_period = member_all_period.resample(time="1d").nearest()
        member_simulations.append(member_all_period)
        member_numbers.append(int(member_folder.split("/")[-1][2:]))
    # all_edel = all_edel.assign_coords({"member": np.arange(17)})

    # date_range(all_edel.coords['time'][0])

    # all_edel_simplified = all_edel.sel(time=time_sampling)
    all_edel = xr.concat(member_simulations, dim=pd.Index(member_numbers, name="member"), coords="all")
    all_edel = all_edel.drop_vars("Projection_Type")
    all_edel = all_edel.rename({"xx": "x", "yy": "y"})
    all_edel = all_edel.rio.write_crs(CRS.from_epsg(2154)).rio.write_coordinate_system()
    if output_file is not None:
        all_edel.to_netcdf(output_file)
    return all_edel


sd_analysis = postprocess_pro(simulation_folder=simulation_folder, output_file=output_file)
