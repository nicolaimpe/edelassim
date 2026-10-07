from datetime import datetime

import numpy as np
import pandas as pd
import xarray as xr
from geospatial_grid.georeferencing import georef_netcdf_rioxarray
from geospatial_grid.gsgrid import GSGrid


def duplicated_particles_from_part_file(part_file: str) -> xr.DataArray:
    date_str = part_file[-14:-4]
    date = datetime.strptime(date_str, "%Y%m%d%H")
    part_df = pd.read_csv(part_file, header=None, usecols=np.arange(0, 17))
    part_da = xr.DataArray(
        data=np.expand_dims(part_df.values, axis=0),
        dims=(
            "time",
            "n_points",
            "member",
        ),
        coords={"time": [date], "n_points": part_df.index, "member": part_df.columns},
    )
    return part_da


def alpha_from_file(alpha_file: str) -> xr.DataArray:
    date_str = alpha_file[-14:-4]
    date = datetime.strptime(date_str, "%Y%m%d%H")
    alpha_df = pd.read_csv(alpha_file, header=None)
    alpha_da = xr.DataArray(
        data=np.expand_dims(alpha_df.values[0][:-1], axis=0),
        dims=(
            "time",
            "n_points",
        ),
        coords={"time": [date], "n_points": alpha_df.columns[:-1]},
    )
    return alpha_da


def concatenate_multiple_assimilation(file_list: list[str], read_function: callable) -> xr.DataArray:
    part_das = []
    for file in sorted(file_list):
        part_das.append(read_function(file))
    part_da = xr.concat(part_das, dim="time").sortby("time")
    return part_da


def duplicated_particles_multiple_assimilation(part_files: list[str]) -> xr.DataArray:
    return concatenate_multiple_assimilation(file_list=part_files, read_function=duplicated_particles_from_part_file)


def alpha_multiple_assimilation(alpha_files: list[str]) -> xr.DataArray:
    return concatenate_multiple_assimilation(file_list=alpha_files, read_function=alpha_from_file)


def duplicated_particles_to_grid(duplicated_particles_da: xr.DataArray, grid: GSGrid) -> xr.DataArray:
    out_da_coords = {}
    out_da_coords.update({"y": grid.ycoords, "x": grid.xcoords})
    out_da_coords.update({"time": duplicated_particles_da.coords["time"], "member": duplicated_particles_da.coords["member"]})
    arr = duplicated_particles_da.transpose("n_points", "time", "member").data
    out_da_values_flipped_y = arr.reshape(
        (grid.height, grid.width, duplicated_particles_da.sizes["time"], duplicated_particles_da.sizes["member"])
    )
    out_da_values = np.flipud(out_da_values_flipped_y)

    out_da = xr.DataArray(data=out_da_values, coords=out_da_coords, dims=("y", "x", "time", "member"))
    return georef_netcdf_rioxarray(data_array=out_da, crs=grid.crs)


def alpha_to_grid(alpha_da: xr.DataArray, grid: GSGrid) -> xr.DataArray:
    out_da_coords = {}
    out_da_coords.update({"y": grid.ycoords, "x": grid.xcoords})
    out_da_coords.update({"time": alpha_da.coords["time"]})
    arr = alpha_da.transpose("n_points", "time").data
    out_da_values_flipped_y = arr.reshape((grid.height, grid.width, alpha_da.sizes["time"]))
    out_da_values = np.flipud(out_da_values_flipped_y)

    out_da = xr.DataArray(data=out_da_values, coords=out_da_coords, dims=("y", "x", "time"))
    return georef_netcdf_rioxarray(data_array=out_da, crs=grid.crs)
