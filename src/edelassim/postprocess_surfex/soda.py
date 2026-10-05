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


def duplicated_particles_multiple_assimilation(part_files: list[str]) -> xr.DataArray:
    part_das = []
    for part_file in sorted(part_files):
        part_das.append(duplicated_particles_from_part_file(part_file))
    part_da = xr.concat(part_das, dim="time").sortby("time")
    return part_da


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
