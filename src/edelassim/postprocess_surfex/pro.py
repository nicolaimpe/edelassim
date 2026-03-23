# Preprocess simulation output (assimilation of snow depth in this case)
import glob
from datetime import timedelta

import numpy as np
import xarray as xr
from pandas import date_range
from pyproj import CRS


def postprocess_pro(simulation_folder: str, output_file: str | None = None) -> xr.Dataset:

    edel_members_files = sorted(glob.glob(f"{simulation_folder}/mb0*/pro/PRO*.nc"))
    all_edel = xr.open_mfdataset(edel_members_files, concat_dim="member", combine="nested")
    all_edel = all_edel.assign_coords({"member": np.arange(17)})

    # date_range(all_edel.coords['time'][0])
    time_sampling = date_range("2022-02-27", "2022-07-31", freq="D") + timedelta(hours=12)

    all_edel_simplified = all_edel.sel(time=time_sampling)
    all_edel_simplified = all_edel_simplified.drop_vars("Projection_Type")
    all_edel_simplified = all_edel_simplified.rename_dims({"xx": "x", "yy": "y"}).rename({"xx": "x", "yy": "y"})
    all_edel_simplified = all_edel_simplified.rio.write_crs(CRS.from_epsg(2154)).rio.write_coordinate_system()
    if output_file is not None:
        all_edel_simplified.to_netcdf(output_file)
    return all_edel_simplified
