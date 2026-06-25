import glob
from typing import List

import numpy as np
import xarray as xr
from geospatial_grid.georeferencing import georef_netcdf_rioxarray
from pyproj import CRS


def compute_snow_thickness_and_mass_from_prep(
    prep_file: str, slope_file: str, crs: CRS, output_file: str | None = None
) -> float:
    # Merci Bastien
    prep_ds = xr.open_dataset(prep_file)
    slope_da = xr.open_dataarray(slope_file).sel(band=1)

    thickness_layers = []
    mass_layers = []

    for i in range(1, 51):  # Boucle sur les 50 couches
        wsn_var = f"WSN_VEG{i}"  # Masse en kg/m2
        rsn_var = f"RSN_VEG{i}"  # Masse volumique en kg/m3

        if wsn_var in prep_ds and rsn_var in prep_ds:
            thickness_layers.extend(prep_ds[wsn_var].values / prep_ds[rsn_var].values)
            mass_layers.extend(prep_ds[wsn_var].values)
        else:
            raise ValueError(f"Variables {wsn_var} ou {rsn_var} manquantes dans le fichier NetCDF")

    total_thickness = np.nansum(thickness_layers, axis=0)
    total_mass = np.nansum(mass_layers, axis=0)
    thickness_dep_tot = prep_ds.data_vars["DEP_TOT"].values[0]

    x_coords = np.unique(prep_ds.data_vars["XX"].values)
    y_coords = np.unique(prep_ds.data_vars["XY"].values)
    new_shape = (len(y_coords), len(x_coords))
    total_thickness_da = xr.DataArray(
        np.reshape(total_thickness, shape=new_shape), coords={"y": y_coords, "x": x_coords}
    ) / np.cos(np.deg2rad(slope_da))
    total_mass_da = xr.DataArray(np.reshape(total_mass, shape=new_shape), coords={"y": y_coords, "x": x_coords})
    thickness_dep_tot = xr.DataArray(
        np.reshape(thickness_dep_tot, shape=new_shape), coords={"y": y_coords, "x": x_coords}
    ) / np.cos(np.deg2rad(slope_da))

    out_dataset = xr.Dataset({"snow_depth": total_thickness_da, "swe": total_mass_da, "dep_tot": thickness_dep_tot})
    out_dataset = out_dataset.reindex(y=total_thickness_da.coords["y"].sortby("y", ascending=False))
    # projection sur la VERTICALE pour être coherent avec sortir des fichiers PRO
    out_dataset = out_dataset
    out_dataset = georef_netcdf_rioxarray(out_dataset, crs=crs)  # add georeferencing
    if output_file is not None:
        out_dataset.to_netcdf(output_file)
    return out_dataset


def compute_all_members_snow_tickness_and_mass(prep_files: List[str], slope_file: str) -> xr.Dataset:

    prep_an_data_list = [
        compute_snow_thickness_and_mass_from_prep(prep_file=prep_an_file, slope_file=slope_file, crs=CRS.from_epsg(2154))
        for prep_an_file in prep_files
    ]
    prep_an = xr.concat(prep_an_data_list, dim="member").reindex(
        member=np.arange(1, len(prep_files) + 1)
    )  # .rename("snow_depth")
    return prep_an
