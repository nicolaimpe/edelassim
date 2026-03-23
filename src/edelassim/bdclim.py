from typing import Tuple

import geopandas as gpd
import numpy as np
import shapely
import xarray as xr
from geospatial_grid.gsgrid import GSGrid
from pyproj import Transformer
from shapely import Point


def find_station_locations(bdclim_dataset: xr.Dataset) -> Tuple[np.ndarray, np.ndarray]:
    transformer = Transformer.from_crs(4326, 2154, always_xy=True)
    x_poste, y_poste = transformer.transform(xx=bdclim_dataset.lon, yy=bdclim_dataset.lat)
    return x_poste, y_poste


def crop_bdclim(bdclim_netcdf_path: str, grid: GSGrid) -> xr.Dataset:
    lon_min, lat_min, lon_max, lat_max = grid.bounds_projected_to_epsg(4326)
    bdclim = xr.open_dataset(bdclim_netcdf_path)
    bdclim_crop = bdclim.where((bdclim.coords["lat"] > lat_min) * (bdclim.coords["lat"] < lat_max), drop=True).where(
        (bdclim.coords["lon"] > lon_min) * (bdclim.coords["lon"] < lon_max), drop=True
    )
    x_poste, y_poste = find_station_locations(bdclim_dataset=bdclim_crop)
    return bdclim_crop.assign_coords({"x": ("num_poste", x_poste), "y": ("num_poste", y_poste)})


def lon_lat_point(ds: xr.Dataset) -> xr.DataArray:
    return xr.DataArray(
        Point(ds.coords["lon"].values[0], ds.coords["lat"].values[0]), coords={"ZS": ds.coords["ZS"].values[0]}
    )


def extract_bdclim_locations_to_shapefile(bdclim_ds: xr.Dataset, export_path: str) -> gpd.GeoDataFrame:
    points = bdclim_ds.groupby("Station_Name").map(lon_lat_point)
    gdf = gpd.GeoDataFrame(
        data={"Station_Name": points.coords["Station_Name"], "ZS": points.coords["ZS"]},
        geometry=points.values,
        crs="EPSG:4326",
    )
    if export_path is not None:
        gdf.to_file(export_path)
    return gdf
