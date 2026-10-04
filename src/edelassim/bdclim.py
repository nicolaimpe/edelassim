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


def extract_bdclim_locations_on_spatial_dataset(bdclim_dataset: xr.Dataset, spatial_dataset: xr.Dataset) -> xr.Dataset:
    x_poste, y_poste = find_station_locations(bdclim_dataset=bdclim_dataset)
    sd_station = bdclim_dataset.where(bdclim_dataset["x"] == x_poste, drop=True).where(
        bdclim_dataset["y"] == y_poste, drop=True
    )
    # Xarray advanced indexing (according to Mistral)
    x_poste_da = xr.DataArray(sd_station.x, dims="num_poste", coords={"num_poste": sd_station.num_poste})
    y_poste_da = xr.DataArray(sd_station.y, dims="num_poste", coords={"num_poste": sd_station.num_poste})
    return spatial_dataset.sel(x=x_poste_da, y=y_poste_da, method="nearest", drop=True)


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


def find_station_imshow_location(x_station: float, y_station: float, grid_coords_x: np.ndarray, grid_coords_y: np.ndarray):
    # To plot over a map plotted via imshow
    col_station = np.abs(grid_coords_x - x_station).argmin()
    row_station = np.abs(grid_coords_y - y_station).argmin()
    x_station_grid_nearest = grid_coords_x[col_station]
    y_station_grid_nearest = grid_coords_y[row_station]
    grid_width = grid_coords_x[-1] - grid_coords_x[0]
    grid_height = grid_coords_y[0] - grid_coords_y[-1]
    x_station_plot_location = col_station + (x_station - x_station_grid_nearest) * len(grid_coords_x) / grid_width
    y_station_plot_location = row_station + (y_station - y_station_grid_nearest) * len(grid_coords_y) / grid_height
    # y_station_plot_location = len(grid_coords_y) - y_station_plot_location
    return x_station_plot_location, y_station_plot_location
