"""Scripts used for generation of an homogenous time stack of satellite data on EDELWEISS grid."""

import glob
import os
from datetime import datetime, timedelta

import xarray as xr
from ndsi_fsc_calibration.regrid import S2TheiaRegrid

from edelassim.evaluations import GrandesRoussesGrid20m
from edelassim.observations import EdelweissGrandesRoussesGrid, reprojection_mf_fsc_l3_to_grid

if __name__ == "__main__":
    sensor = ("S2", "VIIRS")

    #################################### Sentinel-2 regridding ####################################
    if "S2" in sensor:
        folder = "/home/imperatoren/work/edelweiss_assimilation/data/grandesrousses/snow_cover/s2_theia/LIS_FSC_PREOP"
        aoi_files = (
            "/home/imperatoren/work/edelweiss_assimilation/data/grandesrousses/auxiliary/vectorial/grandesrousses_bbox.shp"
        )
        output_folder = "/home/imperatoren/work/edelweiss_assimilation/observations/granderousses/s2"
        grid = GrandesRoussesGrid20m()

        # Mosaic Sentinel-2 tiles
        regridder = S2TheiaRegrid(output_grid=grid, data_folder=folder, output_folder=output_folder)
        out_dataset = regridder.create_time_series(
            roi_shapefile=aoi_files,
            start_date=datetime(year=2021, month=8, day=1),
            end_date=datetime(year=2022, month=7, day=31),
        )
        out_dataset.to_netcdf(f"{output_folder}/spatial.nc")

    #################################### VIIRS regridding ####################################
    if "VIIRS" in sensor:
        input_folder = "/home/imperatoren/work/edelweiss_assimilation/data/france/mf_snow_cover/cms"
        output_file = "/home/imperatoren/work/edelweiss_assimilation/observations/spatial.nc"
        edelweiss_grandesrousses_grid = EdelweissGrandesRoussesGrid()
        viirs_files = glob.glob(f"{input_folder}/*/*/*all.nc")
        viirs_grandesrousses_reprojected = []
        for f in viirs_files:
            viirs = xr.open_dataset(f, engine="rasterio", mask_and_scale=False)
            reproj = reprojection_mf_fsc_l3_to_grid(
                meteofrance_snow_cover=viirs.data_vars["snow_cover_fraction"], output_grid=edelweiss_grandesrousses_grid
            )
            viirs_grandesrousses_reprojected.append(reproj)

        viirs_grandesrousses = xr.concat(viirs_grandesrousses_reprojected, dim="time")
        viirs_grandesrousses.to_netcdf(f"{output_folder}/spatial.nc")
