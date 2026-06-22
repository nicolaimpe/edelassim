from datetime import datetime

import xarray as xr
from mountain_data_binner.mountain_binner import MountainBinnerConfig
from ndsi_fsc_calibration.regrid import S2TheiaRegrid

from edelassim.evaluations import GrandesRoussesGrid20m
from edelassim.snowlines import SnowCoverFractionToSnowline

if __name__ == "__main__":
    folder = "/home/imperatoren/work/edelweiss_assimilation/data/grandesrousses/snow_cover/s2_theia/LIS_FSC_PREOP"
    aoi_files = "/home/imperatoren/work/edelweiss_assimilation/data/grandesrousses/auxiliary/vectorial/grandesrousses_bbox.shp"
    output_folder = "/home/imperatoren/work/edelweiss_assimilation/observations/granderousses/s2"
    grid = GrandesRoussesGrid20m()

    # Mosaic Sentinel-2 tiles
    regridder = S2TheiaRegrid(output_grid=grid, data_folder=folder, output_folder=output_folder)
    # out_dataset = regridder.create_time_series(
    #     roi_shapefile=aoi_files,
    #     start_date=datetime(year=2021, month=11, day=1),
    #     end_date=datetime(year=2021, month=12, day=31),
    # )

    topography_data_folder = "/home/imperatoren/work/edelweiss_assimilation/data/grandesrousses/auxiliary/topography/20m"
    dem_filepath = f"{topography_data_folder}/DEM_GR_UTM_20m.tif"
    slope_filepath = f"{topography_data_folder}/SLP_GR_UTM_20m.tif"
    aspect_filepath = f"{topography_data_folder}/ASP_GR_UTM_20m.tif"

    # Sentinel-2 snowline
    # snowline_calculator = SnowCoverFractionToSnowline(
    #     fsc_sat_image_path=xr.open_dataset(f"{output_folder}/regridded.nc").drop_vars("spatial_ref").isel(time=0),
    #     mnt_data_paths=MountainBinnerConfig(
    #         slope_map_path=slope_filepath, aspect_map_path=aspect_filepath, dem_path=dem_filepath
    #     ),
    # )
    # snowline_calculator.transform(export_path=f"{output_folder}/snowline_paremetrization.nc")

    # EDELWEISS snowline
    from edelassim.observation_operators import dickinson

    folder = "/home/imperatoren/work/edelweiss_assimilation/simulations/postprocess/grandesrousses250m/open_loop/all_members"
    edelweiss_ol = (
        xr.open_dataset(f"{folder}/pro/PRO_GrandesRousses250m_2021080206_2022080106.nc")
        .sel(time="2021-11-05")
        .mean(dim="member")
    )
    edelweiss_ol = edelweiss_ol.assign(
        {"snow_cover_fraction": dickinson(sd=edelweiss_ol.data_vars["DSN_T_ISBA"], a=1.2, b=0.11)}
    )
    edelweiss_ol.to_netcdf("edelweiss_fsc.nc")

    topography_data_folder = "/home/imperatoren/work/edelweiss_assimilation/data/grandesrousses/auxiliary/topography/250m"
    dem_filepath = f"{topography_data_folder}/DEM_GR_L93_250m.tif"
    slope_filepath = f"{topography_data_folder}/SLP_GR_L93_250m.tif"
    aspect_filepath = f"{topography_data_folder}/ASP_GR_L93_250m.tif"
    snowline_calculator = SnowCoverFractionToSnowline(
        fsc_image=edelweiss_ol,
        mnt_data_paths=MountainBinnerConfig(
            slope_map_path=slope_filepath, aspect_map_path=aspect_filepath, dem_path=dem_filepath
        ),
    )
    snowline_calculator.transform(export_path=f"{folder}/snowline_paremetrization.nc")

    # VIIRS snowline
    # from edelassim.observation_operators import dickinson

    # folder = "/home/imperatoren/work/edelweiss_assimilation/observations/granderousses/meteofrance/"
    # viirs = xr.open_dataset(f"{folder}/regrid/mf_fsc_l3_jpss1_grandesrousses_wy_2021_2022.nc").sel(time="2021-11-05")

    # topography_data_folder = "/home/imperatoren/work/edelweiss_assimilation/data/grandesrousses/auxiliary/topography/250m"
    # dem_filepath = f"{topography_data_folder}/DEM_GR_L93_250m.tif"
    # slope_filepath = f"{topography_data_folder}/SLP_GR_L93_250m.tif"
    # aspect_filepath = f"{topography_data_folder}/ASP_GR_L93_250m.tif"
    # snowline_calculator = SnowCoverFractionToSnowline(
    #     fsc_image=viirs,
    #     mnt_data_paths=MountainBinnerConfig(
    #         slope_map_path=slope_filepath, aspect_map_path=aspect_filepath, dem_path=dem_filepath
    #     ),
    # )
    # snowline_calculator.transform(export_path=f"{folder}/snowline_paremetrization.nc")
