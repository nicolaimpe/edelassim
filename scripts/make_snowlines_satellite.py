import logging

import xarray as xr
from mountain_data_binner.mountain_binner import MountainBinnerConfig

from edelassim.snowlines import SnowCoverFractionToSnowline, valid_snow_cover_fraction_s2, valid_snow_cover_fraction_viirs_mf

# Module configuration
logger = logging.getLogger("logger")
logging.basicConfig(level=logging.INFO)


if __name__ == "__main__":
    ################################ User inputs #############################################
    s2_folder = "/home/imperatoren/work/edelweiss_assimilation/observations/grandesrousses250m/s2"
    viirs_folder = "/home/imperatoren/work/edelweiss_assimilation/observations/grandesrousses250m/meteofrance/"
    topography_data_folder = "/home/imperatoren/work/edelweiss_assimilation/data/grandesrousses250m/auxiliary/topography/"

    sensor = ("S2", "VIIRS")
    #################################### Sentinel-2 snowline ####################################
    if "S2" in sensor:
        logger.info("Sentinel-2 snowline calculation")

        dem_filepath = f"{topography_data_folder}/20m/DEM_GR_UTM_20m.tif"
        slope_filepath = f"{topography_data_folder}/20m/SLP_GR_UTM_20m.tif"
        aspect_filepath = f"{topography_data_folder}/20m/ASP_GR_UTM_20m.tif"

        sentinel2_image = xr.open_dataset(f"{s2_folder}/spatial.nc").drop_vars("spatial_ref")
        snowline_calculator = SnowCoverFractionToSnowline(
            fsc_image=valid_snow_cover_fraction_s2(sentinel2_image),
            mnt_data_paths=MountainBinnerConfig(
                slope_map_path=slope_filepath, aspect_map_path=aspect_filepath, dem_path=dem_filepath
            ),
        )

    #################################### VIIRS snowline ####################################
    if "VIIRS" in sensor:
        logger.info("VIIRS snowline calculation")
        viirs = xr.open_dataset(f"{viirs_folder}/spatial.nc")

        topography_data_folder = "/home/imperatoren/work/edelweiss_assimilation/data/grandesrousses250m/auxiliary/topography"
        dem_filepath = f"{topography_data_folder}/250m/DEM_GR_L93_250m.tif"
        slope_filepath = f"{topography_data_folder}/250m/SLP_GR_L93_250m.tif"
        aspect_filepath = f"{topography_data_folder}/250m/ASP_GR_L93_250m.tif"
        snowline_calculator = SnowCoverFractionToSnowline(
            fsc_image=valid_snow_cover_fraction_viirs_mf(viirs),
            mnt_data_paths=MountainBinnerConfig(
                slope_map_path=slope_filepath, aspect_map_path=aspect_filepath, dem_path=dem_filepath
            ),
        )
        snowline_calculator.transform(export_path=f"{viirs_folder}/snowline_paremetrization.nc")

    snowline_calculator.transform(export_path=f"{s2_folder}/snowline_paremetrization.nc")
