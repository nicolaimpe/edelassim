from datetime import datetime

from ndsi_fsc_calibration.regrid import S2TheiaRegrid

from edelassim.evaluations import GrandesRoussesGrid20m

if __name__ == "__main__":
    folder = "/home/imperatoren/work/edelweiss_assimilation/data/grandesrousses/snow_cover/s2_theia/LIS_FSC_PREOP"
    aoi_files = "/home/imperatoren/work/edelweiss_assimilation/data/grandesrousses/auxiliary/vectorial/grandesrousses_bbox.shp"
    output_folder = "/home/imperatoren/work/edelweiss_assimilation/observations/granderousses/s2"
    grid = GrandesRoussesGrid20m()

    # Mosaic Sentinel-2 tiles
    regridder = S2TheiaRegrid(output_grid=grid, data_folder=folder, output_folder=output_folder)
    out_dataset = regridder.create_time_series(
        roi_shapefile=aoi_files,
        start_date=datetime(year=2021, month=8, day=1),
        end_date=datetime(year=2022, month=7, day=31),
    )
