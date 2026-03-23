from turtle import pd
from typing import List

import earthaccess
import numpy as np
import rasterio
import rioxarray
import xarray as xr
from geospatial_grid.georeferencing import georef_netcdf_rioxarray
from geospatial_grid.gsgrid import GSGrid
from geospatial_grid.reprojections import reproject_using_grid
from pyresample import kd_tree
from pyresample.geometry import AreaDefinition, SwathDefinition
from rasterio.enums import Resampling

METEOFRANCE_ARCHIVE_CLASSES = {
    "snow_cover": range(1, 201),
    "no_snow": (0,),
    "clouds": (255,),
    "forest_without_snow": (215,),
    "forest_with_snow": (210,),
    "water": (220,),
    "nodata": (230,),
    "fill": (254,),
}
METEOFRANCE_NEW_CLASSES = {
    "snow_cover": range(1, 201),
    "no_snow": (0,),
    "clouds": (255,),
    "water": (220,),
    "nodata": (230,),
    "fill": (254,),
}


def extract_swath_lon_lats(
    l2_geolocation_data_group: xr.Dataset, bowtie_trim_mask: xr.DataArray | None = None
) -> SwathDefinition:
    if bowtie_trim_mask is not None:
        lons_modif = np.ma.masked_array(l2_geolocation_data_group.data_vars["longitude"], ~bowtie_trim_mask)
        lats_modif = np.ma.masked_array(l2_geolocation_data_group.data_vars["latitude"], ~bowtie_trim_mask)
        swath_def = SwathDefinition(lons=lons_modif, lats=lats_modif)
    else:
        lons = np.ma.masked_array(l2_geolocation_data_group.data_vars["longitude"])
        lats = np.ma.masked_array(l2_geolocation_data_group.data_vars["latitude"])
        swath_def = SwathDefinition(lons=lons, lats=lats)
    return swath_def


def reproject_l2_nasa_to_grid(
    output_grid: GSGrid,
    l2_geolocation_dataset: xr.Dataset,
    l2_dataset: xr.Dataset,
    bowtie_trim_mask: xr.DataArray | None = None,
    output_filename: str | None = None,
    area_name: str = "French Alps",
    area_description: str = "French Alps bounding box",
    radius_of_influence: float = 1000,
    fill_value: int | float = 255,
):
    swath_def = extract_swath_lon_lats(l2_geolocation_data_group=l2_geolocation_dataset, bowtie_trim_mask=bowtie_trim_mask)
    area_def = AreaDefinition(
        area_id=area_name,
        description=area_description,
        proj_id=area_name,
        projection=output_grid.crs,
        width=output_grid.width,
        height=output_grid.height,
        area_extent=output_grid.extent_llx_lly_urx_ury,
    )

    reprojected_data_vars = {}
    for data_var_name, data_var in l2_dataset.items():
        reprojected_l2_data = kd_tree.resample_nearest(
            source_geo_def=swath_def,
            data=data_var.values,
            target_geo_def=area_def,
            radius_of_influence=radius_of_influence,
            fill_value=fill_value,
            nprocs=8,
        )
        # That's ugly sorry me of the future. Basically we need to force the data array georeferecing with its attributes
        # but then we extract it to be able to use xr.assign later
        reprojected_data_vars.update(
            {
                data_var_name: georef_netcdf_rioxarray(
                    xr.DataArray(
                        data=reprojected_l2_data,
                        coords={"y": output_grid.ycoords, "x": output_grid.xcoords},
                    ).rio.write_nodata(fill_value),
                    crs=output_grid.crs,
                )
            }
        )

    output_dataset = xr.Dataset(reprojected_data_vars)

    if output_filename is not None:
        output_dataset.to_netcdf(output_filename)
    return output_dataset


def reproject_l2_geometry_product(l2_nasa_filename: str, output_path: str, output_grid: GSGrid):
    orbit_number = xr.open_dataset(l2_nasa_filename).attrs["OrbitNumber"]
    l2_geoloc = xr.open_dataset(l2_nasa_filename, group="/geolocation_data")
    l2_dataset = xr.open_dataset(l2_nasa_filename, group="/geolocation_data")[["sensor_zenith", "solar_zenith", "height"]]
    reprojected = reproject_l2_nasa_to_grid(
        l2_geolocation_dataset=l2_geoloc,
        l2_dataset=l2_dataset,
        output_grid=output_grid,
        output_filename=None,
        fill_value=255,
    )
    reprojected.attrs["orbit_number"] = orbit_number
    reprojected.to_netcdf(output_path)
    return reprojected


def reproject_l2_radiance_product(l2_geometry_filename: str, l2_radiance_filename: str, output_path: str, output_grid: GSGrid):
    l2_geolocation = xr.open_dataset(l2_geometry_filename, group="/geolocation_data")
    l2_radiance = xr.open_dataset(l2_radiance_filename, group="/observation_data")[["I01", "I03"]]
    l2_radiance_bowtie = xr.open_dataset(l2_radiance_filename, group="/observation_data", mask_and_scale=False).data_vars[
        "I01"
    ]

    bowtie_mask = l2_radiance_bowtie != 65533

    orbit_number = xr.open_dataset(l2_radiance_filename).attrs["orbit_number"]
    if orbit_number != xr.open_dataset(l2_geometry_filename).attrs["OrbitNumber"]:
        print("Problem with orbit number")
    reprojected = reproject_l2_nasa_to_grid(
        l2_geolocation_dataset=l2_geolocation,
        l2_dataset=l2_radiance,
        bowtie_trim_mask=bowtie_mask,
        output_grid=output_grid,
        output_filename=output_path,
        fill_value=65535,
    )
    reprojected.attrs["orbit_number"] = orbit_number
    reprojected.to_netcdf(output_path)
    return reprojected


def mosaic_swaths_l3_grid(swath_filenames: List[str], out_folder: str, nodata_value: int):
    orbit_numbers = [xr.open_dataset(file).attrs["orbit_number"] for file in swath_filenames]
    for unique_orbit_number in list(set(orbit_numbers)):
        orbit_number_filenames = [fn for i, fn in enumerate(swath_filenames) if orbit_numbers[i] == unique_orbit_number]
        orbit_number_ds = xr.open_dataset(orbit_number_filenames[0], mask_and_scale=False)
        if len(orbit_number_filenames) > 1:
            for other_orbit_number_swath in orbit_number_filenames[1:]:
                orbit_number_ds = orbit_number_ds.where(
                    orbit_number_ds != nodata_value, xr.open_dataset(other_orbit_number_swath, mask_and_scale=False)
                )

        orbit_number_ds.to_netcdf(f"{out_folder}/{os.path.basename(orbit_number_filenames[0])}")


def reprojection_l3_meteofrance_to_grid(meteofrance_snow_cover: xr.DataArray, output_grid: GSGrid) -> xr.DataArray:
    # Validity "zombie mask": wherever there is at least one non valid pixel, the output grid pixel is set as invalid (<-> cloud)
    # nasa_dataset = nasa_dataset.where(nasa_dataset <= NASA_CLASSES["snow_cover"][-1], NASA_CLASSES["fill"][0])

    resampled_max = reproject_using_grid(
        meteofrance_snow_cover,
        output_grid=output_grid,
        resampling_method=Resampling.max,
        nodata=METEOFRANCE_ARCHIVE_CLASSES["fill"][0],
    )

    # Tricky forest with snow when resampling using average
    # Whenever a resampled pixel includes forest with snow mask, a quantitative estimation connot be performed unless we choose a FSC value for forest with snow
    # The solution would be to resample forest with snow using max, but this is problematic when forest with snow is next to no snow because it increases the snow detections
    # Therefore we set it to 50% FSC (which means 100 in meteofrance encoding).
    # The contingency analysis will not be biased. The quantitative analysis will be more uncertain and perhaps biaised. The recommendation is to use a forest mask resampled with max for quantitative analysis
    resampled_average = reproject_using_grid(
        meteofrance_snow_cover.where(meteofrance_snow_cover <= METEOFRANCE_ARCHIVE_CLASSES["forest_with_snow"][0], 0)
        .where(meteofrance_snow_cover != METEOFRANCE_ARCHIVE_CLASSES["forest_with_snow"][0], 100)
        .astype("f4"),
        output_grid=output_grid,
        resampling_method=Resampling.average,
    )

    resampled_nearest = reproject_using_grid(
        meteofrance_snow_cover,
        output_grid=output_grid,
        resampling_method=Resampling.nearest,
    )

    water_mask = resampled_nearest == METEOFRANCE_ARCHIVE_CLASSES["water"][0]
    forest_without_snow_mask = resampled_nearest == METEOFRANCE_ARCHIVE_CLASSES["forest_without_snow"][0]
    forest_with_snow_mask = resampled_nearest == METEOFRANCE_ARCHIVE_CLASSES["forest_with_snow"][0]

    cloud_mask = resampled_max == METEOFRANCE_ARCHIVE_CLASSES["clouds"][0]
    nodata_mask = resampled_max == METEOFRANCE_ARCHIVE_CLASSES["nodata"][0]

    invalid_mask = cloud_mask | nodata_mask

    # We exclude these values from the next resampling operations
    valid_qualitative_mask = water_mask | forest_without_snow_mask | forest_with_snow_mask  # | no_snow_mask
    out_snow_cover = resampled_average.where(valid_qualitative_mask == False, resampled_nearest)
    out_snow_cover = out_snow_cover.where(invalid_mask == False, resampled_max)
    return out_snow_cover.astype("u1")


class MeteoFrancePipelineBackup:
    def __init__(
        self,
        meteofrance_archive_filepath: str,
        radiance_filepath: str,
        forest_mask_path: str | None = None,
    ):
        if forest_mask_path:
            self.forest_mask = rasterio.open(forest_mask_path).read(1)
        self.snow_cover_path = meteofrance_archive_filepath
        self.radiance_filepath = radiance_filepath
        self.forest_with_snow_value = 210
        self.forest_without_snow_value = 215
        self.red_band_screen_value = 0.07
        self.max_snow_cover_value = 200
        self.no_snow_value = 0

    def create_new_meteofrance_product(self):
        self.snow_cover = xr.open_dataarray(self.snow_cover_path, mask_and_scale=False)
        self.red_band = xr.open_dataset(self.radiance_filepath, mask_and_scale=False).data_vars["I01"]
        self.swir_band = xr.open_dataset(self.radiance_filepath, mask_and_scale=False).data_vars["I03"]

        ndsi_map = (self.red_band - self.swir_band) / (self.red_band + self.swir_band)
        fsc_map = np.maximum(0, np.minimum(1.45 * ndsi_map - 0.01, 1))

        no_forest = np.where((self.snow_cover == self.forest_without_snow_value), 0, self.snow_cover)
        no_forest_with_snow = np.where(
            (no_forest == self.forest_with_snow_value),
            (fsc_map * 200).astype(np.uint8),
            no_forest,
        )

        low_refl_mask = self.red_band <= self.red_band_screen_value
        modified = np.where(
            no_forest_with_snow > self.max_snow_cover_value,
            no_forest_with_snow,
            np.where(1 - low_refl_mask, no_forest_with_snow, self.no_snow_value),
        )
        # time = datetime.strptime(Path(self.snow_cover_path).name[:13], "%Y%m%d_%H%M")

        return xr.Dataset(
            {"snow_cover_fraction": xr.DataArray(modified, coords=self.snow_cover.coords, dims=self.snow_cover.dims)}
        )


def create_temporal_composite_meteofrance_multiplatform(
    daily_snow_cover_files: List[str], daily_geometry_files: List[str]
) -> xr.Dataset:
    """Create a L3 daily composite form daily L2 swath views using a sensor zenith angle criterion.
    For each pixel we select the "best" observation, i.e. the observation with smaller zenith angle.
    We also make the choice of retrieving some "non-optimal" information.
    If the "best" observation is cloud covered (more generally invalid), we take the other observation,
    even if it has been done at a very high sensor zenith angle.
    This will recover some invalid pixels but at the same time probably introduces false detections
    (more generally "bad" observations)
    """

    # daily_snow_cover_files, daily_geometry_files = (
    #     match_daily_snow_cover_and_geometry_meteofrance(
    #         daily_snow_cover_files, daily_geometry_files
    #     )
    # )
    daily_snow_cover_files.sort()
    daily_geometry_files.sort()

    # This is to account for the fact that NASA and Météo-France L2 come from two very different pipelines
    # Metadata are different (i.e. observation times) and it's possible that there is a different number of daily files
    # This funciton is here to filter this but ideally in a future iteration where sensor zenith angle will
    # be given in the L2 Météo-France this will be useless
    # Read data and assemble in a numpy temporally ordered array
    snow_cover_daily_images = np.array(
        [
            xr.open_dataset(file, mask_and_scale=False).data_vars["snow_cover_fraction"].sel(band=1).drop_vars("band").values
            for file in daily_snow_cover_files
        ]
    )
    view_angles = np.array(
        [xr.open_dataset(file, mask_and_scale=False).data_vars["sensor_zenith"].values for file in daily_geometry_files]
    )

    platform_array = np.zeros_like(snow_cover_daily_images)
    for idx, file in enumerate(daily_snow_cover_files):
        if "SNPP" in file:
            platform_array[idx, :] = 1
        elif "NOAA20" in file:
            platform_array[idx, :] = 2
        elif "NOAA21" in file:
            platform_array[idx, :] = 3
        else:
            raise NotImplementedError
    # Solar zenith angle no observation are encoded as "0", which is also a possible physical value of the incidence angle
    # This is problematic for the algorithm so we correct it.
    # Maybe in the operational product these nodata values will be encoded differently?

    # Compose a view zenith angle data array
    view_angles_daily_array = np.array(view_angles)

    # Sort by view angle
    view_angle_sorting_index = np.argsort(view_angles, axis=0)
    rearrenged_snow_cover = np.take_along_axis(snow_cover_daily_images, view_angle_sorting_index, axis=0)
    rearrenged_view_angle = np.take_along_axis(view_angles_daily_array, view_angle_sorting_index, axis=0)
    rearrenged_platform = np.take_along_axis(platform_array, view_angle_sorting_index, axis=0)

    snow_cover_best_observation = rearrenged_snow_cover[0, :]
    best_observation_angle = rearrenged_view_angle[0, :]
    best_platform = rearrenged_platform[0, :]

    ## In this part we recover observations taken at a worse zenith angle if in the best observation composite the pixel is invalid
    # Intitialize
    out_snow_cover = snow_cover_best_observation
    out_view_angle = best_observation_angle
    out_platform = best_platform

    # Invalid observations
    invalid_masks = rearrenged_snow_cover > METEOFRANCE_NEW_CLASSES["water"][0]
    invalid_mask_out_snow_cover = out_snow_cover > METEOFRANCE_NEW_CLASSES["water"][0]

    for idx in range(snow_cover_daily_images.shape[0]):
        # pixels that are marked as invalid in the best observation but not in another observation
        pixels_to_reverse_mask = invalid_masks[idx] < invalid_mask_out_snow_cover
        out_snow_cover = np.where(pixels_to_reverse_mask, rearrenged_snow_cover[idx], out_snow_cover)
        # Replace data also for view zenith angle and platform
        out_view_angle = np.where(pixels_to_reverse_mask, rearrenged_view_angle[idx], out_view_angle)
        out_platform = np.where(pixels_to_reverse_mask, rearrenged_platform[idx], out_platform)
        invalid_mask_out_snow_cover = out_snow_cover > METEOFRANCE_NEW_CLASSES["water"][0]

    # Here we output in netcdf for export but it can be changed
    sample_data = (
        xr.open_dataset(daily_snow_cover_files[0], decode_cf=True)
        .data_vars["snow_cover_fraction"]
        .sel(band=1)
        .drop_vars("band")
    )
    day_dataset = xr.Dataset(
        {
            "snow_cover_fraction": xr.DataArray(out_snow_cover, dims=sample_data.dims, coords=sample_data.coords),
            "sensor_zenith_angle": xr.DataArray(out_view_angle, dims=sample_data.dims, coords=sample_data.coords),
            "platform": xr.DataArray(out_platform, dims=sample_data.dims, coords=sample_data.coords),
        }
    ).rio.write_crs(sample_data.rio.crs)

    day_dataset.attrs["platform_encoding_values"] = ["1", "2", "3"]
    day_dataset.attrs["platform_encoding_platforms"] = ["SNPP", "JPSS1", "JPSS2"]
    return day_dataset


if __name__ == "__main__":
    import glob
    import os
    from datetime import datetime, timedelta

    import earthaccess
    import pandas as pd
    import rioxarray
    from geospatial_grid.grid_database import LatLon375mGrid
    from pyproj import CRS

    # dates = [
    #     datetime(year=2018, month=1, day=23),
    #     datetime(year=2018, month=3, day=16),
    #     datetime(year=2019, month=5, day=13),
    #     datetime(year=2020, month=5, day=4),
    #     datetime(year=2022, month=2, day=26),
    #     datetime(year=2022, month=5, day=1),
    # ]
    # dates = [
    #     datetime(year=2017, month=3, day=15),
    #     # datetime(year=2018, month=2, day=15),
    #     # datetime(year=2018, month=5, day=11),
    #     # datetime(year=2019, month=3, day=26),
    # ]
    dates = pd.date_range(start="2021/08/01", end="2022/07/31", freq="D")

    folder = "./data/france"
    archive_folder = "/home/imperatoren/work/VIIRS_S2_comparison/data/EOFR62"

    # edelweiss_grandesrousses_grid = GSGrid(
    #     x0=937750, y0=6.46425e06, resolution=250, width=143, height=101, crs=CRS.from_epsg(2154)
    # )
    bassies_grid = GSGrid(
        x0=566402.2257056744,
        y0=6191816.233870121,
        resolution=250.33462315809655,
        width=38,
        height=41,
        crs=CRS.from_epsg(2154),
    )
    grid = LatLon375mGrid()

    ### 0. Forest mask reprojection
    forest_mask_filepath = (
        "/home/imperatoren/work/VIIRS_S2_comparison/data/auxiliary/forest_mask/corine_2018/corine_2018_forest_mask_france.tif"
    )
    resampled_forest_mask_filepath = f"{folder}/forest_mask/corine_max.nc"
    reproject_using_grid(
        rioxarray.open_rasterio(forest_mask_filepath), output_grid=grid, resampling_method=Resampling.max
    ).to_netcdf(resampled_forest_mask_filepath)

    # Earthaccess auth
    earthaccess.login()

    for date in dates:
        try:
            print("Processing", date.strftime("%Y-%m-%d"))
            print("Dowload NASA geometry and radiance products")
            # ## 1. Download VIIRS NASA geometry and radiance
            results = earthaccess.search_data(
                short_name="VNP02IMG",  # ATLAS/ICESat-2 L3A Land Ice Height, VNP10?
                bounding_box=grid.bounds_projected_to_epsg(4326),  # Only include files in area of interest...
                temporal=(date, date + timedelta(days=1)),  # ...and time period of interest
                day_night_flag="day",
            )
            print("Radiance products found")
            print(results)
            radiance_product_filenames = earthaccess.download(results, f"{folder}/viirs_nasa_radiance/downloaded/")

            results = earthaccess.search_data(
                short_name="VNP03IMG",  # ATLAS/ICESat-2 L3A Land Ice Height, VNP10?
                bounding_box=grid.bounds_projected_to_epsg(4326),  # Only include files in area of interest...
                temporal=(date, date + timedelta(days=1)),  # ...and time period of interest
                day_night_flag="day",
            )
            print("Geom products found")
            print(results)
            geometry_product_filenames = earthaccess.download(results, f"{folder}/viirs_nasa_geometry/downloaded/")

            ### 2. Reproject all products to edelweiss grid
            print("Reproject swath products to output grid")
            geometry_product_filenames = sorted(
                glob.glob(f"{folder}/viirs_nasa_geometry/downloaded/*{date.strftime('%Y%j')}*.nc")
            )
            radiance_product_filenames = sorted(
                glob.glob(f"{folder}/viirs_nasa_radiance/downloaded/*{date.strftime('%Y%j')}*.nc")
            )

            for geometry_product_filename, radiance_product_filename in zip(
                geometry_product_filenames, radiance_product_filenames
            ):
                if len(geometry_product_filenames) != len(radiance_product_filenames):
                    print(
                        "Watch out: the lists of geometry, radiance and snow cover files are different. This may cause errors."
                    )

                geometry_reprojected_filename = geometry_product_filename.replace("downloaded", "reprojected_swaths")
                geometry_reprojected = reproject_l2_geometry_product(
                    l2_nasa_filename=geometry_product_filename,
                    output_path=geometry_reprojected_filename,
                    output_grid=grid,
                )

                radiance_reprojected_filename = radiance_product_filename.replace("downloaded", "reprojected_swaths")
                radiance_reprojected = reproject_l2_radiance_product(
                    l2_geometry_filename=geometry_product_filename,
                    l2_radiance_filename=radiance_product_filename,
                    output_path=radiance_reprojected_filename,
                    output_grid=grid,
                )

            geom_swath_fn = sorted(glob.glob(f"{folder}/viirs_nasa_geometry/reprojected_swaths/*{date.strftime('%Y%j')}*.nc"))
            rad_swath_fn = sorted(glob.glob(f"{folder}/viirs_nasa_radiance/reprojected_swaths/*{date.strftime('%Y%j')}*.nc"))

            mosaic_swaths_l3_grid(
                swath_filenames=geom_swath_fn, out_folder=f"{folder}/viirs_nasa_geometry/reprojected", nodata_value=255
            )
            mosaic_swaths_l3_grid(
                swath_filenames=rad_swath_fn,
                out_folder=f"{folder}/viirs_nasa_radiance/reprojected",
                nodata_value=65535,
            )

            reprojected_geometry_filenames = sorted(
                glob.glob(f"{folder}/viirs_nasa_geometry/reprojected/*{date.strftime('%Y%j')}*.nc")
            )
            reprojected_radiance_filenames = sorted(
                glob.glob(f"{folder}/viirs_nasa_radiance/reprojected/*{date.strftime('%Y%j')}*.nc")
            )
            old_snow_cover_filenames = sorted(glob.glob(f"{archive_folder}/VIIRS{date.year}/*{date.strftime('%Y%m%d')}*.LT"))
            snow_cover_reprojected_filenames = []
            for old_snow_cover_filename, reprojected_radiance_filename in zip(
                old_snow_cover_filenames, reprojected_radiance_filenames
            ):
                old_snow_cover = rioxarray.open_rasterio(old_snow_cover_filename)  # , mask_and_scale=False, engine="rasterio")

                old_snow_cover = old_snow_cover.rio.write_crs(4326)

                reprojected_snow_cover = reprojection_l3_meteofrance_to_grid(
                    meteofrance_snow_cover=old_snow_cover, output_grid=grid
                )

                reprojected_snow_cover = reprojected_snow_cover.rio.write_nodata(METEOFRANCE_ARCHIVE_CLASSES["nodata"][0])
                snow_cover_reprojected_filename = (
                    f"{folder}/snow_cover/old_reprojected/{os.path.basename(old_snow_cover_filename.replace('LT', 'nc'))}"
                )
                snow_cover_reprojected_filenames.append(snow_cover_reprojected_filename)
                reprojected_snow_cover.to_netcdf(snow_cover_reprojected_filename)

                ### 3. Create a Météo-France pipeline - like snow cover product
                print("Create a Météo-France pipeline - like snow cover product")
                new_pipeline_product = MeteoFrancePipelineBackup(
                    meteofrance_archive_filepath=snow_cover_reprojected_filename,
                    radiance_filepath=reprojected_radiance_filename,
                    forest_mask_path=resampled_forest_mask_filepath,
                ).create_new_meteofrance_product()
                new_snow_cover_filename = snow_cover_reprojected_filename.replace("old_reprojected", "new")
                new_pipeline_product = new_pipeline_product.expand_dims("time")
                new_pipeline_product = new_pipeline_product.assign_coords(coords={"time": [date]})
                new_pipeline_product.to_netcdf(new_snow_cover_filename)

            ### 4. Create the daily composite
            print("Create the daily composite")

            new_snow_cover_filenames = sorted(glob.glob(f"{folder}/snow_cover/new/*{date.strftime('%Y%m%d')}*.nc"))
            daily_composite = create_temporal_composite_meteofrance_multiplatform(
                daily_snow_cover_files=new_snow_cover_filenames, daily_geometry_files=reprojected_geometry_filenames
            )
            daily_composite = daily_composite.assign_coords(coords={"time": date})
            daily_composite.to_netcdf(f"{folder}/snow_cover/composite/MF_FSC_VNP_{date.strftime('%Y%m%d')}.nc")

            print(geometry_product_filenames)
            print(radiance_product_filenames)
            print(reprojected_geometry_filenames)
            print(reprojected_radiance_filenames)
            print(geom_swath_fn)
            print(rad_swath_fn)
            print(snow_cover_reprojected_filenames)
            print(new_snow_cover_filenames)

            [os.remove(f) for f in geometry_product_filenames]
            [os.remove(f) for f in radiance_product_filenames]
            [os.remove(f) for f in reprojected_radiance_filenames]
            [os.remove(f) for f in reprojected_geometry_filenames]
            [os.remove(f) for f in geom_swath_fn]
            [os.remove(f) for f in rad_swath_fn]
            [os.remove(f) for f in snow_cover_reprojected_filenames]
            [os.remove(f) for f in new_snow_cover_filenames]
        except Exception as e:
            print(f"Problem with date {date}", e)
    # except Exception as e:
    #     print(f"Problem at date {date}", e)
    #     continue
