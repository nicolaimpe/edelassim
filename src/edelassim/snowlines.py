import xarray as xr
from mountain_data_binner.mountain_binner import MountainBinner, MountainBinnerConfig
from ndsi_fsc_calibration.snow_cover_products import S2_CLASSES

# We define Météo-France class encoding via a dictio
METEOFRANCE_CLASSES = {
    "snow_cover": range(1, 201),
    "no_snow": (0,),
    "clouds": (255,),
    "water": (220,),
    "nodata": (230,),
    "fill": (254,),
}


def reduce_fsc_to_pixel_counts_meteofrance(fsc_and_aux_dataset: xr.Dataset) -> xr.Dataset:
    """Produce a synthesis of a snow cover fraction. Intermediary data for snowline calculation.

    :param fsc_and_aux_dataset: a Dataset having a snow_cover_fraction layer and optionally other layers on the same \
    coordinate system (aspect map, slope maps, etc...). It is produced by the MountainParametrization class
    :type fsc_and_aux_dataset: xr.Dataset
    :return: a Dataset contained synthetic information over snow and land cover
    :rtype: xr.Dataset
    """
    # there's a water class for lakes and rivers to consider
    satellite_fsc_data = fsc_and_aux_dataset.data_vars["snow_cover_fraction"]
    # Set water pixels to 0 snow cover
    satellite_fsc_data = satellite_fsc_data.where(satellite_fsc_data != METEOFRANCE_CLASSES["water"], 0)
    valid_data_mask = satellite_fsc_data < METEOFRANCE_CLASSES["nodata"]
    invalid_data_mask = satellite_fsc_data >= METEOFRANCE_CLASSES["nodata"]
    valid_snow_cover_fraction = satellite_fsc_data.where(valid_data_mask) / METEOFRANCE_CLASSES["snow_cover"][-1]
    # snow cover threshold
    snow_cover_mask = valid_snow_cover_fraction >= 0.5
    return xr.Dataset(
        {
            "snow_cover_sum": valid_snow_cover_fraction.sum(),
            "land_sum": (1 - valid_snow_cover_fraction).sum(),
            "n_snow_cover_pixels": snow_cover_mask.sum(),
            "total_valid": valid_data_mask.sum(),
            "total_invalid": invalid_data_mask.sum(),
        }
    )


def reduce_fsc_to_pixel_counts_s2(fsc_and_aux_dataset: xr.Dataset) -> xr.Dataset:
    """Produce a synthesis of a snow cover fraction. Intermediary data for snowline calculation.

    :param fsc_and_aux_dataset: a Dataset having a snow_cover_fraction layer and optionally other layers on the same \
    coordinate system (aspect map, slope maps, etc...). It is produced by the MountainParametrization class
    :type fsc_and_aux_dataset: xr.Dataset
    :return: a Dataset contained synthetic information over snow and land cover
    :rtype: xr.Dataset
    """
    # there's a water class for lakes and rivers to consider
    satellite_fsc_data = fsc_and_aux_dataset.data_vars["snow_cover_fraction"]
    valid_data_mask = satellite_fsc_data < S2_CLASSES["clouds"]
    invalid_data_mask = satellite_fsc_data >= S2_CLASSES["clouds"]
    valid_snow_cover_fraction = satellite_fsc_data.where(valid_data_mask)
    # snow cover threshold
    snow_cover_mask = valid_snow_cover_fraction >= 0.5
    return xr.Dataset(
        {
            "snow_cover_sum": valid_snow_cover_fraction.sum(),
            "land_sum": (1 - valid_snow_cover_fraction).sum(),
            "n_snow_cover_pixels": snow_cover_mask.sum(),
            "total_valid": valid_data_mask.sum(),
            "total_invalid": invalid_data_mask.sum(),
        }
    )


def sum_all_values_below(data_array: xr.DataArray, dim: str, coord: str) -> xr.DataArray:
    """Sort of a magic inliner to obtain the sum of all values above each array element.

    Example: [0, 3, 4, 2, 10] -> [19, 16, 12, 10, 0]"""
    return data_array.sortby(coord, ascending=True).cumsum(dim=dim).sortby(coord)


def sum_all_values_above(data_array: xr.DataArray, dim: str, coord: str) -> xr.DataArray:
    """Sort of a magic inliner to obtain the sum of all values below each array element.

    Example: [0, 3, 4, 2, 10] -> [0, 3, 7, 9, 19]"""
    return data_array.sortby(coord, ascending=False).cumsum(dim=dim).sortby(coord)


def snowline_penalization_function(
    class_distribution_dataset: xr.Dataset,
) -> xr.Dataset:
    """The snowline penalization function, modified Krajci algorithm to calculate the altitude of the snowline
    from satellite snow cover maps.

    See "Détection des lignes d’enneigement avec les images satellites VIIRS et comparaison avec les BERA
    et produits de la chaine de modélisation S2M", rappor de stage de Laura Canterini

    Pavel Krajci, Ladislav Holko, Rui A.P. Perdigão, and Juraj Parajka. Estimation of regional snowline
    elevation (RSLE) from Modis images for seasonally snow covered mountain basins. Journal of Hydrology, 2014.

    Two modifications of the algorithm have been implemented:
    - The use of snow cover fractions instead of binry snow maps to compute the function (stage Laura Canterini)
    - The application of a wight to the lower snow sum in the forest to compensate the fact that the satellite
      typically underestimates snow in the forest (publi VIIRS N. Imperatore)

    :param class_distribution_dataset:  The snow cover map reduced to snow/no snow pixel count in orientation/altitude space
                                        (function reduce_fsc_to_pixel_counts)
    :type class_distribution_dataset: xr.Dataset
    :return: The snowline penalization function calculated with the modified Krajci algorithm in oreintation/altitude space
    :rtype: xr.Dataset
    """
    lower_snow_sum = sum_all_values_below(
        class_distribution_dataset["snow_cover_sum"], dim="altitude_bins", coord="altitude_min"
    )
    higher_land_sum = sum_all_values_above(class_distribution_dataset["land_sum"], dim="altitude_bins", coord="altitude_min")
    snowline_metrics = higher_land_sum + lower_snow_sum
    return snowline_metrics


def to_snowline_parametrization(reduced_dataset: xr.Dataset) -> xr.Dataset:
    """Produce a Dataset that can be used to calculate snowlines.

    :param reduced_dataset: a Dataset built via reduce_fsc_to_pixel_counts
    :type reduced_dataset: xr.Dataset
    :return: a set of metrics useful for snowline calculation
    :rtype: xr.Dataset
    """

    snowline_metrics = snowline_penalization_function(class_distribution_dataset=reduced_dataset)

    total_valid = reduced_dataset.data_vars["total_valid"]
    percentage = reduced_dataset.data_vars["n_snow_cover_pixels"] / total_valid * 100
    return xr.Dataset(
        {
            "snowline_penalization": snowline_metrics,
            "percentage_snow_cover": percentage,
        }
    )


def postprocess_snowline_dataset(
    snowline_parametrization_dataset: xr.Dataset,
) -> xr.Dataset:
    """Clean up for better user experience. Remove some noisy xarray nomenclature.

    :param snowline_parametrization_dataset: a Dataset produced via to_snowline_parametrization
    :type snowline_parametrization_dataset: xr.Dataset
    :return: same as input but without "_bins" for coordinates and spatial_ref veriable
    :rtype: xr.Dataset
    """
    snowline_parametrization_dataset = snowline_parametrization_dataset.rename(
        {coord_name: coord_name.replace("_bins", "") for coord_name in list(snowline_parametrization_dataset.dims)}
    )
    snowline_parametrization_dataset = snowline_parametrization_dataset.drop_vars("spatial_ref")
    return snowline_parametrization_dataset


class SnowCoverFractionToSnowline:
    """Wrap-up this module for more comfortable user experience."""

    def __init__(self, fsc_image: xr.Dataset, mnt_data_paths: MountainBinnerConfig):
        """Initialize by specifying where to find the input necessary data.

        :param fsc_sat_image_path: VIIRS new snow cover composite path
        :type fsc_sat_image_path: str
        :param mnt_data_paths: processed MNT and derived and massif information filepaths
        :type mnt_data_paths: MountainParams
        """
        self.snow_cover_fraction = fsc_image
        self.mnt_data_paths = mnt_data_paths

    def transform(self, export_path: str | None = None) -> xr.Dataset:
        """Concatenate this module functions to pass from a satellite snow cover fraction to snowline diagrams per massif.

        :param export_path: export snowline data to netcdf. If None, no export. defaults to None
        :type export_path: str | None, optional
        :return: a Dataset that can be directly used for snowline caluclation and diagnostics
        :rtype: xr.Dataset
        """
        # Prepare data for reduction (spatially distributed -> semidistributed or categorically distributed)
        mountain_binner = MountainBinner(config=self.mnt_data_paths)
        bins_dictionary = mountain_binner.create_default_bin_dict_from_config(altitude_step=100, altitude_max=3900)
        reduced = mountain_binner.transform(
            distributed_data=self.snow_cover_fraction,
            bin_dict=bins_dictionary,
            function=reduce_fsc_to_pixel_counts_s2,
        )

        # Take only the slopes between 5 and 30 degrees : (bin 5 -> slopes between [5, 30) °])
        # Convert reduction to some metrics that can be directly used for snowline
        snowline_diagram = to_snowline_parametrization(reduced)
        # snowline_diagram = snowline_diagram.swap_dims({"altitude_min": "altitude_bins"})
        # Clean up the resulting Dataset for better user experience
        snowline_diagram_postprocessed = postprocess_snowline_dataset(snowline_diagram)
        # Re-add time dimension
        # print(snowline_diagram_postprocessed)
        # snowline_diagram_postprocessed = snowline_diagram_postprocessed.expand_dims(
        #     {"time": self.snow_cover_fraction.coords["time"]}
        # )
        if export_path is not None:
            snowline_diagram_postprocessed.to_netcdf(export_path)
        return snowline_diagram_postprocessed
