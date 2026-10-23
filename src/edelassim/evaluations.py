import numpy as np
import xarray as xr
from geospatial_grid.gsgrid import GSGrid
from pyproj import CRS
from snowtools.scores.ensemble import EnsembleScores


class GrandesRoussesGrid20m(GSGrid):
    """This grid bound correspond to a bounding box including all mountaineous areas over metropolitan France in WGS84 geographic coordinates."""

    def __init__(self):
        super().__init__(
            x0=736914.0445886956,
            y0=5015138.075400733,
            resolution=20,
            width=1788,
            height=1263,
            crs=CRS.from_epsg(32631),
            name="Sentinel2_GrandesRousses",
        )


def find_common_correspondences(data_1: xr.DataArray, data_2: xr.DataArray) -> tuple[np.ndarray, np.ndarray]:
    data_1_valid_mask = ~np.isnan(data_1)
    data_2_valid_mask = ~np.isnan(data_2)

    union = data_1_valid_mask * data_2_valid_mask

    data_1_correspondences = data_1.where(union).values.flatten()
    data_2_correspondences = data_2.where(union).values.flatten()

    data_1_correspondences = data_1_correspondences[~np.isnan(data_1_correspondences)]
    data_2_correspondences = data_2_correspondences[~np.isnan(data_2_correspondences)]
    return data_1_correspondences, data_2_correspondences


def compute_rmse(diff_array: np.ndarray | xr.DataArray) -> np.ndarray:
    if type(diff_array) == xr.DataArray:
        diff_array = diff_array.values
    n = np.count_nonzero(~np.isnan(diff_array))
    squared_error = diff_array**2
    return np.sqrt(np.nansum(squared_error) / n)


def compute_scores_grouped_data(data: xr.Dataset) -> xr.Dataset:

    mean_ensemble_open_loop = data.data_vars["open_loop"].mean(dim="member")
    mean_ensemble_analysis = data.data_vars["analysis"].mean(dim="member")

    # Ensemble metrics: member mean of the residuals
    crps_open_loop = EnsembleScores(
        timeObs=None, timeSim=None, obs=data.data_vars["reference"].values, ensemble=data.data_vars["open_loop"].values
    ).CRPS()
    crps_analysis = EnsembleScores(
        timeObs=None, timeSim=None, obs=data.data_vars["reference"].values, ensemble=data.data_vars["analysis"].values
    ).CRPS()
    residuals_ensemble_open_loop = data.data_vars["open_loop"] - data.data_vars["reference"]
    residuals_ensemble_analysis = data.data_vars["analysis"] - data.data_vars["reference"]
    mae_ensemble_open_loop = np.abs(residuals_ensemble_open_loop).mean(dim="member").values[0]
    mae_ensemble_analysis = np.abs(residuals_ensemble_analysis).mean(dim="member").values[0]
    sigma_ensemble_open_loop = (
        np.sqrt((data.data_vars["open_loop"] - mean_ensemble_open_loop) ** 2).mean(dim="member").values[0]
    )
    sigma_ensemble_analysis = np.sqrt((data.data_vars["analysis"] - mean_ensemble_analysis) ** 2).mean(dim="member").values[0]

    # Deterministic metrics: residuals of the member mean
    residual_deterministic_open_loop = (mean_ensemble_open_loop - data.data_vars["reference"]).values[0]
    residual_deterministic_analysis = (mean_ensemble_analysis - data.data_vars["reference"]).values[0]

    return xr.Dataset(
        {
            "crps": ("type", [crps_open_loop, crps_analysis]),
            "mae": ("type", [mae_ensemble_open_loop, mae_ensemble_analysis]),
            "sigma": ("type", [sigma_ensemble_open_loop, sigma_ensemble_analysis]),
            "diff_mean_member": ("type", [residual_deterministic_open_loop, residual_deterministic_analysis]),
        },
        coords={"type": ["open_loop", "analysis"]},
    )


def compute_scores_grouped_data_spatial(data: xr.Dataset) -> xr.Dataset:

    mean_ensemble_open_loop = data.data_vars["open_loop"].mean(dim="member")
    mean_ensemble_analysis = data.data_vars["analysis"].mean(dim="member")

    # Ensemble metrics: member mean of the residuals
    residuals_ensemble_open_loop = data.data_vars["open_loop"] - data.data_vars["reference"]
    residuals_ensemble_analysis = data.data_vars["analysis"] - data.data_vars["reference"]
    mae_ensemble_open_loop = np.abs(residuals_ensemble_open_loop).mean(dim="member").unstack().mean(dim=("x", "y")).values[0]
    mae_ensemble_analysis = np.abs(residuals_ensemble_analysis).mean(dim="member").unstack().mean(dim=("x", "y")).values[0]
    sigma_ensemble_open_loop = (
        np.sqrt((data.data_vars["open_loop"] - mean_ensemble_open_loop) ** 2)
        .mean(dim="member")
        .unstack()
        .mean(dim=("x", "y"))
        .values[0]
    )
    sigma_ensemble_analysis = (
        np.sqrt((data.data_vars["analysis"] - mean_ensemble_analysis) ** 2)
        .mean(dim="member")
        .unstack()
        .mean(dim=("x", "y"))
        .values[0]
    )

    # Deterministic metrics: residuals of the member mean
    residual_deterministic_open_loop = (
        (mean_ensemble_open_loop - data.data_vars["reference"]).unstack().mean(dim=("x", "y")).values[0]
    )
    residual_deterministic_analysis = (
        (mean_ensemble_analysis - data.data_vars["reference"]).unstack().mean(dim=("x", "y")).values[0]
    )

    return xr.Dataset(
        {
            "mae": ("type", [mae_ensemble_open_loop, mae_ensemble_analysis]),
            "sigma": ("type", [sigma_ensemble_open_loop, sigma_ensemble_analysis]),
            "diff_mean_member": ("type", [residual_deterministic_open_loop, residual_deterministic_analysis]),
        },
        coords=dict(type=["open_loop", "analysis"]),
    )


# def compute_scores_grouped_data(data: xr.Dataset) -> xr.Dataset:
#     def compute_crpss(data: xr.Dataset) -> xr.DataArray:

#         crps_open_loop = EnsembleScores(
#             timeObs=None, timeSim=None, obs=data.data_vars["reference"].values, ensemble=data.data_vars["open_loop"].values
#         ).CRPS()
#         crps_analysis = EnsembleScores(
#             timeObs=None, timeSim=None, obs=data.data_vars["reference"].values, ensemble=data.data_vars["analysis"].values
#         ).CRPS()
#         crpss = 1 - (crps_analysis / crps_open_loop)
#         return xr.DataArray(crpss)

#     def compute_maes(data: xr.Dataset) -> xr.DataArray:
#         mae_open_loop = np.abs(data.data_vars["open_loop"] - data.data_vars["reference"]).mean(dim="member")
#         mae_analysis = np.abs(data.data_vars["analysis"] - data.data_vars["reference"]).mean(dim="member")
#         maes = 1 - mae_analysis / mae_open_loop
#         return xr.DataArray(maes)

#     def compute_sigmas(data: xr.Dataset) -> xr.DataArray:
#         sigma_open_loop = np.sqrt(
#             ((data.data_vars["open_loop"] - data.data_vars["open_loop"].mean(dim="member")) ** 2).mean(dim="member")
#         )
#         sigma_analysis = np.sqrt(
#             ((data.data_vars["analysis"] - data.data_vars["analysis"].mean(dim="member")) ** 2).mean(dim="member")
#         )
#         sigmas = 1 - sigma_analysis / sigma_open_loop
#         return xr.DataArray(sigmas)

#     return xr.Dataset({"crpss": compute_crpss(data), "maes": compute_maes(data), "sigmas": compute_sigmas(data)})
