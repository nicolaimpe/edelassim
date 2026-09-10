import numpy as np
import xarray as xr
from matplotlib.axes import Axes

from edelassim.snowlines import find_forcing_snowrain_line, find_snowline_from_snow_penalization

############ STATIC
COMPASS_ROSE_DICT = {"N": 0, "NE": 45, "E": 90, "SE": 135, "S": 180, "SW": 225, "W": 270, "NW": 315}


def set_polarplot(
    ax: Axes,
    alt_max: int = 0,
    alt_min: int = 4800,
) -> Axes:

    ax.set_rlim(alt_max, alt_min)
    ax.set_rorigin(alt_max)
    ax.set_theta_direction(-1)  # Clockwise rotation (standard for maps)
    ax.set_theta_offset(np.pi / 2)
    # ax.set_rticks([1000, 2000])
    # ax.rticks(fontsize=9)
    ax.tick_params(axis="both", labelsize="x-small")
    ax.set_xticks(np.deg2rad(list(COMPASS_ROSE_DICT.values())))
    ax.set_xticklabels(list(COMPASS_ROSE_DICT.keys()))
    ax.grid(True)
    ax.set_title("Snowline", va="bottom")
    ax.legend(loc="upper right", bbox_to_anchor=(1.2, 1.2))
    return ax


def plot_snowline_polarplot(
    snowline_per_aspects: np.ndarray,
    ax: Axes,
    alt_max: int = 0,
    alt_min: int = 4800,
    label: str | None = None,
    color: str | None = None,
) -> None:

    set_polarplot(ax=ax, alt_max=alt_max, alt_min=alt_min)
    r = snowline_per_aspects
    theta = np.deg2rad(list(COMPASS_ROSE_DICT.values()))
    r = [*r, r[0]]
    theta = [*theta, theta[0]]
    ax.plot(theta, r, label=label, color=color)
    return ax


def plot_ensemble_snowline_polarplot(
    snowline_per_aspects_upper: np.ndarray,
    snowline_per_aspects_lower: np.ndarray,
    ax: Axes,
    alt_max: int = 0,
    alt_min: int = 4800,
    label: str | None = None,
    color: str | None = None,
) -> None:

    set_polarplot(ax=ax, alt_max=alt_max, alt_min=alt_min)
    theta = np.deg2rad(list(COMPASS_ROSE_DICT.values()))
    theta = [*theta, theta[0]]
    r_upper = snowline_per_aspects_upper
    r_upper = [*r_upper, r_upper[0]]
    r_lower = snowline_per_aspects_lower
    r_lower = [*r_lower, r_lower[0]]
    ax.fill_between(theta, r_lower, r_upper, label=label, color=color, alpha=0.5)
    return ax


def plot_snowline_polarplot_from_semidistributed(
    snowline_parametrization_dataset: xr.Dataset,
    dataset_type: str,
    ax: Axes,
    label: str | None = None,
    color: str | None = None,
):
    if dataset_type == "snow_cover":
        snowline = find_snowline_from_snow_penalization(snowline_parametrization_dataset)
    elif dataset_type == "forcing":
        snowline = find_forcing_snowrain_line(snowline_parametrization_dataset)
    else:
        raise ValueError("Unknown dataset_type argument. Valid choices are 'snow_cover' and 'forcing'")
    # print(snowline.values)
    alt_max = snowline_parametrization_dataset.coords["altitude_max"].max()
    alt_min = snowline_parametrization_dataset.coords["altitude_max"].min()

    return plot_snowline_polarplot(
        snowline_per_aspects=snowline,
        alt_max=alt_max,
        alt_min=alt_min,
        ax=ax,
        label=label,
        color=color,
    )


def plot_ensemble_snowline_polarplot_from_semidistributed(
    snowline_parametrization_dataset: xr.Dataset,
    dataset_type: str,
    ax: Axes,
    label: str | None = None,
    color: str | None = None,
    p_low: int = 10,
    p_high: int = 90,
):

    if dataset_type == "snow_cover":
        snowline = find_snowline_from_snow_penalization(snowline_parametrization_dataset)
        snowline_lower = snowline.quantile(q=p_low / 100, dim="member")
        snowline_upper = snowline.quantile(q=p_high / 100, dim="member")
    elif dataset_type == "forcing":
        snowline = find_forcing_snowrain_line(snowline_parametrization_dataset)
        snowline_lower = snowline.quantile(q=p_low / 100, dim="member")
        snowline_upper = snowline.quantile(q=p_high / 100, dim="member")
    else:
        raise ValueError("Unknown dataset_type argument. Valid choices are 'snow_cover' and 'forcing'")
    # print(snowline.values)
    alt_max = snowline_parametrization_dataset.coords["altitude_max"].max()
    alt_min = snowline_parametrization_dataset.coords["altitude_min"].min()
    return plot_ensemble_snowline_polarplot(
        snowline_per_aspects_lower=snowline_lower,
        snowline_per_aspects_upper=snowline_upper,
        alt_max=alt_max,
        alt_min=alt_min,
        ax=ax,
        label=label,
        color=color,
    )
