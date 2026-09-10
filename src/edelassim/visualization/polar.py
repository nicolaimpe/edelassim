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


def plot_envelop_member_snowline(snowline_ds: xr.Dataset, member: int, label: str, color: str, **plot_kwargs):
    snowline_ensemble = find_snowline_from_snow_penalization(snowline_ds)
    plot_polar_envelop_member(snowline_ensemble, member, color=color, label=label, **plot_kwargs)


def plot_envelop_member_snow_rain(phase_ds: xr.Dataset, member: int, label: str, color: str, **plot_kwargs):
    snowline_ensemble = find_forcing_snowrain_line(phase_ds)
    plot_polar_envelop_member(snowline_ensemble, member, color=color, label=label, **plot_kwargs)


def plot_polar_envelop_member(ensemble_polar_data: xr.Dataset, member: int, color: str, label: str, **plot_kwargs):
    p_low, p_high = 10, 90
    lower_sl = ensemble_polar_data.quantile(q=p_low / 100, dim="member")
    upper_sl = ensemble_polar_data.quantile(q=p_high / 100, dim="member")
    plot_ensemble_snowline_polarplot(upper_sl, lower_sl, color=color, **plot_kwargs)
    member_sl = ensemble_polar_data.sel(member=member)
    plot_snowline_polarplot(member_sl, label=label, color=color, **plot_kwargs)


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
