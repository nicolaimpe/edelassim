import matplotlib.pyplot as plt
import numpy as np
import xarray as xr
from ipywidgets import Dropdown, interact
from matplotlib.axes import Axes
from matplotlib.colors import LinearSegmentedColormap
from matplotlib.figure import Figure
from mpl_toolkits.axes_grid1 import make_axes_locatable
from ndsi_fsc_calibration.snow_cover_products import S2_CLASSES
from scipy.special import logit
from scipy.stats import pearsonr

from edelassim.evaluations import find_common_correspondences
from edelassim.observations import METEOFRANCE_NEW_CLASSES
from edelassim.snowlines import find_forcing_snowrain_line, find_snowline_from_snow_penalization

############ STATIC
COMPASS_ROSE_DICT = {"N": 0, "NE": 45, "E": 90, "SE": 135, "S": 180, "SW": 225, "W": 270, "NW": 315}


# Define color stops at specific values
fsc_color_def_snow_cover = [
    (0.0, (0, 0, 0)),  # 0 -> black, no snow
    (1 / 100, (8 / 255, 51 / 255, 112 / 255)),  # 1 -> light blue, 1% snow
    (100 / 100, (1, 1, 1)),  # 100 -> white, full snow
]

FSC_CMAP_SNOW_COVER = LinearSegmentedColormap.from_list("custom_cmap", fsc_color_def_snow_cover, N=256)


def scatter_logit_plot(
    data1: xr.Dataset, data2: xr.Dataset, ax_normal: Axes, ax_logit: Axes, color: str, title: str, label: str
) -> None:
    data1_correspondences, data2_correspondences = find_common_correspondences(data_1=data1, data_2=data2)
    data1_correspondences = data1_correspondences.ravel()
    data2_correspondences = data2_correspondences.ravel()
    r_coeff = pearsonr(data2_correspondences, data1_correspondences).statistic
    ax_normal.plot(
        data1_correspondences, data2_correspondences, ".", color=color, markersize=0.8, label=f"{label} - r = {r_coeff:.2f}"
    )
    ax_normal.set_title(title)
    ax_logit.plot(logit(data1_correspondences), logit(data2_correspondences), ".", color=color, markersize=0.8)
    ax_logit.set_title(f"logit - {title}")
    ax_normal.legend()
    ax_logit.legend()


def boxplot_logit_plot(
    data1: xr.Dataset,
    data2: xr.Dataset,
    ax_normal: Axes,
    ax_logit: Axes,
    color: str,
    title: str,
    pos: float,
    width: float,
    label: str | None = None,
) -> None:
    data1_correspondences, data2_correspondences = find_common_correspondences(data_1=data1, data_2=data2)
    data1_correspondences = data1_correspondences.ravel()
    data2_correspondences = data2_correspondences.ravel()
    residuals = data2_correspondences - data1_correspondences
    residuals = residuals[~np.isnan(residuals)]
    bp = ax_normal.boxplot(
        residuals, positions=pos, showfliers=False, notch=True, patch_artist=True, widths=width, label=label
    )
    bp["boxes"][0].set_facecolor(color)

    ax_normal.set_title(title)

    residuals_logit = logit(data2_correspondences) - logit(data1_correspondences)
    residuals_logit = residuals_logit[~np.isnan(residuals_logit)]
    residuals_logit = residuals_logit[~np.isinf(residuals_logit)]
    bp = ax_logit.boxplot(
        residuals_logit, positions=pos, showfliers=False, notch=True, patch_artist=True, widths=width, label=label
    )
    bp["boxes"][0].set_facecolor(color)

    ax_logit.set_title(f"logit - {title}")
    ax_normal.legend()
    ax_logit.legend()


def plot_snowline_polarplot(
    snowline_per_aspects: np.ndarray,
    ax: Axes,
    alt_max: int = 0,
    alt_min: int = 4800,
    label: str | None = None,
    color: str | None = None,
) -> None:

    r = snowline_per_aspects
    # print()
    theta = np.deg2rad(list(COMPASS_ROSE_DICT.values()))

    r = [*r, r[0]]
    theta = [*theta, theta[0]]
    ax.set_rlim(alt_max, alt_min)
    ax.set_rorigin(alt_max)
    ax.set_theta_direction(-1)  # Clockwise rotation (standard for maps)
    ax.set_theta_offset(np.pi / 2)
    # ax.set_rticks([1000, 2000])
    # ax.rticks(fontsize=9)
    ax.tick_params(axis="both", labelsize="x-small")
    ax.plot(theta, r, label=label, color=color)
    ax.grid(True)
    ax.set_title("Snowline", va="bottom")
    ax.legend(loc="upper right", bbox_to_anchor=(1.2, 1.2))
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


def add_colorbar(ax, **kwargs):
    """Add a colorbar to the given axes, safely removing any existing one first."""
    # Safely remove existing colorbar and its axes
    if hasattr(ax, "_colorbar") and ax._colorbar is not None:
        try:
            ax._colorbar.remove()
        except (AttributeError, ValueError, KeyError):
            pass  # Already removed or invalid

    if hasattr(ax, "_colorbar_ax") and ax._colorbar_ax is not None:
        try:
            ax._colorbar_ax.remove()
        except (AttributeError, ValueError, KeyError):
            pass

    if ax.images:
        mappable = ax.images[-1]
    elif ax.collections:
        mappable = ax.collections[-1]
    else:
        raise ValueError("No mappable found in axes")

    fig = ax.figure
    divider = make_axes_locatable(ax)
    cax = divider.append_axes("right", size="5%", pad=0.05)
    cb = fig.colorbar(mappable, cax=cax, **kwargs)

    # Store references
    ax._colorbar = cb
    ax._colorbar_ax = cax
    return cb
