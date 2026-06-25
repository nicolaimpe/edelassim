import matplotlib.pyplot as plt
import numpy as np
import xarray as xr
from ipywidgets import Dropdown, interact
from matplotlib.axes import Axes
from matplotlib.colors import LinearSegmentedColormap
from scipy.special import logit
from scipy.stats import pearsonr

from edelassim.evaluations import find_common_correspondences

############ STATIC
COMPASS_ROSE_DICT = {"N": 0, "NE": 45, "E": 90, "SE": 135, "S": 180, "SW": 225, "W": 270, "NW": 315}


# Define color stops at specific values
fsc_color_def_viirs_mf = [
    (0.0, (0, 0, 0)),  # 0 -> black, no snow
    (1 / 255, (8 / 255, 51 / 255, 112 / 255)),  # 1 -> light blue, 1% snow
    (200 / 255, (1, 1, 1)),  # 200 -> white, full snow
    (219 / 255, (1, 1, 1)),  # 200 -> white, full snow
    (220 / 255, (0, 0, 1)),  # 220 -> blue, water
    (230 / 255, (0.5, 0.5, 0.5)),  # 230 -> gray, no data
    (1.0, (0.5, 0.5, 0.5)),  # 255 -> gray
]

FSC_CMAP_VIIRS_MF = LinearSegmentedColormap.from_list("custom_cmap", fsc_color_def_viirs_mf, N=256)

fsc_color_def_s2 = [
    (0.0, (0, 0, 0)),  # 0 -> black, no snow
    (1 / 255, (8 / 255, 51 / 255, 112 / 255)),  # 1 -> light blue, 1% snow
    (100 / 255, (1, 1, 1)),  # 200 -> white, full snow
    (204 / 255, (1, 1, 1)),  # 200 -> white, full snow
    (205 / 255, (0.5, 0.5, 0.5)),  # 205 -> gray, clouds
    (1.0, (0.5, 0.5, 0.5)),  # 255 -> gray, nodata
]

FSC_CMAP_S2 = LinearSegmentedColormap.from_list("custom_cmap", fsc_color_def_s2, N=256)

fsc_color_def_edel = [
    (0.0, (0, 0, 0)),  # 0 -> black, no snow
    (1 / 200, (8 / 255, 51 / 255, 112 / 255)),  # 1 -> light blue, 1% snow
    (200 / 200, (1, 1, 1)),  # 200 -> white, full snow
]

FSC_CMAP_EDEL = LinearSegmentedColormap.from_list("custom_cmap", fsc_color_def_edel, N=256)


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


def plot_snowline_polarplot(snowline_parametrization_dataset: xr.Dataset, ax: Axes, label: str, color: str) -> None:

    # print(snowline_parametrization_dataset)
    snowline_parametrization_dataset = snowline_parametrization_dataset.swap_dims({"altitude": "altitude_min"})

    alt_index = snowline_parametrization_dataset.data_vars["snowline_penalization"].argmin("altitude_min")
    snowline = snowline_parametrization_dataset.isel(altitude_min=list(alt_index)).coords["altitude_min"]
    print(label, snowline.values)
    max_alt = snowline_parametrization_dataset.coords["altitude_max"].max()
    r = snowline
    theta = [np.deg2rad(COMPASS_ROSE_DICT[asp]) for asp in snowline_parametrization_dataset.coords["aspect"].values]

    r = [*r, r[0]]
    theta = [*theta, theta[0]]
    ax.set_rlim(max_alt, snowline_parametrization_dataset.coords["altitude_max"].min())
    ax.set_rorigin(max_alt)
    ax.set_theta_direction(-1)  # Clockwise rotation (standard for maps)
    ax.set_theta_offset(np.pi / 2)
    ax.plot(theta, r, label=label, color=color)
    ax.grid(True)
    ax.set_title("Snowline", va="bottom")
    ax.legend()
    return ax
