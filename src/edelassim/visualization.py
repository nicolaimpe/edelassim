import matplotlib.pyplot as plt
import numpy as np
import xarray as xr
from matplotlib.axes import Axes
from scipy.special import logit
from scipy.stats import pearsonr

from edelassim.evaluations import find_common_correspondences

COMPASS_ROSE_DICT = {"N": 0, "NE": 45, "E": 90, "SE": 135, "S": 180, "SW": 225, "W": 270, "NW": 315}


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


def plot_snowline_polarplot(snowline_parametrization_dataset: xr.Dataset, ax: Axes, label: str) -> None:

    # print(snowline_parametrization_dataset)
    snowline_parametrization_dataset = snowline_parametrization_dataset.swap_dims({"altitude": "altitude_min"})

    alt_index = snowline_parametrization_dataset.data_vars["snowline_penalization"].argmin("altitude_min")
    snowline = snowline_parametrization_dataset.isel(altitude_min=list(alt_index)).coords["altitude_min"] + 50

    max_alt = snowline_parametrization_dataset.coords["altitude_max"].max()
    r = snowline
    # print(max_alt - r)
    theta = [np.deg2rad(COMPASS_ROSE_DICT[asp]) for asp in snowline_parametrization_dataset.coords["aspect"].values]

    r = [*r, r[0]]
    theta = [*theta, theta[0]]
    ax.set_rlim(max_alt, snowline_parametrization_dataset.coords["altitude_max"].min())
    ax.set_rorigin(max_alt)
    ax.set_theta_direction(-1)  # Clockwise rotation (standard for maps)
    ax.set_theta_offset(np.pi / 2)
    ax.plot(theta, r, label=label)
    ax.grid(True)
    ax.set_title("Snowline", va="bottom")
    ax.legend()
    return ax


if __name__ == "__main__":
    s2_folder = "/home/imperatoren/work/edelweiss_assimilation/observations/granderousses/s2"
    snowline_filepath = f"{s2_folder}/snowline_paremetrization.nc"
    # print(xr.open_dataset(snowline_filepath).sel(slope="10 - 30").snowline_penalization.sel(aspect="N"))
    fig, ax = plt.subplots(1, 1, figsize=(5, 8), subplot_kw={"projection": "polar"}, layout="constrained")
    plot_snowline_polarplot(
        snowline_parametrization_dataset=xr.open_dataset(snowline_filepath).sel(slope="10 - 30"), ax=ax, label="Sentinel-2"
    )

    edelweiss_folder = (
        "/home/imperatoren/work/edelweiss_assimilation/simulations/postprocess/grandesrousses250m/open_loop/all_members"
    )
    snowline_filepath = f"{edelweiss_folder}/snowline_paremetrization.nc"
    plot_snowline_polarplot(
        snowline_parametrization_dataset=xr.open_dataset(snowline_filepath).sel(slope="10 - 30"), ax=ax, label="EDELWEISS"
    )

    viirs_folder = "/home/imperatoren/work/edelweiss_assimilation/observations/granderousses/meteofrance/"
    snowline_filepath = f"{viirs_folder}/snowline_paremetrization.nc"
    plot_snowline_polarplot(
        snowline_parametrization_dataset=xr.open_dataset(snowline_filepath).sel(slope="10 - 30"), ax=ax, label="VIIRS"
    )

    plt.show()
    # plt.show()
