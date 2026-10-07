import numpy as np
import xarray as xr
from matplotlib.axes import Axes
from scipy.special import logit
from scipy.stats import pearsonr

from edelassim.evaluations import compute_rmse, find_common_correspondences
from edelassim.visualization.static_files import COLORS, LABELS


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
    data1: xr.Dataset, data2: xr.Dataset, ax_normal: Axes, ax_logit: Axes, pos: float, width: float, **kwargs
) -> None:
    data1_correspondences, data2_correspondences = find_common_correspondences(data_1=data1, data_2=data2)
    data1_correspondences = data1_correspondences.ravel()
    data2_correspondences = data2_correspondences.ravel()
    residuals = data2_correspondences - data1_correspondences
    residuals = residuals[~np.isnan(residuals)]
    bp = ax_normal.boxplot(
        residuals, positions=pos, showfliers=False, notch=True, patch_artist=True, widths=width, label=kwargs["label"]
    )
    bp["boxes"][0].set_facecolor(kwargs["color"])

    residuals_logit = logit(data2_correspondences) - logit(data1_correspondences)
    residuals_logit = residuals_logit[~np.isnan(residuals_logit)]
    residuals_logit = residuals_logit[~np.isinf(residuals_logit)]
    bp = ax_logit.boxplot(
        residuals_logit, positions=pos, showfliers=False, notch=True, patch_artist=True, widths=width, label=kwargs["label"]
    )
    bp["boxes"][0].set_facecolor(kwargs["color"])


def plot_ensemble_envelop(grouped_dataset: xr.Dataset, x_axis_data: np.ndarray, ax: Axes):
    # Should fix this member mean of Pleiades
    qmin, qmax = 0.1, 0.9
    median_openloop = grouped_dataset.data_vars["open_loop"].median(dim="member").values.flatten()
    median_analysis = grouped_dataset.data_vars["analysis"].median(dim="member").values.flatten()
    reference = grouped_dataset.data_vars["bdclim"].mean(dim="member").values.flatten()

    rmse_openloop = compute_rmse(median_openloop - reference)
    rmse_analysis = compute_rmse(median_analysis - reference)
    ax.fill_between(
        x_axis_data,
        grouped_dataset.data_vars["open_loop"].quantile(qmin, dim="member").values.flatten(),
        grouped_dataset.data_vars["open_loop"].quantile(qmax, dim="member").values.flatten(),
        alpha=0.25,
        color=COLORS["edel_ol"],
        label=f"{LABELS['edel_ol']} p{qmin * 100} - p{qmax * 100}",
    )
    ax.fill_between(
        x_axis_data,
        grouped_dataset.data_vars["analysis"].quantile(qmin, dim="member").values.flatten(),
        grouped_dataset.data_vars["analysis"].quantile(qmax, dim="member").values.flatten(),
        alpha=0.25,
        color=COLORS["edel_an"],
        label=f"{LABELS['edel_an']} p{qmin * 100} - p{qmax * 100}",
    )

    ax.plot(
        x_axis_data,
        median_openloop,
        color=COLORS["edel_ol"],
        linewidth=2,
        linestyle="dashed",
        label=f"{LABELS['edel_ol']} median, RMSE={rmse_openloop:.2f}",
    )

    ax.plot(
        x_axis_data,
        median_analysis,
        color=COLORS["edel_an"],
        label=f"{LABELS['edel_an']} median, RMSE={rmse_analysis:.2f}",
        linewidth=2,
        linestyle="dashed",
    )
    ax.plot(x_axis_data, reference, color="black", linestyle="dashed", linewidth=2, label="In situ")
