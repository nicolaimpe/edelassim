from datetime import timedelta

import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import xarray as xr
from matplotlib import cm, colors
from matplotlib.axes import Axes

from edelassim.visualization.day_site import RdGn_colormap
from edelassim.visualization.static_files import get_analysis_station_scores, get_open_loop_station_scores


def day_site_score_plot(score_data: xr.DataArray, ax: Axes):
    score_data = score_data.sortby("elevation")
    good_days_mask = score_data > 0.5
    bad_days_mask = score_data < -0.5
    nanmask = np.isnan(score_data)
    neutral_days_mask = ~(good_days_mask + bad_days_mask + nanmask)
    total_day_site = score_data.count().values
    norm = colors.Normalize(vmin=-1, vmax=1)
    colormap = RdGn_colormap()

    day_of_season = (score_data.time - score_data.time[0]).dt.days

    for i in range(score_data.sizes["num_poste"]):
        barplot_colors = colormap(norm(score_data.isel(num_poste=i)))

        ax.barh(
            y=i,
            width=timedelta(days=1).days,
            height=1,
            left=day_of_season,
            facecolor=barplot_colors,
            edgecolor="black",
            linewidth=0.1,
        )
        # Significantly Bad days to plot before good days so thick line width is not replaces
        ax.barh(
            y=i,
            width=timedelta(days=1).days,
            height=1,
            left=day_of_season.where(bad_days_mask.isel(num_poste=i)),
            facecolor="none",
            edgecolor="black",
            linewidth=0.5,
        )
        # Significantly good days
        ax.barh(
            y=i,
            width=timedelta(days=1).days,
            height=1,
            left=day_of_season.where(good_days_mask.isel(num_poste=i)),
            facecolor="none",
            edgecolor="black",
            linewidth=0.5,
        )
        # No measure days in gray
        ax.barh(
            y=i,
            width=timedelta(days=1).days,
            height=1,
            left=day_of_season.where(nanmask.isel(num_poste=i)),
            facecolor="gray",
            edgecolor="black",
            linewidth=0.1,
        )

    first_of_month = day_of_season.sel(time=day_of_season.coords["time"].dt.day == 1)
    ax.set_xticks(first_of_month.values)
    ax.set_xticklabels(first_of_month.time.dt.strftime("%d-%B").values)
    ax.set_yticks(np.arange(score_data.sizes["num_poste"]))
    ax.set_yticklabels(
        [
            f"{name.title()} - {elevation} m"
            for name, elevation in zip(score_data.coords["station_name"].values, score_data.coords["elevation"].values)
        ]
    )
    ax.set_title(
        f"{score_data.name} - {good_days_mask.sum().values} / {total_day_site} significantly improved, \
            {bad_days_mask.sum().values} / {total_day_site} significantly degraded\
            {neutral_days_mask.sum().values} / {total_day_site} neutral day-site"
    )
    sm = cm.ScalarMappable(cmap=colormap, norm=norm)
    cbar = ax.get_figure().colorbar(sm, ax=ax)
    cbar.set_label(f"{score_data.name}")


if __name__ == "__main__":
    xpids = [
        "assim_viirs_cloudcover055_wy2122_soda_dev",
        "assim_viirs_cloudcover055_wy2122_python_pf",
        "assim_viirs_cloudcover055_wy2122_sigma_obs_var",
    ]
    plot_labels = ["SODA", "Python", "Sigma var"]
    period = slice("2021-11", "2022-06")

    sd_bdclim_ol_scores, fsc_bdclim_ol_scores = get_open_loop_station_scores(period=period)

    for idx, xpid in enumerate(xpids):
        fig, axs = plt.subplots(nrows=2, figsize=(18, 10))
        fig.suptitle(plot_labels[idx])
        sd_assim_scores, fsc_assim_scores = get_analysis_station_scores(xpid=xpid, period=period)

        score = xr.DataArray(1 - sd_assim_scores.data_vars["crps"] / sd_bdclim_ol_scores.data_vars["crps"], name="crpss")
        day_site_score_plot(score_data=score, ax=axs[0])
        axs[0].set_title(f"SD - {axs[0].get_title()}")

        score = xr.DataArray(1 - fsc_assim_scores.data_vars["crps"] / fsc_bdclim_ol_scores.data_vars["crps"], name="crpss")
        day_site_score_plot(score_data=score, ax=axs[1])
        axs[1].set_title(f"FSC -- {axs[1].get_title()}")

    plt.show()
