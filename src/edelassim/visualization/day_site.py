from datetime import timedelta

import matplotlib
import numpy as np
import xarray as xr
from matplotlib import cm, colors
from matplotlib.axes import Axes


def RdGn_colormap():
    top = matplotlib.colormaps["Reds_r"].resampled(128)
    bottom = matplotlib.colormaps["Greens"].resampled(128)
    newcolors = np.vstack((top(np.linspace(0, 1, 128)), bottom(np.linspace(0, 1, 128))))

    return colors.ListedColormap(newcolors, name="RdGn")


def alt_time_diagramm_plot(score_data: xr.Dataset, ax: Axes):
    colormap = RdGn_colormap()
    norm = matplotlib.colors.Normalize(vmin=-0.5, vmax=0.5)

    day_of_season = (score_data.time - score_data.time[0]).dt.days
    for i in range(score_data.sizes["elevation_min"]):
        barplot_colors = colormap(norm(score_data.data_vars["diff_aem"].isel(elevation_min=i)))
        ax.barh(
            y=i,
            width=timedelta(days=1).days,
            height=1,
            left=day_of_season,
            facecolor=barplot_colors,
            edgecolor="black",
            linewidth=0.1,
        )
        ax.barh(
            y=i,
            width=timedelta(days=1).days,
            height=1,
            left=day_of_season.where(np.isnan(score_data.data_vars["diff_aem"]).isel(elevation_min=i)),
            facecolor="gray",
            edgecolor="black",
            linewidth=0.1,
        )

    first_of_month = day_of_season.sel(time=day_of_season.coords["time"].dt.day == 1)
    ax.set_xticks(first_of_month.values)
    ax.set_xticklabels(first_of_month.time.dt.strftime("%d-%B").values)
    ax.set_yticks(np.arange(score_data.sizes["elevation_min"]))
    ax.set_yticklabels(
        [
            f"{((el_max + el_min) / 2):.0f} m"
            for el_min, el_max in zip(score_data.coords["elevation_min"].values, score_data.coords["elevation_max"].values)
        ]
    )
    sm = cm.ScalarMappable(cmap=colormap, norm=norm)
    # ax.get_figure().colorbar(sm, ax=ax)
    ax.get_figure().colorbar(sm, ax=ax).set_label("AEM open loop - AEM assimilation")


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
