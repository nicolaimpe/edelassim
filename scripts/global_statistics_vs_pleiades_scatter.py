from datetime import datetime

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import xarray as xr
from matplotlib.axes import Axes
from matplotlib.gridspec import GridSpec
from xarray.groupers import BinGrouper, UniqueGrouper

from edelassim.bdclim import extract_bdclim_locations_on_spatial_dataset
from edelassim.evaluations import compute_rmse
from edelassim.visualization.scatter import plot_data_assimilation_base_simulation, plot_ensemble_simulation
from edelassim.visualization.static_files import (
    COLORS,
    LABELS,
    get_data_assimilation_simulation,
    get_spatial_datasets,
    topography_data_folder,
)


def make_stats_snow_depth(data: xr.Dataset):
    mean = data.data_vars["spatial"].mean(dim=("stacked_y_x"))
    sum = data.data_vars["spatial"].sum(dim=("stacked_y_x"))
    sigma = np.sqrt(((data.data_vars["spatial"] - mean) ** 2).sum(dim=("stacked_y_x")))
    volume = sum * (250**2)
    return xr.Dataset({"mean": mean, "sum": sum, "volume": volume, "sigma": sigma})


def compute_snow_depth_stats(spatial_data: xr.DataArray, dem: xr.DataArray, altitude_step: int | None = None):
    altitude_bins = BinGrouper(np.arange(1000, 3501, altitude_step), labels=np.arange(1000, 3500, altitude_step))
    grouped_dataset = xr.Dataset({"spatial": spatial_data, "elevation": dem})
    return grouped_dataset.groupby(elevation=altitude_bins).map(make_stats_snow_depth)


def plot_snow_depth_boxplot(spatial_data: xr.DataArray, ax: Axes, offset: float):
    ax.boxplot(spatial_data.snow_depth.values.flatten(), 2, orientation="horizontal")
    return xr.DataArray(0)


def plot_var_boxplot_array(data: xr.Dataset, ax: Axes, var: str, offset: float, width: float = 0.4, color: str = "blue"):
    # print(data.data_vars["open_loop"])
    open_loop_residuals = data.data_vars["snow_depth"].values.flatten()
    # width = 0.4
    # print(width)
    # print(len(data[var]))
    bp = ax.boxplot(
        [open_loop_residuals[~np.isnan(open_loop_residuals)]],
        positions=[data[var].mean() - width / 2 + offset],
        showfliers=False,
        notch=True,
        patch_artist=True,
        widths=width,
        orientation="horizontal",
        label=[LABELS["edel_ol"]],
    )
    bp["boxes"][0].set_facecolor(color)
    # bp["boxes"][1].set_facecolor(COLORS["edel_an"])
    return data


if __name__ == "__main__":
    xpids = ["assim_viirs_cloudcover055_wy2122_python_pf", "assim_viirs_cloudcover055_wy2122_sigma_obs_var"]
    plot_colors = ["magenta", "darkgreen"]
    plot_labels = ["sigma const", "sigma var"]
    period = slice(datetime(year=2021, month=11, day=1), datetime(year=2022, month=6, day=15))
    alt_step = 100

    edelweiss_openloop, viirs, s2, pleiades, bdclim = get_spatial_datasets(period=period)
    edelweiss_openloop = edelweiss_openloop.sel(member=slice(0, None)).data_vars["DSN_T_ISBA"]
    pleiades = pleiades.sel(band=1).data_vars["snow_depth"]
    dem = xr.open_dataarray(f"{topography_data_folder}/250m/DEM_GR_L93_250m.tif").sel(band=1)
    # sd_edel_station_openloop = extract_bdclim_locations_on_spatial_dataset(bdclim, spatial_dataset=edelweiss_openloop)

    ### Snow depth vs altitude scatter
    # Pleiades

    grid_spec = GridSpec(nrows=2, ncols=2, hspace=0.25, height_ratios=(1, 0.33), width_ratios=(1, 0.15))
    n_bars = len(xpids) + 2
    bar_width = (alt_step - alt_step / 4) / n_bars
    bars_offest = n_bars // 2 * bar_width + bar_width / 2

    for pleiades_date in pleiades.coords["time"][1:]:
        fig = plt.figure()
        ax = fig.add_subplot(grid_spec[0])

        stats_ol = compute_snow_depth_stats(
            spatial_data=edelweiss_openloop.sel(time=pleiades_date).where(pleiades.sel(time=pleiades_date) >= 0),
            dem=dem,
            altitude_step=alt_step,
        )

        stats_pleiades = compute_snow_depth_stats(
            spatial_data=pleiades.sel(time=pleiades_date),
            dem=dem,
            altitude_step=alt_step,
        )
        xaxis = stats_ol.coords["elevation_bins"].values
        plot_data_assimilation_base_simulation(
            open_loop=stats_ol.data_vars["mean"],
            x_axis_data=xaxis,
            ax=ax,
            reference=stats_pleiades.data_vars["mean"],
        )
        ax_volume = fig.add_subplot(grid_spec[2])

        ax_volume.bar(
            x=xaxis - bars_offest,
            height=stats_pleiades.data_vars["volume"],
            width=bar_width,
            color="black",
        )
        ax_volume.bar(
            x=xaxis - bars_offest + bar_width,
            height=stats_ol.data_vars["volume"].mean(dim="member"),
            width=bar_width,
            color=COLORS["edel_ol"],
            yerr=compute_rmse(stats_ol.data_vars["volume"] - stats_ol.data_vars["volume"].mean(dim="member")),
        )

        ax_total = fig.add_subplot(grid_spec[3])
        ax_total.bar(
            x=-bars_offest,
            height=stats_pleiades.data_vars["volume"].sum(dim="elevation_bins"),
            width=bar_width,
            color="black",
        )
        ax_total.bar(
            x=-bars_offest + bar_width,
            height=stats_ol.data_vars["volume"].mean(dim="member").sum(dim="elevation_bins"),
            width=bar_width,
            color=COLORS["edel_ol"],
            yerr=compute_rmse(
                stats_ol.data_vars["volume"].sum(dim="elevation_bins")
                - stats_ol.data_vars["volume"].mean(dim="member").sum(dim="elevation_bins")
            ),
        )

        ax_boxplots = fig.add_subplot(grid_spec[1])
        snow_depth_bins = BinGrouper(np.arange(0, 3.1, 0.25), labels=np.arange(0, 3, 0.25))
        xr.Dataset({"snow_depth": pleiades.sel(time=pleiades_date)}).groupby(snow_depth=snow_depth_bins).map(
            plot_var_boxplot_array, ax=ax_boxplots, var="snow_depth", offset=0, width=0.1, color="black"
        )
        xr.Dataset(
            {
                "snow_depth": edelweiss_openloop.where(pleiades.sel(time=pleiades_date) >= 0)
                .sel(time=pleiades_date)
                .mean(dim="member")
            }
        ).groupby(snow_depth=snow_depth_bins).map(
            plot_var_boxplot_array, ax=ax_boxplots, var="snow_depth", offset=0.1, width=0.1, color=COLORS["edel_ol"]
        )
        for idx, xpid in enumerate(xpids):
            edelweiss_assimilation = (
                get_data_assimilation_simulation(xpid, period=period).sel(member=slice(0, None)).data_vars["DSN_T_ISBA"]
            )
            stats_assim = compute_snow_depth_stats(
                spatial_data=edelweiss_assimilation.sel(time=pleiades_date).where(pleiades.sel(time=pleiades_date) >= 0),
                dem=dem,
                altitude_step=alt_step,
            )

            plot_ensemble_simulation(
                simulation_data=stats_assim.data_vars["mean"],
                x_axis_data=xaxis,
                ax=ax,
                color=plot_colors[idx],
                label=plot_labels[idx],
            )
            ax_volume.bar(
                x=xaxis - bars_offest + ((2 + idx) * bar_width),
                height=stats_assim.data_vars["volume"].mean(dim="member"),
                width=bar_width,
                color=plot_colors[idx],
                yerr=compute_rmse(stats_assim.data_vars["volume"] - stats_assim.data_vars["volume"].mean(dim="member")),
            )
            ax_total.bar(
                x=-bars_offest + ((2 + idx) * bar_width),
                height=stats_assim.data_vars["volume"].mean(dim="member").sum(dim="elevation_bins"),
                width=bar_width,
                color=plot_colors[idx],
                yerr=compute_rmse(
                    stats_assim.data_vars["volume"].sum(dim="elevation_bins")
                    - stats_assim.data_vars["volume"].mean(dim="member").sum(dim="elevation_bins")
                ),
            )
            xr.Dataset(
                {
                    "snow_depth": edelweiss_openloop.where(pleiades.sel(time=pleiades_date) >= 0)
                    .sel(time=pleiades_date)
                    .mean(dim="member")
                }
            ).groupby(snow_depth=snow_depth_bins).map(
                plot_var_boxplot_array,
                ax=ax_boxplots,
                var="snow_depth",
                offset=0.1 + 0.1 * (2 + idx),
                width=0.1,
                color=plot_colors[idx],
            )

        ax.set_title(f"Edelweiss vs Pleiades - {pd.Timestamp(pleiades_date.values).strftime('%d-%b-%Y')}- {alt_step} m bins")
        ax.set_ylabel("Mean snow depth [m]")
        ax.set_xlabel("Elevation [m]")
        ax.grid()
        ax.legend()

        ax_volume.grid()
        ax_volume.set_ylabel("Snow volume [m³]")
        ax.set_xlabel("Elevation [m]")

        ax_total.set_title("Total snow volume")
        ax_total.set_ylabel("Snow volume [m³]")
        ax_total.set_xticks([])

    plt.show()
