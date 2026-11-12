from datetime import datetime, timedelta
from glob import glob

import matplotlib
import matplotlib.cm as cm
import matplotlib.colors as colors
import matplotlib.pyplot as plt
import numpy as np
import xarray as xr
from matplotlib.axes import Axes
from matplotlib.gridspec import GridSpec
from xarray.groupers import BinGrouper, UniqueGrouper

from edelassim.bdclim import extract_bdclim_locations_on_spatial_dataset
from edelassim.evaluations import compute_scores
from edelassim.postprocess_surfex.soda import duplicated_particles_multiple_assimilation
from edelassim.snowlines import find_snowline_from_snow_penalization
from edelassim.visualization.scatter import plot_data_assimilation_base_simulation, plot_ensemble_simulation
from edelassim.visualization.static_files import (
    COLORS,
    LABELS,
    get_snowline_simulation,
    get_snowlines,
    get_spatial_datasets,
    list_satellite_clear_dates,
    mask,
    s2_path,
    viirs_file,
    working_folder,
)

if __name__ == "__main__":
    xpids = ["assim_viirs_cloudcover055_wy2122_python_pf", "assim_viirs_cloudcover055_wy2122_sigma_obs_var"]
    plot_colors = ["magenta", "darkgreen"]
    labels = ["sigma const", "sigma var"]
    period = slice(datetime(year=2021, month=11, day=1), datetime(year=2022, month=6, day=15))

    score = "mae_mean_member"

    good_dates_viirs, good_dates_s2 = list_satellite_clear_dates(period=period)

    good_dates_s2.pop(6)
    good_dates_s2.pop(-13)
    # good_dates_s2.pop(9)
    grid_spec = GridSpec(nrows=4, ncols=1, height_ratios=(1, 0.5, 1, 0.5))
    fig = plt.figure(figsize=(18, 14))
    fig.suptitle("Snowline vs time")
    snowline_openloop_edel, snowline_viirs, snowline_s2 = get_snowlines()
    for asp_idx, asp in enumerate(["N", "S"]):
        ol = snowline_openloop_edel.sel(member=slice(0, None)).sel(aspect=asp)
        s2 = snowline_s2.sel(time=good_dates_s2).sel(aspect=asp)
        viirs = snowline_viirs.sel(time=good_dates_viirs).sel(aspect=asp)
        ax = fig.add_subplot(grid_spec[2 * asp_idx])
        ax.set_ylim(500, 3500)
        plot_data_assimilation_base_simulation(open_loop=ol, x_axis_data=ol.coords["time"], ax=ax, reference=None)
        ax.plot(
            s2.coords["time"],
            s2,
            color=COLORS["s2"],
            label=LABELS["s2"],
            marker="s",
            mfc="none",
            linestyle="dashed",
            linewidth=0.5,
        )
        ax.plot(
            viirs.coords["time"],
            viirs,
            color=COLORS["viirs"],
            label=LABELS["viirs"],
            marker="o",
            markersize=2,
            mfc="none",
            linestyle="dashed",
            linewidth=0.5,
        )

        ax_scores = fig.add_subplot(grid_spec[2 * asp_idx + 1])
        ax_scores.set_ylim(0, 600)
        # Sentinel-2
        grouped_data = xr.Dataset(
            {
                "eval": (("member", "time"), ol.sel(time=good_dates_s2).values),
                "reference": (("time"), s2.values),
            },
            coords=snowline_s2.sel(time=good_dates_s2).coords,
        )
        scores_ol = grouped_data.groupby("time").map(compute_scores)
        scores_ol = scores_ol.assign({"mae_mean_member": np.abs(scores_ol.data_vars["diff_mean_member"])})
        time_mean = scores_ol.data_vars[score].mean(dim="time").values
        ax_scores.plot(
            scores_ol.coords["time"],
            scores_ol.data_vars[score],
            linewidth=0,
            marker="s",
            mfc="none",
            color=COLORS["edel_ol"],
            label=f"OL vs S2 - mean: {time_mean:.2f}",
        )
        # VIIRS
        grouped_data = xr.Dataset(
            {
                "eval": (("member", "time"), ol.sel(time=good_dates_viirs).values),
                "reference": (("time"), viirs.values),
            },
            coords=viirs.coords,
        )
        scores_ol = grouped_data.groupby("time").map(compute_scores)
        scores_ol = scores_ol.assign({"mae_mean_member": np.abs(scores_ol.data_vars["diff_mean_member"])})
        time_mean = scores_ol.data_vars[score].mean(dim="time").values
        ax_scores.plot(
            scores_ol.coords["time"],
            scores_ol.data_vars[score],
            linewidth=0,
            marker="o",
            markersize=2,
            mfc="none",
            color=COLORS["edel_ol"],
            label=f"Open loop vs VIIRS - mean: {time_mean:.2f}",
        )
        for xpid_idx, xpid in enumerate(xpids):
            snowline_analysis_edel = find_snowline_from_snow_penalization(get_snowline_simulation(xpid, period=period))

            analysis = snowline_analysis_edel.sel(member=slice(0, None)).sel(aspect=asp)

            plot_ensemble_simulation(
                simulation_data=analysis,
                ax=ax,
                x_axis_data=analysis.coords["time"],
                color=plot_colors[xpid_idx],
                label=labels[xpid_idx],
            )
            # Sentinel-2
            grouped_data = xr.Dataset(
                {
                    "eval": (("member", "time"), analysis.sel(time=good_dates_s2).values),
                    "reference": (("time"), s2.values),
                },
                coords=snowline_s2.sel(time=good_dates_s2).coords,
            )
            scores_analysis = grouped_data.groupby("time").map(compute_scores)
            scores_analysis = scores_analysis.assign(
                {"mae_mean_member": np.abs(scores_analysis.data_vars["diff_mean_member"])}
            )
            time_mean = scores_analysis.data_vars[score].mean(dim="time").values
            ax_scores.plot(
                scores_analysis.coords["time"],
                scores_analysis.data_vars[score],
                linewidth=0,
                marker="s",
                mfc="none",
                color=plot_colors[xpid_idx],
                label=f"Assim vs S2 - mean: {time_mean:.2f}",
            )
            # VIIRS
            grouped_data = xr.Dataset(
                {
                    "eval": (("member", "time"), analysis.sel(time=good_dates_viirs).values),
                    "reference": (("time"), viirs.values),
                },
                coords=viirs.coords,
            )
            scores_analysis = grouped_data.groupby("time").map(compute_scores)
            scores_analysis = scores_analysis.assign(
                {"mae_mean_member": np.abs(scores_analysis.data_vars["diff_mean_member"])}
            )
            time_mean = scores_analysis.data_vars[score].mean(dim="time").values
            ax_scores.plot(
                scores_analysis.coords["time"],
                scores_analysis.data_vars[score],
                linewidth=0,
                marker="o",
                markersize=2,
                mfc="none",
                color=plot_colors[xpid_idx],
                label=f"{labels[xpid_idx]} vs VIIRS - mean: {time_mean:.2f}",
            )

            part_files = glob(f"{working_folder}/simulations/edelweiss/reanalysis/{xpid}/soda/*PART*")
            duplicated_particles = duplicated_particles_multiple_assimilation(part_files=part_files)
            assim_dates = duplicated_particles.sel(time=period).coords["time"]
            for axx in (ax, ax_scores):
                axx.vlines(
                    assim_dates,
                    ymin=axx.get_ylim()[0],
                    ymax=axx.get_ylim()[1],
                    color=plot_colors[xpid_idx],
                    linestyle="dashed",
                    linewidth=0.25,
                    label="assim dates",
                )

        ax.set_title(f"Snowline - {asp}")
        ax.set_ylabel("elevation [m]")
        ax.grid()
        ax.legend(fontsize=6)

        ax_scores.set_title(f"{score} - {asp}")
        ax_scores.set_ylabel("elevation [m]")
        ax_scores.legend(fontsize=6)
        ax_scores.grid()

    fig.subplots_adjust(hspace=0.35)
    plt.show()
