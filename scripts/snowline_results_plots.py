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
from edelassim.evaluations import compute_scores_grouped_data, compute_scores_grouped_data_spatial
from edelassim.postprocess_surfex.soda import duplicated_particles_multiple_assimilation
from edelassim.snowlines import find_snowline_from_snow_penalization
from edelassim.visualization.scatter import plot_ensemble_envelop
from edelassim.visualization.static_files import (
    COLORS,
    LABELS,
    get_snowlines,
    get_spatial_datasets,
    good_dates_s2,
    good_dates_viirs,
    mask,
    s2_path,
    viirs_file,
    working_folder,
)

if __name__ == "__main__":
    xpid = "assim_viirs_cloudcover05_wy2122_d93_50cm"
    period = slice(datetime(year=2021, month=11, day=1), datetime(year=2022, month=5, day=25))
    qmin, qmax = 0.1, 0.9

    part_files = glob(f"{working_folder}/simulations/edelweiss/reanalysis/{xpid}/soda/*PART*")
    duplicated_particles = duplicated_particles_multiple_assimilation(part_files=part_files)
    assim_dates = duplicated_particles.coords["time"]

    score = "mae_mean_member"

    good_dates_viirs = [gd for gd in good_dates_viirs if period.start <= gd <= period.stop]
    good_dates_s2 = [gd for gd in good_dates_s2 if period.start <= gd <= period.stop]

    grid_spec = GridSpec(nrows=4, ncols=1, height_ratios=(1, 0.5, 1, 0.5))
    fig = plt.figure(figsize=(18, 14))
    fig.suptitle("Snowline vs time")

    for i, asp in enumerate(["N", "S"]):
        ax_n = fig.add_subplot(grid_spec[2 * i])

        snowlines = get_snowlines(xpid)
        snowlines = [find_snowline_from_snow_penalization(sl).sel(time=period).sel(aspect=asp) for sl in snowlines]
        snowline_openloop_edel, snowline_analysis_edel, snowline_viirs, snowline_s2 = snowlines

        grouped_data = xr.Dataset(
            {
                "open_loop": (
                    ("member", "time"),
                    snowline_openloop_edel.sel(time=good_dates_s2, member=slice(0, None)).values,
                ),
                "analysis": (("member", "time"), snowline_analysis_edel.sel(time=good_dates_s2, member=slice(0, None)).values),
                "reference": (("time"), snowline_s2.sel(time=good_dates_s2).values),
            },
            coords=snowline_s2.sel(time=good_dates_s2).coords,
        )

        plot_ensemble_envelop(grouped_dataset=grouped_data, x_axis_data=grouped_data.time, ax=ax_n, plot_ref=False)
        ax_n.plot(
            snowline_s2.sel(time=good_dates_s2).time,
            snowline_s2.sel(time=good_dates_s2).values.flatten(),
            color=COLORS["s2"],
            label=LABELS["s2"],
            marker="s",
            mfc="none",
            linestyle="dashed",
            linewidth=0.5,
        )
        ax_n.plot(
            snowline_viirs.sel(time=good_dates_viirs).time,
            snowline_viirs.sel(time=good_dates_viirs).values.flatten(),
            color=COLORS["viirs"],
            label=LABELS["viirs"],
            marker="o",
            markersize=2,
            mfc="none",
            linestyle="dashed",
            linewidth=0.5,
        )
        ax_n.set_title(f"Snowline - {asp}")
        ax_n.set_ylabel("elevation [m]")
        ax_n.grid()
        ax_n.legend(fontsize=6)
        # snowline_openloop_edel = snowline_openloop_edel.drop_vars("altitude")
        # snowline_analysis_edel = snowline_analysis_edel.drop_vars("altitude")
        # snowline_s2 = snowline_s2.drop_vars("altitude")
        ax_scores_n = fig.add_subplot(grid_spec[2 * i + 1])

        # Sentinel-2

        scores = grouped_data.groupby("time").map(compute_scores_grouped_data)
        scores = scores.assign({"mae_mean_member": np.abs(scores.data_vars["diff_mean_member"])})
        time_mean = scores.data_vars[score].sel(type="open_loop").mean(dim="time").values
        ax_scores_n.plot(
            scores.coords["time"],
            scores.data_vars[score].sel(type="open_loop"),
            linewidth=0,
            marker="s",
            mfc="none",
            color=COLORS["edel_ol"],
            label=f"OL vs S2 - mean: {time_mean:.2f}",
        )
        time_mean = scores.data_vars[score].sel(type="analysis").mean(dim="time").values
        ax_scores_n.plot(
            scores.coords["time"],
            scores.data_vars[score].sel(type="analysis"),
            linewidth=0,
            marker="s",
            mfc="none",
            color=COLORS["edel_an"],
            label=f"Assim vs S2 - mean: {time_mean:.2f}",
        )

        # VIIRS
        grouped_data = xr.Dataset(
            {
                "open_loop": (
                    ("member", "time"),
                    snowline_openloop_edel.sel(time=good_dates_viirs, member=slice(0, None)).values,
                ),
                "analysis": (
                    ("member", "time"),
                    snowline_analysis_edel.sel(time=good_dates_viirs, member=slice(0, None)).values,
                ),
                "reference": (("time"), snowline_viirs.sel(time=good_dates_viirs).values),
            },
            coords=snowline_viirs.sel(time=good_dates_viirs).coords,
        )

        scores = grouped_data.groupby("time").map(compute_scores_grouped_data)
        scores = scores.assign({"mae_mean_member": np.abs(scores.data_vars["diff_mean_member"])})
        time_mean = scores.data_vars[score].sel(type="open_loop").mean(dim="time").values
        ax_scores_n.plot(
            scores.coords["time"],
            scores.data_vars[score].sel(type="open_loop"),
            linewidth=0,
            marker="o",
            markersize=2,
            mfc="none",
            color=COLORS["edel_ol"],
            label=f"Open loop vs VIIRS - mean: {time_mean:.2f}",
        )
        time_mean = scores.data_vars[score].sel(type="analysis").mean(dim="time").values
        ax_scores_n.plot(
            scores.coords["time"],
            scores.data_vars[score].sel(type="analysis"),
            linewidth=0,
            marker="o",
            markersize=2,
            mfc="none",
            color=COLORS["edel_an"],
            label=f"Assim vs VIIRS - mean: {time_mean:.2f}",
        )

        ax_scores_n.set_ylim(0, 600)
        ax_scores_n.set_title(f"{score} - {asp}")
        ax_scores_n.set_ylabel("elevation [m]")
        ax_scores_n.legend(fontsize=6)
        ax_scores_n.grid()

        for ax in (ax_n, ax_scores_n):
            ax.vlines(
                assim_dates,
                ymin=ax.get_ylim()[0],
                ymax=ax.get_ylim()[1],
                color="gray",
                linestyle="dashed",
                linewidth=0.25,
                label="assim dates",
            )

    fig.subplots_adjust(hspace=0.35)
    plt.show()
