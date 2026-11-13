from datetime import datetime
from glob import glob

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.gridspec import GridSpec

from edelassim.postprocess_surfex.soda import duplicated_particles_multiple_assimilation
from edelassim.visualization.scatter import plot_data_assimilation_base_simulation, plot_ensemble_simulation
from edelassim.visualization.static_files import (
    COLORS,
    LABELS,
    get_snowline_simulation,
    get_snowlines,
    list_satellite_clear_dates,
    read_bera_snowlines,
    working_folder,
)

if __name__ == "__main__":
    xpids = ["assim_viirs_cloudcover055_wy2122_python_pf", "assim_viirs_cloudcover055_wy2122_sigma_obs_var"]
    plot_colors = ["magenta", "darkgreen"]
    labels = ["sigma const", "sigma var"]
    period = slice(datetime(year=2021, month=11, day=1), datetime(year=2022, month=6, day=15))

    good_dates_viirs, good_dates_s2 = list_satellite_clear_dates(period=period)

    good_dates_s2.pop(6)
    good_dates_s2.pop(-13)
    # good_dates_s2.pop(9)
    grid_spec = GridSpec(nrows=4, ncols=1, height_ratios=(1, 0.5, 1, 0.5))
    fig = plt.figure(figsize=(18, 14))
    fig.suptitle("Snowline vs time")
    snowline_openloop_edel, snowline_viirs, snowline_s2 = get_snowlines(period=period)
    bera_snowlines = read_bera_snowlines().isel(massif=0).sel(time=period)
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
        ax.plot(
            bera_snowlines.coords["time"],
            bera_snowlines.sel(aspect=asp),
            color="orange",
            label="forecasted",
            marker="^",
            markersize=2,
            mfc="none",
            linestyle="dashed",
            linewidth=0.5,
        )

        ax_scores = fig.add_subplot(grid_spec[2 * asp_idx + 1])
        ax_scores.set_ylim(0, 600)
        # Sentinel-2
        aem_s2_ol = np.abs(ol.sel(time=good_dates_s2).mean(dim="member") - s2)
        time_mean = aem_s2_ol.mean(dim="time")
        ax_scores.plot(
            aem_s2_ol.coords["time"],
            aem_s2_ol,
            linewidth=0,
            marker="s",
            mfc="none",
            color=COLORS["edel_ol"],
            label=f"OL vs S2 - mean: {time_mean:.2f}",
        )
        # VIIRS
        aem_viirs_ol = np.abs(ol.sel(time=good_dates_viirs).mean(dim="member") - viirs)
        time_mean = aem_viirs_ol.mean(dim="time")
        ax_scores.plot(
            aem_viirs_ol.coords["time"],
            aem_viirs_ol,
            linewidth=0,
            marker="o",
            markersize=2,
            mfc="none",
            color=COLORS["edel_ol"],
            label=f"Open loop vs VIIRS - mean: {time_mean:.2f}",
        )
        # BERA
        aem_bera_ol = np.abs(ol.mean(dim="member") - bera_snowlines.sel(aspect=asp))
        time_mean = aem_bera_ol.mean(dim="time")
        ax_scores.plot(
            aem_bera_ol.coords["time"],
            aem_bera_ol,
            linewidth=0,
            marker="^",
            markersize=2,
            mfc="none",
            color=COLORS["edel_ol"],
            label=f"Open loop vs forecasted - mean: {time_mean:.2f}",
        )
        for xpid_idx, xpid in enumerate(xpids):
            snowline_analysis_edel = get_snowline_simulation(xpid, period=period)

            analysis = snowline_analysis_edel.sel(member=slice(0, None)).sel(aspect=asp)

            plot_ensemble_simulation(
                simulation_data=analysis,
                ax=ax,
                x_axis_data=analysis.coords["time"],
                color=plot_colors[xpid_idx],
                label=labels[xpid_idx],
            )
            # Sentinel-2
            aem_s2_assim = np.abs(analysis.sel(time=good_dates_s2).mean(dim="member") - s2)
            time_mean = aem_s2_assim.mean(dim="time")
            ax_scores.plot(
                aem_s2_assim.coords["time"],
                aem_s2_assim,
                linewidth=0,
                marker="s",
                mfc="none",
                color=plot_colors[xpid_idx],
                label=f"Assim vs S2 - mean: {time_mean:.2f}",
            )
            # VIIRS
            aem_viirs_assim = np.abs(analysis.sel(time=good_dates_viirs).mean(dim="member") - viirs)
            time_mean = aem_viirs_assim.mean(dim="time")
            ax_scores.plot(
                aem_viirs_assim.coords["time"],
                aem_viirs_assim,
                linewidth=0,
                marker="o",
                markersize=2,
                mfc="none",
                color=plot_colors[xpid_idx],
                label=f"{labels[xpid_idx]} vs VIIRS - mean: {time_mean:.2f}",
            )

            # BERA
            aem_bera_assim = np.abs(analysis.mean(dim="member") - bera_snowlines.sel(aspect=asp))
            time_mean = np.nanmean(aem_bera_assim)  # .mean(dim="time")
            ax_scores.plot(
                aem_bera_assim.coords["time"],
                aem_bera_assim,
                linewidth=0,
                marker="^",
                markersize=2,
                mfc="none",
                color=plot_colors[xpid_idx],
                label=f"{labels[xpid_idx]} vs forecasted - mean: {time_mean:.2f}",
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
                    # label="assim dates",
                )

        ax.set_title(f"Snowline - {asp}")
        ax.set_ylabel("elevation [m]")
        ax.grid()
        ax.legend(fontsize=6)

        ax_scores.set_title(f"Absolute ensemble difference - {asp}")
        ax_scores.set_ylabel("elevation [m]")
        ax_scores.legend(fontsize=6)
        ax_scores.grid()

    fig.subplots_adjust(hspace=0.35)
    plt.show()
