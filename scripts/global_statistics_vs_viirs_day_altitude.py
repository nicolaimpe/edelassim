import matplotlib.pyplot as plt
import numpy as np
import xarray as xr

from edelassim.visualization.day_site import alt_time_diagramm_plot
from edelassim.visualization.static_files import (
    get_snowline_simulation,
    get_snowlines,
    list_satellite_clear_dates,
    read_bera_snowlines,
    simulation_folder,
)


def find_snowline_index(score_dataset: xr.Dataset, snowlines: xr.DataArray):
    return (np.abs(score_dataset.coords["elevation_min"] - snowlines)).argmin(dim="elevation_min")


if __name__ == "__main__":
    xpids = [
        "assim_viirs_cloudcover055_wy2122_soda_dev",
        # "assim_viirs_cloudcover055_wy2122_python_pf",
        # "assim_viirs_cloudcover055_wy2122_sigma_obs_var",
    ]
    plot_labels = ["SODA", "Python", "Sigma var"]
    period = slice("2021-11", "2022-06-15")

    fsc_viirs_ol_scores = xr.open_dataset(f"{simulation_folder}/reanalysis/open_loop/scores/fsc/viirs/scores.nc").sel(
        time=period
    )

    bera_snowlines = read_bera_snowlines().isel(massif=0).sel(time=period)

    snowline_ol, snowline_viirs, _ = get_snowlines(period=period)
    good_dates_viirs, _ = list_satellite_clear_dates(period=period)
    snowline_viirs = snowline_viirs.sel(time=good_dates_viirs)

    # sl_plot_indexes_ol = (
    #     np.abs(fsc_viirs_ol_scores.coords["elevation_min"] - snowline_ol.sel(aspect="N").mean(dim=("member")))
    # ).argmin(dim="elevation_min")

    for idx, xpid in enumerate(xpids):
        fig, ax = plt.subplots(figsize=(18, 8))

        # ax.plot(
        #     (snowline_ol.time - snowline_ol.time[0]).dt.days,
        #     sl_plot_indexes_ol - 0.5,
        #     color=COLORS["edel_ol"],
        #     label=LABELS["edel_ol"],
        #     linewidth=2,
        #     linestyle="dashed",
        # )
        fsc_viirs_assim_scores = xr.open_dataset(f"{simulation_folder}/reanalysis/{xpid}/scores/fsc/viirs/scores.nc").sel(
            time=period
        )
        snowline_assim = get_snowline_simulation(xpid, period=period).sel(member=slice(0, None))

        # score = xr.DataArray(1 - fsc_viirs_assim_scores.data_vars["crps"] / fsc_viirs_ol_scores.data_vars["crps"], name="aem")
        scores = xr.Dataset(
            {
                "sigma_open_loop": fsc_viirs_ol_scores.data_vars["sigma"],
                "diff_aem": xr.DataArray(fsc_viirs_ol_scores.data_vars["aem"] - fsc_viirs_assim_scores.data_vars["aem"]),
            }
        )
        alt_time_diagramm_plot(score_data=scores, ax=ax)
        ### Simulated
        # Find altitude indexes to be plotted as bin numbers
        sl_n_indexes_assim = find_snowline_index(
            score_dataset=fsc_viirs_assim_scores, snowlines=snowline_assim.sel(aspect="N").mean(dim=("member"))
        )
        # sl_s_indexes_assim = find_snowline_index(
        #     score_dataset=fsc_viirs_assim_scores, snowlines=snowline_assim.sel(aspect="S").mean(dim=("member"))
        # )
        # days_of_year = (sl_s_indexes_assim.time - sl_s_indexes_assim.time[0]).dt.days
        # ax.plot(
        #     days_of_year,
        #     sl_n_indexes_assim,
        #     color="darkcyan",
        #     label="Snowline EDELWEISS N",
        #     linestyle="dashed",
        #     linewidth=2,
        # )
        # ax.plot(
        #     days_of_year,
        #     sl_s_indexes_assim,
        #     color="darkorange",
        #     label="Snowline EDELWEISS S",
        #     linestyle="dashed",
        #     linewidth=2,
        # )

        #### BERA
        sl_n_indexes_bera = find_snowline_index(score_dataset=fsc_viirs_assim_scores, snowlines=bera_snowlines.sel(aspect="N"))
        sl_s_indexes_bera = find_snowline_index(score_dataset=fsc_viirs_assim_scores, snowlines=bera_snowlines.sel(aspect="S"))
        days_of_year = (sl_n_indexes_bera.time - sl_n_indexes_assim.time[0]).dt.days
        ax.plot(
            days_of_year, sl_n_indexes_bera, color="darkcyan", label="Snowline forecasters N", linestyle="dashed", linewidth=2
        )
        ax.plot(
            days_of_year,
            sl_s_indexes_bera,
            color="darkorange",
            label="Snowline forecasters S",
            linestyle="dashed",
            linewidth=2,
        )

        ### VIIRS
        # sl_n_indexes_viirs = find_snowline_index(
        #     score_dataset=fsc_viirs_assim_scores, snowlines=snowline_viirs.sel(aspect="N")
        # )
        # sl_s_indexes_viirs = find_snowline_index(
        #     score_dataset=fsc_viirs_assim_scores, snowlines=snowline_viirs.sel(aspect="S")
        # )
        # days_of_year = (sl_n_indexes_viirs.time - sl_n_indexes_viirs.time[0]).dt.days
        # ax.plot(
        #     days_of_year,
        #     sl_n_indexes_viirs,
        #     color="steelblue",
        #     label="Snowline VIIRS N",
        #     linestyle="solid",
        # )
        # ax.plot(
        #     days_of_year,
        #     sl_s_indexes_viirs,
        #     color="darkorange",
        #     label="Snowline VIIRS S",
        #     linestyle="solid",
        # )

        ax.set_title(
            "EDELWEISS FSC (D+1) vs VIIRS FSC (D+1 not assimilated) \n Open loop - assimilation absolute ensemble error (AEM)"
        )
        ax.legend(loc="upper left")
        ax.set_ylim(-0.5, fsc_viirs_assim_scores.sizes["elevation_min"] - 0.5)
        ax.set_xlim(0, fsc_viirs_assim_scores.sizes["time"])
        ax.set_ylabel("Elevation [m]")
        ax.set_xlabel("Time [day]")
        # fig.suptitle(plot_labels[idx])

    plt.show()
