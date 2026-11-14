import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec

from edelassim.bdclim import extract_bdclim_locations_on_spatial_dataset, filter_discontinuous_stations
from edelassim.observation_operators import dickinson
from edelassim.visualization.scatter import plot_data_assimilation_base_simulation, plot_ensemble_simulation
from edelassim.visualization.static_files import (
    COLORS,
    LABELS,
    get_data_assimilation_simulation,
    get_spatial_datasets,
    get_station_location_data,
    list_satellite_clear_dates,
)

if __name__ == "__main__":
    ### Inputs and preprocessing
    xpids = ["assim_viirs_cloudcover055_wy2122_soda_dev", "assim_viirs_cloudcover055_wy2122_python_pf"]
    plot_colors = ["magenta", "darkgreen"]
    plot_labels = ["SODA", "Python"]
    period = slice("2021-11", "2022-06")
    edelweiss_openloop, viirs, s2, _, bdclim = get_spatial_datasets(period=period)
    good_dates_viirs, good_dates_s2 = list_satellite_clear_dates(period=period)
    viirs = viirs.sel(time=good_dates_viirs)
    s2 = s2.sel(time=good_dates_s2)

    bdclim = filter_discontinuous_stations(bdclim)  # .isel(num_poste=slice(0, 1))
    # bdclim = bdclim.sel(num_poste=[5063003, 5079402, 5181002, 38005402, 38191002, 73013001, 73242003, 73306401])
    sd_edel_station_openloop, viirs_station, s2_station = get_station_location_data(
        bdclim=bdclim, edelweiss_openloop=edelweiss_openloop, viirs=viirs, s2=s2
    )
    sd_edel_station_openloop = sd_edel_station_openloop.data_vars["DSN_T_ISBA"].sel(member=slice(0, None))

    ## Temporal simulation of domain wide means on station sites

    fig = plt.figure(figsize=(14, 9))
    fig.suptitle(f"Edelweiss vs in situ ({bdclim.sizes['num_poste']} locations aggregated) ")
    grid_spec = GridSpec(nrows=2, ncols=1, hspace=0.25, height_ratios=(1, 1))
    ax_sd = fig.add_subplot(grid_spec[0])
    ax_fsc = fig.add_subplot(grid_spec[1])
    all_dates = sd_edel_station_openloop.coords["time"].values
    open_loop = sd_edel_station_openloop.groupby("member").mean(dim="num_poste")

    plot_data_assimilation_base_simulation(
        open_loop,
        x_axis_data=all_dates,
        ax=ax_sd,
        reference=bdclim.mean(dim="num_poste"),
    )
    open_loop = dickinson(sd_edel_station_openloop).groupby("member").mean(dim="num_poste")
    reference = dickinson(bdclim).mean(dim="num_poste")
    plot_data_assimilation_base_simulation(open_loop, x_axis_data=all_dates, ax=ax_fsc, reference=reference)

    for idx, xpid in enumerate(xpids):
        edelweiss_analysis = get_data_assimilation_simulation(actual_xpid=xpid, period=period)
        edelweiss_analysis = edelweiss_analysis.data_vars["DSN_T_ISBA"].sel(member=slice(0, None))
        sd_edel_station_analysis = extract_bdclim_locations_on_spatial_dataset(bdclim, spatial_dataset=edelweiss_analysis)
        sd_assim = sd_edel_station_analysis.groupby("member").mean(dim="num_poste")
        plot_ensemble_simulation(sd_assim, ax=ax_sd, x_axis_data=sd_assim.time, color=plot_colors[idx], label=plot_labels[idx])
        fsc_assim = dickinson(sd_edel_station_analysis).groupby("member").mean(dim="num_poste")
        plot_ensemble_simulation(
            fsc_assim, ax=ax_fsc, x_axis_data=fsc_assim.time, color=plot_colors[idx], label=plot_labels[idx]
        )

    ## Satellite FSC data
    dates = viirs_station.coords["time"].values
    obs = viirs_station.data_vars["snow_cover_fraction"].mean(dim="num_poste").values.flatten()
    ax_fsc.plot(dates, obs, color=COLORS["viirs"], linewidth=0, marker=".", markersize=3, label=LABELS["viirs"])
    dates = s2_station.coords["time"].values
    obs = s2_station.data_vars["snow_cover_fraction"].mean(dim="num_poste").values.flatten()
    ax_fsc.plot(dates, obs, color=COLORS["s2"], linewidth=0, marker="s", markersize=6, mfc="none", label=LABELS["s2"])

    ## Finish plot
    ax_sd.set_title("Snow depth")
    ax_sd.set_ylabel("SD [m]")
    ax_fsc.set_title("Fractional snow cover")
    ax_fsc.set_ylabel("FSC [-]")

    [ax.legend() for ax in fig.get_axes()]
    [ax.grid() for ax in fig.get_axes()]
    plt.show()
