from glob import glob

import matplotlib.pyplot as plt
import numpy as np
import xarray as xr
from matplotlib.gridspec import GridSpec
from pyproj import Transformer

from edelassim.bdclim import extract_bdclim_locations_on_spatial_dataset, filter_discontinuous_stations, find_station_locations
from edelassim.evaluations import compute_scores
from edelassim.filter import effective_sample_size_xarray
from edelassim.observation_operators import dickinson
from edelassim.observations import find_clear_dates_s2, find_clear_dates_viirs
from edelassim.postprocess_surfex.soda import (
    alpha_multiple_assimilation,
    alpha_to_grid,
    duplicated_particles_multiple_assimilation,
    duplicated_particles_to_grid,
)
from edelassim.visualization.scatter import plot_data_assimilation_base_simulation, plot_ensemble_simulation
from edelassim.visualization.static_files import (
    COLORS,
    LABELS,
    get_data_assimilation_simulation,
    get_duplicated_particles_assim_dates,
    get_spatial_datasets,
    get_station_location_data,
    list_satellite_clear_dates,
    mask,
    working_folder,
)
from edelassim.visualization.time_series import add_assim_dates

if __name__ == "__main__":
    ### Inputs and preprocessing
    xpids = ["assim_viirs_cloudcover055_wy2122_sigma_obs_var", "assim_viirs_cloudcover055_wy2122_python_pf"]
    colors = ["magenta", "darkgreen"]
    labels = ["sigma var", "sigma const"]
    period = slice("2021-11", "2022-06")
    edelweiss_openloop, viirs, s2, _, bdclim = get_spatial_datasets(period=period)
    good_dates_viirs, good_dates_s2 = list_satellite_clear_dates(period=period)
    viirs = viirs.sel(time=good_dates_viirs)
    s2 = s2.sel(time=good_dates_s2)
    bdclim = filter_discontinuous_stations(bdclim)
    sd_edel_station_openloop, viirs_station, s2_station = get_station_location_data(
        bdclim=bdclim, edelweiss_openloop=edelweiss_openloop, viirs=viirs, s2=s2
    )
    sd_edel_station_openloop = sd_edel_station_openloop.data_vars["DSN_T_ISBA"].sel(member=slice(0, None))
    duplicated_particles, assim_dates = get_duplicated_particles_assim_dates(acutal_xpid=xpids[0])

    ## Temporal simulation of domain wide means on station sites

    # fig = plt.figure(figsize=(14, 9))
    # fig.suptitle(f"Edelweiss vs in situ ({bdclim.sizes['num_poste']} locations aggregated) ")
    # grid_spec = GridSpec(nrows=2, ncols=1, hspace=0.25, height_ratios=(1, 1))
    # ax_sd = fig.add_subplot(grid_spec[0])
    # ax_fsc = fig.add_subplot(grid_spec[1])
    # all_dates = sd_edel_station_openloop.coords["time"].values
    # ol = sd_edel_station_openloop.groupby("member").mean(dim="num_poste")
    # plot_data_assimilation_base_simulation(
    #     open_loop=ol,
    #     x_axis_data=all_dates,
    #     ax=ax_sd,
    #     reference=bdclim.mean(dim="num_poste"),
    # )
    # ol = dickinson(sd_edel_station_openloop).groupby("member").mean(dim="num_poste")
    # plot_data_assimilation_base_simulation(
    #     open_loop=ol,
    #     x_axis_data=all_dates,
    #     ax=ax_fsc,
    #     reference=dickinson(bdclim).mean(dim="num_poste"),
    # )

    # for idx, xpid in enumerate(xpids):
    #     edelweiss_analysis = get_data_assimilation_simulation(actual_xpid=xpid, period=period)
    #     edelweiss_analysis = edelweiss_analysis.data_vars["DSN_T_ISBA"].sel(member=slice(0, None))
    #     sd_edel_station_analysis = extract_bdclim_locations_on_spatial_dataset(bdclim, spatial_dataset=edelweiss_analysis)
    #     sd_assim = sd_edel_station_analysis.groupby("member").mean(dim="num_poste")
    #     plot_ensemble_simulation(
    #         simulation_data=sd_assim,
    #         ax=ax_sd,
    #         x_axis_data=sd_assim.time,
    #         color=colors[idx],
    #         label=labels[idx],
    #     )
    #     fsc_assim = dickinson(sd_edel_station_analysis).groupby("member").mean(dim="num_poste")
    #     plot_ensemble_simulation(
    #         simulation_data=fsc_assim,
    #         ax=ax_fsc,
    #         x_axis_data=fsc_assim.time,
    #         color=colors[idx],
    #         label=labels[idx],
    #     )

    # # Satellite FSC data
    # dates = viirs_station.coords["time"].values
    # obs = viirs_station.data_vars["snow_cover_fraction"].mean(dim="num_poste").values.flatten()
    # ax_fsc.plot(dates, obs, color=COLORS["viirs"], linewidth=0, marker=".", markersize=3, label=LABELS["viirs"])
    # dates = s2_station.coords["time"].values
    # obs = s2_station.data_vars["snow_cover_fraction"].mean(dim="num_poste").values.flatten()
    # ax_fsc.plot(dates, obs, color=COLORS["s2"], linewidth=0, marker="s", markersize=6, mfc="none", label=LABELS["s2"])

    # # Finish plot
    # ax_sd.legend()
    # ax_sd.set_title("Snow depth")
    # ax_sd.set_ylabel("SD [m]")

    # ax_sd.grid()

    # ax_fsc.grid()
    # ax_fsc.set_title("Fractional snow cover")
    # ax_fsc.set_ylabel("FSC [-]")
    # add_assim_dates(assim_dates=assim_dates, ax=ax_fsc)
    # ax_fsc.legend()

    # Scores

    grouped_dataset = xr.Dataset(
        {
            "eval": sd_edel_station_openloop,
            "reference": bdclim,
        },
        coords=sd_edel_station_openloop.coords,
    )

    # Scores on SD
    scores_sd = grouped_dataset.groupby(("num_poste", "time")).map(compute_scores).mean(dim="num_poste")
    print(scores_sd)
    # scores_fsc = (
    #     dickinson(grouped_dataset).groupby(("num_poste", "time")).map(compute_scores_grouped_data).mean(dim="num_poste")
    # )
    # scores = xr.concat([scores_sd, scores_fsc], dim="variable")
    # scores = scores.assign_coords(variable=["snow_depth", "fractional_snow_cover"])
    # # scores = (
    # #     xr.open_dataset("../output_folder/scores_2021_2022.nc")
    # #     .assign_coords(dict(variable=["snow_depth", "fractional_snow_cover"]))
    # #     .mean(dim="num_poste")
    # # )
    # crpss = 1 - scores.data_vars["crps"].sel(type="analysis") / scores.data_vars["crps"].sel(type="open_loop")
    # maes = 1 - scores.data_vars["mae"].sel(type="analysis") / scores.data_vars["mae"].sel(type="open_loop")
    # sigmas = 1 - scores.data_vars["sigma"].sel(type="analysis") / scores.data_vars["sigma"].sel(type="open_loop")

    # ax_scores_sd = fig.add_subplot(grid_spec[2])
    # ax_scores_sd.plot(scores.coords["time"], crpss.sel(variable="snow_depth").values, linewidth=0, marker="^", label="CRPSS")
    # ax_scores_sd.plot(
    #     scores.coords["time"], maes.sel(variable="snow_depth").values, linewidth=0, marker="o", mfc="none", label="MAES"
    # )
    # ax_scores_sd.plot(scores.coords["time"], sigmas.sel(variable="snow_depth").values, linewidth=0, marker="*", label="sigmaS")
    # ax_scores_sd.grid()
    # ax_scores_sd.set_title(f"SD error scores - mean on {bdclim.sizes['num_poste']} sites")
    # ax_scores_sd.set_ylim(-1, 1)
    # ax_scores_sd.vlines(
    #     assim_dates,
    #     ymin=ax_scores_sd.get_ybound()[0],
    #     ymax=ax_scores_sd.get_ybound()[1],
    #     color="gray",
    #     linestyle="dashed",
    #     linewidth=0.5,
    #     label="assim dates",
    # )
    # ax_scores_sd.hlines(
    #     0, scores.coords["time"][0], scores.coords["time"][-1], linewidth=1.5, color="black", linestyle="dashed"
    # )
    # ax_scores_sd.legend()

    # # Scores on FSC
    # ax_scores_fsc = fig.add_subplot(grid_spec[3])
    # # scores = dickinson(grouped_dataset).groupby(("num_poste", "time")).map(compute_scores_grouped_data).mean(dim="num_poste")

    # ax_scores_fsc.plot(
    #     scores.coords["time"], crpss.sel(variable="fractional_snow_cover").values, linewidth=0, marker="^", label="CRPSS"
    # )
    # ax_scores_fsc.plot(
    #     scores.coords["time"],
    #     maes.sel(variable="fractional_snow_cover").values,
    #     linewidth=0,
    #     marker="o",
    #     mfc="none",
    #     label="MAES",
    # )
    # ax_scores_fsc.plot(
    #     scores.coords["time"], sigmas.sel(variable="fractional_snow_cover").values, linewidth=0, marker="*", label="sigmaS"
    # )
    # ax_scores_fsc.grid()
    # ax_scores_fsc.set_title(f"FSC error scores - mean on {bdclim.sizes['num_poste']} sites")
    # ax_scores_fsc.set_ylim(-1, 1)
    # ax_scores_fsc.vlines(
    #     assim_dates,
    #     ymin=ax_scores_sd.get_ybound()[0],
    #     ymax=ax_scores_sd.get_ybound()[1],
    #     color="gray",
    #     linestyle="dashed",
    #     linewidth=0.5,
    #     label="assim dates",
    # )
    # ax_scores_fsc.hlines(
    #     0, scores.coords["time"][0], scores.coords["time"][-1], linewidth=1.5, color="black", linestyle="dashed"
    # )
    # ax_scores_fsc.legend()

    # ## Ensemble monitoring
    # duplicated_particles = duplicated_particles_to_grid(duplicated_particles, grid=EdelweissGrandesRoussesGrid())

    # particle_occurrence = xr.DataArray(
    #     0,
    #     dims=("x", "y", "time", "member"),
    #     coords=duplicated_particles.coords,
    # )

    # for mb in duplicated_particles.coords["member"]:
    #     n = (duplicated_particles == mb + 1).sum(dim=("member"))
    #     particle_occurrence[dict(member=mb)] = n

    # # Normalized Effective sample size mean
    # eff_sample_size = effective_sample_size_xarray(weights=(particle_occurrence / particle_occurrence.sizes["member"]))

    # cloud_mask = np.isnan(viirs)
    # fsc_edel_openloop = dickinson(edelweiss_openloop).sel(time=period)
    # constant_ensemble_mask = (fsc_edel_openloop == 0).all(dim="member") + (fsc_edel_openloop == 1).all(dim="member")
    # mask = cloud_mask.data_vars["snow_cover_fraction"] + constant_ensemble_mask.data_vars["DSN_T_ISBA"]
    # mask = mask.sel(time=duplicated_particles.time, method="nearest")
    # mask = mask.assign_coords({"time": duplicated_particles.time})

    # fig, ax = plt.subplots(figsize=(15, 3))
    # global_ess = eff_sample_size.where(1 - mask).mean(dim=("x", "y"))
    # ax.plot(
    #     global_ess.time, global_ess.values / duplicated_particles.sizes["member"], linewidth=0, marker="*", label="ESS/N_e"
    # )

    # # Alpha mean
    # alpha_files = glob(f"{working_folder}/simulations/edelweiss/reanalysis/{xpid}/soda/*ALPHA*")
    # alpha_coefficients = alpha_to_grid(
    #     alpha_multiple_assimilation(alpha_files=alpha_files), grid=EdelweissGrandesRoussesGrid()
    # )

    # alpha_coeff_mean = alpha_coefficients.where(1 - mask).mean(dim=("x", "y"))
    # ax.plot(alpha_coeff_mean.time, alpha_coeff_mean, linewidth=0, marker="s", label="alpha")
    # ax.set_xticks(global_ess.time, minor=True)
    # ax.grid(which="minor")
    # ax.tick_params(axis="x", which="minor", bottom=False)
    # ax.set_title("Ensemble monitoring - mean on grid points")
    # ax.legend()

    # sigmas = 1 - sigma_ensemble_analysis / sigma_ensemble_open_loop
    # ax.plot(sigmas.time, sigmas, linewidth=0, marker="s", label="sigmas")

    plt.show()
