from glob import glob

import matplotlib.pyplot as plt
import numpy as np
import xarray as xr
from matplotlib.gridspec import GridSpec
from pyproj import Transformer

from edelassim.bdclim import extract_bdclim_locations_on_spatial_dataset, filter_discontinuous_stations, find_station_locations
from edelassim.evaluations import compute_scores_grouped_data
from edelassim.filter import effective_sample_size_xarray
from edelassim.observation_operators import dickinson
from edelassim.observations import EdelweissGrandesRoussesGrid, find_clear_dates_s2, find_clear_dates_viirs
from edelassim.postprocess_surfex.soda import (
    alpha_multiple_assimilation,
    alpha_to_grid,
    duplicated_particles_multiple_assimilation,
    duplicated_particles_to_grid,
)
from edelassim.visualization.scatter import plot_ensemble_envelop
from edelassim.visualization.static_files import COLORS, LABELS, get_spatial_datasets, mask, working_folder

if __name__ == "__main__":
    xpid = "assim_viirs_cloudcover07_wy2122_d93_50cm"
    period = slice("2021-11", "2022-05")
    edelweiss_openloop, edelweiss_analysis, viirs, s2, _, bdclim = get_spatial_datasets(xpid)

    bdclim = filter_discontinuous_stations(bdclim.sel(time=period))

    # Filter masked pixels stations
    mask_stations = extract_bdclim_locations_on_spatial_dataset(bdclim, spatial_dataset=mask)
    mask_stations = mask_stations.assign_coords({"x": bdclim.coords["x"], "y": bdclim.coords["y"]})
    bdclim = bdclim.where(1 - mask_stations, drop=True)

    part_files = glob(f"{working_folder}/simulations/edelweiss/reanalysis/{xpid}/soda/*PART*")
    duplicated_particles = duplicated_particles_multiple_assimilation(part_files=part_files)
    assim_dates = duplicated_particles.coords["time"]

    #############
    sd_edel_station_openloop = extract_bdclim_locations_on_spatial_dataset(bdclim, spatial_dataset=edelweiss_openloop).sel(
        time=period
    )
    sd_edel_station_analysis = extract_bdclim_locations_on_spatial_dataset(bdclim, spatial_dataset=edelweiss_analysis).sel(
        time=period
    )
    # good_dates_viirs = find_clear_dates_viirs(snow_cover_viirs=viirs.data_vars["snow_cover_fraction"])
    viirs_station = (extract_bdclim_locations_on_spatial_dataset(bdclim, spatial_dataset=viirs)).sel(time=period)
    good_dates_viirs_mask = np.isnan(viirs_station.data_vars["snow_cover_fraction"]).sum(dim="num_poste") < 3
    viirs_station = viirs_station.where(good_dates_viirs_mask)

    x_station, y_station = find_station_locations(bdclim_dataset=bdclim)
    trans = Transformer.from_crs(2154, 32631)
    x_station_s2, y_station_s2 = trans.transform(x_station, y_station)
    # Xarray advanced indexing (according to Mistral)
    x_poste_s2_da = xr.DataArray(x_station_s2, dims="num_poste", coords={"num_poste": bdclim.num_poste})
    y_poste_s2_da = xr.DataArray(y_station_s2, dims="num_poste", coords={"num_poste": bdclim.num_poste})
    s2_station = s2.sel(time=period).sel(x=x_poste_s2_da, y=y_poste_s2_da, method="nearest")
    good_dates_s2_mask = np.isnan(s2_station.data_vars["snow_cover_fraction"]).sum(dim="num_poste") < 3
    s2_station = s2_station.where(good_dates_s2_mask)

    ## Bassin wide total
    grouped_dataset = xr.Dataset(
        {
            "open_loop": sd_edel_station_openloop.data_vars["DSN_T_ISBA"].sel(member=slice(0, None)),
            "analysis": sd_edel_station_analysis.data_vars["DSN_T_ISBA"].sel(member=slice(0, None)),
            "reference": bdclim.data_vars["neigetotx"],
        },
        coords=sd_edel_station_openloop.coords,
    )
    fig = plt.figure(figsize=(14, 9))
    fig.suptitle(f"Edelweiss vs in situ ({bdclim.sizes['num_poste']} locations aggregated)")
    grid_spec = GridSpec(nrows=4, ncols=1, hspace=0.25, height_ratios=(1, 1, 0.75, 0.75))
    ax_sd = fig.add_subplot(grid_spec[0])
    # plt.rcParams["font.size"] = 16

    # Snow depth
    grouped_dataset_binned = grouped_dataset.groupby("member").mean(dim="num_poste")
    plot_ensemble_envelop(grouped_dataset=grouped_dataset_binned, x_axis_data=grouped_dataset.coords["time"].values, ax=ax_sd)

    ax_sd.legend()
    ax_sd.set_title("Snow depth")
    ax_sd.set_ylabel("SD [m]")
    ax_sd.vlines(
        assim_dates,
        ymin=ax_sd.get_ybound()[0],
        ymax=ax_sd.get_ybound()[1],
        color="gray",
        linestyle="dashed",
        linewidth=0.5,
        label="assim dates",
    )
    ax_sd.grid()
    # fig, ax = plt.subplots(figsize=(20, 7))
    # plot_ensemble_envelop(grouped_dataset=grouped_dataset_binned, x_axis_data=grouped_dataset.coords["time"].values, ax=ax)
    # ax.vlines(
    #     good_dates_viirs[19:],
    #     0,
    #     1.4,
    #     linestyle="dashed",
    #     color="orange",
    #     linewidth=0.5,
    #     label="Assimilation dates",
    # )
    # ax.set_ylim(0, 1.4)
    # ax.grid()
    # ax.legend()
    # fig.patch.set_alpha(0.0)
    # fig.savefig("/home/imperatoren/work/conferences/jc2/illustrations/res_fig_assim.png", format="png", dpi=400)

    # Fractional snow cover
    grouped_dataset_binned = dickinson(grouped_dataset).groupby("member").mean(dim="num_poste")
    ax_fsc = fig.add_subplot(grid_spec[1])
    plot_ensemble_envelop(grouped_dataset=grouped_dataset_binned, x_axis_data=grouped_dataset.coords["time"].values, ax=ax_fsc)
    ax_fsc.plot(
        viirs_station.coords["time"].values,
        viirs_station.data_vars["snow_cover_fraction"].mean(dim="num_poste").values.flatten(),
        color=COLORS["viirs"],
        linewidth=0,
        marker=".",
        markersize=3,
        label=LABELS["viirs"],
    )
    ax_fsc.plot(
        s2_station.coords["time"].values,
        s2_station.data_vars["snow_cover_fraction"].mean(dim="num_poste").values.flatten(),
        color=COLORS["s2"],
        linewidth=0,
        marker="s",
        markersize=6,
        mfc="none",
        label=LABELS["s2"],
    )

    ax_fsc.grid()

    ax_fsc.set_title("Fractional snow cover")
    ax_fsc.set_ylabel("FSC [-]")
    ax_fsc.vlines(
        assim_dates,
        ymin=ax_fsc.get_ybound()[0],
        ymax=ax_fsc.get_ybound()[1],
        color="gray",
        linestyle="dashed",
        linewidth=0.5,
        label="assim dates",
    )
    ax_fsc.legend()
    # fig.patch.set_alpha(0.0)
    # fig.savefig("/home/imperatoren/work/conferences/jc2/illustrations/temporal.png", format="png", dpi=600)
    ## Bassin wide per altitude band total
    # grouped_dataset = xr.Dataset(
    #     {
    #         "open_loop": sd_edel_station_openloop.data_vars["DSN_T_ISBA"].sel(member=slice(0, None)),
    #         "analysis": sd_edel_station_analysis.data_vars["DSN_T_ISBA"].sel(member=slice(0, None)),
    #         "bdclim": bdclim.data_vars["neigetotx"],
    #     },
    #     coords=sd_edel_station_openloop.coords,
    # )
    # for alt_min in range(1000, 2501, 500):
    #     fig, axs = plt.subplots(2, 1, figsize=(12, 8))
    #     fig.suptitle(f"Edelweiss vs in situ Basin wide {alt_min} - {alt_min + 500}")
    #     num_poste_altitude = (
    #         grouped_dataset.where(
    #             (grouped_dataset.coords["ZS"] > alt_min) * (grouped_dataset.coords["ZS"] <= alt_min + 500), drop=True
    #         )
    #         .coords["num_poste"]
    #         .values
    #     )
    # And then copy paste the bassin wide code
    # CRPSS

    # Scores

    grouped_dataset = xr.Dataset(
        {
            "open_loop": sd_edel_station_openloop.data_vars["DSN_T_ISBA"].sel(member=slice(0, None)),
            "analysis": sd_edel_station_analysis.data_vars["DSN_T_ISBA"].sel(member=slice(0, None)),
            "reference": bdclim.data_vars["neigetotx"],
        },
        coords=sd_edel_station_openloop.coords,
    )

    # Scores on SD
    # scores = grouped_dataset.groupby(("num_poste", "time")).map(compute_scores_grouped_data).mean(dim="num_poste")
    scores = (
        xr.open_dataset("../output_folder/scores_2021_2022.nc")
        .assign_coords(dict(variable=["snow_depth", "fractional_snow_cover"]))
        .mean(dim="num_poste")
    )
    crpss = 1 - scores.data_vars["crps"].sel(type="analysis") / scores.data_vars["crps"].sel(type="open_loop")
    maes = 1 - scores.data_vars["mae"].sel(type="analysis") / scores.data_vars["mae"].sel(type="open_loop")
    sigmas = 1 - scores.data_vars["sigma"].sel(type="analysis") / scores.data_vars["sigma"].sel(type="open_loop")

    ax_scores_sd = fig.add_subplot(grid_spec[2])
    ax_scores_sd.plot(scores.coords["time"], crpss.sel(variable="snow_depth").values, linewidth=0, marker="^", label="CRPSS")
    ax_scores_sd.plot(
        scores.coords["time"], maes.sel(variable="snow_depth").values, linewidth=0, marker="o", mfc="none", label="MAES"
    )
    ax_scores_sd.plot(scores.coords["time"], sigmas.sel(variable="snow_depth").values, linewidth=0, marker="*", label="sigmaS")
    ax_scores_sd.grid()
    ax_scores_sd.set_title(f"SD error scores - mean on {bdclim.sizes['num_poste']} sites")
    ax_scores_sd.set_ylim(-1, 1)
    ax_scores_sd.vlines(
        assim_dates,
        ymin=ax_scores_sd.get_ybound()[0],
        ymax=ax_scores_sd.get_ybound()[1],
        color="gray",
        linestyle="dashed",
        linewidth=0.5,
        label="assim dates",
    )
    ax_scores_sd.hlines(
        0, scores.coords["time"][0], scores.coords["time"][-1], linewidth=1.5, color="black", linestyle="dashed"
    )
    ax_scores_sd.legend()

    # Scores on FSC
    ax_scores_fsc = fig.add_subplot(grid_spec[3])
    # scores = dickinson(grouped_dataset).groupby(("num_poste", "time")).map(compute_scores_grouped_data).mean(dim="num_poste")

    ax_scores_fsc.plot(
        scores.coords["time"], crpss.sel(variable="fractional_snow_cover").values, linewidth=0, marker="^", label="CRPSS"
    )
    ax_scores_fsc.plot(
        scores.coords["time"],
        maes.sel(variable="fractional_snow_cover").values,
        linewidth=0,
        marker="o",
        mfc="none",
        label="MAES",
    )
    ax_scores_fsc.plot(
        scores.coords["time"], sigmas.sel(variable="fractional_snow_cover").values, linewidth=0, marker="*", label="sigmaS"
    )
    ax_scores_fsc.grid()
    ax_scores_fsc.set_title(f"FSC error scores - mean on {bdclim.sizes['num_poste']} sites")
    ax_scores_fsc.set_ylim(-1, 1)
    ax_scores_fsc.vlines(
        assim_dates,
        ymin=ax_scores_sd.get_ybound()[0],
        ymax=ax_scores_sd.get_ybound()[1],
        color="gray",
        linestyle="dashed",
        linewidth=0.5,
        label="assim dates",
    )
    ax_scores_fsc.hlines(
        0, scores.coords["time"][0], scores.coords["time"][-1], linewidth=1.5, color="black", linestyle="dashed"
    )
    ax_scores_fsc.legend()

    # fig, (ax_mae, ax_sigma) = plt.subplots(2, 1, figsize=(13, 6))
    # ax_mae.plot(mae_open_loop.time, mae_open_loop, linewidth=0, marker="o", color=COLORS["edel_ol"], label=LABELS["edel_ol"])
    # ax_mae.plot(mae_analysis.time, mae_analysis, linewidth=0, marker="o", color=COLORS["edel_an"], label=LABELS["edel_an"])
    # ax_mae.grid()
    # ax_mae.set_title(f"MAE - mean on {bdclim.sizes['num_poste']} stations")
    # ax_mae.legend()

    # residuals_ensemble_open_loop = sd_edel_station_openloop.data_vars["DSN_T_ISBA"] - sd_edel_station_openloop.data_vars[
    #     "DSN_T_ISBA"
    # ].mean(dim="member")
    # sigma_ensemble_open_loop = np.sqrt(((residuals_ensemble_open_loop) ** 2).mean(dim="member")).mean(dim="num_poste")
    # residuals_ensemble_analysis = sd_edel_station_analysis.data_vars["DSN_T_ISBA"] - sd_edel_station_analysis.data_vars[
    #     "DSN_T_ISBA"
    # ].mean(dim="member")
    # sigma_ensemble_analysis = np.sqrt(((residuals_ensemble_analysis) ** 2).mean(dim="member")).mean(dim="num_poste")

    # ax_sigma.plot(
    #     sigma_ensemble_open_loop.time,
    #     sigma_ensemble_open_loop,
    #     marker="o",
    #     linewidth=1,
    #     linestyle="dashed",
    #     color=COLORS["edel_ol"],
    #     label=LABELS["edel_ol"],
    # )
    # ax_sigma.plot(
    #     sigma_ensemble_analysis.time,
    #     sigma_ensemble_analysis,
    #     linewidth=1,
    #     linestyle="dashed",
    #     marker="o",
    #     color=COLORS["edel_an"],
    #     label=LABELS["edel_an"],
    # )
    # ax_sigma.grid()
    # ax_sigma.set_title(f"Dispersion - mean on {bdclim.sizes['num_poste']} stations")
    # ax_sigma.legend()

    ## Ensemble monitoring
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
