from glob import glob

import matplotlib.pyplot as plt
import numpy as np
import xarray as xr
from matplotlib.axes import Axes
from pyproj import Transformer
from snowtools.scores.ensemble import EnsembleScores
from xarray.groupers import BinGrouper, UniqueGrouper

from edelassim.bdclim import extract_bdclim_locations_on_spatial_dataset, filter_discontinuous_stations, find_station_locations
from edelassim.evaluations import compute_rmse
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
from edelassim.visualization.static_files import COLORS, LABELS, get_spatial_datasets, topography_data_folder, working_folder

if __name__ == "__main__":
    xpid = "assim_viirs_cloudcover07_wy2122_d93_50cm"
    period = slice("2021-11", "2022-06")
    edelweiss_openloop, edelweiss_analysis, viirs, s2, pleiades, bdclim = get_spatial_datasets(xpid)

    bdclim = filter_discontinuous_stations(bdclim.sel(time=period))
    sd_edel_station_openloop = extract_bdclim_locations_on_spatial_dataset(bdclim, spatial_dataset=edelweiss_openloop).sel(
        time=period
    )
    sd_edel_station_analysis = extract_bdclim_locations_on_spatial_dataset(bdclim, spatial_dataset=edelweiss_analysis).sel(
        time=period
    )
    good_dates_viirs = find_clear_dates_viirs(snow_cover_viirs=viirs.data_vars["snow_cover_fraction"])
    viirs_station = (
        (extract_bdclim_locations_on_spatial_dataset(bdclim, spatial_dataset=viirs))
        .sel(time=good_dates_viirs)
        .sel(time=period)
    )
    good_dates_s2 = find_clear_dates_s2(snow_cover_s2=s2.data_vars["snow_cover_fraction"])
    x_station, y_station = find_station_locations(bdclim_dataset=bdclim)
    trans = Transformer.from_crs(2154, 32631)
    x_station_s2, y_station_s2 = trans.transform(x_station, y_station)
    # Xarray advanced indexing (according to Mistral)
    x_poste_s2_da = xr.DataArray(x_station_s2, dims="num_poste", coords={"num_poste": bdclim.num_poste})
    y_poste_s2_da = xr.DataArray(y_station_s2, dims="num_poste", coords={"num_poste": bdclim.num_poste})
    s2_station = s2.sel(time=good_dates_s2).sel(time=period).sel(x=x_poste_s2_da, y=y_poste_s2_da, method="nearest")

    ## Bassin wide total
    grouped_dataset = xr.Dataset(
        {
            "open_loop": sd_edel_station_openloop.data_vars["DSN_T_ISBA"].sel(member=slice(0, None)),
            "analysis": sd_edel_station_analysis.data_vars["DSN_T_ISBA"].sel(member=slice(0, None)),
            "bdclim": bdclim.data_vars["neigetotx"],
        },
        coords=sd_edel_station_openloop.coords,
    )
    fig, axs = plt.subplots(2, 1, figsize=(14, 7))
    # plt.rcParams["font.size"] = 16
    fig.suptitle(f"Edelweiss vs in situ ({bdclim.sizes['num_poste']} locations aggregated)")
    grouped_dataset_binned = grouped_dataset.groupby("member").mean(dim="num_poste")
    plot_ensemble_envelop(grouped_dataset=grouped_dataset_binned, x_axis_data=grouped_dataset.coords["time"].values, ax=axs[0])
    sd_pleiades_station = extract_bdclim_locations_on_spatial_dataset(bdclim, spatial_dataset=pleiades.sel(time=period))
    axs[0].plot(
        sd_pleiades_station.time,
        sd_pleiades_station.data_vars["snow_depth"].mean(dim=("num_poste")).values,
        marker="*",
        linewidth=0,
        markersize=15,
        color="gold",
        label="Pleiades",
    )
    axs[0].grid()
    # axs[0].legend()
    axs[0].set_ylabel("SD [m]")

    grouped_dataset_binned = dickinson(grouped_dataset).groupby("member").mean(dim="num_poste")
    plot_ensemble_envelop(grouped_dataset=grouped_dataset_binned, x_axis_data=grouped_dataset.coords["time"].values, ax=axs[1])
    axs[1].plot(
        viirs_station.coords["time"].values,
        viirs_station.data_vars["snow_cover_fraction"].mean(dim="num_poste").values.flatten(),
        color=COLORS["viirs"],
        linewidth=0,
        marker=".",
        markersize=3,
        label=LABELS["viirs"],
    )
    axs[1].plot(
        s2_station.coords["time"].values,
        s2_station.data_vars["snow_cover_fraction"].mean(dim="num_poste").values.flatten(),
        color=COLORS["s2"],
        linewidth=0,
        marker="s",
        markersize=6,
        mfc="none",
        label=LABELS["s2"],
    )

    axs[1].grid()
    # axs[1].legend()
    axs[1].set_ylabel("FSC [-]")
    # fig.patch.set_alpha(0.0)
    # fig.savefig("/home/imperatoren/work/conferences/jc2/illustrations/temporal.png", format="png", dpi=600)
    ## Bassin wide per altitude band total
    grouped_dataset = xr.Dataset(
        {
            "open_loop": sd_edel_station_openloop.data_vars["DSN_T_ISBA"].sel(member=slice(0, None)),
            "analysis": sd_edel_station_analysis.data_vars["DSN_T_ISBA"].sel(member=slice(0, None)),
            "bdclim": bdclim.data_vars["neigetotx"],
        },
        coords=sd_edel_station_openloop.coords,
    )
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

    ## Ensemble monitoring
    part_files = glob(f"{working_folder}/simulations/edelweiss/reanalysis/{xpid}/soda/*PART*")
    duplicated_particles = duplicated_particles_multiple_assimilation(part_files=part_files)
    duplicated_particles = duplicated_particles_to_grid(duplicated_particles, grid=EdelweissGrandesRoussesGrid())

    particle_occurrence = xr.DataArray(
        0,
        dims=("x", "y", "time", "member"),
        coords=duplicated_particles.coords,
    )

    for mb in duplicated_particles.coords["member"]:
        n = (duplicated_particles == mb + 1).sum(dim=("member"))
        particle_occurrence[dict(member=mb)] = n

    # Normalized Effective sample size mean
    eff_sample_size = effective_sample_size_xarray(weights=(particle_occurrence / particle_occurrence.sizes["member"]))

    cloud_mask = np.isnan(viirs)
    fsc_edel_openloop = dickinson(edelweiss_openloop).sel(time=period)
    constant_ensemble_mask = (fsc_edel_openloop == 0).all(dim="member") + (fsc_edel_openloop == 1).all(dim="member")
    mask = cloud_mask.data_vars["snow_cover_fraction"] + constant_ensemble_mask.data_vars["DSN_T_ISBA"]
    mask = mask.sel(time=duplicated_particles.time, method="nearest")
    mask = mask.assign_coords({"time": duplicated_particles.time})

    fig, ax = plt.subplots(figsize=(18, 3))
    global_ess = eff_sample_size.where(1 - mask).mean(dim=("x", "y"))
    ax.plot(
        global_ess.time, global_ess.values / duplicated_particles.sizes["member"], linewidth=0, marker="*", label="ESS/N_e"
    )

    # Alpha mean
    alpha_files = glob(f"{working_folder}/simulations/edelweiss/reanalysis/{xpid}/soda/*ALPHA*")
    alpha_coefficients = alpha_to_grid(
        alpha_multiple_assimilation(alpha_files=alpha_files), grid=EdelweissGrandesRoussesGrid()
    )

    alpha_coeff_mean = alpha_coefficients.where(1 - mask).mean(dim=("x", "y"))
    ax.plot(alpha_coeff_mean.time, alpha_coeff_mean, linewidth=0, marker="s", label="alpha")
    ax.set_xticks(global_ess.time, minor=True)
    ax.grid(which="minor")
    ax.tick_params(axis="x", which="minor", bottom=False)
    ax.set_title("Ensemble monitoring - mean on x, y points")
    ax.legend()

    # CRPSS

    def compute_crpss(data: xr.Dataset) -> xr.DataArray:
        crps_open_loop = EnsembleScores(
            timeObs=None, timeSim=None, obs=data.data_vars["bdclim"].values, ensemble=data.data_vars["open_loop"].values
        ).CRPS()
        crps_analysis = EnsembleScores(
            timeObs=None, timeSim=None, obs=data.data_vars["bdclim"].values, ensemble=data.data_vars["analysis"].values
        ).CRPS()
        crpss = 1 - (crps_analysis / crps_open_loop)
        return xr.DataArray(crpss)

    grouped_dataset = xr.Dataset(
        {
            "open_loop": sd_edel_station_openloop.data_vars["DSN_T_ISBA"].sel(member=slice(0, None)),
            "analysis": sd_edel_station_analysis.data_vars["DSN_T_ISBA"].sel(member=slice(0, None)),
            "bdclim": bdclim.data_vars["neigetotx"],
        },
        coords=sd_edel_station_openloop.coords,
    )
    fig, ax = plt.subplots(figsize=(15, 2))
    crpss = grouped_dataset.groupby(("num_poste", "time")).map(compute_crpss)
    crpss_mean = crpss.mean(dim=("num_poste"))
    ax.plot(crpss_mean.time, crpss_mean.values, linewidth=0, marker="^", label="CRPSS")

    mae_open_loop = (
        np.abs(sd_edel_station_openloop.data_vars["DSN_T_ISBA"].mean(dim="num_poste") - bdclim.data_vars["neigetotx"])
        .sel(member=slice(0, None))
        .mean(dim="member")
        .mean(dim="num_poste")
    )
    mae_analysis = (
        np.abs(sd_edel_station_analysis.data_vars["DSN_T_ISBA"].mean(dim="num_poste") - bdclim.data_vars["neigetotx"])
        .sel(member=slice(0, None))
        .mean(dim="member")
        .mean(dim="num_poste")
    )
    maes = 1 - mae_analysis / mae_open_loop
    ax.plot(maes.time, maes, linewidth=0, marker="o", label="MAES")
    ax.grid()
    ax.set_title(f"Temporal evolution error scores - mean on {bdclim.sizes['num_poste']} stations")
    ax.legend()

    fig, ax = plt.subplots(figsize=(15, 2))
    ax.plot(mae_open_loop.time, mae_open_loop, linewidth=0, marker="o", color=COLORS["edel_ol"], label="open loop")
    ax.plot(mae_analysis.time, mae_analysis, linewidth=0, marker="o", color=COLORS["edel_an"], label="assim")
    ax.grid()
    ax.set_title(f"MAE - mean on {bdclim.sizes['num_poste']} stations")
    ax.legend()

    residuals_ensemble_open_loop = sd_edel_station_openloop.data_vars["DSN_T_ISBA"] - sd_edel_station_openloop.data_vars[
        "DSN_T_ISBA"
    ].mean(dim="member")
    sigma_ensemble_open_loop = np.sqrt(((residuals_ensemble_open_loop) ** 2).mean(dim="member")).mean(dim="num_poste")
    residuals_ensemble_analysis = sd_edel_station_analysis.data_vars["DSN_T_ISBA"] - sd_edel_station_analysis.data_vars[
        "DSN_T_ISBA"
    ].mean(dim="member")
    sigma_ensemble_analysis = np.sqrt(((residuals_ensemble_analysis) ** 2).mean(dim="member")).mean(dim="num_poste")

    fig, ax = plt.subplots(figsize=(15, 2))
    ax.plot(
        sigma_ensemble_open_loop.time,
        sigma_ensemble_open_loop,
        linewidth=0,
        marker="o",
        color=COLORS["edel_ol"],
        label="open loop",
    )
    ax.plot(
        sigma_ensemble_analysis.time, sigma_ensemble_analysis, linewidth=0, marker="o", color=COLORS["edel_an"], label="assim"
    )
    ax.grid()
    ax.set_title(f"Dispersion - mean on {bdclim.sizes['num_poste']} stations")
    ax.legend()

    # sigmas = 1 - sigma_ensemble_analysis / sigma_ensemble_open_loop
    # ax.plot(sigmas.time, sigmas, linewidth=0, marker="s", label="sigmas")

    plt.show()
