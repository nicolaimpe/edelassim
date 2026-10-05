from glob import glob

import matplotlib.pyplot as plt
import numpy as np
import xarray as xr
from matplotlib.axes import Axes
from pyproj import Transformer
from xarray.groupers import BinGrouper, UniqueGrouper

from edelassim.bdclim import extract_bdclim_locations_on_spatial_dataset, find_station_locations
from edelassim.filter import effective_sample_size_xarray
from edelassim.observation_operators import dickinson
from edelassim.observations import EdelweissGrandesRoussesGrid, find_clear_dates_s2, find_clear_dates_viirs
from edelassim.postprocess_surfex.soda import duplicated_particles_multiple_assimilation, duplicated_particles_to_grid
from edelassim.visualization.scatter import plot_ensemble_envelop
from edelassim.visualization.static_files import COLORS, LABELS, get_spatial_datasets, topography_data_folder, working_folder

if __name__ == "__main__":
    xpid = "assim_viirs_cloudcover07_wy2122_d93_50cm"
    period = slice("2021-11", "2022-07")
    edelweiss_openloop, edelweiss_analysis, viirs, s2, pleiades, bdclim = get_spatial_datasets(xpid)

    bdclim = bdclim.sel(time=period)
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
    fig, axs = plt.subplots(2, 1, figsize=(12, 8))
    fig.suptitle("Edelweiss vs in situ Basin wide")
    grouped_dataset_binned = grouped_dataset.groupby("member").mean(dim="num_poste")
    plot_ensemble_envelop(grouped_dataset=grouped_dataset_binned, x_axis_data=grouped_dataset.coords["time"].values, ax=axs[0])
    axs[0].plot(
        grouped_dataset.coords["time"].values,
        grouped_dataset_binned.data_vars["bdclim"].mean(dim="member").values.flatten(),
        color="black",
        linestyle="dashed",
        linewidth=2,
        label="In situ",
    )
    axs[0].grid()
    axs[0].legend()
    axs[0].set_ylabel("SD [m]")

    grouped_dataset_binned = dickinson(grouped_dataset).groupby("member").mean(dim="num_poste")
    plot_ensemble_envelop(grouped_dataset=grouped_dataset_binned, x_axis_data=grouped_dataset.coords["time"].values, ax=axs[1])
    axs[1].plot(
        grouped_dataset.coords["time"].values,
        grouped_dataset_binned.data_vars["bdclim"].mean(dim="member").values.flatten(),
        color="black",
        linestyle="dashed",
        linewidth=2,
        label="In situ",
    )
    axs[1].plot(
        viirs_station.coords["time"].values,
        viirs_station.data_vars["snow_cover_fraction"].mean(dim="num_poste").values.flatten(),
        color=COLORS["viirs"],
        linewidth=0,
        marker=".",
        markersize=10,
        label=LABELS["viirs"],
    )
    axs[1].plot(
        s2_station.coords["time"].values,
        s2_station.data_vars["snow_cover_fraction"].mean(dim="num_poste").values.flatten(),
        color=COLORS["s2"],
        linewidth=0,
        marker="s",
        markersize=10,
        mfc="none",
        label=LABELS["s2"],
    )
    axs[1].grid()
    axs[1].legend()
    axs[1].set_ylabel("FSC [-]")

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

    # print(particle_occurrence)
    eff_sample_size = effective_sample_size_xarray(weights=(particle_occurrence / particle_occurrence.sizes["member"]))
    fig, ax = plt.subplots()
    eff_sample_size.sel(time="2021-11-05").drop_vars("time")[:, :, 0].plot.imshow(ax=ax, x="x")

    cloud_mask = np.isnan(viirs)
    fsc_edel_openloop = dickinson(edelweiss_openloop).sel(time=period)
    constant_ensemble_mask = (fsc_edel_openloop == 0).all(dim="member") + (fsc_edel_openloop == 1).all(dim="member")
    mask = cloud_mask.data_vars["snow_cover_fraction"] + constant_ensemble_mask.data_vars["DSN_T_ISBA"]
    mask = mask.sel(time=duplicated_particles.time, method="nearest")
    mask = mask.assign_coords({"time": duplicated_particles.time})

    fig, ax = plt.subplots(figsize=(18, 5))
    global_ess = eff_sample_size.where(1 - mask).mean(dim=("x", "y"))
    print(global_ess)
    ax.plot(
        global_ess.time, global_ess.values / duplicated_particles.sizes["member"], linewidth=0, marker="*", label="ESS/N_e"
    )
    ax.set_xticks(global_ess.time, minor=True)
    ax.grid(which="minor")
    ax.tick_params(axis="x", which="minor", bottom=False)
    ax.set_title("Ensemble monitoring - Basin wide spatial mean")
    ax.legend()
    # ax.set_xticks(global_ess.time)
    # plt.show()
    plt.show()
    # fig, ax = duplicated_particles.sel(time='2021-11-05')
    # def count_member_occurrences(data: xr.DataArray) -> xr.Dataset:
    #     return (data == np.arange(17)).sum(dim=("n_points"))
    # .groupby("member").count(dim="n_poin

    # print(duplicated_particles)
    # particle_occurrence = duplicated_particles == (duplicated_particles.coords["member"] + 11)

    # # Consider only points where assimilation is possible
    # fsc_edel_openloop = dickinson(edelweiss_openloop).stack(points=("y", "x")).sel(time=period)
    # # fig, ax = plt.subplots()
    # # (
    # #     (dickinson(edelweiss_openloop.data_vars["DSN_T_ISBA"]).sel(time="2022-01-15") == 1).all(dim="member")
    # #     + (dickinson(edelweiss_openloop.data_vars["DSN_T_ISBA"]).sel(time="2022-01-15") == 0).all(dim="member")
    # # ).plot.imshow(ax=ax)
    # # plt.show()
    # # (fsc_edel_openloop == 0).all(dim="member")
    # fsc_edel_openloop = fsc_edel_openloop.assign_coords(n_points=("points", np.arange(fsc_edel_openloop.sizes["points"])))
    # fsc_edel_openloop = fsc_edel_openloop.swap_dims({"points": "n_points"})
    # # print((fsc_edel_openloop == 1).all(dim="member") + (fsc_edel_openloop == 0).all(dim="member"))
    # constant_ensemble_mask = (fsc_edel_openloop == 0).all(dim="member") + (fsc_edel_openloop == 1).all(dim="member")
    # constant_ensemble_mask = (constant_ensemble_mask.data_vars["DSN_T_ISBA"]).sel(
    #     time=duplicated_particles.time, method="nearest"
    # )
    # constant_ensemble_mask = constant_ensemble_mask.assign_coords({"time": duplicated_particles.time})

    # # print(viirs.sel(time="2021-12-15").data_vars["snow_cover_fraction"].values)
    # cloud_mask = np.isnan(viirs)
    # cloud_mask = cloud_mask.stack(points=("y", "x")).sel(time=period)
    # cloud_mask = cloud_mask.assign_coords(n_points=("points", np.arange(cloud_mask.sizes["points"])))
    # cloud_mask = cloud_mask.swap_dims({"points": "n_points"})
    # cloud_mask = (cloud_mask.data_vars["snow_cover_fraction"]).sel(time=duplicated_particles.time, method="nearest")
    # cloud_mask = cloud_mask.assign_coords({"time": duplicated_particles.time})
    # # no_assim_points_mask.s
    # # print(duplicated_particles)
    # no_assim_mask = cloud_mask + constant_ensemble_mask
    # # duplicated_particles_assim = duplicated_particles

    # # print(duplicated_particles_assim)
    # # print(duplicated_particles_assim)
    # # duplicated_particles_assim = duplicated_particles.copy(deep=True)
    # # duplicated_particles_assim = [1 - no_assim_points_mask.data_vars["DSN_T_ISBA"]]
    # # print(duplicated_particles_assim)
    # particle_occurrence = xr.DataArray(
    #     np.nan,
    #     dims=("member", "time", "n_points"),
    #     coords=duplicated_particles.coords,
    # )
    # for mb in duplicated_particles.coords["member"]:
    #     n = (duplicated_particles == mb + 1).sum(dim=("member"))
    #     particle_occurrence[dict(member=mb)] = n

    # # print((particle_occurrence / (14443)).sum(dim="member"))
    # # fig, ax = plt.subplots()
    # # ax.imshow(effective_sample_size_xarray(weights=particle_occurrence / (14443 * 17)).values.reshape(101, 143))
    # pseudoweights = particle_occurrence / duplicated_particles.sizes["member"]
    # print(pseudoweights)
    # # print(no_assim_points_mask)
    # # print(pseudoweights)
    # effective_sample_size = effective_sample_size_xarray(weights=pseudoweights)
    # print(effective_sample_size.where(1 - no_assim_mask).mean(dim=("n_points")))

    # effective_sample_size.plot.line(ax=ax)

    # plt.show()
