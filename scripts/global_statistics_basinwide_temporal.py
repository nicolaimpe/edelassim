import matplotlib.pyplot as plt
import numpy as np
import xarray as xr
from matplotlib.axes import Axes
from pyproj import Transformer
from xarray.groupers import BinGrouper, UniqueGrouper

from edelassim.bdclim import extract_bdclim_locations_on_spatial_dataset, find_station_locations
from edelassim.observation_operators import dickinson
from edelassim.observations import find_clear_dates_s2, find_clear_dates_viirs
from edelassim.visualization.scatter import plot_ensemble_envelop
from edelassim.visualization.static_files import COLORS, LABELS, get_spatial_datasets, topography_data_folder

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
    print(s2_station.count())
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
    for alt_min in range(1000, 2501, 500):
        fig, axs = plt.subplots(2, 1, figsize=(12, 8))
        fig.suptitle(f"Edelweiss vs in situ Basin wide {alt_min} - {alt_min + 500}")
        num_poste_altitude = (
            grouped_dataset.where(
                (grouped_dataset.coords["ZS"] > alt_min) * (grouped_dataset.coords["ZS"] <= alt_min + 500), drop=True
            )
            .coords["num_poste"]
            .values
        )

        grouped_dataset_binned = grouped_dataset.sel(num_poste=num_poste_altitude).groupby("member").mean(dim="num_poste")

        plot_ensemble_envelop(
            grouped_dataset=grouped_dataset_binned, x_axis_data=grouped_dataset.coords["time"].values, ax=axs[0]
        )
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

        grouped_dataset_binned = (
            dickinson(grouped_dataset).sel(num_poste=num_poste_altitude).groupby("member").mean(dim="num_poste")
        )
        plot_ensemble_envelop(
            grouped_dataset=grouped_dataset_binned, x_axis_data=grouped_dataset.coords["time"].values, ax=axs[1]
        )
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
            viirs_station.data_vars["snow_cover_fraction"]
            .sel(num_poste=num_poste_altitude)
            .mean(dim="num_poste")
            .values.flatten(),
            color=COLORS["viirs"],
            linewidth=0,
            marker=".",
            markersize=10,
            label=LABELS["viirs"],
        )
        axs[1].plot(
            s2_station.coords["time"].values,
            s2_station.data_vars["snow_cover_fraction"]
            .sel(num_poste=num_poste_altitude)
            .mean(dim="num_poste")
            .values.flatten(),
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
    plt.show()
