import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import xarray as xr
from matplotlib.axes import Axes
from xarray.groupers import BinGrouper, UniqueGrouper

from edelassim.bdclim import extract_bdclim_locations_on_spatial_dataset
from edelassim.visualization.scatter import plot_ensemble_envelop
from edelassim.visualization.static_files import COLORS, LABELS, get_spatial_datasets, topography_data_folder

if __name__ == "__main__":
    xpid = "assim_viirs_cloudcover07_wy2122_d93_50cm"
    period = slice("2021-11", "2022-05")
    edelweiss_openloop, edelweiss_analysis, viirs, s2, pleiades, bdclim = get_spatial_datasets(xpid)

    pleiades = pleiades.sel(time=period).sel(band=1)
    bdclim = bdclim.sel(time=period)
    sd_edel_station_openloop = extract_bdclim_locations_on_spatial_dataset(bdclim, spatial_dataset=edelweiss_openloop)
    sd_edel_station_analysis = extract_bdclim_locations_on_spatial_dataset(bdclim, spatial_dataset=edelweiss_analysis)

    ### Snow depth envelop scatter
    # Pleiades
    snow_depth_bin = 0.1
    bins = BinGrouper(np.arange(0, 4 + 0.01, snow_depth_bin), labels=np.arange(0, 4, snow_depth_bin))

    fig, ax = plt.subplots()
    grouped_dataset = xr.Dataset(
        {
            "open_loop": edelweiss_openloop.data_vars["DSN_T_ISBA"].sel(time=pleiades.coords["time"], member=slice(0, None)),
            "analysis": edelweiss_analysis.data_vars["DSN_T_ISBA"].sel(time=pleiades.coords["time"], member=slice(0, None)),
            "reference": pleiades.data_vars["snow_depth"],
        }
    )
    grouped_dataset_binned = grouped_dataset.groupby(
        reference=bins, member=UniqueGrouper(edelweiss_openloop.sel(member=slice(0, None)).coords["member"])
    ).mean()
    xaxis_data = grouped_dataset_binned.data_vars["reference"].mean(dim="member").values.flatten()
    plot_ensemble_envelop(grouped_dataset=grouped_dataset_binned, x_axis_data=xaxis_data, ax=ax)
    ax.plot([0, 3], [0, 3], color="black", linestyle="dashed")
    ax.set_title(f"Edelweiss vs Pleiades - {snow_depth_bin} m bins")
    ax.set_xlabel("Pleiades SD [m]")
    ax.set_ylabel("Edelweiss SD [m]")
    ax.grid()
    ax.legend()

    # BD clim
    fig, ax = plt.subplots()
    grouped_dataset = xr.Dataset(
        {
            "open_loop": sd_edel_station_openloop.data_vars["DSN_T_ISBA"].sel(member=slice(0, None)),
            "analysis": sd_edel_station_analysis.data_vars["DSN_T_ISBA"].sel(member=slice(0, None)),
            "reference": bdclim.data_vars["neigetotx"],
        },
        coords=sd_edel_station_openloop.coords,
    )

    sd_bins = BinGrouper(np.arange(0, 4 + 0.01, snow_depth_bin), labels=np.arange(0, 4, snow_depth_bin))
    grouped_dataset_binned = grouped_dataset.groupby(
        reference=sd_bins, member=UniqueGrouper(edelweiss_openloop.sel(member=slice(0, None)).coords["member"])
    ).mean()
    xaxis_data = grouped_dataset_binned.data_vars["reference"].mean(dim="member").values.flatten()
    plot_ensemble_envelop(grouped_dataset=grouped_dataset_binned, x_axis_data=xaxis_data, ax=ax)
    ax.plot([0, 3], [0, 3], color="black", linestyle="dashed")
    ax.set_title(f"Edelweiss vs In situ - {snow_depth_bin} m bins")
    ax.set_xlabel("In situ SD [m]")
    ax.set_ylabel("Edelweiss SD [m]")
    ax.grid()
    ax.legend()

    # Snow depth vs altitudes

    for date in pleiades.coords["time"]:
        fig, ax = plt.subplots()
        grouped_dataset = xr.Dataset(
            {
                "open_loop": edelweiss_openloop.data_vars["DSN_T_ISBA"].sel(time=date, member=slice(0, None)),
                "analysis": edelweiss_analysis.data_vars["DSN_T_ISBA"].sel(time=date, member=slice(0, None)),
                "reference": pleiades.data_vars["snow_depth"].sel(time=date),
                "elevation": xr.open_dataarray(
                    f"{topography_data_folder}/250m/DEM_GR_L93_250m.tif",
                ).sel(band=1),
            }
        )
        altitude_step = 100
        altitude_bins = BinGrouper(np.arange(1000, 3501, altitude_step), labels=np.arange(1000, 3500, altitude_step))
        grouped_dataset_binned = grouped_dataset.groupby(
            elevation=altitude_bins, member=UniqueGrouper(edelweiss_openloop.sel(member=slice(0, None)).coords["member"])
        ).mean()
        xaxis_data = grouped_dataset_binned.data_vars["elevation"].mean(dim="member").values.flatten()
        plot_ensemble_envelop(grouped_dataset=grouped_dataset_binned, x_axis_data=xaxis_data, ax=ax)
        ax.set_title(
            f"Edelweiss and Pleiades vs elevation - {pd.Timestamp(date.values).strftime('%Y-%b-%d')} - {altitude_step} m bins"
        )
        ax.set_xlabel("Elevation [m]")
        ax.set_ylabel("Edelweiss SD [m]")
        ax.grid()
        ax.legend()
    # low_edge = edelweiss_analysis.data_vars["DSN_T_ISBA"].sel(time=pleiades.coords["time"]).quantile(0.1, dim="member")
    # high_edge = edelweiss_analysis.data_vars["DSN_T_ISBA"].sel(time=pleiades.coords["time"]).quantile(0.9, dim="member")

    # ax.fill_between(
    #     pleiades.data_vars["snow_depth"].values.flatten(),
    #     low_edge.values.flatten(),
    #     high_edge.values.flatten(),
    #     alpha=0.5,
    #     color=COLORS["edel_an"],
    # )
    # ax.plot(
    #     pleiades.data_vars["snow_depth"].values.flatten(),
    #     edelweiss_analysis.data_vars["DSN_T_ISBA"].sel(member=-1, time=pleiades.coords["time"]).values.flatten(),
    #     linewidth=0,
    #     marker=".",
    #     markersize=1.5,
    #     color=COLORS["edel_an"],
    # )
    # ax.set_xlim(0, 3)
    # ax.set_ylim(0, 3)
    plt.show()
    # grouped_dataset = xr.Dataset(
    #     {
    #         "open_loop": sd_edel_station_openloop,
    #         "analysis": sd_edel_station_analysis,
    #         "pleiades": pleiades,
    #         "elevation": xr.open_dataarray(
    #             f"{topography_data_folder}/250m/DEM_GR_L93_250m.tif",
    #         ).sel(band=1),
    #     }
    # )
