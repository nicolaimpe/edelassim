from collections import Counter

import matplotlib.patches as mpatches
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import xarray as xr
from matplotlib.axes import Axes
from matplotlib.figure import Figure
from mountain_data_binner.mountain_binner import MountainBinner
from xarray.groupers import BinGrouper

from edelassim.bdclim import extract_bdclim_locations_on_spatial_dataset, find_station_locations
from edelassim.visualization.static_files import COLORS, LABELS, get_spatial_datasets, topography_data_folder

pos_month_dict = {
    "Aug": 0,
    "Sep": 1,
    "Oct": 2,
    "Nov": 3,
    "Dec": 4,
    "Jan": 5,
    "Feb": 6,
    "Mar": 7,
    "Apr": 8,
    "May": 9,
    "Jun": 10,
    "Jul": 11,
}
month_pos_dict = {v: k for k, v in pos_month_dict.items()}


def plot_var_boxplot_array(data: xr.Dataset, ax: Axes, var: str, width: float = 0.4):
    # print(data.data_vars["open_loop"])
    open_loop_residuals = data.data_vars["open_loop"].values.flatten()
    analysis_residuals = data.data_vars["analysis"].values.flatten()
    # width = 0.4
    # print(width)
    # print(len(data[var]))
    bp = ax.boxplot(
        [open_loop_residuals[~np.isnan(open_loop_residuals)], analysis_residuals[~np.isnan(analysis_residuals)]],
        positions=[data[var][0] - width / 2, data[var][0] + width / 2],
        showfliers=False,
        notch=True,
        patch_artist=True,
        widths=width,
        label=[LABELS["edel_ol"], LABELS["edel_an"]],
    )
    bp["boxes"][0].set_facecolor(COLORS["edel_ol"])
    bp["boxes"][1].set_facecolor(COLORS["edel_an"])
    return data


def decorate_boxplot(fig: Figure, ax: Axes, xticks: list, xticks_labels: list) -> None:
    ax.grid()
    ax.set_xticks(xticks)
    ax.set_xticklabels(xticks_labels)
    ax.set_ylabel("SD [m]")
    fig.legend(handles=custom_leg, bbox_to_anchor=(1.1, 0.6))
    ax.hlines(0, ax.get_xlim()[0], ax.get_xlim()[1], linewidth=2, color="black", linestyles="dashed")


def assign_altitude_bin_insitu(data: xr.Dataset, altitude_step: float = 500) -> xr.Dataset:
    altitude_bin_coord = [zs // altitude_step for zs in data.coords["ZS"].values]
    return data.assign_coords(altitude_bin=("num_poste", altitude_bin_coord))


def assign_altitude_bin_pleiades(data: xr.Dataset, altitude_step: float = 300) -> xr.Dataset:
    altitude_bin_coord = data.data_vars["altitude"].values // altitude_step
    return data.assign_coords(altitude_bin=("stacked_y_x", altitude_bin_coord))


def assign_aspect_bin_pleiades(data: xr.Dataset) -> xr.Dataset:
    # +22.5 allows to align with rose compass
    aspect_bin_coord = (data.data_vars["aspect"].values + 22.5) // 45
    return data.assign_coords(aspect_bin=("stacked_y_x", aspect_bin_coord))


def assign_month(data: xr.Dataset) -> xr.Dataset:
    month_coord = [pd.Timestamp(t).strftime("%b") for t in data.coords["time"].values]
    month_num_coord = [pos_month_dict[t] for t in month_coord]
    return data.assign_coords(month=("time", month_coord), num_month=("time", month_num_coord))


if __name__ == "__main__":
    xpid = "assim_viirs_cloudcover07_wy2122_d93_50cm"
    period = slice("2021-11", "2022-07")
    edelweiss_openloop, edelweiss_analysis, viirs, s2, pleiades, bdclim = get_spatial_datasets(xpid)

    ### Station plots
    x_poste, y_poste = find_station_locations(bdclim_dataset=bdclim)
    sd_station = bdclim.where(bdclim["x"] == x_poste, drop=True).where(bdclim["y"] == y_poste, drop=True)

    sd_edel_station_openloop = extract_bdclim_locations_on_spatial_dataset(bdclim, spatial_dataset=edelweiss_openloop)
    sd_edel_station_analysis = extract_bdclim_locations_on_spatial_dataset(bdclim, spatial_dataset=edelweiss_analysis)

    residuals_station_openloop = sd_edel_station_openloop.data_vars["DSN_T_ISBA"].sel(
        member=-1, time=period
    ) - sd_station.data_vars["neigetotx"].sel(time=period)
    residuals_station_analysis = sd_edel_station_analysis.data_vars["DSN_T_ISBA"].sel(
        member=-1, time=period
    ) - sd_station.data_vars["neigetotx"].sel(time=period)

    grouped_dataset = xr.Dataset(
        {
            "open_loop": residuals_station_openloop,
            "analysis": residuals_station_analysis,
        }
    )
    custom_leg = [
        mpatches.Patch(color=COLORS["edel_ol"], label=LABELS["edel_ol"]),
        mpatches.Patch(color=COLORS["edel_an"], label=LABELS["edel_an"]),
    ]
    ## Monthly boxplot
    grouped_dataset_month_coord = grouped_dataset.resample(time="1ME").map(assign_month)
    fig, ax = plt.subplots(figsize=(12, 5))
    grouped_dataset_month_coord.groupby("num_month").map(plot_var_boxplot_array, ax=ax, var="num_month")

    ax.set_title("Monthly snow depth residuals  Edelweiss vs in situ - ensemble median")
    xticks = list(set(grouped_dataset_month_coord.coords["num_month"].values))
    decorate_boxplot(fig=fig, ax=ax, xticks=xticks, xticks_labels=[month_pos_dict[num_m] for num_m in xticks])

    ## Altitude boxplot
    bdclim_alt_step = 500
    bins = BinGrouper(np.arange(1000, 3501, bdclim_alt_step), labels=np.arange(1000, 3500, bdclim_alt_step))
    grouped_dataset_altitude_coord = grouped_dataset.groupby(ZS=bins).map(
        assign_altitude_bin_insitu, altitude_step=bdclim_alt_step
    )

    fig, ax = plt.subplots(figsize=(12, 5))
    grouped_dataset_altitude_coord.groupby("altitude_bin").map(plot_var_boxplot_array, ax=ax, var="altitude_bin")

    ax.set_title("Altitude band snow depth residuals Edelweiss vs in situ - ensemble median")
    xticks = list(set(grouped_dataset_altitude_coord.coords["altitude_bin"].values))
    counter = Counter(grouped_dataset_altitude_coord.coords["altitude_bin"].values)
    xticks_labels = [
        f"{bin_label} - {bin_label + bdclim_alt_step} \n n= {counter[bin_idx]} stations"
        for bin_idx, bin_label in enumerate(bins.labels)
    ]
    decorate_boxplot(fig=fig, ax=ax, xticks=xticks, xticks_labels=xticks_labels)

    ### Pleiades boxplots
    pleiades = pleiades.sel(time=period).sel(band=1)
    residuals_pleiades_openloop = (
        edelweiss_openloop.data_vars["DSN_T_ISBA"].sel(member=-1, time=pleiades.coords["time"])
        - pleiades.data_vars["snow_depth"]
    )
    residuals_pleiades_analysis = (
        edelweiss_analysis.data_vars["DSN_T_ISBA"].sel(member=-1, time=pleiades.coords["time"])
        - pleiades.data_vars["snow_depth"]
    )

    # Altitude boxplots
    pleiades_alt_step = 300
    grouped_dataset = xr.Dataset(
        {
            "open_loop": residuals_pleiades_openloop,
            "analysis": residuals_pleiades_analysis,
            "altitude": xr.open_dataarray(
                f"{topography_data_folder}/250m/DEM_GR_L93_250m.tif",
            ).sel(band=1),
        }
    )
    bins = BinGrouper(np.arange(400, 3401, pleiades_alt_step), labels=np.arange(400, 3400, pleiades_alt_step))
    fig, ax = plt.subplots(figsize=(15, 5))

    grouped_dataset_altitude_coord = grouped_dataset.groupby(altitude=bins).map(
        assign_altitude_bin_pleiades, altitude_step=pleiades_alt_step
    )
    grouped_dataset_altitude_coord.groupby("altitude_bin").map(plot_var_boxplot_array, ax=ax, var="altitude_bin")

    ax.set_title("Altitude band snow depth residuals Edelweiss vs Pleiades - ensemble median")
    xticks = np.array(list(set(grouped_dataset_altitude_coord.coords["altitude_bin"].values.flatten())))
    xticks = xticks[~np.isnan(xticks)]
    counter = Counter(grouped_dataset_altitude_coord.coords["altitude_bin"].values.flatten())
    xticks_labels = [
        f"{bin_label} - {bin_label + pleiades_alt_step} \n n= {counter[bin_idx]} points"
        for bin_idx, bin_label in enumerate(bins.labels)
    ]

    decorate_boxplot(fig=fig, ax=ax, xticks=xticks, xticks_labels=xticks_labels)

    # Aspect boxplots
    grouped_dataset = xr.Dataset(
        {
            "open_loop": residuals_pleiades_openloop,
            "analysis": residuals_pleiades_analysis,
            "aspect": MountainBinner.aspect_map_transform(
                xr.open_dataarray(
                    f"{topography_data_folder}/250m/ASP_GR_L93_250m.tif",
                ).sel(band=1)
            ),
        }
    )
    bins = MountainBinner.regular_8_aspect_bins()
    fig, ax = plt.subplots(figsize=(15, 5))
    grouped_dataset_aspect_coord = grouped_dataset.groupby(aspect=bins).map(assign_aspect_bin_pleiades)
    grouped_dataset_aspect_coord.groupby("aspect_bin").map(plot_var_boxplot_array, ax=ax, var="aspect_bin")
    ax.set_title("Aspect snow depth residuals Edelweiss vs Pleiades - ensemble median")
    xticks = np.array(list(set(grouped_dataset_aspect_coord.coords["aspect_bin"].values.flatten())))
    counter = Counter(grouped_dataset_aspect_coord.coords["aspect_bin"].values.flatten())
    xticks_labels = [f"{bin_label}\n n= {counter[bin_idx]} points" for bin_idx, bin_label in enumerate(bins.labels)]
    decorate_boxplot(fig=fig, ax=ax, xticks=xticks, xticks_labels=xticks_labels)

    # Time boxplots
    grouped_dataset = xr.Dataset(
        {
            "open_loop": residuals_pleiades_openloop,
            "analysis": residuals_pleiades_analysis,
        }
    )

    fig, ax = plt.subplots(figsize=(8, 5))
    grouped_dataset_time_coord = grouped_dataset.assign_coords(time_bin=("time", np.arange(grouped_dataset.sizes["time"])))
    grouped_dataset_time_coord.groupby("time_bin").map(plot_var_boxplot_array, ax=ax, var="time_bin")
    ax.set_title("Time snow depth residuals Edelweiss vs Pleiades - ensemble median")
    xticks = grouped_dataset_time_coord.coords["time_bin"].values
    xticks_labels = [pd.Timestamp(date.values).strftime("%b-%d") for date in grouped_dataset.coords["time"]]
    decorate_boxplot(fig=fig, ax=ax, xticks=xticks, xticks_labels=xticks_labels)
    plt.show()
