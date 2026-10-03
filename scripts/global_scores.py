from collections import Counter
from dataclasses import dataclass
from datetime import datetime

import matplotlib.patches as mpatches
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import xarray as xr
from matplotlib.axes import Axes
from mountain_data_binner.mountain_binner import MountainBinner, MountainBinnerConfig
from xarray.groupers import BinGrouper

from edelassim.bdclim import crop_bdclim, find_station_locations
from edelassim.evaluations import compute_rmse
from edelassim.observation_operators import dickinson
from edelassim.observations import valid_snow_cover_fraction_s2
from edelassim.snowlines import find_snowline_from_snow_penalization
from edelassim.visualization.polar import plot_polar_envelop_member, plot_snowline_polarplot, set_polarplot
from edelassim.visualization.static_files import COLORS, LABELS, get_snowlines, get_spatial_datasets, topography_data_folder

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


def assign_altitude_bin_insitu(data: xr.Dataset) -> xr.Dataset:
    altitude_bin_coord = [zs // 500 for zs in data.coords["ZS"].values]
    return data.assign_coords(altitude_bin=("num_poste", altitude_bin_coord))


def assign_altitude_bin_pleiades(data: xr.Dataset) -> xr.Dataset:
    altitude_bin_coord = data.data_vars["altitude"].values // 300
    return data.assign_coords(altitude_bin=("stacked_y_x", altitude_bin_coord))


def assign_aspect_bin_pleiades(data: xr.Dataset) -> xr.Dataset:
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
    snowline_openloop_ds, snowline_edel_ds, snowline_viirs_ds, snowline_s2_ds = get_snowlines(xpid)

    ### Station plots
    x_poste, y_poste = find_station_locations(bdclim_dataset=bdclim)
    sd_station = bdclim.where(bdclim["x"] == x_poste, drop=True).where(bdclim["y"] == y_poste, drop=True)
    # Xarray advanced indexing (according to Mistral)
    x_poste_da = xr.DataArray(sd_station.x, dims="num_poste", coords={"num_poste": sd_station.num_poste})
    y_poste_da = xr.DataArray(sd_station.y, dims="num_poste", coords={"num_poste": sd_station.num_poste})
    sd_edel_station_openloop = edelweiss_openloop.sel(x=x_poste_da, y=y_poste_da, method="nearest", drop=True)
    sd_edel_station_analysis = edelweiss_analysis.sel(x=x_poste_da, y=y_poste_da, method="nearest", drop=True)

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
    ax.grid()
    ax.set_title("Monthly snow depth residuals  Edelweiss vs in situ - ensemble median")
    xticks = list(set(grouped_dataset_month_coord.coords["num_month"].values))
    ax.set_xticks(xticks)
    ax.set_xticklabels([month_pos_dict[num_m] for num_m in xticks])
    ax.set_ylabel("SD [m]")

    fig.legend(handles=custom_leg, bbox_to_anchor=(1, 0.5))
    ## Altitude boxplot
    bins = BinGrouper(np.arange(1000, 3501, 500), labels=np.arange(1000, 3500, 500))
    grouped_dataset_altitude_coord = grouped_dataset.groupby(ZS=bins).map(assign_altitude_bin_insitu)

    fig, ax = plt.subplots(figsize=(12, 5))
    grouped_dataset_altitude_coord.groupby("altitude_bin").map(plot_var_boxplot_array, ax=ax, var="altitude_bin")

    ax.grid()
    ax.set_title("Altitude band snow depth residuals Edelweiss vs in situ - ensemble median")
    xticks = list(set(grouped_dataset_altitude_coord.coords["altitude_bin"].values))
    ax.set_xticks(xticks)
    counter = Counter(grouped_dataset_altitude_coord.coords["altitude_bin"].values)
    ax.set_xticklabels(
        [f"{bin_label} - {bin_label + 500} \n n= {counter[bin_idx]} stations" for bin_idx, bin_label in enumerate(bins.labels)]
    )
    ax.set_ylabel("SD [m]")
    fig.legend(handles=custom_leg, bbox_to_anchor=(1, 0.5))

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
    grouped_dataset = xr.Dataset(
        {
            "open_loop": residuals_pleiades_openloop,
            "analysis": residuals_pleiades_analysis,
            "altitude": xr.open_dataarray(
                f"{topography_data_folder}/250m/DEM_GR_L93_250m.tif",
            ).sel(band=1),
        }
    )
    bins = BinGrouper(np.arange(400, 3401, 300), labels=np.arange(400, 3400, 300))
    fig, ax = plt.subplots(figsize=(15, 5))
    grouped_dataset_altitude_coord = grouped_dataset.groupby(altitude=bins).map(assign_altitude_bin_pleiades)
    grouped_dataset_altitude_coord.groupby("altitude_bin").map(plot_var_boxplot_array, ax=ax, var="altitude_bin")
    ax.grid()
    ax.set_title("Altitude band snow depth residuals Edelweiss vs Pleiades - ensemble median")
    xticks = np.array(list(set(grouped_dataset_altitude_coord.coords["altitude_bin"].values.flatten())))
    xticks = xticks[~np.isnan(xticks)]
    ax.set_xticks(xticks)
    counter = Counter(grouped_dataset_altitude_coord.coords["altitude_bin"].values.flatten())
    ax.set_xticklabels(
        [f"{bin_label} - {bin_label + 300} \n n= {counter[bin_idx]} points" for bin_idx, bin_label in enumerate(bins.labels)]
    )
    ax.set_ylabel("SD [m]")
    fig.legend(handles=custom_leg, bbox_to_anchor=(1, 0.5))

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
    ax.grid()
    ax.set_title("Aspect snow depth residuals Edelweiss vs Pleiades - ensemble median")
    xticks = np.array(list(set(grouped_dataset_aspect_coord.coords["aspect_bin"].values.flatten())))
    ax.set_xticks(xticks)
    counter = Counter(grouped_dataset_aspect_coord.coords["aspect_bin"].values.flatten())
    ax.set_xticklabels([f"{bin_label}\n n= {counter[bin_idx]} points" for bin_idx, bin_label in enumerate(bins.labels)])
    ax.set_ylabel("SD [m]")
    fig.legend(handles=custom_leg, bbox_to_anchor=(1, 0.5))

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
    ax.grid()
    ax.set_title("Time snow depth residuals Edelweiss vs Pleiades - ensemble median")
    xticks = grouped_dataset_time_coord.coords["time_bin"].values
    ax.set_xticks(xticks)
    ax.set_xticklabels([pd.Timestamp(date.values).strftime("%b-%d") for date in grouped_dataset.coords["time"]])
    ax.set_ylabel("SD [m]")
    fig.legend(handles=custom_leg, bbox_to_anchor=(1, 0.5))

    plt.show()

    # plt.show()
    # rain_df["month_year"] = rain_df["date"].apply(lambda x: x.strftime("%b %Y"))

    # fsc_station = dickinson(sd_station.data_vars["neigetotx"], a=0.11, b=2.1)
    # fsc_edel_station_ol = dickinson(sd_edel_station_ol.data_vars["DSN_T_ISBA"], a=0.11, b=2.1)
    # fsc_edel_station_an = dickinson(sd_edel_station_an.data_vars["DSN_T_ISBA"], a=0.11, b=2.1)

    # period = slice("2021-11", "2022-02")
    # # compute_rmse(res_ol.sel(member=-1, time=period).values), compute_rmse(res_an.sel(member=-1, time=period).values)
    # fig, ax = plt.subplots()
    # residuals_median_ol = res_ol.sel(time=period, member=-1).values.flatten()
    # residuals_median_ol = residuals_median_ol[~np.isnan(residuals_median_ol)]
    # residuals_median_an = res_an.sel(time=period, member=-1).values.flatten()
    # residuals_median_an = residuals_median_an[~np.isnan(residuals_median_an)]

    # residuals_median_fsc_ol = (fsc_edel_station_ol - fsc_station).sel(time=period, member=-1).values.flatten()
    # residuals_median_fsc_ol = residuals_median_fsc_ol[~np.isnan(residuals_median_fsc_ol)]
    # residuals_median_fsc_an = (fsc_edel_station_an - fsc_station).sel(time=period, member=-1).values.flatten()
    # residuals_median_fsc_an = residuals_median_fsc_an[~np.isnan(residuals_median_fsc_an)]

    # diff_edel_assim = (edelweiss_an - edelweiss_ol).data_vars["DSN_T_ISBA"]
    # diff_edel_assim_median = diff_edel_assim.sel(time=period, member=-1).values.flatten()
    # diff_edel_assim_median = diff_edel_assim_median[~np.isnan(diff_edel_assim_median)]

    # ax.grid()
    # ax.set_title("Snow depth vs BDCLIM")
    # ax.grid
    # fig, ax = plt.subplots()
    # ax.boxplot(
    #     [residuals_median_fsc_ol, residuals_median_fsc_an],
    #     positions=[1, 2],
    #     showfliers=False,
    #     notch=True,
    #     patch_artist=True,
    #     widths=[0.1, 0.1],
    #     label=["Open loop", "Assimilation"],
    # )
    # ax.set_title("FSC vs BDCLIM")
    # ax.grid()
