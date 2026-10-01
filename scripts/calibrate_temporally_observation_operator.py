from datetime import datetime, timedelta

import matplotlib.pyplot as plt
import numpy as np
import xarray as xr
from pandas import Interval, Timestamp

from edelassim.bdclim import crop_bdclim, find_station_locations
from edelassim.evaluations import compute_rmse
from edelassim.observation_operators import dickinson
from edelassim.observations import (
    EdelweissGrandesRoussesGrid,
    find_clear_dates_viirs,
    scale_virrs,
    valid_snow_cover_fraction_viirs_mf,
)
from edelassim.visualization.scatter import boxplot_logit_plot, find_common_correspondences


def plot_boxplot(residuals, pos, ax, label, color):
    bp = ax.boxplot(
        residuals,
        positions=pos,
        showfliers=False,
        notch=True,
        patch_artist=True,
        widths=0.4,
        label=label,
    )
    bp["boxes"][0].set_facecolor(color)


if __name__ == "__main__":
    ## User inputs
    data_folder = "/home/imperatoren/work/edelweiss_assimilation/data/grandesrousses250m"
    simulation_folder = "/home/imperatoren/work/edelweiss_assimilation/simulations/postprocess/grandesrousses250m"
    forest_mask_path = f"{data_folder}/auxiliary/forest_mask/forest_mask_corine_grandesrousses_max.nc"
    viirs_filepath = f"{data_folder}/../../observations/grandesrousses250m/meteofrance/spatial.nc"
    pleiades_path = f"{data_folder}/../../observations/grandesrousses250m/pleiades_grandesrousses_all.nc"
    bdclim_path = f"{data_folder}/../../observations/grandesrousses250m//bdclim/bdclim.nc"
    LABELS = {"bdclim": "In situ", "pleiades": "Pleiades"}
    COLORS = {"bdclim": "gray", "pleiades": "darkcyan"}
    ## Reading inputs

    forest_mask = xr.open_dataarray(forest_mask_path, engine="rasterio").sel(band=1).drop_vars("band")
    no_assim_all_path = f"{simulation_folder}/open_loop/spatial.nc"
    viirs = (
        valid_snow_cover_fraction_viirs_mf(xr.open_dataset(viirs_filepath).data_vars["snow_cover_fraction"])
        .sel(time=slice("2021-11", "2022-06"))
        .where(forest_mask == 0)
    )
    pleiades = (
        xr.open_dataset(pleiades_path).data_vars["snow_depth"].sel(time=slice("2021-11", "2022-06")).where(forest_mask == 0)
    )

    grandesrousses_grid = EdelweissGrandesRoussesGrid()
    bdclim = crop_bdclim(bdclim_path, grid=grandesrousses_grid).data_vars["neigetotx"].sel(time=slice("2021-11", "2022-06"))
    x_poste, y_poste = find_station_locations(bdclim_dataset=bdclim)

    sd_pleaides = pleiades.sel(x=viirs["x"], y=viirs["y"], method="nearest").sel(band=1)
    sd_pleaides = sd_pleaides.where(sd_pleaides >= 0)
    sd_bdclim = bdclim.sel(time=viirs.coords["time"], method="nearest")
    viirs_station = viirs.sel(x=bdclim.coords["x"], y=bdclim.coords["y"], method="nearest").assign_coords(
        {"time": sd_bdclim.time}
    )
    viirs_pleaides_passes = viirs.sel(time=pleiades.coords["time"], method="nearest").assign_coords({"time": pleiades.time})

    good_dates_viirs = find_clear_dates_viirs(snow_cover_viirs=viirs)
    # good_dates_viirs = viirs.coords["time"]

    ## General residual boxplot for b=1 and b=best value
    fig, ax = plt.subplots(figsize=(8, 5), constrained_layout=True)

    b_values = 1, 1.157, 1.22, 1.367
    a_value = 0.11

    for pos, b in enumerate(b_values):
        # Label only once for automatic legend
        label_bdclim = LABELS["bdclim"] if pos == 0 else None
        label_pleiades = LABELS["pleiades"] if pos == 0 else None

        fsc_bdclim = dickinson(sd_bdclim, b=b, a=a_value)

        station_matches, viirs_matches = find_common_correspondences(
            fsc_bdclim.sel(time=good_dates_viirs), viirs_station.sel(time=good_dates_viirs)
        )
        residuals_station = viirs_matches - station_matches
        plot_boxplot(residuals=residuals_station, pos=[pos - 0.2], ax=ax, label=label_bdclim, color=COLORS["bdclim"])

        fsc_pleiades = dickinson(sd_pleaides, b=b, a=0.11)
        pleiades_matches, viirs_pleiades_matches = find_common_correspondences(fsc_pleiades, viirs_pleaides_passes)
        residuals_pleiades = viirs_pleiades_matches - pleiades_matches
        plot_boxplot(residuals=residuals_pleiades, pos=[pos + 0.2], ax=ax, label=label_pleiades, color=COLORS["pleiades"])

    ax.set_xticks(np.arange(pos + 1))
    ax.set_xticklabels([f"b = {b}" for b in b_values])
    ax.hlines(0, ax.get_xlim()[0], ax.get_xlim()[1], linewidth=2, color="black", linestyles="dashed")
    ax.grid()
    ax.legend()
    ax.set_title("Distribution of the residuals november to june 2021/2022 VIIRS FSC - H(SD,b)")

    ## Monthly boxplots
    fig, axs = plt.subplots(4, 1, figsize=(6, 12), constrained_layout=True)
    fig.suptitle("Observation operator comparison VIIRS - EDELWEISS / VIIRS - REFERENCES")

    period_start, period_end = (datetime(year=2021, month=11, day=1), datetime(year=2022, month=7, day=1))
    interval = timedelta(days=30)
    period_starts = np.arange(period_start, period_end, interval)

    for j, b in enumerate((1, 1.15, 1.22, 2.1)):
        fsc_bdclim = dickinson(sd_bdclim, a=a_value, b=b)
        rmse_stations = []
        bias_stations = []
        sigma_stations = []
        for i, ps in enumerate(period_starts):
            # Label only once for automatic legend
            label_bdclim = LABELS["bdclim"] if i == 0 else None
            label_pleiades = LABELS["pleiades"] if i == 0 else None

            period = slice(ps, np.datetime64(ps + interval))
            station_matches, viirs_matches = find_common_correspondences(
                fsc_bdclim.sel(time=good_dates_viirs).sel(time=period),
                viirs_station.sel(time=good_dates_viirs).sel(time=period),
            )
            residuals_station = viirs_matches - station_matches
            plot_boxplot(residuals=residuals_station, pos=[i - 0.2], ax=axs[j], label=label_bdclim, color=COLORS["bdclim"])
            # if i == 3 or i == 6:
            fsc_pleaides = dickinson(sd_pleaides, a=a_value, b=b)
            pleiades_matches, viirs_pleiades_matches = find_common_correspondences(
                fsc_pleaides.sel(time=period), viirs_pleaides_passes.sel(time=period)
            )
            residuals_pleiades = viirs_pleiades_matches - pleiades_matches
            plot_boxplot(residuals_pleiades, pos=[i + 0.2], ax=axs[j], label=label_pleiades, color=COLORS["pleiades"])

        axs[j].grid("y")
        axs[j].hlines(0, axs[j].get_xlim()[0], axs[j].get_xlim()[1], linewidth=2, color="black", linestyles="dashed")
        axs[j].set_xticks(np.arange(len(period_starts)))
        axs[j].set_xticklabels([Timestamp(ps).strftime("%Y-%m-%d") for ps in period_starts])
        axs[j].set_title(f"b={b} - boxplot of the residuals and RMSE")
        axs[j].set_ylim(-0.6, 0.6)
        axs[j].legend()

## Monthly RMSE ans bias for different values of b

fig, axs = plt.subplots(2, 1, figsize=(15, 5))

period_start, period_end = (datetime(year=2021, month=11, day=1), datetime(year=2022, month=6, day=30))
interval = timedelta(days=3)
period_starts = np.arange(period_start, period_end, interval)

rmse_b_stations = []
bias_b_stations = []
b_values = (1, 1.15, 1.22, 2.1)
for j, b in enumerate(b_values):
    rmse_stations = []
    bias_stations = []
    for ps in period_starts:
        period = slice(ps, np.datetime64(ps + interval))
        fsc_bdclim = dickinson(sd_bdclim, a=a_value, b=b)
        station_matches, viirs_matches = find_common_correspondences(
            fsc_bdclim.sel(time=good_dates_viirs).sel(time=period), viirs_station.sel(time=good_dates_viirs).sel(time=period)
        )
        residuals_station = viirs_matches - station_matches
        rmse_stations.append(compute_rmse(residuals_station))
        bias_stations.append(np.nanmean(residuals_station))

    axs[0].plot(period_starts[~np.isnan(rmse_stations)], np.array(rmse_stations)[~np.isnan(rmse_stations)], label=f"b={b}")
    axs[1].plot(period_starts[~np.isnan(rmse_stations)], np.array(bias_stations)[~np.isnan(rmse_stations)], label=f"b={b}")
    rmse_b_stations.append(rmse_stations)
    bias_b_stations.append(bias_stations)

rmse_b_stations_array = np.array(rmse_b_stations)
best_b_rmse = np.min(rmse_b_stations, axis=0)
axs[0].plot(
    period_starts[~np.isnan(best_b_rmse)],
    best_b_rmse[~np.isnan(best_b_rmse)],
    label="b=best",
    color="black",
    linestyle="--",
    linewidth=2,
)

bias_b_stations_array = np.array(bias_b_stations)
best_b_bias_idx = np.argmin(np.abs(bias_b_stations_array), axis=0)
best_b_bias = np.array([bias_b_stations_array[b_idx, idx] for idx, b_idx in enumerate(best_b_bias_idx)])
axs[1].plot(
    period_starts[~np.isnan(best_b_bias)],
    best_b_bias[~np.isnan(best_b_bias)],
    label="b=best",
    color="black",
    linestyle="--",
    linewidth=2,
)
[ax.legend() for ax in axs]
[ax.grid() for ax in axs]
axs[1].hlines(0, axs[1].get_xlim()[0], axs[1].get_xlim()[1], linewidth=1.5, color="gray", linestyles="dashed")
axs[0].set_title(f"Temporal RMSE VIIRS FSC - H(SD,b), varying b, intervals of {interval.days} days")
axs[1].set_title(f"Temporal bias VIIRS FSC - H(SD,b), varying b, intervals of {interval.days} days")
fig.subplots_adjust(hspace=1)
# plt.show()

## Daily error considering two observation operator strategies
# RMSE
best_b_rmse_idx = np.argmin(rmse_b_stations, axis=0)
best_b = [b_values[idx] for idx in best_b_rmse_idx]
b_data_array = xr.DataArray(data=best_b, coords={"time": period_starts})
b_data_array = b_data_array.reindex(time=sd_bdclim.coords["time"], method="ffill")


def adaptive_observation_operator(ds: xr.Dataset):
    return dickinson(sd=ds.data_vars["snow_depth"], b=ds.data_vars["b"], a=a_value)


fsc_bdclim_adaptive = (
    xr.Dataset({"snow_depth": sd_bdclim, "b": b_data_array}).groupby("time").map(adaptive_observation_operator)
)
rmse_adaptive = np.sqrt(
    ((viirs_station - fsc_bdclim_adaptive) ** 2).sel(time=good_dates_viirs).groupby("time").mean(dim="num_poste")
)

fsc_bdclim_fixed = sd_bdclim.groupby("time").map(dickinson, a=a_value, b=1.22)
rmse_fixed = np.sqrt(
    ((viirs_station - fsc_bdclim_fixed) ** 2).sel(time=good_dates_viirs).groupby("time").mean(dim="num_poste")
)

# Bias
best_b_bias_idx = np.argmin(bias_b_stations, axis=0)
best_b = [b_values[idx] for idx in best_b_rmse_idx]
b_data_array = xr.DataArray(data=best_b, coords={"time": period_starts})
b_data_array = b_data_array.reindex(time=sd_bdclim.coords["time"], method="ffill")
fsc_bdclim_adaptive = (
    xr.Dataset({"snow_depth": sd_bdclim, "b": b_data_array}).groupby("time").map(adaptive_observation_operator)
)
bias_adaptive = (viirs_station - fsc_bdclim_adaptive).sel(time=good_dates_viirs).groupby("time").mean(dim="num_poste")

fsc_bdclim_fixed = sd_bdclim.groupby("time").map(dickinson, a=a_value, b=1.22)

bias_fixed = (viirs_station - fsc_bdclim_fixed).sel(time=good_dates_viirs).groupby("time").mean(dim="num_poste")

fig, axs = plt.subplots(2, 1, figsize=(12, 7))
axs[0].plot(rmse_adaptive.time, rmse_adaptive.values, label=f"adaptive mean RMSE={rmse_adaptive.mean().values:.2f}")
axs[0].plot(rmse_fixed.time, rmse_fixed.values, label=f"fixed mean RMSE={rmse_fixed.mean().values:.2f}")
axs[0].legend()
axs[0].grid()
axs[0].set_title("Daily RMSE per observation operator strategies")

axs[1].plot(bias_adaptive.time, bias_adaptive.values, label=f"adaptive mean MAE={np.abs(bias_adaptive).mean().values:.2f}")
axs[1].plot(bias_fixed.time, bias_fixed.values, label=f"fixed mean MAE={np.abs(bias_fixed).mean().values:.2f}")
axs[1].legend()
axs[1].grid()
axs[1].set_title("Daily bias per observation operator strategies")
plt.show()
