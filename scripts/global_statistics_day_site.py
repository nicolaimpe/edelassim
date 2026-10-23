from datetime import timedelta
from glob import glob

import matplotlib
import matplotlib.cm as cm
import matplotlib.colors as colors
import matplotlib.pyplot as plt
import numpy as np
import xarray as xr
from matplotlib.axes import Axes
from matplotlib.gridspec import GridSpec
from pyproj import Transformer
from snowtools.scores.ensemble import EnsembleScores

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


def compute_rsquare(data: xr.Dataset) -> xr.DataArray:

    r2_open_loop = EnsembleScores(
        timeObs=None, timeSim=None, obs=data.data_vars["reference"].values, ensemble=data.data_vars["open_loop"].values
    ).CRPS()
    r2_analysis = EnsembleScores(
        timeObs=None, timeSim=None, obs=data.data_vars["reference"].values, ensemble=data.data_vars["analysis"].values
    ).CRPS()

    return xr.DataArray(r2_analysis - r2_open_loop)


def postprocess_scores(scores: xr.Dataset, bdclim: xr.Dataset) -> xr.Dataset:
    scores = scores.assign_coords(ZS=("num_poste", bdclim.coords["ZS"].values))
    scores = scores.assign_coords(lat=("num_poste", bdclim.coords["lat"].values))
    scores = scores.assign_coords(lon=("num_poste", bdclim.coords["lon"].values))
    scores = scores.assign_coords(Station_Name=("num_poste", bdclim.coords["Station_Name"].values))
    scores = scores.sortby("ZS")
    return scores


def day_site_difference_plot(score_data: xr.DataArray, ax: Axes):
    top = matplotlib.colormaps["Reds_r"].resampled(128)
    bottom = matplotlib.colormaps["Blues"].resampled(128)
    newcolors = np.vstack((top(np.linspace(0, 1, 128)), bottom(np.linspace(0, 1, 128))))
    newcmp = colors.ListedColormap(newcolors, name="RdGn")
    norm = colors.Normalize(vmin=-0.5, vmax=0.5)

    day_of_season = (score_data.time - score_data.time[0]).dt.days

    for i in range(score_data.sizes["num_poste"]):
        barplot_colors = newcmp(norm(score_data.isel(num_poste=i)))
        # Bad days to plot before good days so thick line width is not replaces
        ax.barh(
            y=i,
            width=timedelta(days=1).days,
            height=1,
            left=day_of_season,
            facecolor=barplot_colors,
            edgecolor="black",
            linewidth=0.1,
        )
    first_of_month = day_of_season.sel(time=day_of_season.coords["time"].dt.day == 1)
    ax.set_xticks(first_of_month.values)
    ax.set_xticklabels(first_of_month.time.dt.strftime("%d-%B").values)
    ax.set_yticks(np.arange(score_data.sizes["num_poste"]))
    ax.set_yticklabels(
        [
            f"{name.title()} - {elevation} m"
            for name, elevation in zip(score_data.coords["Station_Name"].values, score_data.coords["ZS"].values)
        ]
    )
    sm = cm.ScalarMappable(cmap=newcmp, norm=norm)
    cbar = ax.get_figure().colorbar(sm, ax=ax)
    cbar.set_label(f"{score_data.name}")
    ax.set_title(f"{score_data.name}")


def day_site_score_plot(score_data: xr.DataArray, ax: Axes):
    top = matplotlib.colormaps["Reds_r"].resampled(128)
    bottom = matplotlib.colormaps["Greens"].resampled(128)
    newcolors = np.vstack((top(np.linspace(0, 1, 128)), bottom(np.linspace(0, 1, 128))))

    good_days_mask = score_data > 0
    bad_days_mask = score_data < 0
    neutral_days = score_data == 0
    total_day_site = score_data.count().values

    newcmp = colors.ListedColormap(newcolors, name="RdGn")
    norm = colors.Normalize(vmin=-0.5, vmax=0.5)

    day_of_season = (score_data.time - score_data.time[0]).dt.days
    good_days_mask = score_data > 0

    for i in range(score_data.sizes["num_poste"]):
        barplot_colors = newcmp(norm(score_data.isel(num_poste=i)))
        # Bad days to plot before good days so thick line width is not replaces
        ax.barh(
            y=i,
            width=timedelta(days=1).days,
            height=1,
            left=day_of_season.where(~good_days_mask.isel(num_poste=i)),
            facecolor=barplot_colors[~good_days_mask.isel(num_poste=i), :],
            edgecolor="black",
            linewidth=0.1,
        )
        ax.barh(
            y=i,
            width=timedelta(days=1).days,
            height=1,
            left=day_of_season.where(good_days_mask.isel(num_poste=i)),
            facecolor=barplot_colors[good_days_mask.isel(num_poste=i), :],
            edgecolor="black",
            linewidth=1,
        )
        ax.barh(
            y=i,
            width=timedelta(days=1).days,
            height=1,
            left=day_of_season.where(np.isnan(score_data).isel(num_poste=i)),
            facecolor="gray",
            edgecolor="black",
            linewidth=0.1,
        )

    first_of_month = day_of_season.sel(time=day_of_season.coords["time"].dt.day == 1)
    ax.set_xticks(first_of_month.values)
    ax.set_xticklabels(first_of_month.time.dt.strftime("%d-%B").values)
    ax.set_yticks(np.arange(score_data.sizes["num_poste"]))
    ax.set_yticklabels(
        [
            f"{name.title()} - {elevation} m"
            for name, elevation in zip(score_data.coords["Station_Name"].values, score_data.coords["ZS"].values)
        ]
    )
    ax.set_title(
        f"{score_data.name} - {good_days_mask.sum().values} / {total_day_site} improved, \
            {bad_days_mask.sum().values} / {total_day_site} degraded\
            {neutral_days.sum().values} / {total_day_site} neutral day-site"
    )
    sm = cm.ScalarMappable(cmap=newcmp, norm=norm)
    cbar = ax.get_figure().colorbar(sm, ax=ax)
    cbar.set_label(f"{score_data.name}")


def alt_time_diagramm_plot(score_data: xr.DataArray, ax: Axes):
    top = matplotlib.colormaps["Reds_r"].resampled(128)
    bottom = matplotlib.colormaps["Greens"].resampled(128)
    newcolors = np.vstack((top(np.linspace(0, 1, 128)), bottom(np.linspace(0, 1, 128))))

    newcmp = matplotlib.colors.ListedColormap(newcolors, name="RdGn")
    norm = matplotlib.colors.Normalize(vmin=-0.5, vmax=0.5)

    day_of_season = (score_data.time - score_data.time[0]).dt.days
    for i in range(score_data.sizes["elevation_min"]):
        barplot_colors = newcmp(norm(score_data.isel(elevation_min=i)))
        ax.barh(
            y=i,
            width=timedelta(days=1).days,
            height=1,
            left=day_of_season,
            facecolor=barplot_colors,
            edgecolor="black",
            linewidth=0.1,
        )
        ax.barh(
            y=i,
            width=timedelta(days=1).days,
            height=1,
            left=day_of_season.where(np.isnan(score_data).isel(elevation_min=i)),
            facecolor="gray",
            edgecolor="black",
            linewidth=0.1,
        )

    first_of_month = day_of_season.sel(time=day_of_season.coords["time"].dt.day == 1)
    ax.set_xticks(first_of_month.values)
    ax.set_xticklabels(first_of_month.time.dt.strftime("%d-%B").values)
    ax.set_yticks(np.arange(score_data.sizes["elevation_min"]))
    ax.set_yticklabels([f"({el_min} - {el_min + 100}] m" for el_min in score_data.coords["elevation_min"].values])
    ax.set_title(f"Time - altitude of score {score} - FSC VIIRS")

    sm = cm.ScalarMappable(cmap=newcmp, norm=norm)
    ax.get_figure().colorbar(sm, ax=ax)


if __name__ == "__main__":
    xpid = "assim_viirs_cloudcover07_wy2122_d93_50cm"
    period = slice("2021-11", "2022-05")
    edelweiss_openloop, edelweiss_analysis, viirs, s2, _, bdclim = get_spatial_datasets(xpid)

    # bdclim = filter_discontinuous_stations(bdclim.sel(time=period))

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

    grouped_dataset = xr.Dataset(
        {
            "open_loop": sd_edel_station_openloop.data_vars["DSN_T_ISBA"].sel(member=slice(0, None)),
            "analysis": sd_edel_station_analysis.data_vars["DSN_T_ISBA"].sel(member=slice(0, None)),
            "reference": bdclim.data_vars["neigetotx"],
        },
        coords=sd_edel_station_analysis.coords,
    )

    # scores_sd = grouped_dataset.groupby(("num_poste", "time")).map(compute_scores_grouped_data)
    # scores_sd = xr.open_dataset("scores_sd.nc")
    # scores_fsc = dickinson(grouped_dataset).groupby(("num_poste", "time")).map(compute_scores_grouped_data)
    # scores_fsc.to_netcdf("scores_fsc.nc")
    # combined = xr.concat([scores_sd, scores_fsc], dim="variable")
    # combined = combined.assign_coords(dict(variable=["snow_depth", "fractional_snow_cover"]))

    scores = xr.open_dataset("../output_folder/scores_2021_2022.nc")
    scores = scores.assign_coords(dict(variable=["snow_depth", "fractional_snow_cover"]))

    scores = postprocess_scores(scores, bdclim)

    fig, axs = plt.subplots(nrows=2, figsize=(18, 10))

    ####### Difference mean member
    ## FSC
    # Ol
    day_site_difference_plot(
        score_data=scores.sel(type="open_loop", variable="fractional_snow_cover").data_vars["diff_mean_member"], ax=axs[0]
    )
    axs[0].set_title(f"FSC open loop - {axs[0].get_title()}")
    # analysis
    day_site_difference_plot(
        score_data=scores.sel(type="analysis", variable="fractional_snow_cover").data_vars["diff_mean_member"], ax=axs[1]
    )
    axs[1].set_title(f"FSC analysis - {axs[1].get_title()}")
    # Snow depth
    # Ol
    fig, axs = plt.subplots(nrows=2, figsize=(18, 10))
    day_site_difference_plot(
        score_data=scores.sel(type="open_loop", variable="snow_depth").data_vars["diff_mean_member"], ax=axs[0]
    )
    axs[0].set_title(f"SD open loop - {axs[0].get_title()}")
    # Analysis
    day_site_difference_plot(
        score_data=scores.sel(type="analysis", variable="snow_depth").data_vars["diff_mean_member"], ax=axs[1]
    )
    axs[1].set_title(f"SD analysis - {axs[1].get_title()}")

    ############# Difference mean member open loop - analysis
    fig, axs = plt.subplots(nrows=2, figsize=(18, 10))
    scores = scores.assign(
        {
            "diff_mae_mean_member": np.abs(scores.data_vars["diff_mean_member"].sel(type="open_loop"))
            - np.abs(scores.data_vars["diff_mean_member"].sel(type="analysis"))
        }
    )
    # FSC
    day_site_score_plot(score_data=scores.sel(variable="fractional_snow_cover").data_vars["diff_mae_mean_member"], ax=axs[0])
    axs[0].set_title(f"FSC - MAE mean member OL - assim {axs[0].get_title()}")
    # SD
    day_site_score_plot(score_data=scores.sel(variable="snow_depth").data_vars["diff_mae_mean_member"], ax=axs[1])
    axs[1].set_title(f" SD - MAE mean member OL - assim- {axs[1].get_title()}")

    ############### CRPSS
    fig, axs = plt.subplots(nrows=2, figsize=(18, 10))
    scores = scores.assign(
        {"crpss": 1 - scores.data_vars["crps"].sel(type="analysis") / scores.data_vars["crps"].sel(type="open_loop")}
    )
    # FSC
    day_site_score_plot(score_data=scores.sel(variable="fractional_snow_cover").data_vars["crpss"], ax=axs[0])
    axs[0].set_title(f"FSC - {axs[0].get_title()}")
    # SD
    day_site_score_plot(score_data=scores.sel(variable="snow_depth").data_vars["crpss"], ax=axs[1])
    axs[1].set_title(f" SD - {axs[1].get_title()}")

    ################ VIIRS J+1 as reference - altitude bins 100m

    scores = xr.open_dataset("../output_folder/edel_viirs_scores.nc")
    score = "MAE open loop - MAE analysis"
    scores = np.abs(scores.data_vars["mae"].sel(type="open_loop")) - np.abs(scores.data_vars["mae"].sel(type="analysis"))
    fig, ax = plt.subplots(figsize=(12, 5))
    alt_time_diagramm_plot(score_data=scores, ax=ax)
    ax.set_title(f"Edelweiss vs VIIRS FSC (J+1 not assimilated) - {score}")

    plt.show()

    # scores_fsc = dickinson(grouped_dataset).groupby(("num_poste", "time")).map(compute_scores_grouped_data)
    # postprocess_scores(scores_fsc, bdclim).to_netcdf("scores_fsc.nc")
    # scores_sd = xr.open_dataset("scores.nc")
    # scores_fsc = xr.open_dataset("scores_fsc.nc")
    # # fig, (ax_sd, ax_fsc) = plt.subplots(nrows=2, ncols=1, figsize=(14, 8))

    # fig, ax = plt.subplots(figsize=(18, 5))

    # top = matplotlib.colormaps["Reds_r"].resampled(128)
    # bottom = matplotlib.colormaps["Greens"].resampled(128)
    # newcolors = np.vstack((top(np.linspace(0, 1, 128)), bottom(np.linspace(0, 1, 128))))
    # newcmp = colors.ListedColormap(newcolors, name="RdGn")

    # # colors.Colorm
    # # colormap = cm.RdYlGn
    # norm = colors.Normalize(vmin=-0.5, vmax=0.5)

    # scores = xr.open_dataset("../scripts/scores_fsc.nc")
    # scores = scores.sel(num_poste=scores.num_poste[scores.Station_Name != "BESSE"], drop=True)
    # scores = scores.sel(num_poste=scores.num_poste[scores.Station_Name != "Les 2 ALpes Jandri"], drop=True)
    # # scores = scores.sel(num_poste=scores.num_poste[stations!='BESSE'], drop=True)
    # # scores = scores.where([stations!='Besse'])
    # score = "maes"

    # plt.show()
