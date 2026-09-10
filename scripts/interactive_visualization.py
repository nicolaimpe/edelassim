import calendar
import logging
from datetime import datetime, timedelta

import matplotlib.pyplot as plt
import numpy as np
import xarray as xr
from geospatial_grid.georeferencing import georef_netcdf_rioxarray
from pyproj import CRS

from edelassim.observation_operators import dickinson
from edelassim.observations import (
    find_clear_dates_s2,
    find_clear_dates_viirs,
    valid_snow_cover_fraction_s2,
    valid_snow_cover_fraction_viirs_mf,
)
from edelassim.snowlines import find_forcing_snowrain_line, find_snowline_from_snow_penalization
from edelassim.visualization.interactive_plots import InteractiveSeasonExploreButtons
from edelassim.visualization.polar import (  # plot_ensemble_snowline_polarplot,; plot_ensemble_snowline_polarplot_from_semidistributed,; plot_snowline_polarplot_from_semidistributed,
    plot_envelop_member_snow_rain,
    plot_envelop_member_snowline,
    plot_snowline_polarplot,
)
from edelassim.visualization.spatial import (
    FIELD_DIFF_CMAP,
    FSC_CMAP_SNOW_COVER,
    PHASE_CMAP,
    PRECIP_CMAP,
    SNOW_DEPTH_CMAP,
    add_2d_plot,
)


def quantile_index(arr: np.ndarray, q: float):
    """Return the index of the value closest to the q-quantile."""
    q_val = np.quantile(arr, q)
    return np.argmin(np.abs(arr - q_val))


def change_day(delta: int):
    global current_date
    current_date += timedelta(days=delta)
    update_all_plots()


def change_month(delta: int):
    global current_date
    year = current_date.year
    month = current_date.month + delta
    if month > 12:
        month = 1
        year += 1
    elif month < 1:
        month = 12
        year -= 1
    # Handle day overflow (e.g., Jan 31 -> Feb)
    last_day = calendar.monthrange(year, month)[1]
    day = min(current_date.day, last_day)
    current_date = datetime(year, month, day)
    update_all_plots()


def last_good_date(good_dates_list: list[datetime]):
    global current_date
    # Find the last good date before current_date
    for d in reversed(good_dates_list):
        if d < current_date:
            current_date = d
            break
    update_all_plots()


def next_good_date(good_dates_list: list[datetime]):
    global current_date

    # Find the first good date after current_date
    for d in good_dates_list:
        if d > current_date:
            current_date = d
            break
    update_all_plots()


def change_a(delta: int):
    global current_a
    current_a_idx = list(a_values).index(current_a)
    current_a_idx += delta
    current_a = a_values[current_a_idx]
    a_text.set_text(f"a = {current_a}")
    update_all_plots()


def change_mb(delta: int):
    global current_member
    current_mb_idx = list(member_values).index(current_member)
    current_mb_idx += delta
    current_member = member_values[current_mb_idx]
    mb_text.set_text(f"member = {'avg.' if current_member == -1 else current_member}")
    update_all_plots()


def update_spatial_plots():
    [ax.clear() for ax in axs_spatial]
    s2_title = "Sentinel-2 FSC [-]"
    if np.datetime64(current_date) in snow_cover_s2.coords["time"]:
        s2_data = snow_cover_s2.sel(time=current_date)
        add_2d_plot(s2_data, ax_s2, dem_20m, title=s2_title, cmap=FSC_CMAP_SNOW_COVER, vmin=0, vmax=1)
    else:
        ax_s2.set_title(s2_title)
        ax_s2.set_xticks([]), ax_s2.set_yticks([])

    viirs_data = snow_cover_viirs.sel(time=current_date)
    add_2d_plot(viirs_data, ax_viirs, dem_250m, "VIIRS FSC [-]", cmap=FSC_CMAP_SNOW_COVER, vmin=0, vmax=1)

    fsc_edel = dickinson(sd=snow_depth_edel_ol_ds.sel(time=current_date, member=current_member), a=current_a, b=0.11)
    add_2d_plot(fsc_edel, ax_edelweiss, dem_250m, "Edelweiss FSC [-]", cmap=FSC_CMAP_SNOW_COVER, vmin=0, vmax=1)

    diff_edel_viirs = fsc_edel - snow_cover_viirs.sel(time=current_date)

    add_2d_plot(diff_edel_viirs, ax_diff, dem_250m, "Diff FSC Edelweiss - VIIRS [-]", cmap=FIELD_DIFF_CMAP, vmin=-1, vmax=1)

    sd_data = snow_depth_edel_ol_ds.sel(time=current_date, member=current_member)
    add_2d_plot(sd_data, ax_snow_depth, dem_250m, "Snow depth Edelweiss [m]", cmap=SNOW_DEPTH_CMAP, vmin=0.001, vmax=2.5)

    ######### FORCING ###########
    total_precip = forcing.data_vars["precip_total"].sel(time=current_date, member=current_member)
    add_2d_plot(total_precip, ax_precip, dem_250m, "Total precipitation [mm/day]", cmap=PRECIP_CMAP)

    phase_data = forcing.data_vars["phase"].sel(time=current_date, member=current_member)
    add_2d_plot(phase_data, ax_phase, dem_250m, "Precipitation phase [-]", cmap=PHASE_CMAP, vmin=-1, vmax=1)

    fig_maps.suptitle(str(current_date.date()), y=0.98)
    fig_maps.canvas.draw_idle()


def update_snowline_plots():
    [ax.clear() for ax in axs_snowlines]
    # Determine ensemble visualization indexes
    alt_max = snowline_ol_edel_ds.coords["altitude_max"].max()
    alt_min = snowline_ol_edel_ds.coords["altitude_min"].min()
    plot_kwargs = {"ax": ax_snowlines, "alt_min": alt_min, "alt_max": alt_max}

    if np.datetime64(current_date) in snowline_s2_ds.coords["time"]:
        snowline_s2 = find_snowline_from_snow_penalization(snowline_s2_ds.sel(time=current_date))
        plot_snowline_polarplot(snowline_s2, label=LABELS["s2"], color=COLORS["s2"], **plot_kwargs)

    snowline_viirs = find_snowline_from_snow_penalization(snowline_viirs_ds.sel(time=current_date))
    plot_snowline_polarplot(snowline_viirs, label=LABELS["viirs"], color=COLORS["viirs"], **plot_kwargs)

    current_edel_ol_sl = snowline_ol_edel_ds.sel(time=current_date, a=current_a)
    plot_envelop_member_snowline(current_edel_ol_sl, current_member, LABELS["edel_ol"], COLORS["edel_ol"], **plot_kwargs)

    current_edel_an_sl = snowline_an_edel_ds.sel(time=current_date, a=current_a)
    plot_envelop_member_snowline(current_edel_an_sl, current_member, LABELS["edel_an"], COLORS["edel_an"], **plot_kwargs)

    plot_kwargs.update({"ax": ax_snow_rain_line})
    current_forc_ol_srl = snow_rain_forcing_ol_ds.sel(time=current_date)
    current_forc_an_srl = snow_rain_forcing_an_ds.sel(time=current_date)
    # When at least one of the aspects is not covered by the phase field (-> all aktitudes are NaNs) we do not plot
    if not np.any(np.isnan(current_forc_ol_srl.data_vars["phase"]).all(dim="altitude_bins")):
        plot_envelop_member_snow_rain(current_forc_ol_srl, current_member, LABELS["edel_ol"], COLORS["edel_ol"], **plot_kwargs)
        plot_envelop_member_snow_rain(current_forc_an_srl, current_member, LABELS["edel_an"], COLORS["edel_an"], **plot_kwargs)
    else:
        logger.info(f"no phase on day {current_date}")

    fig_snowlines.suptitle(str(current_date.date()), y=0.98)
    fig_snowlines.canvas.draw_idle()


def update_all_plots():
    update_spatial_plots()
    update_snowline_plots()


# Module configuration
logger = logging.getLogger("logger")
logging.basicConfig(level=logging.INFO)
if __name__ == "__main__":
    ################################ User inputs #############################################
    working_folder = "/home/imperatoren/work/edelweiss_assimilation/"
    observation_folder = f"{working_folder}/observations/grandesrousses250m"
    simulation_folder = f"{working_folder}/simulations/postprocess"
    s2_folder = f"{observation_folder}/s2"
    edelweiss_ol_folder = f"{simulation_folder}/grandesrousses250m/open_loop/"
    edelweiss_an_folder = f"{simulation_folder}/reanalysis/assim_viirs_all_clear_dates_november_2021/"
    viirs_folder = f"{observation_folder}/meteofrance/"
    forcing_folder = f"{working_folder}/forcing/grandesrousses250m/open_loop"
    forcing_analysis_folder = f"{working_folder}/forcing/reanalysis/assim_viirs_all_clear_dates_november_2021/"

    topography_data_folder = f"{working_folder}/data/grandesrousses250m/auxiliary/topography/"
    landcover_folder = f"{working_folder}/data/grandesrousses250m/auxiliary/"
    forest_mask_path = f"{landcover_folder}/forest_mask/forest_mask_corine_grandesrousses_max.nc"
    glacier_mask_path = f"{landcover_folder}/glacier_mask/glacier_mask_glims_2022_grandesrousses.nc"
    dem_250m_filepath = f"{topography_data_folder}/250m/DEM_GR_L93_250m.tif"
    dem_20m_filepath = f"{topography_data_folder}/20m/DEM_GR_UTM_20m.tif"
    # Snowline plots
    LABELS = {"s2": "Sentinel-2", "viirs": "VIIRS", "edel_ol": "Edelweiss OL", "edel_an": "Edelweiss assim"}
    COLORS = {"s2": "black", "viirs": "red", "edel_ol": "blue", "edel_an": "purple"}

    # Initial date
    current_date = datetime(2021, 11, 1)

    fig_buttons = plt.figure(figsize=(8, 2))
    buttons = InteractiveSeasonExploreButtons(fig=fig_buttons)
    buttons.btn_d_minus.on_clicked(lambda e: change_day(-1))
    buttons.btn_d_plus.on_clicked(lambda e: change_day(1))

    buttons.btn_m_minus.on_clicked(lambda e: change_month(-1))
    buttons.btn_m_plus.on_clicked(lambda e: change_month(1))

    buttons.btn_next_good_viirs.on_clicked(lambda e: next_good_date(good_dates_viirs))
    buttons.btn_prev_good_viirs.on_clicked(lambda e: last_good_date(good_dates_viirs))
    buttons.btn_next_good_s2.on_clicked(lambda e: next_good_date(good_dates_s2))
    buttons.btn_prev_good_s2.on_clicked(lambda e: last_good_date(good_dates_s2))

    buttons.btn_a_minus.on_clicked(lambda e: change_a(-1))
    buttons.btn_a_plus.on_clicked(lambda e: change_a(1))

    buttons.btn_mb_minus.on_clicked(lambda e: change_mb(-1))
    buttons.btn_mb_plus.on_clicked(lambda e: change_mb(1))

    ########################### Read data ##########################################################

    logger.info("Reading data")
    glacier_mask = xr.open_dataset(glacier_mask_path).data_vars["__xarray_dataarray_variable__"]
    forest_mask = xr.open_dataset(forest_mask_path).sel(band=1).data_vars["__xarray_dataarray_variable__"]
    mask = glacier_mask + forest_mask
    mask = georef_netcdf_rioxarray(mask, crs=CRS.from_epsg(2154))
    snow_cover_s2 = xr.open_dataset(f"{s2_folder}/spatial.nc").data_vars["snow_cover_fraction"]
    mask_20m = mask.rio.reproject_match(snow_cover_s2)
    snow_cover_s2 = valid_snow_cover_fraction_s2(snow_cover_s2.where(1 - mask_20m))

    dem_250m = xr.open_dataarray(dem_250m_filepath).sel(band=1)
    dem_20m = xr.open_dataarray(dem_20m_filepath).sel(band=1)

    snow_cover_viirs = valid_snow_cover_fraction_viirs_mf(
        xr.open_dataset(f"{viirs_folder}/spatial.nc").data_vars["snow_cover_fraction"].where(1 - mask)
    )

    forcing = xr.open_mfdataset(f"{forcing_analysis_folder}/spatial.nc").sortby("y", ascending=False).where(1 - mask)

    snowline_s2_ds = xr.open_dataset(f"{s2_folder}/snowline_paremetrization.nc").sel(slope="8 - 30")
    snowline_ol_edel_ds = xr.open_dataset(f"{edelweiss_ol_folder}/snowline_paremetrization.nc").sel(slope="8 - 30")
    snowline_an_edel_ds = xr.open_dataset(f"{edelweiss_an_folder}/snowline_paremetrization.nc").sel(slope="8 - 30")
    snowline_viirs_ds = xr.open_dataset(f"{viirs_folder}/snowline_paremetrization.nc").sel(slope="8 - 30")

    snow_rain_forcing_ol_ds = xr.open_dataset(f"{forcing_folder}/snowline_parametrization.nc").sel(slope_bins="8 - 30")
    snow_rain_forcing_an_ds = xr.open_dataset(f"{forcing_analysis_folder}/snowline_parametrization.nc").sel(
        slope_bins="8 - 30"
    )
    snow_depth_edel_ol_ds = xr.open_dataset(f"{edelweiss_an_folder}/spatial.nc")
    snow_depth_edel_ol_ds = snow_depth_edel_ol_ds.data_vars["DSN_T_ISBA"].sortby("y", ascending=False).where(1 - mask)

    good_dates_s2 = find_clear_dates_s2(snow_cover_s2=snow_cover_s2)
    good_dates_viirs = find_clear_dates_viirs(snow_cover_viirs=snow_cover_viirs)

    # Slider for observation operator values
    a_values = snowline_ol_edel_ds.coords["a"].values  # or from your DataArray
    # Initial value for observation operator parametrization
    current_a = a_values[0]

    # Slider for member values
    member_values = snowline_ol_edel_ds.coords["member"].values  # or from your DataArray
    # Initial value for observation operator parametrization
    current_member = member_values[0]

    logger.info("Plotting")
    fig_maps = plt.figure(figsize=(20, 10))
    ax_s2 = fig_maps.add_subplot(2, 4, 1)
    ax_edelweiss = fig_maps.add_subplot(2, 4, 2)
    ax_viirs = fig_maps.add_subplot(2, 4, 3)
    ax_diff = fig_maps.add_subplot(2, 4, 4)
    ax_snow_depth = fig_maps.add_subplot(2, 4, 6)
    ax_precip = fig_maps.add_subplot(2, 4, 7)
    ax_phase = fig_maps.add_subplot(2, 4, 8)
    axs_snow_cover = [ax_s2, ax_edelweiss, ax_viirs]
    axs_spatial = [*axs_snow_cover, ax_diff, ax_precip, ax_snow_depth, ax_precip, ax_phase]

    fig_snowlines = plt.figure(figsize=(10, 6))
    ax_snowlines = fig_snowlines.add_subplot(1, 2, 1, projection="polar")
    ax_snow_rain_line = fig_snowlines.add_subplot(1, 2, 2, projection="polar")
    axs_snowlines = [ax_snowlines, ax_snow_rain_line]

    fig_maps.suptitle(str(current_date.date()), y=0.98)
    fig_snowlines.suptitle(str(current_date.date()), y=0.98)

    a_text = fig_buttons.text(s=f"a = {current_a}", y=0.32, x=0.37)
    mb_text = fig_buttons.text(s=f"member = {'avg.' if current_member == -1 else current_member}", y=0.25, x=0.37)

    # Initial plot
    fig_maps.subplots_adjust(bottom=0.05, top=0.96, left=0.05, hspace=1e-4, right=0.96, wspace=1e-4)
    update_all_plots()
    # plt.tight_layout()

    plt.show()
