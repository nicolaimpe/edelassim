import calendar
import logging
from datetime import datetime, timedelta

import matplotlib.pyplot as plt
import numpy as np
import xarray as xr
from geospatial_grid.georeferencing import georef_netcdf_rioxarray
from matplotlib.widgets import Button
from pyproj import CRS

from edelassim.observation_operators import dickinson
from edelassim.observations import (
    find_clear_dates_s2,
    find_clear_dates_viirs,
    valid_snow_cover_fraction_s2,
    valid_snow_cover_fraction_viirs_mf,
)
from edelassim.visualization import (
    FSC_CMAP_SNOW_COVER,
    add_2d_plot,
    plot_ensemble_snowline_polarplot_from_semidistributed,
    plot_snowline_polarplot_from_semidistributed,
)


def quantile_index(arr, q):
    """Return the index of the value closest to the q-quantile."""
    q_val = np.quantile(arr, q)
    return np.argmin(np.abs(arr - q_val))


def change_day(delta):
    global current_date
    current_date += timedelta(days=delta)
    update_plot()


def change_month(delta):
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
    update_plot()


def next_good_date_viirs(event):
    global current_date

    # Find the first good date after current_date
    for d in good_dates_viirs:
        if d > current_date:
            current_date = d
            break
    update_plot()


def prev_good_date_viirs(event):
    global current_date

    # Find the last good date before current_date
    for d in reversed(good_dates_viirs):
        if d < current_date:
            current_date = d
            break

    update_plot()


def next_good_date_s2(event):
    global current_date

    # Find the first good date after current_date
    for d in good_dates_s2:
        if d > current_date:
            current_date = d
            break
    update_plot()


def prev_good_date_s2(event):
    global current_date

    # Find the last good date before current_date
    for d in reversed(good_dates_s2):
        if d < current_date:
            current_date = d
            break

    update_plot()


# def update_a(val):
#     global current_a
#     # Find closest value
#     current_a = min(a_values, key=lambda x: abs(x - val))
#     slider_a.set_val(current_a)  # Snap slider
#     update_plot()


def change_a(delta):
    global current_a
    current_a_idx = list(a_values).index(current_a)
    current_a_idx += delta
    current_a = a_values[current_a_idx]
    a_text.set_text(f"a = {current_a}")
    update_plot()


def change_mb(delta):
    global current_member
    current_mb_idx = list(member_values).index(current_member)
    current_mb_idx += delta
    current_member = member_values[current_mb_idx]
    mb_text.set_text(f"member = {'avg.' if current_member == -1 else current_member}")
    update_plot()


def update_plot():
    [ax.clear() for ax in axs_all]
    # Determine ensemble visualization indexes

    for snowline, label, color in zip(snowline_data, labels, colors):
        if np.datetime64(current_date) in snowline.coords["time"]:
            snowline_to_plot = snowline.sel(time=current_date, slope="8 - 30")
            if label in ("Edelweiss OL", "Edelweiss assim"):
                plot_ensemble_snowline_polarplot_from_semidistributed(
                    snowline_parametrization_dataset=snowline_to_plot.sel(a=current_a),
                    dataset_type="snow_cover",
                    ax=ax_snowlines,
                    color=color,
                )
                snowline_to_plot = snowline_to_plot.sel(a=current_a, member=current_member)
        else:
            continue

        plot_snowline_polarplot_from_semidistributed(
            snowline_parametrization_dataset=snowline_to_plot,
            dataset_type="snow_cover",
            ax=ax_snowlines,
            label=label,
            color=color,
        )

    s2_title = "Sentinel-2 FSC [-]"
    if np.datetime64(current_date) in snow_cover_s2.coords["time"]:
        s2_data = snow_cover_s2.sel(time=current_date)
        add_2d_plot(s2_data, ax_s2, dem_20m, title=s2_title, cmap=FSC_CMAP_SNOW_COVER, vmin=0, vmax=1)
    else:
        ax_s2.set_title(s2_title)
        ax_s2.set_xticks([]), ax_s2.set_yticks([])

    viirs_data = snow_cover_viirs.sel(time=current_date)
    add_2d_plot(viirs_data, ax_viirs, dem_250m, "VIIRS FSC [-]", cmap=FSC_CMAP_SNOW_COVER, vmin=0, vmax=1)

    fsc_edel = dickinson(sd=snow_depth_edel_ol.sel(time=current_date, member=current_member), a=current_a, b=0.11)
    add_2d_plot(fsc_edel, ax_edelweiss, dem_250m, "Edelweiss FSC [-]", cmap=FSC_CMAP_SNOW_COVER, vmin=0, vmax=1)

    diff_edel_viirs = fsc_edel - snow_cover_viirs.sel(time=current_date)
    diff_cmap = plt.get_cmap("coolwarm_r")
    diff_cmap.set_bad("gray")
    add_2d_plot(diff_edel_viirs, ax_diff, dem_250m, "Diff FSC Edelweiss - VIIRS [-]", cmap=diff_cmap, vmin=-1, vmax=1)

    snow_depth_cmap = plt.get_cmap("Blues")
    snow_depth_cmap.set_under("black")
    snow_depth_cmap.set_bad("gray")
    sd_data = snow_depth_edel_ol.sel(time=current_date, member=current_member)
    add_2d_plot(sd_data, ax_snow_depth, dem_250m, "Snow depth Edelweiss [m]", cmap=snow_depth_cmap, vmin=0.001, vmax=2.5)

    ######### FORCING ###########
    total_precip = forcing.data_vars["precip_total"].sel(time=current_date, member=current_member)
    precip_cmap = plt.get_cmap("viridis")
    precip_cmap.set_bad("gray")
    add_2d_plot(total_precip, ax_precip, dem_250m, "Total precipitation [mm/day]", cmap=precip_cmap)

    phase_cmap = plt.get_cmap("Blues_r")
    phase_cmap.set_bad("gray")
    phase_data = forcing.data_vars["phase"].sel(time=current_date, member=current_member)
    add_2d_plot(phase_data, ax_phase, dem_250m, "Precipitation phase [-]", cmap=phase_cmap, vmin=-1, vmax=1)

    try:
        if not np.all(np.isnan(snow_rain_forcing.sel(time=current_date, slope_bins="8 - 30").data_vars["phase"])):
            plot_ensemble_snowline_polarplot_from_semidistributed(
                snowline_parametrization_dataset=snow_rain_forcing.sel(time=current_date, slope_bins="8 - 30"),
                ax=ax_snow_rain_line,
                dataset_type="forcing",
                color="blue",
            )
            current_phase_line = snow_rain_forcing.sel(time=current_date, member=current_member, slope_bins="8 - 30")
            plot_snowline_polarplot_from_semidistributed(
                snowline_parametrization_dataset=current_phase_line,
                ax=ax_snow_rain_line,
                dataset_type="forcing",
                color="blue",
                label="Edelweiss OL",
            )
            plot_ensemble_snowline_polarplot_from_semidistributed(
                snowline_parametrization_dataset=snow_rain_forcing_analysis.sel(time=current_date, slope_bins="8 - 30"),
                ax=ax_snow_rain_line,
                dataset_type="forcing",
                color="purple",
            )
            current_phase_line = snow_rain_forcing_analysis.sel(time=current_date, member=current_member, slope_bins="8 - 30")
            plot_snowline_polarplot_from_semidistributed(
                snowline_parametrization_dataset=current_phase_line,
                ax=ax_snow_rain_line,
                dataset_type="forcing",
                color="purple",
                label="Edelweiss assim",
            )
            ax_snow_rain_line.set_title("Forcing snow-rain line")
        else:
            logger.info(f"no phase on day {current_date}")
    except ValueError as e:
        logger.info(f"Exception caught {e}")

    date_text.set_text(str(current_date.date()))
    # Do not remove ticks for the snowline plot
    fig.canvas.draw_idle()


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
    labels = ("Sentinel-2", "Edelweiss OL", "Edelweiss assim", "VIIRS")
    colors = ("black", "blue", "purple", "red")

    # Create figure and polar axes
    fig = plt.figure(figsize=(20, 10))
    # Initial date
    current_date = datetime(2021, 11, 1)

    ##################### Create button axes
    button_width = 0.07
    button_height = 0.025
    button_y1 = 0.12
    button_y2 = 0.08
    button_x1 = 0.02

    ax_d_minus = plt.axes([button_x1 + button_width, button_y1, button_width, button_height])
    ax_d_plus = plt.axes([button_x1 + 2 * button_width, button_y1, button_width, button_height])
    ax_m_minus = plt.axes([button_x1, button_y1, button_width, button_height])
    ax_m_plus = plt.axes([button_x1 + 3 * button_width, button_y1, button_width, button_height])
    ax_a_minus = plt.axes([0.35, 0.35, 0.03, 0.02])
    ax_a_plus = plt.axes([0.40, 0.35, 0.03, 0.02])
    ax_mb_minus = plt.axes([0.35, 0.20, 0.03, 0.02])
    ax_mb_plus = plt.axes([0.40, 0.20, 0.03, 0.02])

    ax_next_good_viirs = plt.axes([button_x1 + 2 * button_width, button_y2, button_width, button_height])
    ax_prev_good_viirs = plt.axes([button_x1 + button_width, button_y2, button_width, button_height])
    ax_next_good_s2 = plt.axes([button_x1 + 3 * button_width, button_y2, button_width, button_height])
    ax_prev_good_s2 = plt.axes([button_x1, button_y2, button_width, button_height])

    btn_d_minus = Button(ax_d_minus, "D-")
    btn_d_plus = Button(ax_d_plus, "D+")
    btn_m_minus = Button(ax_m_minus, "M-")
    btn_m_plus = Button(ax_m_plus, "M+")
    btn_next_good_viirs = Button(ax_next_good_viirs, "Next good VIIRS")
    btn_prev_good_viirs = Button(ax_prev_good_viirs, "Prev Good VIIRS")
    btn_next_good_s2 = Button(ax_next_good_s2, "Next good S2")
    btn_prev_good_s2 = Button(ax_prev_good_s2, "Prev Good S2")
    btn_a_minus = Button(ax_a_minus, "a-")
    btn_a_plus = Button(ax_a_plus, "a+")
    btn_mb_minus = Button(ax_mb_minus, "member-")
    btn_mb_plus = Button(ax_mb_plus, "member+")

    btn_d_minus.on_clicked(lambda e: change_day(-1))
    btn_d_plus.on_clicked(lambda e: change_day(1))
    btn_m_minus.on_clicked(lambda e: change_month(-1))
    btn_m_plus.on_clicked(lambda e: change_month(1))

    btn_next_good_viirs.on_clicked(next_good_date_viirs)
    btn_prev_good_viirs.on_clicked(prev_good_date_viirs)
    btn_next_good_s2.on_clicked(next_good_date_s2)
    btn_prev_good_s2.on_clicked(prev_good_date_s2)

    btn_a_minus.on_clicked(lambda e: change_a(-1))
    btn_a_plus.on_clicked(lambda e: change_a(1))

    btn_mb_minus.on_clicked(lambda e: change_mb(-1))
    btn_mb_plus.on_clicked(lambda e: change_mb(1))

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

    snowline_s2 = xr.open_dataset(f"{s2_folder}/snowline_paremetrization.nc")
    snowline_ol_edel = xr.open_dataset(f"{edelweiss_ol_folder}/snowline_paremetrization.nc")
    snowline_an_edel = xr.open_dataset(f"{edelweiss_an_folder}/snowline_paremetrization.nc")
    snowline_viirs = xr.open_dataset(f"{viirs_folder}/snowline_paremetrization.nc")
    snowline_data = (snowline_s2, snowline_ol_edel, snowline_an_edel, snowline_viirs)

    snow_rain_forcing = xr.open_dataset(f"{forcing_folder}/snowline_parametrization.nc")
    snow_rain_forcing_analysis = xr.open_dataset(f"{forcing_analysis_folder}/snowline_parametrization.nc")
    snow_depth_edel_ol = xr.open_dataset(f"{edelweiss_an_folder}/spatial.nc")
    snow_depth_edel_ol = snow_depth_edel_ol.data_vars["DSN_T_ISBA"].sortby("y", ascending=False).where(1 - mask)

    good_dates_s2 = find_clear_dates_s2(snow_cover_s2=snow_cover_s2)
    good_dates_viirs = find_clear_dates_viirs(snow_cover_viirs=snow_cover_viirs)

    # Slider for observation operator values
    a_values = snowline_ol_edel.coords["a"].values  # or from your DataArray
    # Initial value for observation operator parametrization
    current_a = a_values[0]

    # Slider for member values
    member_values = snowline_ol_edel.coords["member"].values  # or from your DataArray
    # Initial value for observation operator parametrization
    current_member = member_values[0]

    logger.info("Plotting")
    ax_s2 = fig.add_subplot(3, 4, 1)
    ax_edelweiss = fig.add_subplot(3, 4, 2)
    ax_viirs = fig.add_subplot(3, 4, 3)
    ax_diff = fig.add_subplot(3, 4, 4)
    ax_snow_depth = fig.add_subplot(3, 4, 6)
    ax_precip = fig.add_subplot(3, 4, 7)
    ax_phase = fig.add_subplot(3, 4, 8)
    ax_snowlines = fig.add_subplot(3, 4, 11, projection="polar")
    ax_snow_rain_line = fig.add_subplot(3, 4, 12, projection="polar")
    # ax_temperature = fig.add_subplot(258)
    axs_snow_cover = [ax_s2, ax_edelweiss, ax_viirs]
    axs_all = [*axs_snow_cover, ax_diff, ax_precip, ax_snowlines, ax_snow_depth, ax_precip, ax_phase, ax_snow_rain_line]
    # fig, axs = plt.subplots(1, 2, figsize=(5, 8), subplot_kw={"projection": "polar"}, layout="constrained")
    # fig.subplots_adjust(bottom=0.35)  # Room for buttons
    date_text = fig.suptitle(str(current_date.date()), y=0.98)
    a_text = fig.text(s=f"a = {current_a}", y=0.32, x=0.37)
    mb_text = fig.text(s=f"member = {'avg.' if current_member == -1 else current_member}", y=0.25, x=0.37)

    # Initial plot
    fig.subplots_adjust(bottom=0.05, top=0.96, left=0.05, hspace=1e-4, right=0.96, wspace=1e-4)
    update_plot()
    # plt.tight_layout()

    plt.show()
